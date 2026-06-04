"""Synthetic benchmark metrics for FactorGraph-ST."""

from __future__ import annotations

from itertools import permutations
from math import exp, lgamma, log

import numpy as np


def matched_factor_correlation(estimated: np.ndarray, truth: np.ndarray) -> float:
    """Mean absolute Pearson correlation after best one-to-one factor matching.

    Empty/degenerate inputs (no factors to match) return ``float('nan')``
    rather than silently scoring ``1.0`` — a model that produces zero
    factors is uninformative, not perfect. Callers/benchmark runners should
    treat ``nan`` as "not evaluable". See #79.
    """
    if estimated.size == 0 or truth.size == 0:
        return float("nan")
    corr = _abs_corr_matrix(estimated, truth)
    if corr.size == 0:
        return float("nan")
    # Orient so rows <= cols. The matching value is symmetric, but both the
    # brute-force and greedy searches below assign every row to a distinct
    # column and only ever consider the first ``cols`` rows; with more
    # estimated factors than truth factors (rows > cols) that silently ignored
    # the surplus estimated factors and undercounted recovery (#46). Searching
    # over the larger axis fixes both paths.
    if corr.shape[0] > corr.shape[1]:
        corr = corr.T
    rows, cols = corr.shape
    if max(rows, cols) <= 8:
        best = 0.0
        for perm in permutations(range(cols), min(rows, cols)):
            score = sum(corr[r, c] for r, c in enumerate(perm))
            best = max(best, score)
        return float(best / min(rows, cols))
    # Greedy fallback keeps the package dependency-free for larger matrices.
    used: set[int] = set()
    total = 0.0
    for r in range(rows):
        choices = [(corr[r, c], c) for c in range(cols) if c not in used]
        if not choices:
            break
        val, c = max(choices)
        used.add(c)
        total += float(val)
    return float(total / min(rows, cols))


def section_overlap(Z: np.ndarray, section_id: np.ndarray) -> np.ndarray:
    """Fraction of each factor's activation mass in each section."""
    if Z.ndim != 2:
        raise ValueError("Z must be 2D")
    sections = np.unique(section_id)
    out = np.zeros((Z.shape[1], sections.size), dtype=np.float32)
    mass = Z.sum(axis=0)
    for j, section in enumerate(sections):
        out[:, j] = Z[section_id == section].sum(axis=0) / np.maximum(mass, 1e-12)
    return out


def factor_redundancy(Z: np.ndarray) -> float:
    """Mean absolute off-diagonal correlation among estimated factors.

    Quantifies *within-estimate* redundancy of the recovered factors (the
    columns of ``Z = [Z_shared | Z_private]``). ``0.0`` means the factors are
    mutually uncorrelated (well disentangled); values approaching ``1.0`` mean
    near-duplicate factors (the same program reported multiple times).
    Constant (zero-variance) factors contribute zero correlation, and fewer
    than two factors returns ``0.0`` (no redundancy is possible).
    """
    if Z.ndim != 2:
        raise ValueError("Z must be 2D")
    k = Z.shape[1]
    if k < 2:
        return 0.0
    corr = _abs_corr_matrix(Z, Z)
    off_diagonal = float(corr.sum() - np.trace(corr))
    return off_diagonal / (k * (k - 1))


def shared_private_separation(Z_shared: np.ndarray, Z_private: np.ndarray, section_id: np.ndarray) -> dict[str, float]:
    """Return section-spread summaries for shared and private activations.

    A factor counts as "active" in a section when its per-section mass
    fraction exceeds ``min(0.05, 0.5 / n_sections)`` — the legacy ``0.05``
    floor, but never stricter than half the uniform share. A factor spread
    uniformly over all ``S`` sections has per-section mass ``~1/S``; capping
    the cutoff at ``0.5 / S`` keeps such factors counted as active in every
    section even for large ``S``, which the fixed ``0.05`` cutoff missed once
    ``1/S`` fell below it (``S >= 20``) — silently reporting genuinely-shared
    factors as inactive (#78). For small ``S`` the ``0.05`` floor is retained,
    so shared factors with mildly uneven mass are not spuriously dropped (a
    bare ``0.5 / S`` cutoff is stricter than ``0.05`` for ``S < 10``).
    """
    n_sections = int(np.unique(section_id).size)
    threshold = min(0.05, 0.5 / n_sections) if n_sections else 0.0
    shared = section_overlap(Z_shared, section_id)
    private = section_overlap(Z_private, section_id)
    return {
        "shared_mean_active_sections": float((shared > threshold).sum(axis=1).mean()) if shared.size else 0.0,
        "private_mean_active_sections": float((private > threshold).sum(axis=1).mean()) if private.size else 0.0,
    }


def reconstruction_error(X: np.ndarray, Z: np.ndarray, W: np.ndarray) -> float:
    """Relative Frobenius reconstruction error ``||X - Z @ W.T||_F / ||X||_F``.

    The standard fidelity measure for a factor model ``X ~= Z @ W.T``. Unlike
    recovery metrics it needs no ground-truth factors, so it is the natural
    score on real data. ``0.0`` = perfect reconstruction. When ``X`` is all
    zeros the denominator is zero and the absolute residual norm is returned.
    """
    Xf = np.asarray(X, dtype=np.float64)
    residual = Xf - np.asarray(Z, dtype=np.float64) @ np.asarray(W, dtype=np.float64).T
    denom = float(np.linalg.norm(Xf))
    if denom == 0.0:
        return float(np.linalg.norm(residual))
    return float(np.linalg.norm(residual) / denom)


def held_out_reconstruction_error(
    X: np.ndarray, Z: np.ndarray, *, holdout: float = 0.2, seed: int = 0
) -> float:
    """Refit loadings on a spot subset and score relative error on the rest.

    Detects over/under-fitting and informs ``K`` selection: nonnegative loadings
    ``W`` are refit by least squares on the training spots, then scored as the
    relative Frobenius error on the held-out spots. Returns ``0.0`` when either
    split would be empty (fewer than two spots, or a degenerate ``holdout``).
    """
    Xf = np.asarray(X, dtype=np.float64)
    Zf = np.asarray(Z, dtype=np.float64)
    n = Xf.shape[0]
    if n < 2:
        return 0.0
    n_test = max(1, int(round(holdout * n)))
    if n_test >= n:
        return 0.0
    perm = np.random.default_rng(seed).permutation(n)
    test_idx, train_idx = perm[:n_test], perm[n_test:]
    coef, *_ = np.linalg.lstsq(Zf[train_idx], Xf[train_idx], rcond=None)
    W = np.clip(coef.T, 0.0, None)
    return reconstruction_error(Xf[test_idx], Zf[test_idx], W)


def morans_i(values: np.ndarray, edges: np.ndarray) -> float:
    """Compute Moran's I on graph edges for a numeric score.

    Note: ``morans_i`` is mean-centering and assumes ``values`` is a numeric
    score. Feeding it integer cluster label codes makes the result depend
    on the specific code assigned to each cluster (relabel-sensitive).
    For categorical labels, use :func:`label_invariant_cluster_coherence`.
    """
    x = values.astype(np.float64)
    n = x.size
    if n == 0 or edges.size == 0:
        return 0.0
    centered = x - x.mean()
    denom = float(np.sum(centered**2))
    if denom == 0.0:
        return 0.0
    src, dst = edges
    w = src.size
    return float((n / w) * np.sum(centered[src] * centered[dst]) / denom)


def label_invariant_cluster_coherence(labels: np.ndarray, edges: np.ndarray) -> float:
    """Label-invariant cluster-coherence metric for categorical cluster labels.

    Computes the mean Moran's I across one-hot indicator vectors per class:
    for each class ``c`` present in ``labels``, treat ``1[labels == c]`` as a
    numeric score and compute :func:`morans_i`; return the mean across
    classes.

    This is invariant under any bijective relabel: the *set* of indicator
    vectors does not depend on the integer codes assigned to clusters, so
    permuting labels (e.g., swapping 0 ↔ 1) leaves the metric unchanged.
    """
    if labels.size == 0 or edges.size == 0:
        return 0.0
    classes = np.unique(labels)
    if classes.size == 0:
        return 0.0
    scores = [morans_i((labels == c).astype(np.float64), edges) for c in classes]
    return float(np.mean(scores))


def adjusted_rand_index(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """Dependency-free adjusted Rand index.

    Degenerate inputs return ``float('nan')`` rather than the misleading
    ``1.0`` ("perfect"). Specifically: ``n < 2`` (no pairs to compare) and
    the all-one-cluster case (``denom == 0``) are signalled as
    not-evaluable rather than as a flawless recovery. See #79.
    """
    if labels_true.shape[0] != labels_pred.shape[0]:
        raise ValueError("label arrays must have equal length")
    n = labels_true.size
    if n < 2:
        return float("nan")
    true_vals, true_inv = np.unique(labels_true, return_inverse=True)
    pred_vals, pred_inv = np.unique(labels_pred, return_inverse=True)
    table = np.zeros((true_vals.size, pred_vals.size), dtype=np.int64)
    np.add.at(table, (true_inv.ravel(), pred_inv.ravel()), 1)
    sum_comb = _comb2(table).sum()
    row_comb = _comb2(table.sum(axis=1)).sum()
    col_comb = _comb2(table.sum(axis=0)).sum()
    total = n * (n - 1) / 2
    expected = row_comb * col_comb / total if total else 0.0
    maximum = 0.5 * (row_comb + col_comb)
    denom = maximum - expected
    if denom == 0:
        return float("nan")
    return float((sum_comb - expected) / denom)


def normalized_mutual_information(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """Normalized mutual information between two label partitions.

    ``NMI = I(U; V) / mean(H(U), H(V))`` with the arithmetic mean of the two
    label entropies as the normalizer (the common convention). The result lies
    in ``[0, 1]``: ``1.0`` for partitions that are identical up to relabeling,
    ``0.0`` when one partition is independent of the other.

    Degenerate cases mirror :func:`adjusted_rand_index`: ``n < 2`` (no pairs)
    returns ``float('nan')`` (not evaluable). When *both* partitions collapse
    to a single cluster the normalizer is zero but the partitions are trivially
    identical, so ``1.0`` is returned; when only one collapses, mutual
    information is zero and ``NMI`` is ``0.0``.
    """
    if labels_true.shape[0] != labels_pred.shape[0]:
        raise ValueError("label arrays must have equal length")
    n = labels_true.size
    if n < 2:
        return float("nan")
    table = _contingency(labels_true, labels_pred)
    mi = _mutual_information(table, n)
    h_true = _entropy(table.sum(axis=1), n)
    h_pred = _entropy(table.sum(axis=0), n)
    denom = 0.5 * (h_true + h_pred)
    if denom == 0.0:
        # Both partitions are a single cluster -> trivially identical.
        return 1.0
    return float(min(max(mi / denom, 0.0), 1.0))


def adjusted_mutual_information(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """Adjusted mutual information between two label partitions.

    ``AMI = (MI - E[MI]) / (mean(H(U), H(V)) - E[MI])`` with the expectation
    taken over the hypergeometric model of randomness (Vinh et al., 2010) and
    the arithmetic mean of the label entropies as the normalizer (matching the
    convention of :func:`normalized_mutual_information`). Unlike NMI, AMI
    corrects for the chance agreement that grows with the number of clusters, so
    it is the appropriate concordance measure when the two partitions have
    differing cluster counts. ``1.0`` for identical partitions (up to
    relabeling); ``~0.0`` for independent partitions (can be slightly negative).

    Degenerate cases mirror :func:`adjusted_rand_index`: ``n < 2`` returns
    ``float('nan')``; when both partitions collapse to a single cluster they are
    trivially identical and ``1.0`` is returned.
    """
    if labels_true.shape[0] != labels_pred.shape[0]:
        raise ValueError("label arrays must have equal length")
    n = labels_true.size
    if n < 2:
        return float("nan")
    table = _contingency(labels_true, labels_pred)
    if table.shape[0] == 1 and table.shape[1] == 1:
        # Both partitions are a single cluster -> trivially identical.
        return 1.0
    mi = _mutual_information(table, n)
    emi = _expected_mutual_information(table, n)
    h_true = _entropy(table.sum(axis=1), n)
    h_pred = _entropy(table.sum(axis=0), n)
    normalizer = 0.5 * (h_true + h_pred)
    denom = normalizer - emi
    # Guard against a vanishing/negative denominator (MI, E[MI] and the
    # normalizer nearly coincide) so the ratio stays finite, matching the
    # convention used by scikit-learn's adjusted_mutual_info_score.
    eps = float(np.finfo("float64").eps)
    denom = min(denom, -eps) if denom < 0 else max(denom, eps)
    return float((mi - emi) / denom)


def silhouette(embedding: np.ndarray, labels: np.ndarray) -> float:
    """Mean silhouette coefficient of ``labels`` over a Euclidean ``embedding``.

    For each sample, ``s = (b - a) / max(a, b)`` where ``a`` is the mean
    distance to other samples in its own cluster and ``b`` is the smallest mean
    distance to samples of any other cluster. Singleton clusters contribute
    ``s = 0`` (``a`` undefined), matching the standard convention. The score is
    averaged over all samples and lies in ``[-1, 1]``; higher means denser,
    better-separated clusters.

    Returns ``float('nan')`` when the number of distinct labels is ``< 2`` or
    ``>= n`` (silhouette is undefined for a single cluster or all-singletons).
    """
    X = np.asarray(embedding, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("embedding must be 2D")
    labels = np.asarray(labels)
    n = X.shape[0]
    if labels.shape[0] != n:
        raise ValueError("labels length must match embedding rows")
    classes = np.unique(labels)
    if classes.size < 2 or classes.size >= n:
        return float("nan")
    dist = np.sqrt(np.maximum(_pairwise_sq_dists(X), 0.0))
    masks = {c: labels == c for c in classes}
    sizes = {c: int(m.sum()) for c, m in masks.items()}
    sil = np.zeros(n, dtype=np.float64)
    for i in range(n):
        ci = labels[i]
        a = dist[i, masks[ci]].sum() / (sizes[ci] - 1) if sizes[ci] > 1 else 0.0
        b = min(dist[i, masks[c]].mean() for c in classes if c != ci)
        denom = max(a, b)
        sil[i] = (b - a) / denom if denom > 0 else 0.0
    return float(sil.mean())


def calinski_harabasz(embedding: np.ndarray, labels: np.ndarray) -> float:
    """Calinski-Harabasz index (variance-ratio criterion) of a clustering.

    ``CH = (tr(B) / (k - 1)) / (tr(W) / (n - k))`` where ``B`` is the
    between-cluster dispersion (weighted squared distances of cluster centroids
    to the global mean) and ``W`` is the within-cluster dispersion (squared
    distances of samples to their centroid). Larger is better.

    Returns ``float('nan')`` when ``k < 2`` or ``k >= n`` (the ratio is
    undefined), and likewise when the within-cluster dispersion is zero
    (perfectly coincident points) so no spurious infinity is reported.
    """
    X = np.asarray(embedding, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("embedding must be 2D")
    labels = np.asarray(labels)
    n = X.shape[0]
    if labels.shape[0] != n:
        raise ValueError("labels length must match embedding rows")
    classes = np.unique(labels)
    k = classes.size
    if k < 2 or k >= n:
        return float("nan")
    overall = X.mean(axis=0)
    between = 0.0
    within = 0.0
    for c in classes:
        Xc = X[labels == c]
        centroid = Xc.mean(axis=0)
        between += Xc.shape[0] * float(np.sum((centroid - overall) ** 2))
        within += float(np.sum((Xc - centroid) ** 2))
    if within == 0.0:
        return float("nan")
    return float((between / (k - 1)) * ((n - k) / within))


def boundary_precision(labels_true: np.ndarray, labels_pred: np.ndarray, edges: np.ndarray) -> float:
    """Precision of predicted domain-boundary spots vs ground-truth boundaries.

    A spot is a "boundary" spot when at least one of its spatial neighbors (via
    the ``edges`` adjacency, a ``(2, n_edges)`` COO array) carries a different
    domain label. Treating the predicted boundary set as positives,
    ``precision = |GT_boundary & pred_boundary| / |pred_boundary|``. Returns
    ``float('nan')`` when no boundary is predicted (precision undefined).
    """
    gt = _boundary_mask(labels_true, edges)
    pred = _boundary_mask(_check_pair(labels_true, labels_pred), edges)
    denom = int(pred.sum())
    if denom == 0:
        return float("nan")
    return float(int(np.count_nonzero(gt & pred)) / denom)


def boundary_recall(labels_true: np.ndarray, labels_pred: np.ndarray, edges: np.ndarray) -> float:
    """Recall of predicted domain-boundary spots vs ground-truth boundaries.

    With the boundary definition of :func:`boundary_precision`,
    ``recall = |GT_boundary & pred_boundary| / |GT_boundary|``. Returns
    ``float('nan')`` when the ground truth has no boundary spots (recall
    undefined).
    """
    gt = _boundary_mask(labels_true, edges)
    pred = _boundary_mask(_check_pair(labels_true, labels_pred), edges)
    denom = int(gt.sum())
    if denom == 0:
        return float("nan")
    return float(int(np.count_nonzero(gt & pred)) / denom)


def boundary_f1(labels_true: np.ndarray, labels_pred: np.ndarray, edges: np.ndarray) -> float:
    """F1 of predicted vs ground-truth domain-boundary spots.

    Harmonic mean of :func:`boundary_precision` and :func:`boundary_recall`,
    computed directly as ``2 * TP / (|pred_boundary| + |GT_boundary|)`` so it is
    stable when one side is empty. Returns ``float('nan')`` only when neither
    partition produces any boundary spots (F1 undefined).
    """
    gt = _boundary_mask(labels_true, edges)
    pred = _boundary_mask(_check_pair(labels_true, labels_pred), edges)
    tp = int(np.count_nonzero(gt & pred))
    denom = int(gt.sum()) + int(pred.sum())
    if denom == 0:
        return float("nan")
    return float(2.0 * tp / denom)


def weighted_dice(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """GT-size-weighted Dice overlap after best-overlap domain matching.

    Each ground-truth domain ``A`` is matched to the predicted domain ``B`` with
    which it shares the most spots (best-overlap assignment). The per-domain
    Dice coefficient is ``2 |A & B| / (|A| + |B|)``, and the reported score is
    the average over GT domains weighted by ``|A|`` (GT domain size). The result
    lies in ``[0, 1]``; ``1.0`` iff predicted and GT partitions coincide.
    Returns ``float('nan')`` for empty input.
    """
    if labels_true.shape[0] != labels_pred.shape[0]:
        raise ValueError("label arrays must have equal length")
    if labels_true.size == 0:
        return float("nan")
    table = _contingency(labels_true, labels_pred)
    gt_sizes = table.sum(axis=1).astype(np.float64)
    pred_sizes = table.sum(axis=0).astype(np.float64)
    best_pred = table.argmax(axis=1)
    inter = table[np.arange(table.shape[0]), best_pred].astype(np.float64)
    matched = pred_sizes[best_pred]
    dice = 2.0 * inter / np.maximum(gt_sizes + matched, 1.0)
    total = float(gt_sizes.sum())
    if total == 0.0:
        return float("nan")
    return float(np.sum(gt_sizes * dice) / total)


def _contingency(labels_true: np.ndarray, labels_pred: np.ndarray) -> np.ndarray:
    """Integer contingency table ``table[i, j] = |true==i & pred==j|`` (#100 style)."""
    _, true_inv = np.unique(labels_true, return_inverse=True)
    _, pred_inv = np.unique(labels_pred, return_inverse=True)
    table = np.zeros((int(true_inv.max()) + 1, int(pred_inv.max()) + 1), dtype=np.int64)
    np.add.at(table, (true_inv.ravel(), pred_inv.ravel()), 1)
    return table


def _mutual_information(table: np.ndarray, n: int) -> float:
    """Mutual information (nats) of a contingency ``table`` over ``n`` samples."""
    nz = table > 0
    if not nz.any():
        return 0.0
    nij = table[nz].astype(np.float64)
    outer = (table.sum(axis=1, keepdims=True) @ table.sum(axis=0, keepdims=True))[nz].astype(np.float64)
    mi = float(np.sum((nij / n) * (np.log(nij * n) - np.log(outer))))
    return max(mi, 0.0)


def _entropy(counts: np.ndarray, n: int) -> float:
    """Shannon entropy (nats) of a label-size distribution given ``n`` samples."""
    counts = counts[counts > 0].astype(np.float64)
    if counts.size == 0:
        return 0.0
    p = counts / n
    return float(-np.sum(p * np.log(p)))


def _expected_mutual_information(table: np.ndarray, n: int) -> float:
    """Expected mutual information (nats) under the hypergeometric null.

    The Vinh et al. (2010) closed form used to chance-correct AMI: the
    expectation of :func:`_mutual_information` over all contingency tables that
    share the row/column margins of ``table``. Combinatorial weights are
    evaluated in log space via ``lgamma`` for numerical stability.
    """
    a = table.sum(axis=1)  # row (true) margins
    b = table.sum(axis=0)  # column (pred) margins
    emi = 0.0
    log_n = log(n)
    lg_n1 = lgamma(n + 1)
    for ai_v in a:
        ai = int(ai_v)
        if ai == 0:
            continue
        for bj_v in b:
            bj = int(bj_v)
            if bj == 0:
                continue
            start = max(1, ai + bj - n)
            end = min(ai, bj)
            base = (
                lgamma(ai + 1) + lgamma(bj + 1)
                + lgamma(n - ai + 1) + lgamma(n - bj + 1)
                - lg_n1
            )
            for nij in range(start, end + 1):
                term = (nij / n) * (log_n + log(nij) - log(ai) - log(bj))
                log_weight = (
                    base
                    - lgamma(nij + 1) - lgamma(ai - nij + 1)
                    - lgamma(bj - nij + 1) - lgamma(n - ai - bj + nij + 1)
                )
                emi += term * exp(log_weight)
    return emi


def _pairwise_sq_dists(X: np.ndarray) -> np.ndarray:
    """Dense ``(n, n)`` matrix of squared Euclidean distances among rows of ``X``."""
    sq = np.sum(X * X, axis=1)
    return sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)


def _boundary_mask(labels: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Boolean per-spot mask: ``True`` if any graph neighbor has a different label."""
    n = labels.shape[0]
    mask = np.zeros(n, dtype=bool)
    if edges.size == 0:
        return mask
    src, dst = edges
    diff = labels[src] != labels[dst]
    mask[src[diff]] = True
    mask[dst[diff]] = True
    return mask


def _check_pair(labels_true: np.ndarray, labels_pred: np.ndarray) -> np.ndarray:
    """Validate matching lengths and return ``labels_pred`` (for boundary helpers)."""
    if labels_true.shape[0] != labels_pred.shape[0]:
        raise ValueError("label arrays must have equal length")
    return labels_pred


def _abs_corr_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape[0] != b.shape[0]:
        raise ValueError("factor matrices must have the same row count")
    a0 = a - a.mean(axis=0, keepdims=True)
    b0 = b - b.mean(axis=0, keepdims=True)
    denom = np.linalg.norm(a0, axis=0)[:, None] * np.linalg.norm(b0, axis=0)[None, :]
    corr = (a0.T @ b0) / np.maximum(denom, 1e-12)
    return np.clip(np.abs(corr), 0.0, 1.0)


def _comb2(x: np.ndarray) -> np.ndarray:
    return x * (x - 1) / 2
