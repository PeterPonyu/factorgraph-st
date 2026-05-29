"""Synthetic benchmark metrics for FactorGraph-ST."""

from __future__ import annotations

from itertools import permutations

import numpy as np


def matched_factor_correlation(estimated: np.ndarray, truth: np.ndarray) -> float:
    """Mean absolute Pearson correlation after best one-to-one factor matching."""
    corr = _abs_corr_matrix(estimated, truth)
    if corr.size == 0:
        return 1.0
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


def shared_private_separation(Z_shared: np.ndarray, Z_private: np.ndarray, section_id: np.ndarray) -> dict[str, float]:
    """Return section-spread summaries for shared and private activations."""
    shared = section_overlap(Z_shared, section_id)
    private = section_overlap(Z_private, section_id)
    return {
        "shared_mean_active_sections": float((shared > 0.05).sum(axis=1).mean()) if shared.size else 0.0,
        "private_mean_active_sections": float((private > 0.05).sum(axis=1).mean()) if private.size else 0.0,
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
    """Dependency-free adjusted Rand index."""
    if labels_true.shape[0] != labels_pred.shape[0]:
        raise ValueError("label arrays must have equal length")
    n = labels_true.size
    if n < 2:
        return 1.0
    true_vals, true_inv = np.unique(labels_true, return_inverse=True)
    pred_vals, pred_inv = np.unique(labels_pred, return_inverse=True)
    table = np.zeros((true_vals.size, pred_vals.size), dtype=np.int64)
    for i in range(n):
        table[true_inv[i], pred_inv[i]] += 1
    sum_comb = _comb2(table).sum()
    row_comb = _comb2(table.sum(axis=1)).sum()
    col_comb = _comb2(table.sum(axis=0)).sum()
    total = n * (n - 1) / 2
    expected = row_comb * col_comb / total if total else 0.0
    maximum = 0.5 * (row_comb + col_comb)
    denom = maximum - expected
    if denom == 0:
        return 1.0
    return float((sum_comb - expected) / denom)


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
