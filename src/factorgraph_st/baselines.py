"""Pure-numpy clean-room spatial-domain *baseline* (graph smoothing -> PCA -> k-means).

This module provides a standard, license-free spatial-domain clustering baseline
so the supervised domain metrics (ARI + the NMI/Dice/boundary/silhouette/CH suite)
have an apples-to-apples comparison point against the FactorGraph-ST factor models
(the fixed random-projection MVP in ``model/decoder.py`` and the trained GNMF in
``model/learned.py``). It is intentionally simple and model-free:

1. **Graph smoothing.** The (already-normalized) expression matrix ``X`` is
   averaged over the spatial kNN graph for ``n_hops`` propagation steps, each a
   convex blend ``X <- (1 - alpha) * X + alpha * (S @ X)`` where ``S = D^-1 A`` is
   the row-normalized (symmetrized) adjacency. This is the canonical "smooth the
   features over the neighbor graph" idea shared by many spatial-domain methods,
   implemented from first principles in dense numpy (the adjacency is applied as a
   scatter-add so the ``n_spots x n_spots`` matrix is never materialized).
2. **Linear reduction.** The smoothed, column-centered matrix is reduced to
   ``n_components`` dimensions via a numpy SVD (classical PCA scores). No
   randomness — the SVD is deterministic.
3. **k-means domains.** The PCA scores are clustered into ``n_domains`` domains
   with the same deterministic, seeded k-means used elsewhere in the package
   (:func:`factorgraph_st.model.decoder._kmeans`), so domain assignment is
   bit-for-bit reproducible.

Setting ``n_hops = 0`` disables smoothing and yields a **plain PCA + k-means**
baseline (the natural ablation): with smoothing the baseline should beat plain
PCA on planted spatial domains, while a trained factor model should still beat
the smoothed baseline — i.e. this is a sensible *middle* baseline.

The runtime stays numpy-only (no torch, no scipy, no scikit-learn): every step is
dense numpy, matching the package's numpy-only dependency contract.

Brand-neutral by construction: this is a generic graph-smoothing/PCA/k-means
pipeline with no code, identifiers, or narrative copied from any external method.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from factorgraph_st.model.decoder import _kmeans

# Floor for the row-normalization denominator so isolated (degree-0) spots keep
# their own features instead of producing a divide-by-zero NaN.
_EPS = 1e-12


@dataclass
class SpatialSmoothOutputs:
    """Result of the spatial-smoothing domain baseline.

    Attributes
    ----------
    domain_id:
        ``(n_spots,)`` int64 cluster labels in ``0..n_domains-1`` (fewer when the
        budget exceeds the spot count).
    embedding:
        ``(n_spots, n_components)`` float32 PCA scores of the smoothed expression.
        Exposed so the same intrinsic domain-quality metrics (silhouette /
        Calinski-Harabasz) the factor models report can score this baseline.
    """

    domain_id: np.ndarray
    embedding: np.ndarray


def _row_normalized_smooth(
    X: np.ndarray, edges: np.ndarray, *, alpha: float, n_hops: int
) -> np.ndarray:
    """Smooth ``X`` over the symmetrized kNN graph for ``n_hops`` blended steps.

    Each step computes the neighbor mean ``S @ X`` with ``S = D^-1 A`` (the
    row-normalized symmetrized adjacency) via scatter-add, then returns the
    convex blend ``(1 - alpha) * X + alpha * (S @ X)``. ``n_hops = 0`` returns a
    copy of ``X`` unchanged (the plain-PCA ablation). Spots with no edges keep
    their own row (``S`` row is all-zero, so only the ``(1 - alpha) * X`` term
    survives).
    """
    Xs = np.asarray(X, dtype=np.float64).copy()
    if n_hops <= 0 or edges.size == 0:
        return Xs
    n_spots = Xs.shape[0]
    src, dst = edges
    # Symmetrize so the adjacency is undirected regardless of how edges were built.
    src_sym = np.concatenate([src, dst]).astype(np.int64)
    dst_sym = np.concatenate([dst, src]).astype(np.int64)
    degree = np.bincount(src_sym, minlength=n_spots).astype(np.float64)
    inv_degree = np.where(degree > 0.0, 1.0 / np.maximum(degree, _EPS), 0.0)[:, None]
    for _ in range(int(n_hops)):
        neighbor_sum = np.zeros_like(Xs)
        np.add.at(neighbor_sum, src_sym, Xs[dst_sym])
        neighbor_mean = neighbor_sum * inv_degree
        Xs = (1.0 - alpha) * Xs + alpha * neighbor_mean
    return Xs


def _pca_scores(X: np.ndarray, n_components: int) -> np.ndarray:
    """Deterministic PCA scores: column-center, SVD, project onto top components.

    Returns the ``(n_spots, k)`` score matrix ``U[:, :k] * S[:k]`` where ``k`` is
    clamped to ``min(n_components, n_spots, n_features)``. The SVD sign is pinned
    (largest-magnitude loading made positive per component) so the embedding — and
    therefore the downstream k-means — is reproducible across BLAS backends.
    """
    Xc = np.asarray(X, dtype=np.float64)
    Xc = Xc - Xc.mean(axis=0, keepdims=True)
    n_spots, n_features = Xc.shape
    k = max(1, min(int(n_components), n_spots, n_features))
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    U, S, Vt = U[:, :k], S[:k], Vt[:k]
    # Sign convention: make the largest-|value| entry of each right singular
    # vector positive so U/S are deterministic up to floating point.
    signs = np.sign(Vt[np.arange(k), np.argmax(np.abs(Vt), axis=1)])
    signs[signs == 0.0] = 1.0
    scores = U * S[None, :] * signs[None, :]
    return scores.astype(np.float32)


def spatial_smooth_domains(
    X: np.ndarray,
    edges: np.ndarray,
    *,
    n_domains: int = 5,
    n_components: int = 16,
    alpha: float = 0.5,
    n_hops: int = 2,
    seed: int = 0,
    n_init: int = 4,
    max_iter: int = 100,
) -> SpatialSmoothOutputs:
    """Assign spatial domains by graph-smoothing -> PCA -> k-means (numpy-only).

    Parameters
    ----------
    X:
        ``(n_spots, n_genes)`` normalized expression matrix.
    edges:
        ``(2, n_edges)`` int64 spatial kNN edge list (symmetrized internally).
    n_domains:
        Number of domains ``k`` to cluster into (clamped to ``n_spots``).
    n_components:
        PCA dimensionality of the reduced embedding before k-means.
    alpha:
        Smoothing blend weight in ``[0, 1]`` per hop; ``0`` keeps the original
        features, larger values weight the neighbor mean more.
    n_hops:
        Number of graph-smoothing propagation steps. ``0`` => plain PCA + k-means
        (no spatial smoothing), the natural ablation baseline.
    seed:
        Seeds the k-means initialization; identical ``seed`` => identical labels.
    n_init, max_iter:
        k-means restart count and per-restart iteration cap.

    Returns
    -------
    SpatialSmoothOutputs
        The integer ``domain_id`` labels and the float32 PCA ``embedding``.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1]; got {alpha}")
    if n_hops < 0:
        raise ValueError(f"n_hops must be non-negative; got {n_hops}")
    if n_components <= 0:
        raise ValueError(f"n_components must be positive; got {n_components}")

    Xf = np.asarray(X, dtype=np.float64)
    n_spots = Xf.shape[0]
    if n_spots == 0:
        return SpatialSmoothOutputs(
            domain_id=np.zeros(0, dtype=np.int64),
            embedding=np.zeros((0, min(n_components, 1)), dtype=np.float32),
        )

    smoothed = _row_normalized_smooth(Xf, edges, alpha=alpha, n_hops=n_hops)
    embedding = _pca_scores(smoothed, n_components)

    k = max(1, min(int(n_domains), n_spots))
    if k == 1:
        domain_id = np.zeros(n_spots, dtype=np.int64)
    else:
        domain_id = _kmeans(
            embedding.astype(np.float64), k, seed, n_init, max_iter
        ).astype(np.int64)
    return SpatialSmoothOutputs(domain_id=domain_id, embedding=embedding)
