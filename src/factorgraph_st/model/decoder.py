"""Numpy nonnegative decoder and end-to-end MVP fit/transform helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from factorgraph_st.model.encoder import encode_graph
from factorgraph_st.schemas import validate_outputs


@dataclass
class FactorGraphOutputs:
    H: np.ndarray
    W: np.ndarray
    Z_shared: np.ndarray
    Z_private: np.ndarray
    domain_id: np.ndarray


def decode_factors(
    X: np.ndarray,
    H: np.ndarray,
    K_shared: int = 4,
    K_private: int = 2,
    n_domains: int = 5,
    coords: np.ndarray | None = None,
    seed: int = 0,
) -> FactorGraphOutputs:
    """Decode deterministic nonnegative factors from ``H`` and ``X``.

    This is a lightweight baseline implementation that satisfies the MVP schema:
    activations are rectified embeddings and loadings are least-squares estimates
    clipped to nonnegative values.

    Spatial domains are assigned by a deterministic k-means pass on the
    z-normalized joint feature ``[H | coords]``. Passing ``coords=None``
    falls back to clustering on ``H`` alone (purely embedding-based).
    """
    if K_shared < 0 or K_private < 0:
        raise ValueError("K_shared and K_private must be non-negative")
    K_total = K_shared + K_private
    if K_total <= 0:
        raise ValueError("K_shared + K_private must be positive")

    basis = _positive_basis(H, K_total)
    Z_shared = basis[:, :K_shared].astype(np.float32, copy=False)
    Z_private = basis[:, K_shared:].astype(np.float32, copy=False)
    W = _fit_nonnegative_loadings(X, basis).astype(np.float32, copy=False)
    if coords is None:
        coords = np.zeros((H.shape[0], 0), dtype=np.float32)
    domain_id = _cluster_domains(H, coords, n_domains, seed=seed).astype(np.int64, copy=False)
    return FactorGraphOutputs(H=H, W=W, Z_shared=Z_shared, Z_private=Z_private, domain_id=domain_id)


def fit_transform(
    X: np.ndarray,
    coords: np.ndarray,
    section_id: np.ndarray,
    edges: np.ndarray,
    d: int = 16,
    K_shared: int = 4,
    K_private: int = 2,
    n_domains: int = 5,
    seed: int = 0,
) -> FactorGraphOutputs:
    """Encode inputs, decode nonnegative factors, and validate MVP outputs."""
    H = encode_graph(X, coords, section_id, edges, d=d, seed=seed)
    outputs = decode_factors(
        X,
        H,
        K_shared=K_shared,
        K_private=K_private,
        n_domains=n_domains,
        coords=coords,
        seed=seed,
    )
    validate_outputs(outputs.H, outputs.W, outputs.Z_shared, outputs.Z_private, outputs.domain_id, X.shape[0], X.shape[1])
    return outputs


def _positive_basis(H: np.ndarray, n_components: int) -> np.ndarray:
    if H.shape[0] == 0:
        return np.empty((0, n_components), dtype=np.float32)
    if H.shape[1] >= n_components:
        raw = H[:, :n_components]
    else:
        raw = np.pad(H, ((0, 0), (0, n_components - H.shape[1])))
    raw = raw - raw.min(axis=0, keepdims=True)
    scale = raw.std(axis=0, keepdims=True)
    return (raw / np.maximum(scale, 1e-6) + 1e-6).astype(np.float32)


def _fit_nonnegative_loadings(X: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Solve column-wise nonnegative least-squares for ``W`` given ``Z`` and ``X``.

    For each gene column ``j``, returns the ``W[j, :]`` that minimizes
    ``||X[:, j] - Z @ W[j, :].T||_2`` subject to ``W[j, :] >= 0`` via the
    numpy-only Lawson-Hanson active-set NNLS in :func:`_nnls`. This replaces
    the prior ``clip(lstsq, 0)`` estimator, which is *not* NNLS — clipping an
    unconstrained solution yields a biased, non-optimal ``W`` whenever the
    true optimum has active nonneg constraints. See #83.

    Cost: NNLS is solved independently per gene column, so the loop is
    ``O(n_features)`` solves. Each solve is a small active-set problem over
    ``K = Z.shape[1]`` factors (the active set adds at most ``K`` columns and
    each inner least-squares is ``O(n_spots * K^2)``); since ``K`` is tiny
    (a handful of factors) the per-column work is dominated by the ``Z``
    normal-equation products and the loop scales linearly in the gene count.
    A numpy-only solver keeps the runtime dependency surface at numpy alone.
    """
    if Z.shape[0] == 0:
        # Empty input: return deterministic, finite, nonnegative zeros.
        # np.empty would leak uninitialized memory (often non-finite or
        # negative), violating the nonnegative-loadings contract.
        return np.zeros((X.shape[1], Z.shape[1]), dtype=np.float32)
    Z64 = Z.astype(np.float64, copy=False)
    X64 = X.astype(np.float64, copy=False)
    n_features = X64.shape[1]
    W = np.empty((n_features, Z64.shape[1]), dtype=np.float64)
    for j in range(n_features):
        W[j] = _nnls(Z64, X64[:, j])
    return W.astype(np.float32)


def _nnls(A: np.ndarray, b: np.ndarray, *, tol: float = 1e-10, max_iter: int | None = None) -> np.ndarray:
    """Numpy-only nonnegative least squares: minimize ``||A x - b||_2`` s.t. ``x >= 0``.

    Classic Lawson-Hanson active-set algorithm (matches
    :func:`scipy.optimize.nnls` to floating-point tolerance). Implemented in
    numpy so the runtime package stays numpy-only; ``A`` is the ``(n_spots, K)``
    factor matrix and ``b`` is a single ``(n_spots,)`` gene column.
    """
    A = np.ascontiguousarray(A, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    n = A.shape[1]
    iter_cap = 3 * n if max_iter is None else max_iter
    AtA = A.T @ A
    Atb = A.T @ b
    x = np.zeros(n, dtype=np.float64)
    passive = np.zeros(n, dtype=bool)
    w = Atb - AtA @ x
    outer = 0
    while not passive.all() and np.max(np.where(passive, -np.inf, w)) > tol:
        # Move the most-violating active coordinate into the passive set.
        j = int(np.argmax(np.where(passive, -np.inf, w)))
        passive[j] = True
        while True:
            P = np.where(passive)[0]
            s = np.zeros(n, dtype=np.float64)
            s[P] = np.linalg.lstsq(A[:, P], b, rcond=None)[0]
            if s[P].min() > tol:
                x = s
                break
            # Backtrack toward the unconstrained passive solution until a
            # passive coordinate hits zero, then drop it from the passive set.
            mask = passive & (s <= tol)
            alpha = (x[mask] / (x[mask] - s[mask])).min()
            x = x + alpha * (s - x)
            passive &= x > tol
        w = Atb - AtA @ x
        outer += 1
        if outer > iter_cap:
            break
    return x


def _cluster_domains(
    H: np.ndarray,
    coords: np.ndarray,
    n_domains: int,
    *,
    seed: int = 0,
    n_init: int = 4,
    max_iter: int = 100,
) -> np.ndarray:
    """Assign domain ids by deterministic k-means on z-normalized ``[H | coords]``.

    The previous implementation ignored ``coords`` and partitioned by rank of
    ``H[:, 0]``, producing strip patterns rather than spatial domains. This
    pass uses an actual clustering on the joint feature, with ``n_init``
    restarts picking the lowest-inertia assignment.
    """
    n = H.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    k = max(1, min(int(n_domains), n))
    if k == 1:
        return np.zeros(n, dtype=np.int64)

    parts = []
    for arr in (H, coords):
        if arr.size == 0 or arr.shape[1] == 0:
            continue
        a = np.asarray(arr, dtype=np.float64)
        parts.append((a - a.mean(0, keepdims=True)) / np.maximum(a.std(0, keepdims=True), 1e-6))
    X = np.concatenate(parts, axis=1) if parts else np.zeros((n, 1), dtype=np.float64)

    rng = np.random.default_rng(seed)
    best_inertia, best_labels = np.inf, np.zeros(n, dtype=np.int64)
    for _ in range(max(1, n_init)):
        centers = X[rng.choice(n, size=k, replace=False)].copy()
        labels = np.full(n, -1, dtype=np.int64)
        for _ in range(max_iter):
            dists = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            new_labels = dists.argmin(axis=1).astype(np.int64)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for c in range(k):
                mask = labels == c
                centers[c] = X[mask].mean(0) if mask.any() else X[int(np.argmax(dists.min(1)))]
        inertia = float(((X - centers[labels]) ** 2).sum())
        if inertia < best_inertia:
            best_inertia, best_labels = inertia, labels
    return best_labels
