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
    """Convert ``H`` into a nonnegative per-column-normalized factor basis.

    Constant or near-constant columns (``std < 1e-12``) are explicitly zeroed
    rather than divided by an eps floor. The previous ``raw / max(scale, 1e-6)
    + 1e-6`` formulation amplified microscopic input variation by up to ~1e6×
    on near-constant columns and left a 1e-6 bias term on truly constant
    columns — silently corrupting downstream loadings and clustering with a
    fake signal. See #85.
    """
    if H.shape[0] == 0:
        return np.empty((0, n_components), dtype=np.float32)
    if H.shape[1] >= n_components:
        raw = H[:, :n_components]
    else:
        raw = np.pad(H, ((0, 0), (0, n_components - H.shape[1])))
    raw = raw - raw.min(axis=0, keepdims=True)
    scale = raw.std(axis=0, keepdims=True)
    low_var = scale < 1e-12
    safe_scale = np.where(low_var, 1.0, scale)
    basis = raw / safe_scale
    # Zero out columns that carried no information so the eps floor cannot
    # propagate as a fake bias term.
    basis = np.where(np.broadcast_to(low_var, basis.shape), 0.0, basis)
    return basis.astype(np.float32)


def _fit_nonnegative_loadings(X: np.ndarray, Z: np.ndarray) -> np.ndarray:
    if Z.shape[0] == 0:
        # Empty input: return deterministic, finite, nonnegative zeros.
        # np.empty would leak uninitialized memory (often non-finite or
        # negative), violating the nonnegative-loadings contract.
        return np.zeros((X.shape[1], Z.shape[1]), dtype=np.float32)
    coef, *_ = np.linalg.lstsq(Z.astype(np.float64), X.astype(np.float64), rcond=None)
    return np.clip(coef.T, 0.0, None).astype(np.float32)


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
