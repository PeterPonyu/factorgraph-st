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
) -> FactorGraphOutputs:
    """Decode deterministic nonnegative factors from ``H`` and ``X``.

    This is a lightweight baseline implementation that satisfies the MVP schema:
    activations are rectified embeddings and loadings are least-squares estimates
    clipped to nonnegative values.
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
    domain_id = _cluster_domains(H, n_domains).astype(np.int64, copy=False)
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
    outputs = decode_factors(X, H, K_shared=K_shared, K_private=K_private, n_domains=n_domains)
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
    if Z.shape[0] == 0:
        return np.empty((X.shape[1], Z.shape[1]), dtype=np.float32)
    coef, *_ = np.linalg.lstsq(Z.astype(np.float64), X.astype(np.float64), rcond=None)
    return np.clip(coef.T, 0.0, None).astype(np.float32)


def _cluster_domains(H: np.ndarray, n_domains: int) -> np.ndarray:
    if H.shape[0] == 0:
        return np.empty(0, dtype=np.int64)
    n_labels = min(max(int(n_domains), 1), H.shape[0])
    score = H[:, 0] if H.shape[1] else np.arange(H.shape[0], dtype=np.float32)
    order = np.argsort(score, kind="mergesort")
    labels = np.empty(H.shape[0], dtype=np.int64)
    labels[order] = np.arange(H.shape[0], dtype=np.int64) * n_labels // H.shape[0]
    return labels
