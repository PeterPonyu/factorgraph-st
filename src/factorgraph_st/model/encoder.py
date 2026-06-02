"""Numpy section-aware graph encoder for the MVP output ``H``.

The encoder intentionally avoids deep-learning dependencies. It mean-aggregates
neighbor expression through the provided per-section graph, appends coordinates
and a normalized section feature, then projects deterministically to ``d``.
"""

from __future__ import annotations

import numpy as np

from factorgraph_st.schemas import validate_inputs


def encode_graph(
    X: np.ndarray,
    coords: np.ndarray,
    section_id: np.ndarray,
    edges: np.ndarray,
    d: int = 16,
    seed: int = 0,
) -> np.ndarray:
    """Return a deterministic section-aware embedding with shape ``(n_spots, d)``."""
    validate_inputs(X, coords, section_id, edges)
    if d <= 0:
        raise ValueError(f"d must be positive; got {d}")

    n_spots = X.shape[0]
    if n_spots == 0:
        return np.empty((0, d), dtype=np.float32)

    neighbor_sum = np.zeros_like(X, dtype=np.float32)
    degree = np.zeros(n_spots, dtype=np.float32)
    if edges.size:
        src, dst = edges
        np.add.at(neighbor_sum, src, X[dst])
        np.add.at(degree, src, 1.0)
    neighbor_mean = neighbor_sum / np.maximum(degree[:, None], 1.0)

    # Global per-gene mean subtraction (NOT per-spot library-size correction).
    # This assumes the caller already library-size normalized X (see #197: the
    # real-data runner applies sc.pp.normalize_total -> log1p before encoding).
    # On raw counts the column mean is dominated by high-library-size spots and
    # this centering is biased, so callers must normalize upstream.
    centered_x = X - X.mean(axis=0, keepdims=True)
    centered_coords = coords - coords.mean(axis=0, keepdims=True)
    section_scale = max(int(section_id.max(initial=0)), 1)
    section_feature = (section_id.astype(np.float32) / section_scale)[:, None]
    features = np.concatenate(
        [centered_x, neighbor_mean - X.mean(axis=0, keepdims=True), centered_coords, section_feature],
        axis=1,
    )

    rng = np.random.default_rng(seed)
    projection = rng.normal(0.0, 1.0 / np.sqrt(features.shape[1]), size=(features.shape[1], d))
    return (features @ projection).astype(np.float32)
