"""Deterministic synthetic shared/private factor generator.

Schema mirrors ``docs/SYNTHETIC_BENCHMARK.md``: nonnegative gene loadings
``W``, shared activations ``Z_shared``, section-private activations
``Z_private``, and derived ``domain_id``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SynthInstance:
    X: np.ndarray          # (n_spots, n_genes) float32
    coords: np.ndarray     # (n_spots, 2) float32
    section_id: np.ndarray # (n_spots,) int64
    edges: np.ndarray      # (2, n_spots * k_nn) int64
    W: np.ndarray          # (n_genes, K_shared + K_private) float32, >= 0
    Z_shared: np.ndarray   # (n_spots, K_shared) float32, >= 0
    Z_private: np.ndarray  # (n_spots, K_private) float32, >= 0
    domain_id: np.ndarray  # (n_spots,) int64


def generate_instance(
    n_sections: int = 4,
    n_spots_per_section: int = 2000,
    n_genes: int = 500,
    K_shared: int = 4,
    K_private: int = 2,
    noise_sigma: float = 0.5,
    n_domains: int = 5,
    k_nn: int = 8,
    seed: int = 0,
) -> SynthInstance:
    """Build one synthetic shared/private factor instance.

    Each private factor ``k`` is active only on section ``k % n_sections``,
    so private factors live in a strict subset of sections (one section
    each, unless ``K_private > n_sections``).
    """
    rng = np.random.default_rng(seed)
    n_spots = n_sections * n_spots_per_section
    K_total = K_shared + K_private

    W = rng.exponential(scale=1.0, size=(n_genes, K_total)).astype(np.float32)

    Z_shared = rng.exponential(scale=1.0, size=(n_spots, K_shared)).astype(np.float32)

    section_id = np.repeat(np.arange(n_sections), n_spots_per_section).astype(np.int64)
    Z_private = np.zeros((n_spots, K_private), dtype=np.float32)
    for k in range(K_private):
        s = k % n_sections
        mask = section_id == s
        Z_private[mask, k] = rng.exponential(
            scale=1.0, size=int(mask.sum())
        ).astype(np.float32)

    Z = np.concatenate([Z_shared, Z_private], axis=1)
    mean = Z @ W.T
    X = (mean + rng.normal(0.0, noise_sigma, size=mean.shape)).astype(np.float32)

    coords = rng.uniform(0.0, 1.0, size=(n_spots, 2)).astype(np.float32)
    edges = _build_knn_edges_per_section(coords, section_id, k=k_nn)

    domain_id = (np.argmax(Z_shared, axis=1) % max(n_domains, 1)).astype(np.int64)

    return SynthInstance(
        X=X,
        coords=coords,
        section_id=section_id,
        edges=edges,
        W=W,
        Z_shared=Z_shared,
        Z_private=Z_private,
        domain_id=domain_id,
    )


def _build_knn_edges_per_section(
    coords: np.ndarray, section_id: np.ndarray, k: int
) -> np.ndarray:
    src_chunks: list[np.ndarray] = []
    dst_chunks: list[np.ndarray] = []
    for s in np.unique(section_id):
        idx = np.where(section_id == s)[0]
        pts = coords[idx]
        d2 = np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=-1)
        np.fill_diagonal(d2, np.inf)
        nn = np.argpartition(d2, k, axis=1)[:, :k]
        rows = np.repeat(np.arange(pts.shape[0]), k)
        cols = nn.flatten()
        src_chunks.append(idx[rows])
        dst_chunks.append(idx[cols])
    return np.stack([np.concatenate(src_chunks), np.concatenate(dst_chunks)]).astype(np.int64)
