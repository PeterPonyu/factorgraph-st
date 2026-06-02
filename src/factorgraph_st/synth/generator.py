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
    edges: np.ndarray      # (2, n_edges) int64
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

    Pins global RNG state via ``set_seed(seed)`` so any downstream code that
    reads ``random`` / ``np.random`` / ``torch`` globals (notebooks, scripts)
    sees the same seed as the explicit ``np.random.default_rng(seed)`` used
    locally. Closes the determinism gap reported in #87.
    """
    from factorgraph_st.repro import set_seed
    set_seed(seed)
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
    # Clip additive Gaussian noise so X stays on the nonneg half-line.
    # The nonnegative mean (W, Z >= 0 by construction) plus centered noise
    # would otherwise push entries below 0, silently violating the count-data
    # contract that downstream models / validators assume.
    X = np.clip(mean + rng.normal(0.0, noise_sigma, size=mean.shape), 0.0, None).astype(np.float32)

    coords = rng.uniform(0.0, 1.0, size=(n_spots, 2)).astype(np.float32)
    edges = _build_knn_edges_per_section(coords, section_id, k=k_nn)

    domain_id = _assign_spatial_domains(
        coords=coords,
        section_id=section_id,
        n_domains=n_domains,
        seed=seed,
    )

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
    if k < 0:
        raise ValueError(f"k_nn must be non-negative; got {k}")
    if k == 0 or coords.shape[0] == 0:
        return np.empty((2, 0), dtype=np.int64)

    src_chunks: list[np.ndarray] = []
    dst_chunks: list[np.ndarray] = []
    for s in np.unique(section_id):
        idx = np.where(section_id == s)[0]
        pts = coords[idx]
        if pts.shape[0] < 2:
            continue
        section_k = min(k, pts.shape[0] - 1)
        d2 = np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=-1)
        np.fill_diagonal(d2, np.inf)
        nn = np.argpartition(d2, section_k - 1, axis=1)[:, :section_k]
        rows = np.repeat(np.arange(pts.shape[0]), section_k)
        cols = nn.flatten()
        src_chunks.append(idx[rows])
        dst_chunks.append(idx[cols])
    if not src_chunks:
        return np.empty((2, 0), dtype=np.int64)
    return np.stack([np.concatenate(src_chunks), np.concatenate(dst_chunks)]).astype(np.int64)


def _assign_spatial_domains(
    coords: np.ndarray, section_id: np.ndarray, n_domains: int, seed: int
) -> np.ndarray:
    """Assign deterministic, section-aware spatial domains independent of factor count.

    Each section is partitioned independently: within a section the spots are
    projected onto a **section-specific** random direction and split into up to
    ``n_domains`` contiguous quantile bins. The partition is therefore genuinely
    section-aware — the spatial layout of the domains differs from section to
    section — while the label vocabulary stays bounded by ``n_domains`` (every
    section reuses labels ``0..n_labels-1``), matching the documented
    "produces up to ``n_domains`` labels" contract.

    The previous implementation added a single global ``0.01 * section_id`` term
    to one global projection, which was negligible (~8% of the projection
    spread and a constant per-section offset that almost never reordered spots),
    so the ground-truth domains were effectively section-independent despite the
    "section-aware" docstring (#74). Per-section partitioning makes the section
    dependence real and testable, and yields spatially coherent domains on each
    section's own kNN graph.
    """
    n_spots = coords.shape[0]
    if n_domains <= 0 or n_spots == 0:
        return np.zeros(n_spots, dtype=np.int64)
    n_labels = min(int(n_domains), n_spots)
    labels = np.zeros(n_spots, dtype=np.int64)
    for s in np.unique(section_id):
        idx = np.where(section_id == s)[0]
        m = idx.size
        if m == 0:
            continue
        # Section-specific RNG so every section gets a distinct projection
        # direction; deterministic in ``seed`` and the section index.
        rng = np.random.default_rng(int(seed) + int(s) + 1)
        weights = rng.normal(size=coords.shape[1])
        score = coords[idx] @ weights
        order = np.argsort(score, kind="mergesort")
        k = min(n_labels, m)
        sec_labels = np.empty(m, dtype=np.int64)
        sec_labels[order] = np.arange(m, dtype=np.int64) * k // m
        labels[idx] = sec_labels
    return labels
