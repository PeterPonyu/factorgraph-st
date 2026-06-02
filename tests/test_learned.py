"""Tests for the trained graph-regularized NMF learned model.

These prove the learned model (a) is deterministic, (b) genuinely learns (the
objective decreases monotonically), and (c) beats the fixed random-projection
encoder on both factor recovery and spatial-domain assignment. All data is
tiny-synthetic so the suite runs in the numpy-only CI with no real training.

On ``origin/main`` ``factorgraph_st.model.learned`` does not exist, so every test
here fails to even import (ImportError) -> passes once the module lands.
"""

from __future__ import annotations

import numpy as np

from factorgraph_st.eval.metrics import adjusted_rand_index, matched_factor_correlation
from factorgraph_st.model.decoder import fit_transform
from factorgraph_st.model.learned import fit_gnmf, fit_transform_gnmf
from factorgraph_st.schemas import validate_outputs


def _planted_spatial_instance(
    grid: int = 12,
    n_domains: int = 3,
    genes_per_factor: int = 10,
    noise: float = 0.05,
    seed: int = 0,
):
    """Synthetic data with planted nonnegative, spatially coherent factors.

    Spots lie on a ``grid x grid`` lattice partitioned into ``n_domains`` vertical
    stripes (contiguous, elongated domains). Each stripe activates one nonnegative
    factor that loads a disjoint gene block, so the factors are both spatially
    smooth (neighbors share a stripe) and expression-identifiable. The vertical
    stripes are deliberately elongated, a layout Euclidean k-means on coordinates
    resolves poorly — so a model that recovers the expression factors should beat
    a coordinate-driven baseline on domain assignment.

    Returns ``(X, coords, section_id, edges, Z_true, domain_true)``.
    """
    rng = np.random.default_rng(seed)
    xs, ys = np.meshgrid(np.arange(grid), np.arange(grid))
    coords = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32)
    n = coords.shape[0]

    col = coords[:, 0].astype(np.int64)
    domain = (col * n_domains // grid).astype(np.int64)

    k = n_domains
    Z = np.full((n, k), 0.05, dtype=np.float64)
    for d in range(k):
        Z[domain == d, d] += 1.0
    Z += rng.uniform(0.0, noise, size=Z.shape)

    n_genes = genes_per_factor * k
    W = np.zeros((n_genes, k), dtype=np.float64)
    for f in range(k):
        W[f * genes_per_factor : (f + 1) * genes_per_factor, f] = rng.uniform(
            0.5, 1.5, genes_per_factor
        )

    X = (Z @ W.T + rng.uniform(0.0, noise, size=(n, n_genes))).astype(np.float32)

    # Symmetric kNN spatial graph on the lattice coordinates.
    d2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    knn = 6
    nn = np.argpartition(d2, knn - 1, axis=1)[:, :knn]
    src = np.repeat(np.arange(n), knn)
    dst = nn.ravel()
    edges = np.stack([np.concatenate([src, dst]), np.concatenate([dst, src])]).astype(np.int64)

    section_id = np.zeros(n, dtype=np.int64)
    return X, coords, section_id, edges, Z.astype(np.float32), domain


def test_fit_gnmf_is_deterministic():
    X, _, _, edges, _, _ = _planted_spatial_instance(seed=1)
    a = fit_gnmf(X, edges, n_factors=3, lam=1.0, n_iter=120, seed=7)
    b = fit_gnmf(X, edges, n_factors=3, lam=1.0, n_iter=120, seed=7)
    assert np.array_equal(a.H, b.H)
    assert np.array_equal(a.W, b.W)
    assert np.array_equal(a.objective, b.objective)


def test_gnmf_objective_decreases_monotonically():
    X, _, _, edges, _, _ = _planted_spatial_instance(seed=2)
    res = fit_gnmf(X, edges, n_factors=3, lam=1.0, n_iter=200, seed=0)
    obj = res.objective
    assert obj.size >= 2
    # Multiplicative updates are non-increasing; allow only floating-point slack.
    slack = 1e-6 * (abs(obj[0]) + 1.0)
    assert np.all(np.diff(obj) <= slack)
    # And the fit makes real progress (not a no-op).
    assert obj[-1] < obj[0]


def test_gnmf_recovers_factors_better_than_projection():
    X, coords, section_id, edges, Z_true, _ = _planted_spatial_instance(seed=0)

    out_g, _ = fit_transform_gnmf(
        X, edges, K_shared=2, K_private=1, n_domains=3, lam=1.0, n_iter=300, seed=0
    )
    Z_g = np.concatenate([out_g.Z_shared, out_g.Z_private], axis=1)
    corr_g = matched_factor_correlation(Z_g, Z_true)

    out_p = fit_transform(
        X, coords, section_id, edges, d=6, K_shared=2, K_private=1, n_domains=3, seed=0
    )
    Z_p = np.concatenate([out_p.Z_shared, out_p.Z_private], axis=1)
    corr_p = matched_factor_correlation(Z_p, Z_true)

    assert corr_g > corr_p
    # The learned model should recover the planted factors near-perfectly.
    assert corr_g > 0.9


def test_gnmf_domains_beat_projection_ari():
    X, coords, section_id, edges, _, domain_true = _planted_spatial_instance(seed=0)

    out_g, _ = fit_transform_gnmf(
        X, edges, K_shared=2, K_private=1, n_domains=3, lam=1.0, n_iter=300, seed=0
    )
    ari_g = adjusted_rand_index(domain_true, out_g.domain_id)

    out_p = fit_transform(
        X, coords, section_id, edges, d=6, K_shared=2, K_private=1, n_domains=3, seed=0
    )
    ari_p = adjusted_rand_index(domain_true, out_p.domain_id)

    assert ari_g > ari_p


def test_fit_transform_gnmf_outputs_pass_schema():
    X, _, _, edges, _, _ = _planted_spatial_instance(seed=3)
    out, res = fit_transform_gnmf(
        X, edges, K_shared=2, K_private=1, n_domains=3, lam=1.0, n_iter=150, seed=5
    )
    validate_outputs(
        out.H, out.W, out.Z_shared, out.Z_private, out.domain_id, X.shape[0], X.shape[1]
    )
    # Contract: k = K_shared + K_private factors, nonnegative everywhere.
    assert out.W.shape[1] == 3
    assert out.Z_shared.shape[1] == 2 and out.Z_private.shape[1] == 1
    assert (out.W >= 0).all() and (out.Z_shared >= 0).all() and (out.Z_private >= 0).all()
    assert res.objective.size == res.n_iter_run + 1
