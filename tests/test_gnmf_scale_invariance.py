"""Scale-invariance property test for the learned GNMF model (#392).

THE PROPERTY (the fix's justification): the recovered spatial-domain **partition**
must not depend on a global rescaling of the expression matrix. Feeding raw counts
vs CPM vs log1p of the *same* data — i.e. ``X`` vs ``c·X`` for a constant ``c`` — must
yield the same domains. A spatial-domain method whose output flips when you change
arbitrary units is wrong, regardless of how it scores against any annotation.

Why it was broken: GNMF minimizes ``‖X − HWᵀ‖²_F + λ·Tr(HᵀLH)``. Under ``X → cX`` the
reconstruction term scales as ``c²`` while the Laplacian term scales as ``c`` (since
``H → √c·H``), so a *constant* ``λ`` shifts the recon-vs-smoothness balance and changes
the partition. The fix scales ``λ`` by a degree-1-homogeneous statistic of ``X``
(``λ_eff = λ·RMS(X)``), which exactly compensates the mismatch; combined with the
z-normalization of ``H`` in ``_cluster_domains`` this makes the partition invariant.

These tests assert the post-fix invariant. On the buggy ``main`` (constant ``λ``) the
``test_partition_invariant_to_global_scale`` case FAILS — that failure is the recorded
evidence that the bug is real (see #392).
"""

from __future__ import annotations

import numpy as np
import pytest

from factorgraph_st.eval.metrics import adjusted_rand_index
from factorgraph_st.model.learned import fit_gnmf, fit_transform_gnmf


def _planted_instance(grid: int = 12, n_domains: int = 3, noise: float = 0.05, seed: int = 0):
    """Tiny planted spatial instance: vertical stripes activating disjoint gene blocks.

    Returns ``(X, edges, domain_true)``. Self-contained (no cross-test import).
    """
    rng = np.random.default_rng(seed)
    xs, ys = np.meshgrid(np.arange(grid), np.arange(grid))
    coords = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)
    n = coords.shape[0]
    domain = (coords[:, 0].astype(np.int64) * n_domains // grid).astype(np.int64)

    k = n_domains
    Z = np.full((n, k), 0.05, dtype=np.float64)
    for d in range(k):
        Z[domain == d, d] += 1.0
    Z += rng.uniform(0.0, noise, size=Z.shape)

    genes_per_factor = 10
    n_genes = genes_per_factor * k
    W = np.zeros((n_genes, k), dtype=np.float64)
    for f in range(k):
        W[f * genes_per_factor : (f + 1) * genes_per_factor, f] = rng.uniform(0.5, 1.5, genes_per_factor)
    X = (Z @ W.T + rng.uniform(0.0, noise, size=(n, n_genes))).astype(np.float32)

    d2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    knn = 6
    nn = np.argpartition(d2, knn - 1, axis=1)[:, :knn]
    src = np.repeat(np.arange(n), knn)
    dst = nn.ravel()
    edges = np.stack([np.concatenate([src, dst]), np.concatenate([dst, src])]).astype(np.int64)
    return X, edges, domain


@pytest.mark.parametrize("c", [0.001, 0.5, 10.0, 1000.0])
def test_partition_invariant_to_global_scale(c: float):
    """fit_transform_gnmf(c·X) recovers the SAME partition as fit_transform_gnmf(X).

    This is the core #392 invariant. ARI between the two partitions must be exactly
    1.0 (identical up to relabeling). FAILS on the buggy constant-λ code.
    """
    X, edges, _ = _planted_instance(seed=0)
    base, _ = fit_transform_gnmf(X, edges, K_shared=2, K_private=1, n_domains=3, lam=1.0, n_iter=300, seed=0)
    scaled, _ = fit_transform_gnmf(
        (c * X.astype(np.float64)), edges, K_shared=2, K_private=1, n_domains=3, lam=1.0, n_iter=300, seed=0
    )
    assert adjusted_rand_index(base.domain_id, scaled.domain_id) == pytest.approx(1.0)


def test_factor_scores_scale_as_sqrt_c():
    """fit_gnmf on c·X yields H ≈ √c·H(X): the homogeneity the fix restores.

    A direct check of the mechanism behind the partition invariance — the H update
    is degree-1/2 homogeneous in X once λ is scaled by RMS(X).
    """
    X, edges, _ = _planted_instance(seed=1)
    c = 100.0
    base = fit_gnmf(X, edges, n_factors=3, lam=1.0, n_iter=200, seed=3)
    scaled = fit_gnmf(c * X.astype(np.float64), edges, n_factors=3, lam=1.0, n_iter=200, seed=3)
    ratio = scaled.H / np.maximum(base.H, 1e-9)
    # Off the (near-)zero entries, the ratio concentrates at √c.
    mask = base.H > 1e-3 * float(base.H.max())
    assert np.allclose(ratio[mask], np.sqrt(c), rtol=0.05, atol=0.0)


def test_invariance_holds_across_seeds():
    """The partition invariant holds for multiple model-init seeds, not just one."""
    X, edges, _ = _planted_instance(seed=2)
    for seed in (0, 1, 2):
        base, _ = fit_transform_gnmf(X, edges, K_shared=2, K_private=1, n_domains=3, lam=1.0, n_iter=250, seed=seed)
        scaled, _ = fit_transform_gnmf(
            42.0 * X.astype(np.float64), edges, K_shared=2, K_private=1, n_domains=3, lam=1.0, n_iter=250, seed=seed
        )
        assert adjusted_rand_index(base.domain_id, scaled.domain_id) == pytest.approx(1.0)
