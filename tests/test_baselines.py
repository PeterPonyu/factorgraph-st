"""Tests for the clean-room spatial-smoothing spatial-domain baseline.

These prove the baseline (a) is deterministic and seeded, (b) genuinely uses the
spatial graph (graph smoothing beats the plain PCA ablation on planted spatial
domains), and (c) is a *sensible middle* baseline — meaningfully better than naive
PCA but NOT better than the trained graph-regularized NMF model. All data is
tiny-synthetic so the suite runs in the numpy-only CI with no real data.

On ``origin/main`` ``factorgraph_st.baselines`` does not exist, so every test here
fails to even import (ImportError) -> passes once the module lands.

Planted scenario (the discriminating part): domains are horizontal stripes (vary
with ``y``), while a dominant, spatially-SMOOTH nuisance gradient varies with ``x``
(the orthogonal direction) and loads its own gene block. Per-spot Gaussian noise is
added on top. The three methods then separate cleanly:

* **Plain PCA + k-means** sees raw per-spot noise; its leading variance axes mix
  the nuisance gradient and noise, so it resolves the stripes poorly.
* **Graph smoothing + PCA + k-means** averages the per-spot noise over the kNN
  graph, sharpening the stripe signal -> it beats plain PCA. But the nuisance
  gradient is itself spatially smooth, so smoothing cannot remove it and it caps
  the baseline below the trained model.
* **Graph-regularized NMF** factorizes the nuisance gradient into its own
  nonnegative factor and clusters the stripe factors -> it beats the baseline.
"""

from __future__ import annotations

import numpy as np

from factorgraph_st.baselines import spatial_smooth_domains
from factorgraph_st.eval.metrics import (
    adjusted_rand_index,
    boundary_f1,
    calinski_harabasz,
    normalized_mutual_information,
    silhouette,
    weighted_dice,
)
from factorgraph_st.model.learned import fit_transform_gnmf


def _planted_gradient_instance(
    grid: int = 28,
    n_domains: int = 4,
    genes_per_factor: int = 8,
    noise: float = 1.2,
    grad_amp: float = 2.0,
    knn: int = 6,
    seed: int = 1,
):
    """Planted horizontal-stripe domains under a smooth orthogonal nuisance gradient.

    Returns ``(X, edges, domain_true)``: a ``(grid*grid, n_genes)`` nonnegative
    expression matrix, the symmetric kNN spatial edge list, and the per-spot
    ground-truth domain labels (horizontal stripes).
    """
    rng = np.random.default_rng(seed)
    xs, ys = np.meshgrid(np.arange(grid), np.arange(grid))
    coords = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32)
    n = coords.shape[0]

    # Horizontal stripes (domain varies with the y coordinate).
    domain = (coords[:, 1].astype(np.int64) * n_domains // grid).astype(np.int64)

    k = n_domains
    Z = np.full((n, k), 0.05, dtype=np.float64)
    for d in range(k):
        Z[domain == d, d] += 1.0

    # Dominant, spatially-smooth nuisance gradient along the orthogonal (x) axis.
    grad = grad_amp * (coords[:, 0] / grid)

    n_genes = genes_per_factor * (k + 1)
    W = np.zeros((n_genes, k + 1), dtype=np.float64)
    for f in range(k):
        W[f * genes_per_factor : (f + 1) * genes_per_factor, f] = rng.uniform(
            0.5, 1.5, genes_per_factor
        )
    # The nuisance gradient loads its own disjoint gene block.
    W[k * genes_per_factor : (k + 1) * genes_per_factor, k] = rng.uniform(
        0.5, 1.5, genes_per_factor
    )

    Z_full = np.concatenate([Z, grad[:, None]], axis=1)
    X = (Z_full @ W.T + rng.normal(0.0, noise, size=(n, n_genes))).clip(0.0, None).astype(
        np.float32
    )

    # Symmetric kNN spatial graph on the lattice coordinates.
    d2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    nn = np.argpartition(d2, knn - 1, axis=1)[:, :knn]
    src = np.repeat(np.arange(n), knn)
    dst = nn.ravel()
    edges = np.stack([np.concatenate([src, dst]), np.concatenate([dst, src])]).astype(np.int64)
    return X, edges, domain


def test_spatial_smooth_is_deterministic():
    """Identical seed + inputs => bit-for-bit identical labels and embedding."""
    X, edges, _ = _planted_gradient_instance(seed=3)
    a = spatial_smooth_domains(X, edges, n_domains=4, seed=0)
    b = spatial_smooth_domains(X, edges, n_domains=4, seed=0)
    np.testing.assert_array_equal(a.domain_id, b.domain_id)
    np.testing.assert_array_equal(a.embedding, b.embedding)


def test_spatial_smooth_output_shapes_and_labels():
    """Baseline returns one label per spot and an (n_spots, n_components) embedding."""
    X, edges, domain = _planted_gradient_instance()
    out = spatial_smooth_domains(X, edges, n_domains=4, n_components=16, seed=0)
    assert out.domain_id.shape == (X.shape[0],)
    assert out.domain_id.dtype == np.int64
    assert out.embedding.shape == (X.shape[0], 16)
    assert out.embedding.dtype == np.float32
    # Resolves the full domain budget on this well-separated planted instance.
    assert np.unique(out.domain_id).size == 4


def test_spatial_smooth_beats_plain_pca_on_planted_domains():
    """Graph smoothing (n_hops>0) beats the plain-PCA ablation (n_hops=0) on ARI.

    This is the core value of the baseline: averaging expression over the spatial
    kNN graph denoises the planted stripes that plain PCA + k-means resolves poorly.
    """
    X, edges, domain = _planted_gradient_instance(seed=1)
    plain = spatial_smooth_domains(X, edges, n_domains=4, n_hops=0, seed=0)
    smooth = spatial_smooth_domains(X, edges, n_domains=4, seed=0)  # defaults: alpha=0.5, n_hops=2
    ari_plain = adjusted_rand_index(domain, plain.domain_id)
    ari_smooth = adjusted_rand_index(domain, smooth.domain_id)
    assert ari_smooth > ari_plain + 0.05


def test_spatial_smooth_requires_the_graph():
    """With no edges the baseline cannot smooth and collapses to the plain-PCA result."""
    X, edges, _ = _planted_gradient_instance(seed=1)
    plain = spatial_smooth_domains(X, edges, n_domains=4, n_hops=0, seed=0)
    no_edges = spatial_smooth_domains(
        X, np.empty((2, 0), dtype=np.int64), n_domains=4, seed=0
    )
    np.testing.assert_array_equal(no_edges.domain_id, plain.domain_id)


def test_spatial_smooth_does_not_beat_trained_gnmf():
    """The baseline is a sensible MIDDLE: trained GNMF beats it on the same data.

    GNMF factorizes the nuisance gradient into its own nonnegative factor; the
    smoothing baseline cannot remove a spatially-smooth nuisance, so it sits below
    the trained model while still beating plain PCA (the test above).
    """
    X, edges, domain = _planted_gradient_instance(seed=1)
    smooth = spatial_smooth_domains(X, edges, n_domains=4, seed=0)
    gnmf_out, _ = fit_transform_gnmf(
        X, edges, K_shared=4, K_private=1, n_domains=4, lam=5.0, n_iter=400, seed=0
    )
    ari_smooth = adjusted_rand_index(domain, smooth.domain_id)
    ari_gnmf = adjusted_rand_index(domain, gnmf_out.domain_id.astype(np.int64))
    assert ari_gnmf > ari_smooth + 0.05


def test_baseline_scores_through_the_full_domain_metric_suite():
    """The baseline outputs plug into the SAME eval suite the models report.

    Proves apples-to-apples scoring: ARI / NMI / weighted Dice / boundary-F1 over
    the predicted ``domain_id`` and silhouette / Calinski-Harabasz over the PCA
    ``embedding`` all evaluate to finite numbers in their expected ranges.
    """
    X, edges, domain = _planted_gradient_instance(seed=1)
    out = spatial_smooth_domains(X, edges, n_domains=4, seed=0)
    pred = out.domain_id

    ari = adjusted_rand_index(domain, pred)
    nmi = normalized_mutual_information(domain, pred)
    dice = weighted_dice(domain, pred)
    bf1 = boundary_f1(domain, pred, edges)
    sil = silhouette(out.embedding, pred)
    ch = calinski_harabasz(out.embedding, pred)

    for value in (ari, nmi, dice, bf1, sil, ch):
        assert np.isfinite(value)
    assert 0.0 <= nmi <= 1.0
    assert 0.0 <= dice <= 1.0
    assert 0.0 <= bf1 <= 1.0
    assert -1.0 <= sil <= 1.0
    assert ch > 0.0
