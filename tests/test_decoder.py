"""Regression test for #75: _cluster_domains must actually cluster on coords.

The previous ``_cluster_domains`` ignored spatial coordinates entirely and
partitioned by rank of ``H[:, 0]`` (a "rank stripe"), producing strips
rather than spatial domains. The fix: deterministic k-means on the
z-normalized joint ``[H | coords]`` feature.
"""

from __future__ import annotations

import numpy as np

from factorgraph_st.model.decoder import _cluster_domains


def _purity(pred: np.ndarray, truth: np.ndarray) -> float:
    """Fraction of points whose truth label matches the majority truth label
    in their predicted cluster. 1.0 = perfect; ~1/k = chance."""
    pred = np.asarray(pred)
    truth = np.asarray(truth)
    total = 0
    for c in np.unique(pred):
        in_c = truth[pred == c]
        if in_c.size:
            total += int(np.bincount(in_c).max())
    return total / pred.size


def test_cluster_domains_uses_coords():
    """Two coord-separated clusters with identical H → purity > 0.75 (vs ~0.5 chance)."""
    rng = np.random.default_rng(0)
    n_per = 40
    coords_a = rng.normal(loc=(0.0, 0.0), scale=0.3, size=(n_per, 2))
    coords_b = rng.normal(loc=(10.0, 10.0), scale=0.3, size=(n_per, 2))
    coords = np.concatenate([coords_a, coords_b], axis=0).astype(np.float32)
    # H i.i.d. — not discriminative across the coord clusters.
    H = rng.normal(size=(2 * n_per, 4)).astype(np.float32)
    truth = np.concatenate([np.zeros(n_per, dtype=np.int64), np.ones(n_per, dtype=np.int64)])

    labels = _cluster_domains(H, coords, n_domains=2)
    assert labels.shape == (2 * n_per,) and labels.dtype == np.int64
    assert _purity(labels, truth) > 0.75, (
        "Expected coord-aware clustering to recover the two coord clusters; "
        "the rank-stripe implementation only consults H[:, 0] and fails this."
    )


def test_cluster_domains_deterministic_with_seed():
    rng = np.random.default_rng(0)
    coords = rng.normal(size=(60, 2)).astype(np.float32)
    H = rng.normal(size=(60, 4)).astype(np.float32)
    a = _cluster_domains(H, coords, n_domains=3, seed=0)
    b = _cluster_domains(H, coords, n_domains=3, seed=0)
    np.testing.assert_array_equal(a, b)


def test_cluster_domains_empty_input():
    H = np.zeros((0, 4), dtype=np.float32)
    coords = np.zeros((0, 2), dtype=np.float32)
    labels = _cluster_domains(H, coords, n_domains=3)
    assert labels.shape == (0,) and labels.dtype == np.int64


def _knn_within(coords_sub: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-region kNN edge indices (src, dst) over ``coords_sub``."""
    n = coords_sub.shape[0]
    d2 = ((coords_sub[:, None, :] - coords_sub[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    nn = np.argpartition(d2, k - 1, axis=1)[:, :k]
    return np.repeat(np.arange(n), k), nn.flatten()


def test_spatial_domain_uses_graph():
    """Regression for #101: two coords-separated regions whose coord clouds
    partially overlap (so k-means on [H | coords] alone is unreliable) must
    still be cleanly resolved by ``fit_transform`` when within-region edges
    are passed in. The graph (``edges``) is the signal that disambiguates.

    On main this assertion fails — without graph-aware refinement, purity
    sits near chance (~0.56). After threading ``edges`` through and applying
    one majority-vote smoothing pass over the graph, purity → ~1.0.
    """
    from factorgraph_st.model import fit_transform

    rng = np.random.default_rng(0)
    n_per = 40
    # Moderate coord separation: the clouds overlap enough that pure
    # k-means on [H | coords] confuses ~40% of points.
    coords_a = rng.normal(loc=(0.0, 0.0), scale=0.5, size=(n_per, 2)).astype(np.float32)
    coords_b = rng.normal(loc=(2.0, 0.0), scale=0.5, size=(n_per, 2)).astype(np.float32)
    coords = np.concatenate([coords_a, coords_b], axis=0)
    n = 2 * n_per

    X = rng.exponential(size=(n, 8)).astype(np.float32)
    section_id = np.zeros(n, dtype=np.int64)

    ra, ca = _knn_within(coords_a, k=5)
    rb, cb = _knn_within(coords_b, k=5)
    edges = np.stack(
        [
            np.concatenate([ra, rb + n_per]),
            np.concatenate([ca, cb + n_per]),
        ]
    ).astype(np.int64)
    truth = np.concatenate(
        [np.zeros(n_per, dtype=np.int64), np.ones(n_per, dtype=np.int64)]
    )

    out = fit_transform(
        X, coords, section_id, edges,
        d=8, K_shared=2, K_private=2, n_domains=2, seed=0,
    )
    assert _purity(out.domain_id, truth) > 0.9, (
        "Expected graph-aware clustering to resolve the two regions using "
        "within-region edges; without edges, k-means [H | coords] alone "
        "sits near chance on overlapping clouds."
    )


def test_section_id_used():
    """Regression for #50: decode_factors must consume section_id so train and
    inference share the same private-factor semantics. Two runs with the same
    ``H``/``X`` but permuted ``section_id`` must NOT produce identical outputs;
    on main, ``section_id`` is dropped on the floor and the outputs are equal.
    """
    from factorgraph_st.model.decoder import decode_factors

    rng = np.random.default_rng(0)
    n_per = 12
    n_sections = 3
    n = n_per * n_sections
    X = rng.exponential(size=(n, 8)).astype(np.float32)
    H = rng.normal(size=(n, 8)).astype(np.float32)
    sid_a = np.repeat(np.arange(n_sections, dtype=np.int64), n_per)
    # Different section assignment over the same K=3 sections.
    sid_b = np.tile(np.arange(n_sections, dtype=np.int64), n_per)

    out_a = decode_factors(X, H, K_shared=2, K_private=2, section_id=sid_a)
    out_b = decode_factors(X, H, K_shared=2, K_private=2, section_id=sid_b)

    assert not np.array_equal(out_a.Z_private, out_b.Z_private), (
        "decode_factors ignored section_id; Z_private identical across section "
        "assignments — private factors are not actually section-private."
    )
