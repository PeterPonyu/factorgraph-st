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
