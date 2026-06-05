"""Tests for the eval gates / negative controls (#316, #315, #354).

All fixtures are synthetic and seeded:

* #316 permutation-null calibration — a planted within-cluster-edge graph gives a
  large positive z-score; spatially shuffled labels collapse to z ~ 0.
* #315 GT-free k-selection — the held-out-error knee recovers the planted factor
  count of a low-rank-plus-noise matrix.
* #354 coherence-claim gate — PASS on genuine spatial structure; FAIL on the
  shuffled negative control, on the coordinate-derived negative control, and when
  the leakage check cannot be performed (no coords supplied).

Everything is deterministic for a fixed seed.
"""

from __future__ import annotations

import numpy as np
import pytest

from factorgraph_st.eval.gates import (
    coherence_claim_gate,
    permutation_null_calibration,
    select_n_factors,
)


def _clustered_graph(n_per: int = 12, n_clusters: int = 3):
    """Labels + a graph whose edges live only *within* each cluster.

    With every edge internal to a cluster the one-hot indicators are perfectly
    autocorrelated, so :func:`label_invariant_cluster_coherence` is high while any
    spatial reshuffle of the labels destroys it — the canonical planted-structure
    fixture (mirrors ``tests/test_metrics._toy_graph_with_clusters``).
    """
    labels = np.repeat(np.arange(n_clusters, dtype=np.int64), n_per)
    src, dst = [], []
    for c in range(n_clusters):
        idx = np.where(labels == c)[0]
        for i in range(len(idx) - 1):
            src += [idx[i], idx[i + 1]]
            dst += [idx[i + 1], idx[i]]
    edges = np.array([src, dst], dtype=np.int64)
    return labels, edges


def _planted_lowrank(K: int = 3, n: int = 300, g: int = 50, sigma: float = 1.0, seed: int = 0):
    """Nonnegative ``X ~= Z @ W.T + noise`` with a known number of factors ``K``."""
    rng = np.random.default_rng(seed)
    Z = rng.exponential(1.0, size=(n, K))
    W = rng.exponential(1.0, size=(g, K))
    return np.clip(Z @ W.T + rng.normal(0.0, sigma, size=(n, g)), 0.0, None)


# --------------------------------------------------------------------------- #
# #316 permutation-null calibration
# --------------------------------------------------------------------------- #


def test_perm_null_high_z_on_planted_structure():
    """Genuine within-cluster coherence sits far above its spatial-shuffle null."""
    labels, edges = _clustered_graph()
    cal = permutation_null_calibration(labels, edges, n_shuffles=199, seed=0)
    assert cal.observed > cal.null_mean
    assert cal.z_score > 5.0, cal
    # One-sided empirical p is tiny but never exactly zero (the +1 correction).
    assert 0.0 < cal.p_empirical <= 0.01
    assert cal.n_shuffles == 199


def test_perm_null_near_zero_on_shuffled_labels():
    """Spatially shuffled labels carry no structure -> z ~ 0, p not significant."""
    labels, edges = _clustered_graph()
    shuffled = np.random.default_rng(123).permutation(labels)
    cal = permutation_null_calibration(shuffled, edges, n_shuffles=199, seed=0)
    assert abs(cal.z_score) < 3.0, cal
    assert cal.p_empirical > 0.01


def test_perm_null_morans_metric_path():
    """The Moran's-I metric path also separates structure from its null."""
    # A numeric score that is piecewise-constant within each cluster is strongly
    # autocorrelated on the within-cluster graph.
    labels, edges = _clustered_graph()
    score = labels.astype(np.float64)  # distinct value per within-edge-connected block
    cal = permutation_null_calibration(score, edges, metric="morans_i", n_shuffles=199, seed=0)
    assert cal.metric == "morans_i"
    assert cal.z_score > 5.0, cal


def test_perm_null_is_deterministic():
    """Same seed -> identical calibration."""
    labels, edges = _clustered_graph()
    a = permutation_null_calibration(labels, edges, n_shuffles=99, seed=7)
    b = permutation_null_calibration(labels, edges, n_shuffles=99, seed=7)
    assert (a.observed, a.null_mean, a.null_std, a.z_score, a.p_empirical) == (
        b.observed, b.null_mean, b.null_std, b.z_score, b.p_empirical,
    )


def test_perm_null_rejects_unknown_metric():
    labels, edges = _clustered_graph()
    with pytest.raises(ValueError):
        permutation_null_calibration(labels, edges, metric="nope")


# --------------------------------------------------------------------------- #
# #315 GT-free k-selection
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_k_selection_recovers_planted_count(seed):
    """The held-out-error knee recovers the planted factor count K=3."""
    X = _planted_lowrank(K=3, seed=seed)
    sel = select_n_factors(X, range(1, 9), seed=0)
    assert sel.best_k == 3, (sel.best_k, sel.errors)
    assert sel.candidate_ks == list(range(1, 9))
    assert len(sel.errors) == 8


def test_k_selection_curve_decreases_then_plateaus():
    """Held-out error drops steeply up to the planted K, then flattens."""
    X = _planted_lowrank(K=3, seed=0)
    sel = select_n_factors(X, range(1, 9), seed=0)
    # Relative improvement crosses the knee at K=3: the step into k=3 clears the
    # 5% floor, the step into k=4 does not.
    rel_into_k3 = (sel.errors[1] - sel.errors[2]) / sel.errors[1]
    rel_into_k4 = (sel.errors[2] - sel.errors[3]) / sel.errors[2]
    assert rel_into_k3 >= sel.min_rel_improvement
    assert rel_into_k4 < sel.min_rel_improvement


def test_k_selection_is_deterministic():
    X = _planted_lowrank(K=3, seed=0)
    a = select_n_factors(X, range(1, 7), seed=0)
    b = select_n_factors(X, range(1, 7), seed=0)
    assert a.errors == b.errors
    assert a.best_k == b.best_k


def test_k_selection_empty_candidates_raises():
    X = _planted_lowrank(K=2, seed=0)
    with pytest.raises(ValueError):
        select_n_factors(X, [])


# --------------------------------------------------------------------------- #
# #354 coherence-claim gate
# --------------------------------------------------------------------------- #


def test_gate_passes_on_genuine_structure():
    """High within-cluster coherence + coords that DON'T reproduce the labels."""
    labels, edges = _clustered_graph()
    # Random coords unrelated to the labels: a coords-only k-means cannot recover
    # them, so there is no coordinate leakage.
    coords = np.random.default_rng(0).uniform(size=(labels.size, 2))
    gate = coherence_claim_gate(labels, edges, coords=coords, n_shuffles=99, seed=0)
    assert gate.passed, gate.reason
    assert gate.z_score > 5.0
    assert gate.coord_leakage_ari < 0.5


def test_gate_fails_on_shuffled_negative_control():
    """Spatially shuffled labels do not beat the null -> FAIL."""
    labels, edges = _clustered_graph()
    shuffled = np.random.default_rng(5).permutation(labels)
    coords = np.random.default_rng(0).uniform(size=(labels.size, 2))
    gate = coherence_claim_gate(shuffled, edges, coords=coords, n_shuffles=99, seed=0)
    assert not gate.passed
    assert "null" in gate.reason or "z-score" in gate.reason


def test_gate_fails_on_coordinate_derived_control():
    """#340-style coords-only k-means domains: spatially coherent BY CONSTRUCTION
    but coordinate-leaked, so the gate must FAIL on the leakage criterion."""
    from factorgraph_st.eval.gates import _coord_leakage_ari
    from factorgraph_st.model.decoder import _kmeans

    rng = np.random.default_rng(0)
    centers = np.array([[0.0, 0.0], [50.0, 0.0], [0.0, 50.0]])
    coords = np.vstack([c + rng.normal(0, 1.0, size=(40, 2)) for c in centers])
    # Domains assigned PURELY from coordinates (the negative control).
    z = (coords - coords.mean(0)) / coords.std(0)
    domain_id = _kmeans(z, 3, 0, 4, 100).astype(np.int64)
    # Within-cluster edges so the coherence itself is high.
    src, dst = [], []
    for c in np.unique(domain_id):
        idx = np.where(domain_id == c)[0]
        for i in range(len(idx) - 1):
            src += [idx[i], idx[i + 1]]
            dst += [idx[i + 1], idx[i]]
    edges = np.array([src, dst], dtype=np.int64)

    # The coherence genuinely beats its null (coherent by construction)...
    cal = permutation_null_calibration(domain_id, edges, n_shuffles=99, seed=0)
    assert cal.z_score > 5.0
    # ...but a coords-only k-means reproduces the labels (leakage), so FAIL.
    assert _coord_leakage_ari(domain_id, coords, seed=0) >= 0.5
    gate = coherence_claim_gate(domain_id, edges, coords=coords, n_shuffles=99, seed=0)
    assert not gate.passed
    assert "leak" in gate.reason.lower()


def test_gate_refuses_to_pass_without_leakage_evidence():
    """No coords + no declaration -> the gate must NOT silently pass."""
    labels, edges = _clustered_graph()
    gate = coherence_claim_gate(labels, edges, n_shuffles=99, seed=0)
    assert not gate.passed
    assert "unverifiable" in gate.reason


def test_gate_coords_derived_flag_fails():
    """Declaring the labels coordinate-derived fails the gate outright."""
    labels, edges = _clustered_graph()
    gate = coherence_claim_gate(labels, edges, coords_derived=True, n_shuffles=99, seed=0)
    assert not gate.passed
    assert "coordinate-derived" in gate.reason


def test_gate_is_deterministic():
    labels, edges = _clustered_graph()
    coords = np.random.default_rng(0).uniform(size=(labels.size, 2))
    a = coherence_claim_gate(labels, edges, coords=coords, n_shuffles=99, seed=3)
    b = coherence_claim_gate(labels, edges, coords=coords, n_shuffles=99, seed=3)
    assert (a.passed, a.reason, a.z_score, a.coord_leakage_ari) == (
        b.passed, b.reason, b.z_score, b.coord_leakage_ari,
    )
