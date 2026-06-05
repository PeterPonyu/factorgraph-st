"""Regression test for #73 / #77: label-invariant cluster-coherence metric.

``morans_i`` (Moran's I) is a numeric autocorrelation measure: it
centers ``values`` by the mean and computes ``sum_e (x_i - x̄)(x_j - x̄)``.
Applied to integer cluster labels (``inst.domain_id``), the score
depends on the specific integer codes assigned to each cluster. A label
permutation (``relabel``) that leaves the clustering structurally
identical changes the score. Every reported "domain coherence" headline
number that uses ``morans_i(domain_id, edges)`` is silently
encoding-dependent.

Fix: add a label-invariant cluster-coherence metric. We use the mean
Moran's I across one-hot indicator vectors per class: for each class
``c``, treat ``1[label == c]`` as a numeric score and compute Moran's I;
average across classes. This is invariant under any bijective relabel
because it sums over the *set* of classes, not over their integer codes.
"""

from __future__ import annotations

import importlib.util as _ilu
from pathlib import Path as _Path

import numpy as np
import pytest

from factorgraph_st.eval.metrics import (
    adjusted_rand_index,
    boundary_f1,
    boundary_precision,
    boundary_recall,
    calinski_harabasz,
    factor_covariate_association,
    label_invariant_cluster_coherence,
    matched_factor_correlation,
    normalized_mutual_information,
    shared_private_separation,
    silhouette,
    weighted_dice,
)


def _toy_graph_with_clusters():
    """3 spatially well-separated clusters on a 1D ring of edges."""
    n_per = 12
    labels = np.repeat(np.arange(3, dtype=np.int64), n_per)
    # Edges connecting only within each cluster (perfect coherence).
    src, dst = [], []
    for c in range(3):
        idx = np.where(labels == c)[0]
        for i in range(len(idx) - 1):
            src.append(idx[i])
            dst.append(idx[i + 1])
            src.append(idx[i + 1])
            dst.append(idx[i])
    edges = np.array([src, dst], dtype=np.int64)
    return labels, edges


def _relabel(labels: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    out = labels.copy()
    for k, v in mapping.items():
        out[labels == k] = v
    return out


def test_label_invariance():
    """``label_invariant_cluster_coherence`` must be invariant under any bijective relabel."""
    labels, edges = _toy_graph_with_clusters()
    base = label_invariant_cluster_coherence(labels, edges)
    for mapping in (
        {0: 1, 1: 0, 2: 2},
        {0: 2, 1: 0, 2: 1},
        {0: 7, 1: 3, 2: 11},  # arbitrary positive codes
    ):
        permuted = _relabel(labels, mapping)
        permuted_score = label_invariant_cluster_coherence(permuted, edges)
        assert np.isclose(base, permuted_score), (
            f"score must be relabel-invariant; got {base:.6f} vs {permuted_score:.6f} "
            f"under {mapping}"
        )


def test_label_invariant_cluster_coherence_within_range():
    """Sanity: returned coherence is a Moran's-I average, in [-1, 1] (modulo fp slop)."""
    labels, edges = _toy_graph_with_clusters()
    value = label_invariant_cluster_coherence(labels, edges)
    eps = 1e-9
    assert -1.0 - eps <= value <= 1.0 + eps


def test_label_invariant_cluster_coherence_empty_inputs():
    """Empty labels or empty edges → 0.0 (no crash)."""
    assert label_invariant_cluster_coherence(
        np.zeros(0, dtype=np.int64), np.zeros((2, 0), dtype=np.int64)
    ) == 0.0
    assert label_invariant_cluster_coherence(
        np.array([0, 1, 2], dtype=np.int64), np.zeros((2, 0), dtype=np.int64)
    ) == 0.0


def test_degenerate_inputs_not_perfect():
    """Regression for #79: empty/degenerate inputs must NOT silently score 1.0.

    A model that produces zero factors or a single recovered domain is *not*
    a perfect benchmark recovery; it is an uninformative degenerate run.
    See PR body for the cross-repo sibling reference.
    """
    import math

    # matched_factor_correlation: empty factor matrices -> NaN, not 1.0.
    empty = np.zeros((0, 0), dtype=np.float32)
    score_empty = matched_factor_correlation(empty, empty)
    assert math.isnan(score_empty), (
        f"empty factor matrix must return NaN, got {score_empty!r}"
    )

    # adjusted_rand_index: n < 2 -> NaN, not 1.0.
    score_single = adjusted_rand_index(np.array([0]), np.array([5]))
    assert math.isnan(score_single), (
        f"single-spot ARI must be NaN, got {score_single!r}"
    )

    # adjusted_rand_index: all-one-cluster (denom == 0) -> NaN, not 1.0.
    labels = np.array([0, 0, 0, 0])
    score_collapsed = adjusted_rand_index(labels, np.array([7, 7, 7, 7]))
    assert math.isnan(score_collapsed), (
        f"all-one-cluster ARI must be NaN, got {score_collapsed!r}"
    )


def test_label_invariance_on_real_synth_instance():
    """Robust invariance check on a non-trivial graph from the synthetic generator."""
    from factorgraph_st.synth.generator import generate_instance

    inst = generate_instance(
        n_sections=2, n_spots_per_section=15, n_genes=8, n_domains=3, seed=7
    )
    base = label_invariant_cluster_coherence(inst.domain_id, inst.edges)
    # Apply a non-uniform relabel (codes with different magnitudes).
    unique = np.unique(inst.domain_id)
    mapping = {int(u): int(v) for u, v in zip(unique, [101, 5, 33], strict=False)}
    relabeled = inst.domain_id.copy()
    for k, v in mapping.items():
        relabeled[inst.domain_id == k] = v
    permuted = label_invariant_cluster_coherence(relabeled, inst.edges)
    assert np.isclose(base, permuted), (
        f"label-invariance must hold on real synth graph; "
        f"got base={base} vs permuted={permuted}"
    )


def test_separation_threshold_scales():
    """#78: a factor active uniformly in EVERY section must report ~S active
    sections regardless of the section count S. The old fixed 0.05 threshold
    collapsed once per-section mass (~1/S) fell below 0.05 (S >= 20)."""
    K = 3
    for S in (4, 16, 20, 25):
        n_per = 5
        section_id = np.repeat(np.arange(S, dtype=np.int64), n_per)
        # Uniform activation -> per-section mass fraction == 1/S for every factor.
        Z_shared = np.ones((S * n_per, K), dtype=np.float32)
        Z_private = np.zeros((S * n_per, 1), dtype=np.float32)
        r = shared_private_separation(Z_shared, Z_private, section_id)
        assert r["shared_mean_active_sections"] == float(S), (
            f"S={S}: expected {S} active sections, got "
            f"{r['shared_mean_active_sections']}"
        )


def test_separation_detects_private_factor():
    """A factor concentrated in a single section counts as active in exactly 1."""
    S, n_per = 25, 5
    section_id = np.repeat(np.arange(S, dtype=np.int64), n_per)
    Z_private = np.zeros((S * n_per, 1), dtype=np.float32)
    Z_private[section_id == 7, 0] = 1.0  # all mass in one section
    Z_shared = np.zeros((S * n_per, 1), dtype=np.float32)
    r = shared_private_separation(Z_shared, Z_private, section_id)
    assert r["private_mean_active_sections"] == 1.0


def test_recovery_overcomplete_factors():
    """#46: when estimated factors > truth factors, a perfect match outside the
    first ``n_truth`` estimated columns must still score ~1.0 on BOTH paths."""
    rng = np.random.default_rng(0)

    # Brute-force path (<= 8 columns): 1 truth factor, 4 estimated, col 3 == truth.
    truth = rng.normal(size=(50, 1)).astype(np.float32)
    est = rng.normal(size=(50, 4)).astype(np.float32)
    est[:, 3] = truth[:, 0]
    assert matched_factor_correlation(est, truth) > 0.99

    # Greedy path (> 8 columns): same relationship, perfect copy in the last col.
    est2 = rng.normal(size=(50, 9)).astype(np.float32)
    est2[:, 8] = truth[:, 0]
    assert matched_factor_correlation(est2, truth) > 0.99


def test_recovery_square_unaffected():
    """Square case keeps the existing one-to-one matching semantics."""
    rng = np.random.default_rng(1)
    truth = rng.normal(size=(40, 3)).astype(np.float32)
    est = truth[:, ::-1].copy()  # same factors, permuted columns
    assert matched_factor_correlation(est, truth) > 0.99


# --------------------------------------------------------------------------- #
# Domain-quality metric suite (NMI / silhouette / Calinski-Harabasz /
# boundary precision-recall-F1 / weighted Dice). Deterministic synthetic
# fixtures: a perfect prediction must score the ideal value, a degenerate
# single-label prediction must score sensibly (not "perfect"), and a scrambled
# prediction must score strictly lower than the perfect one.
# --------------------------------------------------------------------------- #


def _line_graph(n: int) -> np.ndarray:
    """Undirected chain graph 0-1-2-...-(n-1) as a (2, 2*(n-1)) COO edge array."""
    src = np.arange(n - 1, dtype=np.int64)
    dst = np.arange(1, n, dtype=np.int64)
    return np.array([np.concatenate([src, dst]), np.concatenate([dst, src])], dtype=np.int64)


def _two_block_labels() -> np.ndarray:
    """10-spot line split into two contiguous domains (one interior boundary)."""
    return np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int64)


def _separated_embedding() -> tuple[np.ndarray, np.ndarray]:
    """Three well-separated Gaussian blobs in 2D with their domain labels."""
    rng = np.random.default_rng(0)
    centers = np.array([[0.0, 0.0], [50.0, 0.0], [0.0, 50.0]])
    labels = np.repeat(np.arange(3, dtype=np.int64), 15)
    X = np.repeat(centers, 15, axis=0) + rng.normal(scale=0.5, size=(45, 2))
    return X.astype(np.float64), labels


def test_nmi_perfect_and_relabel_invariant():
    """Identical-up-to-relabel partitions score NMI == 1.0."""
    true = np.array([0, 0, 1, 1, 2, 2])
    perfect = np.array([7, 7, 3, 3, 9, 9])  # bijective relabel of `true`
    assert normalized_mutual_information(true, perfect) == pytest.approx(1.0)
    assert normalized_mutual_information(true, true.copy()) == pytest.approx(1.0)


def test_nmi_degenerate_and_scrambled():
    """Single-label cases are sensible and a scrambled pred scores lower."""
    # Both partitions a single cluster -> trivially identical -> 1.0.
    assert normalized_mutual_information(np.zeros(6, int), np.zeros(6, int)) == pytest.approx(1.0)
    # One side collapsed, the other informative -> 0.0 (independent), not 1.0.
    assert normalized_mutual_information(np.array([0, 0, 1, 1]), np.zeros(4, int)) == pytest.approx(0.0)
    # n < 2 -> not evaluable.
    assert np.isnan(normalized_mutual_information(np.array([0]), np.array([0])))
    # A scrambled prediction scores strictly below the perfect partition.
    true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    scrambled = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    assert normalized_mutual_information(true, scrambled) < normalized_mutual_information(true, true.copy())


def test_silhouette_separated_vs_degenerate():
    """Well-separated blobs score high; <2 labels is not evaluable (NaN)."""
    X, labels = _separated_embedding()
    value = silhouette(X, labels)
    assert 0.9 < value <= 1.0
    # A single cluster (k < 2) is undefined.
    assert np.isnan(silhouette(X, np.zeros(X.shape[0], dtype=np.int64)))
    # A scrambled label assignment is far less coherent than the true blobs.
    rng = np.random.default_rng(1)
    scrambled = rng.permutation(labels)
    assert silhouette(X, scrambled) < value


def test_calinski_harabasz_separated_vs_overlapping():
    """CH is large for separated blobs, smaller for scrambled, NaN for k<2."""
    X, labels = _separated_embedding()
    separated = calinski_harabasz(X, labels)
    assert separated > 100.0
    rng = np.random.default_rng(2)
    scrambled = rng.permutation(labels)
    assert calinski_harabasz(X, scrambled) < separated
    assert np.isnan(calinski_harabasz(X, np.zeros(X.shape[0], dtype=np.int64)))


def test_boundary_metrics_perfect_and_scrambled():
    """Perfect prediction -> P=R=F1=1; an over-segmented pred has lower precision."""
    true = _two_block_labels()
    edges = _line_graph(true.size)
    perfect = np.where(true == 0, 4, 9)  # bijective relabel -> identical boundaries
    assert boundary_precision(true, perfect, edges) == pytest.approx(1.0)
    assert boundary_recall(true, perfect, edges) == pytest.approx(1.0)
    assert boundary_f1(true, perfect, edges) == pytest.approx(1.0)

    # Alternating labels mark almost every spot as a boundary: recall stays
    # perfect (the true boundary is included) but precision collapses, so F1
    # is strictly below the perfect prediction's F1.
    scrambled = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
    assert boundary_precision(true, scrambled, edges) < 1.0
    assert boundary_f1(true, scrambled, edges) < 1.0


def test_boundary_metrics_no_boundary_is_nan():
    """A single-domain GT has no boundary spots -> recall undefined (NaN)."""
    true = np.zeros(6, dtype=np.int64)
    edges = _line_graph(6)
    # No predicted boundary either -> precision undefined.
    assert np.isnan(boundary_precision(true, np.zeros(6, dtype=np.int64), edges))
    assert np.isnan(boundary_recall(true, np.zeros(6, dtype=np.int64), edges))


def test_weighted_dice_perfect_degenerate_scrambled():
    """Weighted Dice: perfect=1.0, relabel-invariant, scrambled<1, degenerate sensible."""
    true = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
    perfect = np.array([5, 5, 5, 8, 8, 1, 1, 1, 1])  # bijective relabel
    assert weighted_dice(true, perfect) == pytest.approx(1.0)
    assert weighted_dice(true, true.copy()) == pytest.approx(1.0)

    # A single-label prediction cannot exceed the best single-domain overlap.
    single = weighted_dice(true, np.zeros(true.size, dtype=np.int64))
    assert 0.0 < single < 1.0

    # A scrambled prediction overlaps every GT domain only partially.
    scrambled = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    assert weighted_dice(true, scrambled) < 1.0


def test_metric_suite_length_validation():
    """Mismatched label-array lengths raise (no silent broadcasting)."""
    with pytest.raises(ValueError):
        normalized_mutual_information(np.array([0, 1]), np.array([0, 1, 2]))
    with pytest.raises(ValueError):
        weighted_dice(np.array([0, 1]), np.array([0, 1, 2]))
    with pytest.raises(ValueError):
        boundary_f1(np.array([0, 1]), np.array([0, 1, 2]), _line_graph(2))


# --- #340: relabel-invariant domain headline + coordinate-only negative control --

_RUNNER = _Path(__file__).resolve().parents[1] / "scripts" / "run_real_factorgraph.py"


def _load_runner():
    spec = _ilu.spec_from_file_location("_run_real_factorgraph_340", _RUNNER)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_coherence_null_below_structured_coherence():
    """#340: the spatial-shuffle null is well below the true label-invariant
    coherence on a within-cluster-edge graph (so the reported delta is positive)."""
    runner = _load_runner()
    labels, edges = _toy_graph_with_clusters()
    true_coh = label_invariant_cluster_coherence(labels, edges)
    null_coh = runner._coherence_null(labels, edges, n_shuffles=50, seed=0)
    assert true_coh > null_coh + 0.2, (true_coh, null_coh)
    # Determinism: same seed -> same null.
    assert runner._coherence_null(labels, edges, 50, 0) == null_coh


def test_coherence_null_is_relabel_invariant_like_the_metric():
    """The null uses the label-invariant metric, so permuting label CODES (not
    spatial positions) must not change it."""
    runner = _load_runner()
    labels, edges = _toy_graph_with_clusters()
    base = runner._coherence_null(labels, edges, 30, 7)
    relabeled = _relabel(labels, {0: 5, 1: 9, 2: 2})
    assert runner._coherence_null(relabeled, edges, 30, 7) == pytest.approx(base)


def test_coords_negative_control_is_spatially_coherent():
    """#340: coordinate-only k-means domains are spatially coherent BY
    CONSTRUCTION -- this bounds how much domain coherence is attributable to
    position alone. Three well-separated coordinate blobs -> high coherence,
    well above its spatial-shuffle null."""
    runner = _load_runner()
    rng = np.random.default_rng(0)
    centers = np.array([[0.0, 0.0], [50.0, 0.0], [0.0, 50.0]])
    coords = np.vstack([c + rng.normal(0, 1.0, size=(40, 2)) for c in centers])
    domain_id, embedding = runner._coords_domains(coords, n_domains=3, seed=0)
    assert np.unique(domain_id).size == 3
    assert embedding.shape == coords.shape
    edges = runner._build_knn_edges(coords.astype(np.float32), k=6)
    coh = label_invariant_cluster_coherence(domain_id, edges)
    null = runner._coherence_null(domain_id, edges, 30, 0)
    assert coh > 0.5 and coh > null + 0.3, (coh, null)


def test_coords_domains_degenerate_guard():
    """Single requested domain (k<=1) returns an all-zero labelling, not a crash."""
    runner = _load_runner()
    coords = np.random.default_rng(0).normal(size=(20, 2))
    domain_id, embedding = runner._coords_domains(coords, n_domains=1, seed=0)
    assert domain_id.shape == (20,) and np.unique(domain_id).size == 1
    assert embedding.shape == (20, 2)


# --- #366: factor-vs-covariate association (correlation ratio eta^2 / NMI) -----


def test_factor_covariate_eta_sq_perfectly_determined():
    """A factor constant within each covariate group (differing across groups)
    is perfectly explained by it -> eta^2 == 1.0."""
    covariate = np.repeat(np.array([0, 1, 2]), 20)
    # Factor value is a pure function of the group code -> within-group variance 0.
    factor = np.where(covariate == 0, -3.0, np.where(covariate == 1, 0.5, 4.0))
    res = factor_covariate_association(factor, covariate)
    assert res["eta_sq"].shape == (1,)
    assert res["eta_sq"][0] == pytest.approx(1.0)
    assert res["max_eta_sq"] == pytest.approx(1.0)
    assert res["mean_eta_sq"] == pytest.approx(1.0)


def test_factor_covariate_eta_sq_independent_near_zero():
    """A random factor independent of the covariate has eta^2 ~ 0 (only the small
    finite-sample bias ~ (G-1)/(n-1))."""
    rng = np.random.default_rng(0)
    n = 900
    covariate = rng.integers(0, 3, size=n)
    factor = rng.normal(size=(n, 4))  # 4 independent factors
    res = factor_covariate_association(factor, covariate)
    assert res["eta_sq"].shape == (4,)
    assert np.all(res["eta_sq"] >= 0.0)
    assert res["max_eta_sq"] < 0.05, res["eta_sq"]


def test_factor_covariate_mixed_columns_and_summaries():
    """Per-factor scores: a determined column scores ~1, an independent column ~0,
    and the summaries aggregate over the finite per-factor values."""
    rng = np.random.default_rng(1)
    covariate = np.repeat(np.array([0, 1, 2, 3]), 60)
    determined = covariate.astype(np.float64) * 2.0  # eta^2 == 1
    independent = rng.normal(size=covariate.size)  # eta^2 ~ 0
    factors = np.column_stack([determined, independent])
    res = factor_covariate_association(factors, covariate)
    assert res["eta_sq"][0] == pytest.approx(1.0)
    assert res["eta_sq"][1] < 0.05
    assert res["max_eta_sq"] == pytest.approx(1.0)
    assert res["mean_eta_sq"] == pytest.approx(np.mean(res["eta_sq"]))


def test_factor_covariate_determinism():
    """Pure-numpy metric: identical inputs -> bit-identical per-factor arrays and
    summaries across repeated calls."""
    rng = np.random.default_rng(7)
    covariate = rng.integers(0, 5, size=300)
    factors = rng.normal(size=(300, 6))
    a = factor_covariate_association(factors, covariate, n_bins=5)
    b = factor_covariate_association(factors, covariate, n_bins=5)
    assert np.array_equal(a["eta_sq"], b["eta_sq"])
    assert np.array_equal(a["nmi"], b["nmi"])
    assert a["max_eta_sq"] == b["max_eta_sq"]
    assert a["mean_nmi"] == b["mean_nmi"]


def test_factor_covariate_constant_factor_is_nan():
    """A zero-variance (constant) factor is not evaluable -> eta^2 is NaN and is
    excluded from the summaries."""
    covariate = np.repeat(np.array([0, 1, 2]), 10)
    factors = np.column_stack([
        np.full(covariate.size, 3.14),  # constant -> NaN
        covariate.astype(np.float64),  # determined -> 1.0
    ])
    res = factor_covariate_association(factors, covariate)
    assert np.isnan(res["eta_sq"][0])
    assert res["eta_sq"][1] == pytest.approx(1.0)
    # Summaries ignore the NaN column.
    assert res["max_eta_sq"] == pytest.approx(1.0)
    assert res["mean_eta_sq"] == pytest.approx(1.0)


def test_factor_covariate_single_group_is_zero():
    """A covariate with a single group explains no variance -> eta^2 == 0.0."""
    covariate = np.zeros(30, dtype=int)
    factor = np.random.default_rng(0).normal(size=30)
    res = factor_covariate_association(factor, covariate)
    assert res["eta_sq"][0] == 0.0


def test_factor_covariate_nmi_path_detects_nonmonotone_dependence():
    """The discretized-NMI companion flags a factor fully determined by the
    covariate (here a non-monotone group->value map) and stays ~0 for noise."""
    covariate = np.repeat(np.array([0, 1, 2]), 100)
    # Non-monotone mapping (0->5, 1->-5, 2->0): still perfectly determined.
    determined = np.where(covariate == 0, 5.0, np.where(covariate == 1, -5.0, 0.0))
    rng = np.random.default_rng(3)
    independent = rng.normal(size=covariate.size)
    factors = np.column_stack([determined, independent])
    res = factor_covariate_association(factors, covariate, n_bins=8)
    assert "nmi" in res and res["nmi"].shape == (2,)
    assert res["nmi"][0] == pytest.approx(1.0, abs=1e-9)
    assert res["nmi"][1] < 0.2
    assert res["max_nmi"] == pytest.approx(1.0, abs=1e-9)


def test_factor_covariate_length_mismatch_raises():
    """Mismatched factor/covariate lengths fail loudly."""
    with pytest.raises(ValueError):
        factor_covariate_association(np.zeros((10, 2)), np.zeros(9, dtype=int))


def test_factor_covariate_eta_sq_within_unit_range():
    """eta^2 is a variance ratio bounded to [0, 1] for arbitrary inputs."""
    rng = np.random.default_rng(11)
    covariate = rng.integers(0, 4, size=200)
    factors = rng.normal(size=(200, 5)) * rng.normal(size=5)
    res = factor_covariate_association(factors, covariate)
    finite = res["eta_sq"][np.isfinite(res["eta_sq"])]
    assert np.all(finite >= 0.0) and np.all(finite <= 1.0)
