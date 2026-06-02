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

import numpy as np
import pytest

from factorgraph_st.eval.metrics import (
    adjusted_rand_index,
    boundary_f1,
    boundary_precision,
    boundary_recall,
    calinski_harabasz,
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
