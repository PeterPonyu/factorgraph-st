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

from factorgraph_st.eval.metrics import label_invariant_cluster_coherence


def _toy_graph_with_clusters():
    """3 spatially well-separated clusters on a 1D ring of edges."""
    n_per = 12
    labels = np.repeat(np.arange(3, dtype=np.int64), n_per)
    # Edges connecting only within each cluster (perfect coherence).
    src, dst = [], []
    for c in range(3):
        idx = np.where(labels == c)[0]
        for i in range(len(idx) - 1):
            src.append(idx[i]); dst.append(idx[i + 1])
            src.append(idx[i + 1]); dst.append(idx[i])
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


def test_label_invariance_on_real_synth_instance():
    """Robust invariance check on a non-trivial graph from the synthetic generator."""
    from factorgraph_st.synth.generator import generate_instance

    inst = generate_instance(
        n_sections=2, n_spots_per_section=15, n_genes=8, n_domains=3, seed=7
    )
    base = label_invariant_cluster_coherence(inst.domain_id, inst.edges)
    # Apply a non-uniform relabel (codes with different magnitudes).
    unique = np.unique(inst.domain_id)
    mapping = {int(u): int(v) for u, v in zip(unique, [101, 5, 33])}
    relabeled = inst.domain_id.copy()
    for k, v in mapping.items():
        relabeled[inst.domain_id == k] = v
    permuted = label_invariant_cluster_coherence(relabeled, inst.edges)
    assert np.isclose(base, permuted), (
        f"label-invariance must hold on real synth graph; "
        f"got base={base} vs permuted={permuted}"
    )
