"""Tests for #187: spatially-variable / factor-marker gene detection (eval/svg.py).

The fitted decoder produces a gene-loading matrix ``W`` of shape
``(n_genes, K)`` (one column per factor). ``rank_factor_markers`` ranks the
genes per factor by their loading and a fold-change versus the across-factor
background, returning a tidy per-factor structure. Optional permutation-based
significance is exercised on a small synthetic ``W``.
"""

from __future__ import annotations

import numpy as np

from factorgraph_st.eval.svg import FactorMarkers, rank_factor_markers


def _synthetic_W() -> tuple[np.ndarray, list[str]]:
    """4 genes, 2 factors. Each factor is dominated by a distinct gene."""
    # rows = genes, cols = factors. gene0 loads on factor0, gene1 on factor1.
    W = np.array(
        [
            [5.0, 0.1],  # gene0 -> factor0
            [0.1, 4.0],  # gene1 -> factor1
            [0.2, 0.2],  # gene2 background
            [0.05, 0.05],  # gene3 background
        ],
        dtype=np.float32,
    )
    genes = ["gene0", "gene1", "gene2", "gene3"]
    return W, genes


def test_returns_one_entry_per_factor():
    W, genes = _synthetic_W()
    result = rank_factor_markers(W, genes)
    assert isinstance(result, FactorMarkers)
    assert result.n_factors == 2
    assert set(result.per_factor.keys()) == {0, 1}


def test_top_gene_matches_dominant_loading():
    """The top-ranked gene per factor is the one with the highest loading."""
    W, genes = _synthetic_W()
    result = rank_factor_markers(W, genes)
    assert result.per_factor[0][0].gene == "gene0"
    assert result.per_factor[1][0].gene == "gene1"


def test_ranking_sorted_descending_by_score():
    """Within a factor, genes are ordered by descending score."""
    W, genes = _synthetic_W()
    result = rank_factor_markers(W, genes)
    for entries in result.per_factor.values():
        scores = [e.score for e in entries]
        assert scores == sorted(scores, reverse=True)


def test_fold_change_above_one_for_marker():
    """A factor-specific gene has fold-change > 1 vs the background mean."""
    W, genes = _synthetic_W()
    result = rank_factor_markers(W, genes)
    top0 = result.per_factor[0][0]
    assert top0.gene == "gene0"
    assert top0.fold_change > 1.0


def test_top_n_limits_output():
    W, genes = _synthetic_W()
    result = rank_factor_markers(W, genes, top_n=2)
    for entries in result.per_factor.values():
        assert len(entries) == 2


def test_mismatched_gene_names_raises():
    W, _ = _synthetic_W()
    try:
        rank_factor_markers(W, ["only", "three", "names"])
    except ValueError:
        return
    raise AssertionError("expected ValueError on gene-name/row mismatch")


def test_permutation_significance_marks_strong_marker():
    """With permutation significance, a strong marker gets a small p-value."""
    rng = np.random.default_rng(0)
    n_genes = 30
    W = np.abs(rng.normal(0.0, 0.1, size=(n_genes, 3))).astype(np.float32)
    # Plant a very strong marker for factor 0 at gene 0.
    W[0, 0] = 20.0
    genes = [f"g{i}" for i in range(n_genes)]
    result = rank_factor_markers(W, genes, n_permutations=200, seed=0)
    top0 = result.per_factor[0][0]
    assert top0.gene == "g0"
    assert top0.pvalue is not None
    assert top0.pvalue < 0.05
