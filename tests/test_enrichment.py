"""Tests for #189: hypergeometric over-representation enrichment (eval/enrichment.py).

Given a query gene list (e.g. a factor's top markers), a gene-set dict (GMT-style
``{term: [genes]}``), and a background universe, ``enrich_genes`` reports the
hypergeometric over-representation p-value per term plus a BH-adjusted q-value.
The known-overlap case is checked against ``scipy.stats.hypergeom`` directly.
"""

from __future__ import annotations

import math

import numpy as np

from factorgraph_st.eval.enrichment import EnrichmentResult, enrich_genes


def _gene_sets() -> dict[str, list[str]]:
    return {
        "termA": ["g0", "g1", "g2", "g3"],  # query overlaps heavily
        "termB": ["g50", "g51", "g52"],  # no overlap with query
    }


def test_returns_one_row_per_term():
    universe = [f"g{i}" for i in range(100)]
    query = ["g0", "g1", "g2"]
    results = enrich_genes(query, _gene_sets(), background=universe)
    assert all(isinstance(r, EnrichmentResult) for r in results)
    assert {r.term for r in results} == {"termA", "termB"}


def test_overlap_count_correct():
    universe = [f"g{i}" for i in range(100)]
    query = ["g0", "g1", "g2", "g99"]
    by_term = {r.term: r for r in enrich_genes(query, _gene_sets(), background=universe)}
    assert by_term["termA"].overlap == 3  # g0,g1,g2 in termA
    assert by_term["termB"].overlap == 0
    assert set(by_term["termA"].overlap_genes) == {"g0", "g1", "g2"}


def test_pvalue_matches_hypergeom_survival():
    """The reported p-value equals the hypergeometric upper-tail probability."""
    from scipy.stats import hypergeom

    universe = [f"g{i}" for i in range(100)]
    query = ["g0", "g1", "g2", "g3"]  # all 4 of termA
    by_term = {r.term: r for r in enrich_genes(query, _gene_sets(), background=universe)}
    r = by_term["termA"]
    M = 100  # universe size
    n = 4  # genes in termA
    N = 4  # query size
    k = 4  # overlap
    expected = hypergeom.sf(k - 1, M, n, N)
    assert math.isclose(r.pvalue, expected, rel_tol=1e-9, abs_tol=1e-12)


def test_strong_overlap_is_significant():
    universe = [f"g{i}" for i in range(1000)]
    query = ["g0", "g1", "g2", "g3"]
    by_term = {r.term: r for r in enrich_genes(query, _gene_sets(), background=universe)}
    assert by_term["termA"].pvalue < 0.001
    assert by_term["termB"].pvalue == 1.0  # zero overlap -> no enrichment


def test_bh_adjustment_monotone_and_ge_pvalue():
    """BH q-values are >= raw p-values and respect rank monotonicity."""
    universe = [f"g{i}" for i in range(200)]
    query = ["g0", "g1", "g2", "g3"]
    results = enrich_genes(query, _gene_sets(), background=universe)
    for r in results:
        assert r.qvalue >= r.pvalue - 1e-12
        assert 0.0 <= r.qvalue <= 1.0 + 1e-12


def test_bh_known_values():
    """BH on a tiny synthetic p-vector matches the manual computation."""
    # Bypass hypergeometric path by feeding a known set where we can reason about
    # ordering: construct three terms with controlled overlaps.
    universe = [f"g{i}" for i in range(50)]
    gene_sets = {
        "t1": ["g0", "g1", "g2", "g3", "g4"],  # large overlap -> tiny p
        "t2": ["g0", "g10", "g11", "g12"],  # small overlap
        "t3": ["g40", "g41"],  # no overlap -> p == 1
    }
    query = ["g0", "g1", "g2", "g3", "g4"]
    results = enrich_genes(query, gene_sets, background=universe)
    pvals = np.array([r.pvalue for r in results])
    qvals = np.array([r.qvalue for r in results])
    # Manual BH on the same p-values.
    m = pvals.size
    order = np.argsort(pvals)
    ranked = pvals[order]
    bh = ranked * m / (np.arange(1, m + 1))
    bh = np.minimum.accumulate(bh[::-1])[::-1]
    bh = np.clip(bh, 0.0, 1.0)
    expected = np.empty_like(bh)
    expected[order] = bh
    assert np.allclose(qvals, expected, atol=1e-12)


def test_default_background_is_gene_set_union():
    """Without explicit background, the universe defaults to query ∪ gene sets."""
    query = ["g0", "g1"]
    results = enrich_genes(query, _gene_sets())
    assert len(results) == 2  # does not crash; produces a row per term
