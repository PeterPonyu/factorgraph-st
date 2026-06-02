"""Hypergeometric over-representation (pathway) enrichment.

Issue #189. Given a query gene list (e.g. a factor's top markers from
:mod:`factorgraph_st.eval.svg`) and a GMT-style dict ``{term: [genes]}``, this
computes the hypergeometric over-representation p-value of each term against a
background universe, plus a Benjamini-Hochberg adjusted q-value.

Dependency-light: the hypergeometric upper-tail probability uses
``scipy.stats.hypergeom`` when scipy is installed, otherwise a numpy-only
exact log-gamma fallback (so the runtime stays numpy-only). An optional
gseapy/Enrichr backend is intentionally NOT imported here; callers who want a
hosted-Enrichr query should add that behind their own gated import.

Brand-independence: standard hypergeometric ORA + BH; no third-party
spatial-omics source or class names are reused.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

__all__ = ["EnrichmentResult", "enrich_genes", "benjamini_hochberg"]


@dataclass(frozen=True)
class EnrichmentResult:
    """Enrichment of one gene set against the query."""

    term: str
    overlap: int
    term_size: int
    query_size: int
    universe_size: int
    pvalue: float
    qvalue: float
    overlap_genes: tuple[str, ...]


def enrich_genes(
    query: Iterable[str],
    gene_sets: Mapping[str, Iterable[str]],
    *,
    background: Iterable[str] | None = None,
) -> list[EnrichmentResult]:
    """Hypergeometric over-representation enrichment of ``query`` against ``gene_sets``.

    Parameters
    ----------
    query:
        Gene identifiers of interest (e.g. a factor's top markers).
    gene_sets:
        GMT-style mapping ``{term: iterable_of_genes}``.
    background:
        The universe of testable genes. When ``None``, the universe defaults to
        the union of the query and every gene set (a self-contained, if
        optimistic, universe). For real analyses pass the set of all detected
        genes so the hypergeometric parameters are calibrated.

    Returns
    -------
    list[EnrichmentResult]
        One result per term, in the input order of ``gene_sets``. ``pvalue`` is
        the hypergeometric upper-tail probability ``P(X >= overlap)``; ``qvalue``
        is the BH-adjusted p-value across all tested terms. A term with zero
        overlap returns ``pvalue == 1.0``.
    """
    gene_sets = {term: {str(g) for g in genes} for term, genes in gene_sets.items()}

    if background is None:
        universe: set[str] = set()
        for genes in gene_sets.values():
            universe |= genes
        universe |= {str(g) for g in query}
    else:
        universe = {str(g) for g in background}

    # Restrict everything to the universe so the hypergeometric parameters are
    # internally consistent.
    query_set = {str(g) for g in query} & universe
    M = len(universe)  # universe size
    N = len(query_set)  # number of "successes" drawn (query size in universe)

    terms = list(gene_sets.keys())
    pvals = np.empty(len(terms), dtype=np.float64)
    rows: list[dict] = []
    for i, term in enumerate(terms):
        term_genes = gene_sets[term] & universe
        n = len(term_genes)  # successes in the universe (term size)
        overlap_genes = tuple(sorted(query_set & term_genes))
        k = len(overlap_genes)  # observed overlap
        p = _hypergeom_sf(k, M, n, N)
        pvals[i] = p
        rows.append(
            {
                "term": term,
                "overlap": k,
                "term_size": n,
                "query_size": N,
                "universe_size": M,
                "pvalue": p,
                "overlap_genes": overlap_genes,
            }
        )

    qvals = benjamini_hochberg(pvals)
    return [EnrichmentResult(qvalue=float(qvals[i]), **rows[i]) for i in range(len(terms))]


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (FDR q-values).

    Standard step-up procedure: sort ascending, scale each by ``m / rank``,
    enforce monotonicity from the largest rank down, and clip to ``[0, 1]``.
    Returns q-values aligned to the input order.
    """
    p = np.asarray(pvalues, dtype=np.float64)
    m = p.size
    if m == 0:
        return p.copy()
    order = np.argsort(p, kind="stable")
    ranked = p[order]
    scaled = ranked * m / np.arange(1, m + 1)
    # Enforce monotone non-decreasing q-values from the largest p downward.
    scaled = np.minimum.accumulate(scaled[::-1])[::-1]
    scaled = np.clip(scaled, 0.0, 1.0)
    q = np.empty(m, dtype=np.float64)
    q[order] = scaled
    return q


def _hypergeom_sf(k: int, M: int, n: int, N: int) -> float:
    """Upper-tail hypergeometric probability ``P(X >= k)``.

    Parameters mirror ``scipy.stats.hypergeom``: ``M`` universe size, ``n``
    successes in the universe (term size), ``N`` draws (query size), ``k``
    observed overlap. Uses ``scipy.stats.hypergeom.sf(k-1, M, n, N)`` when
    available, otherwise a numpy-only log-gamma summation.
    """
    if k <= 0:
        return 1.0
    if n == 0 or N == 0 or M == 0:
        return 1.0
    try:
        from scipy.stats import hypergeom

        return float(hypergeom.sf(k - 1, M, n, N))
    except ImportError:
        return _hypergeom_sf_numpy(k, M, n, N)


def _hypergeom_sf_numpy(k: int, M: int, n: int, N: int) -> float:
    """Numpy-only ``P(X >= k)`` via summed log-gamma PMF terms.

    PMF(i) = C(n, i) C(M-n, N-i) / C(M, N), summed for ``i`` from ``k`` to the
    support maximum ``min(n, N)``. Computed in log space with ``gammaln`` for
    numerical stability.
    """
    from math import lgamma

    def _log_comb(a: int, b: int) -> float:
        if b < 0 or b > a:
            return -np.inf
        return lgamma(a + 1) - lgamma(b + 1) - lgamma(a - b + 1)

    log_denom = _log_comb(M, N)
    i_max = min(n, N)
    total = 0.0
    for i in range(k, i_max + 1):
        log_p = _log_comb(n, i) + _log_comb(M - n, N - i) - log_denom
        total += float(np.exp(log_p))
    return float(min(max(total, 0.0), 1.0))
