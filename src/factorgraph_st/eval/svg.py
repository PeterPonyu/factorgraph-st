"""Factor-marker / spatially-variable gene detection from the loading matrix.

Issue #187. The numpy decoder (:func:`factorgraph_st.model.decoder.decode_factors`)
produces a nonnegative gene-loading matrix ``W`` of shape ``(n_genes, K)`` — one
column per recovered factor, gauge-fixed so each loading column has unit L2 norm
(see ``_apply_canonical_gauge``). A factor's "marker" genes are the genes that
load highly on that factor *relative to the other factors*.

This module ranks, per factor, the genes by their loading and a fold-change of
that loading versus the across-factor background (the mean loading of the same
gene over the other factors). Optionally, a permutation test over gene labels
estimates how surprising each gene's loading is under the null that loadings are
exchangeable across genes within a factor.

Dependency-light: numpy only. We deliberately do not copy class names or source
from any third-party spatial-omics package; the loading matrix and fold-change
ranking are standard factor-model interpretation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["GeneScore", "FactorMarkers", "rank_factor_markers"]


@dataclass(frozen=True)
class GeneScore:
    """A single ranked gene for one factor."""

    gene: str
    loading: float
    fold_change: float
    score: float
    pvalue: float | None = None


@dataclass(frozen=True)
class FactorMarkers:
    """Per-factor ranked marker genes.

    ``per_factor`` maps factor index ``k`` -> list of :class:`GeneScore`,
    sorted by descending score (the dominant marker first).
    """

    per_factor: dict[int, list[GeneScore]]

    @property
    def n_factors(self) -> int:
        return len(self.per_factor)


def rank_factor_markers(
    W: np.ndarray,
    gene_names: list[str],
    *,
    top_n: int | None = None,
    n_permutations: int = 0,
    seed: int = 0,
    eps: float = 1e-12,
) -> FactorMarkers:
    """Rank factor-marker genes from the gene-loading matrix ``W``.

    Parameters
    ----------
    W:
        Gene-loading matrix of shape ``(n_genes, K)`` (rows = genes, columns =
        factors), as returned by ``decode_factors(...).W``.
    gene_names:
        Length-``n_genes`` gene identifiers aligned to the rows of ``W``.
    top_n:
        If given, keep only the top ``top_n`` genes per factor; otherwise keep
        all genes.
    n_permutations:
        If ``> 0``, run a per-factor permutation test on the gene labels and
        attach an empirical upper-tail p-value to each :class:`GeneScore`. The
        null shuffles which gene receives which loading within a factor, so a
        gene whose loading is extreme relative to the factor's loading
        distribution gets a small p-value. ``0`` (default) skips the test and
        leaves ``pvalue=None``.
    seed:
        Seed for the permutation RNG (only used when ``n_permutations > 0``).
    eps:
        Floor for the background mean in the fold-change denominator.

    Returns
    -------
    FactorMarkers
        Per-factor ranked genes. The ranking score is ``loading * fold_change``,
        which rewards both a large absolute loading and factor specificity.
    """
    W = np.asarray(W, dtype=np.float64)
    if W.ndim != 2:
        raise ValueError(f"W must be 2D (n_genes, K); got shape {W.shape}")
    n_genes, n_factors = W.shape
    if len(gene_names) != n_genes:
        raise ValueError(
            f"len(gene_names)={len(gene_names)} != W.shape[0]={n_genes}; "
            "gene names must align to the rows (genes) of W."
        )

    # Background for gene g, factor k: mean loading of gene g over the OTHER
    # factors. fold_change = loading / background (factor specificity).
    row_sum = W.sum(axis=1, keepdims=True)
    if n_factors > 1:
        background = (row_sum - W) / (n_factors - 1)
    else:
        # A single factor has no "other factors"; compare against the
        # gene-wise mean loading across all genes instead so fold-change is
        # still meaningful (a gene above the typical loading is a marker).
        background = np.full_like(W, float(W.mean()))
    fold_change = W / np.maximum(background, eps)
    score = W * fold_change

    pvalues = _permutation_pvalues(W, n_permutations, seed) if n_permutations > 0 else None

    per_factor: dict[int, list[GeneScore]] = {}
    for k in range(n_factors):
        order = np.argsort(-score[:, k], kind="stable")
        if top_n is not None:
            order = order[:top_n]
        entries = [
            GeneScore(
                gene=gene_names[int(g)],
                loading=float(W[g, k]),
                fold_change=float(fold_change[g, k]),
                score=float(score[g, k]),
                pvalue=None if pvalues is None else float(pvalues[g, k]),
            )
            for g in order
        ]
        per_factor[k] = entries
    return FactorMarkers(per_factor=per_factor)


def _permutation_pvalues(W: np.ndarray, n_permutations: int, seed: int) -> np.ndarray:
    """Empirical upper-tail p-values for each gene's specificity score per factor.

    Null hypothesis: the gene labels are exchangeable within a factor, i.e. a
    gene's specificity score on factor ``k`` is no larger than that of a
    randomly drawn gene. For each factor we permute the per-gene specificity
    scores across genes ``n_permutations`` times and ask, for each gene, how
    often a permuted gene's score reaches at least the observed score. The
    ``(count + 1) / (n + 1)`` add-one estimator keeps p-values strictly
    positive.

    A gene that loads strongly and specifically on one factor sits in the far
    upper tail of the within-factor score distribution, so it earns a small
    p-value. Because the null draws over all ``n_genes`` genes, the achievable
    resolution is ``~1/n_genes`` rather than ``~1/K`` — so clean markers in a
    realistically sized panel become significant. With a single factor there is
    no specificity to test and every p-value is ``1.0``.
    """
    rng = np.random.default_rng(seed)
    n_genes, n_factors = W.shape
    if n_factors < 2:
        return np.ones((n_genes, n_factors), dtype=np.float64)

    row_sum = W.sum(axis=1, keepdims=True)
    background = (row_sum - W) / (n_factors - 1)
    fold = W / np.maximum(background, 1e-12)
    observed = W * fold  # specificity score, shape (n_genes, n_factors)

    pvals = np.ones((n_genes, n_factors), dtype=np.float64)
    for k in range(n_factors):
        col = observed[:, k]
        ge = np.zeros(n_genes, dtype=np.float64)
        for _ in range(n_permutations):
            ge += rng.permutation(col) >= col
        pvals[:, k] = (ge + 1.0) / (n_permutations + 1.0)
    return pvals
