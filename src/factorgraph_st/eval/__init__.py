"""Evaluation metrics for synthetic FactorGraph-ST benchmarks."""

from factorgraph_st.eval.enrichment import (
    EnrichmentResult,
    benjamini_hochberg,
    enrich_genes,
)
from factorgraph_st.eval.metrics import (
    adjusted_rand_index,
    factor_redundancy,
    held_out_reconstruction_error,
    label_invariant_cluster_coherence,
    matched_factor_correlation,
    morans_i,
    reconstruction_error,
    section_overlap,
    shared_private_separation,
)
from factorgraph_st.eval.svg import FactorMarkers, GeneScore, rank_factor_markers

__all__ = [
    "EnrichmentResult",
    "FactorMarkers",
    "GeneScore",
    "adjusted_rand_index",
    "benjamini_hochberg",
    "enrich_genes",
    "factor_redundancy",
    "held_out_reconstruction_error",
    "label_invariant_cluster_coherence",
    "matched_factor_correlation",
    "morans_i",
    "rank_factor_markers",
    "reconstruction_error",
    "section_overlap",
    "shared_private_separation",
]
