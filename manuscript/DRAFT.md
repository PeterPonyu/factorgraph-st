# FactorGraph-ST manuscript seed draft

## Working title

FactorGraph-ST: interpretable graph-factor learning of spatial domains and gene programs across heterogeneous spatial transcriptomics sections

## Draft abstract

Spatial transcriptomics analyses combine many heterogeneous sections, and the resulting embeddings and cluster labels are often hard to relate to specific genes. FactorGraph-ST is a planned method that pairs a spatial graph encoder with nonnegative shared/private factor decoding. The model is intended to produce section-aware embeddings and nonnegative gene loadings that can be read as spatial gene programs. It further aims to separate factors that are conserved across sections from those specific to individual samples. Initial development will compare against public-code interpretable-integration and heterogeneous-graph references without vendoring third-party implementations. All biological and performance claims remain planned until local implementation, tests, and dataset validation are complete.

## Implementation status

The current `src/factorgraph_st/` code is a deterministic, non-learned MVP baseline, not the planned learned model. `encode_graph` is a single neighbor-mean aggregation followed by a fixed random Gaussian projection (no trained parameters); `decode_factors` rectifies the embedding and fits gene loadings by clipped nonnegative least squares (not a learned NMF). The parametric, trained graph encoder described in the abstract is planned and not yet implemented.

## Proposed core contributions

1. A graph-factor schema for spot/cell embeddings, gene loadings, section labels, and spatial-domain assignments.
2. A shared/private factor objective for conserved and section-specific spatial gene programs.
3. A planned validation suite for factor recovery, domain coherence, spatial autocorrelation, and pathway/SVG sanity checks.

## Review questions

- Should first-pass factors be optimized for cross-section alignment or domain interpretability?
- Which synthetic generator should establish known shared/private factor ground truth?
- Which public-code baseline should be executed first if INSPIRE license review remains unresolved?
