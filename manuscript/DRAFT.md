# FactorGraph-ST manuscript seed draft

## Working title

FactorGraph-ST: interpretable graph-factor learning of spatial domains and gene programs across heterogeneous spatial transcriptomics sections

## Draft abstract

Spatial transcriptomics studies often require both integration across heterogeneous sections and interpretable spatial programs, yet opaque embeddings and cluster labels can obscure which genes drive tissue organization. FactorGraph-ST is a planned method for coupling spatial graph encoders with nonnegative shared/private factor decoding. The proposed model will learn section-aware embeddings, decode them into inspectable spatial gene programs, and distinguish conserved tissue factors from sample-specific signals. The first development phase will benchmark against public-code interpretable integration and heterogeneous-graph references without vendoring third-party implementations. All biological and performance claims remain planned until local implementation, tests, and dataset validation are complete.

## Proposed core contributions

1. A graph-factor schema for spot/cell embeddings, gene loadings, section labels, and spatial-domain assignments.
2. A shared/private factor objective for conserved and section-specific spatial gene programs.
3. A planned validation suite for factor recovery, domain coherence, spatial autocorrelation, and pathway/SVG sanity checks.

## Review questions

- Should first-pass factors be optimized for cross-section alignment or domain interpretability?
- Which synthetic generator should establish known shared/private factor ground truth?
- Which public-code baseline should be executed first if INSPIRE license review remains unresolved?
