# FactorGraph-ST manuscript seed draft

## Working title

FactorGraph-ST: interpretable graph-factor learning of spatial domains and gene programs across heterogeneous spatial transcriptomics sections

## Draft abstract

Spatial transcriptomics analyses combine heterogeneous sections, and interpretable gene programs require models whose latent factors can be inspected directly. The code currently implemented in FactorGraph-ST is a graph-regularized nonnegative matrix factorization (GNMF): it learns nonnegative spot factors and gene loadings while penalizing factor variation over a spatial graph Laplacian, then clusters domains from the learned factor scores. This draft therefore reports GNMF as the implemented method and treats the parametric graph neural encoder / shared-private decoder as planned future work, not as an available contribution. All biological and performance claims remain planned until the GNMF implementation is validated on labeled data and compared with reproduced baselines.

## Implementation status

The current default real-data path is GNMF (`scripts/run_real_factorgraph.py --model gnmf` and `src/factorgraph_st/model/learned.py`), not the planned parametric graph neural encoder. GNMF is a trained numpy implementation of graph-regularized nonnegative matrix factorization: it optimizes nonnegative spot factors `H` and gene loadings `W` with a graph-Laplacian smoothness penalty, and domains are clustered from `H` alone. The older fixed random Gaussian projection path remains available only as `--model projection` for ablation/baseline context.

## Proposed core contributions

1. A graph-factor schema for spot/cell embeddings, gene loadings, section labels, and spatial-domain assignments.
2. A shared/private factor objective for conserved and section-specific spatial gene programs.
3. A planned validation suite for factor recovery, domain coherence, spatial autocorrelation, and pathway/SVG sanity checks.

## Review questions

- Should first-pass factors be optimized for cross-section alignment or domain interpretability?
- Which synthetic generator should establish known shared/private factor ground truth?
- Which public-code baseline should be executed first if INSPIRE license review remains unresolved?
