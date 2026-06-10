# FactorGraph-ST manuscript seed draft

## Working title

FactorGraph-ST: interpretable graph-factor learning of spatial domains and gene programs across heterogeneous spatial transcriptomics sections

## Draft abstract

Spatial transcriptomics analyses combine heterogeneous sections, and interpretable gene programs require models whose latent factors can be inspected directly. The code currently implemented in FactorGraph-ST is a graph-regularized nonnegative matrix factorization (GNMF): it learns nonnegative spot factors and gene loadings while penalizing factor variation over a spatial graph Laplacian, then clusters domains from the learned factor scores. This draft therefore reports GNMF as the implemented method and treats the parametric graph neural encoder / shared-private decoder as planned future work, not as an available contribution. All biological and performance claims remain planned until the GNMF implementation is validated on labeled data and compared with reproduced baselines.

## Implementation status

The current default real-data path is GNMF (`scripts/run_real_factorgraph.py --model gnmf` and `src/factorgraph_st/model/learned.py`), not the planned parametric graph neural encoder. GNMF is a trained numpy implementation of graph-regularized nonnegative matrix factorization: it optimizes nonnegative spot factors `H` and gene loadings `W` with a graph-Laplacian smoothness penalty, and domains are clustered from `H` alone. The older fixed random Gaussian projection path remains available only as `--model projection` for ablation/baseline context.

## Results (preliminary — honest-negative)

The implemented GNMF path is currently an honest-negative result, not a validated
improvement. In the reproduced cross-dataset spatial-domain accuracy sweep, trained GNMF is
dominated by trivial coordinate and spatial-smoothing controls on both labeled datasets:

| dataset | coords ARI | GNMF ARI | spatial_smooth ARI |
|---|---:|---:|---:|
| DLPFC / Maynard 2021 Visium | 0.2211 | 0.1556 | 0.2517 |
| STARmap mouse visual cortex / Wang 2018 | 0.2619 | 0.1760 | 0.5798 |

Source: `results/cross_dataset/tables/domain_accuracy.md`. We therefore make **no claim** that
the learned model beats simple spatial or coordinate baselines; published external baselines
have not yet been reproduced in this repository, and all performance claims remain `planned`.

## Proposed core contributions

1. A graph-factor schema for spot/cell embeddings, gene loadings, section labels, and spatial-domain assignments.
2. A shared/private factor objective for conserved and section-specific spatial gene programs.
3. A planned validation suite for factor recovery, domain coherence, spatial autocorrelation, and pathway/SVG sanity checks.

## Review questions

- Should first-pass factors be optimized for cross-section alignment or domain interpretability?
- Which synthetic generator should establish known shared/private factor ground truth?
- Which public-code baseline should be executed first if INSPIRE license review remains unresolved?
