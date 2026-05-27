# FactorGraph-ST first-pass MVP API

Status: design document. All biology and performance claims remain planned until local tests and validated data exist (see `CLAIM_LEDGER.md`). No third-party source is vendored; INSPIRE and HarveST are cited only as comparison references (see `BASELINE_REFERENCES.md`).

## Inputs

| Object | Shape | Type | Notes |
|---|---|---|---|
| `X` spot features | `(n_spots, n_genes)` | float32, dense or CSR | Per-spot/cell expression. |
| `coords` | `(n_spots, 2)` | float32 | 2D coords; 3D deferred. |
| `section_id` | `(n_spots,)` | int | Section index per spot. |
| `edges` | `(2, n_edges)` | int64 COO | k-NN or Delaunay graph over `coords`, built per section. |

## Outputs

| Object | Shape | Type | Notes |
|---|---|---|---|
| `H` spot embedding | `(n_spots, d)` | float32 | Section-aware embedding. |
| `W` gene-factor loading | `(n_genes, K)` | float32, `≥0` | Nonnegative gene loadings. |
| `Z_shared` | `(n_spots, K_s)` | float32, `≥0` | Conserved-factor activations. |
| `Z_private` | `(n_spots, K_p)` | float32, `≥0` | Section-private factor activations. |
| `domain_id` | `(n_spots,)` | int | Spatial-domain assignment. |

## Factor schema

- `K = K_s + K_p`; `K_s` and `K_p` configurable.
- Nonnegativity: `W ≥ 0`, `Z_shared ≥ 0`, `Z_private ≥ 0`.
- `domain_id` is derived from `H` by a configurable clustering step.

## Acceptance matrix

| Output | Claim-ledger row | Future test hook |
|---|---|---|
| `Z_shared`, `Z_private` | "recover interpretable shared/private factors" | `tests/test_factor_recovery.py` |
| `H`, `domain_id` | "align sections while preserving spatial structure" | `tests/test_alignment_and_domains.py` |
| `W`, `domain_id` | "improve domain/gene-program interpretability" | `tests/test_domain_interpretability.py` |

## Out of scope for MVP

- Histology fusion.
- 3D reconstruction.
- Multi-modal (ATAC, protein) input.
- Performance comparison against baselines (deferred until validated; see `BASELINE_REFERENCES.md`).
