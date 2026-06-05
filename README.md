# FactorGraph-ST

FactorGraph-ST is an independent spatial-transcriptomics method project for interpretable graph-factor learning of spatial domains and gene programs across heterogeneous ST sections.

## Scientific focus

The project centers on a future model that combines graph neural encoders with nonnegative shared/private factor decoding so each learned factor can be inspected as a spatial gene program, section effect, or domain-associated signal.

## Current implementation status

The default model in `scripts/run_real_factorgraph.py` is **GNMF** — a genuinely *trained* graph-regularized nonnegative matrix factorization (`model/learned.py`). It minimizes `‖X − H Wᵀ‖²_F + λ·Tr(Hᵀ L H)` over `H, W ≥ 0` by multiplicative updates, so spatial coherence is **learned** through the graph-Laplacian penalty (coordinates are never concatenated into the features) and domains are clustered on the factor scores `H` alone.

The previous default — the non-learned `projection` baseline (a **fixed random Gaussian projection** in `model/encoder.py` with no trained parameters, plus clipped-NNLS gene loadings in `model/decoder.py` and `[H|coords]` domains) — is retained as an opt-in baseline (`--model projection`), alongside `--model spatial_smooth` and the `--model coords` negative control.

Honesty note: GNMF is a trained matrix-factorization model, **not yet** the planned parametric *graph neural* encoder described under Scientific focus — that remains future work. All biological/performance claims stay gated in `CLAIM_LEDGER.md` until validated on labeled data.

## Baseline references

- Primary reference implementation for comparison and design pressure: INSPIRE (`jiazhao97/INSPIRE`).
- Secondary heterogeneous-graph domain/SVG reference: HarveST (`Seven595/HarveST`).

These repositories are treated as prior-art baselines. FactorGraph-ST must keep its own naming, APIs, prose, figures, schemas, and claims independent.

Permitted locations for INSPIRE/HarveST mentions are documented in [`docs/ALLOWED_BASELINE_CONTEXTS.md`](docs/ALLOWED_BASELINE_CONTEXTS.md). References outside those locations are blocked by CI.

## First-pass repository scope

- Baseline provenance lives in `BASELINE_REFERENCES.md`.
- Claim gates live in `CLAIM_LEDGER.md`.
- Baseline checkout commands live in `baseline_repos/README.md`; third-party code is not vendored.
- Draft review notes live in `manuscript/DRAFT.md` and `docs/DRAFT_REVIEW_SUMMARY.md`.

## Non-goals

- Not a missing-gene imputation or denoising project.
- Not a 3D reconstruction project.
- Not a generic clustering benchmark.
- Not histology-first.
- Not dependent on paired protein/ATAC/histology modalities.
