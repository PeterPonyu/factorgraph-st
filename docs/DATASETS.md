# FactorGraph-ST dataset integration guide

This guide maps the **audited, real-world** spatial-transcriptomics (ST) cohorts
that FactorGraph-ST should ingest, and shows how each one supports the method's
core goal. It complements [`DATA.md`](DATA.md) (which covers the small
squidpy/figshare smoke-test stack: DLPFC, mouse-brain pair, MERFISH) — this file
is the roadmap for the production cancer cohorts that justify the method.

Canonical figures and links below come from the program registry
(`ST_research/datasets/DATASET_REGISTRY.md`, audited 2026-05-28). Where a hotlink
could not be confirmed it is marked **⚠️ UNVERIFIED** with the canonical landing
page. Program policy: **raw integer count matrices + spatial metadata only** — no
WSIs, FASTQs, BAMs, or normalized-only objects.

## Method recap (why these datasets)

FactorGraph-ST performs **interpretable cross-section factor learning**: a graph
encoder plus nonnegative shared/private factor decoding, so every learned factor
is inspectable as a spatial gene program, a section effect, or a domain-associated
signal (see [`MVP_DESIGN.md`](MVP_DESIGN.md)). The model consumes **multiple
heterogeneous sections at once** via the input contract
`X (n_spots, n_genes)` · `coords (n_spots, 2)` · `section_id (n_spots,)` ·
`edges (2, n_edges)`. Because shared-vs-private factor separation is only
identifiable when the same biology recurs across genuinely different sections, the
highest-value inputs are **true multi-platform matched cohorts** (the same tumours
profiled on several platforms) and **multiple independent cancer Visium
collections**. FactorGraph-ST is explicitly **not** histology-first and does not
depend on paired protein/ATAC/histology modalities.

## Recommended datasets

Ordered by value to cross-section factor learning. "Fit" = why the dataset
exercises shared/private factor separation across sections.

| # | Dataset | Accession / ID | Platform | Tissue / disease | Size (raw) | Link | Fit for cross-section factor learning |
|---|---------|----------------|----------|------------------|-----------|------|----------------------------------------|
| 1 | **Cervilla 2026 — Visium / Visium HD** | Zenodo **17999961** | Visium v1 / CytAssist / HD | 6 cancer types | 26.0 GB (16 files) | ✅ [zenodo.org/records/17999961](https://zenodo.org/records/17999961) | **Crown jewel, matched cohort.** Same 6 tumour types as records #2/#3 below, on Visium-family platforms — the spot-resolution arm of a true multi-platform matched cohort. Drives shared factors (conserved tumour programs) vs. private factors (platform/section effects). |
| 2 | **Cervilla 2026 — Xenium & CosMx** | Zenodo **17986017** | Xenium + CosMx | 6 cancer types | 23.6 GB (9 files) | ✅ [zenodo.org/records/17986017](https://zenodo.org/records/17986017) | Imaging-ST arm of the **same matched tumours**. Pairing it with record #1 lets shared factors be tested for invariance across spot vs. single-cell platforms — the strongest test of cross-platform shared/private identifiability. |
| 3 | **Cervilla 2026 — Xenium Multi-Tissue & 5K** | Zenodo **18000256** | Xenium Multi-Tissue + 5K panel | 6 cancer types | 46.8 GB (10 files) | ✅ [zenodo.org/records/18000256](https://zenodo.org/records/18000256) | High-plex (5K) imaging arm of the matched cohort; widest gene panel for resolving fine gene-program factors. Completes the three-platform-group matched set. |
| 4 | **Dawo TLS (Kidney + Lung Visium)** | Zenodo **14620362** | Visium v1 | kidney (3) + lung (5) carcinoma + TLS | 2.1 GB (`TLS_VISIUM_USZ.zip`) | ✅ [zenodo.org/records/14620362](https://zenodo.org/records/14620362) | 8 cancer Visium sections across **two organs** with tertiary-lymphoid-structure annotation. Cross-organ sections stress shared (immune/TLS) vs. private (organ-specific) factor separation. Small + fast for first multi-section integration. |
| 5 | **GSE175540 (RCC Visium)** | GEO **GSE175540** | Visium v1 | renal cell carcinoma | ~1.0 GB | ✅ [ncbi.nlm.nih.gov/geo GSE175540](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE175540) | 24 Visium sections (12 FFPE + 12 fresh-frozen) of one cancer type — a large **within-disease, cross-preservation** section set. Ideal for testing private factors that capture FFPE-vs-FF section effects while shared factors track RCC biology. |
| 6 | **HEST-1k subset** | HF `MahmoodLab/hest` | Visium v1/v2 + legacy ST | multi-organ H&E + ST | 8–12 GB (`st/*.h5ad`) | ✅ [huggingface.co/datasets/MahmoodLab/hest](https://huggingface.co/datasets/MahmoodLab/hest) | 1,229 curated samples; take a **cancer Visium subset** spanning many organs as a broad heterogeneous-section pool. Maximises section diversity for learning which factors are conserved vs. dataset-private. |
| 7 | **Xenium Breast Cancer (Janesick)** | 10x `Xenium_V1_human_Breast` demo | Xenium | breast cancer | 0.4–1.5 GB (subset; full outs 8–9 GB) | ✅ [10x Xenium human breast demo](https://www.10xgenomics.com/products/xenium-in-situ/preview-dataset-human-breast) | Canonical single-cell-resolution breast cancer Xenium (313-gene panel). Adds an independent imaging-ST cancer section to complement the Cervilla imaging arm and Visium collections. |
| 8 | **GSE293199 (TNBC Xenium)** | GEO **GSE293199** | Xenium | triple-negative breast cancer | 13.6 GB full (subset ~3 GB) | ✅ [ncbi.nlm.nih.gov/geo GSE293199](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE293199) | TNBC Xenium (280-gene panel), the OmiCLIP source. Pairs with #7 for a **two-collection breast-cancer imaging** comparison — shared factors should recover conserved breast-tumour programs across independent studies. |

### Cohort structure note — the matched multi-platform Cervilla cohort

Datasets **#1–#3 (Cervilla 2026)** are the central asset: the **same six cancer
types profiled across Visium/HD, Xenium/CosMx, and Xenium-MT/5K**. This is a rare
*true multi-platform matched cohort* — the exact regime FactorGraph-ST's
shared/private decomposition is designed for. Recommended first integration is the
**matched samples** (the tumours that appear in more than one of the three records)
so that shared factors can be validated for cross-platform invariance before the
full collection is pulled.

## Local resources & ingestion entry points

- **Ingestion script:** [`scripts/data/fetch_datasets.py`](../scripts/data/fetch_datasets.py)
  is the canonical entry point. It exposes a `DATASETS` registry, per-dataset
  loaders, and the multi-section assembly helpers `build_section_inputs(...)` and
  `build_spatial_graph(...)` that map a list of per-section `AnnData` objects onto
  the `X / coords / section_id / edges` contract. New cohorts are added as
  `Dataset` entries with a loader plus a `data/cards/<id>.yaml` provenance card.
- **Data cards / provenance:** [`data/cards/`](../data/cards/) — one YAML schema
  card per dataset (accession, platform, sections, license).
- **Repo-local raw cache:** `data/raw/<dataset_slug>/` (the script's `RAW_ROOT`),
  for raw counts + spatial metadata only.
- **Program-wide cache convention:** `~/Desktop/ST_research/data_cache/raw/<dataset_slug>/`
  (shared across the ST research program).
- **Verified papers:** `~/Desktop/ST_research/references/` (8 PDFs + index).
- **Provenance + download commands:** `st_dataset_provenance_and_policy.md` in the
  program registry directory (corrected `curl` commands; a few URLs flagged
  ⚠️ UNVERIFIED).
- **Canonical registry:** `~/Desktop/ST_research/datasets/DATASET_REGISTRY.md`
  (audited source of truth for all figures and links here).

### ⚠️ Network footprint — Cervilla is large

The three Cervilla Zenodo records total **~96 GB** of raw payload
(26.0 + 23.6 + 46.8 GB). A **selective matched-sample first pull is ~45–75 GB**
and dominates the FactorGraph-ST data budget. Use a selective-download strategy
(matched tumours first, single platform group at a time) rather than mirroring all
three records up front. `fetch_datasets.py` gates large payloads behind an
explicit `--download` flag, and `--dry-run` plans byte counts without touching the
network — use these before committing the footprint.
