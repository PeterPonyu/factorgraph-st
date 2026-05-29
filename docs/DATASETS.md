# FactorGraph-ST dataset integration guide

This guide maps the **audited, real-world** spatial-transcriptomics (ST) cohorts
that FactorGraph-ST should ingest, and shows how each one supports the method's
core goal. It covers two tiers: **Tier A** — the production cancer cohorts that
justify the method (large, histology-paired raw-count collections) — and
**Tier B** — a small Python-native squidpy/scanpy quick-load stack (DLPFC
12-section, mouse-brain pair, MERFISH) for fast iteration and smoke tests before
committing the Tier A network budget. Both tiers mirror the program registry
(`ST_research/datasets/DATASET_REGISTRY.md`).

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

## Tier A — production cancer cohorts (raw counts + spatial)

Ordered by value to cross-section factor learning. "Fit" = why the dataset
exercises shared/private factor separation across sections. This is the
factorgraph-st slice of **Tier A** in the program registry
(`ST_research/datasets/DATASET_REGISTRY.md`).

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

## Tier B — Python-native quick-load (squidpy/scanpy)

A small, zero-glue stack of one-line loaders (plus the DLPFC figshare benchmark)
for fast iteration, smoke tests, and structure checks before committing the
Tier A network budget. All loaders were confirmed importable + loadable in the
`dl` conda env (scanpy 1.10.4, squidpy 1.6.5, anndata 0.10.7, spatialdata 0.4.0)
on 2026-05-28. These mirror **Tier B** of the program registry
(`ST_research/datasets/DATASET_REGISTRY.md`); full network-verification evidence
is in `ST_research/references/DATASET_CATALOG.md`.

> Quick start
> ```bash
> conda run --no-capture-output -n dl python scripts/data/fetch_datasets.py --list
> conda run --no-capture-output -n dl python scripts/data/fetch_datasets.py \
>     --dataset visium_mouse_brain_pair --dry-run
> ```

### ★ Primary — DLPFC 12-section Visium (Maynard 2021)

The canonical multi-section spatial-domain benchmark: **12 Visium sections, 3
donors, manual cortical-layer labels** (Layer 1–6 + white matter) — the reference
for measuring spatial-domain recovery across heterogeneous sections. Not a
one-line loader, but the primary multi-section domain pick.

- **Access:** figshare article `22004273` (*Visium DLPFC preprocessed .h5ad*,
  DOI `10.6084/m9.figshare.22004273.v2`) → per-section files
  `151507.h5ad … 151676.h5ad` via `https://ndownloader.figshare.com/files/<id>`.
  Real per-section file IDs are recorded in
  [`../data/cards/dlpfc_maynard_2021.yaml`](../data/cards/dlpfc_maynard_2021.yaml)
  and in `scripts/data/fetch_datasets.py`. Alternative: Bioconductor `spatialLIBD`.
- **Platform:** 10x Visium (spot) · **Size:** ~1.3 GB (12 × ~0.1 GB `.h5ad`) ·
  **Sections:** 12 (`151507–151510`, `151669–151676`).
- **Labels:** manual cortical-layer annotation in `obs`.
- **Raw counts:** normalized matrix in `X`; raw UMI counts preserved in `.raw`.
- **License:** CC BY 4.0.

```bash
conda run --no-capture-output -n dl python scripts/data/fetch_datasets.py \
    --dataset dlpfc_maynard_2021 --download    # ~1.3 GB; omit --download to plan only
```

### B4 — Visium adult mouse brain coronal pair (raw in `.raw`)

Two coronal sections of the same adult mouse brain — a fast, fully Python-native
two-section stack for smoke tests and quick domain experiments.

- **Access:** `squidpy.datasets.visium_hne_adata()` (H&E) + `visium_fluo_adata()`
  (fluorescence).
- **Platform:** 10x Visium · **Sections:** 2 (not z-registered) ·
  ~2,688 / ~2,800 spots; ~18k / ~16k genes.
- **Labels:** Leiden clusters in `obs` (clustering, not curated domains).
- **Raw counts:** **yes — in `.raw`** of each section (safe for DL training).
- **License:** 10x public / squidpy example data.
- Card: [`../data/cards/visium_mouse_brain_pair.yaml`](../data/cards/visium_mouse_brain_pair.yaml).
  Shared 2-section anchor across repos in this research program.

### B1 — MERFISH mouse hypothalamus (Moffitt 2018; 8 AP sections)

Single-cell-resolution imaging ST with **8 anterior–posterior levels**
(`obs['Bregma']`) usable as multiple sections — a heterogeneous, non-Visium stress
test for the section-aware encoder.

- **Access:** `squidpy.datasets.merfish()` · **Platform:** MERFISH (imaging) ·
  73,655 cells × 161 genes · 8 AP sections (`obs['Bregma']`; 3D coords in
  `obsm['spatial3d']`) · **Labels:** `obs['Cell_class']`.
- **Raw counts:** the squidpy copy is **normalized**; for DL training pull raw
  counts from Dryad `doi:10.5061/dryad.8t8s248` (CC0, ~1.03 GB).
- Card: [`../data/cards/merfish_hypothalamus_shared.yaml`](../data/cards/merfish_hypothalamus_shared.yaml).
  Shared multi-section anchor across repos in this research program.

### Assembling multi-section inputs + building the kNN graph

`scripts/data/fetch_datasets.py` ships helpers that turn a list of per-section
AnnData objects into the `X / coords / section_id / edges` contract (see
[Method recap](#method-recap-why-these-datasets)), building the per-section graph
with `squidpy.gr.spatial_neighbors`:

```python
import scanpy as sc
import squidpy as sq
from scripts.data.fetch_datasets import build_section_inputs, build_spatial_graph

# 1) Load several sections (e.g. a few DLPFC samples after download)
sections = [sc.read_h5ad(f"data/raw/dlpfc_maynard_2021/{s}.h5ad")
            for s in ("151507", "151508", "151509")]

# 2) Assemble multi-section inputs: X (stacked, raw via .raw), coords,
#    integer section_id, global edges.
inputs = build_section_inputs(sections, n_neighs=6, coord_type="grid", use_raw=True)
X, coords, section_id, edges = (
    inputs["X"], inputs["coords"], inputs["section_id"], inputs["edges"]
)

# Single-section graph build (Visium uses the hex grid; imaging platforms such
# as MERFISH use coord_type="generic"):
sq.gr.spatial_neighbors(sections[0], coord_type="grid", n_neighs=6)
edges_0 = build_spatial_graph(sections[0], n_neighs=6, coord_type="grid")
```

`section_id` is a contiguous integer per section; `edges` are offset so node
indices are global across the concatenated stack — exactly the COO format the
model expects.

### Tier B dataset cards

| Card | Dataset | Registry Tier B |
|---|---|---|
| [`dlpfc_maynard_2021.yaml`](../data/cards/dlpfc_maynard_2021.yaml) | DLPFC 12-section Visium | primary multi-section |
| [`visium_mouse_brain_pair.yaml`](../data/cards/visium_mouse_brain_pair.yaml) | Visium mouse brain coronal pair | B4 |
| [`merfish_hypothalamus_shared.yaml`](../data/cards/merfish_hypothalamus_shared.yaml) | MERFISH hypothalamus (8 AP sections) | B1 |

`data/raw/` and `data/processed/` are git-ignored; `data/cards/` is tracked.

### Baseline data provenance (citation only)

For reproducing prior-art comparison setups, the secondary baseline **HarveST**
publishes a data bundle on Zenodo (`10.5281/zenodo.18532348`, `HarveST_Data.zip`)
containing DLPFC / MOB / HBRC / PDAC sections. The **INSPIRE** (`zenodo 18330972`)
and **HarveST** Zenodo records are otherwise **code-only** (GitHub release
archives), not primary data hosts — FactorGraph-ST therefore pulls its benchmark
data from the platform sources above (figshare / squidpy / Dryad), not baseline
bundles. These names appear here purely as dataset provenance; baseline *method*
details live in [`../BASELINE_REFERENCES.md`](../BASELINE_REFERENCES.md) (see
[`ALLOWED_BASELINE_CONTEXTS.md`](ALLOWED_BASELINE_CONTEXTS.md)).

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

## Extended validation datasets

A raw-count verification pass (2026-05-28) confirmed that **all eight Tier A
cohorts above distribute genuine raw integer count matrices** (not normalized-only
objects), and added cohorts that close cross-study / cross-platform gaps.
Verification methods: direct download + listing of the smallest Cervilla Visium
archive; **remote ZIP central-directory inspection** (HTTP range requests, no full
download) for the imaging arms and the Dawo bundle; and **streamed `tar` header
parsing** for the GEO `*_RAW.tar` archives.

### Raw-count verification — Tier A cohorts

| # | Dataset | Issue | Raw-count artifact (verified) | Verdict |
|---|---------|-------|-------------------------------|---------|
| 1 | Cervilla — Visium/HD (Zenodo 17999961) | [#32](https://github.com/PeterPonyu/factorgraph-st/issues/32) | `visium_*/outs/filtered_feature_bc_matrix.h5` + `outs/spatial/` (Space Ranger); HD bins in `visiumhd_*` | ✅ |
| 2 | Cervilla — Xenium & CosMx (17986017) | [#33](https://github.com/PeterPonyu/factorgraph-st/issues/33) | Xenium `cell_feature_matrix.h5`; CosMx `*_exprMat_file.csv.gz` (+ metadata/tx/polygons/fov) | ✅ |
| 3 | Cervilla — XeniumMT & 5K (18000256) | [#34](https://github.com/PeterPonyu/factorgraph-st/issues/34) | `cell_feature_matrix.h5` per panel (377-gene MT + Human Prime 5K) + transcripts/boundaries | ✅ |
| 4 | Dawo TLS (14620362) | [#35](https://github.com/PeterPonyu/factorgraph-st/issues/35) | `10x_Visium/<sample>/raw_feature_bc_matrix/` (+ `filtered_` + `spatial/`), 8 sections | ✅ |
| 5 | GSE175540 RCC | [#36](https://github.com/PeterPonyu/factorgraph-st/issues/36) | per-GSM `*_filtered_feature_bc_matrix.h5` inside `GSE175540_RAW.tar` (24 sections, FFPE+FF) | ✅ |
| 6 | HEST-1k cancer subset | [#37](https://github.com/PeterPonyu/factorgraph-st/issues/37) | `st/*.h5ad` → `adata.X` raw counts (raw-count QC annotations present; assert int at load) | ✅ |
| 7 | Xenium Breast (Janesick) | [#38](https://github.com/PeterPonyu/factorgraph-st/issues/38) | `*_cell_feature_matrix.h5` (10x bundle, Rep1/Rep2; GEO GSE243280 mirror) | ✅ |
| 8 | GSE293199 TNBC Xenium | [#39](https://github.com/PeterPonyu/factorgraph-st/issues/39) | `cell_feature_matrix.h5` inside nested `*_Xenium_outs.tar.gz` in `GSE293199_RAW.tar` | ✅ ⚠️ single ~14.6 GB tarball (no trivial 3 GB subset) |

No Tier A cohort failed verification, so **no replacements were required**. One
provenance caveat to carry into the loaders: GSE293199 ships a single
~14.6 GB nested `*_Xenium_outs.tar.gz`, so the loader must download the full outs
and extract `cell_feature_matrix.h5` (the registry's "subset ~3 GB" is optimistic).

### New cohorts (extended validation)

| # | Dataset | Accession | Platform | Tissue / disease | Raw-count artifact (verified) | Issue | Fit for cross-section factors |
|---|---------|-----------|----------|------------------|-------------------------------|-------|-------------------------------|
| E1 | **Wu 2021 breast cancer Visium** | Zenodo **4739739** | Visium v1 | human breast cancer (multi-subtype) | `raw_count_matrices.tar.gz` (252.9 MB) + `spatial.tar.gz` | [#42](https://github.com/PeterPonyu/factorgraph-st/issues/42) | Independent multi-section breast Visium; with Tier A #1/#7/#8 enables cross-study/cross-platform conserved breast-tumour factor tests. |
| E2 | **Ji 2020 cutaneous SCC ST** | GEO **GSE144240** | legacy ST array | cutaneous SCC — multi-patient, tumor + matched normal | `*_stdata.tsv.gz` + `*_spot_data-selection-*.tsv.gz` in `GSE144240_RAW.tar` | [#43](https://github.com/PeterPonyu/factorgraph-st/issues/43) | Tumor-vs-normal + patient-private factor structure; adds legacy-ST platform breadth; paired scRNA for factor validation. |

### Raw-count hardening — DLPFC (Tier B)

The Tier B **DLPFC 12-section** entry currently loads the figshare object whose `X`
is *normalized* (raw UMIs kept only in `.raw`). Issue
[#41](https://github.com/PeterPonyu/factorgraph-st/issues/41) hardens its
provenance to a **direct raw-count source**: `spatialLIBD::fetch_data("spe")` exposes
a `counts` assay of raw integer 10x counts, and the spatialLIBD project also
distributes per-sample `h5_raw` / `h5_filtered` Space Ranger matrices — preferable
to a normalized-X object under the raw-counts-only policy. This augments the
existing Tier B DLPFC row rather than adding a new cohort.

## Consolidated ingestion roadmap (registry ↔ issues)

The canonical machine-readable registry is
[`scripts/data/fetch_datasets.py`](../scripts/data/fetch_datasets.py) (`DATASETS`
dict). Source-paper citations are in
[`../manuscript/LITERATURE_LINKS.md`](../manuscript/LITERATURE_LINKS.md). Depth of
this PR = **framework + registry + data cards** (not full runnable loaders): every
dataset has a registry entry, contract mapping, and raw-count policy; downloads are
gated and **only squidpy-backed loaders are wired** — every accession-based direct
URL stays an ⚠️ UNVERIFIED guarded stub until resolved.

| Registry id | Dataset | Tier | Issue(s) | Fetch status |
|---|---|---|---|---|
| `cervilla_2026_visium` | Cervilla Visium/HD | A | [#32](https://github.com/PeterPonyu/factorgraph-st/issues/32) | stub (URL UNVERIFIED) |
| `cervilla_2026_xenium_cosmx` | Cervilla Xenium & CosMx | A | [#33](https://github.com/PeterPonyu/factorgraph-st/issues/33) | stub (URL UNVERIFIED) |
| `cervilla_2026_xenium_mt_5k` | Cervilla Xenium-MT & 5K | A | [#34](https://github.com/PeterPonyu/factorgraph-st/issues/34) | stub (URL UNVERIFIED) |
| `dawo_tls_visium` | Dawo TLS kidney+lung | A | [#35](https://github.com/PeterPonyu/factorgraph-st/issues/35) | stub (URL UNVERIFIED) |
| `gse175540_rcc_visium` | RCC Visium (24 sec) | A | [#36](https://github.com/PeterPonyu/factorgraph-st/issues/36) | stub (URL UNVERIFIED) |
| `hest1k_cancer_subset` | HEST-1k cancer subset | A | [#37](https://github.com/PeterPonyu/factorgraph-st/issues/37) | stub (URL UNVERIFIED) |

Related tracking: [#52](https://github.com/PeterPonyu/factorgraph-st/issues/52)
(fetch CLI test coverage — closed by `tests/data/test_fetch_datasets_cli.py`),
[#130](https://github.com/PeterPonyu/factorgraph-st/issues/130) (meta: surface the
05-29 cards / fetch surface / LITERATURE_LINKS into DRAFT PR #40), and the
data-readiness issues
[#102](https://github.com/PeterPonyu/factorgraph-st/issues/102) (raw-count
distribution assumptions) /
[#103](https://github.com/PeterPonyu/factorgraph-st/issues/103) (ground-truth
metrics on real data).

> Verify the surface offline:
> ```bash
> python scripts/data/fetch_datasets.py --list           # network-free, exit 0
> python scripts/data/fetch_datasets.py --dataset wang_2025_ist_ffpe_xenium --dry-run
> python -m pytest tests/data/ -q                         # registry/CLI smoke tests
> ```
