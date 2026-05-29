# LITERATURE_LINKS — dataset source citations

Citation entries for the source paper of each dataset in the FactorGraph-ST
ingestion registry (`scripts/data/fetch_datasets.py`) and integration guide
(`docs/DATASETS.md`). `citation_key` matches the `Dataset.citation_key` field in
the registry and the `citation_key` in each `data/cards/*.yaml`. Links flagged
**⚠️ UNVERIFIED** where a direct artifact URL could not be confirmed; landing /
DOI pages are canonical. Program policy: raw integer counts + spatial metadata
only.

| citation_key | Datasets (registry ids) | Issues |
|---|---|---|
| `cervilla2026` | cervilla_2026_visium, cervilla_2026_xenium_cosmx, cervilla_2026_xenium_mt_5k | #32–#34 |
| `dawo_tls` | dawo_tls_visium | #35 |
| `gse175540_rcc` | gse175540_rcc_visium | #36 |
| `hest1k` | hest1k_cancer_subset | #37 |
| `janesick2023` | xenium_breast_janesick | #38 |
| `gse293199_tnbc` | gse293199_tnbc_xenium | #39 |
| `maynard2021` | dlpfc_maynard_2021 | #41, #133 |
| `wu2021` | wu_2021_breast_visium | #42 |
| `ji2020` | ji_2020_scc_st | #43 |
| `squidpy2022` | visium_mouse_brain_pair | #44 |
| `moffitt2018` | merfish_hypothalamus_shared | #45 |

---

## Tier A — production cancer cohorts

### `cervilla2026` — Cervilla 2026 matched multi-platform cancer cohort
Cervilla et al. 2026, *Genome Biology* (matched 6-cancer-type cohort profiled on
Visium/HD + Xenium/CosMx + Xenium-MT/5K). Data: Zenodo records
[17999961](https://zenodo.org/records/17999961) (Visium/HD),
[17986017](https://zenodo.org/records/17986017) (Xenium & CosMx),
[18000256](https://zenodo.org/records/18000256) (Xenium-MT & 5K).
Tracks issues #32–#34.

### `dawo_tls` — Dawo TLS kidney + lung Visium
Dawo et al., tertiary-lymphoid-structure carcinoma Visium (kidney + lung).
Data: Zenodo [14620362](https://zenodo.org/records/14620362)
(`TLS_VISIUM_USZ.zip`). Tracks #35.

### `gse175540_rcc` — Renal cell carcinoma Visium (24 sections)
Renal cell carcinoma Visium series (12 FFPE + 12 fresh-frozen). Data: GEO
[GSE175540](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE175540).
Tracks #36.

### `hest1k` — HEST-1k curated H&E + ST
Jaume et al. 2024, *HEST-1k: A Dataset for Spatial Transcriptomics and
Histology Image Analysis* (NeurIPS 2024). Data: Hugging Face
[MahmoodLab/hest](https://huggingface.co/datasets/MahmoodLab/hest) — take a
cancer Visium subset. Tracks #37.

### `janesick2023` — Xenium human breast cancer (Janesick)
Janesick et al. 2023, *High resolution mapping of the tumor microenvironment
using integrated single-cell, spatial and in situ analysis*, *Nature
Communications* 14, 8353. Data: 10x
[Xenium human breast demo](https://www.10xgenomics.com/products/xenium-in-situ/preview-dataset-human-breast)
(GEO GSE243280 mirror). Tracks #38.

### `gse293199_tnbc` — TNBC Xenium (OmiCLIP source)
Triple-negative breast cancer Xenium (280-gene panel; OmiCLIP source). Data:
GEO [GSE293199](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE293199).
Tracks #39.

## Tier B — Python-native quick-load

### `maynard2021` — DLPFC 12-section Visium
Maynard et al. 2021, *Transcriptome-scale spatial gene expression in the human
dorsolateral prefrontal cortex*, *Nature Neuroscience* 24, 425–436.
DOI [10.1038/s41593-020-00787-0](https://doi.org/10.1038/s41593-020-00787-0).
Data: figshare [22004273](https://figshare.com/articles/dataset/22004273)
(DOI 10.6084/m9.figshare.22004273.v2) / Bioconductor `spatialLIBD`. Tracks
#41, #133.

### `squidpy2022` — squidpy example Visium mouse brain
Palla et al. 2022, *Squidpy: a scalable framework for spatial omics analysis*,
*Nature Methods* 19, 171–178.
DOI [10.1038/s41592-021-01358-2](https://doi.org/10.1038/s41592-021-01358-2).
Data loaders: `squidpy.datasets.visium_hne_adata()` / `visium_fluo_adata()`
(10x public mouse brain). Tracks #44.

### `moffitt2018` — MERFISH mouse hypothalamus
Moffitt et al. 2018, *Molecular, spatial, and functional single-cell profiling
of the hypothalamic preoptic region*, *Science* 362, eaau5324.
DOI [10.1126/science.aau5324](https://doi.org/10.1126/science.aau5324).
Data: `squidpy.datasets.merfish()`; raw counts on Dryad
`doi:10.5061/dryad.8t8s248` ⚠️ UNVERIFIED direct URL. Tracks #45.

## Extended validation cohorts

### `wu2021` — Wu 2021 breast cancer Visium
Wu et al. 2021, *A single-cell and spatially resolved atlas of human breast
cancers*, *Nature Genetics* 53, 1334–1347.
DOI [10.1038/s41588-021-00911-1](https://doi.org/10.1038/s41588-021-00911-1).
Data: Zenodo [4739739](https://zenodo.org/records/4739739)
(`raw_count_matrices.tar.gz` + `spatial.tar.gz`). Tracks #42.

### `ji2020` — Ji 2020 cutaneous SCC (legacy ST)
Ji et al. 2020, *Multimodal Analysis of Composition and Spatial Architecture in
Human Squamous Cell Carcinoma*, *Cell* 182, 497–514.
DOI [10.1016/j.cell.2020.05.039](https://doi.org/10.1016/j.cell.2020.05.039).
Data: GEO [GSE144240](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE144240).
Tracks #43.

---

*Baseline-method provenance (HarveST / INSPIRE data bundles) is recorded in
[`../docs/DATASETS.md`](../docs/DATASETS.md) and
[`../BASELINE_REFERENCES.md`](../BASELINE_REFERENCES.md) — those are dataset
provenance only, not method endorsements.*
