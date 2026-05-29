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

---

*Baseline-method provenance (HarveST / INSPIRE data bundles) is recorded in
[`../docs/DATASETS.md`](../docs/DATASETS.md) and
[`../BASELINE_REFERENCES.md`](../BASELINE_REFERENCES.md) — those are dataset
provenance only, not method endorsements.*
