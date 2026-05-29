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

---

## Tier A — production cancer cohorts

### `cervilla2026` — Cervilla 2026 matched multi-platform cancer cohort
Cervilla et al. 2026, *Genome Biology* (matched 6-cancer-type cohort profiled on
Visium/HD + Xenium/CosMx + Xenium-MT/5K). Data: Zenodo records
[17999961](https://zenodo.org/records/17999961) (Visium/HD),
[17986017](https://zenodo.org/records/17986017) (Xenium & CosMx),
[18000256](https://zenodo.org/records/18000256) (Xenium-MT & 5K).
Tracks issues #32–#34.

---

*Baseline-method provenance (HarveST / INSPIRE data bundles) is recorded in
[`../docs/DATASETS.md`](../docs/DATASETS.md) and
[`../BASELINE_REFERENCES.md`](../BASELINE_REFERENCES.md) — those are dataset
provenance only, not method endorsements.*
