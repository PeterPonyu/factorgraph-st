# FactorGraph-ST — references (with code) & datasets

Consolidated reference + dataset index. Paper DOIs verified via Crossref and code
repositories via the GitHub API on 2026-06-09. See `BASELINE_REFERENCES.md`,
`docs/DATASETS.md`, and `manuscript/LITERATURE_LINKS.md` for full provenance.

## Reference papers & method baselines (with public code)

| Role | Method | Venue / year | DOI | Code |
|------|--------|--------------|-----|------|
| Primary | INSPIRE — interpretable, flexible, spatially-aware integration of multiple ST datasets | Nature Genetics 2026 | `10.1038/s41588-026-02579-x` | https://github.com/jiazhao97/INSPIRE (Zenodo 18330972) |
| Secondary | HarveST — heterogeneous graph learning reveals ST patterns | Communications Biology 2026 | `10.1038/s42003-026-09841-2` | https://github.com/Seven595/HarveST (Zenodo 18532348) |

## Datasets (audited registry — `docs/DATASETS.md`)

- ★ Cervilla 2026 matched multi-platform cohort: Zenodo **17999961** (Visium/HD), **17986017** (Xenium+CosMx), **18000256** (Xenium-MT & 5K)
- Dawo TLS kidney+lung Visium: Zenodo **14620362**
- RCC Visium 24 sections: GEO **GSE175540** · TNBC Xenium: GEO **GSE293199**
- Wu 2021 breast Visium: Zenodo **4739739** · Ji 2020 cSCC: GEO **GSE144240**
- Wang 2025 matched TMA: GEO **GSE308148** (Xenium) / **GSE308147** (MERSCOPE) / **GSE308146** (CosMx) / **GSE308145** (scRNA)
- HEST-1k (`MahmoodLab/hest`); Tier B: DLPFC (figshare 22004273 / spatialLIBD), squidpy mouse-brain pair, MERFISH hypothalamus

> Verification: INSPIRE + HarveST DOIs confirmed in Crossref; both code repos live via
> GitHub API. Zenodo 17999961/17986017/18000256/14620362 and GEO GSE293199/GSE175540/
> GSE308148 confirmed accessible (2026-06-09).
