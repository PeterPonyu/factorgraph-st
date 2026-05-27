# FactorGraph-ST baseline references

Verification date: 2026-05-27

## Baseline decision summary

| Role | Baseline | Decision |
|---|---|---|
| Primary | INSPIRE | Use as the primary public-code interpretable graph/factor integration baseline. |
| Secondary | HarveST | Use as public-code heterogeneous graph spatial-domain/SVG reference. |

## Primary baseline: INSPIRE

- Paper title: Interpretable, flexible and spatially aware integration of multiple spatial transcriptomics datasets from diverse sources
- Venue/date: Nature Genetics, published 2026-04-27
- DOI: 10.1038/s41588-026-02579-x
- Article URL: https://www.nature.com/articles/s41588-026-02579-x
- Code URL: https://github.com/jiazhao97/INSPIRE
- Zenodo: https://doi.org/10.5281/zenodo.18330972
- Verification date: 2026-05-27
- Default branch: main
- Observed HEAD SHA: 005d447374fea2820789d82936194690a40f69f0
- Archive status: not archived at GitHub verification time
- GitHub licenseInfo: null; no GitHub repository license inferred
- License note: Article is open access, but GitHub licenseInfo returned null; do not infer a repository software license. Code reuse beyond citation/clone testing requires license review.
- Local use: Primary comparison/reference for adversarial graph integration plus nonnegative matrix factorization-style spatial factors and gene programs.
- Fallback: If public code becomes unavailable, run GitHub/code search for a 2026 public-code interpretable spatial-integration/factor model. If none exists, select the newest suitable 2025 public-code graph-factor or spatial-domain integration reference and label the downgrade explicitly.
- Verification command/evidence:
  - `git ls-remote --heads https://github.com/jiazhao97/INSPIRE.git`
  - `gh repo view jiazhao97/INSPIRE --json licenseInfo,isArchived,defaultBranchRef`

## Secondary baseline: HarveST

- Paper title: HarveST uses a heterogeneous graph learning framework to reveal spatial transcriptomics patterns
- Venue/date: Communications Biology, published 2026-03-27
- DOI: 10.1038/s42003-026-09841-2
- Article URL: https://www.nature.com/articles/s42003-026-09841-2
- Code URL: https://github.com/Seven595/HarveST
- Zenodo: https://doi.org/10.5281/zenodo.18532348
- Verification date: 2026-05-27
- Default branch: main
- Observed HEAD SHA: 27ec12b303dd9a0ac8c9bc6accc99dd24fac48e2
- Archive status: not archived at GitHub verification time
- GitHub licenseInfo: MIT
- License note: GitHub licenseInfo reports MIT. Re-check before code reuse.
- Local use: Secondary reference for heterogeneous spot-gene/gene-gene graph construction, dual-learning spatial domain identification, and RWR-based domain-marker SVG ranking.
- Fallback: If public code becomes unavailable, mark as `deferred-unverified` or replace with another public-code 2026 graph-based spatial-domain/SVG method; do not list as verified public-code baseline.
- Verification command/evidence:
  - `git ls-remote --heads https://github.com/Seven595/HarveST.git`
  - `gh repo view Seven595/HarveST --json licenseInfo,isArchived,defaultBranchRef`
