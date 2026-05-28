# ALLOWED_BASELINE_CONTEXTS

This document is the canonical allowlist of files and directories where
references to the upstream baseline repositories **INSPIRE** (`jiazhao97/INSPIRE`)
and **HarveST** (`Seven595/HarveST`) — including their author handles — are
**permitted**. References in any other location (especially `src/`, `tests/`,
`scripts/`) are prohibited and caught by `scripts/check_independence.sh`.

## Rationale

FactorGraph-ST uses INSPIRE and HarveST as prior-art baselines for comparison
and design pressure only. No third-party code is vendored. References in the
permitted files below are provenance citations, not framing adoption.

## Permitted files and directories

| Path | Purpose / permitted content |
|------|-----------------------------|
| `BASELINE_REFERENCES.md` | Full baseline registry — canonical provenance document |
| `baseline_repos/` | Checked-out baseline source trees (not vendored into src/) |
| `README.md` — "Baseline references" section | One-line citation of each baseline repo |
| `docs/ALLOWED_BASELINE_CONTEXTS.md` | This file (allowlist definition) |
| `docs/INDEPENDENCE_AND_LEAKAGE_AUDIT.md` | Independence audit records |
| `docs/LICENSE_NOTES.md` | License provenance notes |
| `docs/MVP_DESIGN.md` | Provenance disclaimer statement |
| `docs/NAME_AND_BRAND_REVIEW.md` | Name and brand review records |
| `docs/PROVENANCE_CHECKLIST.md` | Provenance checklist |
| `docs/SYNTHETIC_BENCHMARK.md` | Baseline-deferral statements |
| `docs/DRAFT_REVIEW_SUMMARY.md` | Draft review notes referencing baselines |
| `manuscript/DRAFT.md` | Baseline-comparison sections in the manuscript draft |
| `.github/ISSUE_TEMPLATE/*.yml` | Issue templates that reference baselines by name |
| `.github/pull_request_template.md` | PR template references |
| `scripts/check_independence.sh` | CI guard pattern strings (the pattern itself contains the names) |
| `scripts/check_brand_metadata.sh` | Brand metadata guard pattern strings |

## Prohibited zones

The following directories must contain **zero** INSPIRE/HarveST references
(enforced by `scripts/check_independence.sh` Section 2):

- `src/` — production package code
- `tests/` — test suite
- `scripts/` — tooling (except the CI guard scripts themselves, which are excluded by filename)

## Maintenance

When adding a new file that legitimately references INSPIRE or HarveST, add it
to the table above **in the same commit** as the new file, and ensure
`scripts/check_independence.sh` continues to pass.
