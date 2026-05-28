#!/usr/bin/env bash
# Verify that pyproject.toml and CITATION.cff carry the canonical FactorGraph-ST brand names.
# Fails if either file contains a stale or incorrect project name / title.
# Run from repo root.
set -euo pipefail

# Check pyproject.toml project name
grep -qE '^name = "factorgraph-st"' pyproject.toml \
    && echo 'PASS: pyproject.toml name = "factorgraph-st"' \
    || { echo 'FAIL: pyproject.toml name field missing or does not equal "factorgraph-st"' >&2; exit 1; }

# Check CITATION.cff title
grep -qF 'title: "FactorGraph-ST"' CITATION.cff \
    && echo 'PASS: CITATION.cff title = "FactorGraph-ST"' \
    || { echo 'FAIL: CITATION.cff title does not match "FactorGraph-ST"' >&2; exit 1; }
