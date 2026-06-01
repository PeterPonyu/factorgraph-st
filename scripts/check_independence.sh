#!/usr/bin/env bash
# Fail if any cross-brand reference appears in tracked files.
# Run from repo root.
set -euo pipefail

PATTERN='lumina-?st|aether-?3d|NicheLens|nichelens|niche-lens'
# results_contract.py is vendored BYTE-IDENTICALLY from the parent orchestration
# repo and its docstring legitimately names all four sibling projects (it is the
# single canonical results schema). It is byte-locked (see
# tests/test_contract_schema.py), so it is excluded here just like this script
# self-excludes above. __pycache__ holds compiled copies of that same file.
HITS=$(grep -rilE "$PATTERN" . \
    --exclude-dir=.git \
    --exclude-dir=.omc \
    --exclude-dir=baseline_repos \
    --exclude-dir=.venv \
    --exclude-dir=__pycache__ \
    --exclude-dir='*.egg-info' \
    --exclude='check_independence.sh' \
    --exclude='results_contract.py' 2>/dev/null || true)

if [ -n "$HITS" ]; then
    echo "FAIL: cross-brand references found in:" >&2
    echo "$HITS" >&2
    exit 1
fi

echo "PASS: no cross-brand references."

# ── Section 2: INSPIRE / HarveST leakage guard (src/, tests/, scripts/) ──────
# These directories must never import, copy, or mimic INSPIRE/HarveST identifiers.
# Allowed locations are documented in docs/ALLOWED_BASELINE_CONTEXTS.md.
INSPIRE_PATTERN='inspire|harvest|jiazhao97|seven595'
INSPIRE_HITS=$(grep -rilE "$INSPIRE_PATTERN" src/ tests/ scripts/ \
    --exclude-dir=__pycache__ \
    --exclude-dir='*.egg-info' \
    --exclude='check_independence.sh' \
    --exclude='check_brand_metadata.sh' 2>/dev/null || true)

if [ -n "$INSPIRE_HITS" ]; then
    echo "FAIL: INSPIRE/HarveST references found in prohibited zones (src/, tests/, scripts/):" >&2
    echo "$INSPIRE_HITS" >&2
    echo "See docs/ALLOWED_BASELINE_CONTEXTS.md for permitted locations." >&2
    exit 1
fi

echo "PASS: no INSPIRE/HarveST references in prohibited zones."
