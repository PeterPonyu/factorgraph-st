# Contributing to FactorGraph-ST

Thanks for contributing. This repository follows an evidence-gated, independence-first
workflow. This file is the single entry point that consolidates the guardrails that are
otherwise spread across the issue/PR templates and the governance docs
(`CLAIM_LEDGER.md`, `BASELINE_REFERENCES.md`, `docs/PROVENANCE_CHECKLIST.md`,
`docs/ALLOWED_BASELINE_CONTEXTS.md`).

> **Scope note.** FactorGraph-ST is an early-stage synthetic-only MVP. The current
> encoder/decoder is a deterministic, non-learned reference implementation, not a trained
> model. Keep all biological and performance claims marked *planned* until they are backed
> by local tests and dataset validation (see `CLAIM_LEDGER.md`).

## Before you open a PR

Run all of the following from the repository root and confirm each is green:

```bash
# 1. Independence + brand guards must exit 0.
bash scripts/check_independence.sh
bash scripts/check_brand_metadata.sh

# 2. Tests must pass. The package is configured with pythonpath=["src"], so a plain
#    pytest works out-of-the-box; an editable install also works.
python -m pytest -q
# (equivalently: PYTHONPATH=src python -m pytest -q, or `pip install -e ".[test]"` first)
```

If you touch any baseline-aware area, complete `docs/PROVENANCE_CHECKLIST.md` first.

## Claims and evidence

- Do **not** introduce performance or biology claims without local-test evidence. Record
  every claim and its supporting evidence in `CLAIM_LEDGER.md`.
- Do **not** reference sibling brands anywhere in tracked files. The only sanctioned
  exception is the byte-locked vendored `src/factorgraph_st/results_contract.py` (see
  `docs/ALLOWED_BASELINE_CONTEXTS.md`); `scripts/check_independence.sh` enforces this.
- Do **not** copy third-party (baseline) source into this repository, and do not add a git
  submodule that references one. Baseline reuse stays blocked until the upstream license is
  explicitly resolved (see `docs/PROVENANCE_CHECKLIST.md` and `BASELINE_REFERENCES.md`).

## Pull requests

- Every PR must reference its issues with the correct keyword: `Closes #N` (fully resolves,
  auto-closes on merge), `Refs #N` (related, does not close), or `Part of #N`.
- Complete the independence/provenance checklist in `.github/pull_request_template.md`.
- Keep PRs scoped to a single category (hygiene, docs, or a single fix) where practical.

## Branch hygiene and orphan-branch cleanup policy

Temporary branches (audit, parallel-work, or data-integration branches) are pushed, merged,
and then deleted on the remote. Their local tracking refs are left behind as `[gone]`
orphans, which risks basing new work on a stale guard version or re-introducing cross-brand
references. To prevent that:

1. **Always prune before starting work:**
   ```bash
   git fetch --prune
   ```
2. **Surface and delete merged orphans.** A local branch whose upstream is `[gone]` (shown by
   `git branch -vv`) has had its remote deleted. After confirming it is merged to `main`,
   delete it:
   ```bash
   bash scripts/list_orphans.sh        # lists [gone] branches
   git branch -d <branch>              # use -d (safe); -D only if you are sure
   ```
3. **Always rebase new work from a fresh `origin/main`** — never from a stale local `main` or
   an orphaned branch. This guarantees you pick up the current independence guard and the
   vendored `results_contract.py`. Re-run the guard scripts and `pytest -q` after rebasing.
4. **Data cards and `docs/DATASETS.md`:** never add sibling-brand coordination notes. Keep
   cross-brand mentions out of `data/cards/**` and `docs/DATASETS.md`; the only allowed place
   for sibling names is the explicitly excluded vendored contract file.

## Reporting issues

Use the templates under `.github/ISSUE_TEMPLATE/`. Bug reports should include a minimal
reproduction; docs/process issues should name the affected files.
