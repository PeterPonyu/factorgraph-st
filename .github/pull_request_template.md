<!--
FactorGraph-ST pull request template.
-->

## Summary

<!-- One-paragraph description of the change. -->

## Cross-references

<!-- Link every related issue using the correct keyword:
     - `Closes #N`  — fully resolves the issue (auto-closes on merge)
     - `Refs #N`    — partial / related work that does NOT close the issue
     - `Part of #N` — contributes to a larger meta-issue or roadmap

  List ALL issues this PR touches, one per line. Examples:
    Closes #42
    Refs #17, #23
    Part of #130
-->

Closes #

## Test plan

- [ ] <!-- How to verify this change (command, screenshot, etc.). -->

## Independence and provenance checklist

- [ ] No third-party (baseline) source is added to this repository.
- [ ] No git submodule references a baseline repository.
- [ ] `bash scripts/check_independence.sh` exits 0.
- [ ] No cross-brand references.
- [ ] If touching code that touches a baseline area: INSPIRE source has **not** been copied. Reuse remains blocked until its repository license is explicitly resolved (see `docs/PROVENANCE_CHECKLIST.md`).
- [ ] No performance or biology claim is asserted without local-test evidence (see `CLAIM_LEDGER.md`).
- [ ] If `BASELINE_REFERENCES.md` is touched, HEAD / license / provenance fields are re-checked.
