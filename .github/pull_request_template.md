<!--
FactorGraph-ST pull request template.
-->

## Summary

<one-paragraph description of the change>

## Closes / refs

Closes #<issue>

## Independence and provenance checklist

- [ ] No third-party (baseline) source is added to this repository.
- [ ] No git submodule references a baseline repository.
- [ ] `bash scripts/check_independence.sh` exits 0.
- [ ] No cross-brand references (lumina-st, aether-3d, niche-lens-st).
- [ ] If touching code that touches a baseline area: INSPIRE source has **not** been copied. Reuse remains blocked until its repository license is explicitly resolved (see `docs/PROVENANCE_CHECKLIST.md`).
- [ ] No performance or biology claim is asserted without local-test evidence (see `CLAIM_LEDGER.md`).
- [ ] If `BASELINE_REFERENCES.md` is touched, HEAD / license / provenance fields are re-checked.
