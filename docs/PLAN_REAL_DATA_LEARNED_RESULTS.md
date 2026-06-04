# Plan — Real-data learned-method results

**Status:** active · **Milestone:** Real-data learned-method results · **Roadmap issue:** #345
**Created:** 2026-06-04

## Goal
Turn the freshly-integrated learned (graph-regularized NMF) factor model into the
project's first **real-data**, **GT-skeptical** benchmark results — meaningful
metrics emitted through the uniform results contract, consumed by figures/tables.

## Why this is unblocked now
The learned model is integrated into the default `src` path as an opt-in
(`--model gnmf`, default `projection`) — item 1 of #326. This addresses the root
cause flagged in #181: the previous fixed random-projection encoder made
factor-recovery metrics meaningless. With a real learned model available, real
results become interpretable for the first time.

## Sequence (each step = a tracked issue)

| # | Step | Issue | Notes |
|---|------|-------|-------|
| 1 | Integrate learned model (opt-in, non-breaking) | #326 (item 1) | **done** — 152 tests green, `--model {projection,gnmf}` default `projection` |
| 2 | Metric-gating **policy** | #341 | internal metrics default-on; reference-label metrics (NMI/AMI/ARI/Dice/boundary-F1) gated to class-A datasets only |
| 3 | Implement gated metrics | #332 (AMI/NMI), #326 (item 2) | tag each metric `internal` vs `reference_label` |
| 4 | First real-data benchmark | #342 | DLPFC, `projection` vs `gnmf`, emit via contract, record runtime/peak-mem |
| 5 | Consume metrics in figures/tables | #330, #336, #334, #335 | read emitted JSON; no circular accuracy claims |
| 6 | Close the loop | #181, #138 | update root-cause + results trackers; reconcile/close the stranded branch |

## The GT-skeptical metric policy (step 2, the gate before headline numbers)
Two metric families, handled differently:
- **Internal / label-free** — silhouette, Calinski–Harabasz, spatial coherence,
  factor diversity. Valid on *any* dataset (score the partition against the
  data/geometry). **Default-on everywhere.**
- **Reference-label-based** — NMI, AMI, ARI, weighted Dice, boundary-F1. These
  compare predicted domains to annotation labels that are themselves clustering +
  manual annotation, so scoring against them is partly circular for most ST data.
  **Gate** behind a per-dataset `gt_quality: class_A` capability flag (e.g. DLPFC
  cortical layers). On non-class-A data, emit only as labelled supplementary
  diagnostics, never as accuracy headlines.

## Concrete first run (step 4)
```
# class-A dataset already ingested (DLPFC); same seed/config, two encoders
python scripts/run_real_factorgraph.py --model projection ...   # baseline
python scripts/run_real_factorgraph.py --model gnmf       ...   # learned
# -> results/factorgraph-st/{metrics,run_metadata}.json for each, git_sha = HEAD
```
Acceptance: contract-valid metrics for both encoders on a class-A dataset at the
current `git_sha`; internal metrics for both; reference-label metrics only on
class-A; a recorded projection-vs-gnmf comparison.

## Out of scope (separate tracks)
- Cross-repo real-data rollout for lumina-st / aether-3d / niche-lens-st — each
  has its own `[results]` tracker; replicate this runner pattern there once the
  factorgraph path is proven.
- Pushing local commits — operational (the local branches diverge from the
  rewritten remotes; reset-first reconciliation), handled outside this plan.

## Links
- Milestone: `Real-data learned-method results`
- Roadmap issue: #345
- Audit context: `../.omc/audits/FULL_PARITY_AUDIT_2026-06-04.md`,
  `../.omc/audits/_shared/BRANCH_RECONCILIATION_2026-06-04.md`
