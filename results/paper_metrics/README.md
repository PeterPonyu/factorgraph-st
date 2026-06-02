# results/paper_metrics — FactorGraph-ST Contract Stubs

This directory contains per-asset metrics stub files for **factorgraph-st**.
These are **contract stubs** — they establish the claim lifecycle scaffold so
the metrics gate (`scripts/check_metrics_gate.py`) can enforce readiness from
`planned` → `paper_claim_ready: true` before any result appears in prose.

## Current stubs

| File | Asset ID | Role |
|------|----------|------|
| `fg-f1_metrics_stub.json` | FG-F1 | Spatial-domain map (factor spatial maps figure) |
| `fg-f2_metrics_stub.json` | FG-F2 | Factor/SVG interpretability figure |
| `fg-f3_metrics_stub.json` | FG-F3 | Baseline comparison figure (placeholder) |
| `fg-t1_metrics_stub.json` | FG-T1 | Multi-method ARI/NMI comparison table |

## Schema (12 keys)

All stubs share the same 12-key schema:

```
asset_id, project, role, readiness_status, source_metric_reference,
source_script_reference, source_exists, metric_source_exists,
artifact_exists, supports_safe_prose, paper_claim_ready, notes
```

## Status

All stubs are initialised with `paper_claim_ready: false` and
`readiness_status: "planned"`. Factor/SVG metrics (FG-F1, FG-F2, FG-T1)
await the learned-model rebuild tracked in **#181**. The baseline comparison
figure (FG-F3) additionally awaits a baseline runner implementation.

A stub must not be promoted to `paper_claim_ready: true` until:
1. A claim ID is assigned.
2. A benchmark metric artifact is produced and validated.
3. A matching data card exists in `data/cards/`.
4. The rendered figure or table is confirmed reproducible.
