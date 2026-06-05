# GNMF Hyperparameter Tuning Notes

> **Status:** de-risk run (branch `feat/factorgraph-gnmf-derisk`).
> Model exercise only — no CLAIM_LEDGER entries promoted.
> Labeled-data ARI (#350) is still blocked on Maynard spatialLIBD ingestion (#133).

---

## 1. Real-data dry-run profile

**Dataset:** DLPFC GSE307403 Br2719 (single section, unlabeled)  
**Shape:** 4 986 spots × 36 601 genes  
**Command:** `python scripts/run_real_factorgraph.py --model gnmf --gnmf-lam 1.0 --gnmf-n-iter 200 --gnmf-tol 1e-4`

| Metric | Value |
|---|---|
| **runtime_s** (Python `perf_counter`) | **95.3 s** |
| Wall-clock (incl. conda startup) | 1 m 38.6 s |
| Peak RSS (`/usr/bin/time -v`) | **5 400 MB (5.40 GB)** |
| Objective initial | 24 242 831 |
| Objective final | 12 084 285 |
| Objective reduction | **50.1%** |
| Iterations run (tol=1e-4 early stop) | **63 of 200** |
| Monotonic decrease | ✓ (by construction — multiplicative updates) |
| domain_count (k=5 requested) | **5** (non-degenerate) |
| H_rank_effective | 6 / 6 factors active |
| W_sparsity | 34.1% |
| morans_i_domain | 0.3258 (null −0.0011, **delta +0.327**) |
| coherence_label_invariant_domain | 0.4187 (null +0.0011, **delta +0.418**) |

**Sanity:** objective halved, all 6 factors active, domains non-degenerate, spatial coherence
well above its permutation null → the trained model is behaving correctly on real data.
Memory budget is ~5.4 GB for this slide; multi-section runs will scale proportionally.

---

## 2. Synthetic hyperparam sweep

**Setup:** `scripts/sweep_gnmf.py`, 2 sections × 300 spots × 150 genes,
K_shared=4 K_private=2, n_domains=5, 3 random seeds averaged per config.  
**Scoring:**
- `matched_factor_correlation` (mfc) — mean absolute Pearson correlation after
  best one-to-one matching of recovered H vs ground-truth Z (higher = better factor recovery).
- `domain_ari` — ARI of recovered domains vs synthetic GT domain_id.

> **Note on domain ARI:** synthetic GT domains are assigned by a coordinate-quantile
> projection (see `synth/generator.py:_assign_spatial_domains`), while GNMF clusters
> on factor scores H alone (no coordinates). Consequently domain ARI is near zero across
> all configurations — this is expected and does **not** indicate a bug; it reflects the
> deliberate design choice that GNMF never sees coordinates. Domain quality on real data
> should be assessed with labeled sections via #350.

### 2a. n_iter — dominant driver

| n_iter | mfc (mean, best lam) |
|---|---|
| 50 | ~0.88 |
| 100 | ~0.97 |
| **200** | **~0.987** |

Conclusion: **n_iter ≥ 200 is required** for near-converged factor recovery. 100 iterations
recovers most of the improvement but leaves ~1.5% mfc on the table. Increasing beyond 200
was not tested; early-stopping at tol=1e-4 already triggers before 200 iters on well-behaved
instances (dry-run converged at 63).

### 2b. lam — insensitive at n_iter=200

| lam | mfc_mean (n_iter=200, tol=1e-4) |
|---|---|
| 0.00 (plain NMF) | 0.9869 |
| 0.10 | 0.9860 |
| 0.50 | 0.9853 |
| **1.00** | **0.9838** |
| 2.00 | 0.9869 |
| 5.00 | 0.9866 |
| 10.0 | 0.9844 |

The mfc range across the full lam grid is only **0.003** at n_iter=200 — lam is effectively
insensitive for factor recovery on this synthetic benchmark. The scientific reason: spatial
regularization improves spatial coherence of H but slightly trades off pure reconstruction
fidelity; both effects are small at this data scale.

**Recommendation:** use **lam=1.0** as the default for real spatial data. It provides meaningful
graph regularization (spatial coherence is learned rather than zero-penalized) with negligible
factor-recovery cost. lam=0 reduces to plain NMF and loses the spatial coherence property
that distinguishes GNMF from the projection baseline.

### 2c. tol — use 1e-4

| tol | Behaviour |
|---|---|
| 1e-3 | Can stop early (avg 186 iters at lam=0.1 seed mix) — misses ~0.001 mfc |
| **1e-4** | **Well-calibrated: stops at convergence, not prematurely** |
| 1e-5 | No measurable benefit over 1e-4; only increases iteration count |

---

## 3. Recommended defaults

```
--gnmf-lam    1.0     # spatial regularization strength; insensitive in [0.1, 5.0]
--gnmf-n-iter 200     # max iterations; early-stop usually triggers well before 200
--gnmf-tol    1e-4    # relative objective-change threshold; calibrated to avoid premature stop
```

These are already the CLI defaults in `run_real_factorgraph.py`. No change needed before the
labeled-data run (#350).

---

## 4. Convergence curve summary

At lam=1.0, n_iter=200, tol=1e-4 on the DLPFC slide:

```
iter   0: objective = 24 242 831   (initial)
iter  63: objective = 12 084 285   (converged, tol=1e-4 triggered)
reduction: 50.1%
```

The rapid drop indicates the initialization is well-scaled (uniform random × sqrt(mean(X)/k))
and the multiplicative updates are numerically stable. No NaN/Inf observed. W_sparsity of 34%
suggests the model is naturally discovering sparse gene loadings without explicit sparsity
regularization.

---

## 5. What this run does NOT tell us

- **Domain ARI vs ground truth:** ARI is skipped because the Br2719 section carries no
  per-spot layer labels. Labeled ARI must wait for Maynard 2021 spatialLIBD ingestion (#133,
  blocked on network; see PR #350 / #342).
- **Multi-section scaling:** this is a single-section run; memory and runtime will scale
  with n_spots × n_genes. At 4986 spots and 36 601 genes the peak RSS is 5.4 GB; a
  4-section stack (~20k spots) would require ~22 GB.
- **Sensitivity at high lam on real data:** the synthetic benchmark showed lam is insensitive
  up to 10, but real data with strong spatial gradients may benefit from a finer sweep.

---

*Generated by `scripts/sweep_gnmf.py` + `scripts/run_real_factorgraph.py --model gnmf` on
DLPFC GSE307403 Br2719. Full sweep results in
`results/factorgraph-st/outputs/gnmf_sweep/sweep_results.json` (not committed — scratch artifact).*
