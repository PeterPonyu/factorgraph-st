# FactorGraph-ST claim ledger

> **Updated 2026-06-28.** GNMF-era rows retained as history; v10-hybrid evidence adds
> `supported` rows. Do NOT promote anything to `validated` — that requires an explicit
> human decision.

---

## v10-hybrid results (2026-06-28)

| Claim | Evidence | Status |
|---|---|---|
| Spatial contiguity dominance (JOINT claim): contiguity 7/7 datasets (0.91–0.96) **at competitive ARI** (0.47–0.57) — the defensible claim is the joint, not contiguity alone. Contiguity in isolation is gameable: a coords-only expression-blind KMeans reaches contiguity ≥ FactorGraph-ST (DLPFC 0.959 vs 0.926; STARmap 0.949 vs 0.941, the single highest contiguity of any method on STARmap) but collapses on ARI (0.25–0.35). | FAIR_RESCORE multi-seed (n=3): wins STARmap ARI 0.522 (SEDR 0.513), beats SEDR on BRCA1 0.567 (SEDR 0.515), ties DLPFC 0.472 (SEDR 0.492); loses seqFISH 0.383 (SEDR 0.438) and Open-ST 0.149 (SEDR 0.483). Contiguity 0.91–0.96 on all 7 datasets vs baselines 0.50–0.90. Coords-only KMeans control: DLPFC contig 0.959/ARI 0.253, STARmap contig 0.949/ARI 0.350, BRCA1 contig 0.930/ARI 0.374. Full 8-method × 7-dataset leaderboard (ARI + contiguity) now in `manuscript/main.tex` Table 4 (tab:leaderboard), built from `FAIR_RESCORE.json`. Source: `results/benchmark/domain_detection/FAIR_RESCORE.json`; `docs/SCIENCE_GAPS_2026-06-28.md` §3. | **supported** |
| Weak-ARI rows disclosed (honesty): the two lowest-signal datasets (Slide-seqV2 hippocampus ARI 0.068 vs naive 0.063; second Open-ST section ARI 0.189 vs naive 0.172) land near the naive floor with no published comparator scored. Reported in full, not restricted to the favourable 5 datasets. | FAIR_RESCORE; disclosed in `manuscript/main.tex` Table 3 (tab:ari, all 7 rows) and Table 4 leaderboard. | **supported (disclosed)** |
| Spatial mechanism is specific: planted-GT ARI 1.00 coherent / 0.02 scrambled non-spatial | Synthetic planted-ground-truth control experiment | **supported** |
| ASW advantage ("sweeps ASW + contiguity 8/8") | **RETRACTED** — own-latent ASW is uncorrelated with ARI (r≈0.03, 21 method×dataset points). On common X_pca: FactorGraph-ST 0.038 vs SEDR 0.050 on DLPFC (no advantage). Own-latent ASW is inflated by DEC sharpening, not by domain correctness. See `docs/papers/factorgraph/ASW_METHODOLOGY_NOTE.md`. | **retracted** |

---

## GNMF-era rows (history — superseded by v10-hybrid)

All rows below trace to the old GNMF implementation (`--model gnmf`). GNMF was
dominated by trivial baselines on labeled data (L1 internal benchmark, 2026-06-06).
These rows are kept as the falsification record; they are NOT re-grindable claims.

| Claim | Required evidence | Current evidence | Missing evidence | Status |
|---|---|---|---|---|
| Graph-factor learning can recover interpretable shared and private spatial gene programs. | Synthetic shared/private factor benchmark, nonneg factor schema, factor recovery metrics. | GNMF dominated by trivial baselines on labeled data (L1 internal benchmark 2026-06-06; evidence `results/cross_dataset/tables/domain_accuracy.md`, #392). v10-hybrid spatial contiguity result replaces the GNMF claim. | Gene-program interpretability under v10-hybrid not yet benchmarked. | **negative (GNMF path; v10-hybrid supersedes)** |
| FactorGraph-ST can align heterogeneous ST sections while preserving spatial structure. | Multi-section alignment protocol, spatial coherence metrics, held-out section checks. | Not validated under GNMF. Not attempted under v10-hybrid (single-section only). | Cross-section alignment is future work. | **planned** |
| Spatial factors improve domain/gene-program interpretability over generic clustering. | Pre-registered domain metrics, SVG/pathway enrichment sanity checks, baseline comparison. | v10-hybrid contiguity sweep at competitive ARI is the in-scope defensible result. Gene-program interpretability not yet measured. | Pathway/SVG enrichment under v10-hybrid. | **planned** |
