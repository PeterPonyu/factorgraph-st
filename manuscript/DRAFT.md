# FactorGraph-ST manuscript draft

> **v10-hybrid (updated 2026-06-28).** The GNMF-era "all claims planned" framing is
> superseded. GNMF was dominated by trivial baselines on labeled data (L1 internal
> benchmark, 2026-06-06); it is kept as history below. The v10-hybrid result is the
> current defensible claim. The "sweeps ASW + contiguity 8/8" claim is **retracted**
> — see the ASW retraction note.

## Working title

FactorGraph-ST: spatial domain detection via Banksy-augmented variational masked graph
autoencoder — reference-free, spatially-coherent tissue domains, judged honestly

## Draft abstract (v10-hybrid)

Spatial domain detection partitions a tissue into coherent regions from spatial
transcriptomics. We present FactorGraph-ST (v10-hybrid), a variational masked graph
autoencoder whose encoder runs a feature-MLP branch in parallel with a shallow
graph-attention branch, takes a Banksy-style neighbour-augmented input, is trained with
GraphMAE node masking, and reads out domains via a DEC-shaped latent + GMM + spatial
refinement. Across six platforms (10x Visium, STARmap, seqFISH, Slide-seqV2, Open-ST),
under fair common-footing re-scoring (FAIR_RESCORE — every method's predicted labels
scored identically), FactorGraph-ST uniquely sweeps spatial contiguity on all seven
datasets (0.91–0.96 vs 0.50–0.87 for published methods) at competitive ground-truth
agreement: wins STARmap (ARI 0.522 vs SEDR 0.513), beats SEDR on BRCA1 (0.567 vs 0.515),
ties on DLPFC (0.472 vs 0.492); SEDR leads on seqFISH and Open-ST (reported, not hidden).
Embedding tightness (ASW) is a representation property only: own-latent ASW is uncorrelated
with ARI (r≈0.03) and on a common expression space FactorGraph-ST holds no ASW advantage
(0.038 vs SEDR 0.050). The prior "sweeps ASW + contiguity 8/8" framing is **retracted** —
the contiguity sweep stands on 7/7 datasets; the ASW claim does not. A DLPFC case study
shows reference-free recovery of all seven cortical laminar layers.

## v10-hybrid implementation status

The implemented method is **v10-hybrid**: a Banksy-augmented variational masked graph
autoencoder with a DEC/GMM readout. Key components:

## Results — GNMF path (historical, honest-negative)

The original GNMF path is an honest-negative result, retained as historical context (it is
**superseded** by the v10-hybrid method below, not the primary claim). In the reproduced
cross-dataset spatial-domain accuracy sweep, trained GNMF is dominated by trivial coordinate
and spatial-smoothing controls on both labeled datasets:

| dataset | coords ARI | GNMF ARI | spatial_smooth ARI |
|---|---:|---:|---:|
| DLPFC / Maynard 2021 Visium | 0.2211 | 0.1556 | 0.2517 |
| STARmap mouse visual cortex / Wang 2018 | 0.2619 | 0.1760 | 0.5798 |

Source: `results/cross_dataset/tables/domain_accuracy.md` (#392). We make **no claim** that the
GNMF model beats simple spatial or coordinate baselines.

## Proposed core contributions (v10-hybrid)
- **Banksy augmentation**: concatenate X with Gaussian-weighted spatial-neighbour mean
  (λ=0.4) to inject spatial context as input rather than through heavy message passing.
- **Hybrid variational encoder**: feature-MLP branch ∥ GAT branch → concatenated latents →
  reparameterised z (KL w=1e-3).
- **GraphMAE masking**: node-feature masked reconstruction under scaled-cosine error
  (self-supervised, no reference labels required).
- **Readout**: DEC cluster-shaping → GMM (k=#domains) → iterative spatial-majority
  refinement → final domain assignments.

## Core contributions (v10-hybrid; grounded in FAIR_RESCORE)

1. **Spatial contiguity dominance**: 7/7 datasets (0.91–0.96) at competitive ARI — the
   reference-free geometric property a domain method should optimise. Wins STARmap ARI,
   beats SEDR on BRCA1; loses seqFISH and Open-ST (dispersed-cell-type / dense-tumour
   regimes where label-ARI rewards less spatial smoothing).
2. **Specificity control**: synthetic planted-GT confirms the spatial mechanism is not a
   universal smoothing trick (ARI 1.00 on coherent domains / 0.02 on scrambled non-spatial
   domains).
3. **Honest ASW retraction**: own-latent ASW dropped as a domain-quality claim; on a common
   space FactorGraph-ST holds no advantage; ASW retained only as a representation descriptor
   (see `docs/papers/factorgraph/ASW_METHODOLOGY_NOTE.md`).
4. **Case study**: reference-free recovery of all seven DLPFC cortical laminar layers
   (L1–L6, WM) as contiguous superficial-to-deep bands.

## GNMF history (negative — kept for completeness)

The prior default path (`scripts/run_real_factorgraph.py --model gnmf`) implements
graph-regularized NMF: nonneg spot factors H and gene loadings W under a Laplacian
smoothness penalty, domains clustered from H. **GNMF proved negative on labeled benchmark
data** (dominated by trivial baselines, L1 internal results 2026-06-06). The `--model gnmf`
path remains available for ablation/baseline context only. The `--model projection` (random
Gaussian) path is for ablation only. Neither path is credited as a scientific contribution.

## Review questions (open)

- ARI losses on seqFISH and Open-ST: how much is intrinsic dispersed-cell-type / dense-
  tumour tension vs tunable hyperparameters?
- Should future work target the shared/private factor objective (cross-section alignment) or
  consolidate single-section performance first?
- Gene-program interpretability (factor loadings W) under v10-hybrid: not yet benchmarked.
