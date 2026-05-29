# FactorGraph-ST synthetic shared/private factor benchmark

Status: spec document. Defines how evidence will be collected. No performance is reported here.

## Generator

Synthetic instances have `K_shared` factors active across all sections and `K_private` factors active in a subset of sections.

### Generator parameters

| Param | Default | Range | Notes |
|---|---|---|---|
| `n_sections` | 4 | 2-16 | Sections per instance. |
| `n_spots_per_section` | 2000 | 500-10000 | Spots per section. |
| `n_genes` | 500 | 200-2000 | Synthetic gene panel size. |
| `K_shared` | 4 | 1-20 | Factors shared across all sections. |
| `K_private` | 2 | 0-10 | Section-private factors. |
| `noise_sigma` | 0.5 | 0-2 | Gaussian noise on expression. |
| `n_domains` | 5 | 1-20 | Requested number of deterministic spatial domain labels; produces up to this count when enough spots are available, independent of `K_shared`. |
| `k_nn` | 8 | 0-64 | Directed in-section nearest-neighbor edges per spot; clamped to `section_size - 1`, singleton sections emit no edges. |
| `seed` | 0 | any int | Deterministic regeneration. |

### Saved artifacts (per instance)

| Path | Contents |
|---|---|
| `X.h5` | Spot-by-gene expression. |
| `coords.npy` | Per-spot 2D coordinates. |
| `section_id.npy` | Per-spot section index. |
| `edges.npy` | k-NN edges (COO). |
| `truth/W.npy` | Ground-truth gene loadings. |
| `truth/Z_shared.npy` | Ground-truth shared activations. |
| `truth/Z_private.npy` | Ground-truth private activations. |
| `truth/domain_id.npy` | Ground-truth spatial domains. |

## Metrics

| Metric | What it locks | Pass gate |
|---|---|---|
| Matched-factor correlation (Hungarian) | Factor recovery vs truth. | set when first run lands |
| Section-overlap of `Z_shared` vs `Z_private` | Shared / private separation. | set when first run lands |
| Moran's I over `domain_id` | Spatial coherence of recovered domains. | set when first run lands |
| ARI(`domain_id`, truth) | Domain recovery. | set when first run lands |
| Reconstruction error `‖X − Z·Wᵀ‖_F / ‖X‖_F` (+ held-out variant) | Decoder fidelity; computable on real data with no truth. | set when first run lands |
| Factor redundancy (mean abs off-diagonal corr of `Z`) | Disentanglement of recovered factors (lower = less redundant). | set when first run lands |

## Planned test placeholders

- `tests/synth/test_generator_determinism.py`
- `tests/synth/test_factor_recovery.py`
- `tests/synth/test_shared_private_separation.py`
- `tests/synth/test_domain_metrics.py`

## Out of scope

- Real-data execution.
- Comparison runs against INSPIRE or HarveST (deferred; see `BASELINE_REFERENCES.md`).
- Performance claims.
