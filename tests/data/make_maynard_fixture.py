"""Tiny synthetic spatialLIBD-format DLPFC fixture builder (network-free).

Generates a few-hundred-spot AnnData that mimics the Maynard-2021 spatialLIBD
DLPFC layout closely enough to exercise the WHOLE labeled-data chain
(loader -> runner -> ARI/NMI/AMI) without the real ~1.3 GB download:

* ``obsm['spatial']`` spot coordinates laid out as horizontal layer bands so
  the manual layer call is genuinely spatially coherent (a graph-aware domain
  model should recover it, giving a non-trivial ARI).
* a spatialLIBD-style ``obs['layer_guess_reordered_short']`` per-spot layer
  call (``L1``..``WM``) -- the upstream column the loader maps to the canonical
  ``ground_truth_domain``.
* a continuous expression matrix with per-layer mean offsets so domains are
  separable; values are non-count-like so the runner treats them as already
  normalized.

Pure numpy + anndata; deterministic given ``seed``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

#: spatialLIBD short layer labels (six cortical layers + white matter).
_LAYERS: tuple[str, ...] = ("L1", "L2", "L3", "L4", "L5", "L6", "WM")


def build_fixture(
    path: str | Path,
    *,
    n_layers: int = 5,
    rows_per_layer: int = 8,
    cols: int = 8,
    n_genes: int = 40,
    seed: int = 0,
    with_labels: bool = True,
    layer_obs_key: str = "layer_guess_reordered_short",
) -> Path:
    """Write a tiny spatialLIBD-format ``.h5ad`` fixture to ``path``.

    The slide is a ``(n_layers * rows_per_layer) x cols`` grid; each horizontal
    band of ``rows_per_layer`` rows is one cortical layer. Returns ``path``.

    When ``with_labels`` is False the layer obs column is omitted entirely, so
    the runner's GT auto-detect finds nothing and degrades gracefully
    (``ari_vs_gt_available`` stays 0.0) -- the unlabeled control.
    """
    from anndata import AnnData

    path = Path(path)
    rng = np.random.default_rng(seed)
    n_layers = int(min(max(n_layers, 2), len(_LAYERS)))

    n_rows = n_layers * rows_per_layer
    xx, yy = np.meshgrid(np.arange(cols), np.arange(n_rows))
    coords = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32)
    n_spots = coords.shape[0]

    # Layer index per spot from its row band (spatially contiguous bands).
    layer_idx = (coords[:, 1].astype(np.int64) // rows_per_layer)
    layer_idx = np.clip(layer_idx, 0, n_layers - 1)

    # Continuous expression: shared noise + a per-layer mean offset on a distinct
    # gene block so the layers are linearly separable (non-count-like values).
    X = rng.normal(0.0, 0.25, size=(n_spots, n_genes)).astype(np.float32)
    block = max(n_genes // n_layers, 1)
    for li in range(n_layers):
        lo, hi = li * block, min((li + 1) * block, n_genes)
        X[layer_idx == li, lo:hi] += 3.0

    adata = AnnData(X=X)
    adata.obsm["spatial"] = coords
    adata.obs_names = [f"spot_{i:04d}" for i in range(n_spots)]
    adata.var_names = [f"gene_{j:03d}" for j in range(n_genes)]
    if with_labels:
        adata.obs[layer_obs_key] = np.asarray(
            [_LAYERS[i] for i in layer_idx], dtype=object
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(path)
    return path
