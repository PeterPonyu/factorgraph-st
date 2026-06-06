#!/usr/bin/env python
"""Ground-truth-vs-predicted spatial-domain map gallery (#324).

The qualitative anchor of every spatial-domain paper: a side-by-side spatial map
gallery with annotated ground-truth domains next to predicted domains, tiled
across the method rows, so spatial faithfulness is judged directly in tissue
coordinates (distinct from any boundary *metric* -- this is the montage that
makes the metric legible).

Layout: a grid whose first column is the manual GT annotation and whose remaining
columns are one predicted partition per method; every cell is a spatial scatter
(``obsm['spatial']``) colored by domain label, with a shared, **label-matched**
color legend. Predicted integer labels carry no intrinsic correspondence to GT
classes, so each prediction is recolored by majority overlap with the GT
partition (:func:`align_pred_to_gt`) -- the same tissue region keeps the same
color across panels, instead of an arbitrary palette permutation that would make
a good partition look wrong.

HONESTY NOTE: alignment only permutes colors for legibility; it never changes the
partition or any metric. Spots whose GT label is NA/blank are dropped from every
panel (no fabricated labels), exactly as the supervised metrics do.

matplotlib is imported lazily; the alignment helper is numpy-only.

Usage (under the project conda env)::

    conda run --no-capture-output -n dl python \
        scripts/visualize/fig_gt_vs_pred_map_gallery.py \
        --h5ad ../data/processed/starmap_mouse_vcortex_wang2018/anndata.h5ad \
        --gt-obs-key ground_truth \
        --pred gnmf=results/starmap_wang2018/gnmf_seed0/factorgraph-st/outputs/factors.npz \
        --pred coords=results/starmap_wang2018/coords_seed0/factorgraph-st/outputs/factors.npz \
        --pred spatial_smooth=results/starmap_wang2018/smooth_seed0/factorgraph-st/outputs/factors.npz \
        --out /tmp/factorgraph_gt_vs_pred_gallery.png
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

#: NA/blank GT tokens, mirror of ``run_real_factorgraph._GT_NA_TOKENS``.
_GT_NA_TOKENS = frozenset({"", "na", "nan", "none", "unknown", "unlabeled", "unlabelled"})


def align_pred_to_gt(gt_codes: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Relabel ``pred`` so each predicted cluster takes the GT code it most overlaps.

    For each predicted label, the assigned color-code is the GT code that the
    largest share of its spots carry (majority vote via the contingency table).
    This is a pure *color* alignment for the gallery -- it is a many-to-one map
    (two predicted clusters may map to the same GT color) and is never used to
    compute a metric. Returns an int array the same shape as ``pred``.
    """
    gt_codes = np.asarray(gt_codes).astype(np.int64)
    pred = np.asarray(pred).astype(np.int64)
    if pred.size == 0:
        return pred.copy()
    pred_labels = np.unique(pred)
    remap: dict[int, int] = {}
    for p in pred_labels:
        mask = pred == p
        gt_here = gt_codes[mask]
        if gt_here.size == 0:
            remap[int(p)] = int(p)
            continue
        values, counts = np.unique(gt_here, return_counts=True)
        remap[int(p)] = int(values[int(np.argmax(counts))])
    return np.array([remap[int(p)] for p in pred], dtype=np.int64)


def _valid_mask(gt_raw: Sequence[object]) -> np.ndarray:
    raw = np.asarray([str(v) for v in gt_raw])
    return np.array([s.strip().lower() not in _GT_NA_TOKENS for s in raw], dtype=bool)


def render_gallery(
    coords: np.ndarray,
    gt_codes: np.ndarray,
    predictions: Sequence[tuple[str, np.ndarray]],
    out_path: str | Path,
    *,
    title: str = "GT vs predicted spatial domains",
    point_size: float = 6.0,
    dpi: int = 150,
):
    """Render the GT-vs-predicted spatial map gallery; save to ``out_path``.

    ``coords`` is ``(n, 2)``; ``gt_codes`` is the integer GT partition; each
    ``(name, labels)`` in ``predictions`` is recolored to the GT palette via
    :func:`align_pred_to_gt`. Returns the matplotlib Figure (one row, one column
    per {GT, *predictions}). matplotlib is imported lazily.
    """
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")  # headless, deterministic raster backend
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib.lines import Line2D  # noqa: PLC0415

    coords = np.asarray(coords, dtype=float)
    gt_codes = np.asarray(gt_codes).astype(np.int64)
    n_classes = int(gt_codes.max()) + 1 if gt_codes.size else 0
    cmap = plt.get_cmap("tab10" if n_classes <= 10 else "tab20")
    color_for = lambda c: cmap(c % cmap.N)  # noqa: E731

    panels: list[tuple[str, np.ndarray]] = [("ground truth", gt_codes)]
    panels += [(name, align_pred_to_gt(gt_codes, labels)) for name, labels in predictions]

    ncols = len(panels)
    fig, axes = plt.subplots(1, ncols, figsize=(3.4 * ncols, 3.6), squeeze=False)
    for ax, (name, labels) in zip(axes[0], panels, strict=True):
        ax.scatter(
            coords[:, 0], coords[:, 1], c=[color_for(c) for c in labels],
            s=point_size, edgecolor="none",
        )
        ax.set_title(name, fontsize=9)
        ax.set_aspect("equal")
        ax.invert_yaxis()  # image/tissue convention: origin top-left
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=color_for(c),
               markeredgecolor="none", markersize=6, label=f"domain {c}")
        for c in range(n_classes)
    ]
    if handles:
        fig.legend(
            handles=handles, loc="center right", fontsize=7, frameon=False,
            title="GT-matched\ndomain", title_fontsize=7,
        )
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 0.9, 0.96))

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return fig


def _load_domain_id(path: Path) -> np.ndarray:
    """Load a predicted ``domain_id`` from a ``.npz`` file or a runner output dir.

    A directory is scanned for the first ``*.npz`` carrying ``domain_id`` so the
    per-model bundle name (``factors.npz`` / ``baseline_*.npz``) need not be known.
    """
    candidates = sorted(path.glob("*.npz")) if path.is_dir() else [path]
    for npz in candidates:
        with np.load(npz) as data:
            if "domain_id" in data:
                return np.asarray(data["domain_id"]).astype(np.int64)
    raise SystemExit(f"{path}: no .npz with a 'domain_id' array")


def _parse_pred(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise SystemExit(f"--pred expects NAME=PATH, got {spec!r}")
    name, path = spec.split("=", 1)
    return name, Path(path)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad", type=Path, required=True, help="Source AnnData (coords + GT).")
    parser.add_argument("--gt-obs-key", type=str, required=True, help="obs column with per-spot GT domains.")
    parser.add_argument(
        "--pred", action="append", default=[], metavar="NAME=PATH",
        help="A predicted-domains factors.npz, named: gnmf=path/to/factors.npz (repeatable).",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output PNG path.")
    parser.add_argument("--title", type=str, default="GT vs predicted spatial domains")
    parser.add_argument("--point-size", type=float, default=6.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.pred:
        raise SystemExit("provide at least one --pred NAME=PATH")
    import anndata as ad  # noqa: PLC0415

    adata = ad.read_h5ad(args.h5ad)
    if args.gt_obs_key not in adata.obs:
        raise SystemExit(f"--gt-obs-key {args.gt_obs_key!r} not in obs {list(adata.obs.columns)}")
    if "spatial" not in adata.obsm:
        raise SystemExit("AnnData has no obsm['spatial'] coordinates")

    valid = _valid_mask(adata.obs[args.gt_obs_key].to_numpy())
    coords = np.asarray(adata.obsm["spatial"], dtype=float)[valid]
    _, gt_codes = np.unique(
        np.asarray([str(v) for v in adata.obs[args.gt_obs_key].to_numpy()])[valid],
        return_inverse=True,
    )
    gt_codes = gt_codes.astype(np.int64)

    predictions: list[tuple[str, np.ndarray]] = []
    for spec in args.pred:
        name, path = _parse_pred(spec)
        labels = _load_domain_id(path)
        if labels.shape[0] != valid.shape[0]:
            raise SystemExit(
                f"{name}: domain_id length {labels.shape[0]} != n_obs {valid.shape[0]}"
            )
        predictions.append((name, labels[valid]))

    render_gallery(coords, gt_codes, predictions, args.out, title=args.title, point_size=args.point_size)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
