#!/usr/bin/env python
"""Cross-dataset spatial-domain accuracy panel (#322).

The standard headline results figure for a spatial-domain method: grouped bars of
a label-agreement metric (ARI by default) with **x = dataset**, grouped by method
variant, so the reader sees accuracy *and its consistency across tissues/platforms*
in one view. Each bar is the per-seed mean; a per-seed scatter overlay shows the
spread (the issue's "Panel B" -- spread, not just a point estimate). This is the
cross-*dataset* view, complementary to ``fig_domain_comparison.py`` (which compares
methods within a single dataset) and ``fig_gt_skeptical_scorecard.py`` (the
two-axis GT-skeptic scatter).

Input: a long-form record list ``[{dataset, variant, seed, ari, nmi, ami}, ...]``,
exactly what ``collect_real_domain_scorecard.py`` emits as ``per_seed_records.json``.
Pass one or more such JSON files (one per dataset) and they are concatenated, so
the panel grows as labeled datasets land (#133, #391) without code change.

HONESTY NOTE: bars are per-seed means with the real per-seed points overlaid -- no
error bar is hidden and no single seed is cherry-picked. Metrics are only those
the upstream collector computed on each run's labeled subset; nothing is imputed.

matplotlib is imported lazily; the aggregation helper is numpy-only.

Usage::

    python scripts/visualize/fig_cross_dataset_accuracy.py \
        --records results/starmap_wang2018/_scorecard/per_seed_records.json \
        --records results/dlpfc_maynard/_scorecard/per_seed_records.json \
        --metric ari --out /tmp/cross_dataset_accuracy.png
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

_METRICS = ("ari", "nmi", "ami")


def load_records(paths: Sequence[Path]) -> list[dict[str, object]]:
    """Concatenate per-seed record lists from one or more JSON files."""
    records: list[dict[str, object]] = []
    for path in paths:
        loaded = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(loaded, list):
            raise ValueError(f"{path}: expected a JSON list of records")
        records.extend(loaded)
    return records


def aggregate_records(
    records: Sequence[dict[str, object]], metric: str
) -> dict[tuple[str, str], dict[str, object]]:
    """Aggregate to ``(dataset, variant) -> {mean, std, n, values}`` for one metric.

    ``values`` keeps the raw per-seed metric values so the renderer can overlay
    the spread. Raises on an unknown metric so a typo never silently yields an
    empty panel.
    """
    if metric not in _METRICS:
        raise ValueError(f"metric must be one of {_METRICS}; got {metric!r}")
    grouped: dict[tuple[str, str], list[float]] = {}
    for rec in records:
        key = (str(rec["dataset"]), str(rec["variant"]))
        grouped.setdefault(key, []).append(float(rec[metric]))
    out: dict[tuple[str, str], dict[str, object]] = {}
    for key, values in grouped.items():
        arr = np.asarray(values, dtype=float)
        out[key] = {"mean": float(arr.mean()), "std": float(arr.std()), "n": int(arr.size), "values": values}
    return out


def render_cross_dataset_accuracy(
    records: Sequence[dict[str, object]],
    out_path: str | Path,
    *,
    metric: str = "ari",
    title: str | None = None,
    dpi: int = 150,
):
    """Render the grouped cross-dataset accuracy panel; save to ``out_path``.

    x = dataset, bars grouped by variant, height = per-seed mean of ``metric``,
    with the per-seed values scattered over each bar. matplotlib is imported
    lazily. Returns the matplotlib Figure.
    """
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")  # headless, deterministic raster backend
    import matplotlib.pyplot as plt  # noqa: PLC0415

    agg = aggregate_records(records, metric)
    datasets = sorted({d for d, _v in agg})
    variants = sorted({v for _d, v in agg})
    palette = ["#56B4E9", "#E69F00", "#009E73", "#CC79A7", "#0072B2", "#D55E00"]

    fig, ax = plt.subplots(figsize=(1.8 * len(datasets) + 3.0, 4.8))
    n_var = max(len(variants), 1)
    width = 0.8 / n_var
    x = np.arange(len(datasets), dtype=float)

    for vi, variant in enumerate(variants):
        offset = (vi - (n_var - 1) / 2) * width
        heights = [agg.get((d, variant), {}).get("mean", np.nan) for d in datasets]
        ax.bar(
            x + offset, heights, width=width * 0.92, color=palette[vi % len(palette)],
            edgecolor="black", linewidth=0.6, label=variant, zorder=2,
        )
        # per-seed spread overlay
        for di, dataset in enumerate(datasets):
            vals = agg.get((dataset, variant), {}).get("values", [])
            if vals:
                jitter = (np.linspace(-0.25, 0.25, len(vals))) * width
                ax.scatter(
                    np.full(len(vals), x[di] + offset) + jitter, vals,
                    s=14, color="black", alpha=0.6, zorder=3, linewidths=0,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel(f"{metric.upper()} vs annotated domains")
    ax.set_title(title or f"Cross-dataset spatial-domain accuracy ({metric.upper()})")
    ax.legend(fontsize=8, frameon=False, ncol=min(n_var, 3))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(x=0.05)
    ax.axhline(0.0, color="0.7", linewidth=0.6, zorder=1)

    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return fig


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records", type=Path, action="append", required=True,
        help="per_seed_records.json (repeatable; one per dataset).",
    )
    parser.add_argument("--metric", choices=_METRICS, default="ari")
    parser.add_argument("--out", type=Path, required=True, help="Output PNG path.")
    parser.add_argument("--title", type=str, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    records = load_records(args.records)
    if not records:
        raise SystemExit("no records loaded; pass at least one non-empty --records file")
    render_cross_dataset_accuracy(records, args.out, metric=args.metric, title=args.title)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
