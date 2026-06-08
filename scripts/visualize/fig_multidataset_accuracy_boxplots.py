#!/usr/bin/env python
"""Multi-dataset spatial-domain accuracy boxplots + rank aggregation (#307).

A pooled, distribution-first companion to ``fig_cross_dataset_accuracy.py``.
Where that figure keeps **x = dataset** (so consistency across tissues is visible),
this one *pools* every (dataset x seed) point per variant into a boxplot -- one
subplot per metric (ARI, NMI, AMI) -- so the reader sees each variant's full
spread and median across the whole benchmark at a glance (the issue's "Panel A").
A second figure (the issue's "Panel C") is a **rank aggregation**: within every
(dataset, metric) cell variants are ranked by their mean-over-seeds score
(rank 1 = best), and each variant's ranks are averaged across all cells into a
single mean rank (lower = better) -- a single robust ordering that does not let
one easy dataset or one high-variance metric dominate the headline.

Input: a long-form record list ``[{dataset, variant, seed, ari, nmi, ami}, ...]``,
exactly what ``collect_real_domain_scorecard.py`` emits as ``per_seed_records.json``.
Pass one or more such JSON files (one per dataset) and they are concatenated, so
the panels grow as labeled datasets land (#133, #391) without code change.

HONESTY NOTE: boxes are matplotlib defaults (median, IQR box, 1.5xIQR whiskers)
with every real (dataset x seed) point overlaid in black -- no point is hidden and
no seed is cherry-picked. Ranks are computed only from the metrics the upstream
collector actually scored; nothing is imputed and ties get a fair average rank.

matplotlib is imported lazily; every aggregation helper is numpy/stdlib-only.

Usage::

    python scripts/visualize/fig_multidataset_accuracy_boxplots.py \
        --records results/starmap_wang2018/_scorecard/per_seed_records.json \
        --records results/dlpfc_maynard/_scorecard/per_seed_records.json \
        --out /tmp/accuracy_boxplots.png
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


def boxplot_series(
    records: Sequence[dict[str, object]], metric: str
) -> dict[str, list[float]]:
    """Group one metric's values by variant, pooled over all (dataset x seed) points.

    Returns ``{variant -> [values...]}``. Raises on an unknown metric so a typo
    never silently yields an empty panel.
    """
    if metric not in _METRICS:
        raise ValueError(f"metric must be one of {_METRICS}; got {metric!r}")
    series: dict[str, list[float]] = {}
    for rec in records:
        series.setdefault(str(rec["variant"]), []).append(float(rec[metric]))
    return series


def rank_variants_one_cell(variant_to_mean: dict[str, float]) -> dict[str, float]:
    """Rank variants within one (dataset, metric) cell; rank 1 = highest score.

    Ties share the average of the positions they span (e.g. two variants tied
    for best both get rank 1.5). Pure stdlib; no scipy.
    """
    ranks: dict[str, float] = {}
    for variant, mean in variant_to_mean.items():
        greater = sum(1 for other in variant_to_mean.values() if other > mean)
        equal = sum(1 for other in variant_to_mean.values() if other == mean)
        # positions are greater+1 .. greater+equal (1-based); average them
        ranks[variant] = greater + (equal + 1) / 2
    return ranks


def mean_rank(
    records: Sequence[dict[str, object]],
    metrics: Sequence[str] = _METRICS,
) -> dict[str, float]:
    """Average each variant's per-cell rank across all (dataset, metric) cells.

    For every (dataset, metric) cell, variants are ranked by their mean-over-seeds
    score (rank 1 = best). A variant's mean rank is the average of its ranks over
    all cells in which it appears (lower = better). Raises on an unknown metric.
    """
    for metric in metrics:
        if metric not in _METRICS:
            raise ValueError(f"metric must be one of {_METRICS}; got {metric!r}")
    datasets = sorted({str(rec["dataset"]) for rec in records})
    accumulated: dict[str, list[float]] = {}
    for dataset in datasets:
        for metric in metrics:
            sums: dict[str, list[float]] = {}
            for rec in records:
                if str(rec["dataset"]) != dataset:
                    continue
                sums.setdefault(str(rec["variant"]), []).append(float(rec[metric]))
            if not sums:
                continue
            variant_to_mean = {v: float(np.mean(vals)) for v, vals in sums.items()}
            for variant, rank in rank_variants_one_cell(variant_to_mean).items():
                accumulated.setdefault(variant, []).append(rank)
    return {v: float(np.mean(ranks)) for v, ranks in accumulated.items()}


def render_boxplots(
    records: Sequence[dict[str, object]],
    out_path: str | Path,
    *,
    metrics: Sequence[str] = _METRICS,
    title: str | None = None,
    dpi: int = 150,
):
    """Render one boxplot subplot per metric (variants pooled over dataset x seed).

    Each box is matplotlib's default (median, IQR, 1.5xIQR whiskers) with the raw
    per-point values jittered over it in black. matplotlib is imported lazily.
    Returns the matplotlib Figure.
    """
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")  # headless, deterministic raster backend
    import matplotlib.pyplot as plt  # noqa: PLC0415

    variants = sorted({str(rec["variant"]) for rec in records})
    positions = np.arange(len(variants), dtype=float)

    fig, axes = plt.subplots(1, len(metrics), figsize=(3.6 * len(metrics) + 1.0, 4.8), squeeze=False)
    for ax, metric in zip(axes[0], metrics, strict=False):
        series = boxplot_series(records, metric)
        data = [series.get(v, []) for v in variants]
        ax.boxplot(
            [d if d else [np.nan] for d in data],
            positions=positions, widths=0.6, showfliers=False,
        )
        for pos, vals in zip(positions, data, strict=False):
            if vals:
                jitter = np.linspace(-0.18, 0.18, len(vals))
                ax.scatter(
                    np.full(len(vals), pos) + jitter, vals,
                    s=14, color="black", alpha=0.6, zorder=3, linewidths=0,
                )
        ax.set_xticks(positions)
        ax.set_xticklabels(variants, rotation=15, ha="right", fontsize=8)
        ax.set_title(metric.upper())
        ax.set_ylabel(f"{metric.upper()} vs annotated domains")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.axhline(0.0, color="0.7", linewidth=0.6, zorder=1)

    fig.suptitle(title or "Multi-dataset spatial-domain accuracy (pooled over dataset x seed)")
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return fig


def render_mean_rank(
    records: Sequence[dict[str, object]],
    out_path: str | Path,
    *,
    metrics: Sequence[str] = _METRICS,
    title: str | None = None,
    dpi: int = 150,
):
    """Render the mean-rank aggregation as a horizontal bar chart (lower = better).

    matplotlib is imported lazily. Returns the matplotlib Figure.
    """
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")  # headless, deterministic raster backend
    import matplotlib.pyplot as plt  # noqa: PLC0415

    ranks = mean_rank(records, metrics)
    # ascending by mean rank => best (lowest) at the top of a horizontal bar chart
    ordered = sorted(ranks.items(), key=lambda kv: kv[1])
    variants = [v for v, _r in ordered]
    values = [r for _v, r in ordered]
    y = np.arange(len(variants), dtype=float)[::-1]  # best on top

    fig, ax = plt.subplots(figsize=(6.0, 0.6 * len(variants) + 2.0))
    ax.barh(y, values, color="#56B4E9", edgecolor="black", linewidth=0.6, zorder=2)
    for yi, val in zip(y, values, strict=False):
        ax.text(val, yi, f" {val:.2f}", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(variants, fontsize=9)
    ax.set_xlabel("mean rank across (dataset x metric) cells")
    ax.set_title(title or "Rank aggregation across datasets (lower = better)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(y=0.1)

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
    parser.add_argument("--metrics", nargs="+", choices=_METRICS, default=list(_METRICS))
    parser.add_argument("--out", type=Path, required=True, help="Output boxplots PNG path.")
    parser.add_argument("--title", type=str, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    records = load_records(args.records)
    if not records:
        raise SystemExit("no records loaded; pass at least one non-empty --records file")
    render_boxplots(records, args.out, metrics=args.metrics, title=args.title)
    rank_out = args.out.with_name(f"{args.out.stem}_meanrank{args.out.suffix}")
    render_mean_rank(records, rank_out, metrics=args.metrics)
    print(f"wrote {args.out}")
    print(f"wrote {rank_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
