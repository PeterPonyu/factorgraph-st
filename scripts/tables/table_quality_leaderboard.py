#!/usr/bin/env python
"""#335 — multi-metric domain-quality leaderboard (one dataset x many metrics).

For a SINGLE dataset, ranks each method-variant across many quality metrics
(both GT-free and, when available, GT-based). Rows = variants ordered by a
chosen ``primary`` metric (descending, so the best variant sits on top);
columns = ``rank`` + ``variant`` + the sorted union of all metric names.

Data-independent: consumes an in-memory ``scores`` mapping ``variant -> {metric:
value}``. Missing metrics for a variant render as ``n/a`` (never a fabricated
0). An empty input yields a schema-only pending table.

Usage::

    python scripts/tables/table_quality_leaderboard.py --example --out-dir /tmp/tables
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
for _p in (str(_REPO / "src"), str(_REPO / "scripts" / "tables")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from table_emit import Table, finite_float, pending_table, sorted_metric_names, write_table  # noqa: E402

_BASENAME = "quality_leaderboard"
_FIXED_HEADERS = ["rank", "variant"]


def build_quality_leaderboard_table(
    scores: Mapping[str, Mapping[str, float]],
    *,
    primary: str | None = None,
) -> Table:
    """Rank variants by ``primary`` (descending) across all observed metrics.

    ``primary`` defaults to the first metric name in sorted order. Variants whose
    ``primary`` value is not evaluable (``nan`` / missing) sort to the bottom but
    are still listed. Ties and unrankable variants keep a stable, name-sorted
    order so the table is deterministic.
    """
    if not scores:
        return pending_table(
            "domain-quality leaderboard (#335)",
            [*_FIXED_HEADERS, "<metrics>"],
            note="no per-variant quality metrics computed yet; fill once a real dataset is scored",
        )
    metric_names = sorted_metric_names(scores.values())
    if not metric_names:
        raise ValueError("scores contain no metrics; every variant maps to an empty dict")
    key = primary if primary is not None else metric_names[0]
    if key not in metric_names:
        raise ValueError(f"primary metric {key!r} not among observed metrics {metric_names}")

    def _sort_key(variant: str) -> tuple[int, float, str]:
        value = finite_float(scores[variant].get(key))
        # Unrankable (None) -> bucket 1 (after rankable); negate value so higher
        # ranks first; name as final stable tiebreaker.
        if value is None:
            return (1, 0.0, variant)
        return (0, -value, variant)

    ordered = sorted(scores, key=_sort_key)
    headers = [*_FIXED_HEADERS, *metric_names]
    rows: list[list[object]] = []
    for rank, variant in enumerate(ordered, start=1):
        cells = [finite_float(scores[variant].get(m)) for m in metric_names]
        rows.append([rank, variant, *cells])
    return Table(name="domain-quality leaderboard (#335)", headers=headers, rows=rows)


def _example_scores() -> dict[str, dict[str, float]]:
    """Illustrative single-dataset quality scores across GT-free + GT-based metrics."""
    return {
        "gnmf": {"ari": 0.58, "silhouette": 0.31, "coherence": 0.72, "calinski_harabasz": 184.0},
        "projection": {"ari": 0.24, "silhouette": 0.12, "coherence": 0.55, "calinski_harabasz": 96.0},
        "spatial_smooth": {"ari": 0.41, "silhouette": 0.22, "coherence": 0.80, "calinski_harabasz": 131.0},
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=None, help="JSON variant->{metric: value}.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for the emitted table files.")
    parser.add_argument("--primary", type=str, default=None, help="Metric to rank by (default: first sorted).")
    parser.add_argument("--example", action="store_true", help="Build from built-in illustrative scores.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.example:
        scores: Mapping[str, Mapping[str, float]] = _example_scores()
    elif args.scores is not None:
        with open(args.scores, encoding="utf-8") as handle:
            scores = json.load(handle)
    else:
        scores = {}
    table = build_quality_leaderboard_table(scores, primary=args.primary)
    paths = write_table(table, args.out_dir, _BASENAME)
    for fmt, path in sorted(paths.items()):
        print(f"wrote {fmt}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
