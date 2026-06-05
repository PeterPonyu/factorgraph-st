#!/usr/bin/env python
"""GT-skeptical two-axis domain scorecard for FactorGraph-ST (#330).

Plots each method as a single point on two deliberately-orthogonal axes:

  * **x — label-free quality**: a composite of GT-free signals you *can* compute
    without ground-truth domains, namely spatial ``coherence`` (e.g. Moran's I /
    label-invariant cluster coherence) and run-to-run ``stability`` (e.g. mean
    pairwise ARI across seeds). High x means "looks self-consistent and smooth".
  * **y — honest concordance**: the supervised agreement with ground-truth
    domains (``ari`` / ``concordance``), the only number that says the domains
    are *correct*.

The scorecard is "GT-skeptical" because it visually separates *looking good*
(x) from *being right* (y): a method can sit far right (smooth, stable) yet low
(poor ARI) — the upper-left "smooth but wrong" trap. Quadrant guides at the
medians make that trap explicit instead of letting a single headline coherence
number stand in for correctness.

Data helpers are matplotlib-free; matplotlib is imported lazily in
:func:`render_scorecard`.

Usage (under the project conda env)::

    conda run --no-capture-output -n dl python \
        scripts/visualize/fig_gt_skeptical_scorecard.py --example \
        --out /tmp/factorgraph_gt_skeptical_scorecard.png
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np


def _coerce(value: object, name: str, method: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"method {method!r}: {name} must be a real number, got {value!r}")
    return float(value)


def scorecard_points(
    method_scores: Mapping[str, Mapping[str, float]],
) -> list[tuple[str, float, float, float, float]]:
    """Flatten per-method scores to ``(name, x, y, coherence, stability)`` tuples.

    Each method maps to ``coherence`` + ``stability`` (the label-free axis) and
    ``ari`` (the honest concordance axis); ``x`` is the mean of coherence and
    stability. Raises if a required key is missing or non-numeric so a silently
    dropped metric never reads as a real zero.
    """
    if not method_scores:
        raise ValueError("method_scores must be a non-empty mapping")
    points: list[tuple[str, float, float, float, float]] = []
    for method, scores in method_scores.items():
        coherence = _coerce(scores.get("coherence"), "coherence", method)
        stability = _coerce(scores.get("stability"), "stability", method)
        ari = _coerce(scores.get("ari"), "ari", method)
        points.append((method, 0.5 * (coherence + stability), ari, coherence, stability))
    return points


def render_scorecard(
    method_scores: Mapping[str, Mapping[str, float]],
    out_path: str | Path,
    *,
    title: str = "GT-skeptical domain scorecard",
    dpi: int = 150,
):
    """Render the two-axis scorecard scatter; save to ``out_path``.

    matplotlib is imported lazily. Returns the
    :class:`matplotlib.figure.Figure` (one axes, one point per method).
    """
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")  # headless, deterministic raster backend
    import matplotlib.pyplot as plt  # noqa: PLC0415

    points = scorecard_points(method_scores)
    palette = ["#56B4E9", "#E69F00", "#009E73", "#CC79A7", "#0072B2", "#D55E00"]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    xs = [p[1] for p in points]
    ys = [p[2] for p in points]
    stabilities = [p[4] for p in points]
    # Stability also drives marker size so all three label-free signals are read.
    smin, smax = min(stabilities), max(stabilities)
    span = smax - smin
    sizes = [120.0 + 380.0 * ((s - smin) / span if span > 0 else 0.5) for s in stabilities]

    for i, point in enumerate(points):
        name, x, y = point[0], point[1], point[2]
        ax.scatter(x, y, s=sizes[i], color=palette[i % len(palette)], edgecolor="black", linewidth=0.8, zorder=3)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(8, 6), fontsize=8)

    # Median quadrant guides separate "looks good" from "is right".
    if len(points) >= 2:
        ax.axvline(float(np.median(xs)), color="0.6", linestyle="--", linewidth=0.8, zorder=1)
        ax.axhline(float(np.median(ys)), color="0.6", linestyle="--", linewidth=0.8, zorder=1)
        ax.text(
            0.02, 0.98, "smooth/stable but wrong", transform=ax.transAxes,
            fontsize=7, color="0.4", va="top", ha="left", style="italic",
        )
        ax.text(
            0.98, 0.98, "looks good AND correct", transform=ax.transAxes,
            fontsize=7, color="0.4", va="top", ha="right", style="italic",
        )

    ax.set_xlabel("label-free quality  (mean of coherence & stability)")
    ax.set_ylabel("honest concordance  (ARI vs GT domains)")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(0.15)

    fig.tight_layout()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return fig


def _example_scores() -> dict[str, dict[str, float]]:
    """Illustrative per-method scores spanning the trap and the win.

    ``projection`` is smooth-by-construction (coords clustered in) yet low ARI —
    the upper-left "smooth but wrong" trap; ``gnmf`` earns both axes; the
    ``oversmoothed`` decoy is maximally smooth/stable but near-zero ARI.
    """
    return {
        "projection": {"coherence": 0.62, "stability": 0.55, "ari": 0.21},
        "spatial_smooth": {"coherence": 0.58, "stability": 0.70, "ari": 0.39},
        "gnmf": {"coherence": 0.71, "stability": 0.82, "ari": 0.56},
        "oversmoothed": {"coherence": 0.90, "stability": 0.95, "ari": 0.05},
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=None, help="Path to a JSON method -> {coherence,stability,ari}.")
    parser.add_argument("--out", type=Path, required=True, help="Output PNG path.")
    parser.add_argument("--title", type=str, default="GT-skeptical domain scorecard")
    parser.add_argument("--example", action="store_true", help="Render from built-in illustrative scores.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.example:
        method_scores: Mapping[str, Mapping[str, float]] = _example_scores()
    elif args.scores is not None:
        import json  # noqa: PLC0415

        with open(args.scores, encoding="utf-8") as handle:
            method_scores = json.load(handle)
    else:
        raise SystemExit("provide --scores PATH.json (or --example to render the illustrative scorecard).")
    render_scorecard(method_scores, args.out, title=args.title)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
