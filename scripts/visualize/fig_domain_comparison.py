#!/usr/bin/env python
"""Domain-quality method-comparison panel for FactorGraph-ST.

Renders a publication-style grouped bar chart comparing the supervised
domain-quality of the three model paths that emit the SAME domain-metric suite
(#191) against per-spot ground-truth domain labels:

  * ``projection``     -- the default fixed random Gaussian projection encoder
                          + NNLS loadings + ``[H | coords]`` graph-aware k-means
                          domains (the MVP; spatial coords are a clustering
                          input, so its scores are a floor / re-read baseline).
  * ``spatial_smooth`` -- the clean-room graph-smoothing comparison BASELINE
                          (#305): neighbor-average + PCA + k-means, no factors.
  * ``gnmf``           -- the OPT-IN trained graph-regularized NMF (#304):
                          spatial coherence is LEARNED via the Laplacian
                          regularizer and domains are clustered on H alone.

The bars are the supervised domain-quality metrics emitted under each run's
``metrics`` block (the keys written by ``scripts/run_real_factorgraph.py`` only
when usable GT labels are present):

    ari_domain, nmi_domain, boundary_f1_domain, weighted_dice_domain,
    silhouette_domain

This script is a pure VISUALIZATION concern: it CONSUMES the already-emitted
``metrics.json`` bundles and never recomputes a metric. matplotlib is imported
LAZILY inside :func:`render_comparison` so importing this module (e.g. for the
metric-extraction helpers, or under the numpy-only runtime) never pulls in a
plotting dependency.

Usage (under the project conda env)::

    conda run --no-capture-output -n dl python \
        scripts/visualize/fig_domain_comparison.py \
        --projection results/projection/metrics.json \
        --spatial-smooth results/spatial_smooth/metrics.json \
        --gnmf results/gnmf/metrics.json \
        --out /tmp/factorgraph_domain_comparison.png
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

# Ordered (metric-key, display-label) pairs for the comparison panel. Order is
# the x-axis order of the grouped bars. Calinski-Harabasz is intentionally
# excluded: it is unbounded and on a different scale, so it does not share the
# roughly [-1, 1] / [0, 1] axis the other five metrics live on.
DOMAIN_METRICS: tuple[tuple[str, str], ...] = (
    ("ari_domain", "ARI"),
    ("nmi_domain", "NMI"),
    ("boundary_f1_domain", "Boundary F1"),
    ("weighted_dice_domain", "Weighted Dice"),
    ("silhouette_domain", "Silhouette"),
)

# Canonical method order (worst-expected -> best-expected) and display labels.
METHOD_ORDER: tuple[str, ...] = ("projection", "spatial_smooth", "gnmf")
METHOD_LABELS: dict[str, str] = {
    "projection": "projection (MVP floor)",
    "spatial_smooth": "spatial_smooth (baseline)",
    "gnmf": "gnmf (learned)",
}


def load_metrics_bundle(path: str | Path) -> dict[str, object]:
    """Load an emitted results-contract ``metrics.json`` bundle from disk."""
    with open(path, encoding="utf-8") as handle:
        bundle = json.load(handle)
    if not isinstance(bundle, dict):
        raise ValueError(f"{path}: expected a JSON object, got {type(bundle).__name__}")
    return bundle


def extract_domain_quality(bundle: Mapping[str, object]) -> dict[str, float | None]:
    """Pull the comparison metric set out of a metrics bundle (or flat dict).

    Accepts either a full results-contract bundle (with a nested ``metrics``
    map) or an already-flat ``str -> number`` metrics dict. Returns one entry
    per :data:`DOMAIN_METRICS` key; a metric that is absent, ``null`` (the
    contract's coercion of a non-finite value), or non-numeric becomes
    ``None`` so the renderer can mark it as "not evaluable" rather than plot a
    misleading zero.
    """
    raw = bundle.get("metrics", bundle)
    if not isinstance(raw, Mapping):
        raise ValueError("bundle has no 'metrics' mapping and is not itself a mapping")
    out: dict[str, float | None] = {}
    for key, _label in DOMAIN_METRICS:
        value = raw.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            out[key] = None
        else:
            out[key] = float(value)
    return out


def render_comparison(
    method_metrics: Mapping[str, Mapping[str, float | None]],
    out_path: str | Path,
    *,
    title: str = "Domain-quality method comparison (supervised vs GT domains)",
    methods: Sequence[str] = METHOD_ORDER,
    dpi: int = 150,
) -> Path:
    """Render the grouped-bar comparison panel and save it to ``out_path``.

    ``method_metrics`` maps a method name to its domain-quality dict (the shape
    returned by :func:`extract_domain_quality`). Bars are grouped by metric;
    within each group there is one bar per method in ``methods`` order. Missing
    metrics (``None``) are drawn as a zero-height hatched stub and annotated
    ``n/a`` so they read as "not evaluable", never as a real zero score.

    matplotlib is imported here (lazily) so module import stays dependency-free.
    Returns the resolved output :class:`~pathlib.Path`.
    """
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")  # headless, deterministic raster backend
    import matplotlib.pyplot as plt  # noqa: PLC0415

    present = [m for m in methods if m in method_metrics]
    if not present:
        raise ValueError("no known methods present in method_metrics")

    metric_labels = [label for _key, label in DOMAIN_METRICS]
    n_metrics = len(DOMAIN_METRICS)
    n_methods = len(present)

    # Color-blind-friendly qualitative palette (Wong 2011), one hue per method.
    palette = ["#56B4E9", "#E69F00", "#009E73", "#CC79A7", "#0072B2"]
    colors = {m: palette[i % len(palette)] for i, m in enumerate(present)}

    group_width = 0.8
    bar_width = group_width / n_methods
    x = list(range(n_metrics))

    fig, ax = plt.subplots(figsize=(1.7 * n_metrics + 2.0, 5.0))

    for j, method in enumerate(present):
        mvals = method_metrics[method]
        offset = -group_width / 2 + bar_width * (j + 0.5)
        positions = [xi + offset for xi in x]
        heights = []
        missing = []
        for key, _label in DOMAIN_METRICS:
            v = mvals.get(key)
            missing.append(v is None)
            heights.append(0.0 if v is None else float(v))
        bars = ax.bar(
            positions,
            heights,
            width=bar_width * 0.92,
            label=METHOD_LABELS.get(method, method),
            color=colors[method],
            edgecolor="black",
            linewidth=0.6,
        )
        for rect, height, is_missing in zip(bars, heights, missing, strict=True):
            if is_missing:
                rect.set_hatch("///")
                rect.set_facecolor("white")
                rect.set_edgecolor("0.6")
                ax.annotate(
                    "n/a",
                    (rect.get_x() + rect.get_width() / 2, 0.0),
                    textcoords="offset points",
                    xytext=(0, 3),
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    color="0.4",
                )
            else:
                ax.annotate(
                    f"{height:.2f}",
                    (rect.get_x() + rect.get_width() / 2, height),
                    textcoords="offset points",
                    xytext=(0, 2 if height >= 0 else -9),
                    ha="center",
                    va="bottom" if height >= 0 else "top",
                    fontsize=7,
                )

    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Score (higher is better)")
    ax.set_title(title)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(y=0.12)
    fig.tight_layout()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def _example_metrics() -> dict[str, dict[str, float | None]]:
    """Illustrative per-method metrics showing the expected ordering.

    gnmf (learned) > spatial_smooth (baseline) > projection (MVP floor) on the
    supervised metrics. Used by ``--example`` so the panel can be rendered and
    visually reviewed without a real run.
    """
    return {
        "projection": {
            "ari_domain": 0.21,
            "nmi_domain": 0.34,
            "boundary_f1_domain": 0.28,
            "weighted_dice_domain": 0.42,
            "silhouette_domain": 0.08,
        },
        "spatial_smooth": {
            "ari_domain": 0.39,
            "nmi_domain": 0.49,
            "boundary_f1_domain": 0.45,
            "weighted_dice_domain": 0.57,
            "silhouette_domain": 0.18,
        },
        "gnmf": {
            "ari_domain": 0.56,
            "nmi_domain": 0.63,
            "boundary_f1_domain": 0.61,
            "weighted_dice_domain": 0.70,
            "silhouette_domain": 0.27,
        },
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--projection",
        type=Path,
        default=None,
        help="metrics.json for the projection (MVP) run.",
    )
    parser.add_argument(
        "--spatial-smooth",
        type=Path,
        default=None,
        help="metrics.json for the spatial_smooth baseline run.",
    )
    parser.add_argument(
        "--gnmf",
        type=Path,
        default=None,
        help="metrics.json for the gnmf (learned) run.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output PNG path.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Domain-quality method comparison (supervised vs GT domains)",
        help="Figure title.",
    )
    parser.add_argument(
        "--example",
        action="store_true",
        help=(
            "Render from built-in illustrative metrics (expected ordering "
            "gnmf > spatial_smooth > projection) instead of reading JSON bundles."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.example:
        method_metrics = _example_metrics()
    else:
        paths = {
            "projection": args.projection,
            "spatial_smooth": args.spatial_smooth,
            "gnmf": args.gnmf,
        }
        method_metrics = {
            method: extract_domain_quality(load_metrics_bundle(path))
            for method, path in paths.items()
            if path is not None
        }
        if not method_metrics:
            raise SystemExit(
                "no metrics provided: pass --projection/--spatial-smooth/--gnmf "
                "(or --example to render the illustrative panel)."
            )
    out = render_comparison(method_metrics, args.out, title=args.title)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
