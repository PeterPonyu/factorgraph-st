#!/usr/bin/env python
"""Spatial-domain boundary quality & contiguity for FactorGraph-ST (#313).

A domain segmentation can score well on a partition metric (ARI/NMI) yet still
be spatially wrong in two ways this figure makes visible:

  * **bad boundaries** — the predicted domain borders don't land where the true
    borders are (low boundary precision/recall/F1); and
  * **over-segmentation / fragmentation** — a single true region is shattered
    into many disconnected speckles, so each predicted domain is not one
    contiguous patch but a scatter of islands.

The first is scored by reusing the ``boundary_*`` metrics from
:mod:`factorgraph_st.eval.metrics`. The second is scored by a **contiguity**
measure computed here: for each predicted domain, the fraction of its spots that
fall in its single largest spatially-connected component (over the ``edges``
graph). A perfectly contiguous segmentation scores ``1.0``; a fragmented one
drops toward ``0``. The companion **fragmentation ratio** (total connected
components across all predicted domains divided by the number of predicted
domains) is ``1.0`` for one-patch-per-domain and grows as domains shatter.

:func:`compute_boundary_quality` is numpy-only (no network, no plotting) and is
exercised on a tiny synthetic instance in tests; :func:`render_boundary_quality`
imports matplotlib lazily.

Usage (under the project conda env)::

    conda run --no-capture-output -n dl python \
        scripts/visualize/fig_boundary_quality.py --example \
        --out /tmp/factorgraph_boundary_quality.png
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from factorgraph_st.eval.metrics import boundary_f1, boundary_precision, boundary_recall


def _domain_components(mask: np.ndarray, edges: np.ndarray) -> list[int]:
    """Sizes of the connected components of the sub-graph induced on ``mask`` spots.

    Only edges whose BOTH endpoints are in ``mask`` are used, so the connectivity
    is the within-domain spatial connectivity. A masked spot with no in-domain
    edge is its own singleton component. Returns the component sizes (descending);
    an empty mask returns ``[]``.
    """
    members = np.flatnonzero(mask)
    if members.size == 0:
        return []
    # Build adjacency restricted to in-mask spots, keyed by spot index.
    neighbors: dict[int, list[int]] = {int(i): [] for i in members}
    if edges.size:
        src, dst = edges
        both = mask[src] & mask[dst]
        for s, d in zip(src[both].tolist(), dst[both].tolist(), strict=True):
            neighbors[s].append(d)
            neighbors[d].append(s)
    seen: set[int] = set()
    sizes: list[int] = []
    for start in members.tolist():
        if start in seen:
            continue
        size = 0
        stack = [int(start)]
        seen.add(int(start))
        while stack:
            node = stack.pop()
            size += 1
            for nb in neighbors[node]:
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        sizes.append(size)
    return sorted(sizes, reverse=True)


def contiguity(labels_pred: np.ndarray, edges: np.ndarray) -> dict[str, float | int]:
    """Spatial contiguity / fragmentation of a predicted domain partition.

    For every predicted domain, the largest-connected-component fraction
    ``|largest component| / |domain|`` is computed over the within-domain spatial
    graph. Returns:

    * ``contiguity`` — domain-size-weighted mean of those fractions, in ``[0, 1]``
      (``1.0`` = every domain is one contiguous patch);
    * ``n_components`` — total connected components summed over all domains;
    * ``n_domains`` — number of distinct predicted domains; and
    * ``fragmentation_ratio`` — ``n_components / n_domains`` (``1.0`` ideal, grows
      with shattering).

    Empty input returns ``contiguity = nan`` with zero counts.
    """
    labels_pred = np.asarray(labels_pred)
    n = labels_pred.shape[0]
    domains = np.unique(labels_pred)
    n_domains = int(domains.size)
    if n == 0 or n_domains == 0:
        return {"contiguity": float("nan"), "n_components": 0, "n_domains": 0,
                "fragmentation_ratio": float("nan")}
    total = 0.0
    weighted = 0.0
    n_components = 0
    for d in domains:
        mask = labels_pred == d
        sizes = _domain_components(mask, edges)
        domain_size = int(mask.sum())
        n_components += len(sizes)
        if domain_size > 0 and sizes:
            weighted += domain_size * (sizes[0] / domain_size)
            total += domain_size
    cont = float(weighted / total) if total > 0 else float("nan")
    return {
        "contiguity": cont,
        "n_components": int(n_components),
        "n_domains": n_domains,
        "fragmentation_ratio": float(n_components / n_domains) if n_domains else float("nan"),
    }


def compute_boundary_quality(
    labels_true: np.ndarray,
    labels_pred: np.ndarray,
    edges: np.ndarray,
) -> dict[str, float | int]:
    """Bundle the boundary metrics and the contiguity/fragmentation summary.

    Reuses :func:`boundary_precision` / :func:`boundary_recall` / :func:`boundary_f1`
    (predicted vs ground-truth domain borders over ``edges``) and adds the
    :func:`contiguity` summary of the *predicted* partition. Returns a flat dict
    with keys: ``boundary_precision``, ``boundary_recall``, ``boundary_f1``,
    ``contiguity``, ``n_components``, ``n_pred_domains``, ``n_true_domains``,
    ``fragmentation_ratio``. Boundary metrics may be ``nan`` when a side has no
    boundary spots (their documented not-evaluable signal).
    """
    cont = contiguity(labels_pred, edges)
    return {
        "boundary_precision": boundary_precision(labels_true, labels_pred, edges),
        "boundary_recall": boundary_recall(labels_true, labels_pred, edges),
        "boundary_f1": boundary_f1(labels_true, labels_pred, edges),
        "contiguity": cont["contiguity"],
        "n_components": cont["n_components"],
        "n_pred_domains": cont["n_domains"],
        "n_true_domains": int(np.unique(labels_true).size),
        "fragmentation_ratio": cont["fragmentation_ratio"],
    }


def render_boundary_quality(
    summary: dict[str, float | int],
    out_path: str | Path,
    *,
    title: str = "Domain boundary quality & contiguity",
    dpi: int = 150,
):
    """Render boundary-quality bars + over-segmentation counts; save to ``out_path``.

    ``summary`` is the dict returned by :func:`compute_boundary_quality`. Left
    panel: the ``[0, 1]`` quality metrics (boundary P/R/F1 + contiguity) as bars,
    with ``nan`` drawn as a hatched 'n/a' stub. Right panel: the integer domain
    counts (true domains, predicted domains, connected components) exposing
    over-segmentation. matplotlib is imported lazily. Returns the
    :class:`matplotlib.figure.Figure`.
    """
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")  # headless, deterministic raster backend
    import matplotlib.pyplot as plt  # noqa: PLC0415

    quality_keys = [
        ("boundary_precision", "Boundary\nPrecision"),
        ("boundary_recall", "Boundary\nRecall"),
        ("boundary_f1", "Boundary\nF1"),
        ("contiguity", "Contiguity"),
    ]
    count_keys = [
        ("n_true_domains", "true\ndomains"),
        ("n_pred_domains", "pred\ndomains"),
        ("n_components", "connected\ncomponents"),
    ]

    fig, (ax_q, ax_c) = plt.subplots(1, 2, figsize=(10.5, 4.5))

    qvals = [float(summary.get(k, float("nan"))) for k, _ in quality_keys]
    qlabels = [lbl for _, lbl in quality_keys]
    qx = np.arange(len(qvals))
    qheights = [0.0 if not np.isfinite(v) else v for v in qvals]
    palette = ["#56B4E9", "#56B4E9", "#0072B2", "#009E73"]
    bars_q = ax_q.bar(qx, qheights, color=palette, edgecolor="black", linewidth=0.6)
    for rect, v in zip(bars_q, qvals, strict=True):
        if not np.isfinite(v):
            rect.set_hatch("///")
            rect.set_facecolor("white")
            rect.set_edgecolor("0.6")
            ax_q.annotate("n/a", (rect.get_x() + rect.get_width() / 2, 0.0),
                          textcoords="offset points", xytext=(0, 3), ha="center",
                          va="bottom", fontsize=7, color="0.4")
        else:
            ax_q.annotate(f"{v:.2f}", (rect.get_x() + rect.get_width() / 2, v),
                          textcoords="offset points", xytext=(0, 2), ha="center",
                          va="bottom", fontsize=8)
    ax_q.set_xticks(qx)
    ax_q.set_xticklabels(qlabels, fontsize=8)
    ax_q.set_ylabel("score (higher is better)")
    ax_q.set_ylim(0.0, 1.05)
    ax_q.set_title("boundary quality")
    ax_q.spines["top"].set_visible(False)
    ax_q.spines["right"].set_visible(False)

    cvals = [int(summary.get(k, 0)) for k, _ in count_keys]
    clabels = [lbl for _, lbl in count_keys]
    cx = np.arange(len(cvals))
    bars_c = ax_c.bar(cx, cvals, color="#E69F00", edgecolor="black", linewidth=0.6)
    for rect, v in zip(bars_c, cvals, strict=True):
        ax_c.annotate(f"{v}", (rect.get_x() + rect.get_width() / 2, v),
                      textcoords="offset points", xytext=(0, 2), ha="center",
                      va="bottom", fontsize=8)
    ax_c.set_xticks(cx)
    ax_c.set_xticklabels(clabels, fontsize=8)
    ax_c.set_ylabel("count")
    frag = summary.get("fragmentation_ratio", float("nan"))
    frag_txt = f"{float(frag):.2f}" if np.isfinite(float(frag)) else "n/a"
    ax_c.set_title(f"over-segmentation (frag ratio = {frag_txt})")
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)

    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return fig


def _example_summary() -> dict[str, float | int]:
    """Illustrative summary: decent boundaries with mild over-segmentation."""
    return {
        "boundary_precision": 0.58,
        "boundary_recall": 0.64,
        "boundary_f1": 0.61,
        "contiguity": 0.82,
        "n_components": 7,
        "n_pred_domains": 5,
        "n_true_domains": 5,
        "fragmentation_ratio": 1.4,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Output PNG path.")
    parser.add_argument("--title", type=str, default="Domain boundary quality & contiguity")
    parser.add_argument("--example", action="store_true",
                        help="Render from a built-in illustrative summary.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.example:
        raise SystemExit("this script renders from computed/illustrative summaries; pass --example.")
    render_boundary_quality(_example_summary(), args.out, title=args.title)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
