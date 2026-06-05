#!/usr/bin/env python
"""#311 — multi-section integration scorecard for FactorGraph-ST.

A good multi-section integration achieves two things at once: it **mixes** spots
from different sections (removes the batch/section axis) while **conserving**
biology (keeps the spatial domains separated). This figure aggregates both into
a single scorecard panel computed on a tiny synthetic *multi-section* instance.

The two scorecard groups, each in ``[0, 1]`` with higher = better:

  * **batch mixing** (section axis removal, measured on the joint factor scores
    ``H``):
      - ``section_mixing``  — ``1 - sil01(section)``: low section silhouette
        (sections overlap) scores high.
      - ``mixing_gain``     — improvement in section mixing of ``H`` over raw
        expression ``X`` (clamped to ``[0, 1]``).
  * **biology conservation** (spatial-domain structure, on ``H``):
      - ``domain_conservation`` — ``sil01(domain)``: well-separated domains
        score high.
      - ``conservation_gain``   — improvement in domain separation of ``H`` over
        raw ``X`` (clamped to ``[0, 1]``).

where ``sil01(x) = (silhouette(x) + 1) / 2`` maps the silhouette coefficient
from ``[-1, 1]`` to ``[0, 1]``. The headline ``overall`` score is the mean of
the two group means, balancing mixing against conservation.

:func:`compute_integration_scorecard` is numpy-only (reuses the shared
``factorgraph_st.eval.metrics.silhouette`` and ``factorgraph_st.model.fit_gnmf``;
no network, no plotting) and runs on a tiny instance in tests. matplotlib is
imported lazily inside :func:`render_integration_scorecard`.

Usage (under the project conda env)::

    conda run --no-capture-output -n dl python \
        scripts/visualize/fig_integration_scorecard.py --example \
        --out /tmp/factorgraph_integration_scorecard.png
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from factorgraph_st.eval.metrics import silhouette
from factorgraph_st.model.learned import fit_gnmf

# Ordered (attr, display-label, group) spec driving both the dataclass readout
# and the rendered bar order. Two metrics per group.
SCORECARD_METRICS: tuple[tuple[str, str, str], ...] = (
    ("section_mixing", "Section mixing", "batch mixing"),
    ("mixing_gain", "Mixing gain (vs raw)", "batch mixing"),
    ("domain_conservation", "Domain conservation", "biology conservation"),
    ("conservation_gain", "Conservation gain (vs raw)", "biology conservation"),
)
GROUP_ORDER: tuple[str, ...] = ("batch mixing", "biology conservation")


@dataclass
class IntegrationScorecard:
    """Aggregated batch-mixing vs biology-conservation integration scores.

    All four scorecard scores and ``overall`` are in ``[0, 1]`` (higher better),
    or ``nan`` when a silhouette is not evaluable (degenerate labels). The raw
    before/after silhouettes are retained for transparency.
    """

    section_mixing: float
    mixing_gain: float
    domain_conservation: float
    conservation_gain: float
    overall: float
    sil_section_before: float
    sil_section_after: float
    sil_domain_before: float
    sil_domain_after: float


def _sil01(value: float) -> float:
    """Map a silhouette coefficient in ``[-1, 1]`` to ``[0, 1]`` (``nan``-safe)."""
    if not math.isfinite(value):
        return float("nan")
    return (float(value) + 1.0) / 2.0


def _gain(after01: float, before01: float) -> float:
    """Improvement of an after-vs-before ``[0, 1]`` score, clamped to ``[0, 1]``."""
    if not (math.isfinite(after01) and math.isfinite(before01)):
        return float("nan")
    return float(min(1.0, max(0.0, after01 - before01)))


def _nanmean(values: Sequence[float]) -> float:
    """Mean of the finite entries (``nan`` if none are finite)."""
    finite = [v for v in values if math.isfinite(v)]
    return float(sum(finite) / len(finite)) if finite else float("nan")


def compute_integration_scorecard(
    X: np.ndarray,
    edges: np.ndarray,
    section_id: np.ndarray,
    domain_id: np.ndarray,
    *,
    n_factors: int = 6,
    n_iter: int = 120,
    tol: float = 1e-4,
    lam: float = 1.0,
    seed: int = 0,
) -> IntegrationScorecard:
    """Compute the integration scorecard from a joint multi-section GNMF fit.

    Fits graph-regularized NMF jointly across all sections, then scores section
    mixing and domain conservation on the learned factor scores ``H`` relative
    to the raw expression ``X``. Seeded and deterministic; no network.
    """
    X = np.asarray(X, dtype=np.float64)
    section_id = np.asarray(section_id)
    domain_id = np.asarray(domain_id)
    result = fit_gnmf(X, edges, n_factors, lam=lam, n_iter=n_iter, tol=tol, seed=seed)
    H = result.H.astype(np.float64)

    sil_section_before = silhouette(X, section_id)
    sil_section_after = silhouette(H, section_id)
    sil_domain_before = silhouette(X, domain_id)
    sil_domain_after = silhouette(H, domain_id)

    sec_after01 = _sil01(sil_section_after)
    sec_before01 = _sil01(sil_section_before)
    dom_after01 = _sil01(sil_domain_after)
    dom_before01 = _sil01(sil_domain_before)

    # Mixing: low section silhouette (sections overlap) is good -> 1 - sil01.
    section_mixing = float("nan") if math.isnan(sec_after01) else 1.0 - sec_after01
    # A gain here means the section silhouette DROPPED from raw X to H.
    mixing_gain = _gain(1.0 - sec_after01, 1.0 - sec_before01)
    domain_conservation = dom_after01
    conservation_gain = _gain(dom_after01, dom_before01)

    overall = _nanmean(
        [
            _nanmean([section_mixing, mixing_gain]),
            _nanmean([domain_conservation, conservation_gain]),
        ]
    )

    return IntegrationScorecard(
        section_mixing=section_mixing,
        mixing_gain=mixing_gain,
        domain_conservation=domain_conservation,
        conservation_gain=conservation_gain,
        overall=overall,
        sil_section_before=float(sil_section_before),
        sil_section_after=float(sil_section_after),
        sil_domain_before=float(sil_domain_before),
        sil_domain_after=float(sil_domain_after),
    )


_GROUP_COLORS: dict[str, str] = {
    "batch mixing": "#0072B2",          # blue
    "biology conservation": "#009E73",  # green
}


def render_integration_scorecard(
    card: IntegrationScorecard,
    out_path: str | Path,
    *,
    title: str = "Multi-section integration scorecard",
    dpi: int = 150,
):
    """Render the horizontal-bar scorecard and save it to disk.

    Bars are grouped by category (batch mixing vs biology conservation), one
    color per group, each annotated with its value; a not-evaluable (``nan``)
    score is drawn as a hatched ``n/a`` stub. The headline ``overall`` score is
    boxed in the corner. matplotlib is imported lazily. Returns the
    :class:`matplotlib.figure.Figure` (one axis, inspectable for tests).
    """
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")  # headless, deterministic raster backend
    import matplotlib.pyplot as plt  # noqa: PLC0415

    labels = [label for _attr, label, _group in SCORECARD_METRICS]
    groups = [group for _attr, _label, group in SCORECARD_METRICS]
    values = [getattr(card, attr) for attr, _label, _group in SCORECARD_METRICS]
    colors = [_GROUP_COLORS.get(g, "#999999") for g in groups]

    # Top-to-bottom in spec order: reverse y so the first metric sits on top.
    y = list(range(len(labels)))[::-1]

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    for yi, value, color in zip(y, values, colors, strict=True):
        if math.isnan(value):
            ax.barh(yi, 1.0, color="white", edgecolor="0.6", hatch="///", height=0.6)
            ax.annotate(
                "n/a", (0.02, yi), va="center", ha="left", fontsize=8, color="0.4",
            )
        else:
            ax.barh(yi, value, color=color, edgecolor="black", linewidth=0.6, height=0.6)
            ax.annotate(
                f"{value:.2f}",
                (value, yi),
                textcoords="offset points",
                xytext=(4, 0),
                va="center", ha="left", fontsize=9,
            )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("score (higher is better)")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Group legend (color = category).
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=_GROUP_COLORS[g]) for g in GROUP_ORDER
    ]
    ax.legend(handles, list(GROUP_ORDER), loc="lower right", frameon=False, fontsize=8)

    overall_text = "n/a" if math.isnan(card.overall) else f"{card.overall:.2f}"
    ax.annotate(
        f"overall integration\nscore = {overall_text}",
        xy=(0.98, 0.97), xycoords="axes fraction",
        ha="right", va="top", fontsize=10, fontweight="bold",
        bbox={"boxstyle": "round", "facecolor": "#F0E442", "edgecolor": "0.4", "alpha": 0.9},
    )

    fig.tight_layout()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return fig


def _example_scorecard() -> IntegrationScorecard:
    """Compute the scorecard on a small synthetic multi-section instance."""
    from factorgraph_st.synth import generate_instance  # noqa: PLC0415

    inst = generate_instance(
        n_sections=3,
        n_spots_per_section=60,
        n_genes=30,
        K_shared=4,
        K_private=2,
        n_domains=4,
        k_nn=6,
        seed=0,
    )
    return compute_integration_scorecard(
        inst.X, inst.edges, inst.section_id, inst.domain_id,
        n_factors=6, n_iter=80, seed=0,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--npz", type=Path, default=None,
        help="Path to a .npz with arrays X/edges/section_id/domain_id.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output PNG path.")
    parser.add_argument("--title", type=str, default="Multi-section integration scorecard")
    parser.add_argument(
        "--example", action="store_true",
        help="Render from a built-in synthetic multi-section instance.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.example:
        card = _example_scorecard()
    elif args.npz is not None:
        npz = np.load(args.npz)
        card = compute_integration_scorecard(
            npz["X"], npz["edges"], npz["section_id"], npz["domain_id"]
        )
    else:
        raise SystemExit("provide --npz PATH.npz (X/edges/section_id/domain_id) or --example.")
    render_integration_scorecard(card, args.out, title=args.title)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
