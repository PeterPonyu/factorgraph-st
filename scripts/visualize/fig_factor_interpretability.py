#!/usr/bin/env python
"""Factor-interpretability validation for FactorGraph-ST (#310).

A latent factor is only worth interpreting if it is both **spatially coherent**
(its score map varies smoothly over the tissue rather than looking like noise)
and **gene-specific** (its loading vector concentrates on a compact signature of
genes rather than spreading thin across the whole panel). A factor that is one
without the other is a trap: a smooth-but-diffuse factor has no signature to read
off, and a peaky-but-spatially-random factor is a technical artifact, not a
program. This figure scores every factor on both axes at once so neither can be
claimed in isolation.

For each factor ``j`` it reports two GT-free quantities:

  * **spatial coherence** — Moran's I of the score column ``H[:, j]`` over the
    spatial graph ``edges`` (higher = neighbors agree, the map is smooth); and
  * **signature enrichment** — the fraction of the loading column ``W[:, j]``'s
    total mass carried by its top-``top_n`` genes (higher = a compact, readable
    gene signature). This is the simple top-loading enrichment proxy: it needs no
    external gene set, so it works on synthetic data and on real panels alike.

:func:`compute_factor_interpretability` does the numpy-only scoring (no network,
no plotting) and is exercised on a tiny synthetic instance in tests;
:func:`render_interpretability` imports matplotlib lazily.

Usage (under the project conda env)::

    conda run --no-capture-output -n dl python \
        scripts/visualize/fig_factor_interpretability.py --example \
        --out /tmp/factorgraph_factor_interpretability.png
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from factorgraph_st.eval.metrics import morans_i


def signature_enrichment(loadings: np.ndarray, top_n: int) -> np.ndarray:
    """Top-``top_n`` loading-mass fraction per factor (a gene-signature proxy).

    For each factor (column of the nonnegative loading matrix ``W``, shape
    ``(n_genes, n_factors)``) this returns the fraction of the column's total
    absolute mass concentrated in its ``top_n`` largest-magnitude genes:

        enrichment_j = sum(top_n |W[:, j]|) / sum(|W[:, j]|)

    The result lies in ``[top_n / n_genes, 1]``: a uniform loading (no signature)
    sits at the ``top_n / n_genes`` floor, while a spike on exactly ``top_n``
    genes reaches ``1.0``. A zero column (no signal) is not evaluable and scores
    ``float('nan')``. ``top_n`` is clamped to ``[1, n_genes]``.
    """
    W = np.abs(np.asarray(loadings, dtype=np.float64))
    if W.ndim != 2:
        raise ValueError("loadings must be 2D (n_genes, n_factors)")
    n_genes, n_factors = W.shape
    if n_genes == 0 or n_factors == 0:
        return np.full(n_factors, np.nan, dtype=np.float64)
    k = int(min(max(top_n, 1), n_genes))
    out = np.full(n_factors, np.nan, dtype=np.float64)
    for j in range(n_factors):
        col = W[:, j]
        total = float(col.sum())
        if total <= 0.0:
            continue  # zero loading column: no signature -> not evaluable
        top = np.sort(col)[-k:]
        out[j] = float(top.sum() / total)
    return out


def compute_factor_interpretability(
    H: np.ndarray,
    W: np.ndarray,
    edges: np.ndarray,
    *,
    top_n: int = 10,
) -> dict[str, np.ndarray]:
    """Score every factor on spatial coherence and signature enrichment.

    ``H`` is the ``(n_spots, n_factors)`` score matrix and ``W`` the
    ``(n_genes, n_factors)`` loading matrix (same factor order). Returns a dict
    of three aligned per-factor arrays:

    * ``coherence`` — Moran's I of ``H[:, j]`` over ``edges`` (spatial smoothness);
    * ``enrichment`` — top-``top_n`` loading-mass fraction of ``W[:, j]``; and
    * ``factor_index`` — ``0 .. n_factors - 1`` (the plotting/label order).

    Raises if ``H`` and ``W`` disagree on the factor count.
    """
    H = np.asarray(H, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    if H.ndim != 2 or W.ndim != 2:
        raise ValueError("H and W must both be 2D")
    if H.shape[1] != W.shape[1]:
        raise ValueError("H and W must share the factor axis (H.shape[1] == W.shape[1])")
    n_factors = H.shape[1]
    coherence = np.array(
        [morans_i(H[:, j], edges) for j in range(n_factors)], dtype=np.float64
    )
    enrichment = signature_enrichment(W, top_n)
    return {
        "coherence": coherence,
        "enrichment": enrichment,
        "factor_index": np.arange(n_factors, dtype=np.int64),
    }


def render_interpretability(
    scores: dict[str, np.ndarray],
    out_path: str | Path,
    *,
    title: str = "Factor interpretability (spatial coherence vs gene signature)",
    coherence_ref: float = 0.3,
    enrichment_ref: float = 0.5,
    dpi: int = 150,
):
    """Render the per-factor bars + coherence/enrichment scatter; save to ``out_path``.

    ``scores`` is the dict returned by :func:`compute_factor_interpretability`.
    Left panel: grouped bars (coherence, enrichment) per factor. Right panel: a
    coherence (x) vs enrichment (y) scatter with each point labelled by its
    factor index and reference guide-lines at ``coherence_ref`` / ``enrichment_ref``
    — factors in the upper-right quadrant are the coherent, gene-specific
    (interpretable) ones. matplotlib is imported lazily. Returns the
    :class:`matplotlib.figure.Figure`.
    """
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")  # headless, deterministic raster backend
    import matplotlib.pyplot as plt  # noqa: PLC0415

    coherence = np.asarray(scores["coherence"], dtype=np.float64)
    enrichment = np.asarray(scores["enrichment"], dtype=np.float64)
    idx = np.asarray(scores["factor_index"], dtype=np.int64)
    if not (coherence.size == enrichment.size == idx.size) or coherence.size == 0:
        raise ValueError("coherence, enrichment and factor_index must be non-empty arrays of equal length")

    fig, (ax_bar, ax_sc) = plt.subplots(1, 2, figsize=(11.0, 4.5))

    # Left: grouped bars per factor (NaN enrichment drawn as a hatched 'n/a' stub).
    x = np.arange(idx.size)
    bar_w = 0.4
    enr_plot = np.where(np.isfinite(enrichment), enrichment, 0.0)
    ax_bar.bar(x - bar_w / 2, coherence, width=bar_w, color="#0072B2", edgecolor="black",
               linewidth=0.6, label="spatial coherence (Moran's I)")
    bars_e = ax_bar.bar(x + bar_w / 2, enr_plot, width=bar_w, color="#E69F00", edgecolor="black",
                        linewidth=0.6, label="signature enrichment (top-N mass)")
    for rect, finite in zip(bars_e, np.isfinite(enrichment), strict=True):
        if not finite:
            rect.set_hatch("///")
            rect.set_facecolor("white")
            rect.set_edgecolor("0.6")
    ax_bar.axhline(0.0, color="black", linewidth=0.8)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([f"f{int(i)}" for i in idx])
    ax_bar.set_xlabel("factor")
    ax_bar.set_ylabel("score")
    ax_bar.set_title("per-factor scores")
    ax_bar.legend(loc="upper right", frameon=False, fontsize=8)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    # Right: coherence vs enrichment scatter with interpretability quadrant.
    finite = np.isfinite(coherence) & np.isfinite(enrichment)
    ax_sc.axvline(coherence_ref, color="0.7", linestyle="--", linewidth=0.8)
    ax_sc.axhline(enrichment_ref, color="0.7", linestyle="--", linewidth=0.8)
    ax_sc.scatter(coherence[finite], enrichment[finite], s=70, c="#009E73",
                  edgecolor="black", linewidth=0.6, zorder=2)
    for xi, yi, fi in zip(coherence[finite].tolist(), enrichment[finite].tolist(),
                          idx[finite].tolist(), strict=True):
        ax_sc.annotate(f"f{int(fi)}", (xi, yi), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax_sc.set_xlabel("spatial coherence (Moran's I)")
    ax_sc.set_ylabel("signature enrichment (top-N mass)")
    ax_sc.set_title("interpretability map (upper-right = best)")
    ax_sc.spines["top"].set_visible(False)
    ax_sc.spines["right"].set_visible(False)

    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return fig


def _example_scores() -> dict[str, np.ndarray]:
    """Illustrative four-factor scores spanning the interpretability quadrants."""
    return {
        "coherence": np.array([0.62, 0.55, 0.18, 0.40], dtype=np.float64),
        "enrichment": np.array([0.71, 0.30, 0.66, 0.25], dtype=np.float64),
        "factor_index": np.arange(4, dtype=np.int64),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=None,
                        help="Path to a .npz with arrays coherence/enrichment/factor_index.")
    parser.add_argument("--out", type=Path, required=True, help="Output PNG path.")
    parser.add_argument("--title", type=str,
                        default="Factor interpretability (spatial coherence vs gene signature)")
    parser.add_argument("--example", action="store_true", help="Render from built-in illustrative scores.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.example:
        scores = _example_scores()
    elif args.scores is not None:
        data = np.load(args.scores)
        scores = {
            "coherence": data["coherence"],
            "enrichment": data["enrichment"],
            "factor_index": data["factor_index"],
        }
    else:
        raise SystemExit("provide --scores PATH.npz (coherence/enrichment/factor_index) or --example.")
    render_interpretability(scores, args.out, title=args.title)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
