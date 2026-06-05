#!/usr/bin/env python
"""Leave-one-section-out factor/label transfer panel for FactorGraph-ST (#318).

A multi-section factor model is only useful across slices if the programs it
learns on some sections *transfer* to a slice it never saw during fitting. This
figure measures that directly with a leave-one-section-out (LOSO) protocol:

  1. hold out one section, fit graph-regularized NMF on the remaining ``N-1``
     sections (their concatenated expression + within-section spatial graph),
  2. *transfer* the learned gene loadings ``W`` to the held-out slice by solving
     a nonnegative least-squares projection ``X_held ~= H_held @ W.T`` (no refit
     of ``W`` on held-out data — only the per-spot scores ``H_held``),
  3. cluster ``H_held`` into spatial domains and score the recovered partition
     against the held-out slice's ground-truth domains (ARI + NMI).

Repeating over every held-out section yields a per-section transfer score; the
panel draws grouped ARI/NMI bars per section plus the mean. A model that has
learned genuinely shared structure transfers with non-trivial ARI/NMI; one that
memorised section-local quirks collapses on the held-out slice.

:func:`compute_loso_transfer` is numpy-only (no network, no plotting) and runs on
a tiny synthetic multi-section instance in tests; matplotlib is imported lazily
inside :func:`render_loso_transfer`.

Usage (under the project conda env)::

    conda run --no-capture-output -n dl python \
        scripts/visualize/fig_loso_transfer.py --example \
        --out /tmp/factorgraph_loso_transfer.png
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from factorgraph_st.eval.metrics import adjusted_rand_index, normalized_mutual_information
from factorgraph_st.model.learned import fit_gnmf


@dataclass
class LosoTransfer:
    """Per-held-out-section transfer scores.

    ``sections``, ``ari`` and ``nmi`` are aligned arrays; entry ``i`` is the
    transfer result when section ``sections[i]`` was held out. A score may be
    ``nan`` when the held-out partition is degenerate (a single cluster), which
    the renderer marks as ``n/a`` rather than plotting a misleading value.
    """

    sections: np.ndarray
    ari: np.ndarray
    nmi: np.ndarray


def _subset_edges(edges: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Restrict ``edges`` to nodes selected by ``mask`` and reindex to 0..m-1.

    Keeps only edges whose *both* endpoints are in the mask, then remaps the
    surviving node ids to a compact ``0..m-1`` range matching the row order of
    ``X[mask]``.
    """
    edges = np.asarray(edges)
    if edges.size == 0:
        return np.empty((2, 0), dtype=np.int64)
    keep = np.where(mask)[0]
    remap = np.full(mask.shape[0], -1, dtype=np.int64)
    remap[keep] = np.arange(keep.size, dtype=np.int64)
    src, dst = edges
    both = mask[src] & mask[dst]
    return np.stack([remap[src[both]], remap[dst[both]]]).astype(np.int64)


def _kmeans(M: np.ndarray, k: int, seed: int, n_init: int = 4, max_iter: int = 50) -> np.ndarray:
    """Plain deterministic k-means returning integer labels (numpy-only)."""
    X = np.asarray(M, dtype=np.float64)
    n = X.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    k = max(1, min(int(k), n))
    rng = np.random.default_rng(seed)
    best_inertia, best = np.inf, np.zeros(n, dtype=np.int64)
    for _ in range(max(1, n_init)):
        centers = X[rng.choice(n, size=k, replace=False)].copy()
        labels = np.full(n, -1, dtype=np.int64)
        for _ in range(max_iter):
            dists = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            new = dists.argmin(axis=1).astype(np.int64)
            if np.array_equal(new, labels):
                break
            labels = new
            for c in range(k):
                m = labels == c
                if m.any():
                    centers[c] = X[m].mean(axis=0)
        inertia = float(((X - centers[labels]) ** 2).sum())
        if inertia < best_inertia:
            best_inertia, best = inertia, labels
    return best


def _project_scores(X_test: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Nonnegative least-squares projection ``X_test ~= H @ W.T`` -> ``H``.

    Solves ``W @ H.T = X_test.T`` in the least-squares sense and clips the
    result to the nonnegative orthant (the NMF score domain).
    """
    Wf = np.asarray(W, dtype=np.float64)
    coef, *_ = np.linalg.lstsq(Wf, np.asarray(X_test, dtype=np.float64).T, rcond=None)
    return np.clip(coef.T, 0.0, None)


def compute_loso_transfer(
    X: np.ndarray,
    edges: np.ndarray,
    section_id: np.ndarray,
    domain_id: np.ndarray,
    *,
    n_factors: int = 4,
    n_domains: int = 5,
    n_iter: int = 120,
    tol: float = 1e-4,
    lam: float = 1.0,
    seed: int = 0,
) -> LosoTransfer:
    """Run the LOSO factor-transfer protocol and return per-section ARI/NMI.

    For every section: fit GNMF on the other sections, transfer the learned
    loadings ``W`` to the held-out slice via a nonnegative projection, cluster
    the projected scores into ``n_domains`` domains, and score against the
    held-out ground-truth ``domain_id`` (ARI + NMI). Requires at least two
    sections. All randomness is seeded; identical inputs reproduce bitwise.
    """
    X = np.asarray(X, dtype=np.float64)
    section_id = np.asarray(section_id)
    domain_id = np.asarray(domain_id)
    sections = np.unique(section_id)
    if sections.size < 2:
        raise ValueError("LOSO transfer requires at least two sections")

    ari = np.empty(sections.size, dtype=np.float64)
    nmi = np.empty(sections.size, dtype=np.float64)
    for i, s in enumerate(sections.tolist()):
        test_mask = section_id == s
        train_mask = ~test_mask
        train_edges = _subset_edges(edges, train_mask)
        result = fit_gnmf(
            X[train_mask], train_edges, n_factors, lam=lam, n_iter=n_iter, tol=tol, seed=seed
        )
        H_test = _project_scores(X[test_mask], result.W)
        pred = _kmeans(H_test, n_domains, seed=seed)
        truth = domain_id[test_mask]
        ari[i] = adjusted_rand_index(truth, pred)
        nmi[i] = normalized_mutual_information(truth, pred)
    return LosoTransfer(sections=sections, ari=ari, nmi=nmi)


def render_loso_transfer(
    data: LosoTransfer,
    out_path: str | Path,
    *,
    title: str = "Leave-one-section-out factor transfer (held-out domain recovery)",
    dpi: int = 150,
):
    """Render grouped ARI/NMI transfer bars per held-out section; save to disk.

    ``nan`` scores (degenerate held-out partitions) draw as a hatched ``n/a``
    stub. matplotlib is imported lazily. Returns the
    :class:`matplotlib.figure.Figure`.
    """
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")  # headless, deterministic raster backend
    import matplotlib.pyplot as plt  # noqa: PLC0415

    sections = np.asarray(data.sections)
    ari = np.asarray(data.ari, dtype=np.float64)
    nmi = np.asarray(data.nmi, dtype=np.float64)
    if not (sections.size == ari.size == nmi.size):
        raise ValueError("sections, ari and nmi must be arrays of equal length")
    if sections.size == 0:
        raise ValueError("no held-out sections to plot")

    metrics = (("ARI", ari, "#0072B2"), ("NMI", nmi, "#D55E00"))
    n_groups = sections.size
    n_bars = len(metrics)
    group_width = 0.8
    bar_width = group_width / n_bars
    x = list(range(n_groups))

    fig, ax = plt.subplots(figsize=(1.5 * n_groups + 3.0, 4.8))
    for j, (label, vals, color) in enumerate(metrics):
        offset = -group_width / 2 + bar_width * (j + 0.5)
        positions = [xi + offset for xi in x]
        heights = [0.0 if not np.isfinite(v) else float(v) for v in vals]
        bars = ax.bar(
            positions, heights, width=bar_width * 0.92, label=label,
            color=color, edgecolor="black", linewidth=0.6,
        )
        for rect, v in zip(bars, vals.tolist(), strict=True):
            if not np.isfinite(v):
                rect.set_hatch("///")
                rect.set_facecolor("white")
                rect.set_edgecolor("0.6")
                ax.annotate(
                    "n/a", (rect.get_x() + rect.get_width() / 2, 0.0),
                    textcoords="offset points", xytext=(0, 3),
                    ha="center", va="bottom", fontsize=7, color="0.4",
                )
            else:
                ax.annotate(
                    f"{v:.2f}", (rect.get_x() + rect.get_width() / 2, v),
                    textcoords="offset points", xytext=(0, 2 if v >= 0 else -9),
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=7,
                )

    finite_means = [float(np.nanmean(vals)) if np.any(np.isfinite(vals)) else float("nan") for _l, vals, _c in metrics]
    mean_txt = "   ".join(
        f"mean {label}: {m:.2f}" if np.isfinite(m) else f"mean {label}: n/a"
        for (label, _v, _c), m in zip(metrics, finite_means, strict=True)
    )

    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"held-out {int(s)}" for s in sections.tolist()])
    ax.set_ylabel("transfer score (higher is better)")
    ax.set_title(f"{title}\n{mean_txt}", fontsize=11)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(y=0.14)
    fig.tight_layout()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return fig


def _example_transfer() -> LosoTransfer:
    """Compute LOSO transfer on a small synthetic multi-section instance."""
    from factorgraph_st.synth import generate_instance  # noqa: PLC0415

    inst = generate_instance(
        n_sections=3,
        n_spots_per_section=60,
        n_genes=24,
        K_shared=3,
        K_private=2,
        n_domains=4,
        k_nn=6,
        seed=0,
    )
    return compute_loso_transfer(
        inst.X, inst.edges, inst.section_id, inst.domain_id,
        n_factors=4, n_domains=4, n_iter=80, seed=0,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--npz", type=Path, default=None,
        help="Path to a .npz with arrays X/edges/section_id/domain_id.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output PNG path.")
    parser.add_argument(
        "--title", type=str,
        default="Leave-one-section-out factor transfer (held-out domain recovery)",
    )
    parser.add_argument(
        "--example", action="store_true",
        help="Render from a built-in synthetic multi-section instance.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.example:
        data = _example_transfer()
    elif args.npz is not None:
        npz = np.load(args.npz)
        data = compute_loso_transfer(npz["X"], npz["edges"], npz["section_id"], npz["domain_id"])
    else:
        raise SystemExit("provide --npz PATH.npz (X/edges/section_id/domain_id) or --example.")
    render_loso_transfer(data, args.out, title=args.title)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
