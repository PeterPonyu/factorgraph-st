#!/usr/bin/env python
"""Per-domain reliability via perturbation survival for FactorGraph-ST (#329).

A domain label is only trustworthy if it *survives* small perturbations of the
pipeline — if re-seeding the fit or dropping a few spatial edges reshuffles a
region's spots into different domains, that region's "domain" is an artifact of
the run, not a real spatial program. This figure measures that survival without
any ground truth.

Across ``n_perturb`` perturbed re-fits (each combining a **re-seed** and a random
**edge dropout**) it accumulates a **consensus co-association** matrix
``C[i, j]`` = the fraction of perturbations in which spots ``i`` and ``j`` land in
the same domain. A reference clustering (the unperturbed fit) groups spots into
domains, and each reference domain's **survival score** is the mean off-diagonal
co-association among its own spots — how reliably those spots stay together. High
survival = a robust domain; low survival = a fragile one that perturbation breaks
apart.

:func:`compute_perturbation_consensus` does the numpy-only re-fitting and
consensus accumulation (no network, no plotting) and is exercised on a tiny
synthetic instance in tests; :func:`render_reliability` imports matplotlib lazily.

Usage (under the project conda env)::

    conda run --no-capture-output -n dl python \
        scripts/visualize/fig_domain_reliability.py --example \
        --out /tmp/factorgraph_domain_reliability.png
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from factorgraph_st.model.learned import fit_gnmf


def _zscore(arr: np.ndarray) -> np.ndarray:
    """Column z-normalize, leaving zero-variance columns at zero."""
    a = np.asarray(arr, dtype=np.float64)
    if a.size == 0:
        return a
    mean = a.mean(axis=0, keepdims=True)
    std = a.std(axis=0, keepdims=True)
    return np.where(std > 0, (a - mean) / np.maximum(std, 1e-12), 0.0)


def _kmeans_labels(feat: np.ndarray, k: int, *, seed: int, n_init: int = 4, max_iter: int = 50) -> np.ndarray:
    """Deterministic plain k-means labels over rows of ``feat`` (numpy-only)."""
    X = np.asarray(feat, dtype=np.float64)
    n = X.shape[0]
    k = max(1, min(int(k), n))
    if k == 1 or n == 0:
        return np.zeros(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    best_inertia, best_labels = np.inf, np.zeros(n, dtype=np.int64)
    for _ in range(max(1, n_init)):
        centers = X[rng.choice(n, size=k, replace=False)].copy()
        labels = np.full(n, -1, dtype=np.int64)
        for _ in range(max_iter):
            dists = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            new_labels = dists.argmin(axis=1).astype(np.int64)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for c in range(k):
                mask = labels == c
                centers[c] = X[mask].mean(0) if mask.any() else X[int(np.argmax(dists.min(1)))]
        inertia = float(((X - centers[labels]) ** 2).sum())
        if inertia < best_inertia:
            best_inertia, best_labels = inertia, labels
    return best_labels


def _cluster_domains(
    X: np.ndarray, coords: np.ndarray, edges: np.ndarray, n_domains: int, n_factors: int,
    *, lam: float, n_iter: int, seed: int,
) -> np.ndarray:
    """Fit GNMF then k-means on z-scored ``[H | coords]`` -> per-spot domain ids."""
    result = fit_gnmf(X, edges, n_factors, lam=lam, n_iter=n_iter, seed=seed)
    feat = np.hstack([_zscore(result.H), _zscore(coords)])
    return _kmeans_labels(feat, n_domains, seed=seed)


def _drop_edges(edges: np.ndarray, drop_frac: float, rng: np.random.Generator) -> np.ndarray:
    """Randomly keep ``1 - drop_frac`` of the undirected edges (columns of ``edges``)."""
    if edges.size == 0 or drop_frac <= 0.0:
        return edges
    n_edges = edges.shape[1]
    keep = rng.random(n_edges) >= drop_frac
    if not keep.any():  # never return an empty graph (GNMF needs some structure)
        keep[rng.integers(n_edges)] = True
    return edges[:, keep]


def compute_perturbation_consensus(
    X: np.ndarray,
    coords: np.ndarray,
    edges: np.ndarray,
    n_domains: int,
    *,
    n_perturb: int = 8,
    drop_frac: float = 0.1,
    n_factors: int = 4,
    n_iter: int = 60,
    lam: float = 1.0,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Run perturbed re-fits and return consensus + per-domain survival.

    A reference clustering is computed on the full graph at ``seed``. Then for
    each of ``n_perturb`` perturbations the spatial graph is randomly thinned by
    ``drop_frac`` and the model re-fit at a fresh seed; the resulting domains feed
    a consensus co-association matrix ``C`` (``C[i, j]`` = fraction of
    perturbations in which ``i`` and ``j`` co-cluster). Returns a dict with:

    * ``consensus`` — ``(n_spots, n_spots)`` co-association matrix in ``[0, 1]``
      (symmetric, unit diagonal);
    * ``reference_labels`` — ``(n_spots,)`` reference domain ids;
    * ``domain_ids`` — sorted unique reference domain ids; and
    * ``survival`` — per reference domain, the mean off-diagonal co-association
      among its spots (``[0, 1]``; ``nan`` for singleton domains).

    All randomness is seeded, so the result is deterministic for fixed inputs.
    """
    if n_perturb < 1:
        raise ValueError("n_perturb must be >= 1")
    X = np.asarray(X, dtype=np.float64)
    coords = np.asarray(coords, dtype=np.float64)
    n = X.shape[0]

    reference_labels = _cluster_domains(
        X, coords, edges, n_domains, n_factors, lam=lam, n_iter=n_iter, seed=seed
    )

    co_assoc = np.zeros((n, n), dtype=np.float64)
    rng = np.random.default_rng(seed)
    for b in range(n_perturb):
        edges_b = _drop_edges(edges, drop_frac, rng)
        labels_b = _cluster_domains(
            X, coords, edges_b, n_domains, n_factors, lam=lam, n_iter=n_iter, seed=seed + 1 + b
        )
        same = labels_b[:, None] == labels_b[None, :]
        co_assoc += same.astype(np.float64)
    consensus = co_assoc / float(n_perturb)
    if n:
        np.fill_diagonal(consensus, 1.0)

    domain_ids = np.unique(reference_labels)
    survival = np.full(domain_ids.size, np.nan, dtype=np.float64)
    for di, d in enumerate(domain_ids.tolist()):
        idx = np.flatnonzero(reference_labels == d)
        if idx.size < 2:
            continue  # singleton domain: off-diagonal survival not evaluable
        sub = consensus[np.ix_(idx, idx)]
        off = sub.sum() - np.trace(sub)
        survival[di] = float(off / (idx.size * (idx.size - 1)))
    return {
        "consensus": consensus,
        "reference_labels": reference_labels,
        "domain_ids": domain_ids.astype(np.int64),
        "survival": survival,
    }


def render_reliability(
    result: dict[str, np.ndarray],
    out_path: str | Path,
    *,
    title: str = "Per-domain reliability (perturbation survival)",
    dpi: int = 150,
):
    """Render the consensus heatmap + per-domain survival bars; save to ``out_path``.

    ``result`` is the dict returned by :func:`compute_perturbation_consensus`.
    Left panel: the consensus co-association matrix with spots ordered by
    reference domain (block structure = reliable domains). Right panel: per-domain
    survival bars (``nan`` survival drawn as a hatched 'n/a' stub). matplotlib is
    imported lazily. Returns the :class:`matplotlib.figure.Figure`.
    """
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")  # headless, deterministic raster backend
    import matplotlib.pyplot as plt  # noqa: PLC0415

    consensus = np.asarray(result["consensus"], dtype=np.float64)
    reference_labels = np.asarray(result["reference_labels"])
    domain_ids = np.asarray(result["domain_ids"])
    survival = np.asarray(result["survival"], dtype=np.float64)
    if consensus.ndim != 2 or consensus.shape[0] != consensus.shape[1]:
        raise ValueError("consensus must be a square 2D matrix")
    if survival.size != domain_ids.size:
        raise ValueError("survival and domain_ids must have equal length")

    order = np.argsort(reference_labels, kind="stable")
    ordered = consensus[np.ix_(order, order)]

    fig, (ax_hm, ax_bar) = plt.subplots(1, 2, figsize=(11.0, 4.6))

    im = ax_hm.imshow(ordered, cmap="viridis", vmin=0.0, vmax=1.0, interpolation="nearest")
    ax_hm.set_title("consensus co-association (spots ordered by domain)")
    ax_hm.set_xlabel("spot")
    ax_hm.set_ylabel("spot")
    fig.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04, label="co-association")

    bx = np.arange(domain_ids.size)
    heights = np.where(np.isfinite(survival), survival, 0.0)
    bars = ax_bar.bar(bx, heights, color="#009E73", edgecolor="black", linewidth=0.6)
    for rect, v in zip(bars, survival.tolist(), strict=True):
        if not np.isfinite(v):
            rect.set_hatch("///")
            rect.set_facecolor("white")
            rect.set_edgecolor("0.6")
            ax_bar.annotate("n/a", (rect.get_x() + rect.get_width() / 2, 0.0),
                            textcoords="offset points", xytext=(0, 3), ha="center",
                            va="bottom", fontsize=7, color="0.4")
        else:
            ax_bar.annotate(f"{v:.2f}", (rect.get_x() + rect.get_width() / 2, v),
                            textcoords="offset points", xytext=(0, 2), ha="center",
                            va="bottom", fontsize=8)
    ax_bar.set_xticks(bx)
    ax_bar.set_xticklabels([f"d{int(d)}" for d in domain_ids.tolist()])
    ax_bar.set_xlabel("reference domain")
    ax_bar.set_ylabel("survival (mean co-association)")
    ax_bar.set_ylim(0.0, 1.05)
    ax_bar.set_title("per-domain survival")
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return fig


def _example_result() -> dict[str, np.ndarray]:
    """Illustrative two-block consensus: one reliable domain, one fragile."""
    block = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    # Domain 0 cohesive (~0.95 within), domain 1 fragile (~0.45 within).
    within = np.array([0.95, 0.45])
    n = block.size
    consensus = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i == j:
                consensus[i, j] = 1.0
            elif block[i] == block[j]:
                consensus[i, j] = within[block[i]]
            else:
                consensus[i, j] = 0.05
    return {
        "consensus": consensus,
        "reference_labels": block,
        "domain_ids": np.array([0, 1], dtype=np.int64),
        "survival": within.astype(np.float64),
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Output PNG path.")
    parser.add_argument("--title", type=str, default="Per-domain reliability (perturbation survival)")
    parser.add_argument("--example", action="store_true",
                        help="Render from a built-in illustrative consensus.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.example:
        raise SystemExit("this script renders from a computed/illustrative result; pass --example.")
    render_reliability(_example_result(), args.out, title=args.title)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
