#!/usr/bin/env python
"""Cross-method consensus co-association map for FactorGraph-ST (#328).

Without ground-truth domain labels, the trustworthiness of a spatial-domain
partition is best judged by its *stability* across model variants and random
seeds: spot pairs that the methods agree to co-cluster are robust; pairs that
flip with the seed are not. This figure builds that evidence with no GT at all:

  1. cluster the same slice many times under different ``(lambda, seed)`` GNMF
     variants,
  2. accumulate a **co-association matrix** ``C`` where ``C[i, j]`` is the
     fraction of runs that placed spots ``i`` and ``j`` in the same domain,
  3. derive a **consensus** partition by clustering on ``C`` itself, and render
     ``C`` reordered by the consensus labels so robust domains appear as bright
     diagonal blocks.

A sharp block-diagonal ``C`` means the domains are method/seed-stable; a washed-
out ``C`` means the partition is an artefact of a particular run. An optional
second panel maps the consensus labels back onto the spatial coordinates.

:func:`compute_consensus_coassociation` is numpy-only (no network, no plotting)
and runs on a tiny synthetic instance in tests; matplotlib is imported lazily
inside :func:`render_consensus_coassociation`.

Usage (under the project conda env)::

    conda run --no-capture-output -n dl python \
        scripts/visualize/fig_consensus_coassociation.py --example \
        --out /tmp/factorgraph_consensus_coassociation.png
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from factorgraph_st.model.learned import fit_gnmf

# Default GNMF variants (lambda, seed) whose domain partitions are pooled into
# the co-association matrix: two smoothing strengths x three seeds = six runs.
DEFAULT_CONFIGS: tuple[tuple[float, int], ...] = (
    (0.0, 0), (0.0, 1), (0.0, 2),
    (1.0, 0), (1.0, 1), (1.0, 2),
)


@dataclass
class ConsensusCoassociation:
    """Co-association matrix + consensus partition over many clustering runs.

    Attributes
    ----------
    coassoc:
        ``(n_spots, n_spots)`` symmetric matrix; ``coassoc[i, j]`` is the
        fraction of runs that co-clustered spots ``i`` and ``j`` (diagonal 1.0).
    consensus:
        ``(n_spots,)`` consensus domain label per spot (clustered from ``coassoc``).
    n_runs:
        Number of clustering runs pooled into ``coassoc``.
    """

    coassoc: np.ndarray
    consensus: np.ndarray
    n_runs: int


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


def compute_consensus_coassociation(
    X: np.ndarray,
    edges: np.ndarray,
    *,
    n_domains: int = 4,
    n_factors: int = 4,
    configs: Sequence[tuple[float, int]] | None = None,
    n_iter: int = 80,
    tol: float = 1e-4,
) -> ConsensusCoassociation:
    """Pool many GNMF clusterings into a co-association matrix + consensus partition.

    Each ``(lam, seed)`` in ``configs`` fits GNMF and clusters its factor scores
    into ``n_domains`` domains; the binary co-clustering indicator matrices are
    averaged into ``coassoc``. The consensus partition is obtained by clustering
    the rows of ``coassoc``. numpy-only, fully seeded, no network.
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    if n == 0:
        raise ValueError("X must have at least one spot")
    cfgs = DEFAULT_CONFIGS if configs is None else tuple(configs)
    if len(cfgs) == 0:
        raise ValueError("configs must be non-empty")

    coassoc = np.zeros((n, n), dtype=np.float64)
    for lam, seed in cfgs:
        result = fit_gnmf(X, edges, n_factors, lam=float(lam), n_iter=n_iter, tol=tol, seed=int(seed))
        labels = _kmeans(result.H.astype(np.float64), n_domains, seed=int(seed))
        coassoc += (labels[:, None] == labels[None, :]).astype(np.float64)
    coassoc /= float(len(cfgs))
    consensus = _kmeans(coassoc, n_domains, seed=0)
    return ConsensusCoassociation(coassoc=coassoc, consensus=consensus, n_runs=len(cfgs))


def render_consensus_coassociation(
    data: ConsensusCoassociation,
    out_path: str | Path,
    *,
    coords: np.ndarray | None = None,
    title: str = "Cross-method consensus co-association (no ground truth)",
    dpi: int = 150,
):
    """Render the consensus-ordered co-association heatmap (+ optional spatial map).

    Rows/cols of the heatmap are reordered by the consensus partition so stable
    domains form bright diagonal blocks; thin lines mark the block boundaries.
    When ``coords`` is given, a second panel maps consensus labels onto space.
    matplotlib is imported lazily. Returns the
    :class:`matplotlib.figure.Figure`.
    """
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")  # headless, deterministic raster backend
    import matplotlib.pyplot as plt  # noqa: PLC0415

    C = np.asarray(data.coassoc, dtype=np.float64)
    consensus = np.asarray(data.consensus)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("coassoc must be a square matrix")
    if consensus.shape[0] != C.shape[0]:
        raise ValueError("consensus labels must align with coassoc rows")

    order = np.argsort(consensus, kind="mergesort")
    C_ord = C[np.ix_(order, order)]
    sorted_labels = consensus[order]
    boundaries = np.where(np.diff(sorted_labels) != 0)[0] + 0.5

    n_panels = 1 if coords is None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5.6 * n_panels, 5.2), squeeze=False)
    ax_heat = axes[0, 0]
    im = ax_heat.imshow(C_ord, cmap="magma", vmin=0.0, vmax=1.0, aspect="auto", origin="upper")
    for b in boundaries.tolist():
        ax_heat.axhline(b, color="#4dd0e1", linewidth=0.7)
        ax_heat.axvline(b, color="#4dd0e1", linewidth=0.7)
    ax_heat.set_xlabel("spot (consensus order)")
    ax_heat.set_ylabel("spot (consensus order)")
    ax_heat.set_title(f"co-association over {data.n_runs} runs")
    fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04, label="co-clustering fraction")

    if coords is not None:
        coords_arr = np.asarray(coords, dtype=np.float64)
        if coords_arr.shape[0] != C.shape[0] or coords_arr.shape[1] < 2:
            raise ValueError("coords must be (n_spots, 2) aligned with coassoc")
        ax_map = axes[0, 1]
        palette = ["#56B4E9", "#E69F00", "#009E73", "#CC79A7", "#0072B2", "#D55E00", "#F0E442", "#999999"]
        for i, lab in enumerate(np.unique(consensus).tolist()):
            mask = consensus == lab
            ax_map.scatter(
                coords_arr[mask, 0], coords_arr[mask, 1],
                s=12, color=palette[i % len(palette)], edgecolor="none",
                alpha=0.85, label=str(int(lab)),
            )
        ax_map.set_xticks([])
        ax_map.set_yticks([])
        ax_map.set_title("consensus domains in space")
        ax_map.legend(title="domain", loc="best", fontsize=6, title_fontsize=7, frameon=False)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return fig


def _example_instance():
    """A small single-section synthetic slice (kept tiny so the n x n map is small)."""
    from factorgraph_st.synth import generate_instance  # noqa: PLC0415

    return generate_instance(
        n_sections=1,
        n_spots_per_section=80,
        n_genes=24,
        K_shared=3,
        K_private=1,
        n_domains=4,
        k_nn=6,
        seed=0,
    )


def _example_consensus() -> tuple[ConsensusCoassociation, np.ndarray]:
    """Compute consensus co-association on the small synthetic slice."""
    inst = _example_instance()
    data = compute_consensus_coassociation(inst.X, inst.edges, n_domains=4, n_factors=4, n_iter=60)
    return data, inst.coords


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--npz", type=Path, default=None,
        help="Path to a .npz with arrays X/edges (and optional coords).",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output PNG path.")
    parser.add_argument("--title", type=str, default="Cross-method consensus co-association (no ground truth)")
    parser.add_argument(
        "--example", action="store_true",
        help="Render from a built-in synthetic slice.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.example:
        data, coords = _example_consensus()
    elif args.npz is not None:
        npz = np.load(args.npz)
        data = compute_consensus_coassociation(npz["X"], npz["edges"])
        coords = npz["coords"] if "coords" in npz.files else None
    else:
        raise SystemExit("provide --npz PATH.npz (X/edges[/coords]) or --example.")
    render_consensus_coassociation(data, args.out, coords=coords, title=args.title)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
