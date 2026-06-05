#!/usr/bin/env python
"""Qualitative multi-section integration embedding panel for FactorGraph-ST (#323).

When several sections are factored *jointly*, a good integration mixes spots from
different sections while keeping biological spatial domains apart. This figure
shows that qualitatively with a 2x2 grid of 2-D embeddings:

  * **before** (top row) — PCA of the raw expression matrix ``X``. Sections are
    NOT yet integrated, so coloring by section typically shows section-separated
    clouds (a batch axis).
  * **after** (bottom row) — PCA of the joint graph-regularized NMF factor scores
    ``H``. A successful integration *mixes* the section colors here while the
    domain colors stay separated.

Each row is shown twice, colored by **section** (left) and by **ground-truth
domain** (right): good integration = sections become well mixed (left column)
between top and bottom while domains stay/become well separated (right column).

:func:`compute_integration_embedding` is numpy-only (no network, no plotting) and
runs on a tiny synthetic multi-section instance in tests; matplotlib is imported
lazily inside :func:`render_integration_embedding`.

Usage (under the project conda env)::

    conda run --no-capture-output -n dl python \
        scripts/visualize/fig_integration_embedding.py --example \
        --out /tmp/factorgraph_integration_embedding.png
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from factorgraph_st.model.learned import fit_gnmf


@dataclass
class IntegrationEmbedding:
    """Before/after 2-D embeddings with section and domain labels.

    Attributes
    ----------
    before:
        ``(n_spots, 2)`` PCA embedding of the raw expression matrix (pre-integration).
    after:
        ``(n_spots, 2)`` PCA embedding of the joint factor scores (post-integration).
    section_id:
        ``(n_spots,)`` section index per spot (the batch axis to be mixed).
    domain_id:
        ``(n_spots,)`` ground-truth spatial domain per spot (to stay separated).
    """

    before: np.ndarray
    after: np.ndarray
    section_id: np.ndarray
    domain_id: np.ndarray


def _pca_2d(M: np.ndarray) -> np.ndarray:
    """Center ``M`` and project onto its top-2 principal components (numpy-only).

    Returns an ``(n, 2)`` float64 embedding. If fewer than two components exist
    (degenerate input), the missing column(s) are zero-padded so the result is
    always 2-D.
    """
    A = np.asarray(M, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError("M must be 2D")
    n = A.shape[0]
    if n == 0:
        return np.zeros((0, 2), dtype=np.float64)
    Ac = A - A.mean(axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(Ac, full_matrices=False)
    k = min(2, vt.shape[0])
    emb = Ac @ vt[:k].T
    if emb.shape[1] < 2:
        emb = np.column_stack([emb, np.zeros((n, 2 - emb.shape[1]), dtype=np.float64)])
    return emb.astype(np.float64)


def compute_integration_embedding(
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
) -> IntegrationEmbedding:
    """Compute before (raw-X PCA) vs after (joint-factor PCA) 2-D embeddings.

    Fits graph-regularized NMF jointly on all sections, then embeds both the raw
    expression and the learned factor scores into 2-D via PCA. Returns an
    :class:`IntegrationEmbedding`. Seeded and deterministic; no network.
    """
    X = np.asarray(X, dtype=np.float64)
    section_id = np.asarray(section_id)
    domain_id = np.asarray(domain_id)
    result = fit_gnmf(X, edges, n_factors, lam=lam, n_iter=n_iter, tol=tol, seed=seed)
    before = _pca_2d(X)
    after = _pca_2d(result.H.astype(np.float64))
    return IntegrationEmbedding(
        before=before, after=after, section_id=section_id, domain_id=domain_id
    )


def render_integration_embedding(
    data: IntegrationEmbedding,
    out_path: str | Path,
    *,
    title: str = "Multi-section integration embedding (before vs after)",
    dpi: int = 150,
):
    """Render the 2x2 before/after x section/domain scatter grid; save to disk.

    matplotlib is imported lazily. Returns the
    :class:`matplotlib.figure.Figure` (four axes inspectable for tests).
    """
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")  # headless, deterministic raster backend
    import matplotlib.pyplot as plt  # noqa: PLC0415

    before = np.asarray(data.before, dtype=np.float64)
    after = np.asarray(data.after, dtype=np.float64)
    section_id = np.asarray(data.section_id)
    domain_id = np.asarray(data.domain_id)
    n = before.shape[0]
    if not (before.shape == after.shape) or before.shape[1] != 2:
        raise ValueError("before and after must be matching (n, 2) embeddings")
    if not (section_id.shape[0] == domain_id.shape[0] == n):
        raise ValueError("section_id and domain_id must align with the embeddings")

    palette = ["#56B4E9", "#E69F00", "#009E73", "#CC79A7", "#0072B2", "#D55E00", "#F0E442", "#999999"]

    def _scatter(ax, emb, labels, legend_title):
        uniq = np.unique(labels)
        for i, lab in enumerate(uniq.tolist()):
            mask = labels == lab
            ax.scatter(
                emb[mask, 0], emb[mask, 1],
                s=10, color=palette[i % len(palette)], edgecolor="none",
                alpha=0.8, label=str(int(lab)),
            )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(title=legend_title, loc="best", fontsize=6, title_fontsize=7, frameon=False, markerscale=1.2)

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 8.5))
    _scatter(axes[0, 0], before, section_id, "section")
    _scatter(axes[0, 1], before, domain_id, "domain")
    _scatter(axes[1, 0], after, section_id, "section")
    _scatter(axes[1, 1], after, domain_id, "domain")

    axes[0, 0].set_title("before — by section (batch axis)")
    axes[0, 1].set_title("before — by domain")
    axes[1, 0].set_title("after — by section (mixed = good)")
    axes[1, 1].set_title("after — by domain (separated = good)")
    axes[0, 0].set_ylabel("raw expression PCA", fontsize=10)
    axes[1, 0].set_ylabel("joint factor PCA", fontsize=10)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return fig


def _example_embedding() -> IntegrationEmbedding:
    """Compute the integration embedding on a small synthetic multi-section instance."""
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
    return compute_integration_embedding(
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
    parser.add_argument("--title", type=str, default="Multi-section integration embedding (before vs after)")
    parser.add_argument(
        "--example", action="store_true",
        help="Render from a built-in synthetic multi-section instance.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.example:
        data = _example_embedding()
    elif args.npz is not None:
        npz = np.load(args.npz)
        data = compute_integration_embedding(npz["X"], npz["edges"], npz["section_id"], npz["domain_id"])
    else:
        raise SystemExit("provide --npz PATH.npz (X/edges/section_id/domain_id) or --example.")
    render_integration_embedding(data, args.out, title=args.title)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
