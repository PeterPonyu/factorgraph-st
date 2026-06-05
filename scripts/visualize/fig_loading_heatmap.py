#!/usr/bin/env python
"""Factor-loading heatmap + ranked gene-program panel for FactorGraph-ST (#321).

Renders, from a nonnegative gene-loading matrix ``W`` (``n_genes x n_factors``,
the loadings emitted by either the projection decoder or the trained GNMF fit),
two side-by-side views of what each latent factor *is*, in gene-program terms:

  * a **loading heatmap** restricted to the union of the top genes per factor
    (so a 20k-gene panel stays legible), columns = factors, rows = genes; and
  * a **ranked per-factor gene-program panel** listing the top genes and their
    loadings for each factor, the textual "what genes drive factor k" read-out.

This is a pure VISUALIZATION concern over an already-fit ``W``; it never fits a
model. The data helpers (:func:`rank_top_genes`, :func:`top_gene_union`) are
matplotlib-free so they stay covered in the numpy-only test env, and matplotlib
is imported LAZILY inside :func:`render_loading_heatmap`.

Usage (under the project conda env)::

    conda run --no-capture-output -n dl python \
        scripts/visualize/fig_loading_heatmap.py --example \
        --out /tmp/factorgraph_loading_heatmap.png
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np


def _rank_indices(W: np.ndarray, top_n: int) -> list[np.ndarray]:
    """Return, per factor, the gene indices of its ``top_n`` largest loadings.

    Ties are broken by ascending gene index (a stable sort on the negated
    column) so the ranking is deterministic across runs and platforms.
    """
    Wf = np.asarray(W, dtype=np.float64)
    if Wf.ndim != 2:
        raise ValueError(f"W must be 2D (n_genes, n_factors); got shape {Wf.shape}")
    if top_n <= 0:
        raise ValueError(f"top_n must be positive; got {top_n}")
    n_genes, n_factors = Wf.shape
    k = min(top_n, n_genes)
    out: list[np.ndarray] = []
    for j in range(n_factors):
        order = np.argsort(-Wf[:, j], kind="stable")[:k]
        out.append(order.astype(np.int64))
    return out


def rank_top_genes(
    W: np.ndarray,
    top_n: int = 10,
    gene_names: Sequence[str] | None = None,
) -> list[list[tuple[str, float]]]:
    """Rank the top genes per factor as ``(gene_name, loading)`` pairs.

    Returns one list per factor (in factor order), each holding up to ``top_n``
    ``(name, loading)`` pairs sorted by descending loading. ``gene_names``
    defaults to ``g0, g1, ...`` and must match ``W``'s gene axis when supplied.
    """
    Wf = np.asarray(W, dtype=np.float64)
    n_genes = Wf.shape[0] if Wf.ndim == 2 else 0
    names = list(gene_names) if gene_names is not None else [f"g{i}" for i in range(n_genes)]
    if len(names) != n_genes:
        raise ValueError(f"gene_names has {len(names)} entries but W has {n_genes} genes")
    ranked: list[list[tuple[str, float]]] = []
    for j, idx in enumerate(_rank_indices(Wf, top_n)):
        ranked.append([(names[int(i)], float(Wf[int(i), j])) for i in idx])
    return ranked


def top_gene_union(W: np.ndarray, top_n: int = 10) -> np.ndarray:
    """Ordered union of the top-``top_n`` gene indices across all factors.

    Genes are emitted in first-seen order scanning factor 0, 1, ... so the
    heatmap rows group by the factor that first nominated each gene.
    """
    seen: dict[int, None] = {}
    for idx in _rank_indices(W, top_n):
        for i in idx.tolist():
            seen.setdefault(int(i), None)
    return np.array(list(seen), dtype=np.int64)


def render_loading_heatmap(
    W: np.ndarray,
    out_path: str | Path,
    *,
    gene_names: Sequence[str] | None = None,
    factor_names: Sequence[str] | None = None,
    top_n: int = 10,
    title: str = "Factor gene-loadings (top genes per factor)",
    dpi: int = 150,
):
    """Render the loading heatmap + ranked gene-program panel; save to ``out_path``.

    matplotlib is imported lazily so importing this module stays dependency-free.
    Returns the :class:`matplotlib.figure.Figure` (axes inspectable for tests).
    """
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")  # headless, deterministic raster backend
    import matplotlib.pyplot as plt  # noqa: PLC0415

    Wf = np.asarray(W, dtype=np.float64)
    if Wf.ndim != 2 or Wf.size == 0:
        raise ValueError(f"W must be a non-empty 2D array; got shape {Wf.shape}")
    n_genes, n_factors = Wf.shape
    names = list(gene_names) if gene_names is not None else [f"g{i}" for i in range(n_genes)]
    fac_labels = list(factor_names) if factor_names is not None else [f"F{j}" for j in range(n_factors)]
    if len(fac_labels) != n_factors:
        raise ValueError(f"factor_names has {len(fac_labels)} entries but W has {n_factors} factors")

    union = top_gene_union(Wf, top_n)
    sub = Wf[union, :]
    ranked = rank_top_genes(Wf, top_n, names)

    fig, (ax_heat, ax_text) = plt.subplots(
        1, 2, figsize=(2.0 * n_factors + 6.0, 0.32 * len(union) + 2.5),
        gridspec_kw={"width_ratios": [1.4, 1.0]},
    )

    im = ax_heat.imshow(sub, aspect="auto", cmap="magma")
    ax_heat.set_xticks(range(n_factors))
    ax_heat.set_xticklabels(fac_labels)
    ax_heat.set_yticks(range(len(union)))
    ax_heat.set_yticklabels([names[int(i)] for i in union], fontsize=7)
    ax_heat.set_xlabel("factor")
    ax_heat.set_title("loading heatmap (top-gene union)")
    fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04, label="loading")

    # Ranked per-factor gene-program panel as left-aligned text columns.
    ax_text.axis("off")
    ax_text.set_title("ranked gene programs")
    col_w = 1.0 / max(n_factors, 1)
    for j, factor in enumerate(ranked):
        x = j * col_w
        ax_text.text(x, 1.0, fac_labels[j], fontsize=8, fontweight="bold", va="top", transform=ax_text.transAxes)
        for r, (gene, load) in enumerate(factor):
            ax_text.text(
                x, 0.94 - r * 0.045, f"{gene}: {load:.2f}",
                fontsize=6.5, va="top", family="monospace", transform=ax_text.transAxes,
            )

    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return fig


def _example_loadings(seed: int = 0) -> np.ndarray:
    """Illustrative nonnegative ``W`` with a clear per-factor gene program.

    Each factor loads strongly on a disjoint block of genes plus a low
    background, so the heatmap shows a clean block structure for visual review.
    """
    rng = np.random.default_rng(seed)
    n_genes, n_factors = 40, 4
    W = rng.exponential(scale=0.1, size=(n_genes, n_factors))
    block = n_genes // n_factors
    for j in range(n_factors):
        rows = slice(j * block, (j + 1) * block)
        W[rows, j] += rng.uniform(1.0, 2.0, size=block)
    return W.astype(np.float32)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loadings", type=Path, default=None, help="Path to a .npy gene-loading matrix W.")
    parser.add_argument("--out", type=Path, required=True, help="Output PNG path.")
    parser.add_argument("--top-n", type=int, default=10, help="Top genes per factor to rank/show.")
    parser.add_argument("--title", type=str, default="Factor gene-loadings (top genes per factor)")
    parser.add_argument("--example", action="store_true", help="Render from a built-in illustrative W.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.example:
        W = _example_loadings()
    elif args.loadings is not None:
        W = np.load(args.loadings)
    else:
        raise SystemExit("provide --loadings PATH.npy (or --example to render the illustrative panel).")
    render_loading_heatmap(W, args.out, top_n=args.top_n, title=args.title)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
