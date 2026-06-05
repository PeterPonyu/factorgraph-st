#!/usr/bin/env python
"""Shared-vs-section-specific factor decomposition map for FactorGraph-ST (#317).

A multi-section factor model should split its latent programs into two kinds:

  * **shared** factors that carry activation mass across *all* sections (a
    biological program common to every slice), and
  * **section-specific (private)** factors whose mass concentrates in a *strict
    subset* of sections (a slice-local program / batch axis).

This figure renders the per-factor section-mass decomposition as two heatmaps
(rows = factors, columns = sections, cell = fraction of that factor's total
activation mass falling in that section) and annotates the panel with the
shared/private *separation* summary — the mean number of sections each kind of
factor is active in (:func:`factorgraph_st.eval.metrics.shared_private_separation`).
A clean decomposition shows shared rows spread evenly across all columns and
private rows lit in only one (or few) columns.

:func:`compute_shared_private_map` is numpy-only (no network, no plotting) and is
exercised on a tiny synthetic multi-section instance in tests; matplotlib is
imported lazily inside :func:`render_shared_private_map`.

Usage (under the project conda env)::

    conda run --no-capture-output -n dl python \
        scripts/visualize/fig_shared_private_map.py --example \
        --out /tmp/factorgraph_shared_private_map.png
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from factorgraph_st.eval.metrics import section_overlap, shared_private_separation


@dataclass
class SharedPrivateMap:
    """Section-mass decomposition of shared and private factors.

    Attributes
    ----------
    shared_overlap:
        ``(K_shared, n_sections)`` per-factor section-mass fractions (rows sum
        to ~1) for the shared factors.
    private_overlap:
        ``(K_private, n_sections)`` per-factor section-mass fractions for the
        section-specific (private) factors.
    sections:
        ``(n_sections,)`` sorted unique section ids labelling the columns.
    separation:
        Summary dict from :func:`shared_private_separation` with keys
        ``shared_mean_active_sections`` and ``private_mean_active_sections``.
    """

    shared_overlap: np.ndarray
    private_overlap: np.ndarray
    sections: np.ndarray
    separation: dict[str, float]


def compute_shared_private_map(
    Z_shared: np.ndarray, Z_private: np.ndarray, section_id: np.ndarray
) -> SharedPrivateMap:
    """Decompose shared/private factor activations into per-section mass maps.

    ``Z_shared`` is ``(n_spots, K_shared)`` and ``Z_private`` is
    ``(n_spots, K_private)`` nonnegative activation matrices; ``section_id`` is
    the ``(n_spots,)`` section index per spot. Returns a :class:`SharedPrivateMap`
    holding the two section-mass-fraction heatmaps, the column section ids, and
    the shared/private separation summary. No randomness, no network.
    """
    Zs = np.asarray(Z_shared, dtype=np.float64)
    Zp = np.asarray(Z_private, dtype=np.float64)
    if Zs.ndim != 2 or Zp.ndim != 2:
        raise ValueError("Z_shared and Z_private must both be 2D")
    sections = np.unique(np.asarray(section_id))
    shared_overlap = section_overlap(Zs, section_id)
    private_overlap = section_overlap(Zp, section_id)
    separation = shared_private_separation(Zs, Zp, section_id)
    return SharedPrivateMap(
        shared_overlap=shared_overlap,
        private_overlap=private_overlap,
        sections=sections,
        separation=separation,
    )


def render_shared_private_map(
    data: SharedPrivateMap,
    out_path: str | Path,
    *,
    title: str = "Shared vs section-specific factor decomposition",
    dpi: int = 150,
):
    """Render the two section-mass heatmaps + separation annotation; save to disk.

    matplotlib is imported lazily. Returns the
    :class:`matplotlib.figure.Figure` (axes inspectable for tests).
    """
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")  # headless, deterministic raster backend
    import matplotlib.pyplot as plt  # noqa: PLC0415

    shared = np.asarray(data.shared_overlap, dtype=np.float64)
    private = np.asarray(data.private_overlap, dtype=np.float64)
    section_labels = [str(int(s)) for s in np.asarray(data.sections).tolist()]
    n_sections = len(section_labels)

    fig, (ax_shared, ax_private) = plt.subplots(
        1, 2, figsize=(2.2 * max(n_sections, 1) + 3.0, 4.8), layout="constrained"
    )

    last_im = None
    for ax, mat, name in (
        (ax_shared, shared, "shared"),
        (ax_private, private, "section-specific"),
    ):
        if mat.size == 0:
            ax.text(0.5, 0.5, f"no {name} factors", ha="center", va="center", fontsize=10)
            ax.set_axis_off()
            continue
        im = ax.imshow(mat, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
        last_im = im
        ax.set_xticks(range(n_sections))
        ax.set_xticklabels(section_labels)
        ax.set_yticks(range(mat.shape[0]))
        ax.set_yticklabels([f"f{i}" for i in range(mat.shape[0])])
        ax.set_xlabel("section")
        ax.set_title(f"{name} factors")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                ax.annotate(
                    f"{val:.2f}",
                    (j, i),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if val < 0.6 else "black",
                )
    ax_shared.set_ylabel("factor")

    if last_im is not None:
        fig.colorbar(last_im, ax=(ax_shared, ax_private), fraction=0.046, pad=0.04, label="section mass fraction")

    sep = data.separation
    subtitle = (
        f"mean active sections — shared: {sep.get('shared_mean_active_sections', float('nan')):.2f}"
        f"   private: {sep.get('private_mean_active_sections', float('nan')):.2f}"
    )
    fig.suptitle(f"{title}\n{subtitle}", fontsize=11)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return fig


def _example_map() -> SharedPrivateMap:
    """Compute the decomposition map on a small synthetic multi-section instance."""
    from factorgraph_st.synth import generate_instance  # noqa: PLC0415

    inst = generate_instance(
        n_sections=3,
        n_spots_per_section=40,
        n_genes=24,
        K_shared=3,
        K_private=2,
        n_domains=4,
        k_nn=6,
        seed=0,
    )
    return compute_shared_private_map(inst.Z_shared, inst.Z_private, inst.section_id)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--npz",
        type=Path,
        default=None,
        help="Path to a .npz with arrays Z_shared/Z_private/section_id.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output PNG path.")
    parser.add_argument("--title", type=str, default="Shared vs section-specific factor decomposition")
    parser.add_argument(
        "--example",
        action="store_true",
        help="Render from a built-in synthetic multi-section instance.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.example:
        data = _example_map()
    elif args.npz is not None:
        npz = np.load(args.npz)
        data = compute_shared_private_map(npz["Z_shared"], npz["Z_private"], npz["section_id"])
    else:
        raise SystemExit("provide --npz PATH.npz (Z_shared/Z_private/section_id) or --example.")
    render_shared_private_map(data, args.out, title=args.title)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
