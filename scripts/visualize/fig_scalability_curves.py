#!/usr/bin/env python
"""#309 — runtime + peak-memory scalability CURVES vs #spots and #genes.

The companion of the #336 scalability *table* (``scripts/tables/table_scalability.py``):
this script reuses that table's measurement helper (``measure_scalability`` —
imported, never re-implemented) and turns the measured rows into a publication
figure of two sweeps:

  * **vs number of spots** (genes held fixed) — how runtime and peak resident
    memory grow as the section size increases.
  * **vs number of genes** (spots held fixed) — the same two costs as the
    feature dimension increases.

Each sweep is one panel with a twin y-axis: wall-clock **runtime** (left axis,
``time.perf_counter`` via the shared helper) and **peak RSS** (right axis,
``resource.getrusage``). ``resource`` is POSIX-only; on a platform without it
the memory series is all-``nan`` and the panel annotates "peak RSS n/a" rather
than plotting a fabricated curve.

The numpy-only data helpers (:func:`sweep_from_records`, :func:`build_curves`)
carry no plotting or network dependency and run on a handful of in-memory rows;
matplotlib is imported lazily inside :func:`render_scalability_curves`.

Usage (under the project conda env)::

    conda run --no-capture-output -n dl python \
        scripts/visualize/fig_scalability_curves.py --measure \
        --out /tmp/factorgraph_scalability_curves.png
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
for _p in (str(_REPO / "src"), str(_REPO / "scripts" / "tables")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from table_scalability import measure_scalability  # noqa: E402


@dataclass
class Sweep:
    """One swept dimension of the scalability study.

    Attributes
    ----------
    vary:
        The dimension varied across this sweep, ``"n_spots"`` or ``"n_genes"``.
    x:
        ``(n_points,)`` int64 values of the varied dimension, ascending.
    runtime_s:
        ``(n_points,)`` float64 wall-clock runtime per size (seconds).
    peak_rss_mb:
        ``(n_points,)`` float64 peak resident memory per size (MiB); ``nan``
        where ``resource`` is unavailable.
    fixed_label:
        Human-readable note of the dimension held fixed, e.g. ``"n_genes=40"``.
    """

    vary: str
    x: np.ndarray
    runtime_s: np.ndarray
    peak_rss_mb: np.ndarray
    fixed_label: str


@dataclass
class ScalabilityCurves:
    """The two sweeps that make up the scalability figure."""

    spots: Sweep
    genes: Sweep


def sweep_from_records(
    records: Sequence[Mapping[str, float]], vary: str, fixed_label: str
) -> Sweep:
    """Turn measurement rows into a :class:`Sweep`, sorted ascending by ``vary``.

    Each record needs ``vary`` plus ``runtime_s`` / ``peak_rss_mb``. Rows are
    sorted by the varied dimension so the curve reads monotonically regardless
    of input order; a missing / ``None`` memory cell becomes ``nan`` (rendered
    as "n/a"), never a fabricated number.
    """
    if vary not in ("n_spots", "n_genes"):
        raise ValueError("vary must be 'n_spots' or 'n_genes'")
    rows = sorted(records, key=lambda r: int(r[vary]))
    x = np.array([int(r[vary]) for r in rows], dtype=np.int64)
    runtime = np.array([float(r["runtime_s"]) for r in rows], dtype=np.float64)

    def _rss(record: Mapping[str, float]) -> float:
        value = record.get("peak_rss_mb")
        return float("nan") if value is None else float(value)

    rss = np.array([_rss(r) for r in rows], dtype=np.float64)
    return Sweep(vary=vary, x=x, runtime_s=runtime, peak_rss_mb=rss, fixed_label=fixed_label)


def build_curves(
    spots_records: Sequence[Mapping[str, float]],
    genes_records: Sequence[Mapping[str, float]],
    *,
    spots_fixed_label: str,
    genes_fixed_label: str,
) -> ScalabilityCurves:
    """Assemble both sweeps (data-independent; pairs with real benchmark rows)."""
    return ScalabilityCurves(
        spots=sweep_from_records(spots_records, "n_spots", spots_fixed_label),
        genes=sweep_from_records(genes_records, "n_genes", genes_fixed_label),
    )


def measure_scalability_curves(
    spot_ladder: Sequence[int],
    gene_ladder: Sequence[int],
    *,
    n_genes_fixed: int,
    n_spots_per_section_fixed: int,
    n_sections: int = 2,
    n_iter: int = 20,
    seed: int = 0,
) -> ScalabilityCurves:
    """Run real tiny synthetic fits along both ladders; return the two sweeps.

    Delegates the actual timing/RSS measurement to the shared
    :func:`table_scalability.measure_scalability` so the figure and the #336
    table report the *same* numbers from the *same* code path. Deliberately
    small (tiny instances, small ``n_iter``) so it stays test-safe.
    """
    spots_records = measure_scalability(
        [(int(s), int(n_genes_fixed)) for s in spot_ladder],
        n_sections=n_sections,
        n_iter=n_iter,
        seed=seed,
    )
    genes_records = measure_scalability(
        [(int(n_spots_per_section_fixed), int(g)) for g in gene_ladder],
        n_sections=n_sections,
        n_iter=n_iter,
        seed=seed,
    )
    return build_curves(
        spots_records,
        genes_records,
        spots_fixed_label=f"n_genes={int(n_genes_fixed)}",
        genes_fixed_label=f"n_spots={int(n_sections * n_spots_per_section_fixed)}",
    )


_RUNTIME_COLOR = "#0072B2"  # blue   — runtime (left axis)
_MEMORY_COLOR = "#D55E00"   # orange — peak RSS (right axis)


def _plot_sweep(ax, sweep: Sweep, xlabel: str) -> None:
    """Draw one twin-axis panel: runtime (left) + peak RSS (right)."""
    ax.plot(
        sweep.x, sweep.runtime_s,
        marker="o", color=_RUNTIME_COLOR, linewidth=1.8, label="runtime",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("runtime (s)", color=_RUNTIME_COLOR)
    ax.tick_params(axis="y", labelcolor=_RUNTIME_COLOR)
    ax.spines["top"].set_visible(False)
    ax.margins(x=0.08)
    ax.set_title(f"{xlabel}  ({sweep.fixed_label})")

    ax2 = ax.twinx()
    finite = np.isfinite(sweep.peak_rss_mb)
    if finite.any():
        ax2.plot(
            sweep.x[finite], sweep.peak_rss_mb[finite],
            marker="s", color=_MEMORY_COLOR, linewidth=1.8,
            linestyle="--", label="peak RSS",
        )
        ax2.set_ylabel("peak RSS (MiB)", color=_MEMORY_COLOR)
        ax2.tick_params(axis="y", labelcolor=_MEMORY_COLOR)
    else:
        ax2.set_yticks([])
        ax2.annotate(
            "peak RSS n/a",
            xy=(0.97, 0.05), xycoords="axes fraction",
            ha="right", va="bottom", fontsize=8, color="0.4",
        )
    ax2.spines["top"].set_visible(False)


def render_scalability_curves(
    curves: ScalabilityCurves,
    out_path: str | Path,
    *,
    title: str = "FactorGraph-ST scalability (runtime + peak memory)",
    dpi: int = 150,
):
    """Render the two-panel runtime/memory scalability figure; save to disk.

    matplotlib is imported lazily. Returns the
    :class:`matplotlib.figure.Figure` (two panels, each with a twin y-axis, so
    tests can introspect ``fig.axes``).
    """
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")  # headless, deterministic raster backend
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    _plot_sweep(axes[0], curves.spots, "number of spots")
    _plot_sweep(axes[1], curves.genes, "number of genes")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return fig


def _example_curves() -> ScalabilityCurves:
    """Measure both sweeps on small synthetic instances (no labeled data)."""
    return measure_scalability_curves(
        spot_ladder=(40, 80, 120),
        gene_ladder=(30, 60, 90),
        n_genes_fixed=40,
        n_spots_per_section_fixed=40,
        n_sections=2,
        n_iter=30,
        seed=0,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Output PNG path.")
    parser.add_argument(
        "--title",
        type=str,
        default="FactorGraph-ST scalability (runtime + peak memory)",
        help="Figure title.",
    )
    parser.add_argument(
        "--measure",
        action="store_true",
        help="Run real tiny synthetic fits along both ladders (otherwise required).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.measure:
        raise SystemExit(
            "pass --measure to run the (tiny) synthetic fits that fill the curves."
        )
    curves = _example_curves()
    render_scalability_curves(curves, args.out, title=args.title)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
