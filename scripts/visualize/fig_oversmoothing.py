#!/usr/bin/env python
"""Over-smoothing exposure trade-off for FactorGraph-ST (#331).

Spatial smoothing (the GNMF graph-Laplacian penalty ``lam``) trades two things
against each other: turning ``lam`` up makes the factor scores more spatially
**coherent** (neighbors agree) but past a point it washes out the real
**within-spot expression signal** (the reconstruction ``H @ W.T`` stops tracking
``X``). This figure exposes that trade-off explicitly so a high coherence number
can't be sold without showing what it cost.

It renders, across a sweep of regularization strengths ``lam``:

  * left — both metrics vs ``lam`` on a twin axis (coherence rising, signal
    preservation falling), and
  * right — the parametric **trade-off frontier** (signal preservation on x,
    spatial coherence on y) with each point annotated by its ``lam``.

:func:`compute_oversmoothing_sweep` runs the actual GNMF sweep on supplied data
(numpy-only, no network) and is exercised on a tiny synthetic instance in tests;
matplotlib is imported lazily in :func:`render_oversmoothing`.

Usage (under the project conda env)::

    conda run --no-capture-output -n dl python \
        scripts/visualize/fig_oversmoothing.py --example \
        --out /tmp/factorgraph_oversmoothing.png
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from factorgraph_st.eval.metrics import morans_i, reconstruction_error
from factorgraph_st.model.learned import fit_gnmf


def compute_oversmoothing_sweep(
    X: np.ndarray,
    edges: np.ndarray,
    lams: Sequence[float],
    *,
    n_factors: int = 4,
    n_iter: int = 100,
    tol: float = 1e-4,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit GNMF at each ``lam`` and return ``(lams, coherence, signal)`` arrays.

    For each ``lam`` the model is fit and scored by two GT-free metrics:

    * ``coherence`` — mean Moran's I of the factor-score columns over ``edges``
      (spatial smoothness of ``H``), which rises with ``lam``; and
    * ``signal`` — ``max(0, 1 - reconstruction_error(X, H, W))``, the fraction of
      expression signal the factorization still reconstructs, which falls as
      over-smoothing degrades the fit.

    All three returned arrays are aligned and ordered as ``lams`` was given.
    """
    if len(lams) == 0:
        raise ValueError("lams must be non-empty")
    lam_arr = np.asarray(lams, dtype=np.float64)
    coherence = np.empty(lam_arr.size, dtype=np.float64)
    signal = np.empty(lam_arr.size, dtype=np.float64)
    for i, lam in enumerate(lam_arr.tolist()):
        result = fit_gnmf(X, edges, n_factors, lam=lam, n_iter=n_iter, tol=tol, seed=seed)
        H, W = result.H, result.W
        per_factor = [morans_i(H[:, j].astype(np.float64), edges) for j in range(H.shape[1])]
        coherence[i] = float(np.mean(per_factor)) if per_factor else 0.0
        signal[i] = max(0.0, 1.0 - reconstruction_error(X, H, W))
    return lam_arr, coherence, signal


def render_oversmoothing(
    lams: np.ndarray,
    coherence: np.ndarray,
    signal: np.ndarray,
    out_path: str | Path,
    *,
    title: str = "Over-smoothing exposure (coherence vs signal preservation)",
    dpi: int = 150,
):
    """Render the twin-axis sweep + trade-off frontier; save to ``out_path``.

    matplotlib is imported lazily. Returns the
    :class:`matplotlib.figure.Figure` (axes inspectable for tests).
    """
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")  # headless, deterministic raster backend
    import matplotlib.pyplot as plt  # noqa: PLC0415

    lam_arr = np.asarray(lams, dtype=np.float64)
    coh = np.asarray(coherence, dtype=np.float64)
    sig = np.asarray(signal, dtype=np.float64)
    if not (lam_arr.size == coh.size == sig.size) or lam_arr.size == 0:
        raise ValueError("lams, coherence and signal must be non-empty arrays of equal length")

    fig, (ax_sweep, ax_front) = plt.subplots(1, 2, figsize=(11.0, 4.5))

    # Left: both metrics vs lam (twin y-axis).
    ax_coh = ax_sweep
    ax_sig = ax_coh.twinx()
    (l_coh,) = ax_coh.plot(lam_arr, coh, color="#0072B2", marker="o", markersize=4, label="spatial coherence")
    (l_sig,) = ax_sig.plot(lam_arr, sig, color="#D55E00", marker="s", markersize=4, label="signal preservation")
    ax_coh.set_xlabel(r"graph-regularization $\lambda$")
    ax_coh.set_ylabel("spatial coherence (mean Moran's I)", color="#0072B2")
    ax_sig.set_ylabel("signal preservation (1 - recon err)", color="#D55E00")
    ax_coh.tick_params(axis="y", labelcolor="#0072B2")
    ax_sig.tick_params(axis="y", labelcolor="#D55E00")
    ax_coh.set_title("metrics vs smoothing strength")
    ax_coh.legend(handles=[l_coh, l_sig], loc="center right", frameon=False, fontsize=8)

    # Right: parametric trade-off frontier (signal x, coherence y).
    ax_front.plot(sig, coh, color="0.4", linestyle="-", linewidth=1.0, zorder=1)
    sc = ax_front.scatter(sig, coh, c=lam_arr, cmap="viridis", s=70, edgecolor="black", linewidth=0.6, zorder=2)
    for x, y, lam in zip(sig.tolist(), coh.tolist(), lam_arr.tolist(), strict=True):
        ax_front.annotate(f"λ={lam:g}", (x, y), textcoords="offset points", xytext=(6, 4), fontsize=7)
    ax_front.set_xlabel("signal preservation (1 - recon err)")
    ax_front.set_ylabel("spatial coherence (mean Moran's I)")
    ax_front.set_title("trade-off frontier")
    ax_front.spines["top"].set_visible(False)
    ax_front.spines["right"].set_visible(False)
    fig.colorbar(sc, ax=ax_front, fraction=0.046, pad=0.04, label=r"$\lambda$")

    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return fig


def _example_sweep() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Illustrative monotone trade-off: coherence up, signal down with ``lam``."""
    lams = np.array([0.0, 0.5, 1.0, 2.0, 5.0, 10.0], dtype=np.float64)
    coherence = np.array([0.18, 0.34, 0.47, 0.61, 0.78, 0.88], dtype=np.float64)
    signal = np.array([0.92, 0.88, 0.81, 0.69, 0.48, 0.27], dtype=np.float64)
    return lams, coherence, signal


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path, default=None, help="Path to a .npz with arrays lams/coherence/signal.")
    parser.add_argument("--out", type=Path, required=True, help="Output PNG path.")
    parser.add_argument("--title", type=str, default="Over-smoothing exposure (coherence vs signal preservation)")
    parser.add_argument("--example", action="store_true", help="Render from a built-in illustrative sweep.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.example:
        lams, coherence, signal = _example_sweep()
    elif args.sweep is not None:
        data = np.load(args.sweep)
        lams, coherence, signal = data["lams"], data["coherence"], data["signal"]
    else:
        raise SystemExit("provide --sweep PATH.npz (lams/coherence/signal) or --example.")
    render_oversmoothing(lams, coherence, signal, args.out, title=args.title)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
