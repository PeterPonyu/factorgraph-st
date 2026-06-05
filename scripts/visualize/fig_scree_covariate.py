#!/usr/bin/env python
"""Factor scree + factor-vs-covariate association map for FactorGraph-ST (#325).

Two diagnostics over a fitted factor-score matrix ``H`` (``n_spots x n_factors``,
the per-spot scores from the projection decoder or the trained GNMF), rendered
side by side:

  * a **variance-explained scree**: the fraction of total score variance carried
    by each factor (sorted descending), the usual "how many factors actually
    matter" read-out; and
  * a **factor-vs-covariate association map**: for each factor and each supplied
    categorical covariate (e.g. ``section_id`` for a batch/nuisance axis, or a
    domain label for a biological axis) the correlation ratio ``eta^2`` computed
    by :func:`factorgraph_st.eval.metrics.factor_covariate_association`. This
    exposes factors that merely encode batch/section vs factors that track the
    domain structure.

The data helpers are matplotlib-free (covered in the numpy-only env); matplotlib
is imported lazily inside :func:`render_scree_covariate`.

Usage (under the project conda env)::

    conda run --no-capture-output -n dl python \
        scripts/visualize/fig_scree_covariate.py --example \
        --out /tmp/factorgraph_scree_covariate.png
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from factorgraph_st.eval.metrics import factor_covariate_association


def variance_explained(scores: np.ndarray) -> np.ndarray:
    """Per-factor fraction of total score variance (original factor order).

    Returns a length-``n_factors`` array that sums to ``1.0`` (or all-zeros when
    every factor is constant). The renderer sorts this for the scree display
    while keeping the factor labels.
    """
    S = np.asarray(scores, dtype=np.float64)
    if S.ndim != 2:
        raise ValueError(f"scores must be 2D (n_spots, n_factors); got shape {S.shape}")
    var = S.var(axis=0)
    total = float(var.sum())
    if total <= 0.0:
        return np.zeros(S.shape[1], dtype=np.float64)
    return var / total


def covariate_association_matrix(
    scores: np.ndarray,
    covariates: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, list[str]]:
    """Build the ``(n_factors, n_covariates)`` ``eta^2`` association matrix.

    Column ``c`` is the per-factor ``eta_sq`` from
    :func:`factor_covariate_association` of ``scores`` against
    ``covariates[name]``. Constant factors yield ``nan`` (not evaluable), which
    the renderer masks. Returns ``(matrix, names)`` with ``names`` in the
    iteration order of ``covariates``.
    """
    if not covariates:
        raise ValueError("covariates must be a non-empty mapping name -> label array")
    S = np.asarray(scores, dtype=np.float64)
    if S.ndim != 2:
        raise ValueError(f"scores must be 2D (n_spots, n_factors); got shape {S.shape}")
    names = list(covariates)
    matrix = np.full((S.shape[1], len(names)), np.nan, dtype=np.float64)
    for c, name in enumerate(names):
        result = factor_covariate_association(S, np.asarray(covariates[name]))
        matrix[:, c] = np.asarray(result["eta_sq"], dtype=np.float64)
    return matrix, names


def render_scree_covariate(
    scores: np.ndarray,
    covariates: Mapping[str, np.ndarray],
    out_path: str | Path,
    *,
    factor_names: Sequence[str] | None = None,
    title: str = "Factor scree + covariate association",
    dpi: int = 150,
):
    """Render the scree + ``eta^2`` association map; save to ``out_path``.

    matplotlib is imported lazily. Returns the
    :class:`matplotlib.figure.Figure` (axes inspectable for tests).
    """
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")  # headless, deterministic raster backend
    import matplotlib.pyplot as plt  # noqa: PLC0415

    S = np.asarray(scores, dtype=np.float64)
    if S.ndim != 2 or S.size == 0:
        raise ValueError(f"scores must be a non-empty 2D array; got shape {S.shape}")
    n_factors = S.shape[1]
    fac_labels = list(factor_names) if factor_names is not None else [f"F{j}" for j in range(n_factors)]
    if len(fac_labels) != n_factors:
        raise ValueError(f"factor_names has {len(fac_labels)} entries but scores has {n_factors} factors")

    frac = variance_explained(S)
    order = np.argsort(-frac, kind="stable")
    assoc, cov_names = covariate_association_matrix(S, covariates)

    fig, (ax_scree, ax_assoc) = plt.subplots(
        1, 2, figsize=(2.0 * n_factors + 5.0, 4.5),
        gridspec_kw={"width_ratios": [1.2, 1.0]},
    )

    # Scree: sorted variance-explained bars + cumulative line.
    x = np.arange(n_factors)
    ax_scree.bar(x, frac[order], color="#0072B2", edgecolor="black", linewidth=0.5)
    ax_scree.plot(x, np.cumsum(frac[order]), color="#D55E00", marker="o", markersize=4, label="cumulative")
    ax_scree.set_xticks(x)
    ax_scree.set_xticklabels([fac_labels[i] for i in order], rotation=0)
    ax_scree.set_ylabel("variance explained (fraction)")
    ax_scree.set_title("scree (sorted)")
    ax_scree.set_ylim(0.0, 1.05)
    ax_scree.legend(loc="center right", frameon=False, fontsize=8)
    ax_scree.spines["top"].set_visible(False)
    ax_scree.spines["right"].set_visible(False)

    # Association map: factors (rows) x covariates (cols), masking nan.
    masked = np.ma.masked_invalid(assoc)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="0.85")
    im = ax_assoc.imshow(masked, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    ax_assoc.set_xticks(range(len(cov_names)))
    ax_assoc.set_xticklabels(cov_names, rotation=30, ha="right")
    ax_assoc.set_yticks(range(n_factors))
    ax_assoc.set_yticklabels(fac_labels)
    ax_assoc.set_title(r"association ($\eta^2$)")
    for r in range(n_factors):
        for c in range(len(cov_names)):
            v = assoc[r, c]
            txt = "n/a" if not np.isfinite(v) else f"{v:.2f}"
            ax_assoc.text(c, r, txt, ha="center", va="center", fontsize=7, color="white")
    fig.colorbar(im, ax=ax_assoc, fraction=0.046, pad=0.04, label=r"$\eta^2$")

    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return fig


def _example_inputs(seed: int = 0) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Illustrative ``H`` + covariates with a known association structure.

    Factor 0 tracks ``section`` (batch), factor 1 tracks ``domain`` (biology),
    later factors are near-noise — so the scree decays and the association map
    shows one strong cell per structured factor.
    """
    rng = np.random.default_rng(seed)
    n_spots, n_factors = 120, 5
    section = np.repeat(np.arange(3), n_spots // 3)[:n_spots]
    domain = np.tile(np.arange(4), n_spots // 4 + 1)[:n_spots]
    H = rng.normal(0.0, 0.1, size=(n_spots, n_factors))
    H[:, 0] += section.astype(np.float64) * 2.0          # batch-driven factor
    H[:, 1] += domain.astype(np.float64) * 1.5           # domain-driven factor
    H = np.clip(H, 0.0, None).astype(np.float32)
    return H, {"section": section, "domain": domain}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=None, help="Path to a .npy factor-score matrix H.")
    parser.add_argument("--section", type=Path, default=None, help="Path to a .npy section_id covariate.")
    parser.add_argument("--domain", type=Path, default=None, help="Path to a .npy domain_id covariate.")
    parser.add_argument("--out", type=Path, required=True, help="Output PNG path.")
    parser.add_argument("--title", type=str, default="Factor scree + covariate association")
    parser.add_argument("--example", action="store_true", help="Render from built-in illustrative inputs.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.example:
        scores, covariates = _example_inputs()
    elif args.scores is not None:
        scores = np.load(args.scores)
        covariates = {
            name: np.load(path)
            for name, path in (("section", args.section), ("domain", args.domain))
            if path is not None
        }
        if not covariates:
            raise SystemExit("provide at least one of --section/--domain (or --example).")
    else:
        raise SystemExit("provide --scores PATH.npy with covariates (or --example).")
    render_scree_covariate(scores, covariates, args.out, title=args.title)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
