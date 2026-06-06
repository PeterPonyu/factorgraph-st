#!/usr/bin/env python
"""Agreement-vs-reference-complexity diagnostic scatter for FactorGraph-ST (#327).

One point per ``(method/variant, dataset)`` on two deliberately-orthogonal axes:

  * **x -- reference-label ambiguity**: a property of the REFERENCE annotation
    ALONE, with no method/prediction anywhere in it. We define it as the
    *negative* silhouette of the reference partition in expression-PCA space:
    higher silhouette = cleaner, better-separated reference labels =
    *less* ambiguous, so ambiguity ``= -silhouette``. Computed on the labeled
    subset only (NA tokens dropped), over a PCA reduction of the expression.
  * **y -- method-vs-reference ARI**: the supervised agreement (adjusted Rand
    index) between a method's predicted domains and the reference labels, read
    straight from the accuracy scorecards (#322/#330).

The scatter is the GT-skeptical diagnostic that explains WHY the raw accuracy
bars (#322) mislead: apparent accuracy largely tracks *label cleanliness*, not
method quality. A regression line + Pearson R over all points makes the trend
explicit; the expectation is a NEGATIVE correlation (methods score higher where
the reference labels are less ambiguous), so a headline ARI says as much about
the annotation as about the method.

HONESTY NOTE: ``x`` is a property of the reference partition only -- no
prediction enters it -- so the x/y axes are genuinely orthogonal and the
correlation is not circular. The silhouette-in-PCA-space ambiguity proxy is one
defensible operationalization, not the only one; and with only a handful of
datasets the Pearson R is *illustrative*, not conclusive (two datasets pin a
line exactly, R = +/-1 by construction). Nothing is fabricated: a dataset whose
ambiguity or ARI is undefined is dropped rather than zero-filled.

Data helpers (:func:`pearson_r`, :func:`regression_line`, :func:`assemble_points`)
are pure numpy/stdlib and matplotlib-free; matplotlib is imported lazily in
:func:`render`, and anndata only in :func:`reference_ambiguity`.

Usage (under the project conda env)::

    conda run --no-capture-output -n dl python \
        scripts/visualize/fig_agreement_vs_complexity.py \
        --accuracy results/starmap_wang2018/_scorecard/accuracy_results.json \
        --accuracy results/dlpfc_maynard/_scorecard/accuracy_results.json \
        --h5ad starmap_mouse_vcortex_wang2018=../data/processed/starmap_mouse_vcortex_wang2018/anndata.h5ad \
        --h5ad dlpfc_maynard_2021_visium=../data/processed/dlpfc_maynard_2021_visium/anndata.h5ad \
        --gt-obs-key ground_truth \
        --expert starmap_mouse_vcortex_wang2018 --expert dlpfc_maynard_2021_visium \
        --out results/cross_dataset/agreement_vs_complexity.png
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
for _p in (str(_REPO / "src"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

#: NA/blank GT tokens, byte-for-byte mirror of ``run_real_factorgraph._GT_NA_TOKENS``
#: so the labeled subset matches the runner's / collector's ARI exactly.
_GT_NA_TOKENS = frozenset({"", "na", "nan", "none", "unknown", "unlabeled", "unlabelled"})


def pearson_r(x: Sequence[float], y: Sequence[float]) -> float:
    """Pearson correlation coefficient between ``x`` and ``y`` (nan-safe).

    Returns ``float('nan')`` when fewer than two finite paired points are given
    or when either series has zero variance (correlation undefined) -- never a
    misleading ``0.0`` that reads as "measured no relationship".
    """
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if xa.shape != ya.shape:
        raise ValueError("x and y must have equal length")
    finite = np.isfinite(xa) & np.isfinite(ya)
    xa, ya = xa[finite], ya[finite]
    if xa.size < 2:
        return float("nan")
    xc = xa - xa.mean()
    yc = ya - ya.mean()
    denom = float(np.sqrt(np.sum(xc**2) * np.sum(yc**2)))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(xc * yc) / denom)


def regression_line(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    """Least-squares ``(slope, intercept)`` of ``y`` on ``x`` (nan-safe).

    Returns ``(nan, nan)`` when fewer than two finite paired points are given or
    when ``x`` has zero variance (slope undefined).
    """
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if xa.shape != ya.shape:
        raise ValueError("x and y must have equal length")
    finite = np.isfinite(xa) & np.isfinite(ya)
    xa, ya = xa[finite], ya[finite]
    if xa.size < 2:
        return float("nan"), float("nan")
    xc = xa - xa.mean()
    denom = float(np.sum(xc**2))
    if denom == 0.0:
        return float("nan"), float("nan")
    slope = float(np.sum(xc * (ya - ya.mean())) / denom)
    intercept = float(ya.mean() - slope * xa.mean())
    return slope, intercept


def assemble_points(
    accuracy_by_dataset: Mapping[str, Mapping[str, Mapping[str, Mapping[str, float]]]],
    ambiguity_by_dataset: Mapping[str, float],
    *,
    expert_datasets: Iterable[str] = (),
    ari_key: str = "ari",
) -> list[dict]:
    """Build one point per ``(dataset, variant)`` joining ambiguity (x) to ARI (y).

    ``accuracy_by_dataset`` maps a *source label* (e.g. a file id) to the
    ``accuracy_results.json`` schema ``dataset -> variant -> {ari, nmi, ami}``.
    ``ambiguity_by_dataset`` maps the dataset id to its reference-label ambiguity
    (the x value, computed with no method involved). Each output point is
    ``{dataset, variant, ambiguity, ari, expert}`` where ``expert`` is ``True``
    iff the dataset is in ``expert_datasets``.

    A ``(dataset, variant)`` whose dataset has no ambiguity entry, or whose ARI
    is missing/non-finite, is skipped (never zero-filled). Points are sorted by
    ``(dataset, variant)`` for deterministic rendering.
    """
    expert = set(expert_datasets)
    points: list[dict] = []
    for results in accuracy_by_dataset.values():
        for dataset, variants in results.items():
            if dataset not in ambiguity_by_dataset:
                continue
            ambiguity = float(ambiguity_by_dataset[dataset])
            if not np.isfinite(ambiguity):
                continue
            for variant, scores in variants.items():
                raw = scores.get(ari_key)
                if raw is None:
                    continue
                ari = float(raw)
                if not np.isfinite(ari):
                    continue
                points.append(
                    {
                        "dataset": dataset,
                        "variant": variant,
                        "ambiguity": ambiguity,
                        "ari": ari,
                        "expert": dataset in expert,
                    }
                )
    points.sort(key=lambda p: (p["dataset"], p["variant"]))
    return points


def render(
    points: Sequence[dict],
    out_path: str | Path,
    *,
    title: str = "Agreement vs reference-label complexity",
    dpi: int = 150,
):
    """Render the agreement-vs-complexity scatter; save to ``out_path``.

    matplotlib is imported lazily. Points are colored by variant; expert-
    annotated datasets are drawn with a distinct marker (square ``s`` vs circle
    ``o``). A least-squares regression line over ALL points plus the Pearson R is
    overlaid (annotation + title). Returns the :class:`matplotlib.figure.Figure`.
    """
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")  # headless, deterministic raster backend
    import matplotlib.pyplot as plt  # noqa: PLC0415

    palette = ["#56B4E9", "#E69F00", "#009E73", "#CC79A7", "#0072B2", "#D55E00", "#F0E442"]
    variants = sorted({str(p["variant"]) for p in points})
    color_of = {v: palette[i % len(palette)] for i, v in enumerate(variants)}

    xs = [float(p["ambiguity"]) for p in points]
    ys = [float(p["ari"]) for p in points]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    seen_variant: set[str] = set()
    seen_expert: set[bool] = set()
    for p in points:
        variant = str(p["variant"])
        expert = bool(p["expert"])
        x, y = float(p["ambiguity"]), float(p["ari"])
        marker = "s" if expert else "o"
        # Label each variant once (color legend); marker shape legend handled separately.
        label = variant if variant not in seen_variant else None
        seen_variant.add(variant)
        seen_expert.add(expert)
        ax.scatter(
            x, y, s=110, color=color_of[variant], edgecolor="black",
            linewidth=0.8, marker=marker, label=label, zorder=3,
        )
        ax.annotate(
            str(p["dataset"]), (x, y), textcoords="offset points",
            xytext=(7, 5), fontsize=6, color="0.35",
        )

    r = pearson_r(xs, ys)
    slope, intercept = regression_line(xs, ys)
    if np.isfinite(slope) and len(xs) >= 2:
        lo, hi = min(xs), max(xs)
        line_x = np.array([lo, hi], dtype=np.float64)
        ax.plot(
            line_x, slope * line_x + intercept, color="0.4", linestyle="--",
            linewidth=1.2, zorder=2, label="OLS fit",
        )
    r_txt = "n/a" if not np.isfinite(r) else f"{r:+.3f}"
    ax.text(
        0.02, 0.02, f"Pearson R = {r_txt}", transform=ax.transAxes,
        fontsize=9, color="0.2", va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="0.7", alpha=0.85),
    )

    ax.set_xlabel("reference-label ambiguity  (-silhouette of reference in PCA space)")
    ax.set_ylabel("method-vs-reference agreement  (ARI)")
    ax.set_title(f"{title}   (R = {r_txt})")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(0.15)
    if points:
        ax.legend(fontsize=7, frameon=False, loc="best")
        ax.text(
            0.98, 0.98, "square = expert-annotated dataset", transform=ax.transAxes,
            fontsize=7, color="0.4", va="top", ha="right", style="italic",
        )

    fig.tight_layout()

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return fig


def reference_ambiguity(
    h5ad_path: str | Path, gt_obs_key: str, *, n_comps: int = 50
) -> float:
    """Reference-label ambiguity for one dataset: ``-silhouette`` in expression-PCA space.

    Reads the AnnData (lazy anndata import), drops NA-token GT labels, reduces
    the labeled-subset expression to at most ``n_comps`` PCA components (numpy
    SVD on the column-centered dense matrix, or sklearn PCA if available), and
    returns the negative silhouette of the integer-coded REFERENCE labels over
    that embedding. NO prediction enters this value. Returns ``float('nan')``
    when the silhouette is undefined (fewer than two reference labels, etc.).
    """
    import anndata as ad  # noqa: PLC0415  (heavy, lazy)

    from factorgraph_st.eval.metrics import silhouette  # noqa: PLC0415

    adata = ad.read_h5ad(h5ad_path)
    if gt_obs_key not in adata.obs:
        raise SystemExit(
            f"--gt-obs-key {gt_obs_key!r} not in obs columns {list(adata.obs.columns)}"
        )
    raw = np.asarray([str(v) for v in adata.obs[gt_obs_key].to_numpy()])
    valid = np.array([s.strip().lower() not in _GT_NA_TOKENS for s in raw], dtype=bool)
    if valid.sum() < 2:
        return float("nan")
    _, codes = np.unique(raw[valid], return_inverse=True)
    codes = codes.astype(np.int64)

    import scipy.sparse as sp  # noqa: PLC0415

    matrix = adata.X
    if matrix is None:
        raise SystemExit(f"{h5ad_path}: AnnData has no .X expression matrix")
    sub = matrix[valid]
    X = np.asarray(sub.todense(), dtype=np.float64) if sp.issparse(sub) else np.asarray(sub, dtype=np.float64)
    embedding = _pca(X, n_comps=n_comps)
    return float(-silhouette(embedding, codes))


def _pca(X: np.ndarray, *, n_comps: int) -> np.ndarray:
    """Column-centered PCA reduction of ``X`` to at most ``n_comps`` components.

    Uses sklearn's PCA when importable (matches the rest of the pipeline), else a
    dependency-free numpy SVD on the column-centered matrix.
    """
    k = int(min(n_comps, X.shape[0], X.shape[1]))
    if k < 1:
        return X
    try:
        from sklearn.decomposition import PCA  # noqa: PLC0415

        return np.asarray(PCA(n_components=k, svd_solver="auto").fit_transform(X), dtype=np.float64)
    except Exception:  # noqa: BLE001  (sklearn optional -> numpy fallback)
        centered = X - X.mean(axis=0, keepdims=True)
        vt = np.linalg.svd(centered, full_matrices=False)[2]
        return centered @ vt[:k].T


def _parse_h5ad_arg(value: str) -> tuple[str, str]:
    """Parse a ``NAME=PATH`` ``--h5ad`` argument into ``(dataset_id, path)``."""
    if "=" not in value:
        raise argparse.ArgumentTypeError(f"--h5ad expects NAME=PATH, got {value!r}")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError(f"--h5ad expects NAME=PATH, got {value!r}")
    return name, path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--accuracy", type=Path, action="append", required=True,
        help="accuracy_results.json (dataset->variant->{ari,nmi,ami}); repeatable, one per dataset.",
    )
    parser.add_argument(
        "--h5ad", type=_parse_h5ad_arg, action="append", required=True,
        help="NAME=PATH source AnnData for reference-ambiguity; repeatable, NAME is the dataset id.",
    )
    parser.add_argument("--gt-obs-key", type=str, default="ground_truth", help="obs column with per-spot reference labels.")
    parser.add_argument("--expert", type=str, action="append", default=[], help="Dataset id to flag as expert-annotated; repeatable.")
    parser.add_argument("--n-comps", type=int, default=50, help="Max PCA components for the silhouette embedding.")
    parser.add_argument("--out", type=Path, required=True, help="Output PNG path.")
    parser.add_argument("--title", type=str, default="Agreement vs reference-label complexity")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    accuracy_by_dataset: dict[str, dict] = {}
    for path in args.accuracy:
        accuracy_by_dataset[str(path)] = json.loads(Path(path).read_text(encoding="utf-8"))

    ambiguity_by_dataset: dict[str, float] = {}
    for dataset_id, h5ad_path in args.h5ad:
        ambiguity = reference_ambiguity(h5ad_path, args.gt_obs_key, n_comps=args.n_comps)
        ambiguity_by_dataset[dataset_id] = ambiguity
        print(f"  ambiguity[{dataset_id}] = {ambiguity:+.4f}  (= -silhouette of reference in PCA space)")

    points = assemble_points(accuracy_by_dataset, ambiguity_by_dataset, expert_datasets=args.expert)
    if not points:
        raise SystemExit("no (dataset, variant) points to plot -- check --accuracy ids match --h5ad NAMEs")

    xs = [p["ambiguity"] for p in points]
    ys = [p["ari"] for p in points]
    r = pearson_r(xs, ys)
    render(points, args.out, title=args.title)
    print(f"wrote {args.out}  ({len(points)} points, Pearson R = {r:+.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
