#!/usr/bin/env python
"""Real-data GNMF graph-regularization diagnosis (#392).

The synthetic sweep (``scripts/sweep_gnmf.py``) tunes ``lam``/``n_iter``/``tol``
against *known* factors, but it cannot explain why the learned GNMF is
**dominated by the trivial ``coords`` and ``spatial_smooth`` baselines on real
labeled tissue** (STARmap, DLPFC Maynard). This script is that real-data probe.

It sweeps the graph-regularization strength ``lam`` on a labeled dataset, holding
everything else fixed to the ``run_real_factorgraph.py`` defaults (so the numbers
line up with the committed runs), and for each ``lam`` reports, averaged over
seeds:

* **agreement with the manual annotation** -- ARI / NMI / AMI on the labeled
  subset (``accuracy_from_labels``), the only "is it correct" axis;
* **spatial contiguity** -- :func:`edge_label_purity`, the fraction of spatial
  kNN edges whose endpoints share a predicted domain (a label-free "is it
  smooth" axis, high for contiguous domains, ~chance for salt-and-pepper);
* **objective balance** -- :func:`objective_split` decomposes the GNMF objective
  into the reconstruction term and the ``lam``-weighted Laplacian smoothness
  term, so the *relative weight the regularizer actually carries* is explicit.

WHY THIS DIAGNOSES THE GAP. The reconstruction term ``||X - H W^T||_F^2`` scales
with ``n_spots * n_genes``, while ``Tr(H^T L H)`` scales with the (small) factor
count -- so at the default ``lam = 1`` the smoothness term is a fraction of a
percent of the objective. The fit is therefore expression-dominated and the
``H``-only clustering step (no coords, no graph) produces fragmented domains.
Raising ``lam`` by orders of magnitude restores contiguity and lifts ARI, up to
an over-smoothing point where domains collapse. This script makes that whole
curve visible instead of judging GNMF at one (mis-scaled) operating point.

HONESTY: every number is recomputed from a real fit; nothing is imputed. The
recommended ``lam`` is the best *stable* mean ARI (mean maximized among configs
whose across-seed std is within ``--stable-std``), never a lucky single seed.
The script does not claim GNMF beats ``spatial_smooth`` -- it characterizes how
much of the gap is a tuning artifact versus a structural limit of the
factor-bottleneck + cluster-on-H design.

matplotlib is imported lazily and only for ``--fig``; the scoring core is
numpy-only and unit-tested in ``tests/test_diagnose_gnmf_realdata.py``.

Usage::

    conda run --no-capture-output -n dl python scripts/diagnose_gnmf_realdata.py \
        --h5ad ../data/processed/starmap_mouse_vcortex_wang2018/anndata.h5ad \
        --dataset-id starmap_mouse_vcortex_wang2018 \
        --baselines results/starmap_wang2018/_scorecard/accuracy_results.json \
        --out-dir results/starmap_wang2018/_gnmf_diag --fig
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
for _p in (str(_REPO / "src"), str(_REPO / "scripts" / "tables")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

#: NA/blank GT tokens, byte-for-byte mirror of ``run_real_factorgraph._GT_NA_TOKENS``.
_GT_NA_TOKENS = frozenset({"", "na", "nan", "none", "unknown", "unlabeled", "unlabelled"})


# --------------------------------------------------------------------------- #
# numpy-only scoring core (unit-tested; no scanpy / anndata / matplotlib)
# --------------------------------------------------------------------------- #
def edge_label_purity(domain_id: np.ndarray, edges: np.ndarray) -> float:
    """Fraction of spatial kNN edges whose endpoints share a predicted domain.

    A label-free spatial-contiguity proxy: ~1.0 for blocky/contiguous domains,
    near the chance level (``sum p_k^2`` over domain frequencies) for a
    salt-and-pepper partition. Returns ``nan`` for an empty edge list so an
    edgeless input never reads as perfectly pure.
    """
    domain_id = np.asarray(domain_id).astype(np.int64)
    edges = np.asarray(edges)
    if edges.size == 0:
        return float("nan")
    src, dst = edges
    return float((domain_id[src] == domain_id[dst]).mean())


def objective_split(
    X: np.ndarray, H: np.ndarray, W: np.ndarray, edges: np.ndarray, lam: float
) -> dict[str, float]:
    """Split the GNMF objective into reconstruction and ``lam``-weighted smoothness.

    Returns ``{recon, smooth, lam_smooth, smooth_fraction}`` where ``recon`` is
    ``||X - H W^T||_F^2``, ``smooth`` is ``Tr(H^T L H)`` (unweighted), ``lam_smooth``
    is ``lam * smooth`` (its actual contribution to the minimized objective), and
    ``smooth_fraction = lam_smooth / (recon + lam_smooth)`` -- the share of the
    objective the regularizer carries. A ``smooth_fraction`` near zero is the
    signature of an under-weighted graph prior. Uses the same scatter-add
    Laplacian as :mod:`factorgraph_st.model.learned` (no dense ``A``).
    """
    from factorgraph_st.model.learned import _adj_matmul, _graph_terms  # noqa: PLC0415

    Xf = np.clip(np.asarray(X, dtype=np.float64), 0.0, None)
    Hf = np.asarray(H, dtype=np.float64)
    Wf = np.asarray(W, dtype=np.float64)
    n_spots = Xf.shape[0]
    residual = Xf - Hf @ Wf.T
    recon = float(np.sum(residual * residual))
    src_sym, dst_sym, degree = _graph_terms(np.asarray(edges), n_spots)
    AH = _adj_matmul(src_sym, dst_sym, Hf, n_spots)
    smooth = float(np.sum(degree[:, None] * Hf * Hf) - np.sum(Hf * AH))
    lam_smooth = float(lam) * smooth
    denom = recon + lam_smooth
    frac = float(lam_smooth / denom) if denom > 0 else float("nan")
    return {"recon": recon, "smooth": smooth, "lam_smooth": lam_smooth, "smooth_fraction": frac}


def labeled_gt_codes(gt_raw: Sequence[object]) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(valid_mask, gt_codes)`` for the labeled subset of raw GT values.

    Mirrors the collector/runner NA handling so the diagnosis ARI matches the
    committed runs exactly.
    """
    raw = np.asarray([str(v) for v in gt_raw])
    valid = np.array([s.strip().lower() not in _GT_NA_TOKENS for s in raw], dtype=bool)
    _, codes = np.unique(raw[valid], return_inverse=True)
    return valid, codes.astype(np.int64)


def aggregate_sweep(rows: Sequence[dict]) -> list[dict]:
    """Aggregate per-(lam, seed) rows to per-lam mean/std of each scalar metric.

    ``rows`` are the raw per-fit records; the output is one dict per ``lam`` with
    ``{ari,nmi,ami,edge_purity}_{mean,std}``, ``n_seeds``, and the seed-0
    ``smooth_fraction`` (objective balance is seed-stable enough to report once).
    Sorted by ``lam`` ascending.
    """
    by_lam: dict[float, list[dict]] = {}
    for r in rows:
        by_lam.setdefault(float(r["lam"]), []).append(r)
    out: list[dict] = []
    for lam in sorted(by_lam):
        group = by_lam[lam]
        agg: dict[str, object] = {"lam": lam, "n_seeds": len(group)}
        for key in ("ari", "nmi", "ami", "edge_purity"):
            vals = np.asarray([g[key] for g in group], dtype=float)
            agg[f"{key}_mean"] = float(np.nanmean(vals))
            agg[f"{key}_std"] = float(np.nanstd(vals))
        seed0 = min(group, key=lambda g: g["seed"])
        agg["smooth_fraction"] = float(seed0["smooth_fraction"])
        out.append(agg)
    return out


def recommend_lam(agg_rows: Sequence[dict], stable_std: float) -> dict | None:
    """Pick the ``lam`` with the highest mean ARI among across-seed-stable configs.

    "Stable" = ``ari_std <= stable_std``. Falls back to the global max-mean row if
    no config is stable (flagged via ``stable=False``). Returns ``None`` for an
    empty sweep.
    """
    if not agg_rows:
        return None
    stable = [r for r in agg_rows if r["ari_std"] <= stable_std]
    pool = stable or list(agg_rows)
    best = max(pool, key=lambda r: r["ari_mean"])
    return {"lam": best["lam"], "ari_mean": best["ari_mean"], "ari_std": best["ari_std"], "stable": bool(stable)}


# --------------------------------------------------------------------------- #
# heavy real-data path (scanpy / anndata behind lazy imports)
# --------------------------------------------------------------------------- #
def _load_runner():
    """Import ``run_real_factorgraph.py`` as a module to reuse its preprocessing.

    Using the runner's own ``_preprocess`` / ``_build_knn_edges`` guarantees the
    diagnosis sees byte-identical inputs to the committed gnmf runs.
    """
    path = _REPO / "scripts" / "run_real_factorgraph.py"
    spec = importlib.util.spec_from_file_location("_rrf_for_diag", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _baseline_aris(baselines_path: Path | None, dataset_id: str) -> dict[str, float]:
    """Read ``coords``/``spatial_smooth`` reference ARIs from a scorecard JSON.

    Accepts the ``accuracy_results.json`` schema ``dataset -> variant -> {ari,..}``.
    Returns ``{}`` if the file is absent so the figure simply omits the reference
    lines rather than fabricating them.
    """
    if baselines_path is None or not Path(baselines_path).is_file():
        return {}
    data = json.loads(Path(baselines_path).read_text(encoding="utf-8"))
    per_variant = data.get(dataset_id) or next(iter(data.values()), {})
    return {
        v: float(per_variant[v]["ari"])
        for v in ("coords", "spatial_smooth")
        if v in per_variant and "ari" in per_variant[v]
    }


def run_sweep(args: argparse.Namespace) -> dict:
    """Execute the real-data lam sweep and return the full result payload."""
    import anndata as ad  # noqa: PLC0415
    from table_domain_accuracy import accuracy_from_labels  # noqa: PLC0415

    from factorgraph_st.model.learned import fit_transform_gnmf  # noqa: PLC0415

    rrf = _load_runner()
    adata = ad.read_h5ad(args.h5ad)
    rrf._preprocess(adata, already_normalized=args.already_normalized, n_hvg=args.n_hvg)
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    X = np.asarray(X, dtype=np.float64)
    coords = np.asarray(adata.obsm["spatial"], dtype=float)
    edges = rrf._build_knn_edges(coords, args.knn)

    if args.gt_obs_key not in adata.obs:
        raise SystemExit(f"--gt-obs-key {args.gt_obs_key!r} not in obs {list(adata.obs.columns)}")
    valid, gt_codes = labeled_gt_codes(adata.obs[args.gt_obs_key].to_numpy())

    print(
        f"[{args.dataset_id}] X={X.shape} edges={edges.shape[1]} "
        f"labeled={int(valid.sum())}/{valid.size} | lam grid {args.lam_grid} x {args.n_seeds} seeds"
    )
    rows: list[dict] = []
    for lam in args.lam_grid:
        for seed in range(args.n_seeds):
            t0 = time.perf_counter()
            out, res = fit_transform_gnmf(
                X, edges,
                K_shared=args.k_shared, K_private=args.k_private, n_domains=args.n_domains,
                lam=lam, n_iter=args.n_iter, tol=args.tol, seed=seed,
            )
            dom = out.domain_id.astype(np.int64)
            acc = accuracy_from_labels(gt_codes, dom[valid])
            split = objective_split(X, out.H, out.W, edges, lam)
            rows.append({
                "lam": float(lam), "seed": int(seed),
                "ari": float(acc["ari"]), "nmi": float(acc["nmi"]), "ami": float(acc["ami"]),
                "edge_purity": edge_label_purity(dom, edges),
                "smooth_fraction": split["smooth_fraction"],
                "recon": split["recon"], "lam_smooth": split["lam_smooth"],
                "n_iter_run": int(res.n_iter_run), "runtime_s": time.perf_counter() - t0,
            })
            r = rows[-1]
            print(
                f"  lam={lam:>8.1f} seed={seed} ari={r['ari']:.3f} purity={r['edge_purity']:.3f} "
                f"smooth_frac={r['smooth_fraction']:.4f} iters={r['n_iter_run']} ({r['runtime_s']:.1f}s)"
            )

    agg = aggregate_sweep(rows)
    rec = recommend_lam(agg, args.stable_std)
    baselines = _baseline_aris(args.baselines, args.dataset_id)
    return {
        "dataset_id": args.dataset_id,
        "config": {
            "knn": args.knn, "k_shared": args.k_shared, "k_private": args.k_private,
            "n_domains": args.n_domains, "n_iter": args.n_iter, "tol": args.tol,
            "n_seeds": args.n_seeds, "lam_grid": list(args.lam_grid),
            "default_lam": 1.0, "stable_std": args.stable_std,
        },
        "baselines": baselines,
        "recommended": rec,
        "aggregated": agg,
        "raw": rows,
    }


def render_figure(payload: dict, out_path: Path) -> None:
    """Render ARI-vs-lam (left axis) + edge-purity (right axis) with baseline lines."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    agg = payload["aggregated"]
    lams = np.asarray([r["lam"] for r in agg], dtype=float)
    lam_x = np.where(lams > 0, lams, 0.5)  # plot lam=0 at a finite left position on log-x
    ari_m = np.asarray([r["ari_mean"] for r in agg])
    ari_s = np.asarray([r["ari_std"] for r in agg])
    pur_m = np.asarray([r["edge_purity_mean"] for r in agg])

    fig, axL = plt.subplots(figsize=(7.2, 4.6))
    axL.errorbar(lam_x, ari_m, yerr=ari_s, marker="o", color="#0072B2", capsize=3, label="gnmf ARI (mean±sd)", zorder=3)
    for variant, color in (("coords", "#56B4E9"), ("spatial_smooth", "#009E73")):
        if variant in payload.get("baselines", {}):
            axL.axhline(payload["baselines"][variant], ls="--", color=color, lw=1.4, label=f"{variant} ARI", zorder=2)
    rec = payload.get("recommended")
    if rec:
        rx = rec["lam"] if rec["lam"] > 0 else 0.5
        axL.axvline(rx, color="0.4", ls=":", lw=1.0, zorder=1)
    axL.set_xscale("log")
    axL.set_xlabel("graph-regularization strength  λ  (default = 1)")
    axL.set_ylabel("ARI vs annotated domains", color="#0072B2")
    axL.tick_params(axis="y", labelcolor="#0072B2")
    axL.spines["top"].set_visible(False)

    axR = axL.twinx()
    axR.plot(lam_x, pur_m, marker="s", color="#D55E00", lw=1.2, label="edge purity")
    axR.set_ylabel("spatial edge purity", color="#D55E00")
    axR.tick_params(axis="y", labelcolor="#D55E00")
    axR.spines["top"].set_visible(False)

    hL, lL = axL.get_legend_handles_labels()
    hR, lR = axR.get_legend_handles_labels()
    axL.legend(hL + hR, lL + lR, fontsize=8, frameon=False, loc="lower center", ncol=2)
    axL.set_title(f"GNMF λ-sensitivity on real GT — {payload['dataset_id']} (#392)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h5ad", type=Path, required=True)
    p.add_argument("--dataset-id", type=str, required=True)
    p.add_argument("--gt-obs-key", type=str, default="ground_truth")
    p.add_argument("--lam-grid", nargs="+", type=float,
                   default=[0.0, 1.0, 10.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0])
    p.add_argument("--n-seeds", type=int, default=4)
    p.add_argument("--knn", type=int, default=6)
    p.add_argument("--k-shared", type=int, default=4)
    p.add_argument("--k-private", type=int, default=2)
    p.add_argument("--n-domains", type=int, default=5)
    p.add_argument("--n-iter", type=int, default=200)
    p.add_argument("--tol", type=float, default=1e-4)
    p.add_argument("--already-normalized", action="store_true")
    p.add_argument("--n-hvg", type=int, default=0)
    p.add_argument("--stable-std", type=float, default=0.05,
                   help="Max across-seed ARI std for a lam to count as 'stable' when recommending.")
    p.add_argument("--baselines", type=Path, default=None,
                   help="accuracy_results.json for coords/spatial_smooth reference lines.")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--fig", action="store_true", help="Also render the lam-sensitivity figure.")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    payload = run_sweep(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "diag.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out_dir / 'diag.json'}")

    print(f"\n=== {payload['dataset_id']}: per-lam summary ===")
    print(f"{'lam':>9} {'ari':>14} {'edge_purity':>12} {'smooth_frac':>12}")
    for r in payload["aggregated"]:
        print(f"{r['lam']:>9.1f} {r['ari_mean']:>7.3f}±{r['ari_std']:<5.3f} "
              f"{r['edge_purity_mean']:>12.3f} {r['smooth_fraction']:>12.4f}")
    rec = payload["recommended"]
    if rec:
        flag = "" if rec["stable"] else "  [no stable config; global max]"
        print(f"\n>>> recommended lam={rec['lam']:.1f}  ari={rec['ari_mean']:.3f}±{rec['ari_std']:.3f}{flag}")
    if payload.get("baselines"):
        print("    baselines:", ", ".join(f"{k}={v:.3f}" for k, v in payload["baselines"].items()))

    if args.fig:
        render_figure(payload, out_dir / "gnmf_lam_sensitivity.png")
        print(f"wrote {out_dir / 'gnmf_lam_sensitivity.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
