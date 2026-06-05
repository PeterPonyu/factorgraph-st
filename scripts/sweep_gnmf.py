#!/usr/bin/env python
"""Synthetic hyperparam sweep for GNMF: lam / n_iter / tol.

Generates a small synthetic dataset with known shared/private factors,
then sweeps gnmf lam, n_iter, and tol. Scores each configuration with:
  - matched_factor_correlation on the full factor matrix H vs ground-truth Z
  - domain ARI of recovered domain_id vs ground-truth domain_id

Writes sweep results to results/factorgraph-st/outputs/gnmf_sweep/sweep_results.json
and prints a sorted summary table to stdout.

Usage::

    conda run --no-capture-output -n dl python scripts/sweep_gnmf.py
    conda run --no-capture-output -n dl python scripts/sweep_gnmf.py --n-seeds 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# --- project root on sys.path so imports work without editable install -------
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from factorgraph_st.eval.metrics import adjusted_rand_index, matched_factor_correlation
from factorgraph_st.model.learned import fit_transform_gnmf
from factorgraph_st.synth.generator import generate_instance


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--n-sections",
        type=int,
        default=2,
        help="Sections in the synthetic instance (default 2).",
    )
    p.add_argument(
        "--n-spots-per-section",
        type=int,
        default=300,
        help="Spots per section (default 300; bigger = slower but more stable estimates).",
    )
    p.add_argument(
        "--n-genes",
        type=int,
        default=150,
        help="Gene dimension of synthetic data (default 150).",
    )
    p.add_argument(
        "--k-shared",
        type=int,
        default=4,
        help="Ground-truth shared factors (default 4).",
    )
    p.add_argument(
        "--k-private",
        type=int,
        default=2,
        help="Ground-truth private factors (default 2).",
    )
    p.add_argument(
        "--n-domains",
        type=int,
        default=5,
        help="Domains in both synthetic GT and GNMF clustering (default 5).",
    )
    p.add_argument(
        "--n-seeds",
        type=int,
        default=3,
        help="Number of random seeds to average over per config (default 3).",
    )
    p.add_argument(
        "--lam-grid",
        nargs="+",
        type=float,
        default=[0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        help="lam values to sweep (default: 0.0 0.1 0.5 1.0 2.0 5.0 10.0).",
    )
    p.add_argument(
        "--n-iter-grid",
        nargs="+",
        type=int,
        default=[50, 100, 200],
        help="n_iter values to sweep (default: 50 100 200).",
    )
    p.add_argument(
        "--tol-grid",
        nargs="+",
        type=float,
        default=[1e-3, 1e-4, 1e-5],
        help="tol values to sweep (default: 1e-3 1e-4 1e-5).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_REPO / "results" / "factorgraph-st" / "outputs" / "gnmf_sweep",
        help="Directory for sweep_results.json output.",
    )
    return p.parse_args()


def _run_one(
    synth,
    *,
    lam: float,
    n_iter: int,
    tol: float,
    seed: int,
) -> dict:
    """Fit GNMF on a SynthInstance and return a dict of evaluation scores."""
    t0 = time.perf_counter()
    K_shared = synth.Z_shared.shape[1]
    K_private = synth.Z_private.shape[1]
    out, gnmf_result = fit_transform_gnmf(
        synth.X,
        synth.edges,
        K_shared=K_shared,
        K_private=K_private,
        n_domains=int(np.unique(synth.domain_id).size),
        lam=lam,
        n_iter=n_iter,
        tol=tol,
        seed=seed,
    )
    elapsed = time.perf_counter() - t0

    # Full recovered factor matrix H vs GT Z = [Z_shared | Z_private]
    Z_gt = np.concatenate([synth.Z_shared, synth.Z_private], axis=1).astype(np.float64)
    mfc = matched_factor_correlation(out.H.astype(np.float64), Z_gt)

    # Domain ARI: recovered domains vs GT domain_id
    ari = adjusted_rand_index(
        synth.domain_id.astype(np.int64),
        out.domain_id.astype(np.int64),
    )

    return {
        "lam": lam,
        "n_iter": n_iter,
        "tol": tol,
        "seed": seed,
        "matched_factor_correlation": float(mfc) if np.isfinite(mfc) else None,
        "domain_ari": float(ari) if np.isfinite(ari) else None,
        "objective_initial": float(gnmf_result.objective[0]),
        "objective_final": float(gnmf_result.objective[-1]),
        "n_iter_run": int(gnmf_result.n_iter_run),
        "runtime_s": elapsed,
        "converged": bool(gnmf_result.n_iter_run < n_iter),
    }


def main() -> None:
    args = _parse_args()

    print(
        f"Synthetic sweep: {args.n_sections} sections × {args.n_spots_per_section} spots "
        f"× {args.n_genes} genes | K_shared={args.k_shared} K_private={args.k_private} "
        f"n_domains={args.n_domains} seeds={args.n_seeds}"
    )
    print(
        f"Grid: lam={args.lam_grid}  n_iter={args.n_iter_grid}  tol={args.tol_grid}"
    )
    total_runs = len(args.lam_grid) * len(args.n_iter_grid) * len(args.tol_grid) * args.n_seeds
    print(f"Total runs: {total_runs}\n")

    results: list[dict] = []
    run_idx = 0

    for seed in range(args.n_seeds):
        synth = generate_instance(
            n_sections=args.n_sections,
            n_spots_per_section=args.n_spots_per_section,
            n_genes=args.n_genes,
            K_shared=args.k_shared,
            K_private=args.k_private,
            n_domains=args.n_domains,
            seed=seed,
        )
        for lam in args.lam_grid:
            for n_iter in args.n_iter_grid:
                for tol in args.tol_grid:
                    run_idx += 1
                    row = _run_one(synth, lam=lam, n_iter=n_iter, tol=tol, seed=seed)
                    results.append(row)
                    mfc_str = f"{row['matched_factor_correlation']:.4f}" if row["matched_factor_correlation"] is not None else "nan"
                    ari_str = f"{row['domain_ari']:.4f}" if row["domain_ari"] is not None else "nan"
                    print(
                        f"[{run_idx:3d}/{total_runs}] lam={lam:5.2f} n_iter={n_iter:3d} tol={tol:.0e} "
                        f"seed={seed} -> mfc={mfc_str} ari={ari_str} "
                        f"iters={row['n_iter_run']:3d} rt={row['runtime_s']:.2f}s"
                    )

    # Aggregate: mean over seeds per (lam, n_iter, tol)
    from collections import defaultdict

    agg: dict[tuple, dict] = defaultdict(lambda: {"mfc": [], "ari": [], "iters": [], "rt": []})
    for r in results:
        key = (r["lam"], r["n_iter"], r["tol"])
        if r["matched_factor_correlation"] is not None:
            agg[key]["mfc"].append(r["matched_factor_correlation"])
        if r["domain_ari"] is not None:
            agg[key]["ari"].append(r["domain_ari"])
        agg[key]["iters"].append(r["n_iter_run"])
        agg[key]["rt"].append(r["runtime_s"])

    agg_rows = []
    for (lam, n_iter, tol), vals in agg.items():
        mfc_mean = float(np.mean(vals["mfc"])) if vals["mfc"] else float("nan")
        ari_mean = float(np.mean(vals["ari"])) if vals["ari"] else float("nan")
        iters_mean = float(np.mean(vals["iters"]))
        rt_mean = float(np.mean(vals["rt"]))
        agg_rows.append(
            {
                "lam": lam,
                "n_iter": n_iter,
                "tol": tol,
                "mfc_mean": mfc_mean,
                "ari_mean": ari_mean,
                "iters_mean": iters_mean,
                "rt_mean_s": rt_mean,
            }
        )

    # Sort by mfc_mean descending, then ari_mean descending
    agg_rows.sort(key=lambda x: (-x["mfc_mean"] if np.isfinite(x["mfc_mean"]) else float("inf"), -x["ari_mean"] if np.isfinite(x["ari_mean"]) else float("inf")))

    print("\n=== Aggregated results (sorted by mfc_mean desc) ===")
    print(f"{'lam':>6}  {'n_iter':>6}  {'tol':>8}  {'mfc_mean':>9}  {'ari_mean':>9}  {'iters_mean':>10}  {'rt_mean_s':>9}")
    print("-" * 75)
    for r in agg_rows:
        mfc_s = f"{r['mfc_mean']:.4f}" if np.isfinite(r["mfc_mean"]) else "  nan"
        ari_s = f"{r['ari_mean']:.4f}" if np.isfinite(r["ari_mean"]) else "  nan"
        print(
            f"{r['lam']:>6.2f}  {r['n_iter']:>6d}  {r['tol']:>8.0e}  {mfc_s:>9}  {ari_s:>9}  {r['iters_mean']:>10.1f}  {r['rt_mean_s']:>9.2f}"
        )

    # Recommended defaults: top row by mfc_mean
    best = agg_rows[0]
    print(f"\n>>> RECOMMENDED defaults: lam={best['lam']} n_iter={best['n_iter']} tol={best['tol']}")
    print(f"    mfc_mean={best['mfc_mean']:.4f}  ari_mean={best['ari_mean']:.4f}  avg_iters={best['iters_mean']:.1f}")

    # Persist
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "sweep_results.json"
    payload = {
        "config": {
            "n_sections": args.n_sections,
            "n_spots_per_section": args.n_spots_per_section,
            "n_genes": args.n_genes,
            "K_shared": args.k_shared,
            "K_private": args.k_private,
            "n_domains": args.n_domains,
            "n_seeds": args.n_seeds,
            "lam_grid": args.lam_grid,
            "n_iter_grid": args.n_iter_grid,
            "tol_grid": args.tol_grid,
        },
        "recommended": {"lam": best["lam"], "n_iter": best["n_iter"], "tol": best["tol"]},
        "aggregated": agg_rows,
        "raw": results,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
