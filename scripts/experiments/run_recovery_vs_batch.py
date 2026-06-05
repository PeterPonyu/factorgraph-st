#!/usr/bin/env python
"""#312 driver — factor/domain recovery vs increasing unwanted-variation strength.

Thin wrapper over :func:`robustness_harness.recovery_vs_batch`. Regenerates the
synthetic instance at increasing nuisance (batch) strengths — modeled by the
generator's additive-noise floor ``noise_sigma`` — and reports how factor and
domain recovery degrade as the planted signal is increasingly corrupted.

Usage::

    python scripts/experiments/run_recovery_vs_batch.py
    python scripts/experiments/run_recovery_vs_batch.py --batch-grid 0.5 5 15 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
for _p in (str(_REPO / "src"), str(_REPO / "scripts" / "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from robustness_harness import HarnessConfig, json_safe, recovery_vs_batch  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--batch-grid",
        nargs="+",
        type=float,
        default=None,
        help="Nuisance (noise_sigma) strengths to sweep, ascending.",
    )
    p.add_argument("--seed", type=int, default=0, help="Seed for data + model fit.")
    p.add_argument("--n-spots-per-section", type=int, default=HarnessConfig.n_spots_per_section)
    p.add_argument("--n-genes", type=int, default=HarnessConfig.n_genes)
    p.add_argument("--lam", type=float, default=HarnessConfig.lam)
    p.add_argument("--n-iter", type=int, default=HarnessConfig.n_iter)
    p.add_argument(
        "--out",
        type=Path,
        default=_REPO / "results" / "factorgraph-st" / "outputs" / "robustness" / "recovery_vs_batch.json",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    config = HarnessConfig(
        n_spots_per_section=args.n_spots_per_section,
        n_genes=args.n_genes,
        lam=args.lam,
        n_iter=args.n_iter,
    )
    result = recovery_vs_batch(config, batch_grid=args.batch_grid, seed=args.seed)

    print(f"#312 recovery vs unwanted-variation strength (seed={result['seed']})")
    print(f"  {'batch':>8}  {'factor_mfc':>10}  {'domain_ari':>10}")
    for row in result["curve"]:
        print(
            f"  {row['batch_strength']:>8.2f}  {row['matched_factor_correlation']:>10.4f}"
            f"  {row['domain_ari']:>10.4f}"
        )
    print(f"  factor recovery drop (first-last): {result['factor_recovery_drop']:.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(json_safe(result), indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
