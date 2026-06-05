#!/usr/bin/env python
"""#314 driver — factor/domain stability across model initializations (seeds).

Thin wrapper over :func:`robustness_harness.seed_init_stability`. Holds the
synthetic data fixed and refits the GNMF model under several initialization
seeds, reporting how reproducible the recovered factors and domains are.

Usage::

    python scripts/experiments/run_seed_stability.py
    python scripts/experiments/run_seed_stability.py --seeds 0 1 2 3 4
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

from robustness_harness import HarnessConfig, json_safe, seed_init_stability  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3], help="Model init seeds (>=2).")
    p.add_argument("--data-seed", type=int, default=0, help="Seed for the (fixed) synthetic instance.")
    p.add_argument("--n-spots-per-section", type=int, default=HarnessConfig.n_spots_per_section)
    p.add_argument("--n-genes", type=int, default=HarnessConfig.n_genes)
    p.add_argument("--lam", type=float, default=HarnessConfig.lam)
    p.add_argument("--n-iter", type=int, default=HarnessConfig.n_iter)
    p.add_argument(
        "--out",
        type=Path,
        default=_REPO / "results" / "factorgraph-st" / "outputs" / "robustness" / "seed_stability.json",
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
    result = seed_init_stability(config, seeds=args.seeds, data_seed=args.data_seed)

    print(f"#314 seed/init stability over seeds {result['seeds']} (data_seed={result['data_seed']})")
    print(f"  mean factor-matching stability : {result['mean_factor_stability']:.4f}")
    print(f"  mean domain ARI stability      : {result['mean_domain_ari_stability']:.4f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(json_safe(result), indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
