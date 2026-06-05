#!/usr/bin/env python
"""#308 driver — domain robustness to the requested number of domains ``k``.

Thin wrapper over :func:`robustness_harness.robustness_to_k`. Sweeps the
requested domain count around its true value and reports the partition's
self-consistency across initializations at each ``k`` (the recoverable,
GT-free robustness signal — see the harness docstring for why GT-domain
accuracy is near-null by construction here). Optionally applies mild spatial
neighborhood disruption via ``--edge-dropout``.

Usage::

    python scripts/experiments/run_robustness_to_k.py
    python scripts/experiments/run_robustness_to_k.py --k-grid 2 3 4 5 6 --edge-dropout 0.1
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

from robustness_harness import HarnessConfig, json_safe, robustness_to_k  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--k-grid", nargs="+", type=int, default=None, help="Requested domain counts to sweep.")
    p.add_argument("--stability-seeds", nargs="+", type=int, default=[0, 1, 2], help="Init seeds (>=2).")
    p.add_argument("--edge-dropout", type=float, default=0.0, help="Fraction of graph edges to drop in [0,1).")
    p.add_argument("--n-domains", type=int, default=HarnessConfig.n_domains, help="True domain count.")
    p.add_argument("--n-spots-per-section", type=int, default=HarnessConfig.n_spots_per_section)
    p.add_argument("--n-genes", type=int, default=HarnessConfig.n_genes)
    p.add_argument("--lam", type=float, default=HarnessConfig.lam)
    p.add_argument("--n-iter", type=int, default=HarnessConfig.n_iter)
    p.add_argument(
        "--out",
        type=Path,
        default=_REPO / "results" / "factorgraph-st" / "outputs" / "robustness" / "robustness_to_k.json",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    config = HarnessConfig(
        n_domains=args.n_domains,
        n_spots_per_section=args.n_spots_per_section,
        n_genes=args.n_genes,
        lam=args.lam,
        n_iter=args.n_iter,
    )
    result = robustness_to_k(
        config,
        k_grid=args.k_grid,
        stability_seeds=args.stability_seeds,
        edge_dropout=args.edge_dropout,
    )

    print(f"#308 robustness to requested k (true_k={result['true_k']}, edge_dropout={result['edge_dropout']})")
    print(f"  {'k':>4}  {'self_stability':>14}  {'ari_vs_gt':>10}")
    for row in result["curve"]:
        print(f"  {row['requested_k']:>4}  {row['domain_self_stability']:>14.4f}  {row['domain_ari_vs_gt']:>10.4f}")
    print(f"  best_k by self-stability: {result['best_k']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(json_safe(result), indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
