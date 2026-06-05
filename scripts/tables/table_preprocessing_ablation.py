#!/usr/bin/env python
"""#337 — preprocessing x factor-rank ablation table (normalization / k sweep).

Crosses a set of preprocessing choices (``none`` / ``log1p`` / ``total_log1p``)
with a grid of requested factor ranks ``k`` and reports GT-free fit-quality
metrics for each cell of the ablation:

* ``reconstruction_error`` — relative Frobenius error ``||X - Z W^T|| / ||X||``
  (lower is better), the natural no-ground-truth fidelity score.
* ``factor_redundancy`` — mean absolute off-diagonal factor correlation (lower is
  more disentangled).

Two entry points: :func:`build_ablation_table` (data-independent, from in-memory
records) and :func:`run_ablation` (runs the sweep on a TINY synthetic instance,
so the table fills with real numbers without any labeled dataset).

Usage::

    python scripts/tables/table_preprocessing_ablation.py --run --out-dir /tmp/tables
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
for _p in (str(_REPO / "src"), str(_REPO / "scripts" / "tables")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from table_emit import Table, finite_float, write_table  # noqa: E402

HEADERS = ["normalization", "k", "reconstruction_error", "factor_redundancy"]
_BASENAME = "preprocessing_ablation"
_NORMALIZATIONS = ("none", "log1p", "total_log1p")


def _normalize(X, scheme: str):
    """Apply a preprocessing scheme to the count-like matrix ``X`` (numpy array).

    * ``none`` — identity.
    * ``log1p`` — ``log(1 + X)``.
    * ``total_log1p`` — library-size normalization (scale each spot to the median
      row sum) then ``log1p``; the corrected count-data contract (see #197).
    """
    import numpy as np  # noqa: PLC0415

    Xf = np.asarray(X, dtype=np.float64)
    if scheme == "none":
        out = Xf
    elif scheme == "log1p":
        out = np.log1p(Xf)
    elif scheme == "total_log1p":
        row_sums = Xf.sum(axis=1, keepdims=True)
        target = float(np.median(row_sums[row_sums > 0])) if np.any(row_sums > 0) else 1.0
        scaled = Xf * (target / np.maximum(row_sums, 1e-12))
        out = np.log1p(scaled)
    else:
        raise ValueError(f"unknown normalization scheme {scheme!r}; expected one of {_NORMALIZATIONS}")
    return out.astype(np.float32)


def build_ablation_table(records: Sequence[Mapping[str, object]]) -> Table:
    """Turn ablation records into a tidy table sorted by ``(normalization, k)``.

    Each record needs ``normalization`` and ``k``; metric values are coerced to
    finite floats (``None`` -> ``n/a``). Deterministic ordering.
    """
    ordered = sorted(records, key=lambda r: (str(r["normalization"]), int(r["k"])))  # type: ignore[call-overload]
    rows: list[list[object]] = []
    for record in ordered:
        rows.append(
            [
                str(record["normalization"]),
                int(record["k"]),  # type: ignore[call-overload]
                finite_float(record.get("reconstruction_error")),
                finite_float(record.get("factor_redundancy")),
            ]
        )
    return Table(name="preprocessing x factor-rank ablation (#337)", headers=HEADERS, rows=rows)


def run_ablation(
    *,
    normalizations: Sequence[str] = _NORMALIZATIONS,
    k_grid: Sequence[int] = (3, 4, 6),
    n_spots_per_section: int = 40,
    n_genes: int = 30,
    n_iter: int = 30,
    seed: int = 0,
) -> list[dict[str, object]]:
    """Run the normalization x k ablation on one TINY synthetic instance.

    The synthetic instance is held fixed; each ``(normalization, k)`` cell refits
    GNMF on the preprocessed matrix and records GT-free fit-quality metrics.
    Returns records ready for :func:`build_ablation_table`.
    """
    from factorgraph_st.eval.metrics import factor_redundancy, reconstruction_error  # noqa: PLC0415
    from factorgraph_st.model.learned import fit_transform_gnmf  # noqa: PLC0415
    from factorgraph_st.synth.generator import generate_instance  # noqa: PLC0415

    synth = generate_instance(
        n_sections=2,
        n_spots_per_section=n_spots_per_section,
        n_genes=n_genes,
        K_shared=3,
        K_private=1,
        n_domains=4,
        seed=seed,
    )
    records: list[dict[str, object]] = []
    for scheme in normalizations:
        Xp = _normalize(synth.X, scheme)
        for k in k_grid:
            k_shared = max(1, int(k) - 1)
            k_private = int(k) - k_shared
            outputs, _ = fit_transform_gnmf(
                Xp,
                synth.edges,
                K_shared=k_shared,
                K_private=k_private,
                n_domains=4,
                n_iter=n_iter,
                seed=seed,
            )
            recon = reconstruction_error(Xp, outputs.H, outputs.W)
            redundancy = factor_redundancy(outputs.H)
            records.append(
                {
                    "normalization": scheme,
                    "k": int(k),
                    "reconstruction_error": float(recon),
                    "factor_redundancy": float(redundancy),
                }
            )
    return records


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for the emitted table files.")
    parser.add_argument("--run", action="store_true", help="Run the ablation on a tiny synthetic instance.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    records = run_ablation() if args.run else []
    table = build_ablation_table(records)
    paths = write_table(table, args.out_dir, _BASENAME)
    for fmt, path in sorted(paths.items()):
        print(f"wrote {fmt}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
