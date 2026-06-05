#!/usr/bin/env python
"""#336 — runtime + peak-memory scalability table vs #spots and #genes.

Two entry points:

* :func:`build_scalability_table` — data-independent: turns an in-memory list of
  measurement rows ``{n_spots, n_genes, runtime_s, peak_rss_mb}`` into a tidy
  table. This is what real benchmark numbers fill later.
* :func:`measure_scalability` — generates SMALL synthetic instances of increasing
  size, fits the GNMF model on each, and records wall-clock runtime
  (``time.perf_counter``) and peak resident memory (``resource.getrusage``).
  This produces *real* (not fabricated) timing/RSS rows for tiny instances, so
  the table is exercised end-to-end without any labeled dataset.

``resource`` is POSIX-only; on platforms without it the ``peak_rss_mb`` cell is
emitted as ``n/a`` rather than a fabricated number.

Usage::

    python scripts/tables/table_scalability.py --measure --out-dir /tmp/tables
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
for _p in (str(_REPO / "src"), str(_REPO / "scripts" / "tables")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from table_emit import Table, finite_float, write_table  # noqa: E402

HEADERS = ["n_spots", "n_genes", "runtime_s", "peak_rss_mb"]
_BASENAME = "scalability"


def _peak_rss_mb() -> float:
    """Peak resident set size in MiB via ``resource.getrusage``, or ``nan``.

    ``ru_maxrss`` is in kilobytes on Linux and bytes on macOS; we assume the
    Linux convention (this repo's CI target). Returns ``nan`` where ``resource``
    is unavailable (non-POSIX) so the cell renders as ``n/a`` (never fabricated).
    """
    try:
        import resource  # noqa: PLC0415
    except ImportError:
        return float("nan")
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def build_scalability_table(rows: Sequence[Mapping[str, float]]) -> Table:
    """Turn measurement records into a tidy scalability table.

    Each record needs ``n_spots`` and ``n_genes``; ``runtime_s`` / ``peak_rss_mb``
    are coerced to finite floats (or ``None`` -> ``n/a``). Rows are sorted by
    ``(n_spots, n_genes)`` so the size sweep reads monotonically and the table is
    deterministic regardless of input order.
    """
    ordered = sorted(rows, key=lambda r: (int(r["n_spots"]), int(r["n_genes"])))
    out_rows: list[list[object]] = []
    for record in ordered:
        out_rows.append(
            [
                int(record["n_spots"]),
                int(record["n_genes"]),
                finite_float(record.get("runtime_s")),
                finite_float(record.get("peak_rss_mb")),
            ]
        )
    return Table(name="runtime + peak-memory scalability (#336)", headers=HEADERS, rows=out_rows)


def measure_scalability(
    sizes: Sequence[tuple[int, int]],
    *,
    n_sections: int = 2,
    n_iter: int = 30,
    seed: int = 0,
) -> list[dict[str, float]]:
    """Fit GNMF on small synthetic instances of increasing size; time each.

    ``sizes`` is a list of ``(n_spots_per_section, n_genes)`` pairs. For each, a
    synthetic instance is generated and ``fit_transform_gnmf`` is run with a small
    ``n_iter``; runtime and peak RSS are recorded. Returns measurement records
    ready for :func:`build_scalability_table`. Deliberately tiny so it is
    test-safe (well under a second per size).
    """
    from factorgraph_st.model.learned import fit_transform_gnmf  # noqa: PLC0415
    from factorgraph_st.synth.generator import generate_instance  # noqa: PLC0415

    records: list[dict[str, float]] = []
    for n_spots_per_section, n_genes in sizes:
        synth = generate_instance(
            n_sections=n_sections,
            n_spots_per_section=int(n_spots_per_section),
            n_genes=int(n_genes),
            K_shared=3,
            K_private=1,
            n_domains=4,
            seed=seed,
        )
        start = time.perf_counter()
        fit_transform_gnmf(
            synth.X,
            synth.edges,
            K_shared=3,
            K_private=1,
            n_domains=4,
            n_iter=n_iter,
            seed=seed,
        )
        runtime_s = time.perf_counter() - start
        records.append(
            {
                "n_spots": int(n_sections * n_spots_per_section),
                "n_genes": int(n_genes),
                "runtime_s": float(runtime_s),
                "peak_rss_mb": _peak_rss_mb(),
            }
        )
    return records


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for the emitted table files.")
    parser.add_argument(
        "--measure",
        action="store_true",
        help="Run real tiny synthetic fits to fill timing/RSS rows.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.measure:
        rows = measure_scalability([(60, 40), (120, 40), (120, 80)])
    else:
        rows = []
    table = build_scalability_table(rows)
    paths = write_table(table, args.out_dir, _BASENAME)
    for fmt, path in sorted(paths.items()):
        print(f"wrote {fmt}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
