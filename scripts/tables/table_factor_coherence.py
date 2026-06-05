#!/usr/bin/env python
"""#338 — factor diversity / coherence table (per-factor program quality).

One row per learned factor, reporting per-factor program-quality signals:

* ``top_gene_mass`` — fraction of the factor's gene-loading L1 mass carried by
  its top genes (default top-10). High = a focused, interpretable program; low =
  a diffuse one. Diversity across factors is read down this column.
* ``spatial_coherence`` — Moran's I of the factor's per-spot activation over the
  spatial graph (``edges``). High = spatially smooth activation.
* ``max_redundancy`` — the factor's largest absolute correlation with any OTHER
  factor's activation. High = a near-duplicate program (low diversity).

Two entry points: :func:`build_factor_coherence_table` (data-independent, from
in-memory ``W`` / ``H`` / ``edges``) and :func:`run_factor_coherence` (fits the
model on a TINY synthetic instance and builds the table from its factors).

Usage::

    python scripts/tables/table_factor_coherence.py --run --out-dir /tmp/tables
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
for _p in (str(_REPO / "src"), str(_REPO / "scripts" / "tables")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from table_emit import Table, finite_float, pending_table, write_table  # noqa: E402

HEADERS = ["factor", "top_gene_mass", "spatial_coherence", "max_redundancy"]
_BASENAME = "factor_coherence"


def _top_gene_mass(loading, top_k: int):
    import numpy as np  # noqa: PLC0415

    col = np.abs(np.asarray(loading, dtype=np.float64))
    total = float(col.sum())
    if total <= 0.0:
        return float("nan")
    k = min(int(top_k), col.size)
    top = np.sort(col)[::-1][:k]
    return float(top.sum() / total)


def build_factor_coherence_table(W, H, edges, *, top_k: int = 10) -> Table:
    """Build the per-factor quality table from gene loadings ``W`` and scores ``H``.

    ``W`` is ``(n_genes, K)``, ``H`` is ``(n_spots, K)``, ``edges`` is the
    ``(2, n_edges)`` spatial adjacency. Per-factor metrics reuse
    ``factorgraph_st.eval.metrics`` (``morans_i`` for coherence, the shared
    absolute-correlation matrix for redundancy). Non-evaluable values stay
    ``nan`` -> ``n/a``. An empty factor set yields a schema-only pending table.
    """
    import numpy as np  # noqa: PLC0415

    from factorgraph_st.eval.metrics import _abs_corr_matrix, morans_i  # noqa: PLC0415

    W = np.asarray(W, dtype=np.float64)
    H = np.asarray(H, dtype=np.float64)
    if W.ndim != 2 or H.ndim != 2:
        raise ValueError("W and H must be 2D")
    if W.shape[1] != H.shape[1]:
        raise ValueError(f"W has {W.shape[1]} factors but H has {H.shape[1]}")
    edges = np.asarray(edges, dtype=np.int64)
    n_factors = W.shape[1]
    if n_factors == 0:
        return pending_table(
            "per-factor diversity / coherence (#338)",
            HEADERS,
            note="no factors provided; fill from a fitted model's W/H",
        )

    if n_factors >= 2:
        corr = _abs_corr_matrix(H, H)
        np.fill_diagonal(corr, 0.0)
        max_redundancy = corr.max(axis=1)
    else:
        max_redundancy = np.array([float("nan")])

    rows: list[list[object]] = []
    for k in range(n_factors):
        coherence = morans_i(H[:, k], edges) if edges.size else float("nan")
        rows.append(
            [
                k,
                finite_float(_top_gene_mass(W[:, k], top_k)),
                finite_float(coherence),
                finite_float(float(max_redundancy[k])),
            ]
        )
    return Table(name="per-factor diversity / coherence (#338)", headers=HEADERS, rows=rows)


def run_factor_coherence(
    *,
    n_spots_per_section: int = 40,
    n_genes: int = 30,
    n_iter: int = 30,
    top_k: int = 10,
    seed: int = 0,
) -> Table:
    """Fit GNMF on a TINY synthetic instance and build the per-factor table."""
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
    outputs, _ = fit_transform_gnmf(
        synth.X, synth.edges, K_shared=3, K_private=1, n_domains=4, n_iter=n_iter, seed=seed
    )
    return build_factor_coherence_table(outputs.W, outputs.H, synth.edges, top_k=top_k)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for the emitted table files.")
    parser.add_argument("--run", action="store_true", help="Fit a tiny synthetic instance and build the table.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.run:
        table = run_factor_coherence()
    else:
        table = pending_table(
            "per-factor diversity / coherence (#338)",
            HEADERS,
            note="pass --run to fill from a tiny synthetic fit, or call build_factor_coherence_table(W, H, edges)",
        )
    paths = write_table(table, args.out_dir, _BASENAME)
    for fmt, path in sorted(paths.items()):
        print(f"wrote {fmt}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
