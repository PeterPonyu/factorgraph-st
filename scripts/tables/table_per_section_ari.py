#!/usr/bin/env python
"""#339 — per-section ARI table (rows = sections, one column per model variant).

For a multi-section dataset, reports the per-section domain-recovery ARI of each
model variant: rows are sections, columns are ``section`` + one per variant. This
exposes whether a method is uniformly good across sections or carried by a few.

Two entry points: :func:`build_per_section_ari_table` (data-independent, from an
in-memory ``variant -> {section: ari}`` mapping; :func:`per_section_ari` derives
that mapping from label arrays) and :func:`run_per_section_ari` (fits variants on
a TINY synthetic multi-section instance and builds the table).

Usage::

    python scripts/tables/table_per_section_ari.py --run --out-dir /tmp/tables
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

from table_emit import Table, finite_float, pending_table, write_table  # noqa: E402

_BASENAME = "per_section_ari"
_SECTION_HEADER = "section"


def per_section_ari(labels_true, labels_pred, section_id) -> dict[int, float]:
    """Per-section ARI of ``labels_pred`` vs ``labels_true``, keyed by section id.

    Splits the spots by ``section_id`` and computes
    ``adjusted_rand_index`` within each section (a section with ``<2`` spots is
    not evaluable and maps to ``nan``).
    """
    import numpy as np  # noqa: PLC0415

    from factorgraph_st.eval.metrics import adjusted_rand_index  # noqa: PLC0415

    true = np.asarray(labels_true)
    pred = np.asarray(labels_pred)
    sect = np.asarray(section_id)
    if not (true.shape[0] == pred.shape[0] == sect.shape[0]):
        raise ValueError("labels_true, labels_pred, section_id must have equal length")
    out: dict[int, float] = {}
    for s in np.unique(sect):
        mask = sect == s
        if int(mask.sum()) < 2:
            out[int(s)] = float("nan")
        else:
            out[int(s)] = float(adjusted_rand_index(true[mask], pred[mask]))
    return out


def build_per_section_ari_table(per_variant: Mapping[str, Mapping[int, float]]) -> Table:
    """Build the sections x variants ARI table from ``variant -> {section: ari}``.

    Sections (rows) are the sorted union of section ids across variants; columns
    are the sorted variant names. A missing ``(variant, section)`` ARI renders as
    ``n/a``. Empty input yields a schema-only pending table.
    """
    if not per_variant:
        return pending_table(
            "per-section ARI (#339)",
            [_SECTION_HEADER, "<variants>"],
            note="no per-section ARI computed yet; fill from a multi-section dataset",
        )
    variants = sorted(per_variant)
    sections = sorted({int(s) for scores in per_variant.values() for s in scores})
    headers = [_SECTION_HEADER, *variants]
    rows: list[list[object]] = []
    for section in sections:
        cells = [finite_float(per_variant[v].get(section)) for v in variants]
        rows.append([section, *cells])
    return Table(name="per-section ARI (#339)", headers=headers, rows=rows)


def run_per_section_ari(
    *,
    n_sections: int = 3,
    n_spots_per_section: int = 40,
    n_genes: int = 30,
    n_iter: int = 30,
    seed: int = 0,
) -> Table:
    """Fit variants on a TINY synthetic multi-section instance; build the table.

    Two variants are scored per-section against the ground-truth domains: the
    learned ``gnmf`` fit and a ``random`` baseline (a seeded random partition).
    Numbers are real (finite) but small — a demonstration that the table fills
    end-to-end without labeled data.
    """
    import numpy as np  # noqa: PLC0415

    from factorgraph_st.model.learned import fit_transform_gnmf  # noqa: PLC0415
    from factorgraph_st.synth.generator import generate_instance  # noqa: PLC0415

    synth = generate_instance(
        n_sections=n_sections,
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
    n_domains = int(np.unique(synth.domain_id).size)
    random_pred = np.random.default_rng(seed).integers(0, max(1, n_domains), size=synth.domain_id.shape[0])
    per_variant = {
        "gnmf": per_section_ari(synth.domain_id, outputs.domain_id, synth.section_id),
        "random": per_section_ari(synth.domain_id, random_pred, synth.section_id),
    }
    return build_per_section_ari_table(per_variant)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for the emitted table files.")
    parser.add_argument("--run", action="store_true", help="Fit a tiny synthetic multi-section instance.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.run:
        table = run_per_section_ari()
    else:
        table = pending_table(
            "per-section ARI (#339)",
            [_SECTION_HEADER, "<variants>"],
            note="pass --run to fill from a tiny synthetic fit, or call build_per_section_ari_table(...)",
        )
    paths = write_table(table, args.out_dir, _BASENAME)
    for fmt, path in sorted(paths.items()):
        print(f"wrote {fmt}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
