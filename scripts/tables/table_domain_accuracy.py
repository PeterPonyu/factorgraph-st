#!/usr/bin/env python
"""#334 — per-dataset spatial-domain accuracy table (datasets x method-variant).

Emits a tidy long-form table with one row per ``(dataset, variant)`` and the
three label-concordance metrics ARI / NMI / AMI as columns. The generator is
**data-independent**: it consumes an in-memory ``results`` mapping (precomputed
metrics, or label arrays via :func:`accuracy_from_labels`) so the real numbers
fill later (deferred to the labeled-data work, #355 / #133) without touching this
code.

Input shape::

    results = {
        "<dataset>": {
            "<variant>": {"ari": 0.71, "nmi": 0.68, "ami": 0.66},
            ...
        },
        ...
    }

With no results the table is emitted SCHEMA-ONLY with a pending-data marker
(never fabricated zeros).

Usage::

    python scripts/tables/table_domain_accuracy.py --example --out-dir /tmp/tables
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
for _p in (str(_REPO / "src"), str(_REPO / "scripts" / "tables")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from table_emit import Table, finite_float, pending_table, write_table  # noqa: E402

HEADERS = ["dataset", "variant", "ARI", "NMI", "AMI"]
_METRIC_KEYS = ("ari", "nmi", "ami")
_BASENAME = "domain_accuracy"


def accuracy_from_labels(labels_true, labels_pred) -> dict[str, float]:
    """Compute the ``{ari, nmi, ami}`` metric dict from two label arrays.

    Thin reuse of ``factorgraph_st.eval.metrics`` so real / synthetic callers do
    not re-derive concordance. Non-evaluable metrics stay ``nan`` (the emitter
    renders them as ``n/a``, never a fabricated number).
    """
    from factorgraph_st.eval.metrics import (  # noqa: PLC0415
        adjusted_mutual_information,
        adjusted_rand_index,
        normalized_mutual_information,
    )

    return {
        "ari": float(adjusted_rand_index(labels_true, labels_pred)),
        "nmi": float(normalized_mutual_information(labels_true, labels_pred)),
        "ami": float(adjusted_mutual_information(labels_true, labels_pred)),
    }


def build_domain_accuracy_table(
    results: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> Table:
    """Flatten ``results`` to a tidy ARI/NMI/AMI table (rows sorted deterministically).

    Datasets and variants are sorted by name so the table is reproducible
    regardless of input dict ordering. An empty ``results`` yields a schema-only
    pending table.
    """
    if not results:
        return pending_table(
            "per-dataset spatial-domain accuracy (#334)",
            HEADERS,
            note="no labeled datasets available yet; fill via accuracy_from_labels once #355/#133 land",
        )
    rows: list[list[object]] = []
    for dataset in sorted(results):
        variants = results[dataset]
        for variant in sorted(variants):
            metrics = variants[variant]
            rows.append(
                [dataset, variant, *(finite_float(metrics.get(k)) for k in _METRIC_KEYS)]
            )
    return Table(name="per-dataset spatial-domain accuracy (#334)", headers=HEADERS, rows=rows)


def _example_results() -> dict[str, dict[str, dict[str, float]]]:
    """Illustrative (clearly-synthetic) results spanning two datasets x three variants."""
    return {
        "synth_A": {
            "gnmf": {"ari": 0.58, "nmi": 0.61, "ami": 0.59},
            "projection": {"ari": 0.24, "nmi": 0.31, "ami": 0.28},
            "spatial_smooth": {"ari": 0.41, "nmi": 0.45, "ami": 0.43},
        },
        "synth_B": {
            "gnmf": {"ari": 0.52, "nmi": 0.55, "ami": 0.53},
            "projection": {"ari": 0.19, "nmi": 0.27, "ami": 0.24},
            "spatial_smooth": {"ari": 0.37, "nmi": 0.42, "ami": 0.40},
        },
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=None, help="JSON dataset->variant->{ari,nmi,ami}.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for the emitted table files.")
    parser.add_argument("--example", action="store_true", help="Build from built-in illustrative results.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.example:
        results: Mapping[str, Mapping[str, Mapping[str, float]]] = _example_results()
    elif args.results is not None:
        with open(args.results, encoding="utf-8") as handle:
            results = json.load(handle)
    else:
        results = {}
    table = build_domain_accuracy_table(results)
    paths = write_table(table, args.out_dir, _BASENAME)
    for fmt, path in sorted(paths.items()):
        print(f"wrote {fmt}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
