#!/usr/bin/env python
"""Shared tidy-table emitter for the FactorGraph-ST results-table generators.

Every table generator in ``scripts/tables/`` builds a :class:`Table` — a small,
data-independent container of ``headers`` + ``rows`` — and renders it to one or
more on-disk formats (markdown / CSV / JSON) via :func:`write_table`. The same
:class:`Table` object is the structured value the unit tests assert on, so a
generator never needs a separate "for humans" vs "for tests" code path.

Design rules that the generators rely on:

* **Never fabricate numbers.** A table that cannot be filled without real data
  emits its SCHEMA (headers, zero rows) plus a ``pending`` marker via
  :func:`pending_table`, rather than inventing cell values.
* **Non-finite is not a number.** ``None`` and non-finite floats (``nan`` /
  ``inf`` — e.g. a "not evaluable" metric) render as the ``n/a`` sentinel in
  markdown and as empty / ``null`` in CSV / JSON; they are never written as a
  real ``0``.

This module is pure stdlib (no numpy / matplotlib), so it imports cleanly in the
numpy-only runtime env and under the test collector without any path shim.
"""

from __future__ import annotations

import csv
import io
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

Cell = object  # str | int | float | None — kept loose so generators stay simple.

_NA = "n/a"
_MD_FLOAT = "{:.4f}"
_CSV_FLOAT = "{:.6f}"
_RENDERERS = ("md", "csv", "json")


@dataclass
class Table:
    """A tidy table: ``headers`` + ``rows`` (one list of cells per row).

    ``rows`` may be empty — that is the canonical "schema only / pending data"
    shape (pair it with ``pending=True`` and a ``note``). Construction validates
    that every row has exactly ``len(headers)`` cells so a mis-shaped table fails
    loudly at build time instead of silently emitting a ragged file.
    """

    name: str
    headers: list[str]
    rows: list[list[Cell]] = field(default_factory=list)
    note: str = ""
    pending: bool = False

    def __post_init__(self) -> None:
        if not self.headers:
            raise ValueError("Table.headers must be non-empty")
        width = len(self.headers)
        for i, row in enumerate(self.rows):
            if len(row) != width:
                raise ValueError(
                    f"row {i} has {len(row)} cells, expected {width} (headers={self.headers})"
                )


def pending_table(name: str, headers: Sequence[str], note: str) -> Table:
    """Build a schema-only table (headers, zero rows) flagged as pending data.

    Use when a generator's real numbers are not yet available (deferred to the
    data-loading work): the consumer still sees the column contract and an
    explicit ``pending`` marker instead of a fabricated or silently-empty table.
    """
    if not note:
        raise ValueError("pending_table requires a non-empty note explaining what is pending")
    return Table(name=name, headers=list(headers), rows=[], note=note, pending=True)


def _is_nonfinite_float(value: Cell) -> bool:
    return isinstance(value, float) and not math.isfinite(value)


def _fmt_md(value: Cell) -> str:
    if value is None or _is_nonfinite_float(value):
        return _NA
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return _MD_FLOAT.format(value)
    return str(value)


def _fmt_csv(value: Cell) -> object:
    if value is None or _is_nonfinite_float(value):
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return _CSV_FLOAT.format(value)
    return value


def _json_cell(value: Cell) -> Cell:
    # Mirror robustness_harness.json_safe: non-finite floats -> null so the JSON
    # is portable (json.dumps would otherwise emit non-standard NaN/Infinity).
    if _is_nonfinite_float(value):
        return None
    return value


def to_markdown(table: Table) -> str:
    """Render ``table`` as a GitHub-flavored markdown pipe table.

    A pending table is preceded by an HTML comment carrying the ``note`` so the
    "data not yet filled" status survives in rendered markdown.
    """
    lines: list[str] = []
    if table.pending:
        lines.append(f"<!-- pending data: {table.note} -->")
    lines.append("| " + " | ".join(table.headers) + " |")
    lines.append("| " + " | ".join("---" for _ in table.headers) + " |")
    for row in table.rows:
        lines.append("| " + " | ".join(_fmt_md(c) for c in row) + " |")
    return "\n".join(lines) + "\n"


def to_csv(table: Table) -> str:
    """Render ``table`` as CSV (``\\r\\n`` line terminators, header row first)."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(table.headers)
    for row in table.rows:
        writer.writerow([_fmt_csv(c) for c in row])
    return buf.getvalue()


def to_json(table: Table, *, indent: int = 2) -> str:
    """Render ``table`` as a structured JSON object (non-finite floats -> null)."""
    payload = {
        "name": table.name,
        "headers": list(table.headers),
        "rows": [[_json_cell(c) for c in row] for row in table.rows],
        "pending": table.pending,
        "note": table.note,
    }
    return json.dumps(payload, indent=indent) + "\n"


def write_table(
    table: Table,
    out_dir: str | Path,
    basename: str,
    *,
    formats: Iterable[str] = _RENDERERS,
) -> dict[str, Path]:
    """Write ``table`` under ``out_dir`` as ``basename.<fmt>`` for each format.

    Returns a ``{format: path}`` map. Raises ``ValueError`` for an unknown
    format so a typo never silently drops an output.
    """
    renderers = {"md": to_markdown, "csv": to_csv, "json": to_json}
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for fmt in formats:
        if fmt not in renderers:
            raise ValueError(f"unknown format {fmt!r}; expected one of {sorted(renderers)}")
        path = out / f"{basename}.{fmt}"
        path.write_text(renderers[fmt](table), encoding="utf-8")
        written[fmt] = path
    return written


def finite_float(value: object) -> float | None:
    """Coerce ``value`` to a finite float, or ``None`` if it is not evaluable.

    Generators use this to turn metric outputs (which may be ``nan`` for
    "not evaluable") into a clean ``float`` cell or an explicit empty (``None``)
    cell — never a fabricated number.
    """
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def sorted_metric_names(records: Iterable[Mapping[str, object]]) -> list[str]:
    """Stable sorted union of metric keys across a sequence of metric mappings."""
    keys: set[str] = set()
    for record in records:
        keys.update(record.keys())
    return sorted(keys)
