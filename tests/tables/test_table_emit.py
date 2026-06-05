"""Unit tests for the shared tidy-table emitter (``scripts/tables/table_emit.py``)."""

from __future__ import annotations

import json
import math

import pytest

from ._loader import load_script

emit = load_script("table_emit")


def _table():
    return emit.Table(
        name="t",
        headers=["a", "b", "c"],
        rows=[["x", 1, 2.5], ["y", 3, float("nan")]],
    )


def test_table_rejects_ragged_rows():
    with pytest.raises(ValueError, match="expected 3"):
        emit.Table(name="t", headers=["a", "b", "c"], rows=[["x", 1]])


def test_table_rejects_empty_headers():
    with pytest.raises(ValueError, match="non-empty"):
        emit.Table(name="t", headers=[], rows=[])


def test_pending_table_shape_and_flag():
    t = emit.pending_table("p", ["a", "b"], note="waiting on data")
    assert t.pending is True
    assert t.rows == []
    assert t.headers == ["a", "b"]
    assert t.note == "waiting on data"


def test_pending_table_requires_note():
    with pytest.raises(ValueError, match="non-empty note"):
        emit.pending_table("p", ["a"], note="")


def test_markdown_header_rows_and_na():
    md = emit.to_markdown(_table())
    lines = md.strip().splitlines()
    assert lines[0] == "| a | b | c |"
    assert lines[1] == "| --- | --- | --- |"
    assert lines[2] == "| x | 1 | 2.5000 |"
    # nan renders as the n/a sentinel, never a fabricated 0.
    assert lines[3] == "| y | 3 | n/a |"


def test_markdown_pending_comment():
    md = emit.to_markdown(emit.pending_table("p", ["a"], note="later"))
    assert md.splitlines()[0] == "<!-- pending data: later -->"


def test_csv_roundtrip_and_blank_for_nonfinite():
    import csv
    import io

    rows = list(csv.reader(io.StringIO(emit.to_csv(_table()))))
    assert rows[0] == ["a", "b", "c"]
    assert rows[1] == ["x", "1", "2.500000"]
    assert rows[2] == ["y", "3", ""]  # nan -> empty cell


def test_json_structure_and_null_for_nonfinite():
    payload = json.loads(emit.to_json(_table()))
    assert payload["headers"] == ["a", "b", "c"]
    assert payload["pending"] is False
    assert payload["rows"][0] == ["x", 1, 2.5]
    assert payload["rows"][1] == ["y", 3, None]  # nan -> null


def test_write_table_emits_three_formats(tmp_path):
    paths = emit.write_table(_table(), tmp_path, "demo")
    assert set(paths) == {"md", "csv", "json"}
    for fmt, path in paths.items():
        assert path.exists()
        assert path.name == f"demo.{fmt}"
    assert json.loads(paths["json"].read_text())["name"] == "t"


def test_write_table_rejects_unknown_format(tmp_path):
    with pytest.raises(ValueError, match="unknown format"):
        emit.write_table(_table(), tmp_path, "demo", formats=["md", "xlsx"])


def test_finite_float_coerces_and_rejects_nonfinite():
    assert emit.finite_float(2) == 2.0
    assert emit.finite_float("3.5") == 3.5
    assert emit.finite_float(float("nan")) is None
    assert emit.finite_float(float("inf")) is None
    assert emit.finite_float(None) is None
    assert emit.finite_float("not-a-number") is None


def test_sorted_metric_names_union():
    names = emit.sorted_metric_names([{"b": 1, "a": 2}, {"c": 3, "a": 4}])
    assert names == ["a", "b", "c"]


def test_renders_are_deterministic():
    a, b = _table(), _table()
    assert emit.to_markdown(a) == emit.to_markdown(b)
    assert emit.to_csv(a) == emit.to_csv(b)
    assert emit.to_json(a) == emit.to_json(b)


def test_table_cells_are_finite_where_present():
    t = _table()
    for row in t.rows:
        for cell in row:
            if isinstance(cell, float):
                # the only float here is 2.5 (the nan is intentional and tested
                # separately); assert no inf sneaks in.
                assert not math.isinf(cell)
