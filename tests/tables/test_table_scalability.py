"""Tests for the #336 runtime + peak-memory scalability table generator."""

from __future__ import annotations

import math

from ._loader import load_script

mod = load_script("table_scalability")


def _rows():
    return [
        {"n_spots": 240, "n_genes": 80, "runtime_s": 0.9, "peak_rss_mb": 150.0},
        {"n_spots": 120, "n_genes": 40, "runtime_s": 0.3, "peak_rss_mb": 120.0},
        {"n_spots": 120, "n_genes": 80, "runtime_s": 0.5, "peak_rss_mb": 130.0},
    ]


def test_headers():
    table = mod.build_scalability_table(_rows())
    assert table.headers == ["n_spots", "n_genes", "runtime_s", "peak_rss_mb"]


def test_rows_sorted_by_size():
    table = mod.build_scalability_table(_rows())
    assert [(r[0], r[1]) for r in table.rows] == [(120, 40), (120, 80), (240, 80)]


def test_numeric_cells_finite():
    table = mod.build_scalability_table(_rows())
    for row in table.rows:
        assert isinstance(row[0], int) and isinstance(row[1], int)
        assert math.isfinite(row[2]) and math.isfinite(row[3])


def test_nan_rss_renders_na():
    table = mod.build_scalability_table(
        [{"n_spots": 100, "n_genes": 50, "runtime_s": 0.1, "peak_rss_mb": float("nan")}]
    )
    assert table.rows[0][3] is None


def test_empty_rows_ok():
    table = mod.build_scalability_table([])
    assert table.rows == []
    assert table.headers == ["n_spots", "n_genes", "runtime_s", "peak_rss_mb"]


def test_deterministic():
    a = mod.build_scalability_table(_rows())
    b = mod.build_scalability_table(_rows())
    assert a.rows == b.rows


def test_measure_scalability_real_tiny_fit():
    records = mod.measure_scalability([(20, 15), (30, 15)], n_iter=10)
    assert len(records) == 2
    assert [r["n_spots"] for r in records] == [40, 60]  # 2 sections each
    for r in records:
        assert math.isfinite(r["runtime_s"]) and r["runtime_s"] >= 0.0
    # The measured table is well-shaped and its size/runtime cells are finite.
    table = mod.build_scalability_table(records)
    assert len(table.rows) == 2
    for row in table.rows:
        assert math.isfinite(row[2])


def test_main_measure_writes(tmp_path):
    assert mod.main(["--measure", "--out-dir", str(tmp_path)]) == 0
    assert (tmp_path / "scalability.csv").exists()
