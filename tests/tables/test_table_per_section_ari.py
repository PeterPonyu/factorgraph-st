"""Tests for the #339 per-section ARI table generator."""

from __future__ import annotations

import math

import numpy as np

from ._loader import load_script

mod = load_script("table_per_section_ari")


def _per_variant():
    return {
        "gnmf": {0: 0.6, 1: 0.5},
        "random": {0: 0.05, 1: 0.02},
    }


def test_headers_section_then_variants():
    table = mod.build_per_section_ari_table(_per_variant())
    assert table.headers == ["section", "gnmf", "random"]


def test_rows_are_sections():
    table = mod.build_per_section_ari_table(_per_variant())
    assert [r[0] for r in table.rows] == [0, 1]


def test_numeric_cells_finite():
    table = mod.build_per_section_ari_table(_per_variant())
    for row in table.rows:
        for cell in row[1:]:
            assert isinstance(cell, float) and math.isfinite(cell)


def test_missing_section_renders_na():
    pv = {"a": {0: 0.5, 1: 0.4}, "b": {0: 0.3}}
    table = mod.build_per_section_ari_table(pv)
    row1 = next(r for r in table.rows if r[0] == 1)
    # variant "b" has no section 1 -> n/a
    assert row1[2] is None


def test_empty_emits_pending():
    table = mod.build_per_section_ari_table({})
    assert table.pending is True and table.rows == []


def test_per_section_ari_perfect_within_section():
    true = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    pred = true.copy()
    section = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    scores = mod.per_section_ari(true, pred, section)
    assert set(scores) == {0, 1}
    assert scores[0] == 1.0 and scores[1] == 1.0


def test_per_section_ari_singleton_section_nan():
    true = np.array([0, 1, 0])
    pred = np.array([0, 1, 0])
    section = np.array([0, 0, 1])  # section 1 has a single spot
    scores = mod.per_section_ari(true, pred, section)
    assert math.isnan(scores[1])


def test_deterministic():
    a = mod.build_per_section_ari_table(_per_variant())
    b = mod.build_per_section_ari_table(_per_variant())
    assert a.rows == b.rows


def test_run_per_section_ari_tiny_synthetic():
    table = mod.run_per_section_ari(n_sections=3, n_spots_per_section=20, n_genes=15, n_iter=10)
    assert table.headers == ["section", "gnmf", "random"]
    assert [r[0] for r in table.rows] == [0, 1, 2]


def test_main_run_writes(tmp_path):
    assert mod.main(["--run", "--out-dir", str(tmp_path)]) == 0
    assert (tmp_path / "per_section_ari.csv").exists()
