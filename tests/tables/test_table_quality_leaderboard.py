"""Tests for the #335 multi-metric domain-quality leaderboard generator."""

from __future__ import annotations

import json
import math

import pytest

from ._loader import load_script

mod = load_script("table_quality_leaderboard")


def _scores():
    return {
        "projection": {"ari": 0.2, "silhouette": 0.1},
        "gnmf": {"ari": 0.7, "silhouette": 0.3},
        "spatial_smooth": {"ari": 0.4, "silhouette": 0.2},
    }


def test_headers_include_rank_and_sorted_metrics():
    table = mod.build_quality_leaderboard_table(_scores())
    assert table.headers == ["rank", "variant", "ari", "silhouette"]


def test_ranked_descending_by_primary_default_first_metric():
    table = mod.build_quality_leaderboard_table(_scores())
    # default primary = first sorted metric = "ari"; descending order
    assert [r[1] for r in table.rows] == ["gnmf", "spatial_smooth", "projection"]
    assert [r[0] for r in table.rows] == [1, 2, 3]


def test_primary_override_changes_order():
    scores = {
        "a": {"ari": 0.9, "coh": 0.1},
        "b": {"ari": 0.1, "coh": 0.9},
    }
    table = mod.build_quality_leaderboard_table(scores, primary="coh")
    assert [r[1] for r in table.rows] == ["b", "a"]


def test_numeric_cells_finite():
    table = mod.build_quality_leaderboard_table(_scores())
    for row in table.rows:
        for cell in row[2:]:
            assert isinstance(cell, float) and math.isfinite(cell)


def test_missing_metric_renders_na():
    scores = {"a": {"ari": 0.5}, "b": {"ari": 0.4, "silhouette": 0.2}}
    table = mod.build_quality_leaderboard_table(scores)
    # column order: rank, variant, ari, silhouette; "a" lacks silhouette
    row_a = next(r for r in table.rows if r[1] == "a")
    assert row_a[3] is None


def test_unrankable_primary_sorts_to_bottom():
    scores = {"good": {"ari": 0.6}, "bad": {"ari": float("nan")}}
    table = mod.build_quality_leaderboard_table(scores, primary="ari")
    assert [r[1] for r in table.rows] == ["good", "bad"]


def test_unknown_primary_raises():
    with pytest.raises(ValueError, match="not among"):
        mod.build_quality_leaderboard_table(_scores(), primary="nope")


def test_empty_emits_pending():
    table = mod.build_quality_leaderboard_table({})
    assert table.pending is True and table.rows == []


def test_deterministic():
    a = mod.build_quality_leaderboard_table(_scores())
    b = mod.build_quality_leaderboard_table(_scores())
    assert a.rows == b.rows


def test_main_writes_example(tmp_path):
    assert mod.main(["--example", "--out-dir", str(tmp_path)]) == 0
    payload = json.loads((tmp_path / "quality_leaderboard.json").read_text())
    assert payload["headers"][0:2] == ["rank", "variant"]
