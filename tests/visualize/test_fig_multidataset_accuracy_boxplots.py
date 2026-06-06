"""Unit + smoke tests for the multi-dataset accuracy boxplots + rank agg (#307).

The numpy/stdlib-only helpers are tested directly; the matplotlib renders are
guarded with ``importorskip`` and only checked for a valid PNG.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "visualize"
    / "fig_multidataset_accuracy_boxplots.py"
)
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "_fig_multidataset_accuracy_boxplots", _SCRIPT
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _records():
    # Dataset A: coords best, gnmf worst, on all metrics.
    # Dataset B: a deliberate gnmf/coords tie on ari (both mean 0.30).
    return [
        {"dataset": "A", "variant": "gnmf", "seed": 0, "ari": 0.10, "nmi": 0.10, "ami": 0.10},
        {"dataset": "A", "variant": "gnmf", "seed": 1, "ari": 0.20, "nmi": 0.20, "ami": 0.20},
        {"dataset": "A", "variant": "coords", "seed": 0, "ari": 0.50, "nmi": 0.50, "ami": 0.50},
        {"dataset": "A", "variant": "coords", "seed": 1, "ari": 0.60, "nmi": 0.60, "ami": 0.60},
        {"dataset": "A", "variant": "full", "seed": 0, "ari": 0.30, "nmi": 0.30, "ami": 0.30},
        {"dataset": "A", "variant": "full", "seed": 1, "ari": 0.40, "nmi": 0.40, "ami": 0.40},
        {"dataset": "B", "variant": "gnmf", "seed": 0, "ari": 0.30, "nmi": 0.10, "ami": 0.10},
        {"dataset": "B", "variant": "coords", "seed": 0, "ari": 0.30, "nmi": 0.50, "ami": 0.50},
        {"dataset": "B", "variant": "full", "seed": 0, "ari": 0.80, "nmi": 0.30, "ami": 0.30},
    ]


def test_load_records_concatenates(tmp_path):
    mod = _load_module()
    p1 = tmp_path / "a.json"
    p2 = tmp_path / "b.json"
    p1.write_text(json.dumps(_records()[:6]), encoding="utf-8")
    p2.write_text(json.dumps(_records()[6:]), encoding="utf-8")
    recs = mod.load_records([p1, p2])
    assert len(recs) == 9


def test_boxplot_series_pools_over_dataset_and_seed():
    mod = _load_module()
    series = mod.boxplot_series(_records(), "ari")
    # gnmf appears twice in A (0.10, 0.20) and once in B (0.30)
    assert sorted(series["gnmf"]) == [0.10, 0.20, 0.30]
    assert sorted(series["coords"]) == [0.30, 0.50, 0.60]
    assert sorted(series["full"]) == [0.30, 0.40, 0.80]


def test_boxplot_series_rejects_unknown_metric():
    mod = _load_module()
    with pytest.raises(ValueError, match="metric"):
        mod.boxplot_series(_records(), "foo")


def test_rank_variants_one_cell_basic_ordering():
    mod = _load_module()
    ranks = mod.rank_variants_one_cell({"a": 0.5, "b": 0.3, "c": 0.1})
    assert ranks == {"a": 1.0, "b": 2.0, "c": 3.0}


def test_rank_variants_one_cell_tie_gets_average_rank():
    mod = _load_module()
    # two tied for best -> both rank 1.5; the lower one -> rank 3
    ranks = mod.rank_variants_one_cell({"a": 0.5, "b": 0.5, "c": 0.1})
    assert ranks["a"] == pytest.approx(1.5)
    assert ranks["b"] == pytest.approx(1.5)
    assert ranks["c"] == pytest.approx(3.0)


def test_mean_rank_two_datasets_three_metrics_hand_checked():
    mod = _load_module()
    ranks = mod.mean_rank(_records())
    # Cells = {A,B} x {ari,nmi,ami} = 6.
    # Dataset A (all metrics identical ordering): coords=1, full=2, gnmf=3.
    #   -> 3 cells each.
    # Dataset B ari: full(0.80)=1, then coords(0.30)==gnmf(0.30) tie -> both 2.5.
    # Dataset B nmi: coords(0.50)=1, full(0.30)=2, gnmf(0.10)=3.
    # Dataset B ami: coords(0.50)=1, full(0.30)=2, gnmf(0.10)=3.
    # coords ranks: A[1,1,1], B[2.5,1,1] -> mean = 7.5/6 = 1.25
    # full   ranks: A[2,2,2], B[1,2,2]   -> mean = 11/6  ~= 1.8333
    # gnmf   ranks: A[3,3,3], B[2.5,3,3] -> mean = 17.5/6 ~= 2.9167
    assert ranks["coords"] == pytest.approx(7.5 / 6)
    assert ranks["full"] == pytest.approx(11.0 / 6)
    assert ranks["gnmf"] == pytest.approx(17.5 / 6)
    # lower = better: coords ranks best
    assert min(ranks, key=ranks.get) == "coords"


def test_mean_rank_rejects_unknown_metric():
    mod = _load_module()
    with pytest.raises(ValueError, match="metric"):
        mod.mean_rank(_records(), metrics=("ari", "foo"))


def test_render_boxplots_writes_png(tmp_path):
    pytest.importorskip("matplotlib")
    mod = _load_module()
    out = tmp_path / "box.png"
    mod.render_boxplots(_records(), out)
    assert out.is_file()
    assert out.read_bytes()[:8] == _PNG_MAGIC


def test_render_mean_rank_writes_png(tmp_path):
    pytest.importorskip("matplotlib")
    mod = _load_module()
    out = tmp_path / "rank.png"
    mod.render_mean_rank(_records(), out)
    assert out.is_file()
    assert out.read_bytes()[:8] == _PNG_MAGIC
