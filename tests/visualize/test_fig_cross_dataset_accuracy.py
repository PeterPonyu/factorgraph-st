"""Unit + smoke tests for the cross-dataset accuracy panel (#322).

The numpy-only aggregation helper is tested directly; the matplotlib render is
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
    / "fig_cross_dataset_accuracy.py"
)
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _load_module():
    spec = importlib.util.spec_from_file_location("_fig_cross_dataset_accuracy", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _records():
    return [
        {"dataset": "A", "variant": "gnmf", "seed": 0, "ari": 0.10, "nmi": 0.2, "ami": 0.2},
        {"dataset": "A", "variant": "gnmf", "seed": 1, "ari": 0.20, "nmi": 0.3, "ami": 0.3},
        {"dataset": "A", "variant": "coords", "seed": 0, "ari": 0.50, "nmi": 0.6, "ami": 0.6},
        {"dataset": "B", "variant": "gnmf", "seed": 0, "ari": 0.40, "nmi": 0.5, "ami": 0.5},
    ]


def test_aggregate_records_mean_std_and_values():
    mod = _load_module()
    agg = mod.aggregate_records(_records(), "ari")
    a_gnmf = agg[("A", "gnmf")]
    assert a_gnmf["mean"] == pytest.approx(0.15)
    assert a_gnmf["std"] == pytest.approx(0.05)
    assert a_gnmf["n"] == 2
    assert sorted(a_gnmf["values"]) == [0.10, 0.20]
    assert agg[("B", "gnmf")]["mean"] == pytest.approx(0.40)
    assert agg[("A", "coords")]["n"] == 1


def test_aggregate_records_rejects_unknown_metric():
    mod = _load_module()
    with pytest.raises(ValueError, match="metric"):
        mod.aggregate_records(_records(), "foo")


def test_load_records_concatenates(tmp_path):
    mod = _load_module()
    p1 = tmp_path / "a.json"
    p2 = tmp_path / "b.json"
    p1.write_text(json.dumps(_records()[:2]), encoding="utf-8")
    p2.write_text(json.dumps(_records()[2:]), encoding="utf-8")
    recs = mod.load_records([p1, p2])
    assert len(recs) == 4


def test_render_writes_png(tmp_path):
    pytest.importorskip("matplotlib")
    mod = _load_module()
    out = tmp_path / "panel.png"
    mod.render_cross_dataset_accuracy(_records(), out, metric="ari")
    assert out.is_file()
    assert out.read_bytes()[:8] == _PNG_MAGIC
