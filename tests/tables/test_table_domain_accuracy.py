"""Tests for the #334 per-dataset spatial-domain accuracy table generator."""

from __future__ import annotations

import json
import math

import numpy as np

from ._loader import load_script

mod = load_script("table_domain_accuracy")


def _results():
    return {
        "ds2": {"gnmf": {"ari": 0.5, "nmi": 0.6, "ami": 0.55}},
        "ds1": {
            "projection": {"ari": 0.2, "nmi": 0.3, "ami": 0.25},
            "gnmf": {"ari": 0.7, "nmi": 0.65, "ami": 0.6},
        },
    }


def test_headers_and_sorted_rows():
    table = mod.build_domain_accuracy_table(_results())
    assert table.headers == ["dataset", "variant", "ARI", "NMI", "AMI"]
    # datasets then variants sorted -> (ds1, gnmf), (ds1, projection), (ds2, gnmf)
    assert [(r[0], r[1]) for r in table.rows] == [
        ("ds1", "gnmf"),
        ("ds1", "projection"),
        ("ds2", "gnmf"),
    ]


def test_numeric_cells_finite():
    table = mod.build_domain_accuracy_table(_results())
    for row in table.rows:
        for cell in row[2:]:
            assert isinstance(cell, float) and math.isfinite(cell)


def test_empty_results_emit_pending_schema():
    table = mod.build_domain_accuracy_table({})
    assert table.pending is True
    assert table.rows == []
    assert table.headers == ["dataset", "variant", "ARI", "NMI", "AMI"]
    assert table.note


def test_missing_metric_renders_na_not_zero():
    table = mod.build_domain_accuracy_table({"d": {"v": {"ari": 0.4, "nmi": 0.5}}})
    # ami missing -> None cell (n/a), never a fabricated 0.0
    assert table.rows[0][4] is None


def test_deterministic():
    a = mod.build_domain_accuracy_table(_results())
    b = mod.build_domain_accuracy_table(_results())
    assert a.rows == b.rows


def test_accuracy_from_labels_matches_metrics():
    true = np.array([0, 0, 1, 1, 2, 2])
    pred = np.array([0, 0, 1, 1, 2, 2])
    metrics = mod.accuracy_from_labels(true, pred)
    assert set(metrics) == {"ari", "nmi", "ami"}
    assert metrics["ari"] == 1.0
    assert metrics["nmi"] == 1.0


def test_main_writes_example(tmp_path):
    assert mod.main(["--example", "--out-dir", str(tmp_path)]) == 0
    payload = json.loads((tmp_path / "domain_accuracy.json").read_text())
    assert payload["headers"] == ["dataset", "variant", "ARI", "NMI", "AMI"]
    assert len(payload["rows"]) == 6
