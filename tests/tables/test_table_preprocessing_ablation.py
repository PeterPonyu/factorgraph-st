"""Tests for the #337 preprocessing x factor-rank ablation table generator."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ._loader import load_script

mod = load_script("table_preprocessing_ablation")


def _records():
    return [
        {"normalization": "log1p", "k": 6, "reconstruction_error": 0.3, "factor_redundancy": 0.1},
        {"normalization": "none", "k": 4, "reconstruction_error": 0.5, "factor_redundancy": 0.2},
        {"normalization": "none", "k": 6, "reconstruction_error": 0.4, "factor_redundancy": 0.15},
    ]


def test_headers():
    table = mod.build_ablation_table(_records())
    assert table.headers == ["normalization", "k", "reconstruction_error", "factor_redundancy"]


def test_rows_sorted_by_norm_then_k():
    table = mod.build_ablation_table(_records())
    assert [(r[0], r[1]) for r in table.rows] == [("log1p", 6), ("none", 4), ("none", 6)]


def test_numeric_cells_finite():
    table = mod.build_ablation_table(_records())
    for row in table.rows:
        assert math.isfinite(row[2]) and math.isfinite(row[3])


def test_normalize_schemes():
    X = np.array([[0.0, 10.0], [5.0, 5.0]], dtype=np.float32)
    assert np.allclose(mod._normalize(X, "none"), X)
    assert np.allclose(mod._normalize(X, "log1p"), np.log1p(X))
    out = mod._normalize(X, "total_log1p")
    assert out.shape == X.shape and np.isfinite(out).all()


def test_normalize_unknown_raises():
    with pytest.raises(ValueError, match="unknown normalization"):
        mod._normalize(np.zeros((2, 2), dtype=np.float32), "bogus")


def test_run_ablation_tiny_synthetic():
    records = mod.run_ablation(
        normalizations=("none", "log1p"),
        k_grid=(3, 4),
        n_spots_per_section=20,
        n_genes=15,
        n_iter=10,
    )
    assert len(records) == 4  # 2 norms x 2 k
    for r in records:
        assert math.isfinite(r["reconstruction_error"])
        assert math.isfinite(r["factor_redundancy"])
    table = mod.build_ablation_table(records)
    assert len(table.rows) == 4


def test_run_ablation_deterministic():
    a = mod.run_ablation(normalizations=("none",), k_grid=(4,), n_spots_per_section=20, n_genes=15, n_iter=10)
    b = mod.run_ablation(normalizations=("none",), k_grid=(4,), n_spots_per_section=20, n_genes=15, n_iter=10)
    assert a[0]["reconstruction_error"] == b[0]["reconstruction_error"]


def test_main_run_writes(tmp_path):
    assert mod.main(["--run", "--out-dir", str(tmp_path)]) == 0
    assert (tmp_path / "preprocessing_ablation.md").exists()
