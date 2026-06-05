"""Tests for the #338 factor diversity / coherence table generator."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ._loader import load_script

mod = load_script("table_factor_coherence")


def _inputs(seed=0):
    rng = np.random.default_rng(seed)
    W = rng.exponential(size=(12, 3)).astype(np.float64)
    H = rng.exponential(size=(20, 3)).astype(np.float64)
    # simple chain graph edges over the 20 spots
    src = np.arange(19, dtype=np.int64)
    dst = np.arange(1, 20, dtype=np.int64)
    edges = np.stack([src, dst])
    return W, H, edges


def test_headers_and_one_row_per_factor():
    W, H, edges = _inputs()
    table = mod.build_factor_coherence_table(W, H, edges)
    assert table.headers == ["factor", "top_gene_mass", "spatial_coherence", "max_redundancy"]
    assert [r[0] for r in table.rows] == [0, 1, 2]


def test_numeric_cells_finite():
    W, H, edges = _inputs()
    table = mod.build_factor_coherence_table(W, H, edges)
    for row in table.rows:
        assert math.isfinite(row[1])  # top_gene_mass
        assert math.isfinite(row[2])  # coherence
        assert math.isfinite(row[3])  # redundancy


def test_top_gene_mass_in_unit_interval():
    W, H, edges = _inputs()
    table = mod.build_factor_coherence_table(W, H, edges, top_k=4)
    for row in table.rows:
        assert 0.0 <= row[1] <= 1.0 + 1e-9


def test_shape_mismatch_raises():
    W = np.ones((10, 3))
    H = np.ones((10, 2))
    with pytest.raises(ValueError, match="factors"):
        mod.build_factor_coherence_table(W, H, np.empty((2, 0), dtype=np.int64))


def test_empty_factors_pending():
    table = mod.build_factor_coherence_table(
        np.empty((5, 0)), np.empty((7, 0)), np.empty((2, 0), dtype=np.int64)
    )
    assert table.pending is True and table.rows == []


def test_deterministic():
    a = mod.build_factor_coherence_table(*_inputs())
    b = mod.build_factor_coherence_table(*_inputs())
    assert a.rows == b.rows


def test_run_factor_coherence_tiny_synthetic():
    table = mod.run_factor_coherence(n_spots_per_section=20, n_genes=15, n_iter=10)
    assert len(table.rows) == 4  # K_shared(3) + K_private(1)
    for row in table.rows:
        assert math.isfinite(row[1]) and math.isfinite(row[2])


def test_main_run_writes(tmp_path):
    assert mod.main(["--run", "--out-dir", str(tmp_path)]) == 0
    assert (tmp_path / "factor_coherence.json").exists()
