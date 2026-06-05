"""Smoke + unit tests for the #309 scalability curves figure.

The matplotlib render is guarded with ``importorskip``. The numpy-only data
helpers are exercised everywhere; one slow path runs a TINY real synthetic fit
through the shared ``measure_scalability`` helper (small ladder, small n_iter).
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "visualize" / "fig_scalability_curves.py"
)
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _load_module():
    spec = importlib.util.spec_from_file_location("_fig_scalability_curves", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _spots_records():
    # Deliberately out of order to exercise the sort.
    return [
        {"n_spots": 120, "n_genes": 40, "runtime_s": 0.5, "peak_rss_mb": 130.0},
        {"n_spots": 40, "n_genes": 40, "runtime_s": 0.2, "peak_rss_mb": 110.0},
        {"n_spots": 80, "n_genes": 40, "runtime_s": 0.35, "peak_rss_mb": 120.0},
    ]


def _genes_records():
    return [
        {"n_spots": 80, "n_genes": 90, "runtime_s": 0.6, "peak_rss_mb": 140.0},
        {"n_spots": 80, "n_genes": 30, "runtime_s": 0.25, "peak_rss_mb": 115.0},
        {"n_spots": 80, "n_genes": 60, "runtime_s": 0.4, "peak_rss_mb": 125.0},
    ]


def test_sweep_from_records_sorted_and_finite():
    mod = _load_module()
    sweep = mod.sweep_from_records(_spots_records(), "n_spots", "n_genes=40")
    assert sweep.vary == "n_spots"
    assert list(sweep.x) == [40, 80, 120]  # ascending
    assert sweep.runtime_s.shape == (3,)
    assert np.all(np.isfinite(sweep.runtime_s))
    assert np.all(np.isfinite(sweep.peak_rss_mb))
    assert sweep.fixed_label == "n_genes=40"


def test_sweep_from_records_rejects_bad_vary():
    mod = _load_module()
    with pytest.raises(ValueError, match="vary must be"):
        mod.sweep_from_records(_spots_records(), "n_cells", "x")


def test_sweep_none_rss_becomes_nan():
    mod = _load_module()
    sweep = mod.sweep_from_records(
        [{"n_spots": 40, "n_genes": 40, "runtime_s": 0.2, "peak_rss_mb": None}],
        "n_spots",
        "n_genes=40",
    )
    assert math.isnan(sweep.peak_rss_mb[0])


def test_build_curves_assembles_both_sweeps():
    mod = _load_module()
    curves = mod.build_curves(
        _spots_records(), _genes_records(),
        spots_fixed_label="n_genes=40", genes_fixed_label="n_spots=80",
    )
    assert curves.spots.vary == "n_spots"
    assert curves.genes.vary == "n_genes"
    assert list(curves.genes.x) == [30, 60, 90]


def test_render_scalability_curves_writes_valid_png(tmp_path: Path):
    pytest.importorskip("matplotlib")
    mod = _load_module()
    curves = mod.build_curves(
        _spots_records(), _genes_records(),
        spots_fixed_label="n_genes=40", genes_fixed_label="n_spots=80",
    )
    out = tmp_path / "curves.png"
    fig = mod.render_scalability_curves(curves, out)
    raw = out.read_bytes()
    assert len(raw) > 1000
    assert raw[:8] == _PNG_MAGIC
    # Two panels, each with a twin y-axis -> four axes total.
    assert len(fig.axes) == 4


def test_render_handles_all_nan_memory(tmp_path: Path):
    pytest.importorskip("matplotlib")
    mod = _load_module()
    nan_rows = [
        {"n_spots": 40, "n_genes": 40, "runtime_s": 0.2, "peak_rss_mb": None},
        {"n_spots": 80, "n_genes": 40, "runtime_s": 0.3, "peak_rss_mb": None},
    ]
    curves = mod.build_curves(
        nan_rows, nan_rows,
        spots_fixed_label="n_genes=40", genes_fixed_label="n_spots=80",
    )
    out = tmp_path / "curves_nan.png"
    mod.render_scalability_curves(curves, out)
    assert out.read_bytes()[:8] == _PNG_MAGIC


def test_measure_scalability_curves_real_tiny_fit():
    mod = _load_module()
    curves = mod.measure_scalability_curves(
        spot_ladder=(20, 30),
        gene_ladder=(15, 20),
        n_genes_fixed=15,
        n_spots_per_section_fixed=20,
        n_sections=2,
        n_iter=10,
        seed=0,
    )
    # spot ladder is per-section; n_spots = 2 sections * per_section.
    assert list(curves.spots.x) == [40, 60]
    assert list(curves.genes.x) == [15, 20]
    assert np.all(np.isfinite(curves.spots.runtime_s))
    assert np.all(curves.spots.runtime_s >= 0.0)
