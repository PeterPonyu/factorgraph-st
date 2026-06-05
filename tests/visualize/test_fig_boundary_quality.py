"""Smoke + unit tests for the boundary-quality & contiguity figure script (#313).

The numpy-only contiguity/boundary helpers run everywhere; the matplotlib render
is guarded with ``importorskip``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from factorgraph_st.synth import generate_instance

_SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "visualize" / "fig_boundary_quality.py"
)
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _load_module():
    spec = importlib.util.spec_from_file_location("_fig_boundary_quality", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tiny_instance():
    return generate_instance(
        n_sections=2, n_spots_per_section=15, n_genes=12,
        K_shared=2, K_private=1, n_domains=3, k_nn=4, seed=0,
    )


def test_contiguity_perfect_single_chain():
    """A path graph with a single label is fully contiguous (one component)."""
    mod = _load_module()
    labels = np.zeros(5, dtype=np.int64)
    edges = np.array([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=np.int64)
    out = mod.contiguity(labels, edges)
    assert out["contiguity"] == pytest.approx(1.0)
    assert out["n_components"] == 1
    assert out["n_domains"] == 1
    assert out["fragmentation_ratio"] == pytest.approx(1.0)


def test_contiguity_fragmented_domain():
    """One label split across two disconnected pairs -> two components, 0.5 LCC."""
    mod = _load_module()
    labels = np.zeros(4, dtype=np.int64)
    edges = np.array([[0, 2], [1, 3]], dtype=np.int64)  # {0-1} and {2-3}, no link
    out = mod.contiguity(labels, edges)
    assert out["n_components"] == 2
    assert out["contiguity"] == pytest.approx(0.5)
    assert out["fragmentation_ratio"] == pytest.approx(2.0)


def test_compute_boundary_quality_keys_and_ranges():
    mod = _load_module()
    inst = _tiny_instance()
    # Predicted == truth: boundaries perfectly recovered, F1 == 1.
    summary = mod.compute_boundary_quality(inst.domain_id, inst.domain_id, inst.edges)
    expected = {
        "boundary_precision", "boundary_recall", "boundary_f1", "contiguity",
        "n_components", "n_pred_domains", "n_true_domains", "fragmentation_ratio",
    }
    assert set(summary) == expected
    assert summary["boundary_f1"] == pytest.approx(1.0)
    assert 0.0 <= summary["contiguity"] <= 1.0
    assert summary["n_pred_domains"] == summary["n_true_domains"]


def test_render_boundary_quality_writes_valid_png(tmp_path: Path):
    pytest.importorskip("matplotlib")
    mod = _load_module()
    out = tmp_path / "boundary.png"
    fig = mod.render_boundary_quality(mod._example_summary(), out)
    data = out.read_bytes()
    assert len(data) > 1000
    assert data[:8] == _PNG_MAGIC
    assert len(fig.axes) >= 2  # quality panel + counts panel


def test_render_boundary_quality_handles_nan_metric(tmp_path: Path):
    pytest.importorskip("matplotlib")
    mod = _load_module()
    summary = mod._example_summary()
    summary["boundary_precision"] = float("nan")  # not-evaluable -> 'n/a' stub
    out = tmp_path / "boundary_nan.png"
    mod.render_boundary_quality(summary, out)
    assert out.read_bytes()[:8] == _PNG_MAGIC


def test_render_from_computed_summary(tmp_path: Path):
    pytest.importorskip("matplotlib")
    mod = _load_module()
    inst = _tiny_instance()
    summary = mod.compute_boundary_quality(inst.domain_id, inst.domain_id, inst.edges)
    out = tmp_path / "computed.png"
    mod.render_boundary_quality(summary, out)
    assert out.read_bytes()[:8] == _PNG_MAGIC


def test_example_main_renders(tmp_path: Path):
    pytest.importorskip("matplotlib")
    mod = _load_module()
    out = tmp_path / "example.png"
    assert mod.main(["--example", "--out", str(out)]) == 0
    assert out.read_bytes()[:8] == _PNG_MAGIC
