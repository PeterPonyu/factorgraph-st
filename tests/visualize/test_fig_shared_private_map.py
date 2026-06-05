"""Smoke + unit tests for the shared/private decomposition map figure (#317).

The matplotlib render is guarded with ``importorskip`` because the base CI env
installs only ``.[test]`` (numpy + scipy), not a plotting stack. The numpy-only
``compute_shared_private_map`` helper is exercised everywhere on a tiny synthetic
multi-section instance.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from factorgraph_st.synth import generate_instance

_SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "visualize" / "fig_shared_private_map.py"
)
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _load_module():
    spec = importlib.util.spec_from_file_location("_fig_shared_private_map", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _tiny_instance():
    return generate_instance(
        n_sections=3,
        n_spots_per_section=20,
        n_genes=12,
        K_shared=2,
        K_private=2,
        n_domains=3,
        k_nn=4,
        seed=0,
    )


def test_compute_shared_private_map_shapes_and_finite():
    mod = _load_module()
    inst = _tiny_instance()
    data = mod.compute_shared_private_map(inst.Z_shared, inst.Z_private, inst.section_id)
    n_sections = int(np.unique(inst.section_id).size)
    assert data.shared_overlap.shape == (inst.Z_shared.shape[1], n_sections)
    assert data.private_overlap.shape == (inst.Z_private.shape[1], n_sections)
    assert data.sections.tolist() == list(range(n_sections))
    assert np.all(np.isfinite(data.shared_overlap))
    assert np.all(np.isfinite(data.private_overlap))
    # Each factor's section-mass fractions sum to ~1 (mass conservation).
    np.testing.assert_allclose(data.shared_overlap.sum(axis=1), 1.0, atol=1e-5)
    np.testing.assert_allclose(data.private_overlap.sum(axis=1), 1.0, atol=1e-5)
    assert "shared_mean_active_sections" in data.separation
    assert "private_mean_active_sections" in data.separation


def test_private_factors_concentrate_more_than_shared():
    """Synthetic private factors live in one section; shared spread over all."""
    mod = _load_module()
    inst = _tiny_instance()
    data = mod.compute_shared_private_map(inst.Z_shared, inst.Z_private, inst.section_id)
    # Private factors are active in (on average) fewer sections than shared ones.
    assert data.separation["private_mean_active_sections"] < data.separation["shared_mean_active_sections"]


def test_compute_shared_private_map_is_deterministic():
    mod = _load_module()
    inst = _tiny_instance()
    a = mod.compute_shared_private_map(inst.Z_shared, inst.Z_private, inst.section_id)
    b = mod.compute_shared_private_map(inst.Z_shared, inst.Z_private, inst.section_id)
    np.testing.assert_array_equal(a.shared_overlap, b.shared_overlap)
    np.testing.assert_array_equal(a.private_overlap, b.private_overlap)


def test_render_shared_private_map_writes_valid_png(tmp_path: Path):
    pytest.importorskip("matplotlib")
    mod = _load_module()
    inst = _tiny_instance()
    data = mod.compute_shared_private_map(inst.Z_shared, inst.Z_private, inst.section_id)
    out = tmp_path / "shared_private.png"
    fig = mod.render_shared_private_map(data, out)
    raw = out.read_bytes()
    assert len(raw) > 1000
    assert raw[:8] == _PNG_MAGIC
    # Two heatmap panels (shared + private).
    assert len(fig.axes) >= 2


def test_example_main_renders(tmp_path: Path):
    pytest.importorskip("matplotlib")
    mod = _load_module()
    out = tmp_path / "example.png"
    assert mod.main(["--example", "--out", str(out)]) == 0
    assert out.read_bytes()[:8] == _PNG_MAGIC
