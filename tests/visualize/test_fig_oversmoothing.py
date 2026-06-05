"""Smoke + unit tests for the over-smoothing exposure figure script (#331).

matplotlib renders are guarded with ``importorskip``. The GNMF sweep is computed
on a TINY synthetic instance so the numpy-only test stays fast; only the render
needs matplotlib.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from factorgraph_st.synth import generate_instance

_SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "visualize" / "fig_oversmoothing.py"
)
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _load_module():
    spec = importlib.util.spec_from_file_location("_fig_oversmoothing", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tiny_instance():
    return generate_instance(
        n_sections=2,
        n_spots_per_section=15,
        n_genes=12,
        K_shared=2,
        K_private=1,
        n_domains=3,
        k_nn=4,
        seed=0,
    )


def test_compute_oversmoothing_sweep_returns_aligned_finite_arrays():
    mod = _load_module()
    inst = _tiny_instance()
    lams = [0.0, 1.0, 5.0]
    lam_arr, coherence, signal = mod.compute_oversmoothing_sweep(
        inst.X, inst.edges, lams, n_factors=3, n_iter=25, seed=0
    )
    assert lam_arr.shape == coherence.shape == signal.shape == (3,)
    assert np.all(np.isfinite(coherence))
    assert np.all(np.isfinite(signal))
    assert np.all((signal >= 0.0) & (signal <= 1.0))
    assert lam_arr.tolist() == lams


def test_compute_oversmoothing_sweep_is_deterministic():
    mod = _load_module()
    inst = _tiny_instance()
    a = mod.compute_oversmoothing_sweep(inst.X, inst.edges, [0.0, 2.0], n_factors=3, n_iter=25, seed=0)
    b = mod.compute_oversmoothing_sweep(inst.X, inst.edges, [0.0, 2.0], n_factors=3, n_iter=25, seed=0)
    np.testing.assert_array_equal(a[1], b[1])
    np.testing.assert_array_equal(a[2], b[2])


def test_compute_oversmoothing_sweep_rejects_empty_lams():
    mod = _load_module()
    inst = _tiny_instance()
    with pytest.raises(ValueError, match="lams"):
        mod.compute_oversmoothing_sweep(inst.X, inst.edges, [])


def test_render_oversmoothing_writes_valid_png(tmp_path: Path):
    pytest.importorskip("matplotlib")
    mod = _load_module()
    lams, coherence, signal = mod._example_sweep()
    out = tmp_path / "oversmoothing.png"
    fig = mod.render_oversmoothing(lams, coherence, signal, out)
    data = out.read_bytes()
    assert len(data) > 1000
    assert data[:8] == _PNG_MAGIC
    # sweep axes + twin axes + frontier axes (+ colorbar).
    assert len(fig.axes) >= 2


def test_render_oversmoothing_rejects_mismatched_lengths(tmp_path: Path):
    pytest.importorskip("matplotlib")
    mod = _load_module()
    with pytest.raises(ValueError, match="equal length"):
        mod.render_oversmoothing(np.array([0.0, 1.0]), np.array([0.1]), np.array([0.9]), tmp_path / "x.png")


def test_render_from_computed_sweep(tmp_path: Path):
    """End-to-end: compute on a tiny instance, then render the real sweep."""
    pytest.importorskip("matplotlib")
    mod = _load_module()
    inst = _tiny_instance()
    lams, coherence, signal = mod.compute_oversmoothing_sweep(
        inst.X, inst.edges, [0.0, 1.0, 5.0], n_factors=3, n_iter=25, seed=0
    )
    out = tmp_path / "computed.png"
    mod.render_oversmoothing(lams, coherence, signal, out)
    assert out.read_bytes()[:8] == _PNG_MAGIC


def test_example_main_renders(tmp_path: Path):
    pytest.importorskip("matplotlib")
    mod = _load_module()
    out = tmp_path / "example.png"
    assert mod.main(["--example", "--out", str(out)]) == 0
    assert out.read_bytes()[:8] == _PNG_MAGIC
