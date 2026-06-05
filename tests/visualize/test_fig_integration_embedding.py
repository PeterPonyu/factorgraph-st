"""Smoke + unit tests for the integration embedding figure (#323).

matplotlib renders are guarded with ``importorskip``. The joint GNMF fit + PCA
embeddings are computed on a TINY synthetic multi-section instance.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from factorgraph_st.synth import generate_instance

_SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "visualize" / "fig_integration_embedding.py"
)
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _load_module():
    spec = importlib.util.spec_from_file_location("_fig_integration_embedding", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _tiny_instance():
    return generate_instance(
        n_sections=2,
        n_spots_per_section=35,
        n_genes=18,
        K_shared=3,
        K_private=1,
        n_domains=3,
        k_nn=5,
        seed=0,
    )


def _compute(mod, inst):
    return mod.compute_integration_embedding(
        inst.X, inst.edges, inst.section_id, inst.domain_id,
        n_factors=4, n_iter=40, seed=0,
    )


def test_compute_integration_embedding_shapes_and_finite():
    mod = _load_module()
    inst = _tiny_instance()
    data = _compute(mod, inst)
    n = inst.X.shape[0]
    assert data.before.shape == (n, 2)
    assert data.after.shape == (n, 2)
    assert data.section_id.shape == (n,)
    assert data.domain_id.shape == (n,)
    assert np.all(np.isfinite(data.before))
    assert np.all(np.isfinite(data.after))


def test_compute_integration_embedding_is_deterministic():
    mod = _load_module()
    inst = _tiny_instance()
    a = _compute(mod, inst)
    b = _compute(mod, inst)
    np.testing.assert_array_equal(a.before, b.before)
    np.testing.assert_array_equal(a.after, b.after)


def test_pca_2d_always_two_columns():
    mod = _load_module()
    # A single-feature matrix still yields a 2-D (zero-padded) embedding.
    one_feat = np.linspace(0.0, 1.0, 6).reshape(6, 1)
    emb = mod._pca_2d(one_feat)
    assert emb.shape == (6, 2)
    assert np.all(np.isfinite(emb))


def test_render_integration_embedding_writes_valid_png(tmp_path: Path):
    pytest.importorskip("matplotlib")
    mod = _load_module()
    inst = _tiny_instance()
    data = _compute(mod, inst)
    out = tmp_path / "integration.png"
    fig = mod.render_integration_embedding(data, out)
    raw = out.read_bytes()
    assert len(raw) > 1000
    assert raw[:8] == _PNG_MAGIC
    # 2x2 grid of scatter panels.
    assert len(fig.axes) == 4


def test_example_main_renders(tmp_path: Path):
    pytest.importorskip("matplotlib")
    mod = _load_module()
    out = tmp_path / "example.png"
    assert mod.main(["--example", "--out", str(out)]) == 0
    assert out.read_bytes()[:8] == _PNG_MAGIC
