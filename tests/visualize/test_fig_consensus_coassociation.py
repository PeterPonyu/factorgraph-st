"""Smoke + unit tests for the consensus co-association figure (#328).

matplotlib renders are guarded with ``importorskip``. The multi-run GNMF
consensus is computed on a TINY synthetic slice so the n x n co-association
matrix stays small and the numpy-only test stays fast.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from factorgraph_st.synth import generate_instance

_SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "visualize" / "fig_consensus_coassociation.py"
)
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _load_module():
    spec = importlib.util.spec_from_file_location("_fig_consensus_coassociation", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _tiny_instance():
    return generate_instance(
        n_sections=1,
        n_spots_per_section=40,
        n_genes=14,
        K_shared=3,
        K_private=1,
        n_domains=3,
        k_nn=5,
        seed=0,
    )


def _compute(mod, inst, configs=None):
    return mod.compute_consensus_coassociation(
        inst.X, inst.edges, n_domains=3, n_factors=3, n_iter=40, configs=configs,
    )


def test_coassociation_is_valid_similarity_matrix():
    mod = _load_module()
    inst = _tiny_instance()
    data = _compute(mod, inst)
    n = inst.X.shape[0]
    C = data.coassoc
    assert C.shape == (n, n)
    assert np.all(np.isfinite(C))
    assert np.all((C >= 0.0) & (C <= 1.0))
    np.testing.assert_allclose(C, C.T, atol=1e-9)  # symmetric
    np.testing.assert_allclose(np.diag(C), 1.0, atol=1e-9)  # self co-assoc = 1
    assert data.consensus.shape == (n,)
    assert data.n_runs == len(mod.DEFAULT_CONFIGS)


def test_coassociation_is_deterministic():
    mod = _load_module()
    inst = _tiny_instance()
    a = _compute(mod, inst)
    b = _compute(mod, inst)
    np.testing.assert_array_equal(a.coassoc, b.coassoc)
    np.testing.assert_array_equal(a.consensus, b.consensus)


def test_identical_runs_give_binary_coassociation():
    """With one repeated config, co-association is exactly a 0/1 block matrix."""
    mod = _load_module()
    inst = _tiny_instance()
    data = _compute(mod, inst, configs=[(1.0, 0)])
    uniq = np.unique(data.coassoc)
    assert set(uniq.tolist()).issubset({0.0, 1.0})


def test_compute_rejects_empty_configs():
    mod = _load_module()
    inst = _tiny_instance()
    with pytest.raises(ValueError, match="configs"):
        mod.compute_consensus_coassociation(inst.X, inst.edges, configs=[])


def test_render_consensus_writes_valid_png(tmp_path: Path):
    pytest.importorskip("matplotlib")
    mod = _load_module()
    inst = _tiny_instance()
    data = _compute(mod, inst)
    out = tmp_path / "consensus.png"
    fig = mod.render_consensus_coassociation(data, out, coords=inst.coords)
    raw = out.read_bytes()
    assert len(raw) > 1000
    assert raw[:8] == _PNG_MAGIC
    # heatmap + spatial map (+ colorbar axis).
    assert len(fig.axes) >= 2


def test_render_consensus_heatmap_only(tmp_path: Path):
    pytest.importorskip("matplotlib")
    mod = _load_module()
    inst = _tiny_instance()
    data = _compute(mod, inst)
    out = tmp_path / "consensus_heat.png"
    mod.render_consensus_coassociation(data, out, coords=None)
    assert out.read_bytes()[:8] == _PNG_MAGIC


def test_example_main_renders(tmp_path: Path):
    pytest.importorskip("matplotlib")
    mod = _load_module()
    out = tmp_path / "example.png"
    assert mod.main(["--example", "--out", str(out)]) == 0
    assert out.read_bytes()[:8] == _PNG_MAGIC
