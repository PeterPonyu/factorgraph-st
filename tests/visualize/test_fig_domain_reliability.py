"""Smoke + unit tests for the per-domain reliability figure script (#329).

The perturbation-consensus computation runs on a TINY synthetic instance so the
numpy-only path stays fast; only the render needs matplotlib (importorskip).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from factorgraph_st.synth import generate_instance

_SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "visualize" / "fig_domain_reliability.py"
)
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _load_module():
    spec = importlib.util.spec_from_file_location("_fig_domain_reliability", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tiny_instance():
    return generate_instance(
        n_sections=2, n_spots_per_section=12, n_genes=12,
        K_shared=2, K_private=1, n_domains=3, k_nn=4, seed=0,
    )


def _compute(mod):
    inst = _tiny_instance()
    return inst, mod.compute_perturbation_consensus(
        inst.X, inst.coords, inst.edges, n_domains=3,
        n_perturb=4, drop_frac=0.1, n_factors=3, n_iter=25, seed=0,
    )


def test_consensus_is_valid_similarity_matrix():
    mod = _load_module()
    inst, res = _compute(mod)
    n = inst.X.shape[0]
    C = res["consensus"]
    assert C.shape == (n, n)
    assert np.all(np.isfinite(C))
    assert np.all((C >= 0.0) & (C <= 1.0))
    np.testing.assert_allclose(np.diag(C), 1.0)
    np.testing.assert_allclose(C, C.T)  # symmetric


def test_survival_aligned_and_in_range():
    mod = _load_module()
    _inst, res = _compute(mod)
    domain_ids = res["domain_ids"]
    survival = res["survival"]
    assert survival.shape == domain_ids.shape
    finite = survival[np.isfinite(survival)]
    assert np.all((finite >= 0.0) & (finite <= 1.0))


def test_consensus_is_deterministic():
    mod = _load_module()
    inst = _tiny_instance()
    kw = dict(n_domains=3, n_perturb=4, drop_frac=0.1, n_factors=3, n_iter=25, seed=0)
    a = mod.compute_perturbation_consensus(inst.X, inst.coords, inst.edges, **kw)
    b = mod.compute_perturbation_consensus(inst.X, inst.coords, inst.edges, **kw)
    np.testing.assert_array_equal(a["consensus"], b["consensus"])
    np.testing.assert_array_equal(a["reference_labels"], b["reference_labels"])


def test_compute_rejects_zero_perturb():
    mod = _load_module()
    inst = _tiny_instance()
    with pytest.raises(ValueError, match="n_perturb"):
        mod.compute_perturbation_consensus(inst.X, inst.coords, inst.edges, n_domains=3, n_perturb=0)


def test_render_reliability_writes_valid_png(tmp_path: Path):
    pytest.importorskip("matplotlib")
    mod = _load_module()
    out = tmp_path / "reliability.png"
    fig = mod.render_reliability(mod._example_result(), out)
    data = out.read_bytes()
    assert len(data) > 1000
    assert data[:8] == _PNG_MAGIC
    assert len(fig.axes) >= 2  # heatmap + survival bars (+ colorbar)


def test_render_reliability_rejects_non_square_consensus(tmp_path: Path):
    pytest.importorskip("matplotlib")
    mod = _load_module()
    bad = mod._example_result()
    bad["consensus"] = np.zeros((3, 4), dtype=np.float64)
    with pytest.raises(ValueError, match="square"):
        mod.render_reliability(bad, tmp_path / "x.png")


def test_render_from_computed_consensus(tmp_path: Path):
    pytest.importorskip("matplotlib")
    mod = _load_module()
    _inst, res = _compute(mod)
    out = tmp_path / "computed.png"
    mod.render_reliability(res, out)
    assert out.read_bytes()[:8] == _PNG_MAGIC


def test_example_main_renders(tmp_path: Path):
    pytest.importorskip("matplotlib")
    mod = _load_module()
    out = tmp_path / "example.png"
    assert mod.main(["--example", "--out", str(out)]) == 0
    assert out.read_bytes()[:8] == _PNG_MAGIC
