"""Smoke + unit tests for the leave-one-section-out transfer figure (#318).

matplotlib renders are guarded with ``importorskip``. The GNMF LOSO protocol is
computed on a TINY synthetic multi-section instance so the numpy-only test stays
fast; only the render needs matplotlib.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from factorgraph_st.synth import generate_instance

_SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "visualize" / "fig_loso_transfer.py"
)
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _load_module():
    spec = importlib.util.spec_from_file_location("_fig_loso_transfer", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _tiny_instance():
    return generate_instance(
        n_sections=3,
        n_spots_per_section=40,
        n_genes=16,
        K_shared=3,
        K_private=1,
        n_domains=3,
        k_nn=5,
        seed=0,
    )


def _compute(mod, inst):
    return mod.compute_loso_transfer(
        inst.X, inst.edges, inst.section_id, inst.domain_id,
        n_factors=3, n_domains=3, n_iter=40, seed=0,
    )


def test_compute_loso_transfer_one_result_per_section():
    mod = _load_module()
    inst = _tiny_instance()
    res = _compute(mod, inst)
    n_sections = int(np.unique(inst.section_id).size)
    assert res.sections.shape == res.ari.shape == res.nmi.shape == (n_sections,)
    assert res.sections.tolist() == list(range(n_sections))
    # Scores are finite-or-nan and, when finite, within metric bounds.
    for arr, lo in ((res.ari, -0.5), (res.nmi, 0.0)):
        finite = arr[np.isfinite(arr)]
        assert np.all(finite >= lo)
        assert np.all(finite <= 1.0 + 1e-9)


def test_compute_loso_transfer_is_deterministic():
    mod = _load_module()
    inst = _tiny_instance()
    a = _compute(mod, inst)
    b = _compute(mod, inst)
    np.testing.assert_array_equal(a.ari, b.ari)
    np.testing.assert_array_equal(a.nmi, b.nmi)


def test_compute_loso_transfer_requires_two_sections():
    mod = _load_module()
    inst = generate_instance(
        n_sections=1, n_spots_per_section=30, n_genes=12,
        K_shared=2, K_private=1, n_domains=3, k_nn=4, seed=0,
    )
    with pytest.raises(ValueError, match="two sections"):
        mod.compute_loso_transfer(inst.X, inst.edges, inst.section_id, inst.domain_id)


def test_subset_edges_reindexes_and_filters():
    mod = _load_module()
    # nodes 0..4; keep {1,3,4}. Edge (0,1) drops (0 excluded); (1,3),(3,4) survive.
    edges = np.array([[0, 1, 3, 2], [1, 3, 4, 0]], dtype=np.int64)
    mask = np.array([False, True, False, True, True])
    sub = mod._subset_edges(edges, mask)
    # remap: 1->0, 3->1, 4->2. Surviving edges: (1,3)->(0,1), (3,4)->(1,2).
    assert sub.shape[0] == 2
    pairs = {tuple(p) for p in sub.T.tolist()}
    assert pairs == {(0, 1), (1, 2)}


def test_render_loso_transfer_writes_valid_png(tmp_path: Path):
    pytest.importorskip("matplotlib")
    mod = _load_module()
    inst = _tiny_instance()
    res = _compute(mod, inst)
    out = tmp_path / "loso.png"
    fig = mod.render_loso_transfer(res, out)
    raw = out.read_bytes()
    assert len(raw) > 1000
    assert raw[:8] == _PNG_MAGIC
    assert len(fig.axes) >= 1


def test_render_loso_transfer_handles_nan(tmp_path: Path):
    pytest.importorskip("matplotlib")
    mod = _load_module()
    data = mod.LosoTransfer(
        sections=np.array([0, 1]),
        ari=np.array([0.4, np.nan]),
        nmi=np.array([np.nan, 0.6]),
    )
    out = tmp_path / "loso_nan.png"
    mod.render_loso_transfer(data, out)
    assert out.read_bytes()[:8] == _PNG_MAGIC


def test_example_main_renders(tmp_path: Path):
    pytest.importorskip("matplotlib")
    mod = _load_module()
    out = tmp_path / "example.png"
    assert mod.main(["--example", "--out", str(out)]) == 0
    assert out.read_bytes()[:8] == _PNG_MAGIC
