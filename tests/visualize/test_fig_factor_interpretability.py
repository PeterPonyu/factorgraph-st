"""Smoke + unit tests for the factor-interpretability figure script (#310).

The numpy-only scoring helpers run everywhere; the matplotlib render is guarded
with ``importorskip`` because the base CI test env installs only ``.[test]``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from factorgraph_st.synth import generate_instance

_SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "visualize" / "fig_factor_interpretability.py"
)
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _load_module():
    spec = importlib.util.spec_from_file_location("_fig_factor_interpretability", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tiny_instance():
    return generate_instance(
        n_sections=2, n_spots_per_section=15, n_genes=12,
        K_shared=2, K_private=1, n_domains=3, k_nn=4, seed=0,
    )


def test_signature_enrichment_bounds_and_spike():
    mod = _load_module()
    n_genes, n_factors, top_n = 20, 3, 5
    W = np.zeros((n_genes, n_factors), dtype=np.float64)
    W[:, 0] = 1.0  # uniform -> enrichment at the top_n/n_genes floor
    W[:top_n, 1] = 1.0  # spike on exactly top_n genes -> 1.0
    # column 2 stays zero -> nan (not evaluable)
    enr = mod.signature_enrichment(W, top_n)
    assert enr[0] == pytest.approx(top_n / n_genes)
    assert enr[1] == pytest.approx(1.0)
    assert np.isnan(enr[2])


def test_compute_factor_interpretability_aligned_finite():
    mod = _load_module()
    inst = _tiny_instance()
    H = inst.Z_shared  # (n_spots, K_shared) nonnegative scores
    W = inst.W[:, : H.shape[1]]  # (n_genes, K_shared) matching factor count
    scores = mod.compute_factor_interpretability(H, W, inst.edges, top_n=4)
    assert set(scores) == {"coherence", "enrichment", "factor_index"}
    k = H.shape[1]
    assert scores["coherence"].shape == scores["enrichment"].shape == scores["factor_index"].shape == (k,)
    assert np.all(np.isfinite(scores["coherence"]))
    assert np.all(np.isfinite(scores["enrichment"]))  # nonneg loadings -> evaluable
    assert scores["factor_index"].tolist() == list(range(k))


def test_compute_factor_interpretability_rejects_mismatched_factor_axis():
    mod = _load_module()
    inst = _tiny_instance()
    with pytest.raises(ValueError, match="factor axis"):
        mod.compute_factor_interpretability(inst.Z_shared, inst.W[:, :1], inst.edges)


def test_render_interpretability_writes_valid_png(tmp_path: Path):
    pytest.importorskip("matplotlib")
    mod = _load_module()
    out = tmp_path / "interp.png"
    fig = mod.render_interpretability(mod._example_scores(), out)
    data = out.read_bytes()
    assert len(data) > 1000
    assert data[:8] == _PNG_MAGIC
    assert len(fig.axes) >= 2  # bar panel + scatter panel


def test_render_interpretability_handles_nan_enrichment(tmp_path: Path):
    pytest.importorskip("matplotlib")
    mod = _load_module()
    scores = mod._example_scores()
    scores["enrichment"][1] = np.nan  # a not-evaluable factor renders as a stub
    out = tmp_path / "interp_nan.png"
    mod.render_interpretability(scores, out)
    assert out.read_bytes()[:8] == _PNG_MAGIC


def test_render_interpretability_rejects_empty(tmp_path: Path):
    pytest.importorskip("matplotlib")
    mod = _load_module()
    empty = {
        "coherence": np.array([], dtype=np.float64),
        "enrichment": np.array([], dtype=np.float64),
        "factor_index": np.array([], dtype=np.int64),
    }
    with pytest.raises(ValueError, match="non-empty"):
        mod.render_interpretability(empty, tmp_path / "x.png")


def test_example_main_renders(tmp_path: Path):
    pytest.importorskip("matplotlib")
    mod = _load_module()
    out = tmp_path / "example.png"
    assert mod.main(["--example", "--out", str(out)]) == 0
    assert out.read_bytes()[:8] == _PNG_MAGIC
