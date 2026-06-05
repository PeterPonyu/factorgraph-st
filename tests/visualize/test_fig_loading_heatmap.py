"""Smoke + unit tests for the factor-loading heatmap figure script (#321).

The matplotlib render is guarded with ``importorskip("matplotlib")`` because the
base CI test env installs only ``.[test]`` (numpy + scipy), not a plotting
stack. The pure ranking helpers are exercised without matplotlib so they stay
covered everywhere.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "visualize" / "fig_loading_heatmap.py"
)
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _load_module():
    spec = importlib.util.spec_from_file_location("_fig_loading_heatmap", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _block_loadings() -> np.ndarray:
    """Tiny W where factor k loads on its own disjoint gene block."""
    W = np.full((9, 3), 0.01, dtype=np.float32)
    for k in range(3):
        W[3 * k : 3 * (k + 1), k] = np.array([3.0, 2.0, 1.0], dtype=np.float32) + k
    return W


def test_rank_top_genes_picks_block_per_factor():
    mod = _load_module()
    ranked = mod.rank_top_genes(_block_loadings(), top_n=2)
    assert len(ranked) == 3  # one list per factor
    # Factor 0's top genes are its own block (indices 0,1), in descending loading.
    names = [name for name, _load in ranked[0]]
    assert names == ["g0", "g1"]
    loads = [load for _name, load in ranked[0]]
    assert loads[0] >= loads[1]


def test_rank_top_genes_respects_custom_names_and_validates_length():
    mod = _load_module()
    names = [f"GENE_{i}" for i in range(9)]
    ranked = mod.rank_top_genes(_block_loadings(), top_n=1, gene_names=names)
    assert ranked[1][0][0] == "GENE_3"  # factor 1 block starts at index 3
    with pytest.raises(ValueError, match="gene_names"):
        mod.rank_top_genes(_block_loadings(), gene_names=["only_one"])


def test_top_gene_union_is_first_seen_order():
    mod = _load_module()
    union = mod.top_gene_union(_block_loadings(), top_n=1)
    # One gene per factor, first-seen scanning factors 0,1,2.
    assert union.tolist() == [0, 3, 6]


def test_render_loading_heatmap_writes_valid_png(tmp_path: Path):
    pytest.importorskip("matplotlib")
    mod = _load_module()
    out = tmp_path / "heatmap.png"
    fig = mod.render_loading_heatmap(_block_loadings(), out, top_n=2)
    assert out.exists()
    data = out.read_bytes()
    assert len(data) > 1000
    assert data[:8] == _PNG_MAGIC
    # Heatmap axes + ranked-text axes + colorbar axes.
    assert len(fig.axes) >= 2
    assert np.all(np.isfinite([c for ax in fig.axes for c in ax.get_position().bounds]))


def test_render_rejects_empty_W(tmp_path: Path):
    pytest.importorskip("matplotlib")
    mod = _load_module()
    with pytest.raises(ValueError, match="non-empty"):
        mod.render_loading_heatmap(np.empty((0, 0)), tmp_path / "x.png")


def test_example_main_renders(tmp_path: Path):
    pytest.importorskip("matplotlib")
    mod = _load_module()
    out = tmp_path / "example.png"
    assert mod.main(["--example", "--out", str(out)]) == 0
    assert out.read_bytes()[:8] == _PNG_MAGIC
