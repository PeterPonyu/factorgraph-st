"""Smoke + unit tests for the #311 multi-section integration scorecard.

matplotlib renders are guarded with ``importorskip``. The joint GNMF fit + the
silhouette-based scores are computed on a TINY synthetic multi-section instance.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from factorgraph_st.synth import generate_instance

_SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "visualize" / "fig_integration_scorecard.py"
)
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _load_module():
    spec = importlib.util.spec_from_file_location("_fig_integration_scorecard", _SCRIPT)
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
    return mod.compute_integration_scorecard(
        inst.X, inst.edges, inst.section_id, inst.domain_id,
        n_factors=4, n_iter=40, seed=0,
    )


def test_scorecard_scores_in_unit_range_and_finite():
    mod = _load_module()
    card = _compute(mod, _tiny_instance())
    for attr in ("section_mixing", "mixing_gain", "domain_conservation", "conservation_gain", "overall"):
        value = getattr(card, attr)
        assert math.isfinite(value), attr
        assert 0.0 <= value <= 1.0, (attr, value)


def test_scorecard_overall_is_group_mean():
    mod = _load_module()
    card = _compute(mod, _tiny_instance())
    mixing = (card.section_mixing + card.mixing_gain) / 2.0
    conservation = (card.domain_conservation + card.conservation_gain) / 2.0
    assert card.overall == pytest.approx((mixing + conservation) / 2.0)


def test_scorecard_is_deterministic():
    mod = _load_module()
    inst = _tiny_instance()
    a = _compute(mod, inst)
    b = _compute(mod, inst)
    assert a == b


def test_sil01_maps_unit_interval():
    mod = _load_module()
    assert mod._sil01(-1.0) == pytest.approx(0.0)
    assert mod._sil01(0.0) == pytest.approx(0.5)
    assert mod._sil01(1.0) == pytest.approx(1.0)
    assert math.isnan(mod._sil01(float("nan")))


def test_gain_clamped_to_unit_interval():
    mod = _load_module()
    assert mod._gain(0.8, 0.5) == pytest.approx(0.3)
    assert mod._gain(0.4, 0.9) == 0.0  # negative improvement clamps to 0
    assert mod._gain(1.0, 0.0) == 1.0


def test_render_integration_scorecard_writes_valid_png(tmp_path: Path):
    pytest.importorskip("matplotlib")
    mod = _load_module()
    card = _compute(mod, _tiny_instance())
    out = tmp_path / "scorecard.png"
    fig = mod.render_integration_scorecard(card, out)
    raw = out.read_bytes()
    assert len(raw) > 1000
    assert raw[:8] == _PNG_MAGIC
    # Single scorecard axis with one bar per scorecard metric.
    assert len(fig.axes) == 1
    assert len(fig.axes[0].patches) >= len(mod.SCORECARD_METRICS)


def test_render_handles_nan_score(tmp_path: Path):
    pytest.importorskip("matplotlib")
    mod = _load_module()
    card = _compute(mod, _tiny_instance())
    card.section_mixing = float("nan")  # force a not-evaluable bar
    out = tmp_path / "scorecard_nan.png"
    mod.render_integration_scorecard(card, out)
    assert out.read_bytes()[:8] == _PNG_MAGIC


def test_example_main_renders(tmp_path: Path):
    pytest.importorskip("matplotlib")
    mod = _load_module()
    out = tmp_path / "example.png"
    assert mod.main(["--example", "--out", str(out)]) == 0
    assert out.read_bytes()[:8] == _PNG_MAGIC


def test_finite_scores_use_finite_silhouettes():
    mod = _load_module()
    card = _compute(mod, _tiny_instance())
    for attr in ("sil_section_before", "sil_section_after", "sil_domain_before", "sil_domain_after"):
        assert np.isfinite(getattr(card, attr)), attr
