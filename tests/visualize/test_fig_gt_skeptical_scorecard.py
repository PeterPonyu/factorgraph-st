"""Smoke + unit tests for the GT-skeptical domain scorecard figure script (#330).

matplotlib renders are guarded with ``importorskip``; the pure point-flattening
helper runs in the numpy-only env.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "visualize"
    / "fig_gt_skeptical_scorecard.py"
)
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _load_module():
    spec = importlib.util.spec_from_file_location("_fig_gt_skeptical_scorecard", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _scores() -> dict[str, dict[str, float]]:
    return {
        "projection": {"coherence": 0.6, "stability": 0.5, "ari": 0.2},
        "gnmf": {"coherence": 0.7, "stability": 0.8, "ari": 0.55},
        "oversmoothed": {"coherence": 0.9, "stability": 0.95, "ari": 0.05},
    }


def test_scorecard_points_flattens_and_combines_axes():
    mod = _load_module()
    points = mod.scorecard_points(_scores())
    assert len(points) == 3
    names = [p[0] for p in points]
    assert names == ["projection", "gnmf", "oversmoothed"]
    # x = mean(coherence, stability); y = ari.
    name, x, y, coherence, stability = points[1]
    assert name == "gnmf"
    assert x == pytest.approx(0.5 * (0.7 + 0.8))
    assert y == pytest.approx(0.55)
    assert (coherence, stability) == pytest.approx((0.7, 0.8))


def test_scorecard_points_rejects_missing_metric():
    mod = _load_module()
    with pytest.raises(ValueError, match="ari"):
        mod.scorecard_points({"m": {"coherence": 0.5, "stability": 0.5}})


def test_scorecard_points_rejects_empty():
    mod = _load_module()
    with pytest.raises(ValueError, match="non-empty"):
        mod.scorecard_points({})


def test_render_scorecard_writes_valid_png(tmp_path: Path):
    pytest.importorskip("matplotlib")
    mod = _load_module()
    out = tmp_path / "scorecard.png"
    fig = mod.render_scorecard(_scores(), out)
    data = out.read_bytes()
    assert len(data) > 1000
    assert data[:8] == _PNG_MAGIC
    assert len(fig.axes) == 1
    # One scatter collection point per method.
    n_points = sum(coll.get_offsets().shape[0] for coll in fig.axes[0].collections)
    assert n_points == len(_scores())


def test_main_reads_json_scores(tmp_path: Path):
    pytest.importorskip("matplotlib")
    mod = _load_module()
    scores_path = tmp_path / "scores.json"
    scores_path.write_text(json.dumps(_scores()), encoding="utf-8")
    out = tmp_path / "fromjson.png"
    assert mod.main(["--scores", str(scores_path), "--out", str(out)]) == 0
    assert out.read_bytes()[:8] == _PNG_MAGIC


def test_example_main_renders(tmp_path: Path):
    pytest.importorskip("matplotlib")
    mod = _load_module()
    out = tmp_path / "example.png"
    assert mod.main(["--example", "--out", str(out)]) == 0
    assert out.read_bytes()[:8] == _PNG_MAGIC
