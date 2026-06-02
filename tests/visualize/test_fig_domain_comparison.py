"""Smoke + unit tests for the domain-quality method-comparison figure script.

The matplotlib render is guarded with ``importorskip("matplotlib")`` because
the base CI test env installs only ``.[test]`` (numpy + scipy), not a plotting
stack. The pure metric-extraction helpers are exercised without matplotlib so
they stay covered everywhere.
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
    / "fig_domain_comparison.py"
)

# PNG 8-byte magic signature.
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _load_module():
    spec = importlib.util.spec_from_file_location("_fig_domain_comparison", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _synthetic_metrics() -> dict[str, dict[str, float]]:
    """Small per-method metric dicts with the expected ordering."""
    return {
        "projection": {
            "ari_domain": 0.20,
            "nmi_domain": 0.30,
            "boundary_f1_domain": 0.25,
            "weighted_dice_domain": 0.40,
            "silhouette_domain": 0.05,
        },
        "spatial_smooth": {
            "ari_domain": 0.40,
            "nmi_domain": 0.50,
            "boundary_f1_domain": 0.45,
            "weighted_dice_domain": 0.55,
            "silhouette_domain": 0.15,
        },
        "gnmf": {
            "ari_domain": 0.55,
            "nmi_domain": 0.62,
            "boundary_f1_domain": 0.60,
            "weighted_dice_domain": 0.70,
            "silhouette_domain": 0.25,
        },
    }


def test_extract_domain_quality_from_full_bundle():
    """A full results-contract bundle exposes the five comparison metrics."""
    mod = _load_module()
    bundle = {
        "schema_version": "1.0.0",
        "project": "factorgraph-st",
        "metrics": {
            "ari_domain": 0.5,
            "nmi_domain": 0.6,
            "boundary_f1_domain": 0.55,
            "weighted_dice_domain": 0.7,
            "silhouette_domain": 0.2,
            "calinski_harabasz_domain": 1234.5,  # present but intentionally unused
        },
    }
    got = mod.extract_domain_quality(bundle)
    assert set(got) == {key for key, _ in mod.DOMAIN_METRICS}
    assert got["ari_domain"] == pytest.approx(0.5)
    assert got["silhouette_domain"] == pytest.approx(0.2)
    assert "calinski_harabasz_domain" not in got


def test_extract_domain_quality_missing_and_null_become_none():
    """Absent / null / non-numeric metrics coerce to None (not a fake zero)."""
    mod = _load_module()
    got = mod.extract_domain_quality({"metrics": {"ari_domain": None, "nmi_domain": 0.4}})
    assert got["ari_domain"] is None  # explicit null
    assert got["nmi_domain"] == pytest.approx(0.4)
    assert got["boundary_f1_domain"] is None  # absent key


def test_load_metrics_bundle_roundtrip(tmp_path: Path):
    mod = _load_module()
    payload = {"metrics": {"ari_domain": 0.3}}
    p = tmp_path / "metrics.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    assert mod.load_metrics_bundle(p) == payload


def test_render_comparison_writes_valid_png(tmp_path: Path):
    """Smoke: rendering synthetic metrics produces a valid non-empty PNG."""
    pytest.importorskip("matplotlib")
    mod = _load_module()
    out = tmp_path / "panel.png"
    result = mod.render_comparison(_synthetic_metrics(), out)
    assert result == out
    assert out.exists()
    data = out.read_bytes()
    assert len(data) > 1000  # a real raster, not an empty stub
    assert data[:8] == _PNG_MAGIC


def test_render_comparison_handles_missing_metric(tmp_path: Path):
    """A None metric renders without error (drawn as an 'n/a' stub)."""
    pytest.importorskip("matplotlib")
    mod = _load_module()
    metrics = _synthetic_metrics()
    metrics["projection"]["silhouette_domain"] = None  # type: ignore[assignment]
    out = tmp_path / "panel_missing.png"
    mod.render_comparison(metrics, out)
    assert out.read_bytes()[:8] == _PNG_MAGIC


def test_render_comparison_rejects_empty_methods(tmp_path: Path):
    pytest.importorskip("matplotlib")
    mod = _load_module()
    with pytest.raises(ValueError, match="no known methods"):
        mod.render_comparison({"unknown_method": {"ari_domain": 0.1}}, tmp_path / "x.png")
