"""Unit tests for the agreement-vs-reference-complexity figure helpers (#327).

Exercises ONLY the pure numpy/stdlib core (``pearson_r``, ``regression_line``,
``assemble_points``) -- no anndata, no silhouette, no matplotlib render. The
``reference_ambiguity`` collector is data-dependent and verified by the
end-to-end render, not here.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "visualize"
    / "fig_agreement_vs_complexity.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("_fig_agreement_vs_complexity", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mod = _load_module()


def test_pearson_r_perfectly_correlated():
    r = mod.pearson_r([1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0])
    assert math.isclose(r, 1.0, abs_tol=1e-12)


def test_pearson_r_perfectly_anticorrelated():
    r = mod.pearson_r([1.0, 2.0, 3.0, 4.0], [8.0, 6.0, 4.0, 2.0])
    assert math.isclose(r, -1.0, abs_tol=1e-12)


def test_pearson_r_too_few_points_is_nan():
    assert math.isnan(mod.pearson_r([1.0], [2.0]))
    assert math.isnan(mod.pearson_r([], []))


def test_pearson_r_zero_variance_is_nan():
    assert math.isnan(mod.pearson_r([3.0, 3.0, 3.0], [1.0, 2.0, 3.0]))


def test_regression_line_recovers_slope_and_intercept():
    # y = 2x + 1
    slope, intercept = mod.regression_line([0.0, 1.0, 2.0, 3.0], [1.0, 3.0, 5.0, 7.0])
    assert math.isclose(slope, 2.0, abs_tol=1e-12)
    assert math.isclose(intercept, 1.0, abs_tol=1e-12)


def test_regression_line_too_few_points_is_nan():
    slope, intercept = mod.regression_line([1.0], [2.0])
    assert math.isnan(slope) and math.isnan(intercept)


def test_assemble_points_count_and_join():
    accuracy = {
        "fileA": {
            "dsA": {
                "coords": {"ari": 0.30, "nmi": 0.4, "ami": 0.35},
                "gnmf": {"ari": 0.10, "nmi": 0.2, "ami": 0.15},
            }
        },
        "fileB": {
            "dsB": {
                "spatial_smooth": {"ari": 0.55, "nmi": 0.6, "ami": 0.5},
            }
        },
    }
    ambiguity = {"dsA": -0.20, "dsB": -0.45}
    points = mod.assemble_points(accuracy, ambiguity, expert_datasets={"dsB"})

    assert len(points) == 3  # 2 variants for dsA + 1 for dsB
    by_key = {(p["dataset"], p["variant"]): p for p in points}

    # ambiguity joined from the per-dataset map (x), ari from the scorecard (y)
    assert math.isclose(by_key[("dsA", "coords")]["ambiguity"], -0.20)
    assert math.isclose(by_key[("dsA", "coords")]["ari"], 0.30)
    assert math.isclose(by_key[("dsA", "gnmf")]["ari"], 0.10)
    assert math.isclose(by_key[("dsB", "spatial_smooth")]["ambiguity"], -0.45)
    assert math.isclose(by_key[("dsB", "spatial_smooth")]["ari"], 0.55)


def test_assemble_points_expert_flag_membership():
    accuracy = {"f": {"dsA": {"m": {"ari": 0.5}}, "dsB": {"m": {"ari": 0.6}}}}
    ambiguity = {"dsA": -0.1, "dsB": -0.2}
    points = mod.assemble_points(accuracy, ambiguity, expert_datasets={"dsA"})
    flag = {p["dataset"]: p["expert"] for p in points}
    assert flag == {"dsA": True, "dsB": False}


def test_assemble_points_skips_datasets_without_ambiguity():
    accuracy = {"f": {"dsA": {"m": {"ari": 0.5}}, "dsMissing": {"m": {"ari": 0.6}}}}
    ambiguity = {"dsA": -0.1}  # dsMissing has no ambiguity -> dropped, never zero-filled
    points = mod.assemble_points(accuracy, ambiguity)
    assert [p["dataset"] for p in points] == ["dsA"]
