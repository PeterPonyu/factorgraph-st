"""Smoke + unit tests for the scree + covariate-association figure script (#325).

matplotlib renders are guarded with ``importorskip``; the pure helpers
(variance-explained, eta^2 association matrix) run in the numpy-only env.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "visualize" / "fig_scree_covariate.py"
)
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _load_module():
    spec = importlib.util.spec_from_file_location("_fig_scree_covariate", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _structured_inputs():
    """Tiny H where factor 0 tracks section and factor 1 tracks domain."""
    rng = np.random.default_rng(0)
    n = 40
    section = np.repeat(np.arange(2), n // 2)
    domain = np.tile(np.arange(4), n // 4)
    H = rng.normal(0.0, 0.05, size=(n, 3))
    H[:, 0] += section * 3.0
    H[:, 1] += domain * 2.0
    H = np.clip(H, 0.0, None).astype(np.float32)
    return H, {"section": section, "domain": domain}


def test_variance_explained_sums_to_one_and_is_finite():
    mod = _load_module()
    H, _ = _structured_inputs()
    frac = mod.variance_explained(H)
    assert frac.shape == (3,)
    assert np.all(np.isfinite(frac))
    assert frac.sum() == pytest.approx(1.0)


def test_variance_explained_all_constant_is_zeros():
    mod = _load_module()
    frac = mod.variance_explained(np.ones((10, 4), dtype=np.float32))
    assert frac.tolist() == [0.0, 0.0, 0.0, 0.0]


def test_covariate_association_matrix_shape_and_signal():
    mod = _load_module()
    H, covariates = _structured_inputs()
    matrix, names = mod.covariate_association_matrix(H, covariates)
    assert names == ["section", "domain"]
    assert matrix.shape == (3, 2)
    # Factor 0 should associate most with section; factor 1 with domain.
    assert matrix[0, 0] > matrix[0, 1]
    assert matrix[1, 1] > matrix[1, 0]
    assert np.all((matrix[np.isfinite(matrix)] >= 0.0) & (matrix[np.isfinite(matrix)] <= 1.0))


def test_covariate_association_requires_covariates():
    mod = _load_module()
    H, _ = _structured_inputs()
    with pytest.raises(ValueError, match="covariates"):
        mod.covariate_association_matrix(H, {})


def test_render_scree_covariate_writes_valid_png(tmp_path: Path):
    pytest.importorskip("matplotlib")
    mod = _load_module()
    H, covariates = _structured_inputs()
    out = tmp_path / "scree.png"
    fig = mod.render_scree_covariate(H, covariates, out)
    data = out.read_bytes()
    assert len(data) > 1000
    assert data[:8] == _PNG_MAGIC
    assert len(fig.axes) >= 2  # scree + association map (+ colorbar)


def test_example_main_renders(tmp_path: Path):
    pytest.importorskip("matplotlib")
    mod = _load_module()
    out = tmp_path / "example.png"
    assert mod.main(["--example", "--out", str(out)]) == 0
    assert out.read_bytes()[:8] == _PNG_MAGIC
