"""Regression tests for #69: ``_positive_basis`` must not silently pad H
with zero columns when ``d < K_shared + K_private``.

Before the fix, the pad-and-normalize path of ``_positive_basis`` left every
padded column at a constant ``1e-6`` after the ``raw/scale + 1e-6`` step
(std = 0 for zero columns, clamped to 1e-6, so each column becomes that
constant). Downstream factors silently carry no signal, yet validate_outputs
still passes because nonneg/finite are satisfied. The fix: raise
``ValueError`` so callers cannot supply ``d < K_total`` by accident.
"""

from __future__ import annotations

import numpy as np
import pytest

from factorgraph_st.model.decoder import _positive_basis, decode_factors


def test_dim_mismatch_raises():
    """``_positive_basis`` must raise when H.shape[1] < n_components."""
    H = np.random.default_rng(0).normal(size=(20, 2)).astype(np.float32)
    with pytest.raises(ValueError, match="d.*<.*K_shared.*K_private|H.shape"):
        _positive_basis(H, n_components=5)


def test_decode_factors_rejects_short_embedding():
    """``decode_factors`` propagates the guard so callers learn at the boundary."""
    rng = np.random.default_rng(0)
    H = rng.normal(size=(20, 2)).astype(np.float32)
    X = rng.exponential(size=(20, 6)).astype(np.float32)
    with pytest.raises(ValueError):
        decode_factors(X, H, K_shared=3, K_private=2)


def test_positive_basis_exact_dim_ok():
    """No regression: ``d == K_total`` and ``d > K_total`` still succeed."""
    H = np.random.default_rng(0).normal(size=(20, 5)).astype(np.float32)
    out_exact = _positive_basis(H, n_components=5)
    out_wider = _positive_basis(H, n_components=4)
    assert out_exact.shape == (20, 5) and out_wider.shape == (20, 4)
    assert np.isfinite(out_exact).all() and np.isfinite(out_wider).all()


def test_positive_basis_empty_input_still_returns_empty():
    """Empty inputs continue to short-circuit (no guard regression on n=0)."""
    H = np.zeros((0, 2), dtype=np.float32)
    out = _positive_basis(H, n_components=5)
    assert out.shape == (0, 5) and out.dtype == np.float32
