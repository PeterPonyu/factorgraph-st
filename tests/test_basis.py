"""Regression tests for ``_positive_basis``.

Two independent failure modes are covered here:

#69 — ``_positive_basis`` must not silently pad ``H`` with zero columns when
``d < K_shared + K_private``. Before the fix, the pad-and-normalize path left
every padded column at a constant ``1e-6`` after the ``raw/scale + 1e-6`` step
(std = 0 for zero columns, clamped to 1e-6, so each column becomes that
constant). Downstream factors silently carry no signal, yet validate_outputs
still passes because nonneg/finite are satisfied. The fix: raise ``ValueError``
so callers cannot supply ``d < K_total`` by accident.

#85 — low-variance column handling. A constant or near-constant encoder
embedding column (``std → 0``) was divided by a hard 1e-6 eps floor. With a tiny
but nonzero perturbation the floor activates and amplifies microscopic variation
by up to 1e6×; worse, the trailing ``+ 1e-6`` offset leaves *truly* constant
columns contributing a 1e-6 bias to every factor activation. This propagates to
loadings and clustering as a fake "signal" with no information content. The fix
detects low-variance columns (``std < tol``) and zeros them rather than dividing
by the floor + adding a bias.
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


def test_low_variance_column_no_explosion():
    """Near-constant column with var ~ 1e-15 produces finite, bounded basis values."""
    n_spots = 100
    n_components = 3
    rng = np.random.default_rng(0)
    H = np.zeros((n_spots, n_components), dtype=np.float32)
    # Column 0: well-conditioned signal.
    H[:, 0] = rng.normal(size=n_spots).astype(np.float32)
    # Column 1: near-constant — variance ~ 1e-15 (one float-precision-level
    # perturbation on top of a constant baseline).
    H[:, 1] = 5.0
    H[50, 1] = np.float32(5.0 + 1e-7)
    # Column 2: exactly constant.
    H[:, 2] = 3.0

    basis = _positive_basis(H, n_components)

    # No inf/NaN anywhere — the headline acceptance criterion.
    assert np.isfinite(basis).all(), (
        f"basis must be finite; found non-finite values:\n{basis}"
    )

    # Schema: factor activations are nonneg.
    assert (basis >= 0).all(), "basis must be nonnegative"

    # An exactly constant column carries zero information — its contribution
    # must not leak through as a fake constant bias term. Bound the maximum
    # magnitude of the constant column.
    assert basis[:, 2].max() < 1e-7, (
        f"constant column should not contribute a >1e-7 bias; "
        f"got max={basis[:, 2].max():.3e}"
    )

    # Near-constant column likewise must not produce activations of order
    # 1.0 from a 1e-7 raw perturbation (would be a ~1e7× amplification).
    assert basis[:, 1].max() < 1e-3, (
        f"near-constant column should not amplify to O(1); "
        f"got max={basis[:, 1].max():.3e}"
    )


def test_well_conditioned_columns_unchanged():
    """Existing well-conditioned columns must still normalize to the same scale."""
    rng = np.random.default_rng(1)
    n_spots = 200
    n_components = 4
    H = rng.normal(size=(n_spots, n_components)).astype(np.float32)
    basis = _positive_basis(H, n_components)
    assert np.isfinite(basis).all()
    assert (basis >= 0).all()
    # Each column should have meaningful spread (std of a z-scored, min-shifted
    # column with a normal distribution is ~1).
    assert (basis.std(axis=0) > 0.5).all(), (
        f"well-conditioned columns should retain unit-scale variation; "
        f"got stds={basis.std(axis=0)}"
    )
