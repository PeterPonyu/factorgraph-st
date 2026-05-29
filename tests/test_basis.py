"""Regression tests for ``_positive_basis`` low-variance column handling (#85).

A constant or near-constant encoder embedding column (``std → 0``) was
divided by a hard 1e-6 eps floor. With a tiny but nonzero perturbation
the floor activates and amplifies microscopic variation by up to 1e6×;
worse, the trailing ``+ 1e-6`` offset leaves *truly* constant columns
contributing a 1e-6 bias to every factor activation. This propagates to
loadings and clustering as a fake "signal" with no information content.

The fix detects low-variance columns (``std < tol``) and zeros them
rather than dividing by the floor + adding a bias.
"""

from __future__ import annotations

import numpy as np

from factorgraph_st.model.decoder import _positive_basis


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
