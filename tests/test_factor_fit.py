"""Regression test for #47: _fit_nonnegative_loadings on empty input.

When ``Z`` has zero rows (empty embedding), ``_fit_nonnegative_loadings``
in ``factorgraph_st.model.decoder`` previously returned
``np.empty((X.shape[1], Z.shape[1]))``. ``np.empty`` allocates
uninitialized memory, so the returned ``W`` contains arbitrary, often
non-finite, junk bytes (e.g., ``-3.6e+17``, ``NaN``, ``inf`` depending
on the allocator state).

The function is documented as returning nonnegative loadings, and the
output schema validator expects finite, nonnegative values. Returning
uninitialized memory silently produces a non-finite or negative ``W``
that downstream consumers (decoder, validators, evaluation metrics) may
quietly accept or crash on.

The minimal fix is to return ``np.zeros(shape, dtype=float32)``: this
matches the nonnegative contract, is deterministic, and is the natural
zero element for a linear-regression coefficient matrix with zero
samples.
"""

from __future__ import annotations

import numpy as np

from factorgraph_st.model.decoder import _fit_nonnegative_loadings


def test_empty_input_finite_W():
    """Empty Z must produce a finite, nonneg W (the nonnegative-loadings contract)."""
    # Z has zero rows (no samples / empty embedding).
    n_features = 5
    n_components = 3
    X = np.zeros((0, n_features), dtype=np.float32)
    Z = np.zeros((0, n_components), dtype=np.float32)

    W = _fit_nonnegative_loadings(X, Z)

    assert W.shape == (n_features, n_components), (
        f"expected shape ({n_features}, {n_components}); got {W.shape}"
    )
    assert np.isfinite(W).all(), (
        f"W must be finite on empty input; found non-finite values: {W}"
    )
    assert (W >= 0).all(), (
        f"W must be nonnegative (function name says so); found negatives: {W}"
    )


def test_empty_input_deterministic_W():
    """Repeated calls on the same empty input must return the same W (no allocator noise)."""
    X = np.zeros((0, 4), dtype=np.float32)
    Z = np.zeros((0, 2), dtype=np.float32)
    W1 = _fit_nonnegative_loadings(X, Z)
    W2 = _fit_nonnegative_loadings(X, Z)
    np.testing.assert_array_equal(W1, W2)


def test_empty_input_various_shapes():
    """Empty Z with assorted (n_features, n_components) shapes all return finite, nonneg W."""
    for n_features, n_components in [(1, 1), (7, 3), (100, 8)]:
        X = np.zeros((0, n_features), dtype=np.float32)
        Z = np.zeros((0, n_components), dtype=np.float32)
        W = _fit_nonnegative_loadings(X, Z)
        assert W.shape == (n_features, n_components)
        assert np.isfinite(W).all(), (
            f"non-finite W at shape ({n_features}, {n_components}): {W}"
        )
        assert (W >= 0).all(), (
            f"negative W at shape ({n_features}, {n_components}): {W}"
        )
