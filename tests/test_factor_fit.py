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


def test_loadings_are_NNLS():
    """Regression for #83: _fit_nonnegative_loadings must solve NNLS, not clipped LS.

    Build a problem with sparse nonneg true loadings so the unconstrained
    least-squares solution has negative entries. Clipping those to 0 is
    biased; a proper NNLS solve recovers the true nonneg optimum and has
    strictly lower (or equal) reconstruction error than the clipped LS
    estimate.
    """
    from scipy.optimize import nnls

    from factorgraph_st.model.decoder import _fit_nonnegative_loadings

    rng = np.random.default_rng(0)
    n_spots, n_components, n_features = 50, 3, 4
    # Correlated nonneg factors so the bias from clipping is non-trivial.
    Z = rng.exponential(scale=1.0, size=(n_spots, n_components)).astype(np.float64)
    # Sparse nonneg true loadings (each gene loads on a subset of factors).
    W_true = np.array(
        [
            [1.0, 0.0, 2.0],
            [0.0, 1.5, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    X = (Z @ W_true.T + rng.normal(0.0, 0.05, size=(n_spots, n_features))).astype(np.float32)

    W = _fit_nonnegative_loadings(X, Z.astype(np.float32))

    # Reference NNLS per gene column.
    W_ref = np.column_stack(
        [nnls(Z, X.astype(np.float64)[:, j])[0] for j in range(n_features)]
    ).T

    np.testing.assert_allclose(
        W.astype(np.float64), W_ref, atol=1e-4,
        err_msg="W must match scipy.optimize.nnls (not clipped lstsq)",
    )

    # Sanity: NNLS reconstruction error ≤ clipped-LS reconstruction error.
    coef_ls, *_ = np.linalg.lstsq(Z, X.astype(np.float64), rcond=None)
    W_clip = np.clip(coef_ls.T, 0.0, None)
    err_nnls = float(np.linalg.norm(X.astype(np.float64) - Z @ W.astype(np.float64).T))
    err_clip = float(np.linalg.norm(X.astype(np.float64) - Z @ W_clip.T))
    assert err_nnls <= err_clip + 1e-6, (
        f"NNLS error must be ≤ clipped-LS error; got NNLS={err_nnls:.6f}, clip={err_clip:.6f}"
    )

    # Confirm unconstrained LS actually produced negatives on this problem
    # (so the regression is non-trivial — clipping really did change things).
    assert (coef_ls.T < 0).any(), "test fixture should produce negative LS coefs"


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
