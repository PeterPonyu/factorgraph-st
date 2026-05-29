"""Regression test for #88: reconstruction-error / decoder-fidelity metric.

The decoder defines ``X ~= Z @ W.T`` but the eval surface never measured
reconstruction quality — the most basic fidelity check for a factor model and
the only one that generalizes to real data (no ground-truth factors).
``reconstruction_error`` returns the relative Frobenius error, and
``held_out_reconstruction_error`` refits loadings on a spot subset and scores
the rest (overfitting / K-selection signal).
"""

from __future__ import annotations

import numpy as np

from factorgraph_st.eval.metrics import (
    held_out_reconstruction_error,
    reconstruction_error,
)


def test_perfect_reconstruction_is_zero():
    """Exact low-rank data reconstructs with ~0 relative error."""
    rng = np.random.default_rng(0)
    Z = rng.exponential(size=(100, 4))
    W = rng.exponential(size=(20, 4))
    X = Z @ W.T
    assert reconstruction_error(X, Z, W) < 1e-6


def test_error_decreases_as_K_approaches_truth():
    """Error falls monotonically with K and hits ~0 at the true rank."""
    rng = np.random.default_rng(0)
    K_true = 4
    X = rng.exponential(size=(150, K_true)) @ rng.exponential(size=(30, K_true)).T
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    errs = [reconstruction_error(X, U[:, :K] * S[:K], Vt[:K].T) for K in range(1, K_true + 2)]
    assert all(errs[i + 1] <= errs[i] + 1e-9 for i in range(len(errs) - 1)), errs
    assert errs[K_true - 1] < 1e-6


def test_all_zero_X_returns_finite():
    """Zero target gives a finite score (no divide-by-zero)."""
    X = np.zeros((10, 5))
    Z = np.zeros((10, 3))
    W = np.zeros((5, 3))
    assert reconstruction_error(X, Z, W) == 0.0


def test_held_out_error_low_for_clean_low_rank_data():
    """Held-out spots reconstruct well when data is exactly low-rank."""
    rng = np.random.default_rng(0)
    Z = rng.exponential(size=(200, 4))
    W = rng.exponential(size=(25, 4))
    X = Z @ W.T
    err = held_out_reconstruction_error(X, Z, holdout=0.3, seed=0)
    assert 0.0 <= err < 1e-6


def test_held_out_degenerate_split_returns_zero():
    """Fewer than two spots cannot be split; returns 0.0."""
    assert held_out_reconstruction_error(np.ones((1, 3)), np.ones((1, 2))) == 0.0
