"""Regression test for #92: decoder normalizes factors to a canonical gauge.

The factorization ``X ~= [Z_shared | Z_private] @ W.T`` has a scale
indeterminacy: ``(Z @ D) @ (W @ inv(D)).T`` reconstructs identically for any
positive diagonal ``D``. The decoder now pins the gauge so every loading
(``W``) column has unit L2 norm, with the scale folded into the matching
activation column. Before the fix, loading columns had arbitrary magnitudes,
so the unit-norm invariant failed and factor strengths were not comparable.
"""

from __future__ import annotations

import numpy as np

from factorgraph_st.model.decoder import _apply_canonical_gauge, fit_transform
from factorgraph_st.synth.generator import generate_instance


def _fit_small():
    inst = generate_instance(
        n_sections=2,
        n_spots_per_section=40,
        n_genes=30,
        K_shared=3,
        K_private=2,
        n_domains=3,
        k_nn=5,
        seed=0,
    )
    return fit_transform(
        inst.X,
        inst.coords,
        inst.section_id,
        inst.edges,
        d=8,
        K_shared=3,
        K_private=2,
        n_domains=3,
        seed=0,
    )


def test_W_columns_unit_norm():
    """Every non-degenerate loading column lands on the unit sphere."""
    out = _fit_small()
    norms = np.linalg.norm(out.W, axis=0)
    assert np.allclose(norms[norms > 1e-6], 1.0, atol=1e-5), norms


def test_gauge_preserves_reconstruction():
    """The gauge transform is reconstruction-preserving and nonneg-preserving."""
    rng = np.random.default_rng(0)
    Z_shared = rng.exponential(size=(20, 3)).astype(np.float32)
    Z_private = rng.exponential(size=(20, 2)).astype(np.float32)
    W = rng.exponential(size=(15, 5)).astype(np.float32)

    before = np.concatenate([Z_shared, Z_private], axis=1) @ W.T
    Zs, Zp, Wn = _apply_canonical_gauge(Z_shared, Z_private, W)
    after = np.concatenate([Zs, Zp], axis=1) @ Wn.T

    np.testing.assert_allclose(before, after, rtol=1e-4, atol=1e-4)
    assert np.allclose(np.linalg.norm(Wn, axis=0), 1.0, atol=1e-5)
    assert (Zs >= 0).all() and (Zp >= 0).all() and (Wn >= 0).all()


def test_two_fits_reach_same_canonical_gauge():
    """Two fits of the same instance converge to the same canonical gauge."""
    out1, out2 = _fit_small(), _fit_small()
    np.testing.assert_allclose(
        np.linalg.norm(out1.W, axis=0), np.linalg.norm(out2.W, axis=0), atol=1e-5
    )
    np.testing.assert_allclose(out1.W, out2.W, atol=1e-5)
