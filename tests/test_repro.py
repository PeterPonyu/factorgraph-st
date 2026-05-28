"""Regression tests for #87: ``set_seed`` controls every random source the
package can reach (Python ``random``, NumPy globals, torch if present), and
the synthetic generator + ``fit_transform`` wire it in so end-to-end runs are
reproducible under a single seed.
"""

from __future__ import annotations

import random as _py_random

import numpy as np

import factorgraph_st
from factorgraph_st import set_seed
from factorgraph_st.model import fit_transform
from factorgraph_st.synth import generate_instance


def test_set_seed_is_public():
    """``set_seed`` is re-exported from the package root."""
    assert callable(factorgraph_st.set_seed)
    assert factorgraph_st.set_seed is set_seed


def test_set_seed_pins_global_rngs():
    """Same seed → identical draws from Python ``random`` and NumPy globals."""
    set_seed(123)
    py_a = [_py_random.random() for _ in range(4)]
    np_a = np.random.rand(4)

    set_seed(123)
    py_b = [_py_random.random() for _ in range(4)]
    np_b = np.random.rand(4)

    assert py_a == py_b
    np.testing.assert_array_equal(np_a, np_b)


def test_two_runs_same_seed_match():
    """Wiring check: synthetic generator + fit_transform match across runs.

    Two end-to-end runs with the same seed must produce identical synthetic
    data **and** identical decoded factors.
    """
    kw = dict(n_sections=2, n_spots_per_section=12, n_genes=8, K_shared=2, K_private=1, seed=42)
    a = generate_instance(**kw)
    b = generate_instance(**kw)
    np.testing.assert_array_equal(a.X, b.X)
    np.testing.assert_array_equal(a.W, b.W)
    np.testing.assert_array_equal(a.Z_shared, b.Z_shared)
    np.testing.assert_array_equal(a.Z_private, b.Z_private)
    np.testing.assert_array_equal(a.domain_id, b.domain_id)

    out_a = fit_transform(a.X, a.coords, a.section_id, a.edges, d=8, K_shared=2, K_private=1, seed=7)
    out_b = fit_transform(b.X, b.coords, b.section_id, b.edges, d=8, K_shared=2, K_private=1, seed=7)
    np.testing.assert_array_equal(out_a.H, out_b.H)
    np.testing.assert_array_equal(out_a.W, out_b.W)
    np.testing.assert_array_equal(out_a.Z_shared, out_b.Z_shared)
    np.testing.assert_array_equal(out_a.Z_private, out_b.Z_private)
    np.testing.assert_array_equal(out_a.domain_id, out_b.domain_id)


def test_different_seed_diverges():
    """Different seeds must produce different draws (set_seed is not a no-op)."""
    set_seed(1)
    a = np.random.rand(4)
    set_seed(2)
    b = np.random.rand(4)
    assert not np.array_equal(a, b)
