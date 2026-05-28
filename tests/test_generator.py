"""Regression test for #76 / #90: synthetic X must be nonnegative.

The synthetic generator in ``factorgraph_st.synth.generator`` produces
``X = mean + N(0, noise_sigma)`` where ``mean = Z @ W.T`` is nonnegative by
construction (W, Z >= 0 via the exponential prior). Adding a centered
Gaussian to a nonnegative mean drives many entries below zero — silently
violating the spatial-transcriptomics count-data contract that the
downstream model and validators assume.

Fix: clip the noisy mean to the nonnegative half-line (or sample from a
nonnegative-support noise distribution such as Poisson).
"""

from __future__ import annotations

import pytest

from factorgraph_st.synth.generator import generate_instance


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_X_nonnegative(seed: int) -> None:
    """X.min() must be >= 0 across multiple seeds (count-data contract)."""
    inst = generate_instance(
        n_sections=2,
        n_spots_per_section=40,
        n_genes=12,
        K_shared=3,
        K_private=1,
        noise_sigma=0.5,
        k_nn=3,
        seed=seed,
    )
    x_min = float(inst.X.min())
    assert x_min >= 0.0, (
        f"X must be nonnegative (count-like); got X.min()={x_min:.4f} at seed={seed}"
    )
