"""Regression test for #91: factor-redundancy / disentanglement metric.

No existing metric measures *within-estimate* collinearity among the
recovered factors (columns of ``Z = [Z_shared | Z_private]``). A redundant
factor set (the same program reported K times) is uninterpretable yet
invisible to recovery-vs-truth metrics. ``factor_redundancy`` scores this as
the mean absolute off-diagonal correlation: high for near-duplicate factors,
low for mutually uncorrelated ones.
"""

from __future__ import annotations

import numpy as np

from factorgraph_st.eval.metrics import factor_redundancy


def test_duplicated_factors_score_high():
    """Three identical factors are maximally redundant (~1.0)."""
    rng = np.random.default_rng(0)
    f = rng.normal(size=(300, 1))
    Z = np.hstack([f, f, f])
    assert factor_redundancy(Z) > 0.99


def test_orthogonal_factors_score_low():
    """Independent random factors are nearly non-redundant."""
    rng = np.random.default_rng(1)
    Z = rng.normal(size=(300, 4))
    assert factor_redundancy(Z) < 0.2


def test_duplicates_more_redundant_than_independent():
    """The metric orders a duplicated set above an independent set."""
    rng = np.random.default_rng(2)
    base = rng.normal(size=(300, 1))
    duplicated = np.hstack([base, base + 1e-3 * rng.normal(size=(300, 1))])
    independent = rng.normal(size=(300, 2))
    assert factor_redundancy(duplicated) > factor_redundancy(independent)


def test_constant_factor_and_singleton_edge_cases():
    """Constant factors contribute no correlation; <2 factors returns 0.0."""
    rng = np.random.default_rng(3)
    Z = np.hstack([rng.normal(size=(50, 1)), np.zeros((50, 1))])
    assert factor_redundancy(Z) == 0.0
    assert factor_redundancy(rng.normal(size=(50, 1))) == 0.0
