"""Parity of ``adjusted_mutual_information`` against scikit-learn.

AMI chance-corrects mutual information over the hypergeometric null (Vinh et
al., 2010). This pins the numpy-only implementation to scikit-learn's reference
``adjusted_mutual_info_score`` (arithmetic normalizer) plus the documented
degenerate-case behaviour.
"""

import numpy as np
import pytest

from factorgraph_st.eval import adjusted_mutual_information

sk = pytest.importorskip("sklearn.metrics")


def _cases(rng):
    return [
        (np.array([0, 0, 1, 1, 2, 2]), np.array([0, 0, 1, 1, 2, 2])),  # identical
        (np.array([0, 0, 1, 1, 2, 2]), np.array([2, 2, 0, 0, 1, 1])),  # relabeled
        (np.array([0, 0, 0, 1, 1, 1]), np.array([0, 1, 0, 1, 0, 1])),  # ~independent
        (rng.integers(0, 4, 200), rng.integers(0, 3, 200)),  # differing k
        (rng.integers(0, 7, 500), rng.integers(0, 5, 500)),
        (rng.integers(0, 2, 50), rng.integers(0, 9, 50)),  # asymmetric k
    ]


def test_ami_matches_sklearn():
    rng = np.random.default_rng(0)
    for t, p in _cases(rng):
        got = adjusted_mutual_information(t.astype(np.int64), p.astype(np.int64))
        exp = sk.adjusted_mutual_info_score(t, p, average_method="arithmetic")
        assert np.isclose(got, exp, atol=1e-9), (got, exp)


def test_ami_identical_is_one():
    a = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    assert adjusted_mutual_information(a, a) == pytest.approx(1.0)


def test_ami_both_single_cluster_is_one():
    a = np.zeros(5, dtype=np.int64)
    assert adjusted_mutual_information(a, a) == 1.0


def test_ami_nan_below_two_samples():
    a = np.array([0], dtype=np.int64)
    assert np.isnan(adjusted_mutual_information(a, a))


def test_ami_length_mismatch_raises():
    with pytest.raises(ValueError):
        adjusted_mutual_information(np.array([0, 1]), np.array([0, 1, 1]))
