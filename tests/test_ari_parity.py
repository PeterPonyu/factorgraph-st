"""Numerical-parity guard for the vectorized adjusted_rand_index (#100/#104).

Asserts the np.add.at contingency-table implementation equals an independent
brute-force pair-counting reference on fixtures + edge cases, so the perf
change is provably behavior-preserving.
"""
from itertools import combinations

import numpy as np
import pytest

from factorgraph_st.eval.metrics import adjusted_rand_index


def _ref_ari(a: np.ndarray, b: np.ndarray) -> float:
    """Independent O(n^2) pair-counting Adjusted Rand Index reference."""
    n = len(a)
    if n < 2:
        return float("nan")
    tp = fp = fn = tn = 0
    for i, j in combinations(range(n), 2):
        same_a = a[i] == a[j]
        same_b = b[i] == b[j]
        if same_a and same_b:
            tp += 1
        elif same_a and not same_b:
            fp += 1
        elif not same_a and same_b:
            fn += 1
        else:
            tn += 1
    index = tp
    expected = (tp + fp) * (tp + fn) / (tp + fp + fn + tn)
    maximum = 0.5 * ((tp + fp) + (tp + fn))
    denom = maximum - expected
    if denom == 0:
        return float("nan")
    return float((index - expected) / denom)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_vectorized_ari_matches_bruteforce(seed: int) -> None:
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 4, size=40)
    b = rng.integers(0, 5, size=40)
    got = adjusted_rand_index(a, b)
    ref = _ref_ari(a, b)
    assert np.isclose(got, ref, atol=1e-9), f"vectorized={got} ref={ref}"


def test_identical_partitions_is_one() -> None:
    a = np.array([0, 0, 1, 1, 2, 2])
    assert np.isclose(adjusted_rand_index(a, a.copy()), 1.0)


def test_degenerate_inputs_return_nan() -> None:
    # n < 2 and the all-one-cluster case are not-evaluable, not "perfect" (#79).
    assert np.isnan(adjusted_rand_index(np.array([0]), np.array([0])))
    assert np.isnan(adjusted_rand_index(np.zeros(5, int), np.zeros(5, int)))
