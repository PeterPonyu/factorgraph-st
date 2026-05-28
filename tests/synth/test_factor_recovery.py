import numpy as np

from factorgraph_st.eval.metrics import matched_factor_correlation


def test_matched_factor_correlation_permutation_invariant():
    truth = np.arange(24, dtype=np.float32).reshape(6, 4)
    estimate = truth[:, [2, 0, 3, 1]]
    assert matched_factor_correlation(estimate, truth) == 1.0
