import numpy as np

from factorgraph_st.eval.metrics import adjusted_rand_index, morans_i
from factorgraph_st.synth.generator import generate_instance


def test_adjusted_rand_index_perfect_and_randomish_cases():
    labels = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    assert adjusted_rand_index(labels, labels.copy()) == 1.0
    shuffled = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
    assert adjusted_rand_index(labels, shuffled) < 0.2


def test_morans_i_range_on_generator_domains():
    inst = generate_instance(n_sections=2, n_spots_per_section=10, n_domains=3, seed=13)
    value = morans_i(inst.domain_id, inst.edges)
    assert -1.0 <= value <= 1.0
