import numpy as np

from factorgraph_st.synth import generate_instance

KW = dict(
    n_sections=2,
    n_spots_per_section=20,
    n_genes=10,
    K_shared=2,
    K_private=1,
    n_domains=3,
    k_nn=3,
)


def test_same_seed_same_output():
    a = generate_instance(**KW, seed=0)
    b = generate_instance(**KW, seed=0)
    np.testing.assert_array_equal(a.X, b.X)
    np.testing.assert_array_equal(a.W, b.W)
    np.testing.assert_array_equal(a.Z_shared, b.Z_shared)
    np.testing.assert_array_equal(a.Z_private, b.Z_private)
    np.testing.assert_array_equal(a.coords, b.coords)
    np.testing.assert_array_equal(a.section_id, b.section_id)
    np.testing.assert_array_equal(a.edges, b.edges)
    np.testing.assert_array_equal(a.domain_id, b.domain_id)


def test_different_seed_diverges():
    a = generate_instance(**KW, seed=0)
    b = generate_instance(**KW, seed=1)
    assert not np.array_equal(a.W, b.W)
