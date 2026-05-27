import numpy as np

from factorgraph_st.synth import generate_instance


def test_shapes_match_documented_schema():
    inst = generate_instance(
        n_sections=3,
        n_spots_per_section=50,
        n_genes=20,
        K_shared=3,
        K_private=2,
        n_domains=4,
        k_nn=4,
        seed=42,
    )
    n_spots = 3 * 50
    K_total = 3 + 2
    assert inst.X.shape == (n_spots, 20)
    assert inst.X.dtype == np.float32
    assert inst.coords.shape == (n_spots, 2)
    assert inst.section_id.shape == (n_spots,)
    assert inst.section_id.dtype == np.int64
    assert inst.edges.shape == (2, n_spots * 4)
    assert inst.edges.dtype == np.int64
    assert inst.W.shape == (20, K_total)
    assert inst.Z_shared.shape == (n_spots, 3)
    assert inst.Z_private.shape == (n_spots, 2)
    assert inst.domain_id.shape == (n_spots,)
    assert inst.domain_id.dtype == np.int64


def test_edges_point_to_valid_spots():
    inst = generate_instance(
        n_sections=2,
        n_spots_per_section=30,
        n_genes=10,
        K_shared=2,
        K_private=1,
        n_domains=3,
        k_nn=3,
        seed=7,
    )
    n_spots = 60
    assert inst.edges.min() >= 0
    assert inst.edges.max() < n_spots
