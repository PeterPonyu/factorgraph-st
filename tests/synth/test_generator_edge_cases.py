import numpy as np

from factorgraph_st.synth.generator import _build_knn_edges_per_section, generate_instance


def test_noise_free_expression_matches_factor_product():
    inst = generate_instance(n_sections=2, n_spots_per_section=8, n_genes=5, noise_sigma=0.0, seed=1)
    Z = np.concatenate([inst.Z_shared, inst.Z_private], axis=1)
    np.testing.assert_allclose(inst.X, Z @ inst.W.T, rtol=1e-6, atol=1e-6)


def test_noise_sigma_increases_residual_scale():
    base = dict(n_sections=2, n_spots_per_section=20, n_genes=8, seed=2)
    quiet = generate_instance(**base, noise_sigma=0.01)
    noisy = generate_instance(**base, noise_sigma=1.0)
    quiet_mean = np.concatenate([quiet.Z_shared, quiet.Z_private], axis=1) @ quiet.W.T
    noisy_mean = np.concatenate([noisy.Z_shared, noisy.Z_private], axis=1) @ noisy.W.T
    assert np.std(noisy.X - noisy_mean) > np.std(quiet.X - quiet_mean) * 20


def test_k_private_zero_and_wraparound_locality():
    zero = generate_instance(n_sections=3, n_spots_per_section=5, K_private=0, seed=3)
    assert zero.Z_private.shape == (15, 0)
    assert zero.W.shape[1] == zero.Z_shared.shape[1]

    wrapped = generate_instance(n_sections=2, n_spots_per_section=5, K_private=5, seed=4)
    for k in range(5):
        active_sections = np.unique(wrapped.section_id[wrapped.Z_private[:, k] > 0])
        assert active_sections.tolist() == [k % 2]


def test_requested_domain_count_independent_of_K_shared():
    inst = generate_instance(n_sections=2, n_spots_per_section=25, K_shared=2, n_domains=5, seed=5)
    assert len(np.unique(inst.domain_id)) == 5


def test_knn_clamps_to_available_neighbors_and_handles_singletons():
    coords = np.array([[0, 0], [1, 0], [0, 1], [10, 10]], dtype=np.float32)
    section_id = np.array([0, 0, 0, 1], dtype=np.int64)
    edges = _build_knn_edges_per_section(coords, section_id, k=3)
    assert edges.shape == (2, 6)
    assert not np.any(edges == 3)


def test_knn_boundary_section_size_minus_one():
    coords = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
    section_id = np.zeros(3, dtype=np.int64)
    edges = _build_knn_edges_per_section(coords, section_id, k=2)
    assert edges.shape == (2, 6)
