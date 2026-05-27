from factorgraph_st.synth import generate_instance


def test_truth_W_nonnegative():
    inst = generate_instance(
        n_sections=2,
        n_spots_per_section=20,
        n_genes=10,
        K_shared=2,
        K_private=1,
        k_nn=3,
        seed=0,
    )
    assert (inst.W >= 0).all()


def test_truth_Z_shared_nonnegative():
    inst = generate_instance(
        n_sections=2,
        n_spots_per_section=20,
        n_genes=10,
        K_shared=2,
        K_private=1,
        k_nn=3,
        seed=0,
    )
    assert (inst.Z_shared >= 0).all()


def test_truth_Z_private_nonnegative():
    inst = generate_instance(
        n_sections=2,
        n_spots_per_section=20,
        n_genes=10,
        K_shared=2,
        K_private=1,
        k_nn=3,
        seed=0,
    )
    assert (inst.Z_private >= 0).all()


def test_private_factor_lives_in_strict_subset_of_sections():
    n_sections = 4
    K_private = 2
    inst = generate_instance(
        n_sections=n_sections,
        n_spots_per_section=40,
        n_genes=8,
        K_shared=2,
        K_private=K_private,
        k_nn=3,
        seed=0,
    )
    for k in range(K_private):
        nonzero_rows = inst.Z_private[:, k] > 0
        sections_seen = set(inst.section_id[nonzero_rows].tolist()) if nonzero_rows.any() else set()
        assert len(sections_seen) < n_sections, (
            f"private factor {k} active in every section: {sections_seen}"
        )
