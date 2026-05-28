from factorgraph_st.eval.metrics import shared_private_separation
from factorgraph_st.synth.generator import generate_instance


def test_shared_private_section_overlap_separates_generator_truth():
    inst = generate_instance(n_sections=3, n_spots_per_section=8, K_private=3, seed=12)
    result = shared_private_separation(inst.Z_shared, inst.Z_private, inst.section_id)
    assert result["shared_mean_active_sections"] == 3.0
    assert result["private_mean_active_sections"] == 1.0
