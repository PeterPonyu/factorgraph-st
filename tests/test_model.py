from factorgraph_st.model import encode_graph, fit_transform
from factorgraph_st.schemas import validate_outputs
from factorgraph_st.synth.generator import generate_instance


def test_encoder_uses_graph_and_returns_valid_H_shape():
    inst = generate_instance(n_sections=2, n_spots_per_section=10, n_genes=6, seed=10)
    H_with_edges = encode_graph(inst.X, inst.coords, inst.section_id, inst.edges, d=5, seed=0)
    H_without_edges = encode_graph(inst.X, inst.coords, inst.section_id, inst.edges[:, :0], d=5, seed=0)
    assert H_with_edges.shape == (20, 5)
    assert not (H_with_edges == H_without_edges).all()


def test_fit_transform_outputs_pass_schema():
    inst = generate_instance(n_sections=2, n_spots_per_section=10, n_genes=6, K_shared=2, K_private=1, seed=11)
    out = fit_transform(inst.X, inst.coords, inst.section_id, inst.edges, d=6, K_shared=2, K_private=1, seed=11)
    validate_outputs(out.H, out.W, out.Z_shared, out.Z_private, out.domain_id, inst.X.shape[0], inst.X.shape[1])
