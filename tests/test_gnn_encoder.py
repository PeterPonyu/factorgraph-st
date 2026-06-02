"""Tests for the opt-in trainable GNN encoder (issue #181, PR 1/N).

The default fit path remains the fixed random projection (`encoder="random"`),
so these tests assert (a) back-compat of the default, (b) the GNN encoder
trains (loss decreases), (c) it returns the documented shape and integrates
with `fit_transform`, (d) determinism for a fixed seed, and (e) a clear error
when torch is unavailable.

Torch-dependent tests use ``pytest.importorskip("torch")`` so they skip
gracefully in numpy-only environments.
"""

from __future__ import annotations

import numpy as np
import pytest

from factorgraph_st.model import encode_graph, fit_transform
from factorgraph_st.synth.generator import generate_instance


def _instance():
    return generate_instance(
        n_sections=2, n_spots_per_section=20, n_genes=8, K_shared=2, K_private=1, seed=3
    )


def test_default_fit_path_is_random_projection_unchanged():
    """Default fit_transform must be byte-identical to the legacy random path."""
    inst = _instance()
    out_default = fit_transform(
        inst.X, inst.coords, inst.section_id, inst.edges, d=6, K_shared=2, K_private=1, seed=3
    )
    out_explicit_random = fit_transform(
        inst.X, inst.coords, inst.section_id, inst.edges, d=6, K_shared=2, K_private=1,
        seed=3, encoder="random",
    )
    # Legacy random encoder output is reproduced exactly by both calls.
    H_legacy = encode_graph(inst.X, inst.coords, inst.section_id, inst.edges, d=6, seed=3)
    np.testing.assert_array_equal(out_default.H, H_legacy)
    np.testing.assert_array_equal(out_default.H, out_explicit_random.H)
    np.testing.assert_array_equal(out_default.W, out_explicit_random.W)


def test_unknown_encoder_raises():
    inst = _instance()
    with pytest.raises(ValueError, match="encoder"):
        fit_transform(
            inst.X, inst.coords, inst.section_id, inst.edges, d=6,
            K_shared=2, K_private=1, seed=3, encoder="bogus",
        )


def test_gnn_encoder_missing_torch_error(monkeypatch):
    """Selecting the GNN encoder without torch raises a clear ImportError.

    Simulates a torch-absent environment: build inputs first (so any incidental
    torch use during setup is unaffected), then drop torch from ``sys.modules``
    and make ``import torch`` fail. The GNN encoder must surface a clear,
    actionable error rather than a bare ModuleNotFoundError.
    """
    import builtins
    import sys

    import factorgraph_st.model.gnn_encoder as gnn

    inst = _instance()  # construct BEFORE patching imports

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("No module named 'torch'")
        return real_import(name, *args, **kwargs)

    # Drop any cached torch so _require_torch's `import torch` re-imports (and
    # thus hits the fake importer) instead of returning the cached module.
    for mod in list(sys.modules):
        if mod == "torch" or mod.startswith("torch."):
            monkeypatch.delitem(sys.modules, mod, raising=False)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="GNN encoder requires"):
        gnn.encode_graph_gnn(inst.X, inst.coords, inst.section_id, inst.edges, d=6, seed=0)


def test_gnn_encoder_shape_and_dtype():
    pytest.importorskip("torch")
    from factorgraph_st.model.gnn_encoder import encode_graph_gnn

    inst = _instance()
    H = encode_graph_gnn(
        inst.X, inst.coords, inst.section_id, inst.edges, d=6, seed=0, epochs=5
    )
    assert H.shape == (inst.X.shape[0], 6)
    assert H.dtype == np.float32
    assert np.isfinite(H).all()


def test_gnn_encoder_loss_decreases():
    pytest.importorskip("torch")
    from factorgraph_st.model.gnn_encoder import encode_graph_gnn

    inst = _instance()
    _, history = encode_graph_gnn(
        inst.X, inst.coords, inst.section_id, inst.edges, d=6, seed=0,
        epochs=40, return_history=True,
    )
    assert len(history) >= 5
    # Loss at the end should be meaningfully below the start.
    assert history[-1] < history[0]


def test_gnn_encoder_deterministic_for_fixed_seed():
    pytest.importorskip("torch")
    from factorgraph_st.model.gnn_encoder import encode_graph_gnn

    inst = _instance()
    H1 = encode_graph_gnn(inst.X, inst.coords, inst.section_id, inst.edges, d=6, seed=7, epochs=10)
    H2 = encode_graph_gnn(inst.X, inst.coords, inst.section_id, inst.edges, d=6, seed=7, epochs=10)
    np.testing.assert_array_equal(H1, H2)


def test_gnn_encoder_integrates_with_fit_transform():
    pytest.importorskip("torch")

    inst = _instance()
    out = fit_transform(
        inst.X, inst.coords, inst.section_id, inst.edges, d=6,
        K_shared=2, K_private=1, seed=5, encoder="gnn", encoder_kwargs={"epochs": 15},
    )
    from factorgraph_st.schemas import validate_outputs

    validate_outputs(
        out.H, out.W, out.Z_shared, out.Z_private, out.domain_id,
        inst.X.shape[0], inst.X.shape[1],
    )
    # GNN output should differ from the random projection on the same inputs.
    H_random = encode_graph(inst.X, inst.coords, inst.section_id, inst.edges, d=6, seed=5)
    assert not np.array_equal(out.H, H_random)
