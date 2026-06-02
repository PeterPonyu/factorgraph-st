"""Model helpers for FactorGraph-ST."""

from factorgraph_st.model.decoder import FactorGraphOutputs, decode_factors, fit_transform
from factorgraph_st.model.encoder import encode_graph

# GNN encoder symbols are imported lazily via gnn_encoder.* to keep the package
# numpy-only at import time; importing them here would not pull torch (the module
# only requires torch when its functions are *called*), so re-export for
# discoverability per the issue's acceptance criterion.
from factorgraph_st.model.gnn_encoder import GNNEncoder, encode_graph_gnn

__all__ = [
    "FactorGraphOutputs",
    "GNNEncoder",
    "decode_factors",
    "encode_graph",
    "encode_graph_gnn",
    "fit_transform",
]
