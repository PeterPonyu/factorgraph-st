# FactorGraph-ST src

The code here is a **deterministic, non-learned MVP baseline**, not the planned learned model. `model/encoder.py:encode_graph` is a single neighbor-mean aggregation over the spatial graph followed by a fixed random Gaussian projection (no trained parameters, no multi-hop message passing). `model/decoder.py:decode_factors` rectifies the embedding and fits gene loadings by clipped nonnegative least squares (not a learned NMF). The parametric, trained graph encoder is planned but not yet implemented. All performance and biological claims remain gated by the claim ledger.
