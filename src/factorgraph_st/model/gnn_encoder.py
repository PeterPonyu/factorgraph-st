"""Trainable GNN encoder for FactorGraph-ST (issue #181, PR 1/N).

This is an **opt-in** alternative to the fixed random Gaussian projection in
:mod:`factorgraph_st.model.encoder`. The random projection remains the default
fit path; selecting ``encoder="gnn"`` in :func:`factorgraph_st.model.fit_transform`
routes here instead.

Scope / honesty
---------------
This PR adds the *machinery* for a trainable encoder. It does NOT claim
scientific validity: flipping the default to ``gnn`` and validating
factor-recovery against the synthetic benchmark are later PRs in the #181
series. The acceptance criterion "factor-recovery metrics measurably higher for
GNN" is explicitly deferred (see the issue staging).

Architecture
------------
A 2-layer message-passing encoder implemented in plain torch using
``index_add_`` scatter over the kNN edge list — matching the repo's existing
numpy ``np.add.at`` scatter idiom in :func:`encoder.encode_graph` and avoiding a
torch-geometric / DGL dependency. Each layer computes a mean aggregation of
neighbor hidden states, concatenates it with the node's own hidden state, and
applies a linear + ReLU. A linear head produces the latent ``Z`` of width ``d``.

Input features are constructed identically to the random encoder
(:func:`encoder.encode_graph`): neighbor-mean-aggregated expression, mean-centered
expression, mean-centered coordinates, and a normalized section feature. This
keeps the two encoders comparable on the same input geometry.

Self-supervised objective
--------------------------
``loss = recon + lambda_smooth * smooth``

- **recon**: a linear decoder maps ``Z`` back to the input feature matrix; the
  mean-squared reconstruction error is a standard autoencoder objective that
  forces ``Z`` to retain the dominant feature structure.
- **smooth**: a graph-Laplacian smoothness penalty (mean squared latent
  difference across kNN edges) encodes the spatial prior that neighboring spots
  share latent factors — a numpy/torch-native analogue of the graph-regularized
  objectives used by spatial factor models, expressed in clean-room terms (no
  baseline class names / narrative).

Both terms are unsupervised (no ground-truth labels), so the encoder is a
genuinely trained model rather than a fixed projection.

Determinism
-----------
Seeding pins Python ``random``, NumPy, and torch (CPU + CUDA) via
:func:`factorgraph_st.repro.set_seed`, and the optimizer / parameter init are
seeded so a fixed ``seed`` yields byte-identical embeddings on the same device.
"""

from __future__ import annotations

import numpy as np

from factorgraph_st.schemas import validate_inputs

__all__ = ["encode_graph_gnn", "GNNEncoder"]


def _require_torch():
    """Import torch or raise a clear, actionable error.

    The GNN encoder is gated behind the optional ``model`` extra so the core
    package stays numpy-only (``pip install factorgraph-st`` does not pull
    torch). Selecting ``encoder="gnn"`` without torch installed surfaces this
    message instead of a bare ``ModuleNotFoundError``.
    """
    try:
        import torch  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise ImportError(
            "GNN encoder requires PyTorch. Install the optional extra: "
            "`pip install 'factorgraph-st[model]'` (or `pip install torch`)."
        ) from exc
    return torch


def _build_features(
    X: np.ndarray,
    coords: np.ndarray,
    section_id: np.ndarray,
    edges: np.ndarray,
) -> np.ndarray:
    """Construct the same node feature matrix the random encoder projects.

    Mirrors :func:`factorgraph_st.model.encoder.encode_graph` so the GNN sees an
    identical input geometry (neighbor-mean expression, centered expression,
    centered coords, normalized section feature). Kept in float64 for the
    feature build; the torch graph runs in float32.
    """
    n_spots = X.shape[0]
    neighbor_sum = np.zeros_like(X, dtype=np.float32)
    degree = np.zeros(n_spots, dtype=np.float32)
    if edges.size:
        src, dst = edges
        np.add.at(neighbor_sum, src, X[dst])
        np.add.at(degree, src, 1.0)
    neighbor_mean = neighbor_sum / np.maximum(degree[:, None], 1.0)

    x_mean = X.mean(axis=0, keepdims=True)
    centered_x = X - x_mean
    centered_coords = coords - coords.mean(axis=0, keepdims=True)
    section_scale = max(int(section_id.max(initial=0)), 1)
    section_feature = (section_id.astype(np.float32) / section_scale)[:, None]
    features = np.concatenate(
        [centered_x, neighbor_mean - x_mean, centered_coords, section_feature],
        axis=1,
    )
    return features.astype(np.float32)


def _make_encoder_cls():
    """Define the torch ``GNNEncoder`` lazily (only when torch is present)."""
    torch = _require_torch()
    nn = torch.nn

    class _GNNEncoder(nn.Module):
        """2-layer mean-aggregation message-passing encoder + linear latent head."""

        def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
            super().__init__()
            # Each layer takes [self_hidden | neighbor_mean_hidden] -> hidden.
            self.lin1 = nn.Linear(in_dim * 2, hidden_dim)
            self.lin2 = nn.Linear(hidden_dim * 2, hidden_dim)
            self.head = nn.Linear(hidden_dim, out_dim)
            self.act = nn.ReLU()

        @staticmethod
        def _aggregate(h, src, dst, deg):
            """Mean-aggregate neighbor hidden states via index_add_ scatter."""
            agg = torch.zeros_like(h)
            if src.numel() > 0:
                agg.index_add_(0, src, h.index_select(0, dst))
            return agg / deg

        def forward(self, x, src, dst, deg):
            agg = self._aggregate(x, src, dst, deg)
            h = self.act(self.lin1(torch.cat([x, agg], dim=1)))
            agg2 = self._aggregate(h, src, dst, deg)
            h = self.act(self.lin2(torch.cat([h, agg2], dim=1)))
            return self.head(h)

    return _GNNEncoder


# Public alias resolved lazily; importing the module never requires torch.
def GNNEncoder(*args, **kwargs):  # noqa: N802 - factory mimics a class
    """Construct the torch GNN encoder module (requires the ``model`` extra)."""
    return _make_encoder_cls()(*args, **kwargs)


def encode_graph_gnn(
    X: np.ndarray,
    coords: np.ndarray,
    section_id: np.ndarray,
    edges: np.ndarray,
    d: int = 16,
    seed: int = 0,
    *,
    hidden_dim: int = 32,
    epochs: int = 200,
    lr: float = 1e-2,
    lambda_smooth: float = 0.5,
    return_history: bool = False,
):
    """Train the GNN encoder and return the latent embedding ``H`` ``(n_spots, d)``.

    Parameters mirror :func:`encoder.encode_graph` (``X``, ``coords``,
    ``section_id``, ``edges``, ``d``, ``seed``) plus training hyperparameters.
    When ``return_history`` is True, returns ``(H, loss_history)`` where
    ``loss_history`` is a list of per-epoch total-loss floats (used by tests to
    assert the loss decreases).

    Raises ``ImportError`` (with install guidance) if torch is unavailable.
    """
    torch = _require_torch()
    validate_inputs(X, coords, section_id, edges)
    if d <= 0:
        raise ValueError(f"d must be positive; got {d}")

    n_spots = X.shape[0]
    if n_spots == 0:
        empty = np.empty((0, d), dtype=np.float32)
        return (empty, []) if return_history else empty

    from factorgraph_st.repro import set_seed

    set_seed(seed)

    features_np = _build_features(X, coords, section_id, edges)
    device = torch.device("cpu")
    x = torch.from_numpy(features_np).to(device)
    in_dim = x.shape[1]

    if edges.size:
        src = torch.from_numpy(np.ascontiguousarray(edges[0])).long().to(device)
        dst = torch.from_numpy(np.ascontiguousarray(edges[1])).long().to(device)
    else:
        src = torch.empty(0, dtype=torch.long, device=device)
        dst = torch.empty(0, dtype=torch.long, device=device)
    # Degree (clamped at 1 to avoid div-by-zero on isolated nodes), matching the
    # numpy encoder's ``np.maximum(degree, 1.0)``.
    deg = torch.zeros(n_spots, 1, device=device)
    if src.numel() > 0:
        deg.index_add_(0, src, torch.ones(src.numel(), 1, device=device))
    deg = deg.clamp_min(1.0)

    # Seed parameter init deterministically (independent of any prior torch RNG
    # consumption inside set_seed).
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    torch.manual_seed(int(seed))

    enc = _make_encoder_cls()(in_dim, hidden_dim, d).to(device)
    decoder = torch.nn.Linear(d, in_dim).to(device)
    params = list(enc.parameters()) + list(decoder.parameters())
    opt = torch.optim.Adam(params, lr=lr)

    history: list[float] = []
    enc.train()
    decoder.train()
    for _ in range(int(epochs)):
        opt.zero_grad()
        z = enc(x, src, dst, deg)
        recon = decoder(z)
        loss_recon = torch.mean((recon - x) ** 2)
        if src.numel() > 0:
            diff = z.index_select(0, src) - z.index_select(0, dst)
            loss_smooth = torch.mean(torch.sum(diff * diff, dim=1))
        else:
            loss_smooth = torch.zeros((), device=device)
        loss = loss_recon + lambda_smooth * loss_smooth
        loss.backward()
        opt.step()
        history.append(float(loss.detach().cpu()))

    enc.eval()
    with torch.no_grad():
        H = enc(x, src, dst, deg).detach().cpu().numpy().astype(np.float32)

    return (H, history) if return_history else H
