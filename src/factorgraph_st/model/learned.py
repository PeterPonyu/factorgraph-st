"""Pure-numpy *trained* graph-regularized nonnegative matrix factorization (GNMF).

This module replaces the fixed random Gaussian projection "encoder"
(``model/encoder.py``) with a genuinely trained factor model. It minimizes

    ||X - H @ W.T||_F^2  +  lam * Tr(H.T @ L @ H)

over ``H >= 0`` (per-spot factor scores, ``n_spots x n_factors``) and
``W >= 0`` (gene loadings, ``n_genes x n_factors``), where ``L = D - A`` is the
unnormalized Laplacian of the spatial kNN graph supplied via ``edges``. The
optimization uses the standard GNMF multiplicative-update rules (Cai et al.,
2011), which keep both factors nonnegative and drive the objective monotonically
non-increasing.

Two design properties matter for scientific honesty (see the FactorGraph section
of the reference-results integration issues):

* **Spatial coherence is LEARNED via the graph regularizer**, not baked into the
  feature matrix. Coordinates are never concatenated into ``X``; the only spatial
  signal the model sees is the graph Laplacian penalty ``Tr(H.T L H)``, which
  encourages neighboring spots to share factor scores.
* **Domains are clustered on the factor scores ``H`` alone**, never on
  ``[H | coords]``. Because coordinates are not a clustering input, Moran's I on
  the resulting ``domain_id`` is no longer high by construction.

The runtime stays numpy-only: no torch, no scipy. All arithmetic is dense numpy,
with the adjacency applied as a scatter-add over the (symmetrized) edge list so
the ``n_spots x n_spots`` matrix is never materialized.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from factorgraph_st.model.decoder import (
    FactorGraphOutputs,
    _apply_canonical_gauge,
    _cluster_domains,
)
from factorgraph_st.schemas import validate_outputs

# Floor added to multiplicative-update denominators so a zero denominator can
# never produce a NaN/Inf; small enough not to perturb the converged optimum.
_EPS = 1e-12


@dataclass
class GNMFResult:
    """Outputs of a graph-regularized NMF fit.

    Attributes
    ----------
    H:
        ``(n_spots, n_factors)`` nonnegative per-spot factor scores (float32).
    W:
        ``(n_genes, n_factors)`` nonnegative gene loadings (float32).
    objective:
        Per-iteration objective trace including the initial value, length
        ``n_iter_run + 1`` (float64). Non-increasing by construction.
    n_iter_run:
        Number of multiplicative-update iterations actually executed (<=
        ``n_iter``; fewer when the relative objective change falls below ``tol``).
    lam_effective:
        The graph-regularization weight actually used internally,
        ``lam * RMS(X)`` (see :func:`fit_gnmf`). Recorded for transparency so the
        scale-relative weighting behind a run is never implicit. ``0.0`` only for
        the degenerate empty-input early return.
    """

    H: np.ndarray
    W: np.ndarray
    objective: np.ndarray
    n_iter_run: int
    lam_effective: float = 0.0


def _graph_terms(edges: np.ndarray, n_spots: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(src_sym, dst_sym, degree)`` for the symmetrized graph.

    The edge list is symmetrized (each ``(s, d)`` contributes both directions) so
    the implied adjacency ``A`` is symmetric and ``L = D - A`` is a valid
    unnormalized Laplacian. ``degree[i]`` is the (multiplicity-weighted) number of
    symmetrized edges incident to spot ``i``. Self-loops contribute equally to a
    spot's degree and its ``A @ H`` row, so they cancel in ``L`` and are harmless.
    """
    if edges.size == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.zeros(n_spots, dtype=np.float64),
        )
    src, dst = edges
    src_sym = np.concatenate([src, dst])
    dst_sym = np.concatenate([dst, src])
    degree = np.bincount(src_sym, minlength=n_spots).astype(np.float64)
    return src_sym.astype(np.int64), dst_sym.astype(np.int64), degree


def _adj_matmul(src_sym: np.ndarray, dst_sym: np.ndarray, H: np.ndarray, n_spots: int) -> np.ndarray:
    """Compute ``A @ H`` for the symmetrized adjacency without materializing ``A``."""
    out = np.zeros((n_spots, H.shape[1]), dtype=np.float64)
    if src_sym.size:
        np.add.at(out, src_sym, H[dst_sym])
    return out


def fit_gnmf(
    X: np.ndarray,
    edges: np.ndarray,
    n_factors: int,
    *,
    lam: float = 1.0,
    n_iter: int = 200,
    tol: float = 1e-4,
    seed: int = 0,
) -> GNMFResult:
    """Fit graph-regularized NMF ``X ~= H @ W.T`` with a Laplacian smoothness prior.

    Parameters
    ----------
    X:
        ``(n_spots, n_genes)`` nonnegative expression matrix. Negative entries are
        clipped to zero (NMF requires a nonnegative target).
    edges:
        ``(2, n_edges)`` int64 spatial kNN edge list. Symmetrized internally; the
        induced Laplacian ``L = D - A`` supplies the spatial smoothness penalty.
    n_factors:
        Number of latent factors ``k`` (must be positive).
    lam:
        Graph-regularization strength (``lam >= 0``), interpreted *relative to the
        data scale*: internally the Laplacian penalty is weighted by
        ``lam_eff = lam * RMS(X)`` with ``RMS(X) = sqrt(mean(X^2))`` (#392). Because
        ``RMS`` is degree-1-homogeneous in ``X``, this makes the fit -- and the
        domain partition clustered from ``H`` -- **invariant to a global rescaling
        of** ``X`` (e.g. raw counts vs CPM vs log1p): under ``X -> cX`` the
        reconstruction term scales as ``c^2`` and the Laplacian term as ``c``, and
        scaling ``lam_eff`` by ``c`` exactly compensates so ``H -> sqrt(c) H`` and
        the z-normalized clustering is unchanged. A bare constant ``lam`` (the prior
        behavior) did NOT have this property -- the recovered domains depended on
        the arbitrary units of ``X``. ``lam = 0`` still recovers plain NMF; larger
        values pull neighboring spots toward equal factor scores.
    n_iter:
        Maximum number of multiplicative-update iterations.
    tol:
        Relative objective-change threshold for early stopping.
    seed:
        Seeds the nonnegative initialization; identical ``seed`` yields a
        bitwise-identical fit.

    Returns
    -------
    GNMFResult
        Nonnegative ``H``/``W`` (float32), the float64 objective trace, and the
        number of iterations run.
    """
    if n_factors <= 0:
        raise ValueError(f"n_factors must be positive; got {n_factors}")
    if lam < 0:
        raise ValueError(f"lam must be non-negative; got {lam}")
    if n_iter <= 0:
        raise ValueError(f"n_iter must be positive; got {n_iter}")

    Xf = np.clip(np.asarray(X, dtype=np.float64), 0.0, None)
    n_spots, n_genes = Xf.shape
    if n_spots == 0 or n_genes == 0:
        empty_H = np.zeros((n_spots, n_factors), dtype=np.float32)
        empty_W = np.zeros((n_genes, n_factors), dtype=np.float32)
        return GNMFResult(H=empty_H, W=empty_W, objective=np.zeros(1, dtype=np.float64), n_iter_run=0)

    src_sym, dst_sym, degree = _graph_terms(edges, n_spots)
    degree_col = degree[:, None]

    # Scale-relative regularization (#392): weight the Laplacian penalty by the
    # data RMS so the recon (~c^2) and smoothness (~c) terms stay in fixed balance
    # under a global rescale X -> cX. This makes the recovered domain partition
    # invariant to the arbitrary units of X (counts / CPM / log1p). A bare constant
    # `lam` did not, so domains depended on input scale.
    data_rms = float(np.sqrt(np.mean(Xf * Xf)))
    lam_eff = float(lam) * (data_rms if np.isfinite(data_rms) and data_rms > _EPS else 1.0)

    # Seeded nonnegative init scaled so H @ W.T has the magnitude of X. Drawing
    # from a strictly-positive uniform avoids exact zeros, which multiplicative
    # updates can never escape.
    rng = np.random.default_rng(seed)
    scale = np.sqrt(max(Xf.mean(), _EPS) / n_factors)
    H = (rng.random((n_spots, n_factors)) + _EPS) * scale
    W = (rng.random((n_genes, n_factors)) + _EPS) * scale

    def objective(H: np.ndarray, W: np.ndarray) -> float:
        residual = Xf - H @ W.T
        recon = float(np.sum(residual * residual))
        # Tr(H.T L H) = Tr(H.T D H) - Tr(H.T A H); both terms via scatter-add.
        AH = _adj_matmul(src_sym, dst_sym, H, n_spots)
        smooth = float(np.sum(degree_col * H * H) - np.sum(H * AH))
        return recon + lam_eff * smooth

    trace = [objective(H, W)]
    n_run = 0
    for _ in range(n_iter):
        # --- H update: H *= (X W + lam A H) / (H (W.T W) + lam D H) -----------
        AH = _adj_matmul(src_sym, dst_sym, H, n_spots)
        numer_H = Xf @ W + lam_eff * AH
        denom_H = H @ (W.T @ W) + lam_eff * degree_col * H
        H = H * numer_H / (denom_H + _EPS)

        # --- W update: W *= (X.T H) / (W (H.T H)) -----------------------------
        numer_W = Xf.T @ H
        denom_W = W @ (H.T @ H)
        W = W * numer_W / (denom_W + _EPS)

        n_run += 1
        obj = objective(H, W)
        prev = trace[-1]
        trace.append(obj)
        if prev > _EPS and abs(prev - obj) / prev < tol:
            break

    return GNMFResult(
        H=H.astype(np.float32),
        W=W.astype(np.float32),
        objective=np.asarray(trace, dtype=np.float64),
        n_iter_run=n_run,
        lam_effective=lam_eff,
    )


def fit_transform_gnmf(
    X: np.ndarray,
    edges: np.ndarray,
    *,
    K_shared: int = 4,
    K_private: int = 2,
    n_domains: int = 5,
    lam: float = 1.0,
    n_iter: int = 200,
    tol: float = 1e-4,
    seed: int = 0,
) -> tuple[FactorGraphOutputs, GNMFResult]:
    """Fit GNMF and package the result as :class:`FactorGraphOutputs`.

    The ``k = K_shared + K_private`` jointly-learned factors are split by the same
    column convention the projection decoder uses: the first ``K_shared`` columns
    of ``H`` become ``Z_shared`` and the remainder ``Z_private``. Loadings are
    taken directly from the GNMF fit (they already satisfy ``X ~= Z @ W.T``) and
    re-gauged to unit-norm columns via the shared decoder helper. Private factors
    are NOT section-gated: unlike the projection path, the learned model is left
    free to discover structure rather than having section identity baked in.

    Domains are clustered on the factor scores ``H`` ALONE (plain k-means, no
    coordinates), which is what removes the Moran's-I-by-construction artifact.

    Returns the validated outputs and the raw :class:`GNMFResult` (objective trace
    et al.) so callers can record learning evidence.
    """
    if K_shared < 0 or K_private < 0:
        raise ValueError("K_shared and K_private must be non-negative")
    n_factors = K_shared + K_private
    if n_factors <= 0:
        raise ValueError("K_shared + K_private must be positive")

    from factorgraph_st.repro import set_seed

    set_seed(seed)
    result = fit_gnmf(X, edges, n_factors, lam=lam, n_iter=n_iter, tol=tol, seed=seed)

    H = result.H
    Z_shared = H[:, :K_shared].astype(np.float32, copy=False)
    Z_private = H[:, K_shared:].astype(np.float32, copy=False)
    Z_shared, Z_private, W = _apply_canonical_gauge(Z_shared, Z_private, result.W)

    # Cluster domains on the factor scores alone (no coordinates, no graph): the
    # learned H already carries the spatial structure absorbed via the Laplacian.
    empty_coords = np.zeros((H.shape[0], 0), dtype=np.float32)
    domain_id = _cluster_domains(H, empty_coords, n_domains, edges=None, seed=seed).astype(
        np.int64, copy=False
    )

    outputs = FactorGraphOutputs(H=H, W=W, Z_shared=Z_shared, Z_private=Z_private, domain_id=domain_id)
    validate_outputs(
        outputs.H, outputs.W, outputs.Z_shared, outputs.Z_private, outputs.domain_id, X.shape[0], X.shape[1]
    )
    return outputs, result
