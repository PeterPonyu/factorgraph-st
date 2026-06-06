"""Model-side domain-detection benchmark adapter for FactorGraph-ST (#368).

The cross-project ``domain_detection`` comparison suite (parent repo
``scripts/benchmark/factorgraph_st``, contract
``benchmark_contracts/factorgraph_st_benchmark_methods.json``) scores *our* model
alongside clean-room public-competitor adapters under one env + seed. This module
is the in-repo entry point the suite calls to run the learned FactorGraph-ST model
and get back the contract's ``common_output`` prediction arrays:

    {labels, embedding, factors, loadings} + a model-side provenance contribution

It is a thin, brand-clean wrapper over :func:`factorgraph_st.model.learned.fit_transform_gnmf`
-- the graph-regularized nonnegative factorization (the trained model whose
spatial coherence is LEARNED via the graph Laplacian, with domains clustered on
the factor scores ``H`` alone). No metrics are computed here: the suite owns
scoring and emission, so per the provenance-tagged metric contract the supervised
``ground_truth_ari`` belongs in the suite's *metrics* dict (never run_metadata top
level) and this adapter's :class:`DomainDetectionResult.provenance` flows into the
emission's model/eval_policy provenance block.

Preprocessing (normalization / HVG) is applied identically *upstream of every
adapter* by the suite (recorded in ``run_metadata.normalization``); this adapter
therefore consumes ``adata.X`` as given and only builds the kNN spatial graph the
model needs. The numpy-only :func:`run_domain_detection_arrays` core takes raw
arrays so it is testable without AnnData; :func:`run_domain_detection` is the
AnnData-shaped convenience the suite uses (duck-typed -- it reads ``adata.X`` and
``adata.obsm['spatial']`` without importing anndata).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

from factorgraph_st.model.learned import fit_transform_gnmf
from factorgraph_st.schemas import validate_inputs

#: This adapter is a pure, seeded numpy computation: same inputs + seed -> identical
#: outputs. Recorded in provenance so the suite can tag the reproducibility level.
REPRODUCIBILITY_LEVEL = "deterministic_seeded"

#: Brand-neutral functional id for *our* model in the comparison contract.
METHOD_ID = "factorgraph_st_gnmf"


@dataclass(frozen=True)
class DomainDetectionResult:
    """The contract ``common_output`` prediction arrays + a provenance contribution.

    Attributes
    ----------
    labels:
        ``(n_spots,)`` int64 predicted domain assignment (the required output).
    embedding:
        ``(n_spots, n_factors)`` latent embedding. For the learned model the
        embedding IS the per-spot factor-score matrix ``H``.
    factors:
        ``(n_spots, n_factors)`` per-spot factor scores (``H``); enables the
        suite's ``factor_coherence`` metric.
    loadings:
        ``(n_factors, n_genes)`` gene loadings (``W.T``); enables ``factor_diversity``.
    provenance:
        Model-side provenance contribution (method id, family, hyperparameters,
        seed, reproducibility level, optimizer trace). The suite folds this into
        the emission's model/eval_policy provenance block. Carries NO ground-truth
        metric: ``ground_truth_ari`` is the suite's to compute and place in metrics.
    """

    labels: np.ndarray
    embedding: np.ndarray
    factors: np.ndarray
    loadings: np.ndarray
    provenance: dict[str, Any]


def build_knn_edges(coords: np.ndarray, k: int) -> np.ndarray:
    """Undirected, de-duplicated kNN spatial edges as a ``(2, n_edges)`` int64 array.

    Each spot connects to its ``k`` nearest neighbors (self excluded); the edge set
    is symmetrized so every spatial relationship appears once per direction. This
    mirrors the real-data runner's edge build so the adapter and the runner score
    the same graph.
    """
    coords = np.asarray(coords, dtype=np.float64)
    n = coords.shape[0]
    k_eff = min(k, max(n - 1, 0))
    if n == 0 or k_eff == 0:
        return np.empty((2, 0), dtype=np.int64)
    tree = cKDTree(coords)
    # query k+1 to drop the self-match in column 0.
    _, idx = tree.query(coords, k=k_eff + 1)
    idx = np.atleast_2d(idx)
    src = np.repeat(np.arange(n, dtype=np.int64), k_eff)
    dst = idx[:, 1:].reshape(-1).astype(np.int64)
    a = np.concatenate([src, dst])
    b = np.concatenate([dst, src])
    pairs = np.unique(np.stack([a, b], axis=1), axis=0)
    return pairs.T.astype(np.int64)


def run_domain_detection_arrays(
    X: np.ndarray,
    coords: np.ndarray,
    *,
    seed: int,
    n_domains: int,
    k_shared: int = 4,
    k_private: int = 2,
    lam: float = 1.0,
    n_iter: int = 200,
    tol: float = 1e-4,
    knn: int = 6,
) -> DomainDetectionResult:
    """Run the learned model on raw arrays; return the contract outputs (numpy-only).

    ``X`` is the ``(n_spots, n_genes)`` expression matrix as the suite normalized it
    (no further normalization here); ``coords`` is ``(n_spots, 2)``. Builds the kNN
    spatial graph, fits the graph-regularized NMF, and packages
    ``{labels, embedding, factors, loadings}`` plus a model-side provenance dict.
    """
    X = np.ascontiguousarray(X, dtype=np.float32)
    coords = np.asarray(coords, dtype=np.float32)
    section_id = np.zeros(X.shape[0], dtype=np.int64)
    edges = build_knn_edges(coords, knn)
    validate_inputs(X, coords, section_id, edges)

    out, gnmf = fit_transform_gnmf(
        X,
        edges,
        K_shared=k_shared,
        K_private=k_private,
        n_domains=n_domains,
        lam=lam,
        n_iter=n_iter,
        tol=tol,
        seed=seed,
    )

    provenance: dict[str, Any] = {
        "method": METHOD_ID,
        "model_family": "graph_regularized_nonnegative_factorization",
        "reproducibility_level": REPRODUCIBILITY_LEVEL,
        "seed": int(seed),
        "n_factors": int(out.H.shape[1]),
        "hyperparameters": {
            "K_shared": int(k_shared),
            "K_private": int(k_private),
            "n_domains": int(n_domains),
            "lam": float(lam),
            "n_iter": int(n_iter),
            "tol": float(tol),
            "knn": int(knn),
        },
        "gnmf_objective_initial": float(gnmf.objective[0]) if gnmf.objective.size else None,
        "gnmf_objective_final": float(gnmf.objective[-1]) if gnmf.objective.size else None,
        "gnmf_n_iter_run": int(gnmf.n_iter_run),
    }

    return DomainDetectionResult(
        labels=np.asarray(out.domain_id, dtype=np.int64),
        embedding=np.asarray(out.H, dtype=np.float32),
        factors=np.asarray(out.H, dtype=np.float32),
        loadings=np.asarray(out.W, dtype=np.float32).T,  # (n_genes, k) -> (k, n_genes)
        provenance=provenance,
    )


def run_domain_detection(adata: Any, *, seed: int, n_domains: int, **kwargs: Any) -> DomainDetectionResult:
    """AnnData-shaped entry point the suite calls (duck-typed; no anndata import).

    Reads ``adata.X`` (densified if sparse) and ``adata.obsm['spatial']`` and
    delegates to :func:`run_domain_detection_arrays`. Extra keyword arguments
    (``k_shared``, ``lam``, ``knn`` ...) are forwarded to the core.
    """
    X = adata.X
    X = np.asarray(X.todense()) if sp.issparse(X) else np.asarray(X)
    if "spatial" not in adata.obsm:
        raise ValueError("adata.obsm['spatial'] is required for domain detection")
    coords = np.asarray(adata.obsm["spatial"])
    return run_domain_detection_arrays(X, coords, seed=seed, n_domains=n_domains, **kwargs)
