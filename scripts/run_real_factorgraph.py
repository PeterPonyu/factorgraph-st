#!/usr/bin/env python
"""Real-data results path for FactorGraph-ST on the DLPFC Visium dataset.

Loads a real ``.h5ad`` spatial-transcriptomics matrix, builds a single-section
kNN spatial graph, runs the FactorGraph-ST MVP ``fit_transform``, computes
INTRINSIC (unsupervised) metrics only, and emits the uniform computational
results contract (``metrics.json`` + ``run_metadata.json`` + ``outputs/``) via
the vendored :mod:`factorgraph_st.results_contract`.

SCIENTIFIC-HONESTY NOTE (see consensus plan §4.3): the FactorGraph-ST "model"
is NOT a learned factor model. The encoder is a fixed random Gaussian
projection of neighbor-aggregated features (``encoder.py``: a single seeded
``rng.normal`` projection, no training/backprop), the factor basis is a
deterministic rectified-and-normalized transform of ``H`` (``_positive_basis``),
and loadings are a numpy-only NNLS fit. Spatial coordinates + neighbor means are
baked into the projected features (``encoder.py``). Domains are assigned by a
deterministic GRAPH-AWARE k-means pass on the joint ``[H | coords]`` feature
(``decoder.py:_cluster_domains``) -- because coordinates are an explicit
clustering input, Moran's I on ``domain_id`` is HIGH BY CONSTRUCTION (it partly
re-reads out the coordinates fed in), not evidence of learned spatial biology.
This runner pairs every reported ``morans_i_domain`` with a permutation-shuffled
null (``morans_i_domain_null``) and records a mandatory interpretability block so
the value is only ever interpreted as a delta over chance.

This script lives entirely in the runner: densify, conditional log1p, and the
kNN edge build are runner-local; no model source is touched.

Usage (under the project conda env)::

    conda run --no-capture-output -n dl python scripts/run_real_factorgraph.py \
        --h5ad ../data/processed/dlpfc_GSE307403_Br2719/anndata.h5ad
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import scanpy as sc
import scipy.sparse as sp
from scipy.spatial import cKDTree

from factorgraph_st import results_contract
from factorgraph_st.eval.metrics import morans_i, shared_private_separation
from factorgraph_st.model.decoder import fit_transform
from factorgraph_st.schemas import validate_inputs

# Default dataset path, relative to the PARENT orchestration repo root (this
# package lives at ``<parent>/factorgraph-st`` so the data dir is one level up).
_DEFAULT_H5AD = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "processed"
    / "dlpfc_GSE307403_Br2719"
    / "anndata.h5ad"
)

#: Raw-count heuristic threshold for the conditional log1p guard.
_RAW_COUNT_MAX_THRESHOLD = 50.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--h5ad",
        type=Path,
        default=_DEFAULT_H5AD,
        help="Path to the input AnnData .h5ad (default: DLPFC GSE307403 Br2719).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Results root (default: <repo>/results).",
    )
    parser.add_argument("--knn", type=int, default=6, help="k for kNN spatial edges.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--d", type=int, default=16, help="Embedding dim d.")
    parser.add_argument("--k-shared", type=int, default=4, help="Number of shared factors.")
    parser.add_argument("--k-private", type=int, default=2, help="Number of private factors.")
    parser.add_argument("--n-domains", type=int, default=5, help="Number of domains.")
    parser.add_argument(
        "--n-null-shuffles",
        type=int,
        default=10,
        help="Number of permutation shuffles for the Moran's I null (>=10).",
    )
    parser.add_argument(
        "--already-normalized",
        action="store_true",
        help="Skip the conditional log1p; assume X is already normalized.",
    )
    return parser.parse_args()


def _looks_like_raw_counts(X) -> bool:
    """Heuristic: integer-valued OR a large max value indicates raw counts."""
    x_max = float(X.max()) if X.size or sp.issparse(X) else 0.0
    if x_max > _RAW_COUNT_MAX_THRESHOLD:
        return True
    # Integer-valued check on a bounded sample (avoid densifying the full matrix).
    sample = X[: min(X.shape[0], 100)]
    sample = sample.toarray() if sp.issparse(sample) else np.asarray(sample)
    return bool(np.allclose(sample, np.round(sample)))


def _build_knn_edges(coords: np.ndarray, k: int) -> np.ndarray:
    """Build undirected kNN spatial edges as a ``(2, n_edges)`` int64 COO array.

    Each spot connects to its ``k`` nearest neighbors (excluding itself). The
    edge set is symmetrized (undirected) and de-duplicated so every spatial
    relationship appears once per direction.
    """
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
    # Symmetrize + dedupe.
    a = np.concatenate([src, dst])
    b = np.concatenate([dst, src])
    pairs = np.unique(np.stack([a, b], axis=1), axis=0)
    return pairs.T.astype(np.int64)


def _factor_stats(out) -> dict[str, float]:
    """Intrinsic structural summaries of the recovered factors and domains."""
    H = out.H
    # Effective rank of H via singular values (numpy rank with default tol).
    h_rank = int(np.linalg.matrix_rank(H.astype(np.float64))) if H.size else 0
    W = out.W
    w_sparsity = float((W == 0).mean()) if W.size else 0.0
    domains, counts = np.unique(out.domain_id, return_counts=True)
    domain_count = int(domains.size)
    # Shannon entropy (nats) of the per-domain size distribution.
    if counts.size:
        p = counts.astype(np.float64) / counts.sum()
        size_entropy = float(-(p * np.log(p)).sum())
    else:
        size_entropy = 0.0
    return {
        "H_rank_effective": float(h_rank),
        "W_sparsity": w_sparsity,
        "domain_count": float(domain_count),
        "domain_size_entropy": size_entropy,
    }


def _morans_null(domain_id: np.ndarray, edges: np.ndarray, n_shuffles: int, seed: int) -> float:
    """Mean Moran's I over permutation-shuffled domain labels (chance baseline)."""
    rng = np.random.default_rng(seed)
    vals = []
    base = domain_id.astype(np.float64)
    for _ in range(max(int(n_shuffles), 1)):
        vals.append(morans_i(rng.permutation(base), edges))
    return float(np.mean(vals))


def main() -> None:
    args = _parse_args()
    h5ad_path = args.h5ad.resolve()

    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    t0 = time.perf_counter()

    # --- 1. Load --------------------------------------------------------------
    adata = sc.read_h5ad(h5ad_path)
    n_obs, n_vars = int(adata.shape[0]), int(adata.shape[1])

    # --- 2. Conditional log1p -------------------------------------------------
    if args.already_normalized:
        normalization = {"applied": False, "method": "none"}
    elif _looks_like_raw_counts(adata.X):
        sc.pp.log1p(adata)
        normalization = {"applied": True, "method": "log1p"}
    else:
        normalization = {"applied": False, "method": "none"}

    # --- 3. Densify + section + edges ----------------------------------------
    X = adata.X
    X = np.asarray(X.todense(), dtype=np.float32) if sp.issparse(X) else np.asarray(X, dtype=np.float32)
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    section_id = np.zeros(n_obs, dtype=np.int64)
    edges = _build_knn_edges(coords, args.knn)

    # --- 4. Validate + fit ----------------------------------------------------
    validate_inputs(X, coords, section_id, edges)
    out = fit_transform(
        X,
        coords,
        section_id,
        edges,
        d=args.d,
        K_shared=args.k_shared,
        K_private=args.k_private,
        n_domains=args.n_domains,
        seed=args.seed,
    )

    # --- 5. Intrinsic metrics (NO ARI / GT) ----------------------------------
    morans_real = morans_i(out.domain_id.astype(np.float64), edges)
    morans_null = _morans_null(out.domain_id, edges, args.n_null_shuffles, args.seed)
    sep = shared_private_separation(out.Z_shared, out.Z_private, section_id)

    # DEGENERACY GUARD: main's decoder assigns domains by graph-aware clustering
    # that collapses each connected component of ``edges`` to a single label
    # (decoder.py:_cluster_components). A single-section kNN graph over one
    # contiguous slide is fully connected (one component), so ``domain_id``
    # degenerates to a SINGLE domain regardless of ``n_domains`` -- and every
    # domain-derived metric (Moran's I) is trivially 0. We detect and record this
    # explicitly so the zeros are never mistaken for a computed spatial signal.
    n_domains_found = int(np.unique(out.domain_id).size)
    domain_degenerate = n_domains_found <= 1

    metrics: dict[str, float] = {
        "morans_i_domain": morans_real,
        "morans_i_domain_null": morans_null,
        "morans_i_domain_delta": morans_real - morans_null,
        "domain_clustering_degenerate": float(domain_degenerate),
        "shared_mean_active_sections": sep["shared_mean_active_sections"],
        "private_mean_active_sections": sep["private_mean_active_sections"],
        "n_edges": float(edges.shape[1]),
    }
    metrics.update(_factor_stats(out))

    runtime_s = time.perf_counter() - t0

    # --- 6. Save native outputs ----------------------------------------------
    if args.results_dir is not None:
        results_root = args.results_dir.resolve()
    else:
        results_root = Path(__file__).resolve().parents[1] / "results"
    outputs_dir = results_root / "factorgraph-st" / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    factors_npz = outputs_dir / "factors.npz"
    np.savez_compressed(
        factors_npz,
        H=out.H,
        W=out.W,
        Z_shared=out.Z_shared,
        Z_private=out.Z_private,
        domain_id=out.domain_id,
    )

    # --- 7. Emit contract -----------------------------------------------------
    interpretability = {
        "model_is_learned": False,
        "encoder": (
            "fixed random Gaussian projection of neighbor-aggregated features "
            "(encoder.py, single seeded rng.normal), no training/backprop"
        ),
        "factor_basis": "deterministic rectified+normalized transform of H (_positive_basis)",
        "loadings": "numpy-only NNLS fit (_fit_nonnegative_loadings)",
        "domain_assignment": (
            "graph-aware k-means on joint [H | coords] (decoder.py:_cluster_domains); "
            "coords are an explicit clustering input"
        ),
        "caveat": (
            "morans_i_domain is high by construction: spatial coords are baked into "
            "the encoder features AND used directly as a k-means clustering input, so "
            "domain_id partly re-reads out the coordinates; interpret only as the delta "
            "over morans_i_domain_null, never as learned spatial structure"
        ),
        "domain_clustering_degenerate": bool(domain_degenerate),
        "domain_count_found": n_domains_found,
        "domain_degeneracy_reason": (
            "main's decoder clusters domains graph-aware: each connected component of "
            "the kNN graph collapses to one label (decoder.py:_cluster_components). A "
            "single-section graph over one contiguous DLPFC slide is fully connected "
            "(1 component), so domain_id degenerates to a single domain regardless of "
            "n_domains and morans_i_domain is trivially 0. Multi-section runs (multiple "
            "disconnected components) would yield non-trivial domains. This is recorded "
            "so the domain metrics are not mistaken for a computed spatial signal."
            if domain_degenerate
            else "domain clustering produced multiple domains"
        ),
    }
    run_metadata = {
        "dataset_paths": [str(h5ad_path)],
        "n_obs": n_obs,
        "n_vars": n_vars,
        "seed": args.seed,
        "runtime_s": runtime_s,
        "started_utc": started_utc,
        "device": "cpu",
        "deterministic": True,
        "num_threads": 1,
        "reproducibility_level": "seeded",
        "normalization": normalization,
        "interpretability": interpretability,
        "notes": (
            "Single-section run (section_id all zeros); shared_private_separation "
            "degenerates (single_section=True). FactorGraph-ST MVP is a fixed "
            "random projection + NNLS loadings + graph-aware k-means domains, not a "
            "learned model -- see interpretability block."
        ),
    }

    paths = results_contract.write_results(
        project="factorgraph-st",
        dataset_card_id=results_contract.dataset_card_id([str(h5ad_path)]),
        metrics=metrics,
        outputs={"factors": str(factors_npz)},
        run_metadata=run_metadata,
        results_dir=results_root,
    )

    print(f"Wrote metrics:      {paths['metrics']}")
    print(f"Wrote run_metadata: {paths['run_metadata']}")
    print(f"Wrote outputs:      {factors_npz}")
    print(f"morans_i_domain={morans_real:.6f} null={morans_null:.6f} "
          f"delta={morans_real - morans_null:.6f} runtime_s={runtime_s:.2f}")


if __name__ == "__main__":
    main()
