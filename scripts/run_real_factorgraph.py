#!/usr/bin/env python
"""Real-data results path for FactorGraph-ST on the DLPFC Visium dataset.

Loads a real ``.h5ad`` spatial-transcriptomics matrix, builds a single-section
kNN spatial graph, runs the FactorGraph-ST MVP ``fit_transform``, computes
INTRINSIC (unsupervised) metrics and -- WHEN per-spot ground-truth domain
labels are present in ``adata.obs`` (e.g. the Maynard 2021 spatialLIBD DLPFC
``layer_guess`` / ``ground_truth_domain`` column) -- the supervised Adjusted
Rand Index of the recovered domains vs those labels (#179). ARI is computed
only when a usable GT column is found; it is silently skipped otherwise, and
labels are never fabricated or joined across donors. Emits the uniform
computational results contract (``metrics.json`` + ``run_metadata.json`` +
``outputs/``) via the vendored :mod:`factorgraph_st.results_contract`.

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

This script lives entirely in the runner: densify, normalization
(``normalize_total`` -> ``log1p`` on count-like input, optional HVG selection),
and the kNN edge build are runner-local; no model source is touched.

Usage (under the project conda env)::

    conda run --no-capture-output -n dl python scripts/run_real_factorgraph.py \
        --h5ad ../data/processed/dlpfc_GSE307403_Br2719/anndata.h5ad
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

from factorgraph_st import results_contract
from factorgraph_st.baselines import spatial_smooth_domains
from factorgraph_st.eval.metrics import (
    adjusted_rand_index,
    boundary_f1,
    boundary_precision,
    boundary_recall,
    calinski_harabasz,
    label_invariant_cluster_coherence,
    morans_i,
    normalized_mutual_information,
    shared_private_separation,
    silhouette,
    weighted_dice,
)
from factorgraph_st.model.decoder import _kmeans, fit_transform
from factorgraph_st.model.learned import fit_transform_gnmf
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
        help="Skip normalize_total + log1p; assume X is already normalized.",
    )
    parser.add_argument(
        "--target-sum",
        type=float,
        default=1e4,
        help="Library-size target for sc.pp.normalize_total (default 1e4).",
    )
    parser.add_argument(
        "--n-hvg",
        type=int,
        default=0,
        help=(
            "If > 0, select this many highly variable genes (and subset X) "
            "after normalization. Default 0 = keep all genes."
        ),
    )
    parser.add_argument(
        "--gt-obs-key",
        type=str,
        default=None,
        help=(
            "obs column holding per-spot ground-truth domain labels for the "
            "supervised ARI (#179). Default None = auto-detect "
            f"({', '.join(_GT_OBS_KEY_CANDIDATES)}); ARI is skipped if none present."
        ),
    )
    parser.add_argument(
        "--model",
        choices=("projection", "gnmf", "spatial_smooth", "coords"),
        default="gnmf",
        help=(
            "Factor model / baseline. 'gnmf' (default) = the trained graph-regularized "
            "NMF (model/learned.py): spatial coherence is learned via the graph "
            "Laplacian and domains are clustered on H alone. 'projection' = the "
            "non-learned baseline (the fixed random Gaussian projection encoder + NNLS "
            "loadings + [H|coords] domains). 'spatial_smooth' = the "
            "OPT-IN clean-room comparison BASELINE (baselines.py): neighbor-average "
            "the expression over the kNN graph, reduce by PCA, then k-means into "
            "domains. It emits only the domain-metric block (no factor model). "
            "'coords' = the COORDINATE-ONLY negative control (#340): k-means on the "
            "raw (x, y) coordinates alone, to bound how much domain spatial-coherence "
            "is attributable to position rather than to a factor model."
        ),
    )
    parser.add_argument(
        "--gnmf-lam",
        type=float,
        default=1.0,
        help="GNMF graph-regularization strength lam (only used when --model gnmf).",
    )
    parser.add_argument(
        "--gnmf-n-iter",
        type=int,
        default=200,
        help="GNMF maximum multiplicative-update iterations (only when --model gnmf).",
    )
    parser.add_argument(
        "--gnmf-tol",
        type=float,
        default=1e-4,
        help="GNMF relative objective-change early-stop tolerance (only when --model gnmf).",
    )
    parser.add_argument(
        "--smooth-alpha",
        type=float,
        default=0.5,
        help="Graph-smoothing blend weight per hop in [0,1] (only when --model spatial_smooth).",
    )
    parser.add_argument(
        "--smooth-hops",
        type=int,
        default=2,
        help=(
            "Number of graph-smoothing hops; 0 = plain PCA (only when "
            "--model spatial_smooth)."
        ),
    )
    parser.add_argument(
        "--smooth-n-components",
        type=int,
        default=16,
        help="PCA dimensionality before k-means (only when --model spatial_smooth).",
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


def _preprocess(adata, *, already_normalized: bool, n_hvg: int, target_sum: float = 1e4) -> dict:
    """Apply the standard ST preprocessing in place and return a manifest dict.

    Closes #197. The previous path applied only a heuristic ``log1p`` and never
    library-size normalized, so per-spot total-count variance dominated the
    encoder input (the encoder column-mean-centers but assumes mean-zero-ish
    continuous values). For count-like input we now run the standard
    ``sc.pp.normalize_total`` (library-size correction to ``target_sum``)
    BEFORE ``sc.pp.log1p``, optionally followed by highly-variable-gene
    selection. The applied steps are recorded in the returned dict so the run
    manifest is unambiguous about what normalization happened.

    Returns a dict with keys ``applied`` (bool), ``method`` (str),
    ``target_sum`` (only when normalization ran), ``hvg_applied`` (bool) and
    ``n_genes_used`` (int).
    """
    # scanpy is a heavy opt-in dependency (`.[real-data]`/`.[data]`); import it
    # lazily here so merely importing this script — e.g. test_metrics.py loading
    # `_coherence_null` via spec_from_file_location — does not require scanpy in
    # the base/`.[test]` CI env (matches the lazy-matplotlib convention; #340).
    import scanpy as sc

    if already_normalized:
        norm = {"applied": False, "method": "none"}
    elif _looks_like_raw_counts(adata.X):
        # Library-size normalize first, THEN log1p (standard ST pipeline).
        sc.pp.normalize_total(adata, target_sum=target_sum)
        sc.pp.log1p(adata)
        norm = {
            "applied": True,
            "method": "normalize_total+log1p",
            "target_sum": float(target_sum),
        }
    else:
        # Already continuous / non-count-like: do not touch values.
        norm = {"applied": False, "method": "none"}

    hvg_applied = False
    if n_hvg and n_hvg > 0 and adata.n_vars > n_hvg:
        # HVG selection is gated behind --n-hvg (default off). Subset to the
        # top-dispersion genes so the encoder sees informative columns only.
        sc.pp.highly_variable_genes(adata, n_top_genes=int(n_hvg))
        adata._inplace_subset_var(adata.var["highly_variable"].to_numpy())
        hvg_applied = True
    norm["hvg_applied"] = hvg_applied
    norm["n_genes_used"] = int(adata.n_vars)
    return norm


#: Candidate ``adata.obs`` columns that may carry per-spot ground-truth spatial
#: domain labels, in priority order. The Maynard 2021 spatialLIBD DLPFC prep
#: writes ``ground_truth_domain`` (from the spatialLIBD ``layer_guess_*`` field);
#: ``layer_guess`` is the canonical upstream name. Auto-detection binds to the
#: first present column unless ``--gt-obs-key`` is given explicitly.
_GT_OBS_KEY_CANDIDATES = (
    "layer_guess",
    "layer_guess_reordered_short",
    "ground_truth_domain",
    "ground_truth",
    "spatial_domain",
)

#: Tokens treated as "no label" and excluded from ARI on BOTH partitions.
_GT_NA_TOKENS = frozenset({"", "na", "nan", "none", "unknown", "unlabeled", "unlabelled"})


def _resolve_gt_key(adata, gt_obs_key: str | None) -> str | None:
    """Return the obs column holding GT domain labels, or ``None`` if absent.

    An explicit ``--gt-obs-key`` that is missing fails loudly (no silent
    fallback); auto-detection scans ``_GT_OBS_KEY_CANDIDATES`` in order.
    """
    if gt_obs_key:
        if gt_obs_key not in adata.obs.columns:
            raise KeyError(
                f"--gt-obs-key={gt_obs_key!r} not found in adata.obs "
                f"(available: {list(adata.obs.columns)})"
            )
        return gt_obs_key
    for key in _GT_OBS_KEY_CANDIDATES:
        if key in adata.obs.columns:
            return key
    return None


def _compute_gt_ari(
    adata, domain_id: np.ndarray, gt_obs_key: str | None
) -> tuple[float | None, str | None, int]:
    """Adjusted Rand Index of ``domain_id`` vs per-spot GT labels (#179).

    Guard: ARI is computed ONLY when a usable GT obs column is present with at
    least two distinct non-NA labels across at least two spots. Spots whose GT
    label is NA/blank are dropped from BOTH partitions so we never fabricate
    labels or join across donors. Returns ``(ari, key_used, n_labeled)``; the
    ARI is ``None`` (skipped) when no usable GT is found.
    """
    key = _resolve_gt_key(adata, gt_obs_key)
    if key is None:
        return None, None, 0
    raw = adata.obs[key].astype(str).to_numpy()
    valid = np.array([s.strip().lower() not in _GT_NA_TOKENS for s in raw], dtype=bool)
    n_labeled = int(valid.sum())
    if n_labeled < 2:
        return None, key, n_labeled
    gt = raw[valid]
    pred = np.asarray(domain_id)[valid].astype(np.int64)
    _, gt_codes = np.unique(gt, return_inverse=True)
    if np.unique(gt_codes).size < 2:
        # Only one GT class present among labeled spots -> ARI not evaluable.
        return None, key, n_labeled
    ari = adjusted_rand_index(gt_codes.astype(np.int64), pred)
    return (float(ari) if np.isfinite(ari) else None), key, n_labeled


def _compute_gt_domain_metrics(
    adata, domain_id: np.ndarray, embedding: np.ndarray, edges: np.ndarray, gt_obs_key: str | None
) -> dict[str, float] | None:
    """Full supervised domain-quality suite vs per-spot GT labels.

    Completes the supervised parity set alongside the ARI of :func:`_compute_gt_ari`:
    NMI, weighted Dice and boundary precision/recall/F1 compare the recovered
    ``domain_id`` against the GT labels, while silhouette and Calinski-Harabasz
    score the *predicted* domains' compactness in the recovered ``embedding``
    (the factor scores ``H`` for the model paths, or the baseline's PCA scores).
    All are computed ONLY when a usable GT obs column is present with at least
    two distinct non-NA labels; spots whose GT label is NA/blank are dropped from
    every computation (no fabrication / no cross-donor join), and the spatial
    graph is restricted to the labeled subset (edges remapped to compact indices)
    so the boundary set is well defined. Returns ``None`` when no usable GT is
    found; otherwise a ``str -> float`` dict (non-finite values are coerced to
    JSON ``null`` by the results contract).
    """
    key = _resolve_gt_key(adata, gt_obs_key)
    if key is None:
        return None
    raw = adata.obs[key].astype(str).to_numpy()
    valid = np.array([s.strip().lower() not in _GT_NA_TOKENS for s in raw], dtype=bool)
    if int(valid.sum()) < 2:
        return None
    _, gt_codes = np.unique(raw[valid], return_inverse=True)
    if np.unique(gt_codes).size < 2:
        return None
    gt_codes = gt_codes.astype(np.int64)
    pred = np.asarray(domain_id)[valid].astype(np.int64)

    # Restrict the spatial graph to labeled spots and remap to compact indices
    # so boundary spots are defined purely among spots that carry a GT label.
    n_all = np.asarray(domain_id).shape[0]
    remap = np.full(n_all, -1, dtype=np.int64)
    remap[np.flatnonzero(valid)] = np.arange(int(valid.sum()), dtype=np.int64)
    src, dst = edges
    keep = valid[src] & valid[dst]
    sub_edges = np.stack([remap[src[keep]], remap[dst[keep]]]).astype(np.int64)

    embedding = np.asarray(embedding, dtype=np.float64)[valid]

    return {
        "nmi_domain": normalized_mutual_information(gt_codes, pred),
        "weighted_dice_domain": weighted_dice(gt_codes, pred),
        "boundary_precision_domain": boundary_precision(gt_codes, pred, sub_edges),
        "boundary_recall_domain": boundary_recall(gt_codes, pred, sub_edges),
        "boundary_f1_domain": boundary_f1(gt_codes, pred, sub_edges),
        # silhouette / Calinski-Harabasz score the predicted clustering's
        # compactness in the recovered embedding (intrinsic domain quality).
        "silhouette_domain": silhouette(embedding, pred),
        "calinski_harabasz_domain": calinski_harabasz(embedding, pred),
    }


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


def _coherence_null(domain_id: np.ndarray, edges: np.ndarray, n_shuffles: int, seed: int) -> float:
    """Mean label-invariant cluster coherence over spatially-shuffled labels.

    The chance baseline for :func:`label_invariant_cluster_coherence`: permuting
    which spot carries which label destroys spatial structure, so the mean
    coherence over shuffles is the no-structure reference. The reported
    ``coherence_label_invariant_domain`` should be read as its delta over this
    null, exactly as ``morans_i_domain`` is read against ``morans_i_domain_null``.
    """
    rng = np.random.default_rng(seed)
    base = np.asarray(domain_id)
    vals = [
        label_invariant_cluster_coherence(rng.permutation(base), edges)
        for _ in range(max(int(n_shuffles), 1))
    ]
    return float(np.mean(vals))


def _coords_domains(coords: np.ndarray, n_domains: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Coordinate-only negative-control domains: k-means on z-normalized (x, y).

    This is the explicit ablation for #340: it clusters on *nothing but the raw
    spatial coordinates*. Any spatial-coherence metric (Moran's I or
    label-invariant coherence) on these domains is high purely because the labels
    ARE the coordinates — so it bounds how much "spatial structure" is attributable
    to position alone rather than to a learned factor model. Returns
    ``(domain_id, embedding)`` with the normalized coords as the embedding so the
    intrinsic domain-quality suite scores it on the same footing as the models.
    """
    xy = np.asarray(coords, dtype=np.float64)
    std = xy.std(axis=0)
    std[std == 0.0] = 1.0
    embedding = ((xy - xy.mean(axis=0)) / std).astype(np.float32)
    n = embedding.shape[0]
    k = int(min(max(n_domains, 1), max(n, 1)))
    if n == 0 or k <= 1:
        return np.zeros(n, dtype=np.int64), embedding
    domain_id = _kmeans(embedding.astype(np.float64), k, seed, n_init=4, max_iter=100)
    return domain_id.astype(np.int64), embedding


def main() -> None:
    args = _parse_args()
    h5ad_path = args.h5ad.resolve()

    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    t0 = time.perf_counter()

    # --- 1. Load --------------------------------------------------------------
    import scanpy as sc  # lazy: heavy opt-in dep; keep module import scanpy-free

    adata = sc.read_h5ad(h5ad_path)
    n_obs, n_vars = int(adata.shape[0]), int(adata.shape[1])

    # --- 2. Normalize (normalize_total -> log1p) + optional HVG --------------
    normalization = _preprocess(
        adata,
        already_normalized=args.already_normalized,
        n_hvg=args.n_hvg,
        target_sum=args.target_sum,
    )
    n_vars = int(adata.shape[1])  # may shrink if HVG selection ran

    # --- 3. Densify + section + edges ----------------------------------------
    X = adata.X
    X = np.asarray(X.todense(), dtype=np.float32) if sp.issparse(X) else np.asarray(X, dtype=np.float32)
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    section_id = np.zeros(n_obs, dtype=np.int64)
    edges = _build_knn_edges(coords, args.knn)

    # --- 4. Validate + fit ----------------------------------------------------
    validate_inputs(X, coords, section_id, edges)
    gnmf_result = None
    baseline_out = None
    if args.model == "spatial_smooth":
        # OPT-IN clean-room comparison BASELINE (baselines.py). NOT a factor
        # model: it neighbor-averages the (normalized) expression over the kNN
        # graph, reduces by PCA, then k-means into domains. ``out`` is left None
        # so the factor-model-specific blocks (shared/private separation, factor
        # stats, factors.npz) are skipped; only the domain-metric block is
        # emitted, scored by the SAME eval suite and behind the SAME GT guard as
        # the models, so the comparison is apples-to-apples.
        baseline_out = spatial_smooth_domains(
            X,
            edges,
            n_domains=args.n_domains,
            n_components=args.smooth_n_components,
            alpha=args.smooth_alpha,
            n_hops=args.smooth_hops,
            seed=args.seed,
        )
        out = None
        domain_id = baseline_out.domain_id
        embedding = baseline_out.embedding
    elif args.model == "gnmf":
        # OPT-IN trained model: graph-regularized NMF (numpy-only). Spatial
        # coherence is LEARNED via the Laplacian regularizer on `edges`, not by
        # baking coords into the features, and domains are clustered on the
        # factor scores H ALONE -- so morans_i_domain is no longer high by
        # construction. We still emit the Moran's permutation null for honesty.
        out, gnmf_result = fit_transform_gnmf(
            X,
            edges,
            K_shared=args.k_shared,
            K_private=args.k_private,
            n_domains=args.n_domains,
            lam=args.gnmf_lam,
            n_iter=args.gnmf_n_iter,
            tol=args.gnmf_tol,
            seed=args.seed,
        )
        domain_id = out.domain_id
        embedding = out.H
    elif args.model == "coords":
        # COORDINATE-ONLY negative control (#340): cluster on the raw (x, y)
        # coordinates alone. ``out`` is left None (no factor structure), so only
        # the domain-metric block is emitted, scored by the SAME eval suite and
        # behind the SAME GT guard as the models. A high morans_i_domain /
        # coherence here is the by-construction ceiling that the factor models
        # must be read against.
        out = None
        domain_id, embedding = _coords_domains(coords, args.n_domains, args.seed)
    else:
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
        domain_id = out.domain_id
        embedding = out.H

    # --- 5. Intrinsic metrics (+ supervised ARI iff GT labels present, #179) -
    morans_real = morans_i(domain_id.astype(np.float64), edges)
    morans_null = _morans_null(domain_id, edges, args.n_null_shuffles, args.seed)
    # Relabel-invariant headline (#340): morans_i on integer label CODES is
    # relabel-sensitive (its own docstring warns to use the label-invariant form
    # for categorical labels). label_invariant_cluster_coherence averages Moran's
    # I over per-class one-hot indicators, so it is invariant to which integer is
    # assigned to which domain. Reported with its own spatial-shuffle null.
    coherence_real = label_invariant_cluster_coherence(domain_id, edges)
    coherence_null = _coherence_null(domain_id, edges, args.n_null_shuffles, args.seed)

    # Supervised domain ARI vs per-spot GT labels (e.g. Maynard DLPFC layer
    # labels). Computed ONLY when a usable GT obs column is present; skipped
    # (no fabrication / no wrong-donor join) for label-less datasets like the
    # active Br2719 section, whose obs carries no domain annotation.
    ari_domain, gt_key_used, n_gt_labeled = _compute_gt_ari(
        adata, domain_id, args.gt_obs_key
    )
    # Full supervised domain-quality suite (NMI / weighted Dice / boundary
    # P-R-F1 / silhouette / Calinski-Harabasz), behind the SAME GT-label guard
    # as the ARI above: emitted only when usable per-spot GT labels are present.
    gt_domain_metrics = _compute_gt_domain_metrics(
        adata, domain_id, embedding, edges, args.gt_obs_key
    )

    # DEGENERACY GUARD: the decoder assigns domains by graph-aware clustering
    # that distributes the domain budget across connected components and runs a
    # per-spot k-means WITHIN each component (decoder.py:_cluster_components).
    # A single-section kNN graph over one contiguous slide is one connected
    # component, but it is now sub-partitioned into multiple domains (#183), so
    # ``domain_id`` should no longer collapse to a single label. We keep this
    # guard as a safety check: if some pathological input still yields a single
    # domain, every domain-derived metric (Moran's I) is trivially 0, so we
    # detect and record it explicitly rather than report the zeros as signal.
    n_domains_found = int(np.unique(domain_id).size)
    domain_degenerate = n_domains_found <= 1

    if out is not None:
        # Factor-model paths (projection / gnmf): emit the shared/private factor
        # separation alongside the domain block. Built as the original literal so
        # the projection/gnmf metric set stays byte-for-byte unchanged.
        sep = shared_private_separation(out.Z_shared, out.Z_private, section_id)
        metrics: dict[str, float] = {
            "morans_i_domain": morans_real,
            "morans_i_domain_null": morans_null,
            "morans_i_domain_delta": morans_real - morans_null,
            "coherence_label_invariant_domain": coherence_real,
            "coherence_label_invariant_domain_null": coherence_null,
            "coherence_label_invariant_domain_delta": coherence_real - coherence_null,
            "domain_clustering_degenerate": float(domain_degenerate),
            "shared_mean_active_sections": sep["shared_mean_active_sections"],
            "private_mean_active_sections": sep["private_mean_active_sections"],
            "n_edges": float(edges.shape[1]),
            "ari_vs_gt_available": float(ari_domain is not None),
            "domain_metric_suite_available": float(gt_domain_metrics is not None),
        }
    else:
        # spatial_smooth BASELINE: domain-metric block only (no factor structure,
        # so no shared/private separation or factor stats are meaningful).
        metrics = {
            "morans_i_domain": morans_real,
            "morans_i_domain_null": morans_null,
            "morans_i_domain_delta": morans_real - morans_null,
            "coherence_label_invariant_domain": coherence_real,
            "coherence_label_invariant_domain_null": coherence_null,
            "coherence_label_invariant_domain_delta": coherence_real - coherence_null,
            "domain_clustering_degenerate": float(domain_degenerate),
            "n_edges": float(edges.shape[1]),
            "ari_vs_gt_available": float(ari_domain is not None),
            "domain_metric_suite_available": float(gt_domain_metrics is not None),
        }
    # ``ari_domain`` is recorded ONLY when GT labels were present and evaluable,
    # so a label-less run never emits a misleading ARI value of any kind.
    if ari_domain is not None:
        metrics["ari_domain"] = ari_domain
    # The supervised domain-quality suite is emitted as a block when GT labels
    # are present (non-finite entries become JSON null via the results contract).
    if gt_domain_metrics is not None:
        metrics.update(gt_domain_metrics)
    # Factor stats are factor-model-only (skipped for the baseline).
    if out is not None:
        metrics.update(_factor_stats(out))
    # Learning evidence is emitted ONLY on the trained path so the projection
    # run's metrics stay byte-for-byte unchanged.
    if gnmf_result is not None:
        metrics["gnmf_objective_initial"] = float(gnmf_result.objective[0])
        metrics["gnmf_objective_final"] = float(gnmf_result.objective[-1])
        metrics["gnmf_n_iter_run"] = float(gnmf_result.n_iter_run)
        metrics["gnmf_lam"] = float(args.gnmf_lam)

    runtime_s = time.perf_counter() - t0

    # --- 6. Save native outputs ----------------------------------------------
    if args.results_dir is not None:
        results_root = args.results_dir.resolve()
    else:
        results_root = Path(__file__).resolve().parents[1] / "results"
    outputs_dir = results_root / "factorgraph-st" / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    if out is not None:
        output_npz = outputs_dir / "factors.npz"
        np.savez_compressed(
            output_npz,
            H=out.H,
            W=out.W,
            Z_shared=out.Z_shared,
            Z_private=out.Z_private,
            domain_id=out.domain_id,
        )
        outputs_payload = {"factors": str(output_npz)}
    elif baseline_out is not None:
        # spatial_smooth baseline: persist the PCA embedding + domain labels.
        output_npz = outputs_dir / "baseline_spatial_smooth.npz"
        np.savez_compressed(
            output_npz,
            embedding=baseline_out.embedding,
            domain_id=baseline_out.domain_id,
        )
        outputs_payload = {"baseline_spatial_smooth": str(output_npz)}
    else:
        # coords negative control (#340): persist the normalized-coordinate
        # embedding + the coordinate-only domain labels (no factor matrices).
        output_npz = outputs_dir / "baseline_coords.npz"
        np.savez_compressed(output_npz, embedding=embedding, domain_id=domain_id)
        outputs_payload = {"baseline_coords": str(output_npz)}

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
            "domain clustering unexpectedly collapsed to a single label despite the "
            "intra-component partition (decoder.py:_cluster_components, #183). With one "
            "domain morans_i_domain is trivially 0; this is recorded so the metric is "
            "not mistaken for a computed spatial signal. Inspect the input graph/budget."
            if domain_degenerate
            else "domain clustering produced multiple domains"
        ),
    }
    if args.model == "gnmf" and gnmf_result is not None:
        # OPT-IN trained path: replace the projection caveat with the honest
        # learned-model description. The morans-by-construction artifact is gone
        # because coordinates are never a clustering input here.
        interpretability = {
            "model_is_learned": True,
            "encoder": (
                "trained graph-regularized NMF (model/learned.py): multiplicative "
                "updates minimizing ||X - H W^T||_F^2 + lam*Tr(H^T L H) over H,W>=0, "
                "L = D - A the unnormalized Laplacian of the spatial kNN graph"
            ),
            "factor_basis": "learned nonnegative factor scores H (gauge-normalized loadings)",
            "loadings": "learned nonnegative gene loadings W from the GNMF fit",
            "domain_assignment": (
                "k-means on the learned factor scores H ALONE (no coords, no graph); "
                "spatial coherence is learned via the Laplacian regularizer, not baked "
                "into the features"
            ),
            "caveat": (
                "morans_i_domain is NOT high by construction here: coordinates are never "
                "an encoder feature nor a clustering input, so domain_id cannot re-read "
                "out the coordinates. It is still reported against morans_i_domain_null "
                "for honesty."
            ),
            "lam": float(args.gnmf_lam),
            "n_iter_run": int(gnmf_result.n_iter_run),
            "objective_initial": float(gnmf_result.objective[0]),
            "objective_final": float(gnmf_result.objective[-1]),
            "domain_clustering_degenerate": bool(domain_degenerate),
            "domain_count_found": n_domains_found,
            "domain_degeneracy_reason": (
                "domain clustering collapsed to a single label; with one domain "
                "morans_i_domain is trivially 0. Inspect n_domains / the learned H."
                if domain_degenerate
                else "domain clustering produced multiple domains"
            ),
        }
    elif args.model == "spatial_smooth":
        # OPT-IN clean-room comparison BASELINE: replace the factor-model
        # description with the honest graph-smoothing pipeline description.
        interpretability = {
            "model_is_learned": False,
            "encoder": (
                "clean-room spatial-smoothing baseline (baselines.py): neighbor-average "
                "the normalized expression over the symmetrized kNN graph for "
                f"n_hops={args.smooth_hops} blended steps (alpha={args.smooth_alpha}), "
                "no training/backprop"
            ),
            "factor_basis": (
                f"numpy PCA (SVD) of the smoothed expression to {args.smooth_n_components} "
                "components; no nonnegative factor model"
            ),
            "loadings": "not applicable (no factor model; baseline has no gene loadings)",
            "domain_assignment": (
                "k-means on the PCA embedding of the graph-smoothed expression "
                "(baselines.py:spatial_smooth_domains); coordinates are NOT a clustering "
                "input -- the only spatial signal is the neighbor averaging"
            ),
            "caveat": (
                "comparison baseline only: graph smoothing injects spatial coherence via "
                "neighbor averaging, so morans_i_domain reflects that smoothing prior and "
                "should be read as the delta over morans_i_domain_null. With n_hops=0 this "
                "reduces to a plain PCA + k-means ablation (no spatial smoothing)."
            ),
            "smooth_alpha": float(args.smooth_alpha),
            "smooth_hops": int(args.smooth_hops),
            "smooth_n_components": int(args.smooth_n_components),
            "domain_clustering_degenerate": bool(domain_degenerate),
            "domain_count_found": n_domains_found,
            "domain_degeneracy_reason": (
                "baseline clustering collapsed to a single label; with one domain "
                "morans_i_domain is trivially 0. Inspect n_domains / the kNN graph."
                if domain_degenerate
                else "baseline clustering produced multiple domains"
            ),
        }
    elif args.model == "coords":
        # COORDINATE-ONLY negative control (#340): domains are k-means on the raw
        # (x, y) coordinates, so every spatial-coherence metric is high BY
        # CONSTRUCTION. This run exists to bound that ceiling, not to claim quality.
        interpretability = {
            "model_is_learned": False,
            "encoder": (
                "coordinate-only negative control (#340): no expression model at all; "
                "domains are k-means on z-normalized (x, y) coordinates"
            ),
            "factor_basis": "not applicable (no factor model; embedding is the normalized coords)",
            "loadings": "not applicable (no factor model; control has no gene loadings)",
            "domain_assignment": (
                "k-means on the raw (x, y) coordinates ALONE; coordinates are the only "
                "input, so domain_id is a direct partition of space"
            ),
            "caveat": (
                "negative control: morans_i_domain AND coherence_label_invariant_domain are "
                "high purely because the labels ARE the coordinates. Read this as the "
                "by-construction ceiling that any factor model must be compared against; "
                "it is never evidence of a learned spatial signal."
            ),
            "domain_clustering_degenerate": bool(domain_degenerate),
            "domain_count_found": n_domains_found,
            "domain_degeneracy_reason": (
                "coordinate k-means collapsed to a single label; with one domain "
                "morans_i_domain is trivially 0. Inspect n_domains / the coordinates."
                if domain_degenerate
                else "coordinate clustering produced multiple domains"
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
        "ground_truth_ari": {
            "available": ari_domain is not None,
            "obs_key_used": gt_key_used,
            "n_labeled_spots": n_gt_labeled,
            "ari_domain": ari_domain,
            "note": (
                "Supervised ARI of recovered domains vs per-spot GT labels (#179). "
                "Computed only when a usable GT obs column is present; NA/blank "
                "spots are dropped from both partitions. No labels are fabricated "
                "and no cross-donor join is performed."
            ),
        },
        "notes": (
            "Single-section run (section_id all zeros); shared_private_separation "
            "degenerates (single_section=True). FactorGraph-ST MVP is a fixed "
            "random projection + NNLS loadings + graph-aware k-means domains, not a "
            "learned model -- see interpretability block."
        ),
    }
    if args.model == "gnmf":
        # Override only on the trained path; the projection run_metadata above is
        # left byte-for-byte unchanged.
        run_metadata["model"] = "gnmf"
        run_metadata["notes"] = (
            "OPT-IN trained graph-regularized NMF (model/learned.py): X ~= H W^T "
            "minimized with a graph-Laplacian smoothness penalty (lam="
            f"{args.gnmf_lam}); spatial coherence is LEARNED via the graph, not baked "
            "into features, and domains are clustered on the factor scores H alone. "
            "Single-section run (section_id all zeros); shared_private_separation "
            "degenerates (single_section=True)."
        )
    elif args.model == "spatial_smooth":
        # Override only on the baseline path; the projection run_metadata above is
        # left byte-for-byte unchanged.
        run_metadata["model"] = "spatial_smooth"
        run_metadata["notes"] = (
            "OPT-IN clean-room comparison BASELINE (baselines.py): graph-smoothing "
            f"(alpha={args.smooth_alpha}, n_hops={args.smooth_hops}) of the normalized "
            f"expression -> PCA ({args.smooth_n_components} components) -> k-means "
            "domains. Not a factor model; only the domain-metric block is emitted "
            "(no shared/private separation or factor stats), scored by the SAME eval "
            "suite as the models for an apples-to-apples comparison. n_hops=0 reduces "
            "to a plain PCA + k-means ablation."
        )

    paths = results_contract.write_results(
        project="factorgraph-st",
        dataset_card_id=results_contract.dataset_card_id([str(h5ad_path)]),
        metrics=metrics,
        outputs=outputs_payload,
        run_metadata=run_metadata,
        results_dir=results_root,
    )

    print(f"Wrote metrics:      {paths['metrics']}")
    print(f"Wrote run_metadata: {paths['run_metadata']}")
    print(f"Wrote outputs:      {output_npz}")
    print(f"morans_i_domain={morans_real:.6f} null={morans_null:.6f} "
          f"delta={morans_real - morans_null:.6f} runtime_s={runtime_s:.2f}")
    print(f"coherence_label_invariant_domain={coherence_real:.6f} "
          f"null={coherence_null:.6f} delta={coherence_real - coherence_null:.6f}")
    if gnmf_result is not None:
        print(f"model=gnmf lam={args.gnmf_lam} objective "
              f"{gnmf_result.objective[0]:.4f} -> {gnmf_result.objective[-1]:.4f} "
              f"over {gnmf_result.n_iter_run} iters")
    if baseline_out is not None:
        print(f"model=spatial_smooth alpha={args.smooth_alpha} hops={args.smooth_hops} "
              f"n_components={args.smooth_n_components} domains_found={n_domains_found}")
    if ari_domain is not None:
        print(f"ari_domain={ari_domain:.6f} (vs obs[{gt_key_used!r}], "
              f"{n_gt_labeled} labeled spots)")
    else:
        print("ari_domain=SKIPPED (no usable ground-truth domain labels in obs)")


if __name__ == "__main__":
    main()
