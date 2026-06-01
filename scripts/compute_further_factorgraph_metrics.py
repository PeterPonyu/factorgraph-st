#!/usr/bin/env python
"""Further reference-grounded computational results for FactorGraph-ST.

Loads the real DLPFC Visium dataset (``anndata.h5ad``) plus the previously
emitted MVP factor outputs (``results/factorgraph-st/outputs/factors.npz``) and
APPENDS four families of further metrics to ``results/factorgraph-st/metrics.json``
(and a ``further_results`` provenance block to ``run_metadata.json``) via the
vendored :mod:`factorgraph_st.results_contract`. The existing run-path metrics and
ALL pre-existing honesty caveats are preserved -- this script merges, never
discards.

SCIENTIFIC-HONESTY NOTE (see consensus plan G4): the FactorGraph-ST "model" is
NOT a learned factor model. The encoder is a fixed random Gaussian projection of
neighbor-aggregated features (``encoder.py``: a single seeded ``rng.normal``
projection, no training/backprop), loadings are a numpy-only NNLS fit, and
spatial coordinates + neighbor means are baked into the projected features
(``encoder.py``). ``domain_id`` comes from a deterministic GRAPH-AWARE k-means
pass on the joint ``[H | coords]`` feature (``decoder.py:_cluster_domains``) --
coordinates are an explicit clustering input. Therefore EVERY ``domain_id``-
derived metric is HIGH BY CONSTRUCTION (it partly re-reads out the coordinates
fed in) and carries ``model_is_learned: false`` plus a paired permutation null.
Only the gene-level Moran's I family is model-INDEPENDENT (it characterizes the
REAL data and is the single cleanest honest win).

The four families:

  1. Gene-level Moran's I (MODEL-INDEPENDENT, HONEST): distribution over the
     top-HVG genes on the kNN spatial graph + top-N spatially-variable genes.
  2. PAS + CHAOS of ``domain_id`` WITH permutation nulls (MVP-characterization;
     interpretability caveat retained; reported as delta over null).
  3. spatialLIBD cortical-layer GT fetch + ARI/NMI floor. The donor here is
     Br2719 (GSE307403, capture V12F14-053_A1). We ATTEMPT to fetch the LIBD
     HumanPilot per-barcode manual layer map and join on the spot barcodes, but
     a donor-safety guard refuses any join whose barcode overlap is merely the
     shared Visium whitelist (i.e. the labels are from a DIFFERENT donor). If
     the join is unsafe / the source is unreachable, ``gt_fetch`` is recorded as
     ``"blocked: <reason>"`` and ARI/NMI are SKIPPED cleanly -- GT is NEVER
     fabricated. If a genuine same-donor join succeeds, ARI/NMI are reported as
     a BASELINE FLOOR (random-projection MVP approx chance), with the GT source
     URL and ``model_is_learned: false``.
  4. NMF-contrast: sklearn NMF (k = factorgraph K_shared) on the same X +
     ``matched_factor_correlation`` between the MVP Z_shared and the NMF factors,
     quantifying the gap to a real factor model.

Usage (under the project conda env)::

    conda run --no-capture-output -n dl python \
        scripts/compute_further_factorgraph_metrics.py \
        --h5ad ../data/processed/dlpfc_GSE307403_Br2719/anndata.h5ad
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import scanpy as sc
import scipy.sparse as sp
from scipy.spatial import cKDTree

from factorgraph_st import results_contract
from factorgraph_st.eval.metrics import matched_factor_correlation, morans_i

# ---- paths --------------------------------------------------------------- #
_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_H5AD = (
    _REPO_ROOT.parents[0]
    / "data"
    / "processed"
    / "dlpfc_GSE307403_Br2719"
    / "anndata.h5ad"
)
_DEFAULT_FACTORS = _REPO_ROOT / "results" / "factorgraph-st" / "outputs" / "factors.npz"

#: Raw-count heuristic threshold for the conditional log1p guard (mirrors the
#: run-path runner so the gene-level Moran's I is computed on the same scale).
_RAW_COUNT_MAX_THRESHOLD = 50.0

#: LIBD HumanPilot per-barcode manual cortical-layer map (Maynard 2021). Columns:
#: barcode, sample_id, layer. This is the ONLY barcode-joinable per-spot manual
#: layer GT the LIBD project publishes as a committed flat file. NOTE: its donors
#: are Br5292/Br5595/Br8100 (samples 151507-151676), NOT Br2719 -- the donor
#: guard below uses this to refuse a spurious whitelist-collision join.
_HUMANPILOT_LAYER_MAP_URL = (
    "https://raw.githubusercontent.com/LieberInstitute/HumanPilot/master/"
    "10X/barcode_level_layer_map.tsv"
)

#: Minimum fraction of OUR spots that must carry a SAME-DONOR label for the join
#: to be considered genuine rather than a shared-Visium-whitelist collision.
_GT_MIN_COVERAGE = 0.5


# --------------------------------------------------------------------------- #
# args
# --------------------------------------------------------------------------- #
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5ad", type=Path, default=_DEFAULT_H5AD)
    parser.add_argument("--factors", type=Path, default=_DEFAULT_FACTORS)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--knn", type=int, default=6, help="k for kNN spatial edges.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--n-hvg",
        type=int,
        default=2000,
        help="Number of highly-variable genes for the gene-level Moran's I scan.",
    )
    parser.add_argument(
        "--top-n-svg", type=int, default=20, help="Number of top SVGs to report."
    )
    parser.add_argument(
        "--n-null-shuffles",
        type=int,
        default=20,
        help="Permutation shuffles for the PAS/CHAOS nulls (>=10).",
    )
    parser.add_argument(
        "--no-network",
        action="store_true",
        help="Skip the GT fetch entirely (records gt_fetch=blocked: network disabled).",
    )
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# shared helpers (runner-local; no model source touched)
# --------------------------------------------------------------------------- #
def _looks_like_raw_counts(X) -> bool:
    x_max = float(X.max()) if (sp.issparse(X) or X.size) else 0.0
    if x_max > _RAW_COUNT_MAX_THRESHOLD:
        return True
    sample = X[: min(X.shape[0], 100)]
    sample = sample.toarray() if sp.issparse(sample) else np.asarray(sample)
    return bool(np.allclose(sample, np.round(sample)))


def _build_knn_edges(coords: np.ndarray, k: int) -> np.ndarray:
    """Undirected, de-duplicated kNN spatial edges as ``(2, n_edges)`` int64."""
    n = coords.shape[0]
    k_eff = min(k, max(n - 1, 0))
    if n == 0 or k_eff == 0:
        return np.empty((2, 0), dtype=np.int64)
    tree = cKDTree(coords)
    _, idx = tree.query(coords, k=k_eff + 1)
    idx = np.atleast_2d(idx)
    src = np.repeat(np.arange(n, dtype=np.int64), k_eff)
    dst = idx[:, 1:].reshape(-1).astype(np.int64)
    a = np.concatenate([src, dst])
    b = np.concatenate([dst, src])
    pairs = np.unique(np.stack([a, b], axis=1), axis=0)
    return pairs.T.astype(np.int64)


def _knn_index(coords: np.ndarray, k: int) -> np.ndarray:
    """Per-spot neighbor index array ``(n, k)`` (self excluded)."""
    n = coords.shape[0]
    k_eff = min(k, max(n - 1, 0))
    if n == 0 or k_eff == 0:
        return np.empty((n, 0), dtype=np.int64)
    tree = cKDTree(coords)
    _, idx = tree.query(coords, k=k_eff + 1)
    idx = np.atleast_2d(idx)
    return idx[:, 1:].astype(np.int64)


# --------------------------------------------------------------------------- #
# 1. gene-level Moran's I (MODEL-INDEPENDENT, HONEST)
# --------------------------------------------------------------------------- #
def _gene_level_morans(
    X: np.ndarray, var_names: np.ndarray, edges: np.ndarray, n_hvg: int, top_n: int
) -> tuple[dict[str, float], list[dict]]:
    """Distribution of per-gene Moran's I over the top-``n_hvg`` HVGs.

    Characterizes the REAL spatial structure of the data; INDEPENDENT of the MVP
    model. Returns a flat metric dict plus a ranked top-SVG table for the
    provenance block.
    """
    # Variance-rank HVG selection on the (log) expression so the scan is bounded
    # and dominated by genes with signal (matches the spirit of seurat HVGs
    # without pulling in a heavyweight dependency).
    variances = X.var(axis=0)
    n_keep = int(min(n_hvg, X.shape[1]))
    hvg_idx = np.argsort(variances)[::-1][:n_keep]

    vals = np.empty(n_keep, dtype=np.float64)
    for i, g in enumerate(hvg_idx):
        vals[i] = morans_i(X[:, g].astype(np.float64), edges)

    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        finite = np.array([0.0])

    order = np.argsort(vals)[::-1]
    top = []
    for rank, pos in enumerate(order[: int(top_n)], start=1):
        top.append(
            {
                "rank": rank,
                "gene": str(var_names[hvg_idx[pos]]),
                "morans_i": float(vals[pos]),
            }
        )

    metrics = {
        "gene_morans_i_mean": float(np.mean(finite)),
        "gene_morans_i_median": float(np.median(finite)),
        "gene_morans_i_std": float(np.std(finite)),
        "gene_morans_i_max": float(np.max(finite)),
        "gene_morans_i_p90": float(np.percentile(finite, 90)),
        "gene_morans_i_frac_above_0p1": float(np.mean(finite > 0.1)),
        "gene_morans_i_n_genes": float(n_keep),
    }
    return metrics, top


# --------------------------------------------------------------------------- #
# 2. PAS + CHAOS of domain_id WITH permutation nulls (MVP-characterization)
# --------------------------------------------------------------------------- #
def _pas(domain_id: np.ndarray, neighbor_idx: np.ndarray) -> float:
    """Proportion of Abnormal Spots (Dong 2025): fraction of spots whose label
    differs from the MAJORITY of their spatial kNN neighbors. Lower = more
    spatially coherent domains."""
    if neighbor_idx.size == 0:
        return 0.0
    neigh_labels = domain_id[neighbor_idx]  # (n, k)
    same = (neigh_labels == domain_id[:, None]).mean(axis=1)
    return float(np.mean(same < 0.5))


def _chaos(domain_id: np.ndarray, coords: np.ndarray) -> float:
    """Spatial CHAOS (Dong 2025): mean nearest-neighbor distance computed WITHIN
    each domain, averaged over domains and normalized. Lower = more spatially
    compact domains."""
    dists = []
    for d in np.unique(domain_id):
        pts = coords[domain_id == d]
        if pts.shape[0] < 2:
            continue
        tree = cKDTree(pts)
        nn, _ = tree.query(pts, k=2)
        dists.append(float(np.mean(nn[:, 1])))
    if not dists:
        return 0.0
    # Normalize by the global nearest-neighbor scale so the value is comparable
    # across domain counts.
    gtree = cKDTree(coords)
    gnn, _ = gtree.query(coords, k=2)
    scale = float(np.mean(gnn[:, 1])) or 1.0
    return float(np.mean(dists) / scale)


def _pas_chaos_with_nulls(
    domain_id: np.ndarray,
    coords: np.ndarray,
    neighbor_idx: np.ndarray,
    n_shuffles: int,
    seed: int,
) -> dict[str, float]:
    pas_real = _pas(domain_id, neighbor_idx)
    chaos_real = _chaos(domain_id, coords)
    rng = np.random.default_rng(seed)
    pas_null, chaos_null = [], []
    for _ in range(max(int(n_shuffles), 1)):
        perm = rng.permutation(domain_id)
        pas_null.append(_pas(perm, neighbor_idx))
        chaos_null.append(_chaos(perm, coords))
    pas_null_m = float(np.mean(pas_null))
    chaos_null_m = float(np.mean(chaos_null))
    return {
        "domain_pas": pas_real,
        "domain_pas_null": pas_null_m,
        # PAS is lower-is-better, so the informative signal is null - real.
        "domain_pas_delta": pas_null_m - pas_real,
        "domain_chaos": chaos_real,
        "domain_chaos_null": chaos_null_m,
        "domain_chaos_delta": chaos_null_m - chaos_real,
    }


# --------------------------------------------------------------------------- #
# 3. spatialLIBD cortical-layer GT fetch + ARI/NMI floor (NEVER fabricate)
# --------------------------------------------------------------------------- #
def _fetch_humanpilot_layer_map(timeout: float = 60.0) -> str | None:
    """Download the LIBD HumanPilot per-barcode manual layer map (TSV text).

    Returns the raw text on success, ``None`` on any network failure. Uses the
    ambient proxy env (urllib honors ``http(s)_proxy``)."""
    try:
        req = urllib.request.Request(
            _HUMANPILOT_LAYER_MAP_URL, headers={"User-Agent": "factorgraph-st/1.0"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return resp.read().decode("utf-8")
    except (urllib.error.URLError, OSError, TimeoutError):
        return None


def _gt_floor(
    barcodes: np.ndarray,
    domain_id: np.ndarray,
    no_network: bool,
) -> tuple[dict[str, float], dict]:
    """Attempt the spatialLIBD layer-GT join and compute the ARI/NMI floor.

    The donor-safety guard is the crux of honesty: the only barcode-joinable
    LIBD layer map (HumanPilot) is for donors Br5292/Br5595/Br8100, NOT our
    Br2719. Visium reuses one barcode whitelist across slides, so a naive
    barcode intersection is NEARLY TOTAL yet corresponds to DIFFERENT tissue
    locations -- joining it fabricates GT. We therefore record ``gt_fetch`` as
    blocked unless a genuine same-donor source is wired in (none is publicly
    available for Br2719 / GSE307403 / capture V12F14-053_A1)."""
    info: dict = {
        "gt_donor": "Br2719",
        "gt_capture": "V12F14-053_A1",
        "gt_accession": "GSE307403 / GSM9223405",
        "gt_source_attempted": _HUMANPILOT_LAYER_MAP_URL,
    }

    if no_network:
        info["gt_fetch"] = "blocked: network disabled (--no-network)"
        return {}, info

    text = _fetch_humanpilot_layer_map()
    if text is None:
        info["gt_fetch"] = (
            "blocked: HumanPilot layer map unreachable "
            f"({_HUMANPILOT_LAYER_MAP_URL})"
        )
        return {}, info

    # Parse barcode -> {sample_id -> layer}. Column 1 of the HumanPilot map is the
    # Maynard 2021 sample_id (151507-151676), which belongs to donors
    # Br5292/Br5595/Br8100 -- NOT Br2719. Compute the barcode overlap to make the
    # whitelist-collision explicit in the record.
    map_barcodes: set[str] = set()
    map_sample_ids: set[str] = set()
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        map_barcodes.add(parts[0])
        map_sample_ids.add(parts[1])
    our = set(str(b) for b in barcodes)
    overlap = len(our & map_barcodes)
    coverage = overlap / max(len(our), 1)

    info["gt_source_sample_ids"] = sorted(map_sample_ids)
    info["gt_source_donors"] = ["Br5292", "Br5595", "Br8100"]
    info["gt_barcode_overlap"] = overlap
    info["gt_barcode_coverage"] = round(coverage, 4)

    # Donor-safety guard: the fetched map (Maynard 2021 samples 151507-151676)
    # is from donors Br5292/Br5595/Br8100, NOT Br2719. Any barcode overlap is the
    # shared Visium whitelist, NOT a real per-spot match. Refuse -> never fabricate.
    donor_match = "Br2719" in map_sample_ids  # by construction False (sample ids only)
    if not donor_match:
        info["gt_fetch"] = (
            "blocked: only barcode-joinable LIBD layer map is HumanPilot "
            "(Maynard 2021 samples 151507-151676, donors Br5292/Br5595/Br8100), "
            "which does NOT contain donor Br2719. The high barcode overlap "
            f"({overlap}/{len(our)} = {coverage:.1%}) is the shared 10x Visium "
            "barcode whitelist across slides, NOT a per-spot correspondence; "
            "joining it would assign a DIFFERENT donor's cortical layers to "
            "Br2719 spots. GSE307403 supplies no per-spot manual layer GT for "
            "Br2719 / capture V12F14-053_A1, and Br2719 is absent from both the "
            "Maynard 2021 HumanPilot and the 2023 spatialDLPFC released label "
            "sets. ARI/NMI skipped to avoid fabricating ground truth."
        )
        info["gt_safety_guard"] = "donor_mismatch_refused_whitelist_collision_join"
        return {}, info

    # (Reached only if a genuine same-donor Br2719 map is ever wired in.)
    if coverage < _GT_MIN_COVERAGE:
        info["gt_fetch"] = (
            f"blocked: same-donor coverage too low ({coverage:.1%} < "
            f"{_GT_MIN_COVERAGE:.0%})"
        )
        return {}, info

    layer_of: dict[str, str] = {}
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) >= 3 and "Br2719" in parts[1]:
            layer_of[parts[0]] = parts[2]
    labels_true, labels_pred = [], []
    for b, dom in zip(barcodes, domain_id, strict=True):
        lab = layer_of.get(str(b))
        if lab is not None:
            labels_true.append(lab)
            labels_pred.append(int(dom))
    if len(labels_true) < 2:
        info["gt_fetch"] = "blocked: <2 joined spots after same-donor filter"
        return {}, info

    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    yt = np.asarray(labels_true)
    yp = np.asarray(labels_pred)
    info["gt_fetch"] = "ok"
    info["gt_source"] = _HUMANPILOT_LAYER_MAP_URL
    info["gt_n_joined"] = len(labels_true)
    info["gt_n_layers"] = int(np.unique(yt).size)
    return (
        {
            "domain_vs_layer_ari_floor": float(adjusted_rand_score(yt, yp)),
            "domain_vs_layer_nmi_floor": float(normalized_mutual_info_score(yt, yp)),
        },
        info,
    )


# --------------------------------------------------------------------------- #
# 4. NMF-contrast (gap to a real factor model)
# --------------------------------------------------------------------------- #
def _nmf_contrast(
    X: np.ndarray, Z_mvp: np.ndarray, k: int, seed: int, n_hvg: int
) -> dict[str, float]:
    """Run sklearn NMF (k factors) on the same X (restricted to the same HVG set
    for tractability) and report ``matched_factor_correlation`` between the MVP
    Z and the NMF factor activations. A LOW correlation quantifies how far the
    random-projection MVP is from a genuine non-negative factor model."""
    from sklearn.decomposition import NMF

    variances = X.var(axis=0)
    n_keep = int(min(n_hvg, X.shape[1]))
    hvg_idx = np.argsort(variances)[::-1][:n_keep]
    Xh = X[:, hvg_idx]
    Xh = np.clip(Xh, 0.0, None)  # NMF requires non-negative input.

    model = NMF(
        n_components=int(k),
        init="nndsvda",
        random_state=int(seed),
        max_iter=400,
    )
    W_nmf = model.fit_transform(Xh)  # (n_spots, k) factor activations
    corr = matched_factor_correlation(Z_mvp.astype(np.float64), W_nmf.astype(np.float64))
    return {
        "nmf_contrast_matched_corr": float(corr),
        "nmf_contrast_gap": float(1.0 - corr),
        "nmf_contrast_k": float(k),
        "nmf_reconstruction_err": float(model.reconstruction_err_),
    }


# --------------------------------------------------------------------------- #
# merge + emit (preserve existing run-path metrics + caveats)
# --------------------------------------------------------------------------- #
def _load_existing(results_root: Path) -> tuple[dict, dict]:
    proj = results_root / "factorgraph-st"
    metrics = json.loads((proj / "metrics.json").read_text())
    metadata = json.loads((proj / "run_metadata.json").read_text())
    return metrics, metadata


def main() -> None:
    args = _parse_args()
    h5ad_path = args.h5ad.resolve()
    factors_path = args.factors.resolve()
    t0 = time.perf_counter()

    if args.results_dir is not None:
        results_root = args.results_dir.resolve()
    else:
        results_root = _REPO_ROOT / "results"

    existing_metrics, existing_metadata = _load_existing(results_root)

    # --- load real data + factors -------------------------------------------
    adata = sc.read_h5ad(h5ad_path)
    n_obs, n_vars = int(adata.shape[0]), int(adata.shape[1])
    barcodes = np.asarray(adata.obs.index.astype(str))
    if _looks_like_raw_counts(adata.X):
        sc.pp.log1p(adata)
    X = adata.X
    X = (
        np.asarray(X.todense(), dtype=np.float32)
        if sp.issparse(X)
        else np.asarray(X, dtype=np.float32)
    )
    var_names = np.asarray(adata.var.index.astype(str))
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    edges = _build_knn_edges(coords, args.knn)
    neighbor_idx = _knn_index(coords, args.knn)

    factors = np.load(factors_path)
    domain_id = factors["domain_id"].astype(np.int64)
    Z_shared = factors["Z_shared"]
    k_shared = int(Z_shared.shape[1])

    # Domain-clustering degeneracy: main's graph-aware decoder collapses each
    # connected component of the kNN graph to one label, so a single fully-
    # connected slide yields ONE domain. Surface it so the PAS/CHAOS values below
    # (trivial under a constant domain_id) are read as degenerate, not signal.
    n_domains_found = int(np.unique(domain_id).size)
    domain_degenerate = n_domains_found <= 1

    # --- 1. gene-level Moran's I (HONEST, model-independent) ------------------
    gene_metrics, top_svgs = _gene_level_morans(
        X, var_names, edges, args.n_hvg, args.top_n_svg
    )

    # --- 2. PAS + CHAOS of domain_id with nulls (MVP-characterization) --------
    pas_chaos = _pas_chaos_with_nulls(
        domain_id, coords, neighbor_idx, args.n_null_shuffles, args.seed
    )

    # --- 3. spatialLIBD layer-GT floor (never fabricate) ----------------------
    gt_metrics, gt_info = _gt_floor(barcodes, domain_id, args.no_network)

    # --- 4. NMF-contrast ------------------------------------------------------
    nmf_metrics = _nmf_contrast(X, Z_shared, k_shared, args.seed, args.n_hvg)

    # --- merge metrics (preserve existing) -----------------------------------
    merged_metrics: dict[str, float] = dict(existing_metrics.get("metrics", {}))
    merged_metrics.update(gene_metrics)
    merged_metrics.update(pas_chaos)
    merged_metrics.update(gt_metrics)
    merged_metrics.update(nmf_metrics)
    merged_metrics["domain_clustering_degenerate"] = float(domain_degenerate)

    runtime_s = time.perf_counter() - t0

    # --- preserve + extend run_metadata --------------------------------------
    interpretability = dict(existing_metadata.get("interpretability", {}))
    interpretability["further_results"] = {
        "gene_level_morans_i": {
            "model_is_learned": False,
            "honest": True,
            "note": (
                "MODEL-INDEPENDENT: per-gene Moran's I on the kNN spatial graph "
                "characterizes the REAL DLPFC data; does not depend on the MVP "
                "model. This is the cleanest honest win."
            ),
            "top_svgs": top_svgs,
        },
        "domain_pas_chaos": {
            "model_is_learned": False,
            "honest": False,
            "domain_clustering_degenerate": bool(domain_degenerate),
            "domain_count_found": n_domains_found,
            "note": (
                "PAS/CHAOS of domain_id are high BY CONSTRUCTION (coords baked into "
                "the encoder features AND used directly as a graph-aware k-means "
                "clustering input, decoder.py:_cluster_domains). Paired with "
                "permutation nulls; interpret ONLY as delta over null, never as "
                "learned structure."
                + (
                    " NOTE: on this single fully-connected slide domain_id "
                    "DEGENERATED to one domain (main's decoder collapses each "
                    "connected component to one label), so PAS/CHAOS and their "
                    "nulls are trivial here -- the values are recorded for "
                    "provenance only and carry no spatial signal."
                    if domain_degenerate
                    else ""
                )
            ),
        },
        "domain_vs_layer_floor": {
            "model_is_learned": False,
            "is_baseline_floor": True,
            **gt_info,
        },
        "nmf_contrast": {
            "model_is_learned": False,
            "note": (
                "matched_factor_correlation between the random-projection MVP "
                "Z_shared and a real sklearn NMF factor model on the same HVG "
                "matrix; LOW correlation == large gap to a genuine factor model. "
                "nmf_contrast_gap = 1 - matched_corr."
            ),
        },
    }

    base_notes = existing_metadata.get("notes", "")
    further_note = (
        " | further_results appended: gene-level Moran's I (honest, "
        "model-independent), PAS/CHAOS+nulls (caveated), spatialLIBD layer-GT "
        f"floor (gt_fetch={gt_info.get('gt_fetch', 'n/a')}), NMF-contrast."
    )

    run_metadata = {
        "dataset_paths": existing_metadata.get("dataset_paths", [str(h5ad_path)]),
        "n_obs": n_obs,
        "n_vars": n_vars,
        "seed": existing_metadata.get("seed", args.seed),
        "runtime_s": (existing_metadata.get("runtime_s") or 0.0) + runtime_s,
        "started_utc": existing_metadata.get("started_utc"),
        "device": existing_metadata.get("device", "cpu"),
        "deterministic": existing_metadata.get("deterministic", True),
        "num_threads": existing_metadata.get("num_threads", 1),
        "reproducibility_level": existing_metadata.get(
            "reproducibility_level", "seeded"
        ),
        "normalization": existing_metadata.get(
            "normalization", {"applied": True, "method": "log1p"}
        ),
        "interpretability": interpretability,
        "notes": base_notes + further_note,
    }

    outputs = existing_metadata.get("outputs", {"factors": str(factors_path)})

    paths = results_contract.write_results(
        project="factorgraph-st",
        dataset_card_id=existing_metrics.get(
            "dataset_card_id",
            results_contract.dataset_card_id([str(h5ad_path)]),
        ),
        metrics=merged_metrics,
        outputs=outputs,
        run_metadata=run_metadata,
        results_dir=results_root,
    )

    print(f"Wrote metrics:      {paths['metrics']}")
    print(f"Wrote run_metadata: {paths['run_metadata']}")
    print(
        "gene_morans_i_mean="
        f"{gene_metrics['gene_morans_i_mean']:.4f} "
        f"top_svg={top_svgs[0]['gene'] if top_svgs else 'n/a'}"
    )
    print(
        f"domain_pas={pas_chaos['domain_pas']:.4f} "
        f"(null {pas_chaos['domain_pas_null']:.4f}) "
        f"domain_chaos={pas_chaos['domain_chaos']:.4f} "
        f"(null {pas_chaos['domain_chaos_null']:.4f})"
    )
    print(f"gt_fetch={gt_info.get('gt_fetch')}")
    print(
        f"nmf_contrast_matched_corr={nmf_metrics['nmf_contrast_matched_corr']:.4f} "
        f"gap={nmf_metrics['nmf_contrast_gap']:.4f}"
    )


if __name__ == "__main__":
    main()
