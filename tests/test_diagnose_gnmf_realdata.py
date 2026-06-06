"""Unit tests for the numpy-only core of the #392 real-data GNMF diagnosis.

Heavy real-data paths (scanpy/anndata) are NOT exercised here; only the
pure-numpy scoring/aggregation helpers, which carry the diagnostic logic.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]


def _load_module():
    path = _REPO / "scripts" / "diagnose_gnmf_realdata.py"
    spec = importlib.util.spec_from_file_location("diagnose_gnmf_realdata", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


diag = _load_module()


# --- edge_label_purity ----------------------------------------------------- #
def test_edge_purity_all_same_domain_is_one():
    edges = np.array([[0, 1, 2], [1, 2, 3]])
    assert diag.edge_label_purity(np.zeros(4, dtype=int), edges) == 1.0


def test_edge_purity_all_cross_domain_is_zero():
    # every edge connects a 0 to a 1
    edges = np.array([[0, 2], [1, 3]])
    assert diag.edge_label_purity(np.array([0, 1, 0, 1]), edges) == 0.0


def test_edge_purity_mixed_fraction():
    # 2 of 4 edges are within-domain
    edges = np.array([[0, 1, 2, 3], [1, 2, 3, 0]])
    dom = np.array([0, 0, 1, 1])  # edges: (0,0)same (0,1)diff (1,1)same (1,0)diff -> 0.5
    assert diag.edge_label_purity(dom, edges) == 0.5


def test_edge_purity_empty_edges_is_nan():
    assert math.isnan(diag.edge_label_purity(np.array([0, 1]), np.empty((2, 0), dtype=int)))


# --- objective_split ------------------------------------------------------- #
def test_objective_split_recon_matches_frobenius():
    rng = np.random.default_rng(0)
    X = rng.random((6, 5))
    H = rng.random((6, 2))
    W = rng.random((5, 2))
    edges = np.array([[0, 1, 2], [1, 2, 3]])
    split = diag.objective_split(X, H, W, edges, lam=2.0)
    expected_recon = float(np.sum((np.clip(X, 0, None) - H @ W.T) ** 2))
    assert split["recon"] == pytest.approx(expected_recon)
    assert split["lam_smooth"] == pytest.approx(2.0 * split["smooth"])


def test_objective_split_constant_H_has_zero_smoothness():
    # L @ 1 = 0, so a spatially constant H carries no Laplacian penalty.
    X = np.ones((4, 3))
    H = np.ones((4, 2))
    W = np.ones((3, 2))
    edges = np.array([[0, 1, 2], [1, 2, 3]])
    split = diag.objective_split(X, H, W, edges, lam=5.0)
    assert split["smooth"] == pytest.approx(0.0, abs=1e-9)
    assert split["smooth_fraction"] == pytest.approx(0.0, abs=1e-9)


def test_objective_split_fraction_grows_with_lam():
    rng = np.random.default_rng(1)
    X = rng.random((8, 4))
    H = rng.random((8, 2))
    W = rng.random((4, 2))
    edges = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
    lo = diag.objective_split(X, H, W, edges, lam=1.0)["smooth_fraction"]
    hi = diag.objective_split(X, H, W, edges, lam=1000.0)["smooth_fraction"]
    assert hi > lo


# --- labeled_gt_codes ------------------------------------------------------ #
def test_labeled_gt_codes_drops_na_tokens():
    raw = ["L1", "nan", "L2", "", "L1", "unknown"]
    valid, codes = diag.labeled_gt_codes(raw)
    assert valid.tolist() == [True, False, True, False, True, False]
    assert codes.shape == (3,)
    # two distinct surviving labels (L1, L2)
    assert set(codes.tolist()) == {0, 1}


# --- aggregate_sweep ------------------------------------------------------- #
def test_aggregate_sweep_means_and_sorted():
    rows = [
        {"lam": 10.0, "seed": 0, "ari": 0.2, "nmi": 0.3, "ami": 0.3, "edge_purity": 0.5, "smooth_fraction": 0.1},
        {"lam": 10.0, "seed": 1, "ari": 0.4, "nmi": 0.5, "ami": 0.5, "edge_purity": 0.7, "smooth_fraction": 0.1},
        {"lam": 1.0, "seed": 0, "ari": 0.1, "nmi": 0.1, "ami": 0.1, "edge_purity": 0.4, "smooth_fraction": 0.01},
    ]
    agg = diag.aggregate_sweep(rows)
    assert [r["lam"] for r in agg] == [1.0, 10.0]  # sorted ascending
    lam10 = agg[1]
    assert lam10["ari_mean"] == pytest.approx(0.3)
    assert lam10["ari_std"] == pytest.approx(0.1)
    assert lam10["n_seeds"] == 2


# --- recommend_lam --------------------------------------------------------- #
def test_recommend_lam_prefers_stable_max():
    agg = [
        {"lam": 1.0, "ari_mean": 0.16, "ari_std": 0.01},
        {"lam": 300.0, "ari_mean": 0.29, "ari_std": 0.02},
        {"lam": 1000.0, "ari_mean": 0.30, "ari_std": 0.18},  # higher mean but UNSTABLE
    ]
    rec = diag.recommend_lam(agg, stable_std=0.05)
    assert rec["lam"] == 300.0  # the unstable 1000 is excluded
    assert rec["stable"] is True


def test_recommend_lam_falls_back_when_none_stable():
    agg = [
        {"lam": 1.0, "ari_mean": 0.16, "ari_std": 0.2},
        {"lam": 300.0, "ari_mean": 0.29, "ari_std": 0.3},
    ]
    rec = diag.recommend_lam(agg, stable_std=0.05)
    assert rec["lam"] == 300.0
    assert rec["stable"] is False


def test_recommend_lam_empty_is_none():
    assert diag.recommend_lam([], stable_std=0.05) is None
