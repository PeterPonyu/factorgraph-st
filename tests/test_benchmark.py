"""Tests for the model-side domain-detection benchmark adapter (#368).

All numpy-only: the adapter core takes raw arrays, and the AnnData entry point is
exercised with a duck-typed stand-in (a namespace with ``.X`` + ``.obsm``) so no
anndata dependency is pulled into the test.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from factorgraph_st import benchmark


def _toy(seed: int = 0):
    """A tiny 3-block synthetic: spatially-contiguous blocks with block-specific genes."""
    rng = np.random.default_rng(seed)
    n_per, n_genes = 16, 12
    blocks = []
    coords = []
    X = []
    for b in range(3):
        xc = rng.normal(b * 10.0, 0.5, size=(n_per, 1))
        yc = rng.normal(0.0, 0.5, size=(n_per, 1))
        coords.append(np.hstack([xc, yc]))
        expr = rng.gamma(0.3, 1.0, size=(n_per, n_genes))
        expr[:, b * 4 : b * 4 + 4] += 5.0  # block-specific genes
        X.append(expr)
        blocks.append(np.full(n_per, b))
    return np.vstack(X).astype(np.float32), np.vstack(coords).astype(np.float32), np.concatenate(blocks)


def test_build_knn_edges_symmetric_no_self_loops():
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    edges = benchmark.build_knn_edges(coords, k=2)
    assert edges.shape[0] == 2
    src, dst = edges
    assert not np.any(src == dst)  # no self loops
    pairset = set(map(tuple, edges.T.tolist()))
    assert all((d, s) in pairset for s, d in pairset)  # symmetric


def test_build_knn_edges_empty():
    assert benchmark.build_knn_edges(np.empty((0, 2)), k=3).shape == (2, 0)


def test_run_domain_detection_arrays_shapes_and_contract():
    X, coords, _ = _toy()
    res = benchmark.run_domain_detection_arrays(
        X, coords, seed=0, n_domains=3, k_shared=2, k_private=1, n_iter=50
    )
    n, g = X.shape
    k = 3  # k_shared + k_private
    assert res.labels.shape == (n,)
    assert res.labels.dtype == np.int64
    assert res.labels.min() >= 0 and res.labels.max() < 3
    assert res.embedding.shape == (n, k)
    assert res.factors.shape == (n, k)
    assert res.loadings.shape == (k, g)  # (n_factors, n_genes)
    # provenance carries the model contribution but NEVER a ground-truth metric.
    assert res.provenance["method"] == benchmark.METHOD_ID
    assert res.provenance["reproducibility_level"] == "deterministic_seeded"
    assert res.provenance["seed"] == 0
    assert res.provenance["n_factors"] == k
    assert res.provenance["hyperparameters"]["n_domains"] == 3
    assert "ground_truth_ari" not in res.provenance
    assert "ari" not in res.provenance


def test_run_domain_detection_arrays_is_deterministic():
    X, coords, _ = _toy()
    a = benchmark.run_domain_detection_arrays(X, coords, seed=7, n_domains=3, n_iter=50)
    b = benchmark.run_domain_detection_arrays(X, coords, seed=7, n_domains=3, n_iter=50)
    assert np.array_equal(a.labels, b.labels)
    assert np.allclose(a.embedding, b.embedding)


def test_run_domain_detection_anndata_duck_typed_matches_core():
    X, coords, _ = _toy()
    adata = types.SimpleNamespace(X=X, obsm={"spatial": coords})
    via_adata = benchmark.run_domain_detection(adata, seed=3, n_domains=3, n_iter=40)
    via_arrays = benchmark.run_domain_detection_arrays(X, coords, seed=3, n_domains=3, n_iter=40)
    assert np.array_equal(via_adata.labels, via_arrays.labels)


def test_run_domain_detection_requires_spatial():
    X, _, _ = _toy()
    adata = types.SimpleNamespace(X=X, obsm={})
    with pytest.raises(ValueError, match="spatial"):
        benchmark.run_domain_detection(adata, seed=0, n_domains=3)
