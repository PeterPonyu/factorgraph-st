"""Regression for #197: count-like input must get library-size normalization
(``normalize_total``) BEFORE ``log1p``, and the runner must record what it did.

The old runner applied only a heuristic ``log1p`` (no ``normalize_total``), so
raw library-size variance dominated the encoder input. These tests pin the
corrected preprocessing contract in ``scripts/run_real_factorgraph.py``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scanpy")
pytest.importorskip("anndata")

import anndata as ad  # noqa: E402

_RUNNER = (
    Path(__file__).resolve().parents[1] / "scripts" / "run_real_factorgraph.py"
)


def _load_runner():
    spec = importlib.util.spec_from_file_location("_run_real_factorgraph", _RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _count_adata(seed: int = 0) -> ad.AnnData:
    """Synthetic count-like AnnData with strongly varying library sizes."""
    rng = np.random.default_rng(seed)
    n_obs, n_vars = 40, 12
    base = rng.poisson(3.0, size=(n_obs, n_vars)).astype(np.float32)
    # Inject 10x library-size variation across spots so normalize_total has a
    # visible effect on row sums.
    depth = rng.integers(1, 11, size=n_obs).astype(np.float32)[:, None]
    X = base * depth
    return ad.AnnData(X=X)


def test_normalize_total_applied_to_counts():
    """Count-like input → method records normalize_total+log1p and target_sum."""
    mod = _load_runner()
    adata = _count_adata()
    norm = mod._preprocess(adata, already_normalized=False, n_hvg=0)
    assert norm["applied"] is True
    assert norm["method"] == "normalize_total+log1p"
    assert norm["target_sum"] == pytest.approx(1e4)


def test_row_sums_normalized_pre_log1p():
    """After preprocessing, per-spot expression is library-size corrected.

    With ``normalize_total`` the pre-log row sums are equalized; the old
    log1p-only path left raw (wildly varying) library sizes intact. We verify
    the row-sum coefficient of variation collapses relative to the raw input.
    """
    mod = _load_runner()
    adata = _count_adata()
    raw_sums = np.asarray(adata.X).sum(axis=1)
    raw_cv = raw_sums.std() / raw_sums.mean()

    mod._preprocess(adata, already_normalized=False, n_hvg=0)
    # Invert log1p to recover the normalized (pre-log) counts.
    norm_counts = np.expm1(np.asarray(adata.X))
    norm_sums = norm_counts.sum(axis=1)
    norm_cv = norm_sums.std() / max(norm_sums.mean(), 1e-12)

    assert raw_cv > 0.3, "fixture should have strong library-size variation"
    assert norm_cv < raw_cv / 5, (
        f"normalize_total did not equalize library sizes (raw_cv={raw_cv:.3f} "
        f"norm_cv={norm_cv:.3f})"
    )


def test_already_normalized_skips_both():
    """``--already-normalized`` records method 'none' and leaves X untouched."""
    mod = _load_runner()
    adata = _count_adata()
    before = np.asarray(adata.X).copy()
    norm = mod._preprocess(adata, already_normalized=True, n_hvg=0)
    assert norm["applied"] is False
    assert norm["method"] == "none"
    np.testing.assert_array_equal(np.asarray(adata.X), before)


def test_hvg_selection_subsets_genes():
    """``n_hvg > 0`` subsets genes and records the count used."""
    mod = _load_runner()
    adata = _count_adata()
    norm = mod._preprocess(adata, already_normalized=False, n_hvg=5)
    assert adata.n_vars <= 5
    assert norm["hvg_applied"] is True
    assert norm["n_genes_used"] == adata.n_vars
