"""Tests for #200: UMAP/PCA factor-embedding visualisation (factorgraph_st.viz).

``embed_2d`` reduces the factor embedding ``H`` (shape ``(n_spots, d)``) to 2D
(UMAP/PCA if available, numpy-SVD PCA fallback otherwise) and
``plot_embedding`` writes a non-interactive (Agg) PNG scatter coloured by a
per-spot label. Tests run headless on a tiny synthetic ``H``.
"""

from __future__ import annotations

import numpy as np
import pytest

mpl = pytest.importorskip("matplotlib")

from factorgraph_st.viz import embed_2d, plot_embedding


def _tiny_H(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Two well-separated blobs in 6D with cluster labels."""
    rng = np.random.default_rng(seed)
    a = rng.normal(0.0, 0.3, size=(20, 6)) + np.array([5, 0, 0, 0, 0, 0])
    b = rng.normal(0.0, 0.3, size=(20, 6)) + np.array([-5, 0, 0, 0, 0, 0])
    H = np.vstack([a, b]).astype(np.float32)
    labels = np.array([0] * 20 + [1] * 20, dtype=np.int64)
    return H, labels


def test_embed_2d_shape_svd_fallback():
    H, _ = _tiny_H()
    emb = embed_2d(H, method="pca", seed=0)
    assert emb.shape == (40, 2)
    assert np.isfinite(emb).all()


def test_embed_2d_auto_does_not_crash():
    H, _ = _tiny_H()
    emb = embed_2d(H, method="auto", seed=0)
    assert emb.shape == (40, 2)


def test_plot_embedding_writes_png(tmp_path):
    H, labels = _tiny_H()
    out = tmp_path / "factors.png"
    written = plot_embedding(H, labels=labels, out_path=str(out), method="pca", seed=0)
    assert out.exists()
    assert out.stat().st_size > 0
    assert str(written) == str(out)


def test_plot_embedding_accepts_precomputed_embedding(tmp_path):
    H, labels = _tiny_H()
    emb = embed_2d(H, method="pca", seed=0)
    out = tmp_path / "pre.png"
    plot_embedding(emb, labels=labels, out_path=str(out))
    assert out.exists() and out.stat().st_size > 0


def test_plot_embedding_no_labels(tmp_path):
    H, _ = _tiny_H()
    out = tmp_path / "nolabel.png"
    plot_embedding(H, labels=None, out_path=str(out), method="pca")
    assert out.exists() and out.stat().st_size > 0


def test_matplotlib_uses_agg_backend():
    import matplotlib

    # Importing factorgraph_st.viz must force a headless backend.
    import factorgraph_st.viz  # noqa: F401

    assert matplotlib.get_backend().lower() == "agg"
