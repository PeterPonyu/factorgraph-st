"""Non-interactive 2D visualisation of the factor embedding ``H``.

Issue #200. Reduces the factor embedding ``H`` (shape ``(n_spots, d)``, the
encoder output / ``FactorGraphOutputs.H``) to two dimensions and writes a PNG
scatter coloured by a per-spot label (e.g. ``domain_id`` or a cluster).

Dimensionality reduction backends, in preference order for ``method="auto"``:

1. UMAP (``umap-learn``) — if installed.
2. scanpy's PCA (``scanpy``) — if installed.
3. numpy SVD PCA — always available (no optional dependency).

matplotlib is required and is an opt-in ``[viz]`` extra in ``pyproject.toml``.
Importing this module forces the headless ``Agg`` backend so it never tries to
open a display.

Brand-independence: standard PCA/UMAP scatter; no third-party spatial-omics
source or class names reused.
"""

from __future__ import annotations

import matplotlib
import numpy as np

# Force a non-interactive backend BEFORE pyplot is imported anywhere so this is
# safe to import in headless CI / batch jobs. The pyplot import is deliberately
# placed AFTER matplotlib.use(...) (E402 / import-order is load-bearing here;
# see the per-file-ignore for this module in pyproject.toml).
matplotlib.use("Agg")
import matplotlib.pyplot as plt

__all__ = ["embed_2d", "plot_embedding"]


def embed_2d(H: np.ndarray, *, method: str = "auto", seed: int = 0) -> np.ndarray:
    """Reduce ``H`` of shape ``(n_spots, d)`` to a ``(n_spots, 2)`` embedding.

    Parameters
    ----------
    H:
        Factor embedding (encoder output).
    method:
        ``"auto"`` tries UMAP, then scanpy PCA, then a numpy-SVD PCA fallback.
        ``"umap"`` / ``"pca"`` force a specific backend (``"umap"`` falls back
        to PCA if ``umap-learn`` is absent).
    seed:
        Random seed forwarded to UMAP / used for deterministic PCA.
    """
    H = np.asarray(H, dtype=np.float64)
    if H.ndim != 2:
        raise ValueError(f"H must be 2D (n_spots, d); got shape {H.shape}")
    if H.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64)
    if H.shape[1] <= 2:
        # Already low-dimensional: pad to 2 columns if needed.
        if H.shape[1] == 2:
            return H.copy()
        return np.column_stack([H[:, 0], np.zeros(H.shape[0])])

    if method in ("auto", "umap"):
        emb = _try_umap(H, seed)
        if emb is not None:
            return emb
        if method == "umap":
            # explicit umap requested but unavailable -> fall through to PCA
            pass

    if method == "auto":
        emb = _try_scanpy_pca(H)
        if emb is not None:
            return emb

    return _pca_svd(H)


def plot_embedding(
    H_or_embedding: np.ndarray,
    *,
    labels: np.ndarray | None = None,
    out_path: str,
    method: str = "auto",
    seed: int = 0,
    title: str | None = None,
    point_size: float = 12.0,
) -> str:
    """Write a 2D scatter PNG of the factor embedding coloured by ``labels``.

    ``H_or_embedding`` may be either the raw embedding ``H`` (``d > 2``, reduced
    via :func:`embed_2d`) or a precomputed ``(n_spots, 2)`` embedding (used
    as-is). Returns the path written.
    """
    arr = np.asarray(H_or_embedding, dtype=np.float64)
    emb = arr if arr.ndim == 2 and arr.shape[1] == 2 else embed_2d(arr, method=method, seed=seed)

    fig, ax = plt.subplots(figsize=(6, 5))
    try:
        if labels is None:
            ax.scatter(emb[:, 0], emb[:, 1], s=point_size, c="#3b5b92")
        else:
            labels = np.asarray(labels)
            classes = np.unique(labels)
            cmap = plt.get_cmap("tab20" if classes.size > 10 else "tab10")
            for i, c in enumerate(classes):
                mask = labels == c
                ax.scatter(
                    emb[mask, 0],
                    emb[mask, 1],
                    s=point_size,
                    color=cmap(i % cmap.N),
                    label=str(c),
                )
            if classes.size <= 20:
                ax.legend(title="label", fontsize="x-small", markerscale=1.5, loc="best")
        ax.set_xlabel("dim 1")
        ax.set_ylabel("dim 2")
        if title:
            ax.set_title(title)
        fig.tight_layout()
        fig.savefig(out_path, dpi=120)
    finally:
        plt.close(fig)
    return out_path


def _try_umap(H: np.ndarray, seed: int) -> np.ndarray | None:
    try:
        import umap  # type: ignore
    except ImportError:
        return None
    n_neighbors = min(15, max(2, H.shape[0] - 1))
    reducer = umap.UMAP(n_components=2, random_state=seed, n_neighbors=n_neighbors)
    return np.asarray(reducer.fit_transform(H), dtype=np.float64)


def _try_scanpy_pca(H: np.ndarray) -> np.ndarray | None:
    try:
        import scanpy as sc  # type: ignore
        from anndata import AnnData  # type: ignore
    except ImportError:
        return None
    adata = AnnData(H.astype(np.float32))
    sc.pp.pca(adata, n_comps=2)
    return np.asarray(adata.obsm["X_pca"][:, :2], dtype=np.float64)


def _pca_svd(H: np.ndarray) -> np.ndarray:
    """Deterministic 2-component PCA via numpy SVD (no optional dependency)."""
    centered = H - H.mean(axis=0, keepdims=True)
    # full_matrices=False keeps this efficient for tall-skinny H.
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T
