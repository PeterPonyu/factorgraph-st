"""Synthetic benchmark metrics for FactorGraph-ST."""

from __future__ import annotations

from itertools import permutations

import numpy as np


def matched_factor_correlation(estimated: np.ndarray, truth: np.ndarray) -> float:
    """Mean absolute Pearson correlation after best one-to-one factor matching."""
    corr = _abs_corr_matrix(estimated, truth)
    if corr.size == 0:
        return 1.0
    rows, cols = corr.shape
    if max(rows, cols) <= 8:
        best = 0.0
        for perm in permutations(range(cols), min(rows, cols)):
            score = sum(corr[r, c] for r, c in enumerate(perm))
            best = max(best, score)
        return float(best / min(rows, cols))
    # Greedy fallback keeps the package dependency-free for larger matrices.
    used: set[int] = set()
    total = 0.0
    for r in range(rows):
        choices = [(corr[r, c], c) for c in range(cols) if c not in used]
        if not choices:
            break
        val, c = max(choices)
        used.add(c)
        total += float(val)
    return float(total / min(rows, cols))


def section_overlap(Z: np.ndarray, section_id: np.ndarray) -> np.ndarray:
    """Fraction of each factor's activation mass in each section."""
    if Z.ndim != 2:
        raise ValueError("Z must be 2D")
    sections = np.unique(section_id)
    out = np.zeros((Z.shape[1], sections.size), dtype=np.float32)
    mass = Z.sum(axis=0)
    for j, section in enumerate(sections):
        out[:, j] = Z[section_id == section].sum(axis=0) / np.maximum(mass, 1e-12)
    return out


def shared_private_separation(Z_shared: np.ndarray, Z_private: np.ndarray, section_id: np.ndarray) -> dict[str, float]:
    """Return section-spread summaries for shared and private activations."""
    shared = section_overlap(Z_shared, section_id)
    private = section_overlap(Z_private, section_id)
    return {
        "shared_mean_active_sections": float((shared > 0.05).sum(axis=1).mean()) if shared.size else 0.0,
        "private_mean_active_sections": float((private > 0.05).sum(axis=1).mean()) if private.size else 0.0,
    }


def morans_i(values: np.ndarray, edges: np.ndarray) -> float:
    """Compute Moran's I on graph edges for numeric labels or scores."""
    x = values.astype(np.float64)
    n = x.size
    if n == 0 or edges.size == 0:
        return 0.0
    centered = x - x.mean()
    denom = float(np.sum(centered**2))
    if denom == 0.0:
        return 0.0
    src, dst = edges
    w = src.size
    return float((n / w) * np.sum(centered[src] * centered[dst]) / denom)


def adjusted_rand_index(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """Dependency-free adjusted Rand index."""
    if labels_true.shape[0] != labels_pred.shape[0]:
        raise ValueError("label arrays must have equal length")
    n = labels_true.size
    if n < 2:
        return 1.0
    true_vals, true_inv = np.unique(labels_true, return_inverse=True)
    pred_vals, pred_inv = np.unique(labels_pred, return_inverse=True)
    table = np.zeros((true_vals.size, pred_vals.size), dtype=np.int64)
    np.add.at(table, (true_inv.ravel(), pred_inv.ravel()), 1)
    sum_comb = _comb2(table).sum()
    row_comb = _comb2(table.sum(axis=1)).sum()
    col_comb = _comb2(table.sum(axis=0)).sum()
    total = n * (n - 1) / 2
    expected = row_comb * col_comb / total if total else 0.0
    maximum = 0.5 * (row_comb + col_comb)
    denom = maximum - expected
    if denom == 0:
        return 1.0
    return float((sum_comb - expected) / denom)


def _abs_corr_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape[0] != b.shape[0]:
        raise ValueError("factor matrices must have the same row count")
    a0 = a - a.mean(axis=0, keepdims=True)
    b0 = b - b.mean(axis=0, keepdims=True)
    denom = np.linalg.norm(a0, axis=0)[:, None] * np.linalg.norm(b0, axis=0)[None, :]
    corr = (a0.T @ b0) / np.maximum(denom, 1e-12)
    return np.clip(np.abs(corr), 0.0, 1.0)


def _comb2(x: np.ndarray) -> np.ndarray:
    return x * (x - 1) / 2
