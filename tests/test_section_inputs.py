"""Regression tests for #71: ``build_section_inputs`` must emit dense float32 X.

Real Visium / AnnData loads return SciPy sparse matrices of int / float64,
so the first call into ``encode_graph`` / ``fit_transform`` previously
failed at ``validate_inputs`` or silently broke on
``np.isfinite(sparse).all()``. ``build_section_inputs`` now enforces the
documented dense-``float32``-finite contract at the data-bridge boundary.
"""

from __future__ import annotations

import numpy as np
import pytest

from factorgraph_st.data import build_section_inputs
from factorgraph_st.schemas import validate_inputs


class _SparseLike:
    """Minimal scipy.sparse duck type (no scipy dependency required)."""

    def __init__(self, dense: np.ndarray) -> None:
        self._dense = dense
        self.shape = dense.shape

    def toarray(self) -> np.ndarray:
        return self._dense.copy()


def test_X_dense_float32():
    """Sparse-like + float64 blocks → stacked dense float32 X passing validate_inputs."""
    rng = np.random.default_rng(0)
    blocks = [
        {
            "X": _SparseLike(rng.integers(0, 5, size=(6, 4)).astype(np.int32)),
            "coords": rng.uniform(size=(6, 2)).astype(np.float64),
        },
        {
            "X": rng.uniform(size=(5, 4)).astype(np.float64),
            "coords": rng.uniform(size=(5, 2)).astype(np.float64),
        },
    ]
    out = build_section_inputs(blocks)

    assert isinstance(out["X"], np.ndarray) and not hasattr(out["X"], "toarray")
    assert out["X"].dtype == np.float32 and np.isfinite(out["X"]).all()
    assert out["X"].shape == (11, 4)
    assert out["coords"].dtype == np.float32
    assert out["section_id"].dtype == np.int64
    np.testing.assert_array_equal(out["section_id"], [0] * 6 + [1] * 5)
    assert out["edges"].dtype == np.int64 and out["edges"].shape == (2, 0)
    validate_inputs(out["X"], out["coords"], out["section_id"], out["edges"])


def test_rejects_non_finite_X():
    block = {
        "X": np.array([[1.0, np.nan], [0.0, 1.0]], dtype=np.float64),
        "coords": np.zeros((2, 2), dtype=np.float32),
    }
    with pytest.raises(ValueError, match="non-finite"):
        build_section_inputs([block])


def test_rejects_empty_blocks():
    with pytest.raises(ValueError, match="non-empty"):
        build_section_inputs([])
