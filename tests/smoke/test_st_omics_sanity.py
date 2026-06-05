"""Smoke / sanity for factorgraph-st using very small real-ST-format data.

Real cards use AnnData with .raw or X=raw int counts, obsm['spatial'],
per-section or slice_id. Not bare scRNA (no coords, no section, normalized X).

Exercises the build_section_inputs bridge + strict schema contract (dense float32 etc)
and the run_real path patterns.
"""

from __future__ import annotations

import numpy as np
import pytest

from factorgraph_st.data.build import build_section_inputs
from factorgraph_st.schemas import validate_inputs


def _tiny_st_blocks(n_sections: int = 2, n_per: int = 7, n_genes: int = 5, seed: int = 2024):
    """Return list of dict blocks mimicking loaded per-section ST AnnData (raw counts + spatial)."""
    rng = np.random.default_rng(seed)
    blocks = []
    for s in range(n_sections):
        X = rng.poisson(1.9, size=(n_per, n_genes)).astype(np.float32)
        coords = rng.uniform(10, 500, size=(n_per, 2)).astype(np.float32)
        blocks.append({
            "X": X,
            "coords": coords,
            "section_id": s,
        })
    return blocks


def test_build_section_inputs_from_st_format():
    """build_section_inputs (used by real loaders) must accept and validate tiny ST blocks."""
    blocks = _tiny_st_blocks()
    out = build_section_inputs(blocks)
    assert out["X"].dtype == np.float32
    assert out["X"].shape[0] == 14
    assert out["coords"].shape == (14, 2)
    assert out["section_id"].dtype.kind in "iu"
    assert out["edges"].shape[0] == 2
    validate_inputs(out["X"], out["coords"], out["section_id"], out["edges"])


def test_st_blocks_must_have_spatial_and_counts_shape():
    """Missing spatial or shape mismatch in real ST block must fail early (contract)."""
    blocks = _tiny_st_blocks(n_per=4)
    bad = blocks[0].copy()
    bad["coords"] = bad["coords"][:3]  # mismatch
    with pytest.raises(ValueError, match="X.shape|coords.shape"):
        build_section_inputs([bad, blocks[1]])


def test_section_id_from_obs_pattern_like_real_cards():
    """Simulate the 'factorize obs column for section' pattern used in data cards."""
    rng = np.random.default_rng(11)
    n = 9
    X = rng.poisson(2, (n, 3)).astype(np.float32)
    coords = rng.random((n, 2)).astype(np.float32)
    # like 'slice_id' or 'fov' column -> codes
    sec_codes = np.array([0]*4 + [1]*5, dtype=np.int64)
    # build would take per block, but direct validate for stacked
    edges = np.empty((2, 0), dtype=np.int64)
    validate_inputs(X, coords, sec_codes, edges)
    assert np.unique(sec_codes).size == 2
