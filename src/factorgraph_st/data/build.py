"""Stack per-section expression blocks into a validated MVP input dict.

Closes #71: callers previously had to densify and cast ``X`` themselves
before passing it to ``encode_graph`` / ``fit_transform``, because the
dense-``float32``-finite contract advertised by ``docs/MVP_DESIGN.md``
was not enforced at the data-bridge boundary. Real Visium / AnnData loads
return SciPy CSR / CSC matrices with ``int`` or ``float64`` dtype, so any
first real-data run failed at ``validate_inputs`` or silently broke on
``np.isfinite(sparse).all()``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from factorgraph_st.schemas import validate_inputs


def build_section_inputs(
    blocks: Sequence[Mapping[str, Any]],
    *,
    edges: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Stack per-section blocks into a validated MVP input dict.

    Each block must provide ``"X"`` (dense ``ndarray`` or any SciPy-sparse
    matrix-like exposing ``.toarray()``; arbitrary dtype) and ``"coords"``
    ``(n_spots, 2)``. ``section_id`` defaults to the block index. The
    returned dict satisfies ``validate_inputs``: dense ``float32`` finite
    ``X``, ``float32`` ``coords``, ``int64`` ``section_id`` / ``edges``.
    """
    if len(blocks) == 0:
        raise ValueError("blocks must be a non-empty sequence")
    x_chunks, coord_chunks, sec_chunks = [], [], []
    for i, b in enumerate(blocks):
        if "X" not in b or "coords" not in b:
            raise ValueError(f"blocks[{i}] missing required key 'X' or 'coords'")
        x = b["X"].toarray() if hasattr(b["X"], "toarray") else np.asarray(b["X"])
        x = np.ascontiguousarray(x, dtype=np.float32)
        coords = np.ascontiguousarray(np.asarray(b["coords"]), dtype=np.float32)
        if x.ndim != 2 or coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(
                f"blocks[{i}] expects 2D X and (n_spots, 2) coords; "
                f"got X.shape={x.shape}, coords.shape={coords.shape}"
            )
        if x.shape[0] != coords.shape[0]:
            raise ValueError(
                f"blocks[{i}] X.shape[0]={x.shape[0]} != coords.shape[0]={coords.shape[0]}"
            )
        x_chunks.append(x)
        coord_chunks.append(coords)
        sec_chunks.append(np.full(x.shape[0], int(b.get("section_id", i)), dtype=np.int64))

    X = np.concatenate(x_chunks, axis=0)
    if not np.isfinite(X).all():
        raise ValueError("stacked X contains non-finite entries after densification")
    out = {
        "X": X,
        "coords": np.concatenate(coord_chunks, axis=0),
        "section_id": np.concatenate(sec_chunks, axis=0),
        "edges": (
            np.empty((2, 0), dtype=np.int64)
            if edges is None
            else np.ascontiguousarray(np.asarray(edges), dtype=np.int64)
        ),
    }
    validate_inputs(out["X"], out["coords"], out["section_id"], out["edges"])
    return out
