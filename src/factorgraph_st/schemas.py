"""Schema validators for FactorGraph-ST MVP inputs and outputs.

Mirrors the contract in ``docs/MVP_DESIGN.md``. The key FG-specific invariant
is nonnegativity of the gene loading ``W`` and the factor activations
``Z_shared`` / ``Z_private``.
"""

from __future__ import annotations

import numpy as np


class SchemaError(ValueError):
    """Raised when an input or output object violates the MVP schema."""


def validate_inputs(
    X: np.ndarray,
    coords: np.ndarray,
    section_id: np.ndarray,
    edges: np.ndarray,
) -> None:
    """Validate MVP inputs.

    Raises ``SchemaError`` on any violation. Returns ``None`` on success.
    """
    if X.ndim != 2:
        raise SchemaError(f"X must be 2D; got ndim={X.ndim}")
    if X.dtype != np.float32:
        raise SchemaError(f"X must be float32; got {X.dtype}")
    if not np.isfinite(X).all():
        raise SchemaError("X must be finite")
    n_spots = X.shape[0]

    if coords.ndim != 2 or coords.shape[1] != 2:
        raise SchemaError(f"coords must be (n_spots, 2); got shape={coords.shape}")
    if coords.dtype != np.float32:
        raise SchemaError(f"coords must be float32; got {coords.dtype}")
    if not np.isfinite(coords).all():
        raise SchemaError("coords must be finite")
    if coords.shape[0] != n_spots:
        raise SchemaError(f"coords n_spots={coords.shape[0]} != X n_spots={n_spots}")

    if section_id.ndim != 1 or section_id.shape[0] != n_spots:
        raise SchemaError(f"section_id must be (n_spots,); got shape={section_id.shape}")
    if section_id.dtype.kind not in ("i", "u"):
        raise SchemaError(f"section_id must be integer dtype; got {section_id.dtype}")

    if edges.ndim != 2 or edges.shape[0] != 2:
        raise SchemaError(f"edges must be (2, n_edges); got shape={edges.shape}")
    if edges.dtype != np.int64:
        raise SchemaError(f"edges must be int64; got {edges.dtype}")
    if edges.size and (edges.min() < 0 or edges.max() >= n_spots):
        raise SchemaError("edges contain indices outside [0, n_spots)")


def validate_outputs(
    H: np.ndarray,
    W: np.ndarray,
    Z_shared: np.ndarray,
    Z_private: np.ndarray,
    domain_id: np.ndarray,
    n_spots: int,
    n_genes: int,
) -> None:
    """Validate MVP outputs, including nonnegativity of factor matrices."""
    if H.ndim != 2 or H.shape[0] != n_spots:
        raise SchemaError(f"H must be (n_spots, d); got shape={H.shape}")
    if H.dtype != np.float32:
        raise SchemaError(f"H must be float32; got {H.dtype}")

    if W.ndim != 2 or W.shape[0] != n_genes:
        raise SchemaError(f"W must be (n_genes, K); got shape={W.shape}")
    if W.dtype != np.float32:
        raise SchemaError(f"W must be float32; got {W.dtype}")
    if not np.isfinite(W).all():
        raise SchemaError("W must be finite")
    if (W < 0).any():
        raise SchemaError("W must be nonnegative")

    if Z_shared.ndim != 2 or Z_shared.shape[0] != n_spots:
        raise SchemaError(f"Z_shared must be (n_spots, K_shared); got shape={Z_shared.shape}")
    if Z_shared.dtype != np.float32:
        raise SchemaError(f"Z_shared must be float32; got {Z_shared.dtype}")
    if not np.isfinite(Z_shared).all():
        raise SchemaError("Z_shared must be finite")
    if (Z_shared < 0).any():
        raise SchemaError("Z_shared must be nonnegative")

    if Z_private.ndim != 2 or Z_private.shape[0] != n_spots:
        raise SchemaError(f"Z_private must be (n_spots, K_private); got shape={Z_private.shape}")
    if Z_private.dtype != np.float32:
        raise SchemaError(f"Z_private must be float32; got {Z_private.dtype}")
    if not np.isfinite(Z_private).all():
        raise SchemaError("Z_private must be finite")
    if (Z_private < 0).any():
        raise SchemaError("Z_private must be nonnegative")

    K = Z_shared.shape[1] + Z_private.shape[1]
    if W.shape[1] != K:
        raise SchemaError(
            f"W columns {W.shape[1]} != K_shared + K_private = {K}"
        )

    if domain_id.ndim != 1 or domain_id.shape[0] != n_spots:
        raise SchemaError(f"domain_id must be (n_spots,); got shape={domain_id.shape}")
    if domain_id.dtype != np.int64:
        raise SchemaError(f"domain_id must be int64; got {domain_id.dtype}")
    if domain_id.size and domain_id.min() < 0:
        raise SchemaError("domain_id must be non-negative")
