"""spatialLIBD / Maynard-2021 DLPFC loader + ground-truth layer wiring.

Closes the labeled-data half of #133 / #355 / #347: makes the supervised
ARI/NMI/AMI path turnkey so that the moment the Maynard-2021 spatialLIBD DLPFC
download lands, :mod:`scripts.run_real_factorgraph` emits the supervised
domain-quality suite with ``ari_vs_gt_available == 1.0``.

The spatialLIBD release annotates each Visium spot with a manual cortical
*layer* call (``Layer1``..``Layer6`` + ``WM``) under one of several obs column
names (``layer_guess``, ``layer_guess_reordered_short`` ...). The runner's GT
auto-detect (``scripts/run_real_factorgraph.py:_GT_OBS_KEY_CANDIDATES``) keys off
``ground_truth_domain`` (and ``layer_guess``). This module normalizes whatever
upstream layer column is present into the canonical ``ground_truth_domain`` obs
column so the runner's supervised branch flips on without guessing.

HONESTY GUARANTEE: labels are copied verbatim from the upstream layer column.
Nothing is fabricated and no cross-donor join is performed; spots the upstream
prep left blank/NA stay blank (the runner drops them from both partitions).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from anndata import AnnData

#: Canonical obs column the runner's GT auto-detect binds to first. Writing the
#: layer call here makes the supervised ARI/NMI/AMI branch turnkey.
CANONICAL_GT_KEY = "ground_truth_domain"

#: Upstream spatialLIBD obs columns that may carry the manual per-spot layer
#: call, in priority order. ``layer_guess_reordered_short`` is the short
#: (``L1``..``WM``) form; ``layer_guess`` is the canonical long name; the
#: remaining are observed variants across spatialLIBD release vintages.
SPATIALLIBD_LAYER_KEYS: tuple[str, ...] = (
    "layer_guess_reordered_short",
    "layer_guess_reordered",
    "layer_guess",
    "spatialLIBD",
    "Layer",
    "layer",
)


def resolve_layer_key(adata: AnnData, source_key: str | None = None) -> str | None:
    """Return the obs column holding the manual layer call, or ``None``.

    An explicit ``source_key`` that is missing fails loudly (no silent
    fallback); auto-detection scans :data:`SPATIALLIBD_LAYER_KEYS` in order and
    returns ``None`` when none are present (caller decides whether that is an
    error or a graceful skip).
    """
    if source_key is not None:
        if source_key not in adata.obs.columns:
            raise KeyError(
                f"source_key={source_key!r} not found in adata.obs "
                f"(available: {list(adata.obs.columns)})"
            )
        return source_key
    for key in SPATIALLIBD_LAYER_KEYS:
        if key in adata.obs.columns:
            return key
    return None


def wire_ground_truth_layers(
    adata: AnnData, *, source_key: str | None = None
) -> str:
    """Copy the upstream layer call into ``obs['ground_truth_domain']`` in place.

    Resolves the source layer column (explicit ``source_key`` or auto-detect
    over :data:`SPATIALLIBD_LAYER_KEYS`) and writes it verbatim as a string
    dtype to :data:`CANONICAL_GT_KEY`. Returns the source column name used.

    Raises ``KeyError`` when no usable layer column is found -- the prepare path
    must not silently produce a label-less object and call it "labeled".
    """
    key = resolve_layer_key(adata, source_key)
    if key is None:
        raise KeyError(
            "no spatialLIBD layer column found in adata.obs "
            f"(looked for {list(SPATIALLIBD_LAYER_KEYS)}); "
            f"available columns: {list(adata.obs.columns)}"
        )
    # Verbatim copy as string: never recode, never fabricate. NA stays NA so the
    # runner's GT guard drops unlabeled spots from both partitions.
    adata.obs[CANONICAL_GT_KEY] = adata.obs[key].astype(str).to_numpy()
    return key


def load_spatiallibd_h5ad(
    path: str | Path, *, source_key: str | None = None
) -> AnnData:
    """Load a spatialLIBD-format DLPFC ``.h5ad`` with GT layers wired in.

    Reads ``path`` via :func:`anndata.read_h5ad`, validates that spatial
    coordinates are present (``obsm['spatial']``), and wires the manual layer
    call into ``obs['ground_truth_domain']`` via :func:`wire_ground_truth_layers`
    so the runner's supervised branch turns on. ``anndata`` is imported lazily so
    importing this module stays dependency-light.
    """
    from anndata import read_h5ad

    adata = read_h5ad(Path(path))
    if "spatial" not in adata.obsm:
        raise KeyError(
            "spatialLIBD h5ad missing obsm['spatial'] spot coordinates "
            f"(obsm keys: {list(adata.obsm.keys())})"
        )
    wire_ground_truth_layers(adata, source_key=source_key)
    return adata
