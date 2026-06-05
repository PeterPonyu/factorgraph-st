#!/usr/bin/env python3
"""Prepare the Maynard-2021 / spatialLIBD DLPFC dataset for the labeled path.

This is the ingestion entry point referenced by the processed-dataset manifest
(``data/processed/dlpfc_maynard_2021_visium/manifest.json``) and tracks
#133 / #355 / #347. It turns a downloaded spatialLIBD-format ``.h5ad`` into a
prepared ``anndata.h5ad`` whose ``obs['ground_truth_domain']`` carries the
manual per-spot cortical-layer call, so that running::

    python scripts/run_real_factorgraph.py --h5ad <prepared>/anndata.h5ad

emits the supervised ARI/NMI suite with ``ari_vs_gt_available == 1.0`` -- no
runner change needed (its GT auto-detect already binds to ``ground_truth_domain``).

Modes:

* ``--src IN.h5ad --out OUT.h5ad``  Prepare from an already-downloaded
  spatialLIBD h5ad: wire the layer labels and write the prepared object. This
  is the turnkey path the moment the real download lands. Network-free.
* ``--download``  Fetch the raw spatialLIBD release first. Currently DEFERRED:
  the spatialLIBD proxy is down (see the manifest ``next_steps``), so this
  raises ``NotImplementedError`` pointing at the tracking issues rather than
  inventing a URL or writing placeholder bytes -- exactly the
  ``scripts/data/fetch_datasets.py`` policy.

Per-spot ground-truth labels are copied verbatim from the upstream layer column
(:mod:`factorgraph_st.data.maynard`). Nothing is fabricated and no cross-donor
join is performed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# This script lives under scripts/data/; make the installed package importable
# when run directly without an editable install on sys.path.
_SRC = Path(__file__).resolve().parents[2] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from factorgraph_st.data.maynard import (  # noqa: E402  (after sys.path shim)
    CANONICAL_GT_KEY,
    load_spatiallibd_h5ad,
)

#: Tracking issues for the labeled DLPFC unblock.
_ISSUE_REFS = "#133, #355, #347"

#: Default processed-dataset output directory (parent orchestration repo layout:
#: this package lives at ``<parent>/factorgraph-st`` so data is one level up).
_DEFAULT_OUT = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "processed"
    / "dlpfc_maynard_2021_visium"
    / "anndata.h5ad"
)


def prepare(src: Path, out: Path, *, source_key: str | None = None) -> dict[str, object]:
    """Wire GT layers into ``src`` and write the prepared object to ``out``.

    Returns a small summary dict (counts + the source layer column used) so the
    caller / CLI can report what happened without re-reading the file.
    """
    import numpy as np

    adata = load_spatiallibd_h5ad(src, source_key=source_key)
    labels = adata.obs[CANONICAL_GT_KEY].astype(str).to_numpy()
    na_tokens = {"", "na", "nan", "none", "unknown", "unlabeled", "unlabelled"}
    labeled = np.array([s.strip().lower() not in na_tokens for s in labels], dtype=bool)
    n_labeled = int(labeled.sum())
    n_classes = int(np.unique(labels[labeled]).size) if n_labeled else 0

    out.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out)
    return {
        "out": str(out),
        "n_spots": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "n_labeled_spots": n_labeled,
        "n_layer_classes": n_classes,
        "gt_obs_key": CANONICAL_GT_KEY,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="prepare_dlpfc_maynard_2021.py",
        description="Prepare the Maynard-2021/spatialLIBD DLPFC labeled dataset "
        f"(tracks {_ISSUE_REFS}).",
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=None,
        help="Path to a downloaded spatialLIBD-format .h5ad to prepare from.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=_DEFAULT_OUT,
        help=f"Output prepared .h5ad path (default: {_DEFAULT_OUT}).",
    )
    parser.add_argument(
        "--source-key",
        type=str,
        default=None,
        help="Explicit upstream obs layer column (default: auto-detect "
        "spatialLIBD layer columns).",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Fetch the raw spatialLIBD release first (currently DEFERRED -- raises).",
    )
    args = parser.parse_args(argv)

    if args.download:
        raise NotImplementedError(
            "spatialLIBD download is DEFERRED (proxy down -- see the processed "
            f"manifest next_steps; tracked by {_ISSUE_REFS}). Restore network "
            "access to spatial.libd.org, then re-run with --src <downloaded.h5ad> "
            "--out <prepared.h5ad>. Never invent a URL or write placeholder bytes."
        )

    if args.src is None:
        parser.error("provide --src <downloaded spatialLIBD .h5ad> (or --download).")

    if not args.src.is_file():
        print(f"error: --src not found: {args.src}", file=sys.stderr)
        return 2

    summary = prepare(args.src.resolve(), args.out.resolve(), source_key=args.source_key)
    print(
        f"[ok] prepared {summary['out']}: {summary['n_spots']} spots, "
        f"{summary['n_genes']} genes, {summary['n_labeled_spots']} labeled "
        f"({summary['n_layer_classes']} layer classes) -> "
        f"obs[{summary['gt_obs_key']!r}]"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
