#!/usr/bin/env python3
"""Unified dataset fetch/prepare framework for FactorGraph-ST.

This is the canonical ingestion entry point referenced by
``docs/DATASETS.md``. It exposes:

* :data:`DATASETS` — a machine-readable registry of every audited cohort
  (Tier A cancer collections, Tier B Python-native quick-load stack, the
  extended-validation cohorts, and the Wang 2025 crown-jewel matched
  multi-platform TMA).  Each entry records the accession, landing URL,
  citation key, the ``X / coords / section_id / edges`` contract mapping,
  the raw-count policy, and the GitHub issue(s) that track its ingestion.
* :func:`build_spatial_graph` / :func:`build_section_inputs` — helpers that
  turn a list of per-section ``AnnData`` objects into the model's
  ``X / coords / section_id / edges`` input contract (lazy ``squidpy`` /
  ``scipy`` imports; only needed for real assembly, never for ``--list`` /
  ``--dry-run``).
* a small argparse CLI: ``--list`` and ``--dry-run`` are completely
  network-free; ``--download`` is the only path that may touch the network.

Invariants (enforced by tests, see issue #52):

* **Never writes placeholder bytes.** A fetch only ever writes the genuine
  bytes returned by the source. URLs that have *not* been verified as
  direct, downloadable artifacts raise ``NotImplementedError`` pointing at
  the tracking issue — we never invent a URL or stub a file on disk.
* **Raw integer count matrices + spatial metadata only** — no WSIs, FASTQs,
  BAMs, or normalized-only objects (program policy).
* Large payloads are gated behind an explicit ``--download``; ``--dry-run``
  plans byte counts without touching the network.

No real download is executed unless a human passes ``--download`` for a
dataset whose direct URL is verified.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only
    import numpy as np

# Repo-local raw cache (git-ignored). Program-wide shared cache convention is
# ``~/Desktop/ST_research/data_cache/raw/<dataset_slug>/``.
RAW_ROOT = Path(__file__).resolve().parents[2] / "data" / "raw"
CARDS_ROOT = Path(__file__).resolve().parents[2] / "data" / "cards"

GB = 1024**3
MB = 1024**2


@dataclass(frozen=True)
class Dataset:
    """One registry entry: accession + contract mapping + provenance.

    ``url_verified`` records whether a *direct, downloadable* artifact URL has
    been confirmed. ``landing_url`` is always the human landing page. When
    ``url_verified`` is ``False`` the fetch path raises ``NotImplementedError``
    referencing ``issues`` — we never invent a direct URL or write placeholder
    bytes.
    """

    id: str
    title: str
    platform: str
    tier: str  # "A" | "B" | "extended"
    accession: str
    landing_url: str
    citation_key: str
    issues: tuple[int, ...]
    raw_count_policy: str
    contract: dict[str, str]
    source_kind: str  # "squidpy" | "zenodo" | "geo" | "figshare" | "hf"
    url_verified: bool = False
    download_url: str | None = None
    approx_bytes: int | None = None
    card: str | None = None
    notes: str = ""

    @property
    def issue_refs(self) -> str:
        return ", ".join(f"#{n}" for n in self.issues)


def _human_bytes(n: int | None) -> str:
    if n is None:
        return "unknown size"
    if n >= GB:
        return f"{n / GB:.1f} GB"
    if n >= MB:
        return f"{n / MB:.0f} MB"
    return f"{n} B"


# --------------------------------------------------------------------------- #
# Registry — every audited cohort.  Sizes mirror docs/DATASETS.md and the
# program registry (ST_research/datasets/DATASET_REGISTRY.md, audited 05-28/29).
# Direct file URLs are intentionally left unverified (url_verified=False) unless
# the source is a programmatic squidpy loader: Zenodo/GEO/figshare/HF record
# pages are landing pages, not confirmed direct artifacts, so guessing a file
# URL would violate the "never invent URLs" policy.
# --------------------------------------------------------------------------- #
_CONTRACT_VISIUM = {
    "X": "(n_spots, n_genes) float32 raw counts from filtered_feature_bc_matrix.h5",
    "coords": "(n_spots, 2) float32 spot centroids from outs/spatial/",
    "section_id": "int per Visium section (capture area)",
    "edges": "(2, n_edges) int64 hex-grid kNN via build_spatial_graph(coord_type='grid')",
}
_CONTRACT_IMAGING = {
    "X": "(n_cells, n_genes) float32 raw counts from cell_feature_matrix / exprMat / cell_by_gene",
    "coords": "(n_cells, 2) float32 cell centroids from cell_metadata / boundaries",
    "section_id": "int per (platform, section/TMA, tissue)",
    "edges": "(2, n_edges) int64 generic kNN via build_spatial_graph(coord_type='generic')",
}

DATASETS: dict[str, Dataset] = {
    d.id: d
    for d in (
        # ---- Tier A — production cancer cohorts -------------------------- #
        Dataset(
            id="cervilla_2026_visium",
            title="Cervilla 2026 — Visium / Visium HD (6 cancer types)",
            platform="Visium v1 / CytAssist / HD",
            tier="A",
            accession="Zenodo 17999961",
            landing_url="https://zenodo.org/records/17999961",
            citation_key="cervilla2026",
            issues=(32,),
            raw_count_policy="filtered_feature_bc_matrix.h5 (Space Ranger); HD bins; raw int counts",
            contract=_CONTRACT_VISIUM,
            source_kind="zenodo",
            approx_bytes=26 * GB,
            notes="Crown-jewel matched-cohort Visium arm (with #33/#34). Pull matched samples first.",
        ),
        Dataset(
            id="cervilla_2026_xenium_cosmx",
            title="Cervilla 2026 — Xenium & CosMx (6 cancer types)",
            platform="Xenium + CosMx",
            tier="A",
            accession="Zenodo 17986017",
            landing_url="https://zenodo.org/records/17986017",
            citation_key="cervilla2026",
            issues=(33,),
            raw_count_policy="Xenium cell_feature_matrix.h5; CosMx *_exprMat_file.csv.gz; raw int counts",
            contract=_CONTRACT_IMAGING,
            source_kind="zenodo",
            approx_bytes=int(23.6 * GB),
            notes="Imaging-ST arm of the matched Cervilla tumours.",
        ),
        Dataset(
            id="cervilla_2026_xenium_mt_5k",
            title="Cervilla 2026 — Xenium Multi-Tissue & 5K (6 cancer types)",
            platform="Xenium Multi-Tissue + Human Prime 5K",
            tier="A",
            accession="Zenodo 18000256",
            landing_url="https://zenodo.org/records/18000256",
            citation_key="cervilla2026",
            issues=(34,),
            raw_count_policy="cell_feature_matrix.h5 per panel (377-gene MT + 5K); raw int counts",
            contract=_CONTRACT_IMAGING,
            source_kind="zenodo",
            approx_bytes=int(46.8 * GB),
            notes="High-plex (5K) imaging arm; completes the three-platform matched set.",
        ),
        Dataset(
            id="dawo_tls_visium",
            title="Dawo TLS (Kidney + Lung Visium)",
            platform="Visium v1",
            tier="A",
            accession="Zenodo 14620362",
            landing_url="https://zenodo.org/records/14620362",
            citation_key="dawo_tls",
            issues=(35,),
            raw_count_policy="10x_Visium/<sample>/raw_feature_bc_matrix/ + spatial/; 8 sections; raw int counts",
            contract=_CONTRACT_VISIUM,
            source_kind="zenodo",
            approx_bytes=int(2.1 * GB),
            notes="8 cancer Visium sections, two organs + TLS annotation. Small/fast first integration.",
        ),
        Dataset(
            id="gse175540_rcc_visium",
            title="GSE175540 (RCC Visium, 24 sections FFPE+FF)",
            platform="Visium v1",
            tier="A",
            accession="GEO GSE175540",
            landing_url="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE175540",
            citation_key="gse175540_rcc",
            issues=(36,),
            raw_count_policy="per-GSM *_filtered_feature_bc_matrix.h5 inside GSE175540_RAW.tar; raw int counts",
            contract=_CONTRACT_VISIUM,
            source_kind="geo",
            approx_bytes=1 * GB,
            notes="Within-disease, cross-preservation (FFPE vs FF) section set.",
        ),
        Dataset(
            id="hest1k_cancer_subset",
            title="HEST-1k cancer Visium subset",
            platform="Visium v1/v2 + legacy ST",
            tier="A",
            accession="HF MahmoodLab/hest",
            landing_url="https://huggingface.co/datasets/MahmoodLab/hest",
            citation_key="hest1k",
            issues=(37,),
            raw_count_policy="st/*.h5ad -> adata.X raw counts; assert integer at load",
            contract=_CONTRACT_VISIUM,
            source_kind="hf",
            approx_bytes=10 * GB,
            notes="Take a cancer Visium subset spanning many organs for section diversity.",
        ),
        Dataset(
            id="xenium_breast_janesick",
            title="Xenium Breast Cancer (Janesick)",
            platform="Xenium",
            tier="A",
            accession="10x Xenium_V1_human_Breast demo (GEO GSE243280 mirror)",
            landing_url="https://www.10xgenomics.com/products/xenium-in-situ/preview-dataset-human-breast",
            citation_key="janesick2023",
            issues=(38,),
            raw_count_policy="*_cell_feature_matrix.h5 (Rep1/Rep2); raw int counts",
            contract=_CONTRACT_IMAGING,
            source_kind="geo",
            approx_bytes=int(1.5 * GB),
            notes="Canonical single-cell-resolution breast Xenium (313-gene panel).",
        ),
        Dataset(
            id="gse293199_tnbc_xenium",
            title="GSE293199 (TNBC Xenium)",
            platform="Xenium",
            tier="A",
            accession="GEO GSE293199",
            landing_url="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE293199",
            citation_key="gse293199_tnbc",
            issues=(39,),
            raw_count_policy="cell_feature_matrix.h5 inside nested *_Xenium_outs.tar.gz in GSE293199_RAW.tar",
            contract=_CONTRACT_IMAGING,
            source_kind="geo",
            approx_bytes=int(14.6 * GB),
            notes="Single ~14.6 GB nested tarball (no trivial 3 GB subset). OmiCLIP source, 280-gene panel.",
        ),
        # ---- Tier B — Python-native quick-load --------------------------- #
        Dataset(
            id="dlpfc_maynard_2021",
            title="DLPFC 12-section Visium (Maynard 2021 / spatialLIBD)",
            platform="Visium v1",
            tier="B",
            accession="figshare 22004273 (DOI 10.6084/m9.figshare.22004273.v2); spatialLIBD",
            landing_url="https://figshare.com/articles/dataset/22004273",
            citation_key="maynard2021",
            issues=(41, 133),
            raw_count_policy=(
                "figshare X is normalized (raw UMIs in .raw); prefer spatialLIBD::fetch_data('spe') "
                "counts assay or per-sample h5_raw Space Ranger matrices for raw int counts (#41)"
            ),
            contract=_CONTRACT_VISIUM,
            source_kind="figshare",
            approx_bytes=int(1.3 * GB),
            card="dlpfc_maynard_2021.yaml",
            notes="Primary multi-section domain benchmark: 12 sections, 3 donors, manual layer labels.",
        ),
    )
}


# --------------------------------------------------------------------------- #
# Multi-section assembly helpers (lazy squidpy / scipy imports).
# These map a list of per-section AnnData onto X / coords / section_id / edges.
# --------------------------------------------------------------------------- #
def build_spatial_graph(
    adata: Any,
    n_neighs: int = 6,
    coord_type: str = "grid",
) -> "np.ndarray":
    """Build a single-section spatial kNN graph as ``(2, n_edges)`` int64 COO.

    Visium uses ``coord_type='grid'`` (hex grid); imaging platforms (Xenium,
    MERSCOPE, CosMx, MERFISH, legacy ST) use ``coord_type='generic'``.
    Imports ``squidpy`` / ``scipy`` lazily so the registry/CLI stay
    import-light and network-free.
    """
    import numpy as np
    import squidpy as sq

    sq.gr.spatial_neighbors(adata, coord_type=coord_type, n_neighs=n_neighs)
    conn = adata.obsp["spatial_connectivities"].tocoo()
    edges = np.vstack([conn.row, conn.col]).astype(np.int64)
    return edges


def build_section_inputs(
    sections: Sequence[Any],
    n_neighs: int = 6,
    coord_type: str = "grid",
    use_raw: bool = True,
) -> dict[str, "np.ndarray"]:
    """Stack per-section ``AnnData`` into the MVP input contract.

    Returns a dict with ``X`` (stacked raw counts, ``float32``), ``coords``
    (``float32``), ``section_id`` (contiguous integer per section), and
    ``edges`` (``int64`` COO, node indices offset to be global across the
    concatenated stack). Imports ``numpy`` / ``scipy`` lazily.
    """
    import numpy as np
    from scipy.sparse import issparse

    if len(sections) == 0:
        raise ValueError("sections must be a non-empty sequence")

    x_chunks: list[np.ndarray] = []
    coord_chunks: list[np.ndarray] = []
    sec_chunks: list[np.ndarray] = []
    edge_chunks: list[np.ndarray] = []
    offset = 0
    for i, adata in enumerate(sections):
        source = adata.raw if (use_raw and adata.raw is not None) else adata
        x = source.X
        x = x.toarray() if issparse(x) else np.asarray(x)
        x = np.ascontiguousarray(x, dtype=np.float32)
        coords = np.ascontiguousarray(
            np.asarray(adata.obsm["spatial"]), dtype=np.float32
        )
        if x.shape[0] != coords.shape[0]:
            raise ValueError(
                f"sections[{i}] X rows {x.shape[0]} != coords rows {coords.shape[0]}"
            )
        n = x.shape[0]
        x_chunks.append(x)
        coord_chunks.append(coords)
        sec_chunks.append(np.full(n, i, dtype=np.int64))
        edges = build_spatial_graph(adata, n_neighs=n_neighs, coord_type=coord_type)
        edge_chunks.append(edges + offset)
        offset += n

    return {
        "X": np.concatenate(x_chunks, axis=0),
        "coords": np.concatenate(coord_chunks, axis=0),
        "section_id": np.concatenate(sec_chunks, axis=0),
        "edges": (
            np.concatenate(edge_chunks, axis=1)
            if edge_chunks
            else np.empty((2, 0), dtype=np.int64)
        ),
    }


# --------------------------------------------------------------------------- #
# Network layer — only ever called with a verified direct URL.
# --------------------------------------------------------------------------- #
def _http_download(url: str, dest: Path, *, chunk: int = 1 << 20) -> int:
    """Stream ``url`` to ``dest``; return bytes written.

    Refuses to create a placeholder: if the response yields zero bytes the
    partial file is removed and an error is raised. Only invoked for datasets
    whose direct download URL has been verified.
    """
    from urllib.request import urlopen

    dest.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urlopen(url) as resp, open(tmp, "wb") as fh:  # noqa: S310 (verified URL)
        while True:
            block = resp.read(chunk)
            if not block:
                break
            fh.write(block)
            written += len(block)
    if written == 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"refusing to write placeholder: {url} returned 0 bytes")
    tmp.rename(dest)
    return written


def _fetch_squidpy(ds: Dataset, dest: Path) -> None:
    """Materialize a squidpy-backed dataset via its programmatic loader."""
    import squidpy as sq

    dest.mkdir(parents=True, exist_ok=True)
    if ds.id == "visium_mouse_brain_pair":
        hne = sq.datasets.visium_hne_adata()
        fluo = sq.datasets.visium_fluo_adata()
        hne.write_h5ad(dest / "visium_hne.h5ad")
        fluo.write_h5ad(dest / "visium_fluo.h5ad")
    elif ds.id == "merfish_hypothalamus_shared":
        adata = sq.datasets.merfish()
        adata.write_h5ad(dest / "merfish.h5ad")
    else:  # pragma: no cover - defensive
        raise NotImplementedError(f"no squidpy loader wired for {ds.id}")


def fetch(ds: Dataset, *, download: bool = False, dry_run: bool = False) -> int:
    """Plan or perform a fetch for one dataset.

    With ``dry_run`` (or without ``download``) this only prints the plan and
    touches no network. With ``download`` it materializes squidpy-backed
    datasets and downloads verified direct URLs; unverified URLs raise
    ``NotImplementedError`` referencing the tracking issue.
    """
    dest = RAW_ROOT / ds.id
    plan = (
        f"[plan] {ds.id}: {ds.title}\n"
        f"        platform={ds.platform} | accession={ds.accession}\n"
        f"        size~={_human_bytes(ds.approx_bytes)} | dest={dest}\n"
        f"        landing={ds.landing_url}\n"
        f"        raw-count policy: {ds.raw_count_policy}\n"
        f"        tracked by: {ds.issue_refs}"
    )
    print(plan)
    if dry_run or not download:
        return 0

    if ds.source_kind == "squidpy":
        _fetch_squidpy(ds, dest)
        print(f"[ok] {ds.id} materialized via squidpy -> {dest}")
        return 0
    if ds.url_verified and ds.download_url:
        n = _http_download(ds.download_url, dest / Path(ds.download_url).name)
        print(f"[ok] {ds.id} downloaded {_human_bytes(n)} -> {dest}")
        return 0
    raise NotImplementedError(
        f"{ds.id}: direct download URL UNVERIFIED — see issue {ds.issue_refs}. "
        f"Resolve the artifact URL on {ds.landing_url} before enabling download; "
        f"never invent a URL or write placeholder bytes."
    )


def print_list() -> None:
    """Print every registry entry. Network-free."""
    by_tier: dict[str, list[Dataset]] = {}
    for ds in DATASETS.values():
        by_tier.setdefault(ds.tier, []).append(ds)
    labels = {"A": "Tier A (production cancer)", "B": "Tier B (quick-load)", "extended": "Extended validation"}
    for tier in ("A", "B", "extended"):
        items = by_tier.get(tier, [])
        if not items:
            continue
        print(f"\n== {labels.get(tier, tier)} ==")
        for ds in items:
            flag = "verified-loader" if ds.url_verified else "URL UNVERIFIED"
            print(
                f"  {ds.id:<32} {ds.platform:<28} {_human_bytes(ds.approx_bytes):>10}"
                f"  [{flag}]  ({ds.issue_refs})"
            )
    print(f"\n{len(DATASETS)} datasets. Use --dataset <id> --dry-run to plan a fetch.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fetch_datasets.py",
        description="FactorGraph-ST dataset fetch/prepare framework (registry-driven).",
    )
    parser.add_argument("--list", action="store_true", help="list all datasets and exit (network-free)")
    parser.add_argument("--dataset", metavar="ID", help="dataset id to fetch/plan")
    parser.add_argument("--dry-run", action="store_true", help="plan the fetch without touching the network")
    parser.add_argument(
        "--download",
        action="store_true",
        help="actually download (gated; only verified direct URLs / squidpy loaders)",
    )
    args = parser.parse_args(argv)

    if args.list or not args.dataset:
        if not args.list and not args.dataset:
            parser.print_help()
            return 0
        print_list()
        return 0

    if args.dataset not in DATASETS:
        print(f"error: unknown dataset '{args.dataset}'. Use --list to see ids.", file=sys.stderr)
        return 2

    return fetch(DATASETS[args.dataset], download=args.download, dry_run=args.dry_run)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
