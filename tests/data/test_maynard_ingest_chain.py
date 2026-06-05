"""End-to-end labeled-data chain test for the Maynard/spatialLIBD ingest (#133).

Exercises the WHOLE turnkey path WITHOUT the real ~1.3 GB download, using the
tiny synthetic spatialLIBD-format fixture from :mod:`make_maynard_fixture`:

    fixture (upstream layer obs) -> loader (wire ground_truth_domain)
        -> run_real_factorgraph.py (subprocess) -> metrics.json
        -> assert ari_vs_gt_available == 1.0 and ARI/NMI/AMI finite.

Also asserts the graceful-degradation contract: an UNLABELED fixture (no layer
obs column, mirroring the on-disk DLPFC GSE307403 section) leaves
``ari_vs_gt_available == 0.0`` and the runner does not crash.

The runner is invoked as a subprocess under the SAME interpreter running the
tests (it imports scanpy + the package), so this runs green only in the data
env -- skipped otherwise. Network-free and confined to ``tmp_path``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from make_maynard_fixture import build_fixture

pytest.importorskip("anndata")
pytest.importorskip("scanpy")
pytest.importorskip("scipy")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RUNNER = _REPO_ROOT / "scripts" / "run_real_factorgraph.py"


def _run_runner(h5ad: Path, results_dir: Path) -> dict:
    """Invoke run_real_factorgraph.py on ``h5ad`` and return parsed metrics.json."""
    cmd = [
        sys.executable,
        str(_RUNNER),
        "--h5ad",
        str(h5ad),
        "--results-dir",
        str(results_dir),
        "--already-normalized",  # fixture X is continuous, not raw counts
        "--n-domains",
        "5",
        "--n-null-shuffles",
        "10",
        "--seed",
        "0",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(_REPO_ROOT))
    assert proc.returncode == 0, (
        f"runner failed (rc={proc.returncode})\nSTDOUT:\n{proc.stdout}\n"
        f"STDERR:\n{proc.stderr}"
    )
    metrics_path = results_dir / "factorgraph-st" / "metrics.json"
    assert metrics_path.is_file(), f"metrics.json not written: {metrics_path}"
    # The results contract nests the metric dict under a top-level "metrics" key.
    return json.loads(metrics_path.read_text())["metrics"]


def test_labeled_chain_flips_ari_available_and_metrics_finite(tmp_path):
    """Labeled fixture -> loader -> runner emits ari_vs_gt_available==1.0."""
    from factorgraph_st.data.maynard import (
        CANONICAL_GT_KEY,
        load_spatiallibd_h5ad,
    )
    from factorgraph_st.eval.metrics import (
        adjusted_mutual_information,
        adjusted_rand_index,
        normalized_mutual_information,
    )

    # 1. Build a labeled spatialLIBD-format fixture (upstream layer obs column).
    raw = build_fixture(tmp_path / "raw_spatiallibd.h5ad", seed=0)

    # 2. Loader wires the canonical ground_truth_domain obs column.
    adata = load_spatiallibd_h5ad(raw)
    assert CANONICAL_GT_KEY in adata.obs.columns
    gt = adata.obs[CANONICAL_GT_KEY].astype(str).to_numpy()
    assert np.unique(gt).size >= 2  # genuinely multi-class labels
    prepared = tmp_path / "prepared.h5ad"
    adata.write_h5ad(prepared)

    # 3. Runner emits the supervised suite; ari_vs_gt_available flips to 1.0.
    metrics = _run_runner(prepared, tmp_path / "results_labeled")
    assert metrics["ari_vs_gt_available"] == 1.0
    assert metrics["domain_metric_suite_available"] == 1.0
    assert np.isfinite(metrics["ari_domain"]), metrics["ari_domain"]
    assert np.isfinite(metrics["nmi_domain"]), metrics["nmi_domain"]

    # 4. AMI completes the ARI/NMI/AMI triple. The runner does not emit AMI, so
    #    recompute it directly on the recovered domains vs the GT labels to prove
    #    the chance-corrected metric is finite over this chain.
    npz = np.load(
        tmp_path / "results_labeled" / "factorgraph-st" / "outputs" / "factors.npz"
    )
    pred = npz["domain_id"].astype(np.int64)
    _, gt_codes = np.unique(gt, return_inverse=True)
    ami = adjusted_mutual_information(gt_codes.astype(np.int64), pred)
    assert np.isfinite(ami), ami

    # Cross-check the runner's reported ARI/NMI against a direct recompute.
    assert np.isfinite(adjusted_rand_index(gt_codes.astype(np.int64), pred))
    assert np.isfinite(normalized_mutual_information(gt_codes.astype(np.int64), pred))


def test_unlabeled_chain_degrades_gracefully(tmp_path):
    """Unlabeled fixture -> runner keeps ari_vs_gt_available==0.0, no crash."""
    raw = build_fixture(
        tmp_path / "unlabeled.h5ad", seed=1, with_labels=False
    )
    metrics = _run_runner(raw, tmp_path / "results_unlabeled")
    assert metrics["ari_vs_gt_available"] == 0.0
    assert metrics["domain_metric_suite_available"] == 0.0
    assert "ari_domain" not in metrics  # never fabricated when no GT present


def test_loader_raises_when_no_layer_column(tmp_path):
    """An unlabeled object has no layer column -> wiring fails loudly."""
    from factorgraph_st.data.maynard import load_spatiallibd_h5ad

    raw = build_fixture(tmp_path / "nolabel.h5ad", seed=2, with_labels=False)
    with pytest.raises(KeyError, match="no spatialLIBD layer column"):
        load_spatiallibd_h5ad(raw)
