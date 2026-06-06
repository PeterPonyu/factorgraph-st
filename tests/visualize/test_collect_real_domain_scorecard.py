"""Unit tests for the real-domain scorecard collector (#322 #330).

Pure aggregation helpers run in the numpy-only env; the end-to-end ``collect``
(which reads an AnnData) is exercised against a tiny on-disk fixture so the
metrics.json -> scores.json / accuracy_results.json bridge is covered without the
heavy runner.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "visualize"
    / "collect_real_domain_scorecard.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("_collect_real_domain_scorecard", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_parse_run_dirname():
    mod = _load_module()
    assert mod.parse_run_dirname("gnmf_seed0") == ("gnmf", 0)
    assert mod.parse_run_dirname("spatial_smooth_seed12") == ("spatial_smooth", 12)
    assert mod.parse_run_dirname("_scorecard") is None
    assert mod.parse_run_dirname("gnmf") is None


def test_pairwise_stability_identical_is_one():
    mod = _load_module()
    a = np.array([0, 0, 1, 1, 2, 2])
    assert mod.pairwise_stability([a, a.copy(), a.copy()]) == pytest.approx(1.0)


def test_pairwise_stability_single_run_is_none():
    mod = _load_module()
    assert mod.pairwise_stability([np.array([0, 1, 2])]) is None
    assert mod.pairwise_stability([]) is None


def test_labeled_gt_codes_drops_na_tokens():
    mod = _load_module()
    valid, codes = mod.labeled_gt_codes(["L1", "L2", "unknown", "", "L1"])
    assert valid.tolist() == [True, True, False, False, True]
    # codes are over the valid subset only (3 spots, 2 classes), L1 == L1.
    assert codes.shape == (3,)
    assert codes[0] == codes[2]
    assert codes[0] != codes[1]


def test_summarize_method_means_and_stability():
    mod = _load_module()
    gt_valid = np.array([True, True, True, True])
    gt_codes = np.array([0, 0, 1, 1])
    # Two seeds: seed0 perfect, seed1 swaps two spots.
    metrics_list = [
        {"coherence_label_invariant_domain_delta": 0.8},
        {"coherence_label_invariant_domain_delta": 0.6},
    ]
    label_arrays = [np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1])]
    summary = mod.summarize_method(metrics_list, label_arrays, gt_valid, gt_codes)
    assert summary["coherence"] == pytest.approx(0.7)  # mean(0.8, 0.6)
    assert summary["n_seeds"] == 2.0
    assert summary["stability"] is not None and 0.0 <= summary["stability"] <= 1.0
    # ari is the mean of the per-seed ARI vs GT; seed0 is perfect (1.0).
    assert 0.0 <= summary["ari"] <= 1.0


def test_build_scores_omits_single_seed_methods():
    mod = _load_module()
    summaries = {
        "gnmf": {"coherence": 0.2, "stability": 0.45, "ari": 0.18, "nmi": 0.2, "ami": 0.2, "n_seeds": 4.0},
        "oneoff": {"coherence": 0.9, "stability": None, "ari": 0.5, "nmi": 0.5, "ami": 0.5, "n_seeds": 1.0},
    }
    scores = mod.build_scores(summaries)
    assert set(scores) == {"gnmf"}  # single-seed method dropped (no fabricated stability)
    assert scores["gnmf"] == {"coherence": 0.2, "stability": 0.45, "ari": 0.18}


def test_build_accuracy_results_shape():
    mod = _load_module()
    summaries = {
        "gnmf": {"coherence": 0.2, "stability": 0.45, "ari": 0.18, "nmi": 0.23, "ami": 0.21, "n_seeds": 4.0},
    }
    res = mod.build_accuracy_results("starmap", summaries)
    assert res == {"starmap": {"gnmf": {"ari": 0.18, "nmi": 0.23, "ami": 0.21}}}


def _write_run(root: Path, method: str, seed: int, domain_id: np.ndarray, coherence_delta: float):
    run = root / f"{method}_seed{seed}" / "factorgraph-st"
    (run / "outputs").mkdir(parents=True, exist_ok=True)
    (run / "metrics.json").write_text(
        json.dumps({"metrics": {"coherence_label_invariant_domain_delta": coherence_delta}}),
        encoding="utf-8",
    )
    np.savez(run / "outputs" / "factors.npz", domain_id=domain_id.astype(np.int64))


def test_collect_end_to_end(tmp_path):
    anndata = pytest.importorskip("anndata")
    mod = _load_module()

    gt = np.array(["A", "A", "A", "B", "B", "B"])
    adata = anndata.AnnData(
        X=np.zeros((6, 2), dtype=np.float32),
        obs={"ground_truth": gt},
    )
    adata.obsm["spatial"] = np.zeros((6, 2), dtype=np.float32)
    h5ad = tmp_path / "data.h5ad"
    adata.write_h5ad(h5ad)

    runs = tmp_path / "runs"
    perfect = np.array([0, 0, 0, 1, 1, 1])
    _write_run(runs, "gnmf", 0, perfect, 0.5)
    _write_run(runs, "gnmf", 1, perfect, 0.7)

    scores, accuracy, summaries = mod.collect(runs, h5ad, "ground_truth", "toy")
    assert set(scores) == {"gnmf"}
    assert scores["gnmf"]["ari"] == pytest.approx(1.0)  # perfect partition vs GT
    assert scores["gnmf"]["coherence"] == pytest.approx(0.6)
    assert scores["gnmf"]["stability"] == pytest.approx(1.0)  # identical across seeds
    assert accuracy["toy"]["gnmf"]["ami"] == pytest.approx(1.0)
    assert summaries["gnmf"]["n_seeds"] == 2.0
