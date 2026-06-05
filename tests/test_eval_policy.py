"""Tests for the GT-metric emission policy (#341).

Two layers, all synthetic and seeded:

* PURE policy unit tests (no heavy deps) over
  :func:`factorgraph_st.eval.policy.resolve_eval_policy` /
  :func:`~factorgraph_st.eval.policy.infer_dataset_class`: class-A datasets emit
  GT metrics; non-A / GT-absent datasets suppress them; the presence-based guard
  is a strict subset of the class gate (forced class-A on a label-less dataset
  still suppresses); determinism.

* END-TO-END runner tests (skipped if scanpy/anndata are absent) that run
  ``scripts/run_real_factorgraph.py`` on a tiny synthetic ``.h5ad`` and assert
  the emitted ``metrics.json`` / ``run_metadata.json`` honor the policy:
  internal/label-free metrics are ALWAYS present; GT-based metrics appear ONLY
  for class A; the policy marker is recorded.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from factorgraph_st.eval.policy import (
    DATASET_CLASSES,
    EvalPolicy,
    infer_dataset_class,
    resolve_eval_policy,
)

_RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "run_real_factorgraph.py"

# Internal / label-free metrics that must ALWAYS be emitted (default-on).
_INTERNAL_KEYS = (
    "morans_i_domain",
    "morans_i_domain_null",
    "morans_i_domain_delta",
    "coherence_label_invariant_domain",
    "coherence_label_invariant_domain_null",
    "coherence_label_invariant_domain_delta",
)
# GT-based metrics that are gated to class-A datasets.
_GT_KEYS = (
    "ari_domain",
    "nmi_domain",
    "weighted_dice_domain",
    "boundary_f1_domain",
    "boundary_precision_domain",
    "boundary_recall_domain",
)


# --------------------------------------------------------------------------- #
# PURE policy unit tests (data-independent)
# --------------------------------------------------------------------------- #


def test_infer_class_a_when_usable_gt_present():
    """Usable per-spot GT with >=2 classes infers class A."""
    assert infer_dataset_class(gt_present=True, n_gt_classes=3) == "A"
    assert infer_dataset_class(gt_present=True, n_gt_classes=2) == "A"


@pytest.mark.parametrize(
    "gt_present,n_classes",
    [(False, 0), (True, 1), (True, 0)],
)
def test_infer_unknown_without_usable_gt(gt_present, n_classes):
    """GT absent, or fewer than two classes, never infers class A."""
    assert infer_dataset_class(gt_present=gt_present, n_gt_classes=n_classes) == "unknown"


def test_resolve_inferred_class_a_emits_gt_metrics():
    """Class-A (inferred) emits GT metrics."""
    p = resolve_eval_policy(gt_present=True, n_gt_classes=3)
    assert p == EvalPolicy(
        dataset_class="A",
        dataset_class_source="inferred",
        gt_metrics_emitted=True,
        reason=p.reason,
    )
    assert p.gt_metrics_emitted is True
    assert "EMITTED" in p.reason


def test_resolve_gt_absent_suppresses_but_class_unknown():
    """No usable GT -> inferred unknown, GT metrics suppressed."""
    p = resolve_eval_policy(gt_present=False, n_gt_classes=0)
    assert p.dataset_class == "unknown"
    assert p.gt_metrics_emitted is False
    assert "SUPPRESSED" in p.reason


def test_resolve_forced_non_a_suppresses_even_with_gt():
    """Forcing a non-A class suppresses GT metrics even when GT is present."""
    for cls in ("B", "unknown"):
        p = resolve_eval_policy(gt_present=True, n_gt_classes=4, dataset_class=cls)
        assert p.dataset_class == cls
        assert p.dataset_class_source == "explicit"
        assert p.gt_metrics_emitted is False
        assert "!= A" in p.reason


def test_resolve_presence_guard_is_subset_of_class_gate():
    """Forced class-A on a label-less dataset still suppresses (no fabrication)."""
    p = resolve_eval_policy(gt_present=False, n_gt_classes=0, dataset_class="A")
    assert p.dataset_class == "A"
    assert p.gt_metrics_emitted is False
    assert "no usable per-spot GT" in p.reason


def test_resolve_rejects_unknown_class():
    with pytest.raises(ValueError):
        resolve_eval_policy(gt_present=True, n_gt_classes=2, dataset_class="C")


def test_resolve_is_deterministic():
    """The policy is a pure function: identical inputs -> identical output."""
    kw = dict(gt_present=True, n_gt_classes=3, dataset_class="A")
    assert resolve_eval_policy(**kw) == resolve_eval_policy(**kw)


def test_dataset_classes_constant():
    assert DATASET_CLASSES == ("A", "B", "unknown")


# --------------------------------------------------------------------------- #
# END-TO-END runner tests (synthetic .h5ad)
# --------------------------------------------------------------------------- #


def _load_runner():
    spec = importlib.util.spec_from_file_location("_run_real_factorgraph_policy", _RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_synthetic_h5ad(path: Path, *, with_gt: bool, n_per: int = 20, seed: int = 0):
    """Tiny 3-cluster synthetic AnnData; optional aligned per-spot GT labels."""
    ad = pytest.importorskip("anndata")

    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.0], [30.0, 0.0], [0.0, 30.0]])
    coords = np.vstack(
        [c + rng.normal(0.0, 1.0, size=(n_per, 2)) for c in centers]
    ).astype(np.float32)
    n = coords.shape[0]
    # Continuous, already-normalized expression so the runner skips scanpy norm.
    X = rng.random((n, 8)).astype(np.float32)
    obs = None
    if with_gt:
        import pandas as pd  # noqa: PLC0415

        labels = np.repeat(np.array(["L1", "L2", "L3"]), n_per)
        obs = pd.DataFrame({"layer_guess": labels}, index=[str(i) for i in range(n)])
    a = ad.AnnData(X=X, obs=obs)
    a.obsm["spatial"] = coords
    a.write_h5ad(path)
    return n


def _run(mod, monkeypatch, h5ad: Path, results_dir: Path, extra=()):
    pytest.importorskip("scanpy")
    argv = [
        "run_real_factorgraph.py",
        "--h5ad", str(h5ad),
        "--results-dir", str(results_dir),
        "--model", "coords",  # cheapest path (no fit), still emits the full suite
        "--already-normalized",
        "--n-domains", "3",
        "--knn", "4",
        "--seed", "0",
        *extra,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    mod.main()
    proj = results_dir / "factorgraph-st"
    metrics = json.loads((proj / "metrics.json").read_text())["metrics"]
    meta = json.loads((proj / "run_metadata.json").read_text())
    return metrics, meta


def test_runner_class_a_emits_gt_metrics(tmp_path, monkeypatch):
    """Class-A synthetic (GT present, >=2 classes): GT + internal metrics emitted."""
    pytest.importorskip("scanpy")
    mod = _load_runner()
    h5ad = tmp_path / "classA.h5ad"
    _write_synthetic_h5ad(h5ad, with_gt=True)
    metrics, meta = _run(mod, monkeypatch, h5ad, tmp_path / "resA")

    for k in _INTERNAL_KEYS:
        assert k in metrics, f"internal metric {k} must always be present"
    for k in _GT_KEYS:
        assert k in metrics, f"class-A run must emit GT metric {k}"
    assert metrics["dataset_class_is_a"] == 1.0
    assert metrics["gt_metrics_gated"] == 0.0
    assert metrics["ari_vs_gt_available"] == 1.0
    assert metrics["domain_metric_suite_available"] == 1.0
    assert meta["interpretability"]["eval_policy"]["dataset_class"] == "A"
    assert meta["interpretability"]["eval_policy"]["gt_metrics_emitted"] is True
    assert "EVAL POLICY (#341)" in meta["notes"]
    assert "EMITTED" in meta["notes"]


def test_runner_forced_non_a_suppresses_gt_keeps_internal(tmp_path, monkeypatch):
    """GT present but forced --dataset-class unknown: GT suppressed, internal kept."""
    pytest.importorskip("scanpy")
    mod = _load_runner()
    h5ad = tmp_path / "forced.h5ad"
    _write_synthetic_h5ad(h5ad, with_gt=True)
    metrics, meta = _run(
        mod, monkeypatch, h5ad, tmp_path / "resForced",
        extra=("--dataset-class", "unknown"),
    )

    for k in _INTERNAL_KEYS:
        assert k in metrics, f"internal metric {k} must remain present when gated"
    for k in _GT_KEYS:
        assert k not in metrics, f"GT metric {k} must be suppressed for non-A class"
    assert metrics["dataset_class_is_a"] == 0.0
    assert metrics["gt_metrics_gated"] == 1.0
    assert metrics["ari_vs_gt_available"] == 0.0
    assert metrics["domain_metric_suite_available"] == 0.0
    ep = meta["interpretability"]["eval_policy"]
    assert ep["dataset_class"] == "unknown"
    assert ep["dataset_class_source"] == "explicit"
    assert ep["gt_metrics_emitted"] is False


def test_runner_gt_absent_suppresses_gt_keeps_internal(tmp_path, monkeypatch):
    """No GT obs column: inferred unknown -> GT suppressed, internal still emitted."""
    pytest.importorskip("scanpy")
    mod = _load_runner()
    h5ad = tmp_path / "noGt.h5ad"
    _write_synthetic_h5ad(h5ad, with_gt=False)
    metrics, meta = _run(mod, monkeypatch, h5ad, tmp_path / "resNoGt")

    for k in _INTERNAL_KEYS:
        assert k in metrics
    for k in _GT_KEYS:
        assert k not in metrics
    assert metrics["gt_metrics_gated"] == 1.0
    ep = meta["interpretability"]["eval_policy"]
    assert ep["dataset_class"] == "unknown"
    assert ep["dataset_class_source"] == "inferred"
    assert ep["gt_metrics_emitted"] is False


def test_runner_is_deterministic(tmp_path, monkeypatch):
    """Two class-A runs with the same seed emit identical metrics."""
    pytest.importorskip("scanpy")
    mod = _load_runner()
    h5ad = tmp_path / "det.h5ad"
    _write_synthetic_h5ad(h5ad, with_gt=True)
    m1, _ = _run(mod, monkeypatch, h5ad, tmp_path / "det1")
    m2, _ = _run(mod, monkeypatch, h5ad, tmp_path / "det2")
    assert m1 == m2
