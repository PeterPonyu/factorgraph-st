"""Smoke + unit tests for the GT-vs-predicted spatial map gallery (#324).

The numpy-only color-alignment helper is tested directly; the matplotlib render
is guarded with ``importorskip`` and only checked for a valid PNG.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "visualize"
    / "fig_gt_vs_pred_map_gallery.py"
)
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _load_module():
    spec = importlib.util.spec_from_file_location("_fig_gt_vs_pred_map_gallery", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_align_pred_to_gt_recolors_by_majority():
    mod = _load_module()
    gt = np.array([0, 0, 0, 1, 1, 1])
    # pred uses different integers but the SAME partition (just relabeled 0->7, 1->3).
    pred = np.array([7, 7, 7, 3, 3, 3])
    aligned = mod.align_pred_to_gt(gt, pred)
    assert aligned.tolist() == gt.tolist()


def test_align_pred_to_gt_majority_vote_with_noise():
    mod = _load_module()
    gt = np.array([0, 0, 0, 1, 1, 1])
    # pred cluster 5 is mostly GT-0 (one stray GT-1 spot) -> recolors to 0.
    pred = np.array([5, 5, 5, 5, 9, 9])
    aligned = mod.align_pred_to_gt(gt, pred)
    assert aligned[0] == 0
    assert aligned[4] == 1  # cluster 9 is all GT-1


def test_align_pred_to_gt_empty():
    mod = _load_module()
    out = mod.align_pred_to_gt(np.array([], dtype=int), np.array([], dtype=int))
    assert out.shape == (0,)


def test_render_gallery_writes_png(tmp_path):
    pytest.importorskip("matplotlib")
    mod = _load_module()
    rng = np.random.default_rng(0)
    coords = rng.random((30, 2))
    gt = np.array([0] * 15 + [1] * 15)
    pred = np.array([3] * 15 + [8] * 15)
    out = tmp_path / "gallery.png"
    mod.render_gallery(coords, gt, [("gnmf", pred)], out)
    assert out.is_file()
    assert out.read_bytes()[:8] == _PNG_MAGIC
