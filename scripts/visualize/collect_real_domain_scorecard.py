#!/usr/bin/env python
"""Collect real ``run_real_factorgraph.py`` outputs into figure-ready JSON (#322 #330).

The GT-skeptical scorecard (``fig_gt_skeptical_scorecard.py``, #330) and the
per-dataset accuracy table (``table_domain_accuracy.py``, #334/#322) are both
deliberately *data-independent*: each consumes a small JSON mapping so the real
numbers fill in once a labeled dataset lands (#133/#355). This collector is that
bridge. It scans a directory of runner outputs laid out as::

    <runs-root>/<method>_seed<N>/factorgraph-st/metrics.json
    <runs-root>/<method>_seed<N>/factorgraph-st/outputs/factors.npz   # carries domain_id

groups the per-seed runs by method, and emits two files:

  * ``scores.json`` -- ``method -> {coherence, stability, ari}`` for the
    scorecard. ``coherence`` is the chance-corrected label-invariant cluster
    coherence (``coherence_label_invariant_domain_delta`` from ``metrics.json``,
    i.e. the value already read against its spatial-shuffle null); ``stability``
    is the mean *pairwise* ARI across the per-seed predicted ``domain_id`` arrays
    (run-to-run self-consistency, NOT agreement with GT); ``ari`` is the mean
    ``ari_domain`` (vs GT) across seeds.
  * ``accuracy_results.json`` -- ``dataset -> variant -> {ari, nmi, ami}`` for the
    accuracy table, recomputed from the GT labels and predicted ``domain_id`` on
    the labeled subset via :func:`accuracy_from_labels` (so AMI -- not emitted by
    the runner -- is filled consistently with ARI/NMI).

GT-SKEPTIC NOTE: ``coherence``/``stability`` are the label-FREE axis ("looks
self-consistent and smooth") and ``ari`` is the only number that says the domains
are *correct*. The collector keeps them separate exactly so a smooth-but-wrong
method cannot hide behind a single headline coherence value. No metric is ever
fabricated: a method with fewer than two seeds gets ``stability = null`` (the
scorecard requires >=2 seeds to plot it) rather than a fake ``1.0``.

This module is matplotlib-free; it only reads JSON + ``.npz`` and reuses the
project metrics. Render the scorecard/table from its outputs::

    python scripts/visualize/collect_real_domain_scorecard.py \
        --runs-root results/starmap_wang2018 \
        --h5ad ../data/processed/starmap_mouse_vcortex_wang2018/anndata.h5ad \
        --gt-obs-key ground_truth --dataset-id starmap_mouse_vcortex_wang2018 \
        --out-dir results/starmap_wang2018/_scorecard
    python scripts/visualize/fig_gt_skeptical_scorecard.py \
        --scores results/starmap_wang2018/_scorecard/scores.json --out /tmp/scorecard.png
    python scripts/tables/table_domain_accuracy.py \
        --results results/starmap_wang2018/_scorecard/accuracy_results.json --out-dir /tmp/tables
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
for _p in (str(_REPO / "src"), str(_REPO / "scripts" / "tables")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from table_domain_accuracy import accuracy_from_labels  # noqa: E402

from factorgraph_st.eval.metrics import adjusted_rand_index  # noqa: E402

#: NA/blank GT tokens, byte-for-byte mirror of ``run_real_factorgraph._GT_NA_TOKENS``
#: so the collector's labeled subset matches the runner's ARI/NMI exactly.
_GT_NA_TOKENS = frozenset({"", "na", "nan", "none", "unknown", "unlabeled", "unlabelled"})

#: ``<method>_seed<N>`` run-directory naming emitted by the runner conventions.
_RUN_DIR_RE = re.compile(r"^(?P<method>.+)_seed(?P<seed>\d+)$")


def parse_run_dirname(name: str) -> tuple[str, int] | None:
    """Split a ``<method>_seed<N>`` directory name into ``(method, seed)``.

    Returns ``None`` for names that do not match the convention so unrelated
    directories under the runs root are skipped rather than mis-grouped.
    """
    match = _RUN_DIR_RE.match(name)
    if match is None:
        return None
    return match.group("method"), int(match.group("seed"))


def group_run_dirs(runs_root: Path) -> dict[str, list[Path]]:
    """Map ``method -> [run-dir, ...]`` (seed-sorted) under ``runs_root``.

    Only immediate child directories named ``<method>_seed<N>`` that actually
    contain a ``factorgraph-st/metrics.json`` are included.
    """
    grouped: dict[str, list[tuple[int, Path]]] = {}
    for child in sorted(Path(runs_root).iterdir()):
        if not child.is_dir():
            continue
        parsed = parse_run_dirname(child.name)
        if parsed is None:
            continue
        method, seed = parsed
        if not (child / "factorgraph-st" / "metrics.json").is_file():
            continue
        grouped.setdefault(method, []).append((seed, child))
    return {m: [p for _s, p in sorted(runs)] for m, runs in grouped.items()}


def _load_metrics(run_dir: Path) -> dict[str, object]:
    bundle = json.loads((run_dir / "factorgraph-st" / "metrics.json").read_text(encoding="utf-8"))
    raw = bundle.get("metrics", bundle)
    if not isinstance(raw, Mapping):
        raise ValueError(f"{run_dir}: metrics.json has no usable 'metrics' mapping")
    return dict(raw)


def _load_domain_id(run_dir: Path) -> np.ndarray:
    """Load the predicted ``domain_id`` from a run's outputs.

    The runner names the outputs bundle per model (``factors.npz`` for the factor
    models, ``baseline_coords.npz`` / ``baseline_spatial_smooth.npz`` for the
    baselines), but every variant carries a ``domain_id`` array. Scan the outputs
    dir for the first ``.npz`` that contains it rather than hard-coding a name.
    """
    outputs = run_dir / "factorgraph-st" / "outputs"
    for npz in sorted(outputs.glob("*.npz")):
        with np.load(npz) as data:
            if "domain_id" in data:
                return np.asarray(data["domain_id"]).astype(np.int64)
    raise FileNotFoundError(f"{outputs}: no .npz with a 'domain_id' array")


def pairwise_stability(label_arrays: Sequence[np.ndarray]) -> float | None:
    """Mean pairwise ARI across per-seed predicted-label arrays (run-to-run agreement).

    This is the *label-free* stability signal: it measures how reproducible the
    predicted partition is across model-init seeds, independent of any ground
    truth. Returns ``None`` when fewer than two arrays are given (stability is
    undefined for a single run -- never fabricate it as a perfect ``1.0``).
    """
    arrays = [np.asarray(a).astype(np.int64) for a in label_arrays]
    if len(arrays) < 2:
        return None
    scores = [
        adjusted_rand_index(arrays[i], arrays[j])
        for i in range(len(arrays))
        for j in range(i + 1, len(arrays))
    ]
    return float(np.mean(scores))


def labeled_gt_codes(gt_raw: Sequence[object]) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(valid_mask, gt_codes)`` for the labeled subset of raw GT values.

    NA/blank tokens (:data:`_GT_NA_TOKENS`, case-insensitive) are dropped, exactly
    as the runner does, and the surviving labels are integer-coded. ``gt_codes`` is
    aligned to the *valid* spots only.
    """
    raw = np.asarray([str(v) for v in gt_raw])
    valid = np.array([s.strip().lower() not in _GT_NA_TOKENS for s in raw], dtype=bool)
    _, codes = np.unique(raw[valid], return_inverse=True)
    return valid, codes.astype(np.int64)


def summarize_method(
    metrics_list: Sequence[Mapping[str, object]],
    label_arrays: Sequence[np.ndarray],
    gt_valid: np.ndarray,
    gt_codes: np.ndarray,
) -> dict[str, float | None]:
    """Aggregate one method's per-seed runs into scorecard + accuracy numbers.

    ``coherence`` and ``ari`` are means across seeds; ``ari``/``nmi``/``ami`` are
    recomputed from ``(gt_codes, domain_id)`` on the labeled subset (so AMI is
    consistent with ARI/NMI and matches the runner's NA handling); ``stability``
    is the mean pairwise ARI across the per-seed predictions.
    """
    if not metrics_list:
        raise ValueError("metrics_list must be non-empty")
    coherences = [float(m["coherence_label_invariant_domain_delta"]) for m in metrics_list]
    accs = [
        accuracy_from_labels(gt_codes, np.asarray(dom).astype(np.int64)[gt_valid])
        for dom in label_arrays
    ]
    mean = lambda key: float(np.mean([a[key] for a in accs]))  # noqa: E731
    return {
        "coherence": float(np.mean(coherences)),
        "stability": pairwise_stability(label_arrays),
        "ari": mean("ari"),
        "nmi": mean("nmi"),
        "ami": mean("ami"),
        "n_seeds": float(len(metrics_list)),
    }


def build_scores(summaries: Mapping[str, Mapping[str, float | None]]) -> dict[str, dict[str, float]]:
    """Project per-method summaries to the scorecard schema ``{coherence,stability,ari}``.

    A method whose ``stability`` is ``None`` (single seed) is omitted from the
    scorecard -- the scorecard's stability axis is undefined for it, and emitting a
    placeholder would read as a real measurement.
    """
    scores: dict[str, dict[str, float]] = {}
    for method, summary in summaries.items():
        if summary.get("stability") is None:
            continue
        scores[method] = {
            "coherence": float(summary["coherence"]),
            "stability": float(summary["stability"]),
            "ari": float(summary["ari"]),
        }
    return scores


def build_accuracy_results(
    dataset_id: str, summaries: Mapping[str, Mapping[str, float | None]]
) -> dict[str, dict[str, dict[str, float]]]:
    """Project per-method summaries to the accuracy-table schema ``dataset->variant->{ari,nmi,ami}``."""
    variants = {
        method: {k: float(summary[k]) for k in ("ari", "nmi", "ami")}
        for method, summary in summaries.items()
    }
    return {dataset_id: variants}


def collect(
    runs_root: Path, h5ad: Path, gt_obs_key: str, dataset_id: str
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, dict[str, float]]], dict[str, dict]]:
    """End-to-end collection: returns ``(scores, accuracy_results, per_method_summary)``."""
    import anndata as ad  # noqa: PLC0415  (heavy, lazy)

    adata = ad.read_h5ad(h5ad)
    if gt_obs_key not in adata.obs:
        raise SystemExit(
            f"--gt-obs-key {gt_obs_key!r} not in obs columns {list(adata.obs.columns)}"
        )
    gt_valid, gt_codes = labeled_gt_codes(adata.obs[gt_obs_key].to_numpy())

    grouped = group_run_dirs(runs_root)
    if not grouped:
        raise SystemExit(f"no <method>_seed<N>/factorgraph-st/metrics.json runs under {runs_root}")

    summaries: dict[str, dict[str, float | None]] = {}
    for method, run_dirs in grouped.items():
        metrics_list = [_load_metrics(d) for d in run_dirs]
        label_arrays = [_load_domain_id(d) for d in run_dirs]
        summaries[method] = summarize_method(metrics_list, label_arrays, gt_valid, gt_codes)

    return build_scores(summaries), build_accuracy_results(dataset_id, summaries), summaries


def collect_records(runs_root: Path, h5ad: Path, gt_obs_key: str, dataset_id: str) -> list[dict[str, object]]:
    """Per-seed long-form accuracy records for the cross-dataset panel (#322).

    Returns one row per ``(dataset, variant, seed)`` with ``ari``/``nmi``/``ami``
    recomputed on the labeled subset -- the un-aggregated counterpart of
    :func:`collect`, so the cross-dataset figure can show per-seed *spread* (not
    just the mean). The seed is read back from each ``<method>_seed<N>`` dir name.
    """
    import anndata as ad  # noqa: PLC0415  (heavy, lazy)

    adata = ad.read_h5ad(h5ad)
    if gt_obs_key not in adata.obs:
        raise SystemExit(f"--gt-obs-key {gt_obs_key!r} not in obs columns {list(adata.obs.columns)}")
    gt_valid, gt_codes = labeled_gt_codes(adata.obs[gt_obs_key].to_numpy())

    records: list[dict[str, object]] = []
    for method, run_dirs in group_run_dirs(runs_root).items():
        for run_dir in run_dirs:
            parsed = parse_run_dirname(run_dir.name)
            seed = parsed[1] if parsed is not None else -1
            dom = _load_domain_id(run_dir).astype(np.int64)[gt_valid]
            acc = accuracy_from_labels(gt_codes, dom)
            records.append(
                {
                    "dataset": dataset_id,
                    "variant": method,
                    "seed": int(seed),
                    "ari": float(acc["ari"]),
                    "nmi": float(acc["nmi"]),
                    "ami": float(acc["ami"]),
                }
            )
    return records


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True, help="Dir of <method>_seed<N> runner outputs.")
    parser.add_argument("--h5ad", type=Path, required=True, help="Source AnnData (for GT labels).")
    parser.add_argument("--gt-obs-key", type=str, required=True, help="obs column with per-spot GT domains.")
    parser.add_argument("--dataset-id", type=str, required=True, help="Dataset id used as the table row key.")
    parser.add_argument(
        "--out-dir", type=Path, required=True, help="Directory for scores.json + accuracy_results.json."
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    scores, accuracy, summaries = collect(args.runs_root, args.h5ad, args.gt_obs_key, args.dataset_id)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "scores.json").write_text(json.dumps(scores, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "accuracy_results.json").write_text(
        json.dumps(accuracy, indent=2, sort_keys=True), encoding="utf-8"
    )
    records = collect_records(args.runs_root, args.h5ad, args.gt_obs_key, args.dataset_id)
    (out_dir / "per_seed_records.json").write_text(
        json.dumps(records, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"wrote {out_dir / 'scores.json'} ({len(scores)} methods on the scorecard)")
    print(f"wrote {out_dir / 'accuracy_results.json'}")
    print(f"wrote {out_dir / 'per_seed_records.json'} ({len(records)} per-seed rows)")
    for method in sorted(summaries):
        s = summaries[method]
        stab = "n/a" if s["stability"] is None else f"{s['stability']:.3f}"
        print(
            f"  {method:16s} ari={s['ari']:.3f} nmi={s['nmi']:.3f} ami={s['ami']:.3f} "
            f"coherence={s['coherence']:.3f} stability={stab} (n_seeds={int(s['n_seeds'])})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
