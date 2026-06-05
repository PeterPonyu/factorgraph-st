"""Domain-quality metric emission policy (#341).

Data-independent policy deciding WHICH metrics a run emits:

* internal / label-free metrics (Moran's I delta, label-invariant coherence,
  silhouette, Calinski-Harabasz, factor stats) are ALWAYS emitted (default-on);
* ground-truth-based metrics (ARI / NMI / AMI, boundary precision/recall/F1,
  weighted Dice) are emitted ONLY for **class-A** datasets -- those with
  trustworthy per-spot ground-truth domain labels.

This module loads no data and reads no dataset cards. It operates purely on a
small set of booleans/ints the runner already discovered (whether usable
per-spot GT is present and how many distinct GT classes it carries) plus an
optional explicit ``--dataset-class`` override, so the policy is unit-testable
in isolation and never depends on disk layout.

The dataset *class* is an explicit policy axis distinct from mere GT *presence*:
the presence-based guard is kept as a strict SUBSET of the class gate, so forcing
``--dataset-class A`` on a label-less dataset still suppresses GT metrics (labels
are never fabricated) and a class-B/unknown dataset that happens to carry labels
still has its GT metrics gated off (its labels are not certified trustworthy).
"""

from __future__ import annotations

from dataclasses import dataclass

#: Allowed dataset classes. ``A`` = trustworthy per-spot GT (GT metrics emitted);
#: ``B`` = labels present but not certified trustworthy; ``unknown`` = unclassified.
DATASET_CLASSES = ("A", "B", "unknown")


@dataclass(frozen=True)
class EvalPolicy:
    """Resolved GT-metric emission policy for a single run (#341).

    ``dataset_class`` is the resolved class (explicit override or inferred);
    ``dataset_class_source`` is ``"explicit"`` or ``"inferred"``;
    ``gt_metrics_emitted`` is ``True`` iff GT-based metrics should be emitted;
    ``reason`` is a human-readable explanation of the gating decision.
    """

    dataset_class: str
    dataset_class_source: str
    gt_metrics_emitted: bool
    reason: str


def infer_dataset_class(*, gt_present: bool, n_gt_classes: int) -> str:
    """Infer the dataset class from GT availability (data-independent).

    Returns ``"A"`` iff usable per-spot GT is present AND at least two distinct
    classes are observed; otherwise ``"unknown"``. Class ``B`` is never inferred
    -- distinguishing "labels present but untrustworthy" from class A is a human
    judgement supplied explicitly via ``--dataset-class``.
    """
    if gt_present and n_gt_classes >= 2:
        return "A"
    return "unknown"


def resolve_eval_policy(
    *,
    gt_present: bool,
    n_gt_classes: int,
    dataset_class: str | None = None,
) -> EvalPolicy:
    """Resolve the GT-metric emission policy for a run (#341).

    ``dataset_class`` is the explicit ``--dataset-class`` override (``None`` =
    infer via :func:`infer_dataset_class`). GT-based metrics are emitted ONLY when
    the resolved class is ``"A"`` AND usable per-spot GT is actually present with
    at least two classes -- the presence-based guard is a strict subset of the
    class gate, so forcing ``--dataset-class A`` on a label-less dataset still
    suppresses GT metrics (labels are never fabricated). Deterministic and pure.
    """
    if dataset_class is not None:
        if dataset_class not in DATASET_CLASSES:
            raise ValueError(
                f"dataset_class must be one of {DATASET_CLASSES}, got {dataset_class!r}"
            )
        resolved = dataset_class
        source = "explicit"
    else:
        resolved = infer_dataset_class(gt_present=gt_present, n_gt_classes=n_gt_classes)
        source = "inferred"

    gt_usable = bool(gt_present and n_gt_classes >= 2)
    emit = (resolved == "A") and gt_usable

    if emit:
        reason = (
            f"dataset_class={resolved} ({source}) with usable per-spot GT "
            f"({n_gt_classes} classes): GT-based metrics EMITTED"
        )
    elif resolved != "A":
        reason = (
            f"dataset_class={resolved} ({source}) != A: GT-based metrics SUPPRESSED "
            "(only class-A datasets carry trustworthy per-spot ground truth)"
        )
    else:
        reason = (
            f"dataset_class={resolved} ({source}) but no usable per-spot GT "
            f"(gt_present={gt_present}, n_gt_classes={n_gt_classes}): GT-based metrics "
            "SUPPRESSED (presence guard -- labels are never fabricated)"
        )

    return EvalPolicy(
        dataset_class=resolved,
        dataset_class_source=source,
        gt_metrics_emitted=emit,
        reason=reason,
    )


__all__ = ["DATASET_CLASSES", "EvalPolicy", "infer_dataset_class", "resolve_eval_policy"]
