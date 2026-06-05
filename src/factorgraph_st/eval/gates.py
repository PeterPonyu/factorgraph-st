"""Evaluation gates and negative controls for spatial-coherence claims.

Three GT-free / data-independent routines, validated on the synthetic
generator:

* :func:`permutation_null_calibration` (#316) — a permutation-null negative
  control for a spatial-coherence metric (Moran's I or label-invariant cluster
  coherence). Shuffling *which spot carries which label* destroys spatial
  structure while preserving the marginal label distribution, so the metric
  recomputed over many shuffles is the no-structure reference. The routine
  reports the observed value against that null (mean/std, a z-score, and a
  one-sided empirical p).
* :func:`select_n_factors` (#315) — GT-free model selection for the number of
  factors ``k`` via :func:`held_out_reconstruction_error`: sweep candidate
  ``k``, score each by held-out reconstruction error, and pick the minimizer.
* :func:`coherence_claim_gate` (#354) — a PASS/FAIL gate a spatial-coherence
  claim must clear *before* it is reported. It requires (a) the observed
  coherence to exceed its permutation null by a margin (and a z-score floor) and
  (b) the labels not to be coordinate-derived/leaked (a coords-only k-means must
  not reproduce them). It refuses to certify a claim it cannot check, rather than
  silently passing.

All routines are pure NumPy and deterministic for a fixed ``seed``; they reuse
the metrics in :mod:`factorgraph_st.eval.metrics`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from factorgraph_st.eval.metrics import (
    adjusted_rand_index,
    held_out_reconstruction_error,
    label_invariant_cluster_coherence,
    morans_i,
)

# Spatial-coherence metrics share the ``(values, edges) -> float`` signature, so
# the permutation-null machinery is metric-agnostic. ``coherence`` is
# relabel-invariant (the default); ``morans_i`` is the numeric autocorrelation.
_COHERENCE_METRICS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "coherence": label_invariant_cluster_coherence,
    "morans_i": morans_i,
}


@dataclass
class NullCalibration:
    """Permutation-null calibration of a spatial-coherence metric (#316).

    ``observed`` is the metric on the real labels; ``null_mean`` / ``null_std``
    summarize the metric over ``n_shuffles`` spatial permutations of the labels.
    ``z_score = (observed - null_mean) / null_std`` (``inf`` when the null is
    degenerate and the observed value differs from it). ``p_empirical`` is the
    one-sided ``P(null >= observed)`` with the standard ``+1`` correction (the
    observed value counts as one realization), so it is never exactly zero.
    """

    metric: str
    observed: float
    null_mean: float
    null_std: float
    z_score: float
    p_empirical: float
    n_shuffles: int


def permutation_null_calibration(
    values: np.ndarray,
    edges: np.ndarray,
    *,
    metric: str = "coherence",
    n_shuffles: int = 199,
    seed: int = 0,
) -> NullCalibration:
    """Calibrate a spatial-coherence metric against its permutation null (#316).

    ``values`` are the per-spot labels (for ``metric="coherence"``) or a numeric
    score (for ``metric="morans_i"``); ``edges`` is the ``(2, n_edges)`` COO
    adjacency. Each of the ``n_shuffles`` permutations reassigns ``values`` across
    spots (seeded), destroying spatial structure, and the metric is recomputed.
    Returns a :class:`NullCalibration`. Deterministic for a fixed ``seed``.
    """
    if metric not in _COHERENCE_METRICS:
        raise ValueError(
            f"metric must be one of {sorted(_COHERENCE_METRICS)}, got {metric!r}"
        )
    fn = _COHERENCE_METRICS[metric]
    arr = np.asarray(values)
    observed = float(fn(arr, edges))

    n_shuffles = max(int(n_shuffles), 1)
    rng = np.random.default_rng(seed)
    null = np.empty(n_shuffles, dtype=np.float64)
    for i in range(n_shuffles):
        null[i] = fn(rng.permutation(arr), edges)

    null_mean = float(null.mean())
    null_std = float(null.std())
    if null_std > 0.0:
        z = (observed - null_mean) / null_std
    elif observed == null_mean:
        z = 0.0
    else:
        z = float("inf") if observed > null_mean else float("-inf")
    p_empirical = float((1 + int(np.count_nonzero(null >= observed))) / (n_shuffles + 1))
    return NullCalibration(
        metric=metric,
        observed=observed,
        null_mean=null_mean,
        null_std=null_std,
        z_score=float(z),
        p_empirical=p_empirical,
        n_shuffles=n_shuffles,
    )


@dataclass
class FactorSelection:
    """GT-free selection of the number of factors ``k`` (#315).

    ``errors[i]`` is the held-out reconstruction error at ``candidate_ks[i]``;
    ``best_k`` is the knee of that curve — the largest ``k`` still reached by
    consecutive relative improvements of at least ``min_rel_improvement``. The
    knee (point of diminishing returns), not the global minimum, is the right
    GT-free choice: each added factor can only *lower* held-out error here, so a
    pure argmin would always pick the largest candidate.
    """

    candidate_ks: list[int]
    errors: list[float]
    best_k: int
    min_rel_improvement: float


def _knee_index(errors: Sequence[float], min_rel_improvement: float) -> int:
    """Index of the held-out-error knee (last step with a worthwhile drop).

    Walk from the smallest ``k`` and keep accepting the next candidate while it
    lowers the error by at least ``min_rel_improvement`` (relative); stop at the
    first step whose improvement falls below that floor (or that increases the
    error). The returned index is the largest ``k`` worth keeping — adding more
    factors past it buys < ``min_rel_improvement`` and risks fitting noise.
    """
    best = 0
    for i in range(1, len(errors)):
        prev = errors[i - 1]
        if prev <= 0.0:
            break
        rel = (prev - errors[i]) / abs(prev)
        if rel >= min_rel_improvement:
            best = i
        else:
            break
    return best


def _nmf_scores(X: np.ndarray, k: int, seed: int = 0, *, n_iter: int = 300) -> np.ndarray:
    """Rank-``k`` nonnegative factor scores via Lee-Seung multiplicative updates.

    The dependency-free default factorization for :func:`select_n_factors`. Plain
    NMF (no graph regularization — this is a reconstruction sweep, not a fit) of
    ``X ~= Z @ W.T`` with ``Z, W >= 0``, matching the nonnegativity that
    :func:`held_out_reconstruction_error` assumes (it refits and clips ``W`` to
    the nonnegative orthant). Held-out error bottoms out near the true factor
    count and rises again as surplus factors begin fitting noise. Deterministic
    for a fixed ``seed``.
    """
    Xf = np.clip(np.asarray(X, dtype=np.float64), 0.0, None)
    n, g = Xf.shape
    k = int(min(max(k, 1), max(n, 1), max(g, 1)))
    rng = np.random.default_rng(seed)
    eps = 1e-9
    Z = rng.uniform(0.1, 1.0, size=(n, k))
    W = rng.uniform(0.1, 1.0, size=(g, k))
    for _ in range(max(int(n_iter), 1)):
        Z *= (Xf @ W) / (Z @ (W.T @ W) + eps)
        W *= (Xf.T @ Z) / (W @ (Z.T @ Z) + eps)
    return Z


def select_n_factors(
    X: np.ndarray,
    candidate_ks: Sequence[int],
    *,
    holdout: float = 0.2,
    seed: int = 0,
    min_rel_improvement: float = 0.05,
    fit_fn: Callable[[np.ndarray, int, int], np.ndarray] | None = None,
) -> FactorSelection:
    """Select the factor count ``k`` by held-out reconstruction error (#315).

    For each candidate ``k`` the loadings are obtained from ``fit_fn(X, k, seed)``
    (default :func:`_nmf_scores`) and scored by
    :func:`held_out_reconstruction_error`, a ground-truth-free measure that refits
    ``W`` on a spot subset and scores the relative error on the rest. The chosen
    ``k`` is the **knee** of that curve (:func:`_knee_index`): the largest ``k``
    still earning a relative error drop of at least ``min_rel_improvement``.
    Selecting the knee rather than the raw minimum recovers the planted factor
    count — added factors only ever lower this error, so an argmin would trivially
    return the largest candidate. Deterministic for a fixed ``seed``.
    """
    ks = [int(k) for k in candidate_ks]
    if not ks:
        raise ValueError("candidate_ks must be non-empty")
    fit = fit_fn or _nmf_scores
    errors = [
        float(held_out_reconstruction_error(X, fit(X, k, seed), holdout=holdout, seed=seed))
        for k in ks
    ]
    best_k = ks[_knee_index(errors, min_rel_improvement)]
    return FactorSelection(
        candidate_ks=ks,
        errors=errors,
        best_k=int(best_k),
        min_rel_improvement=float(min_rel_improvement),
    )


@dataclass
class GateResult:
    """Verdict of the spatial-coherence claim gate (#354).

    ``passed`` is ``True`` only when the observed coherence clears the
    permutation null by ``min_margin`` (and ``min_z``) *and* the labels are not
    coordinate-derived. ``reason`` is a human-readable PASS message or a
    ``"; "``-joined list of every failed criterion. ``coord_leakage_ari`` is the
    ARI between the claimed labels and a coords-only k-means (``nan`` when not
    assessed).
    """

    passed: bool
    reason: str
    observed: float
    null_mean: float
    z_score: float
    p_empirical: float
    coord_leakage_ari: float


def _coord_leakage_ari(labels: np.ndarray, coords: np.ndarray, *, seed: int = 0) -> float:
    """ARI between ``labels`` and a coords-only k-means with the same #clusters.

    Quantifies how much the labelling is recoverable from raw position alone: an
    ARI near ``1.0`` means the labels essentially *are* the coordinates (the #340
    coordinate-only negative control), so any spatial-coherence claim about them
    is leaked. Returns ``nan`` when ARI is undefined (fewer than two classes or
    spots). Reuses the package k-means (:func:`factorgraph_st.model.decoder._kmeans`).
    """
    from factorgraph_st.model.decoder import _kmeans

    lab = np.asarray(labels)
    xy = np.asarray(coords, dtype=np.float64)
    if lab.shape[0] != xy.shape[0]:
        raise ValueError("labels and coords must have equal length")
    n = xy.shape[0]
    k = int(np.unique(lab).size)
    if k < 2 or n < 2:
        return float("nan")
    std = xy.std(axis=0)
    std[std == 0.0] = 1.0
    z = (xy - xy.mean(axis=0)) / std
    coord_labels = _kmeans(z, min(k, n), seed, n_init=4, max_iter=100)
    return float(adjusted_rand_index(lab, coord_labels))


def coherence_claim_gate(
    labels: np.ndarray,
    edges: np.ndarray,
    *,
    coords: np.ndarray | None = None,
    coords_derived: bool = False,
    metric: str = "coherence",
    n_shuffles: int = 199,
    seed: int = 0,
    min_margin: float = 0.1,
    min_z: float = 3.0,
    max_coord_ari: float = 0.5,
) -> GateResult:
    """Gate a spatial-coherence claim: PASS only if real and not leaked (#354).

    Two independent criteria must both hold:

    1. **Beats its null** — the observed coherence exceeds its permutation null
       (:func:`permutation_null_calibration`) by at least ``min_margin`` and has a
       z-score of at least ``min_z``.
    2. **Not coordinate-derived** — the labels are not reproducible from raw
       position. Provide ``coords`` to check this quantitatively (a coords-only
       k-means with ARI ``>= max_coord_ari`` fails the gate), or set
       ``coords_derived=True`` to declare them leaked outright. If neither is
       supplied the gate **fails**: it refuses to certify a leak-free claim it
       cannot verify rather than silently passing.

    Returns a :class:`GateResult`. Deterministic for a fixed ``seed``.
    """
    cal = permutation_null_calibration(
        labels, edges, metric=metric, n_shuffles=n_shuffles, seed=seed
    )
    margin = cal.observed - cal.null_mean
    reasons: list[str] = []
    if margin < min_margin:
        reasons.append(
            f"coherence margin over null {margin:.4f} < min_margin {min_margin}"
        )
    if cal.z_score < min_z:
        reasons.append(f"z-score {cal.z_score:.3f} < min_z {min_z}")

    coord_ari = float("nan")
    if coords_derived:
        reasons.append("labels declared coordinate-derived (coords_derived=True)")
    elif coords is not None:
        coord_ari = _coord_leakage_ari(labels, coords, seed=seed)
        if np.isfinite(coord_ari) and coord_ari >= max_coord_ari:
            reasons.append(
                f"coordinate-leakage: coords-only k-means reproduces labels "
                f"(ARI {coord_ari:.3f} >= {max_coord_ari})"
            )
    else:
        reasons.append(
            "coordinate-leakage unverifiable: pass `coords` (or set coords_derived) "
            "— refusing to certify a leak-free claim silently"
        )

    passed = not reasons
    reason = (
        "PASS: coherence exceeds its permutation null and is not coordinate-derived"
        if passed
        else "; ".join(reasons)
    )
    return GateResult(
        passed=passed,
        reason=reason,
        observed=cal.observed,
        null_mean=cal.null_mean,
        z_score=cal.z_score,
        p_empirical=cal.p_empirical,
        coord_leakage_ari=coord_ari,
    )
