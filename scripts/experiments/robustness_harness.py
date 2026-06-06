"""Data-independent robustness harness for the synthetic FactorGraph-ST model.

One shared harness backs three thin drivers (see ``run_*`` scripts in this
directory), each probing a different axis of robustness on the *synthetic*
generator only (no labeled real data required):

* :func:`seed_init_stability` (#314) — how reproducible the recovered factors and
  domains are across random initializations of the model fit, holding the data
  fixed.
* :func:`robustness_to_k` (#308) — how domain recovery degrades when the requested
  number of domains ``k`` is mis-specified around its true value (optionally with
  mild spatial-graph disruption).
* :func:`recovery_vs_batch` (#312) — how factor / domain recovery degrades as the
  strength of unwanted (nuisance) variation in the synthetic data rises.

Every public function returns a plain JSON-serializable ``dict`` (lists of
floats, never numpy arrays) so the drivers can dump results directly. All runs
are deterministic in their seeds and intended to be SMALL (fast, test-friendly).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations

import numpy as np

from factorgraph_st.eval.metrics import adjusted_rand_index, matched_factor_correlation
from factorgraph_st.model.learned import fit_transform_gnmf
from factorgraph_st.synth.generator import SynthInstance, generate_instance


@dataclass
class HarnessConfig:
    """Small synthetic + GNMF configuration shared by every robustness driver.

    Defaults are deliberately tiny so the harness runs in well under a second and
    is safe to exercise from the test suite. Drivers expose these as CLI flags.
    """

    n_sections: int = 2
    n_spots_per_section: int = 80
    n_genes: int = 50
    K_shared: int = 3
    K_private: int = 1
    n_domains: int = 4
    k_nn: int = 6
    noise_sigma: float = 0.5
    # GNMF fit hyperparameters. NOTE (#392): ``fit_gnmf`` now interprets ``lam``
    # relative to the data scale (effective weight = ``lam * RMS(X)``) so the fit is
    # invariant to a global rescale of X. This harness's synthetic instance has
    # ``RMS(X) ~= 5.37``, so to keep the model-selection property (#308) evaluated at
    # the SAME effective regularization it was validated at before the fix
    # (effective ~= 1.0), the default is set to ``1 / RMS(X) ~= 0.186`` -- a units
    # conversion that preserves the prior operating point, NOT a value searched to
    # make the curve peak.
    lam: float = 0.186
    n_iter: int = 80
    tol: float = 1e-4


def _generate(config: HarnessConfig, *, seed: int, noise_sigma: float | None = None) -> SynthInstance:
    """Build one synthetic instance from ``config`` (optionally overriding noise)."""
    return generate_instance(
        n_sections=config.n_sections,
        n_spots_per_section=config.n_spots_per_section,
        n_genes=config.n_genes,
        K_shared=config.K_shared,
        K_private=config.K_private,
        noise_sigma=config.noise_sigma if noise_sigma is None else noise_sigma,
        n_domains=config.n_domains,
        k_nn=config.k_nn,
        seed=seed,
    )


def _fit(
    synth: SynthInstance,
    config: HarnessConfig,
    *,
    requested_n_domains: int,
    seed: int,
    edges: np.ndarray | None = None,
):
    """Fit GNMF on ``synth`` and return its validated outputs.

    ``requested_n_domains`` is the *requested* domain count handed to the
    clustering step (the knob #308 sweeps); ``edges`` lets a caller substitute a
    perturbed spatial graph without rebuilding the instance.
    """
    K_shared = int(synth.Z_shared.shape[1])
    K_private = int(synth.Z_private.shape[1])
    out, _ = fit_transform_gnmf(
        synth.X,
        synth.edges if edges is None else edges,
        K_shared=K_shared,
        K_private=K_private,
        n_domains=requested_n_domains,
        lam=config.lam,
        n_iter=config.n_iter,
        tol=config.tol,
        seed=seed,
    )
    return out


def _ground_truth_factors(synth: SynthInstance) -> np.ndarray:
    """Concatenated GT factor matrix ``Z = [Z_shared | Z_private]`` as float64."""
    return np.concatenate([synth.Z_shared, synth.Z_private], axis=1).astype(np.float64)


def _disrupt_edges(edges: np.ndarray, dropout: float, *, seed: int) -> np.ndarray:
    """Randomly drop a fraction of (symmetrizable) edges — mild graph disruption.

    ``dropout`` in ``[0, 1)`` is the fraction of edge columns removed. ``0.0``
    returns the graph unchanged. Deterministic in ``seed``.
    """
    if dropout <= 0.0 or edges.size == 0:
        return edges
    if not 0.0 <= dropout < 1.0:
        raise ValueError(f"dropout must be in [0, 1); got {dropout}")
    n_edges = edges.shape[1]
    keep = np.random.default_rng(seed).random(n_edges) >= dropout
    return edges[:, keep]


def seed_init_stability(
    config: HarnessConfig | None = None,
    *,
    seeds: list[int] | None = None,
    data_seed: int = 0,
) -> dict:
    """#314 — factor/domain reproducibility across model initializations.

    The synthetic data is held FIXED (one instance at ``data_seed``); only the
    model fit seed — which seeds the GNMF nonnegative initialization — varies over
    ``seeds``. Stability is the mean over all unordered seed PAIRS of:

    * ``matched_factor_correlation`` between the two recovered factor matrices ``H``
      (factor-matching stability), and
    * ``adjusted_rand_index`` between the two recovered ``domain_id`` partitions
      (domain stability).

    A model that is robust to initialization recovers near-identical factors and
    domains regardless of seed, driving both means toward ``1.0``.
    """
    config = config or HarnessConfig()
    seeds = list(seeds) if seeds is not None else [0, 1, 2, 3]
    if len(seeds) < 2:
        raise ValueError("seed_init_stability needs at least two seeds to form a pair")

    synth = _generate(config, seed=data_seed)
    fits = [_fit(synth, config, requested_n_domains=config.n_domains, seed=s) for s in seeds]
    factor_mats = [f.H.astype(np.float64) for f in fits]
    domains = [f.domain_id.astype(np.int64) for f in fits]

    factor_pairs: list[float] = []
    domain_pairs: list[float] = []
    for i, j in combinations(range(len(seeds)), 2):
        factor_pairs.append(float(matched_factor_correlation(factor_mats[i], factor_mats[j])))
        domain_pairs.append(float(adjusted_rand_index(domains[i], domains[j])))

    return {
        "experiment": "seed_init_stability",
        "issue": 314,
        "config": asdict(config),
        "data_seed": data_seed,
        "seeds": seeds,
        "factor_stability_pairs": factor_pairs,
        "domain_ari_pairs": domain_pairs,
        "mean_factor_stability": _nanmean(factor_pairs),
        "mean_domain_ari_stability": _nanmean(domain_pairs),
    }


def robustness_to_k(
    config: HarnessConfig | None = None,
    *,
    k_grid: list[int] | None = None,
    stability_seeds: list[int] | None = None,
    edge_dropout: float = 0.0,
) -> dict:
    """#308 — domain robustness as the requested domain count ``k`` is mis-specified.

    Holds the synthetic data fixed (true domain count ``config.n_domains``) and
    sweeps the *requested* number of domains ``k`` over ``k_grid``.

    Note on the primary metric: in this generator the ground-truth domains are
    coordinate-derived random partitions that are **independent of the planted
    expression signal**, so GT-domain ``adjusted_rand_index`` is near-null at
    every ``k`` by construction (it is still reported as ``domain_ari_vs_gt`` for
    transparency, not as an accuracy claim). The recoverable, GT-free robustness
    signal is the partition's **self-consistency**: at each ``k`` the model is
    refit over ``stability_seeds`` (different initializations) and we report the
    mean pairwise ``adjusted_rand_index`` between the resulting ``domain_id``
    partitions (``domain_self_stability``). A robust ``k`` yields the same domain
    call regardless of seed; over-specified ``k`` fragments the partition and
    self-stability falls. ``edge_dropout`` optionally removes a fraction of
    spatial-graph edges first (mild neighborhood disruption), applied identically
    across the sweep so the curve isolates the ``k`` effect.
    """
    config = config or HarnessConfig()
    true_k = config.n_domains
    if k_grid is None:
        k_grid = [max(2, true_k - 2), max(2, true_k - 1), true_k, true_k + 1, true_k + 2]
    stability_seeds = list(stability_seeds) if stability_seeds is not None else [0, 1, 2]
    if len(stability_seeds) < 2:
        raise ValueError("robustness_to_k needs at least two stability_seeds to form a pair")

    synth = _generate(config, seed=stability_seeds[0])
    edges = _disrupt_edges(synth.edges, edge_dropout, seed=stability_seeds[0])
    gt_domains = synth.domain_id.astype(np.int64)

    curve: list[dict] = []
    for k in k_grid:
        partitions = [
            _fit(synth, config, requested_n_domains=int(k), seed=s, edges=edges).domain_id.astype(np.int64)
            for s in stability_seeds
        ]
        pair_aris = [
            float(adjusted_rand_index(partitions[i], partitions[j]))
            for i, j in combinations(range(len(stability_seeds)), 2)
        ]
        ari_vs_gt = float(adjusted_rand_index(gt_domains, partitions[0]))
        curve.append(
            {
                "requested_k": int(k),
                "domain_self_stability": _nanmean(pair_aris),
                "domain_ari_vs_gt": ari_vs_gt,
            }
        )

    finite = [
        (row["requested_k"], row["domain_self_stability"])
        for row in curve
        if np.isfinite(row["domain_self_stability"])
    ]
    best_k = max(finite, key=lambda kv: kv[1])[0] if finite else None
    return {
        "experiment": "robustness_to_k",
        "issue": 308,
        "config": asdict(config),
        "stability_seeds": stability_seeds,
        "edge_dropout": edge_dropout,
        "true_k": true_k,
        "k_grid": [int(k) for k in k_grid],
        "curve": curve,
        "best_k": best_k,
        "self_stability_at_true_k": next(
            (r["domain_self_stability"] for r in curve if r["requested_k"] == true_k), None
        ),
    }


def recovery_vs_batch(
    config: HarnessConfig | None = None,
    *,
    batch_grid: list[float] | None = None,
    seed: int = 0,
) -> dict:
    """#312 — factor/domain recovery vs increasing unwanted-variation strength.

    Uses the synthetic generator's additive-noise floor ``noise_sigma`` as a
    data-independent proxy for unwanted (nuisance/batch) variation: each grid
    point regenerates the instance at a larger ``noise_sigma`` (a stronger
    nuisance corruption of the planted signal) and refits. At each strength we
    report:

    * ``matched_factor_correlation`` of recovered ``H`` vs the planted GT factors,
    * domain ``adjusted_rand_index`` vs the GT domains.

    Both are expected to DEGRADE (trend downward) as the nuisance strength rises.
    """
    config = config or HarnessConfig()
    if batch_grid is None:
        batch_grid = [0.5, 5.0, 15.0, 30.0]

    curve: list[dict] = []
    for sigma in batch_grid:
        synth = _generate(config, seed=seed, noise_sigma=float(sigma))
        out = _fit(synth, config, requested_n_domains=config.n_domains, seed=seed)
        mfc = float(matched_factor_correlation(out.H.astype(np.float64), _ground_truth_factors(synth)))
        ari = float(adjusted_rand_index(synth.domain_id.astype(np.int64), out.domain_id.astype(np.int64)))
        curve.append({"batch_strength": float(sigma), "matched_factor_correlation": mfc, "domain_ari": ari})

    mfc_series = [row["matched_factor_correlation"] for row in curve]
    return {
        "experiment": "recovery_vs_batch",
        "issue": 312,
        "config": asdict(config),
        "seed": seed,
        "batch_grid": [float(s) for s in batch_grid],
        "curve": curve,
        "factor_recovery_drop": _finite_drop(mfc_series),
        "domain_ari_drop": _finite_drop([row["domain_ari"] for row in curve]),
    }


def json_safe(obj):
    """Recursively convert ``obj`` to valid JSON: non-finite floats become ``None``.

    ``json.dumps`` emits the non-standard ``NaN``/``Infinity`` tokens by default;
    drivers route results through this first so the on-disk JSON is portable.
    """
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    return obj


def _nanmean(values: list[float]) -> float:
    """Mean over the finite entries of ``values`` (``nan`` if none are finite)."""
    finite = [v for v in values if np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


def _finite_drop(series: list[float]) -> float:
    """First-minus-last difference over finite endpoints (``nan`` if unavailable).

    Positive when the series degrades (falls) from start to end of the sweep.
    """
    finite = [v for v in series if np.isfinite(v)]
    if len(finite) < 2:
        return float("nan")
    return float(finite[0] - finite[-1])
