"""Fast, seeded tests for the synthetic robustness harness (#314 / #308 / #312).

Each test runs the harness on a tiny synthetic instance and asserts the curve is
finite and well-shaped and exhibits the expected trend:

* #314 — recovered factors/domains are stable (reproducible) across model inits.
* #308 — domain self-consistency peaks at the matched ``k`` and degrades when
  ``k`` is heavily over-specified.
* #312 — factor recovery degrades monotonically as nuisance strength rises.

The harness is fully deterministic in its seeds, so determinism is also asserted.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

# The harness lives under scripts/experiments/, which pytest does not put on the
# path (only src/ is, via pyproject pythonpath). Add it explicitly.
_EXPERIMENTS = Path(__file__).resolve().parents[1] / "scripts" / "experiments"
if str(_EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS))

from robustness_harness import (  # noqa: E402
    HarnessConfig,
    json_safe,
    recovery_vs_batch,
    robustness_to_k,
    seed_init_stability,
)

# Tiny config shared across tests: fast (<1s total) yet enough signal to be stable.
_CONFIG = HarnessConfig()


def test_seed_init_stability_high_and_well_shaped() -> None:
    """#314: factor and domain recovery are reproducible across init seeds."""
    seeds = [0, 1, 2]
    result = seed_init_stability(_CONFIG, seeds=seeds)

    n_pairs = len(seeds) * (len(seeds) - 1) // 2
    assert len(result["factor_stability_pairs"]) == n_pairs
    assert len(result["domain_ari_pairs"]) == n_pairs
    assert all(math.isfinite(v) for v in result["factor_stability_pairs"])
    assert all(math.isfinite(v) for v in result["domain_ari_pairs"])

    # Pairwise correlations/ARIs are bounded; stability should be high.
    assert all(0.0 <= v <= 1.0 for v in result["factor_stability_pairs"])
    assert result["mean_factor_stability"] > 0.8
    assert result["mean_domain_ari_stability"] > 0.5


def test_seed_init_stability_deterministic() -> None:
    """#314: identical seeds -> bitwise-identical JSON-safe result."""
    a = seed_init_stability(_CONFIG, seeds=[0, 1, 2])
    b = seed_init_stability(_CONFIG, seeds=[0, 1, 2])
    assert json_safe(a) == json_safe(b)


def test_robustness_to_k_peaks_at_true_k() -> None:
    """#308: domain self-stability is highest at the matched k and finite throughout."""
    result = robustness_to_k(_CONFIG)
    curve = result["curve"]

    assert len(curve) == len(result["k_grid"])
    stabilities = [row["domain_self_stability"] for row in curve]
    assert all(math.isfinite(v) and -1.0 <= v <= 1.0 for v in stabilities)
    assert all(math.isfinite(row["domain_ari_vs_gt"]) for row in curve)

    # "Stability high at matched k": the self-stability curve peaks at the true k.
    assert result["best_k"] == result["true_k"]
    assert result["self_stability_at_true_k"] > 0.5

    # Heavily over-specified k fragments the partition -> lower self-stability.
    stab_at_true = result["self_stability_at_true_k"]
    stab_at_max_k = next(
        row["domain_self_stability"] for row in curve if row["requested_k"] == max(result["k_grid"])
    )
    assert stab_at_true > stab_at_max_k


def test_robustness_to_k_edge_dropout_runs() -> None:
    """#308: mild neighborhood disruption keeps the curve finite and well-shaped."""
    result = robustness_to_k(_CONFIG, k_grid=[3, 4, 5], edge_dropout=0.1)
    assert len(result["curve"]) == 3
    assert all(math.isfinite(row["domain_self_stability"]) for row in result["curve"])


def test_robustness_to_k_deterministic() -> None:
    """#308: repeated calls are identical."""
    a = robustness_to_k(_CONFIG)
    b = robustness_to_k(_CONFIG)
    assert json_safe(a) == json_safe(b)


def test_recovery_vs_batch_degrades_monotonically() -> None:
    """#312: factor recovery falls monotonically as nuisance strength rises."""
    batch_grid = [0.5, 5.0, 15.0, 30.0]
    result = recovery_vs_batch(_CONFIG, batch_grid=batch_grid)
    curve = result["curve"]

    assert len(curve) == len(batch_grid)
    assert [row["batch_strength"] for row in curve] == batch_grid

    mfc = [row["matched_factor_correlation"] for row in curve]
    assert all(math.isfinite(v) for v in mfc)
    # Strictly decreasing factor recovery with rising nuisance strength.
    assert all(mfc[i] > mfc[i + 1] for i in range(len(mfc) - 1))
    # Reported first-minus-last drop is positive (net degradation).
    assert result["factor_recovery_drop"] > 0.3


def test_recovery_vs_batch_deterministic() -> None:
    """#312: repeated calls are identical."""
    a = recovery_vs_batch(_CONFIG, batch_grid=[0.5, 10.0])
    b = recovery_vs_batch(_CONFIG, batch_grid=[0.5, 10.0])
    assert json_safe(a) == json_safe(b)
