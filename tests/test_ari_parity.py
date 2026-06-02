"""Numerical-parity guard for the vectorized adjusted_rand_index (#100/#104).

Asserts the np.add.at contingency-table implementation equals an independent
brute-force pair-counting reference on fixtures + edge cases, so the perf
change is provably behavior-preserving.

Also pins the #179 ground-truth ARI ingestion in the real-data runner: ARI is
computed against the per-spot GT obs column WHEN PRESENT (auto-detecting
``layer_guess`` / ``ground_truth_domain``) and SKIPPED when absent, never
fabricating labels nor joining across donors.
"""
import importlib.util
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

from factorgraph_st.eval.metrics import adjusted_rand_index

_RUNNER = (
    Path(__file__).resolve().parents[1] / "scripts" / "run_real_factorgraph.py"
)


def _load_runner():
    spec = importlib.util.spec_from_file_location("_run_real_factorgraph", _RUNNER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ref_ari(a: np.ndarray, b: np.ndarray) -> float:
    """Independent O(n^2) pair-counting Adjusted Rand Index reference."""
    n = len(a)
    if n < 2:
        return float("nan")
    tp = fp = fn = tn = 0
    for i, j in combinations(range(n), 2):
        same_a = a[i] == a[j]
        same_b = b[i] == b[j]
        if same_a and same_b:
            tp += 1
        elif same_a and not same_b:
            fp += 1
        elif not same_a and same_b:
            fn += 1
        else:
            tn += 1
    index = tp
    expected = (tp + fp) * (tp + fn) / (tp + fp + fn + tn)
    maximum = 0.5 * ((tp + fp) + (tp + fn))
    denom = maximum - expected
    if denom == 0:
        return float("nan")
    return float((index - expected) / denom)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_vectorized_ari_matches_bruteforce(seed: int) -> None:
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 4, size=40)
    b = rng.integers(0, 5, size=40)
    got = adjusted_rand_index(a, b)
    ref = _ref_ari(a, b)
    assert np.isclose(got, ref, atol=1e-9), f"vectorized={got} ref={ref}"


def test_identical_partitions_is_one() -> None:
    a = np.array([0, 0, 1, 1, 2, 2])
    assert np.isclose(adjusted_rand_index(a, a.copy()), 1.0)


def test_degenerate_inputs_return_nan() -> None:
    # n < 2 and the all-one-cluster case are not-evaluable, not "perfect" (#79).
    assert np.isnan(adjusted_rand_index(np.array([0]), np.array([0])))
    assert np.isnan(adjusted_rand_index(np.zeros(5, int), np.zeros(5, int)))


# --- #179: ground-truth ARI ingestion in the real-data runner ----------------

ad = pytest.importorskip("anndata")


def _adata_with_obs(obs: dict | None, n: int = 12):
    """Minimal AnnData carrying optional obs columns (X/obsm unused by ARI)."""
    import pandas as pd  # noqa: PLC0415

    X = np.zeros((n, 3), dtype=np.float32)
    obs_df = pd.DataFrame(obs, index=[str(i) for i in range(n)]) if obs else None
    return ad.AnnData(X=X, obs=obs_df)


def test_gt_ari_skipped_when_absent():
    """No GT obs column → ARI is skipped (None), never fabricated (#179)."""
    mod = _load_runner()
    adata = _adata_with_obs({"in_tissue": np.ones(12, dtype=int)})  # Br2719-like
    domain_id = np.array([0, 1] * 6)
    ari, key, n_labeled = mod._compute_gt_ari(adata, domain_id, None)
    assert ari is None and key is None and n_labeled == 0
    # And the runner's auto-detect resolver agrees there is no GT key.
    assert mod._resolve_gt_key(adata, None) is None


def test_gt_ari_computed_when_layer_guess_present():
    """``layer_guess`` present and perfectly aligned → ARI == 1.0 (#179)."""
    mod = _load_runner()
    labels = np.array(["L1", "L1", "L2", "L2", "L3", "L3"] * 2)
    adata = _adata_with_obs({"layer_guess": labels}, n=12)
    # domain_id is a relabeling of the same partition → perfect ARI.
    domain_id = np.array([5, 5, 8, 8, 2, 2] * 2)
    ari, key, n_labeled = mod._compute_gt_ari(adata, domain_id, None)
    assert key == "layer_guess"
    assert n_labeled == 12
    assert ari == pytest.approx(1.0)
    # Matches the library metric computed directly on encoded labels.
    _, gt_codes = np.unique(labels, return_inverse=True)
    assert ari == pytest.approx(adjusted_rand_index(gt_codes, domain_id))


def test_gt_ari_autodetects_ground_truth_domain():
    """Maynard prep writes ``ground_truth_domain`` → auto-detected (#179)."""
    mod = _load_runner()
    labels = np.array(["WM", "WM", "Layer1", "Layer1", "Layer2", "Layer2"] * 2)
    adata = _adata_with_obs({"ground_truth_domain": labels}, n=12)
    domain_id = np.array([1, 1, 0, 0, 2, 2] * 2)
    ari, key, n_labeled = mod._compute_gt_ari(adata, domain_id, None)
    assert key == "ground_truth_domain"
    assert n_labeled == 12
    assert np.isfinite(ari)


def test_gt_ari_drops_na_spots():
    """NA/blank GT spots are excluded from BOTH partitions (no fabrication)."""
    mod = _load_runner()
    labels = np.array(["L1", "L1", "NA", "L2", "L2", "nan"])
    adata = _adata_with_obs({"layer_guess": labels}, n=6)
    domain_id = np.array([3, 3, 9, 7, 7, 1])
    ari, key, n_labeled = mod._compute_gt_ari(adata, domain_id, None)
    assert key == "layer_guess"
    assert n_labeled == 4  # two NA-like spots dropped
    assert np.isfinite(ari)


def test_gt_ari_explicit_missing_key_raises():
    """An explicit --gt-obs-key that is absent fails loudly (no silent skip)."""
    mod = _load_runner()
    adata = _adata_with_obs({"layer_guess": np.array(["L1", "L2"] * 6)}, n=12)
    with pytest.raises(KeyError):
        mod._compute_gt_ari(adata, np.zeros(12, dtype=int), "does_not_exist")


# --- domain-quality metric suite wiring in the real-data runner ---------------

import types  # noqa: E402


def _fake_out(domain_id: np.ndarray, embedding: np.ndarray):
    """Minimal stand-in for the model output (only domain_id + H are used)."""
    return types.SimpleNamespace(domain_id=domain_id, H=embedding)


def _line_edges(n: int) -> np.ndarray:
    src = np.arange(n - 1, dtype=np.int64)
    dst = np.arange(1, n, dtype=np.int64)
    return np.array([np.concatenate([src, dst]), np.concatenate([dst, src])], dtype=np.int64)


def test_gt_domain_metrics_skipped_when_absent():
    """No GT obs column -> the whole suite is skipped (None), never fabricated."""
    mod = _load_runner()
    adata = _adata_with_obs({"in_tissue": np.ones(12, dtype=int)})  # Br2719-like
    out = _fake_out(np.array([0, 1] * 6), np.zeros((12, 2), dtype=np.float64))
    assert mod._compute_gt_domain_metrics(adata, out.domain_id, out.H, _line_edges(12), None) is None


def test_gt_domain_metrics_perfect_prediction():
    """A perfect (relabel-only) prediction scores NMI=Dice=boundary-F1=1.0."""
    mod = _load_runner()
    layers = np.array(["L1"] * 4 + ["L2"] * 4 + ["L3"] * 4)
    adata = _adata_with_obs({"layer_guess": layers}, n=12)
    # domain_id is a bijective relabel of the GT partition (contiguous blocks).
    domain_id = np.array([5] * 4 + [8] * 4 + [2] * 4, dtype=np.int64)
    # Well-separated per-domain embedding (small jitter so the within-cluster
    # dispersion is non-zero and Calinski-Harabasz is well defined).
    centers = np.array([[0.0, 0.0], [40.0, 0.0], [0.0, 40.0]])
    embedding = np.repeat(centers, 4, axis=0) + np.random.default_rng(0).normal(scale=0.3, size=(12, 2))
    out = _fake_out(domain_id, embedding)

    suite = mod._compute_gt_domain_metrics(adata, out.domain_id, out.H, _line_edges(12), None)
    assert suite is not None
    assert suite["nmi_domain"] == pytest.approx(1.0)
    assert suite["weighted_dice_domain"] == pytest.approx(1.0)
    assert suite["boundary_f1_domain"] == pytest.approx(1.0)
    assert suite["boundary_precision_domain"] == pytest.approx(1.0)
    assert suite["boundary_recall_domain"] == pytest.approx(1.0)
    assert np.isfinite(suite["silhouette_domain"])
    assert np.isfinite(suite["calinski_harabasz_domain"])


def test_gt_domain_metrics_drops_na_spots():
    """NA/blank GT spots are excluded before any metric is computed."""
    mod = _load_runner()
    layers = np.array(["L1", "L1", "NA", "L2", "L2", "nan"])
    adata = _adata_with_obs({"layer_guess": layers}, n=6)
    domain_id = np.array([3, 3, 9, 7, 7, 1], dtype=np.int64)
    embedding = np.array([[0.0], [0.1], [9.0], [5.0], [5.1], [1.0]])
    out = _fake_out(domain_id, embedding)
    suite = mod._compute_gt_domain_metrics(adata, out.domain_id, out.H, _line_edges(6), None)
    assert suite is not None
    # Four labeled spots ([L1,L1,L2,L2]) remain; the relabeling is perfect.
    assert suite["nmi_domain"] == pytest.approx(1.0)
    assert suite["weighted_dice_domain"] == pytest.approx(1.0)
