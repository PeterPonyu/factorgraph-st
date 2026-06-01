"""Network-free smoke tests for the dataset fetch CLI (closes #52).

Covers the deterministic, network-free contract of
``scripts/data/fetch_datasets.py``:

* ``--list`` returns 0 and emits every registry id.
* ``--dry-run`` plans a fetch without importing squidpy / touching the network.
* an unverified-URL ``--download`` raises ``NotImplementedError`` instead of
  writing placeholder bytes.

``squidpy`` / ``scipy`` are never imported by these paths, so the test runs in
the core install (numpy + pytest) with no optional data extras.
"""

from __future__ import annotations

import sys

import pytest
from fetch_datasets import DATASETS, fetch, main


def test_list_is_network_free(capsys):
    assert main(["--list"]) == 0
    out = capsys.readouterr().out
    for ds_id in DATASETS:
        assert ds_id in out


def test_dry_run_plans_without_network(capsys):
    assert main(["--dataset", "visium_mouse_brain_pair", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "[plan]" in out
    # squidpy must not have been imported by the dry-run path.
    assert "squidpy" not in sys.modules


def test_no_args_prints_help(capsys):
    assert main([]) == 0
    assert "usage" in capsys.readouterr().out.lower()


def test_unknown_dataset_errors():
    assert main(["--dataset", "does_not_exist"]) == 2


def test_unverified_download_refuses_placeholder():
    # cervilla_2026_visium has no verified direct URL -> must raise, never stub.
    ds = DATASETS["cervilla_2026_visium"]
    assert ds.url_verified is False
    with pytest.raises(NotImplementedError, match="URL UNVERIFIED"):
        fetch(ds, download=True)


def test_every_dataset_tracks_an_issue():
    # Each registry entry must reference at least one GitHub tracking issue.
    for ds in DATASETS.values():
        assert ds.issues, f"{ds.id} has no tracking issue"


def test_registry_covers_all_data_issues():
    # Sanity: the registry collectively references every [data] issue in scope.
    covered = {n for ds in DATASETS.values() for n in ds.issues}
    expected = {32, 33, 34, 35, 36, 37, 38, 39, 41, 42, 43, 44, 45, 53, 132, 133}
    assert expected.issubset(covered)
