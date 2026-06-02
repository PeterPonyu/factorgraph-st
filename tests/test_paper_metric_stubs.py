"""
Tests for factorgraph-st per-repo metrics-stub contract.

Asserts that each fg-* stub in results/paper_metrics/ is:
  - valid JSON
  - contains all 12 required schema keys
  - has paper_claim_ready == False
  - has project == "factorgraph-st"
  - has readiness_status in the allowed set
"""

import json
import pathlib

import pytest

PAPER_METRICS_DIR = pathlib.Path(__file__).parent.parent / "results" / "paper_metrics"

REQUIRED_KEYS = {
    "asset_id",
    "project",
    "role",
    "readiness_status",
    "source_metric_reference",
    "source_script_reference",
    "source_exists",
    "metric_source_exists",
    "artifact_exists",
    "supports_safe_prose",
    "paper_claim_ready",
    "notes",
}

ALLOWED_READINESS = {"planned", "contract-only", "in-progress", "complete"}

EXPECTED_FG_STUBS = [
    "fg-f1_metrics_stub.json",
    "fg-f2_metrics_stub.json",
    "fg-f3_metrics_stub.json",
    "fg-t1_metrics_stub.json",
]


def _load_stub(filename: str) -> dict:
    path = PAPER_METRICS_DIR / filename
    assert path.exists(), f"Stub file not found: {path}"
    with path.open() as fh:
        return json.load(fh)


@pytest.mark.parametrize("filename", EXPECTED_FG_STUBS)
def test_stub_is_valid_json(filename):
    """Each fg-* stub must be parseable as JSON without error."""
    stub = _load_stub(filename)
    assert isinstance(stub, dict), f"{filename}: expected a JSON object at top level"


@pytest.mark.parametrize("filename", EXPECTED_FG_STUBS)
def test_stub_has_required_keys(filename):
    """Each fg-* stub must contain all 12 required schema keys."""
    stub = _load_stub(filename)
    missing = REQUIRED_KEYS - stub.keys()
    assert not missing, f"{filename}: missing required keys: {sorted(missing)}"


@pytest.mark.parametrize("filename", EXPECTED_FG_STUBS)
def test_stub_paper_claim_ready_is_false(filename):
    """paper_claim_ready must be false — stubs are not yet claim-ready."""
    stub = _load_stub(filename)
    assert stub["paper_claim_ready"] is False, (
        f"{filename}: expected paper_claim_ready=false, got {stub['paper_claim_ready']!r}"
    )


@pytest.mark.parametrize("filename", EXPECTED_FG_STUBS)
def test_stub_project_is_factorgraph_st(filename):
    """project field must identify this repo."""
    stub = _load_stub(filename)
    assert stub["project"] == "factorgraph-st", (
        f"{filename}: expected project='factorgraph-st', got {stub['project']!r}"
    )


@pytest.mark.parametrize("filename", EXPECTED_FG_STUBS)
def test_stub_readiness_status_is_valid(filename):
    """readiness_status must be one of the allowed lifecycle values."""
    stub = _load_stub(filename)
    assert stub["readiness_status"] in ALLOWED_READINESS, (
        f"{filename}: readiness_status {stub['readiness_status']!r} not in {ALLOWED_READINESS}"
    )


@pytest.mark.parametrize("filename", EXPECTED_FG_STUBS)
def test_stub_boolean_fields_are_false(filename):
    """All boolean gate fields must be False in stub state."""
    stub = _load_stub(filename)
    bool_fields = [
        "source_exists",
        "metric_source_exists",
        "artifact_exists",
        "supports_safe_prose",
        "paper_claim_ready",
    ]
    for field in bool_fields:
        assert stub[field] is False, (
            f"{filename}: expected {field}=false, got {stub[field]!r}"
        )


def test_all_expected_fg_stubs_present():
    """Directory must contain exactly the expected fg-* stub files."""
    present = {p.name for p in PAPER_METRICS_DIR.glob("fg-*_metrics_stub.json")}
    expected = set(EXPECTED_FG_STUBS)
    missing = expected - present
    assert not missing, f"Missing fg-* stubs: {sorted(missing)}"
