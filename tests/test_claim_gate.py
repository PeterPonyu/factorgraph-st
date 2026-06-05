"""Tests for the repo-local claim-ledger + manuscript over-claim gate (#357).

Loads ``scripts/check_claim_gate.py`` directly (it is a script, not a package
module) and exercises: ledger parsing, the evidence requirement for non-planned
rows, the conservative manuscript over-claim heuristic, and an end-to-end run
that asserts the REAL repo ledger + manuscript currently pass the gate.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_GATE_PATH = _REPO_ROOT / "scripts" / "check_claim_gate.py"


def _load_gate():
    spec = importlib.util.spec_from_file_location("check_claim_gate", _GATE_PATH)
    assert spec and spec.loader, f"cannot load gate from {_GATE_PATH}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # so @dataclass can resolve cls.__module__
    spec.loader.exec_module(mod)
    return mod


gate = _load_gate()


_HEADER = (
    "| Claim | Required evidence | Current evidence | Missing evidence | Status |\n"
    "|---|---|---|---|---|\n"
)


def _ledger(*rows: str) -> str:
    return _HEADER + "".join(rows)


def test_parse_ledger_reads_five_columns():
    text = _ledger("| C one | req | cur | none | planned |\n")
    rows = gate.parse_ledger(text)
    assert len(rows) == 1
    assert rows[0].claim == "C one"
    assert rows[0].status == "planned"
    assert rows[0].is_planned


def test_planned_row_needs_no_evidence():
    rows = gate.parse_ledger(_ledger("| C | req | none | everything | planned |\n"))
    assert gate.check_ledger(rows) == []


def test_supported_row_with_evidence_passes():
    # concrete pointer (#384) in current-evidence, nothing missing -> no violation
    rows = gate.parse_ledger(
        _ledger("| C | req | merged in #384, tests/test_x.py | none | supported |\n")
    )
    assert gate.check_ledger(rows) == []


def test_supported_row_without_evidence_fails():
    rows = gate.parse_ledger(
        _ledger("| C | req | baseline references only | data + tests | supported |\n")
    )
    violations = gate.check_ledger(rows)
    # one for the missing pointer, one for the non-empty 'Missing evidence'
    assert len(violations) == 2
    assert any("concrete pointer" in v for v in violations)
    assert any("Missing evidence" in v for v in violations)


def test_supported_row_pointer_but_missing_nonempty_fails():
    rows = gate.parse_ledger(
        _ledger("| C | req | proven in #99 | still need replication | validated |\n")
    )
    violations = gate.check_ledger(rows)
    assert len(violations) == 1
    assert "Missing evidence" in violations[0]


def test_manuscript_overclaim_of_planned_claim_fails():
    rows = gate.parse_ledger(
        _ledger("| Method recovers spatial gene programs | req | none | impl | planned |\n")
    )
    manuscript = {
        "draft.md": "Our method recovers spatial gene programs across all sections."
    }
    violations = gate.check_manuscript(rows, manuscript)
    assert len(violations) == 1
    assert "still-'planned'" in violations[0]


def test_manuscript_hedged_sentence_passes():
    rows = gate.parse_ledger(
        _ledger("| Method recovers spatial gene programs | req | none | impl | planned |\n")
    )
    # 'planned' is a hedge token -> sentence is downgraded, not flagged
    manuscript = {
        "draft.md": "The method is planned to recover spatial gene programs."
    }
    assert gate.check_manuscript(rows, manuscript) == []


def test_real_repo_ledger_and_manuscript_pass_gate():
    """The shipped CLAIM_LEDGER.md (all 'planned') + hedged manuscript must PASS."""
    report = gate.run_gate(_REPO_ROOT / "CLAIM_LEDGER.md", _REPO_ROOT / "manuscript")
    assert report.checked_rows >= 1, "expected to parse real ledger rows"
    assert report.passed, report.render()


def test_main_exit_code_on_real_repo_is_zero():
    assert gate.main([]) == 0
