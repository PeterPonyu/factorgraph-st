#!/usr/bin/env python
"""#357 — repo-local claim-ledger + manuscript over-claim gate.

A small, dependency-free guard so production-ready evidence discipline is
*mechanically* enforced inside this repo:

1. **Ledger gate** — parse ``CLAIM_LEDGER.md`` (a markdown table with columns
   ``Claim | Required evidence | Current evidence | Missing evidence | Status``).
   Any row whose ``Status`` is **not** ``planned`` (e.g. ``supported`` /
   ``validated``) MUST carry a concrete evidence pointer in ``Current evidence``
   (a PR/commit/test/figure path or dataset id) AND an empty / ``none``
   ``Missing evidence``. A non-planned claim with no real evidence FAILS.

2. **Manuscript guard** — scan ``manuscript/*.md`` for assertive achievement
   language that contradicts ledger status: a sentence that asserts a *planned*
   claim as already achieved (assertive verb + claim topic overlap, with no
   hedging) FAILS. Conservative by design — the current manuscript hedges every
   claim ("planned" / "intended" / "aims" / "will"), so it passes cleanly.

This gate is **independent** of the META-owned parent ``../scripts/check_claim_ledger.py``
and never imports or depends on it.

Usage::

    python scripts/check_claim_gate.py                 # check repo defaults
    python scripts/check_claim_gate.py --ledger X.md --manuscript-dir manuscript

Exit code 0 == PASS, 1 == FAIL (violations printed). Never silently passes.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# A status of exactly "planned" exempts a row from the evidence requirement.
PLANNED_STATUS = "planned"

# Values in "Missing evidence" that count as "nothing missing".
_EMPTY_TOKENS = {"", "-", "—", "none", "n/a", "na", "nothing"}

# Concrete evidence pointers accepted in "Current evidence" for a non-planned row.
# (PR/issue ref, commit sha, test path/name, figure/artifact path, dataset id.)
_EVIDENCE_PATTERNS = (
    re.compile(r"#\d+"),                                  # PR / issue reference
    re.compile(r"\b[0-9a-f]{7,40}\b"),                    # commit sha
    re.compile(r"\btest_[\w/]+|/tests?/|\btests?/"),      # test path / name
    re.compile(r"\.(?:png|pdf|svg|csv|json|npz|h5ad)\b"), # artifact / figure file
    re.compile(r"\bfigures?/|\bresults?/"),               # artifact directory
    re.compile(r"\b(?:GSE|GSM|SRR|E-MTAB-|EGAD)\d+", re.I),  # dataset accession id
)

# Assertive achievement language (claim asserted as already done).
_ASSERTIVE = (
    "achiev", "outperform", "recover", "improv", "demonstrat", "shows",
    "shown", "establish", "confirm", "prove", "proven", "surpass",
    "exceed", "attain", "validat", "we show", "results show",
)

# Hedging language that downgrades a sentence to non-assertive (still planned).
_HEDGE = (
    "planned", "plan ", "plans ", "intend", "aim", "will ", "would ",
    "could ", "can ", "may ", "might ", "propose", "proposal", "expect",
    "anticipat", "future", "not yet", "remain", "deterministic", "mvp",
    "baseline reference", "is intended", "seek", "goal",
)

_STOPWORDS = {
    "the", "and", "can", "for", "with", "over", "that", "this", "from",
    "into", "across", "while", "their", "than", "are", "its", "via", "such",
    "more", "less", "each", "any", "all", "not", "yet", "but", "out",
}


@dataclass
class ClaimRow:
    """One parsed row of the claim ledger."""

    claim: str
    required_evidence: str
    current_evidence: str
    missing_evidence: str
    status: str
    line_no: int = 0

    @property
    def is_planned(self) -> bool:
        return self.status.strip().lower() == PLANNED_STATUS


@dataclass
class GateReport:
    """Structured result of a gate run."""

    violations: list[str] = field(default_factory=list)
    checked_rows: int = 0
    checked_manuscripts: int = 0

    @property
    def passed(self) -> bool:
        return not self.violations

    def render(self) -> str:
        lines = [
            f"claim-gate: parsed {self.checked_rows} ledger row(s), "
            f"scanned {self.checked_manuscripts} manuscript file(s)."
        ]
        if self.passed:
            lines.append("PASS: claim ledger + manuscript consistent with evidence.")
        else:
            lines.append(f"FAIL: {len(self.violations)} violation(s):")
            lines.extend(f"  - {v}" for v in self.violations)
        return "\n".join(lines)


def parse_ledger(text: str) -> list[ClaimRow]:
    """Parse the 5-column markdown claim-ledger table into rows.

    Recognises the header row by its column names and skips the ``---`` separator.
    Rows with fewer than 5 cells are ignored (defensive against prose lines).
    """
    rows: list[ClaimRow] = []
    seen_header = False
    for i, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 5:
            continue
        lowered = [c.lower() for c in cells]
        if not seen_header:
            if "claim" in lowered[0] and "status" in lowered[-1]:
                seen_header = True
            continue
        if set("".join(cells)) <= set("-: "):  # markdown separator row
            continue
        rows.append(
            ClaimRow(
                claim=cells[0],
                required_evidence=cells[1],
                current_evidence=cells[2],
                missing_evidence=cells[3],
                status=cells[4],
                line_no=i,
            )
        )
    return rows


def _has_evidence_pointer(text: str) -> bool:
    return any(p.search(text) for p in _EVIDENCE_PATTERNS)


def _is_empty_missing(text: str) -> bool:
    return text.strip().lower() in _EMPTY_TOKENS


def check_ledger(rows: list[ClaimRow]) -> list[str]:
    """Return a list of violation strings for non-planned rows lacking evidence."""
    violations: list[str] = []
    for row in rows:
        if row.is_planned:
            continue
        loc = f"ledger row (line {row.line_no}, status '{row.status}')"
        if not _has_evidence_pointer(row.current_evidence):
            violations.append(
                f"{loc}: status is not 'planned' but 'Current evidence' has no "
                f"concrete pointer (PR/commit/test/figure/dataset): "
                f"{row.current_evidence!r}"
            )
        if not _is_empty_missing(row.missing_evidence):
            violations.append(
                f"{loc}: status is not 'planned' but 'Missing evidence' is "
                f"non-empty: {row.missing_evidence!r}"
            )
    return violations


def _keywords(text: str) -> set[str]:
    words = re.findall(r"[a-z][a-z\-]{2,}", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) >= 4}


def _split_sentences(text: str) -> list[str]:
    # Strip markdown headers/bullets to plain prose, then split on sentence enders.
    cleaned = re.sub(r"(?m)^[#>\-\*\d\.\)\s]+", " ", text)
    cleaned = cleaned.replace("\n", " ")
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if s.strip()]


def check_manuscript(rows: list[ClaimRow], texts: dict[str, str]) -> list[str]:
    """Flag manuscript sentences that assert a *planned* claim as achieved.

    Heuristic and conservative: a sentence is an over-claim only when it
    (a) shares >= 2 distinctive keywords with a planned claim,
    (b) contains assertive achievement language, and
    (c) contains no hedging language.
    """
    planned = [r for r in rows if r.is_planned]
    if not planned:
        return []
    claim_kw = [(r, _keywords(r.claim)) for r in planned]

    violations: list[str] = []
    for name, text in texts.items():
        for sentence in _split_sentences(text):
            low = sentence.lower()
            if sentence.rstrip().endswith("?"):
                continue  # a question is not an assertion of achievement
            if any(h in low for h in _HEDGE):
                continue
            if not any(a in low for a in _ASSERTIVE):
                continue
            sent_kw = _keywords(sentence)
            for _row, kw in claim_kw:
                overlap = sent_kw & kw
                if len(overlap) >= 2:
                    violations.append(
                        f"manuscript '{name}': asserts a still-'planned' claim as "
                        f"achieved (overlap {sorted(overlap)}): {sentence!r}"
                    )
                    break
    return violations


def run_gate(ledger_path: Path, manuscript_dir: Path | None) -> GateReport:
    """Run both gates and return a structured report."""
    report = GateReport()
    if not ledger_path.exists():
        report.violations.append(f"ledger not found: {ledger_path}")
        return report

    rows = parse_ledger(ledger_path.read_text(encoding="utf-8"))
    report.checked_rows = len(rows)
    if not rows:
        report.violations.append(f"no ledger rows parsed from {ledger_path}")
        return report

    report.violations.extend(check_ledger(rows))

    texts: dict[str, str] = {}
    if manuscript_dir and manuscript_dir.is_dir():
        for md in sorted(manuscript_dir.glob("*.md")):
            texts[md.name] = md.read_text(encoding="utf-8")
    report.checked_manuscripts = len(texts)
    report.violations.extend(check_manuscript(rows, texts))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Repo-local claim-ledger + manuscript gate (#357).")
    repo_root = Path(__file__).resolve().parent.parent
    parser.add_argument("--ledger", type=Path, default=repo_root / "CLAIM_LEDGER.md")
    parser.add_argument("--manuscript-dir", type=Path, default=repo_root / "manuscript")
    args = parser.parse_args(argv)

    report = run_gate(args.ledger, args.manuscript_dir)
    print(report.render())
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
