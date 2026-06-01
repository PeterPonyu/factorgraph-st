"""Regression test for #48: tests must collect under Python 3.10.

``tomllib`` was added to the stdlib in Python 3.11. The package declares
``requires-python = ">=3.10"`` in ``pyproject.toml``, so any test module
with a bare top-level ``import tomllib`` breaks ``pytest --collect-only``
on a declared-supported 3.10 interpreter with ``ImportError``.

The minimal fix is to import ``tomllib`` defensively, falling back to the
backport ``tomli`` on Python < 3.11:

    try:
        import tomllib
    except ImportError:  # Python < 3.11
        import tomli as tomllib

This regression test fails on the buggy main (bare ``import tomllib`` in
``tests/test_packaging_contract.py``) and passes once the fallback is in
place. It is a static-AST check so it runs on any Python version that
pytest itself can collect on, rather than relying on a 3.10 interpreter
being available in the test environment.
"""

from __future__ import annotations

import ast
import pathlib


def _has_import_error_handler(try_node: ast.Try) -> bool:
    """Return True if ``try_node`` catches ``ImportError`` (or bare ``except``)."""
    for handler in try_node.handlers:
        if handler.type is None:
            return True
        if isinstance(handler.type, ast.Name) and handler.type.id == "ImportError":
            return True
        if isinstance(handler.type, ast.Tuple) and any(
            isinstance(elt, ast.Name) and elt.id == "ImportError"
            for elt in handler.type.elts
        ):
            return True
    return False


def _bare_tomllib_imports(tree: ast.Module) -> list[int]:
    """Return line numbers of ``import tomllib`` NOT guarded by ``try/except ImportError``."""

    guarded_lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Try) and _has_import_error_handler(node):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Import):
                    for alias in sub.names:
                        if alias.name == "tomllib":
                            guarded_lines.add(sub.lineno)

    bare_lines: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "tomllib" and node.lineno not in guarded_lines:
                    bare_lines.append(node.lineno)
    return bare_lines


def test_no_bare_tomllib_import_breaks_py310_collection():
    """Bare ``import tomllib`` in test modules breaks ``--collect-only`` on Python 3.10.

    Every ``import tomllib`` in ``tests/`` must be wrapped in
    ``try/except ImportError`` with a ``tomli`` fallback so the test
    suite collects on the lowest declared-supported interpreter.
    """
    tests_dir = pathlib.Path(__file__).resolve().parent
    offenders: list[str] = []
    for path in sorted(tests_dir.rglob("test_*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            # If a test file is unparsable, that is a different bug; skip.
            continue
        for lineno in _bare_tomllib_imports(tree):
            offenders.append(f"{path.relative_to(tests_dir.parent)}:{lineno}")

    assert not offenders, (
        "Found bare `import tomllib` (no `try/except ImportError` fallback) at: "
        f"{offenders}. This breaks `pytest --collect-only` on Python 3.10, "
        "which is declared supported via `requires-python = \">=3.10\"`. "
        "Wrap with `try: import tomllib\\nexcept ImportError: import tomli as tomllib` "
        "and add `tomli; python_version < '3.11'` to the test extra."
    )
