"""Shared loader for the ``scripts/tables/*.py`` table generators under test.

The generators are standalone scripts (not an installed package), so tests load
them by file path via importlib. Each generator's own ``sys.path`` shim runs
first during ``exec_module``, registering ``scripts/tables`` on the path so its
``from table_emit import ...`` resolves — including for ``table_emit`` itself.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_TABLES_DIR = Path(__file__).resolve().parents[2] / "scripts" / "tables"


def load_script(stem: str) -> ModuleType:
    """Import ``scripts/tables/<stem>.py`` as a uniquely-named module."""
    path = _TABLES_DIR / f"{stem}.py"
    if str(_TABLES_DIR) not in sys.path:
        sys.path.insert(0, str(_TABLES_DIR))
    spec = importlib.util.spec_from_file_location(f"_tables_{stem}", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so @dataclass (which looks up sys.modules[__module__]
    # on Python 3.13) and any self-referential imports resolve cleanly.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod
