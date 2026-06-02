"""Make ``scripts/data/fetch_datasets.py`` importable as a module.

The fetch surface lives under ``scripts/`` (not the installed package), so add
the ``scripts/data`` directory to ``sys.path`` for these tests. Network-free.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS_DATA = Path(__file__).resolve().parents[2] / "scripts" / "data"
if str(_SCRIPTS_DATA) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DATA))
