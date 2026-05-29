"""Determinism control for FactorGraph-ST.

Closes #87. The package already threads explicit ``np.random.default_rng(seed)``
through the synthetic generator and ``fit_transform``, but global state from
the standard library ``random`` module and optionally PyTorch is uncontrolled.
``set_seed`` provides a single entry point that seeds every random source the
package can reach without taking on hard dependencies (torch stays optional).
"""

from __future__ import annotations

import random as _py_random
import sys

import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python ``random``, NumPy's legacy global RNG, and torch if loaded.

    The package's own RNG plumbing already takes ``seed`` explicitly through
    ``generate_instance`` and ``fit_transform``; this helper additionally pins
    the **global** random sources that downstream code (notebooks, scripts,
    third-party callers) may rely on.

    PyTorch is seeded only if it has already been imported in the current
    process, so ``factorgraph_st`` retains its numpy-only dependency set.
    """
    s = int(seed)
    _py_random.seed(s)
    np.random.seed(s)
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.manual_seed(s)
        cuda = getattr(torch, "cuda", None)
        if cuda is not None and getattr(cuda, "is_available", lambda: False)():
            cuda.manual_seed_all(s)
