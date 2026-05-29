"""FactorGraph-ST top-level package.

The package exposes schema validators, a deterministic synthetic generator, and
lightweight numpy MVP model/evaluation helpers. Performance and biology claims
remain planned (see CLAIM_LEDGER.md).
"""

from factorgraph_st.repro import set_seed

__version__ = "0.0.0.dev0"

__all__ = ["__version__", "set_seed"]
