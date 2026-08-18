"""Thin entry point / compatibility layer for the sweep implementation.

The implementation now lives in ``nsrom.bifurcation.sweep``. The original
``scripts/sweep.py`` defined only importable functions and had no ``__main__``
body, so there is no standalone workload to invoke here; this module simply
re-exports the package API so existing ``from sweep import ...`` usages and
``python scripts/sweep.py`` keep working. New code should import from
``nsrom.bifurcation.sweep``.
"""
from nsrom.bifurcation.sweep import *  # noqa: F401,F403
