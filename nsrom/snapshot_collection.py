"""Compatibility shim — snapshot collection moved to ``nsrom.snapshots.collection``.

Retained because the manuscript figure directory ``paper_figures/`` and the
manually-run ``scripts/fom_bifurcation_diagram_fom.py`` import
``from nsrom.snapshot_collection import ...`` and are out of scope for this
refactor. New code should import from ``nsrom.snapshots.collection``.
"""
from nsrom.snapshots.collection import *  # noqa: F401,F403
from nsrom.snapshots.collection import __all__  # noqa: F401
