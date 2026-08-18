"""Compatibility shim — clustering moved to ``nsrom.clustering.building``.

Retained because the manuscript figure directory ``paper_figures/`` imports
``from nsrom.cluster_building import ...`` and is out of scope for this refactor.
New code should import from ``nsrom.clustering.building``.
"""
from nsrom.clustering.building import *  # noqa: F401,F403
