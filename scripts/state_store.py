"""Compatibility shim — the state store now lives in ``nsrom.io.state_store``.

Retained only so ``from state_store import ...`` keeps resolving for the
manuscript figure directories (``section_6_figures/`` and ``paper_figures/``),
which add ``scripts/`` to ``sys.path`` and are out of scope for this refactor.
New code should import from ``nsrom.io.state_store`` directly.
"""
from nsrom.io.state_store import StateWriter, StatePoint, StateStore  # noqa: F401
