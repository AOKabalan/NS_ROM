"""Compatibility shim — the local-ROM implementation moved to ``nsrom.rom.local``.

Retained because ``run_all.sh`` / ``run_gaps.sh`` preflight-check the package with
``python -c "import nsrom, nsrom.local_rom, ..."`` (the old path), and those
operational drivers are out of scope for this refactor. Re-exports the full
public API (including re-exported dependencies such as ``SubMeshDEIM``). New code
should import from ``nsrom.rom.local``.
"""
from nsrom.rom.local import *  # noqa: F401,F403
