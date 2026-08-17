"""Runnable entry point for snapshot generation.

The reusable implementation and its configuration (parameter grid, branches,
mesh/output paths) live in ``nsrom.snapshots.generation``; edit the constants at
the top of that module to change what is generated. This script only invokes it.

Usage:
    python scripts/generate_snapshots.py            # generate snapshots
    python scripts/generate_snapshots.py --preview  # plot the grid, no solves
"""
from nsrom.snapshots.generation import main

if __name__ == "__main__":
    main()
