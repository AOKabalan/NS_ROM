#!/usr/bin/env python
"""
Regenerate every paper figure in one go.

Runs each plotting script in turn. They read the .npz / .csv produced by the
compute scripts, so this is FAST -- it does not re-run any FOM solves. Use it
after editing nsrom/plotting/style.py to restyle every figure at once.

    python make_figures.py                 # draft: mathtext, single column
    python make_figures.py --usetex        # final: real LaTeX fonts
    python make_figures.py --width double --usetex

Per-figure arguments (input paths, --xlabel, --zoom, ...) live in the FIGURES
list below; the flags above are passed to every script that accepts them.
"""

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# (script filename, list of args specific to that figure)
FIGURES = [
    ("pinball_geometry_figure.py", ["mesh/mid_pinball.msh", "--width", "linewidth"]),
    ("plot_bifurcation_diagram.py", ["--width", "0.7linewidth"]),
    ("plot_critical_curve.py",      ["--width", "linewidth"]),
    ("plot_cumulative_energy.py", ["--width", "linewidth"]),
]

def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--usetex", action="store_true",
                   help="render all figures through a real LaTeX install")
    p.add_argument("--width", choices=["single", "double"],
                   help="column width applied to every figure")
    p.add_argument("--outdir", help="output directory applied to every figure")
    args = p.parse_args()

    common = []
    if args.usetex:
        common.append("--usetex")
    if args.width:
        common += ["--width", args.width]
    if args.outdir:
        common += ["--outdir", args.outdir]

    for script, extra in FIGURES:
        cmd = [sys.executable, str(HERE / script), *extra, *common]
        print("\n»", " ".join(cmd))
        subprocess.run(cmd, check=True)

    print("\nAll figures regenerated.")


if __name__ == "__main__":
    main()
