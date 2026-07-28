#!/usr/bin/env python
"""
Publication-quality C_L(Re) pitchfork bifurcation diagram for the fluidic
pinball, rendered from the .npz written by compute_bifurcation_diagram.py.

Style (fonts, sizes, palette, column widths) comes from nsrom.plotting.style,
shared with every other figure script.

Reads (keys in the .npz):
    Re_c, mu_c              critical Reynolds number and leading real mu
    sym_Re, sym_CL          symmetric branch (split at Re_c by stability)
    pos_Re, pos_CL          asymmetric C_L > 0 branch
    neg_Re, neg_CL          asymmetric C_L < 0 branch
    pos_mirror, neg_mirror  whether a branch is a reflected fallback

Produces (PDF + PNG in --outdir):
    bifurcation_diagram.{pdf,png}

Usage:
    python plot_bifurcation_diagram.py
    python plot_bifurcation_diagram.py figures/bifurcation_diagram.npz \
        --outdir figs --width double --usetex
"""

import argparse
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nsrom.plotting.style import (
    set_style, resolve_width, SINGLE_COL, DOUBLE_COL, C_POS, C_NEG, C_SYM,
)


# ----------------------------------------------------------------------
# IO
# ----------------------------------------------------------------------

def read_npz(path):
    """-> dict with scalars as floats/bools and branches as float arrays."""
    d = np.load(path, allow_pickle=True)
    return {
        "Re_c": float(d["Re_c"]),
        "mu_c": float(d["mu_c"]),
        "sym_Re": np.asarray(d["sym_Re"], float),
        "sym_CL": np.asarray(d["sym_CL"], float),
        "pos_Re": np.asarray(d["pos_Re"], float),
        "pos_CL": np.asarray(d["pos_CL"], float),
        "neg_Re": np.asarray(d["neg_Re"], float),
        "neg_CL": np.asarray(d["neg_CL"], float),
        "pos_mirror": bool(d["pos_mirror"]),
        "neg_mirror": bool(d["neg_mirror"]),
    }


# ----------------------------------------------------------------------
# Figure: C_L(Re) bifurcation diagram
# ----------------------------------------------------------------------

def split_at(Re, CL, Re_c):
    """Split a curve at Re_c, inserting an interpolated junction point so the
    stable/unstable segments meet with no gap."""
    order = np.argsort(Re)
    Re, CL = Re[order], CL[order]
    cl_c = float(np.interp(Re_c, Re, CL))

    lo = Re <= Re_c
    hi = Re >= Re_c
    Re_lo = np.append(Re[lo], Re_c)
    CL_lo = np.append(CL[lo], cl_c)
    Re_hi = np.insert(Re[hi], 0, Re_c)
    CL_hi = np.insert(CL[hi], 0, cl_c)
    return (Re_lo, CL_lo), (Re_hi, CL_hi)


def plot_bifurcation(data, width):
    Re_c = data["Re_c"]
    fig, ax = plt.subplots(figsize=(width, 0.82 * width))

    # Computed symmetric branch, split at Re_c by stability.
    (Re_lo, CL_lo), (Re_hi, CL_hi) = split_at(
        data["sym_Re"], data["sym_CL"], Re_c)

    ax.plot(Re_lo, CL_lo, "-", color=C_SYM, lw=1.3, zorder=2,
            label="symmetric (stable)")
    ax.plot(Re_hi, CL_hi, "--", color=C_SYM, lw=1.3, zorder=2,
            label="symmetric (unstable)")

    # Asymmetric branches.
    Re_p, CL_p = data["pos_Re"], data["pos_CL"]
    Re_n, CL_n = data["neg_Re"], data["neg_CL"]

    if Re_p.size:
        lab = r"asymmetric, $C_L > 0$" + (
            " (reflected)" if data["pos_mirror"] else "")
        ax.plot(Re_p, CL_p, "-o", color=C_POS, ms=3, mew=0, lw=1.2,
                zorder=3, label=lab)

    if Re_n.size:
        lab = r"asymmetric, $C_L < 0$" + (
            " (reflected)" if data["neg_mirror"] else "")
        ax.plot(Re_n, CL_n, "-s", color=C_NEG, ms=3, mew=0, lw=1.2,
                zorder=3, label=lab)

    ax.axvline(Re_c, color="0.35", ls=":", lw=1.0, zorder=1,
               label=rf"$\mathrm{{Re}}_c = {Re_c:.2f}$")

    ax.set_xlabel(r"$\mathrm{Re}$")
    ax.set_ylabel(r"$C_L$")
    ax.legend(loc="upper left", handlelength=1.4,
              borderaxespad=0.4, labelspacing=0.35)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("npz", nargs="?",
                   default="figures/bifurcation_diagram.npz",
                   help="input .npz from compute_bifurcation_diagram.py")
    p.add_argument("--outdir", default="figures")
    p.add_argument("--width", default="single",
                help="preset ('single','double','linewidth','0.7linewidth') "
                        "or a width in inches, e.g. 5.79")
    p.add_argument("--usetex", action="store_true")
    args = p.parse_args()

    set_style(args.usetex)
    os.makedirs(args.outdir, exist_ok=True)
    W = resolve_width(args.width)
    data = read_npz(args.npz)
    print(f"Re_c = {data['Re_c']:.4f},  mu_c = {data['mu_c']:+.3e}")
    if data["pos_mirror"]:
        print("note: C_L>0 branch is a reflected fallback")
    if data["neg_mirror"]:
        print("note: C_L<0 branch is a reflected fallback")

    fig = plot_bifurcation(data, W)
    for ext in ("pdf", "png"):
        f = os.path.join(args.outdir, f"bifurcation_diagram.{ext}")
        fig.savefig(f)
        print(f"wrote {f}")


if __name__ == "__main__":
    main()
