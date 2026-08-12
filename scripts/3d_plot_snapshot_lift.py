#!/usr/bin/env python
"""
3D scatter of C_L (or C_D) over the (Re, amplitude) parameter space, read
directly from the snapshot collection written by the generator.

Reads <snapdir>/metadata.npz (falls back to snapshot_dofs.npz), i.e. the
lightweight file -- no Firedrake needed.

Axes: x = Re, y = amplitude, z = C_L.  Points are coloured by branch, with
thin guide lines joining each branch along Re at fixed amplitude, and an
optional grey shadow on the floor showing (Re, amp) coverage.

Usage:
    python plot_snapshot_lift.py snapshots_sparse
    python plot_snapshot_lift.py snapshots_sparse --quantity drag
    python plot_snapshot_lift.py snapshots_sparse --elev 22 --azim -128 --no-lines
"""

import argparse
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.lines import Line2D

try:
    from nsrom.plotting.style import set_style, resolve_width, SINGLE_COL, DOUBLE_COL
except ImportError:
    SINGLE_COL, DOUBLE_COL = 3.40, 7.00

    def resolve_width(w):
        return {"single": SINGLE_COL, "double": DOUBLE_COL}.get(w, float(w))

    def set_style(usetex=False):
        plt.rcParams.update({
            "font.family": "serif", "font.size": 9, "axes.labelsize": 10,
            "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
            "legend.frameon": False, "axes.linewidth": 0.8,
            "savefig.dpi": 600, "savefig.bbox": "tight",
            "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.fontset": "cm",
        })
        if usetex:
            plt.rcParams.update({"text.usetex": True})


CB = {0: "0.45", 1: "#1f4e79", 2: "#b3541e"}
MK = {0: "o", 1: "^", 2: "v"}


def read(snapdir):
    path = os.path.join(snapdir, "metadata.npz")
    if not os.path.exists(path):
        path = os.path.join(snapdir, "snapshot_dofs.npz")
    if not os.path.exists(path):
        raise SystemExit(f"no metadata.npz or snapshot_dofs.npz in {snapdir}")

    d = np.load(path, allow_pickle=True)
    par = np.asarray(d["parameters"], float)
    n = par.shape[0]

    bid = np.asarray(d["branch_ids"], int) if "branch_ids" in d else np.zeros(n, int)
    if "branch_names_keys" in d:
        names = dict(zip(np.asarray(d["branch_names_keys"], int),
                         [str(v) for v in d["branch_names_values"]]))
    else:
        names = {k: f"branch {k}" for k in np.unique(bid)}

    return {
        "Re": par[:, 0],
        "amp": par[:, 1],
        "CL": np.asarray(d["lift_coefficients"], float),
        "CD": np.asarray(d["drag_coefficients"], float),
        "bid": bid,
        "names": names,
        "path": path,
    }


def plot3d(data, y, zlabel, width, elev, azim, lines, shadow, swap=False,
           flip_re=False, flip_amp=False):
    Re, amp, bid, names = data["Re"], data["amp"], data["bid"], data["names"]
    if swap:
        X, Y = amp, Re
        xlabel, ylabel = "amplitude", r"$\mathrm{Re}$"
    else:
        X, Y = Re, amp
        xlabel, ylabel = r"$\mathrm{Re}$", "amplitude"

    fig = plt.figure(figsize=(width, 0.80 * width))
    ax = fig.add_subplot(111, projection="3d")

    finite = np.isfinite(y)
    zmin = float(np.min(y[finite])) if finite.any() else 0.0
    zmax = float(np.max(y[finite])) if finite.any() else 1.0
    pad = 0.08 * max(zmax - zmin, 1e-12)
    zfloor = zmin - pad

    if shadow:
        ax.scatter(X, Y, np.full_like(X, zfloor), c="0.85", s=3,
                   marker=".", linewidths=0, depthshade=False, zorder=0)

    for k in sorted(np.unique(bid)):
        mk = bid == k
        if lines:
            for v in np.unique(Y[mk]):
                m = mk & np.isclose(Y, v)
                if m.sum() < 2:
                    continue
                o = np.argsort(X[m])
                ax.plot(X[m][o], Y[m][o], y[m][o], "-",
                        color=CB.get(k, "0.3"), lw=0.6, alpha=0.55, zorder=1)
        ax.scatter(X[mk], Y[mk], y[mk], color=CB.get(k, "0.3"),
                   marker=MK.get(k, "o"), s=11, linewidths=0.2,
                   edgecolors="white", depthshade=False, zorder=2)

    if np.any(np.isclose(amp, 0.0)):
        xs = [0.0, 0.0] if swap else [X.min(), X.max()]
        ys = [Y.min(), Y.max()] if swap else [0.0, 0.0]
        ax.plot(xs, ys, [0.0, 0.0], color="0.6", lw=0.7, ls=":", zorder=1)

    ax.set_xlabel(xlabel, labelpad=2)
    ax.set_ylabel(ylabel, labelpad=2)
    ax.set_zlabel(zlabel, labelpad=2)
    ax.set_zlim(zfloor, zmax + pad)
    if flip_re:
        (ax.invert_yaxis if swap else ax.invert_xaxis)()
    if flip_amp:
        (ax.invert_xaxis if swap else ax.invert_yaxis)()
    ax.view_init(elev=elev, azim=azim)
    ax.tick_params(pad=1)
    try:
        ax.set_box_aspect((1.0, 0.85, 0.65))
    except AttributeError:
        pass
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0.0)
        pane._axinfo["grid"].update(color="0.9", linewidth=0.5)

    handles = [Line2D([], [], color=CB.get(k, "0.3"), marker=MK.get(k, "o"),
                      ms=4, mew=0, lw=0.8, label=names.get(k, f"branch {k}"))
               for k in sorted(np.unique(bid))]
    ax.legend(handles=handles, loc="upper left", handlelength=1.6,
              labelspacing=0.3, borderaxespad=0.0)

    fig.tight_layout()
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("snapdir", nargs="?", default="snapshots_sparse")
    p.add_argument("--quantity", choices=("lift", "drag"), default="lift")
    p.add_argument("--outdir", default="figures")
    p.add_argument("--width", default="single")
    p.add_argument("--elev", type=float, default=20.0)
    p.add_argument("--azim", type=float, default=-130.0)
    p.add_argument("--swap", action="store_true",
                   help="put amplitude on x and Re on y")
    p.add_argument("--flip-re", action="store_true",
                   help="reverse the Re axis (decreasing outward)")
    p.add_argument("--flip-amp", action="store_true",
                   help="reverse the amplitude axis")
    p.add_argument("--no-lines", dest="lines", action="store_false")
    p.add_argument("--no-shadow", dest="shadow", action="store_false")
    p.add_argument("--usetex", action="store_true")
    a = p.parse_args()

    set_style(a.usetex)
    os.makedirs(a.outdir, exist_ok=True)

    d = read(a.snapdir)
    y = d["CL"] if a.quantity == "lift" else d["CD"]
    zlabel = r"$C_L$" if a.quantity == "lift" else r"$C_D$"

    print(f"read {d['path']}")
    print(f"  {y.size} snapshots, {np.unique(d['Re']).size} Re x "
          f"{np.unique(d['amp']).size} amp")
    for k in sorted(np.unique(d["bid"])):
        m = d["bid"] == k
        print(f"  {d['names'].get(k, k):<12} {m.sum():4d} snapshots, "
              f"Re in [{d['Re'][m].min():.2f}, {d['Re'][m].max():.2f}], "
              f"amp in [{d['amp'][m].min():+.3f}, {d['amp'][m].max():+.3f}], "
              f"z in [{np.nanmin(y[m]):+.4e}, {np.nanmax(y[m]):+.4e}]")
    if np.isnan(y).any():
        print(f"  WARNING: {int(np.isnan(y).sum())} NaN values")

    fig = plot3d(d, y, zlabel, resolve_width(a.width),
                 a.elev, a.azim, a.lines, a.shadow, a.swap,
                 a.flip_re, a.flip_amp)
    tag = "".join(t for t, on in (("_swap", a.swap), ("_flipre", a.flip_re),
                                  ("_flipamp", a.flip_amp)) if on)
    for ext in ("pdf", "png"):
        f = os.path.join(a.outdir, f"snapshot_{a.quantity}_3d{tag}.{ext}")
        fig.savefig(f)
        print(f"wrote {f}")


if __name__ == "__main__":
    main()
