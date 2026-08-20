#!/usr/bin/env python
"""
Plot C_L (or C_D) over the (Re, amplitude) parameter space, read directly from
the snapshot collection written by the generator.

Reads <snapdir>/metadata.npz (falls back to snapshot_dofs.npz), i.e. the
lightweight file -- no Firedrake needed.

Panels:
    (a) C_L vs Re, one curve per amplitude, coloured by amplitude
    (b) C_L vs amplitude, one curve per Re (subsampled), coloured by Re
    (c) (Re, amp) coverage, coloured by C_L, marker by branch

In one-parameter mode (a single amplitude) only panel (a) is drawn.

Usage:
    python plot_snapshot_lift.py snapshots_sparse
    python plot_snapshot_lift.py snapshots_sparse --quantity drag --outdir figs
"""

import argparse
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, "render"))
try:
    from style import set_style, resolve_width, SINGLE_COL, DOUBLE_COL
except ImportError:
    SINGLE_COL, DOUBLE_COL = 3.40, 7.00

    def resolve_width(w):
        return {"single": SINGLE_COL, "double": DOUBLE_COL}.get(w, float(w))

    def set_style(usetex=False):
        plt.rcParams.update({
            "font.family": "serif", "font.size": 9, "axes.labelsize": 10,
            "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7,
            "legend.frameon": False, "axes.linewidth": 0.8,
            "xtick.direction": "in", "ytick.direction": "in",
            "xtick.top": True, "ytick.right": True,
            "savefig.dpi": 600, "savefig.bbox": "tight",
            "pdf.fonttype": 42, "ps.fonttype": 42,
            "mathtext.fontset": "cm",
        })
        if usetex:
            plt.rcParams.update({"text.usetex": True})


LS = {0: "-", 1: "--", 2: ":"}
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


def branch_legend(ax, bids, names, loc="best"):
    handles = [Line2D([], [], color="0.3", ls=LS.get(k, "-"), marker=MK.get(k, "o"),
                      ms=3, mew=0, lw=0.9, label=names.get(k, f"branch {k}"))
               for k in sorted(np.unique(bids))]
    ax.legend(handles=handles, loc=loc, handlelength=1.8, labelspacing=0.3)


def plot(data, y, ylabel, width):
    Re, amp, bid, names = data["Re"], data["amp"], data["bid"], data["names"]
    amp_u, re_u = np.unique(amp), np.unique(Re)
    single_amp = amp_u.size == 1

    ncol = 1 if single_amp else 3
    fig, axes = plt.subplots(1, ncol, figsize=(width, 0.78 * width / max(ncol * 0.55, 1)))
    axes = np.atleast_1d(axes)

    cmap_a = plt.get_cmap("coolwarm")
    norm_a = mcolors.Normalize(amp_u.min(), amp_u.max()) if not single_amp \
        else mcolors.Normalize(-1, 1)

    ax = axes[0]
    for a in amp_u:
        for k in np.unique(bid):
            m = np.isclose(amp, a) & (bid == k)
            if not m.any():
                continue
            o = np.argsort(Re[m])
            ax.plot(Re[m][o], y[m][o], ls=LS.get(k, "-"), marker=MK.get(k, "o"),
                    color=cmap_a(norm_a(a)) if not single_amp else "#1f4e79",
                    ms=2.5, mew=0, lw=0.9)
    ax.axhline(0.0, color="0.6", lw=0.6, zorder=0)
    ax.set_xlabel(r"$\mathrm{Re}$")
    ax.set_ylabel(ylabel)
    branch_legend(ax, bid, names, loc="upper left")

    if not single_amp:
        sm = plt.cm.ScalarMappable(cmap=cmap_a, norm=norm_a)
        cb = fig.colorbar(sm, ax=ax, pad=0.02)
        cb.set_label("amplitude", fontsize=8)

        ax = axes[1]
        cmap_r = plt.get_cmap("viridis")
        norm_r = mcolors.Normalize(re_u.min(), re_u.max())
        sel = re_u[np.linspace(0, re_u.size - 1, min(re_u.size, 6)).astype(int)]
        for r in sel:
            for k in np.unique(bid):
                m = np.isclose(Re, r) & (bid == k)
                if not m.any():
                    continue
                o = np.argsort(amp[m])
                ax.plot(amp[m][o], y[m][o], ls=LS.get(k, "-"), marker=MK.get(k, "o"),
                        color=cmap_r(norm_r(r)), ms=2.5, mew=0, lw=0.9)
        ax.axhline(0.0, color="0.6", lw=0.6, zorder=0)
        ax.axvline(0.0, color="0.6", lw=0.6, zorder=0)
        ax.set_xlabel("amplitude")
        ax.set_ylabel(ylabel)
        sm = plt.cm.ScalarMappable(cmap=cmap_r, norm=norm_r)
        cb = fig.colorbar(sm, ax=ax, pad=0.02)
        cb.set_label(r"$\mathrm{Re}$", fontsize=8)

        ax = axes[2]
        vmax = float(np.nanmax(np.abs(y))) or 1.0
        norm_y = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        for k in np.unique(bid):
            m = bid == k
            ax.scatter(Re[m], amp[m], c=y[m], cmap="RdBu_r", norm=norm_y,
                       marker=MK.get(k, "o"), s=12, linewidths=0.2,
                       edgecolors="0.3")
        ax.set_xlabel(r"$\mathrm{Re}$")
        ax.set_ylabel("amplitude")
        sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=norm_y)
        cb = fig.colorbar(sm, ax=ax, pad=0.02)
        cb.set_label(ylabel, fontsize=8)

    fig.tight_layout()
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("snapdir", nargs="?", default="snapshots_sparse")
    p.add_argument("--quantity", choices=("lift", "drag"), default="lift")
    p.add_argument("--outdir", default="figures")
    p.add_argument("--width", default="double")
    p.add_argument("--usetex", action="store_true")
    a = p.parse_args()

    set_style(a.usetex)
    os.makedirs(a.outdir, exist_ok=True)

    d = read(a.snapdir)
    y = d["CL"] if a.quantity == "lift" else d["CD"]
    ylabel = r"$C_L$" if a.quantity == "lift" else r"$C_D$"

    print(f"read {d['path']}")
    print(f"  {y.size} snapshots, {np.unique(d['Re']).size} Re x "
          f"{np.unique(d['amp']).size} amp")
    for k in sorted(np.unique(d["bid"])):
        m = d["bid"] == k
        print(f"  {d['names'].get(k, k):<12} {m.sum():4d} snapshots, "
              f"Re in [{d['Re'][m].min():.2f}, {d['Re'][m].max():.2f}], "
              f"{ylabel[1:-1]} in [{y[m].min():+.4e}, {y[m].max():+.4e}]")
    if np.isnan(y).any():
        print(f"  WARNING: {int(np.isnan(y).sum())} NaN values")

    fig = plot(d, y, ylabel, resolve_width(a.width))
    for ext in ("pdf", "png"):
        f = os.path.join(a.outdir, f"snapshot_{a.quantity}.{ext}")
        fig.savefig(f)
        print(f"wrote {f}")


if __name__ == "__main__":
    main()
