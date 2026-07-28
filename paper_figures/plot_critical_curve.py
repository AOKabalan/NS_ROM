#!/usr/bin/env python
"""
Publication-quality figures for the reduced critical curve Re_c(amp).

Style (fonts, sizes, palette, column widths) comes from nsrom.plotting.style,
shared with every other figure script.

Reads the two CSVs written by save_critical_curve:

    critical_curve.csv : amp, Re_c, mu_c, found, kind
    critical_sweep.csv : amp, Re, mu, cluster, converged, iters

and produces (PDF + PNG in --outdir):

    critical_curve.{pdf,png} : Re_c vs amp, detection kind by marker
                               (crossing / rounded / extrapolated),
                               unstable region shaded.
    mu_sweeps.{pdf,png}      : (a) mu(Re) per amp, colored by amp, with
                               non-converged points as a rug on the axis;
                               (b) zoom around mu = 0 with the located
                               Re_c marked on the zero line.

Usage:
    python plot_critical_curve.py                       # default filenames
    python plot_critical_curve.py curve.csv sweep.csv --outdir figs
    python plot_critical_curve.py --width double --usetex \
        --xlabel '$a$' --zoom 0.04
"""

import argparse
import csv
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory

from nsrom.plotting.style import set_style, resolve_width, SINGLE_COL, DOUBLE_COL, C_POS, C_NEG

# ----------------------------------------------------------------------
# Marker styling by detection kind (main colours from the shared palette)
# ----------------------------------------------------------------------

KIND_STYLE = {
    # kind -> (marker, facecolor, edgecolor, label)
    "crossing":     ("o", C_POS,  C_POS,     "crossing ($\\mu = 0$)"),
    "rounded":      ("D", "none", C_NEG,     "rounded (closest approach)"),
    "extrapolated": ("s", "none", "#5c5c5c", "extrapolated (verify)"),
    "":             ("x", "#888888", "#888888", "not found"),
}


# ----------------------------------------------------------------------
# IO
# ----------------------------------------------------------------------

def read_curve(path):
    """-> list of dicts {amp, Re_c, mu_c, found, kind}, sorted by amp."""
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "amp": float(r["amp"]),
                "Re_c": float(r["Re_c"]),          # 'nan' parses fine
                "mu_c": float(r["mu_c"]),
                "found": r["found"].strip() == "True",
                "kind": (r.get("kind") or "").strip(),
            })
    return sorted(rows, key=lambda d: d["amp"])


def read_sweep(path):
    """-> {amp: dict(Re, mu, converged)} with arrays SORTED by Re
    (pin/bisection points are appended out of order in the log)."""
    by_amp = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            a = float(r["amp"])
            d = by_amp.setdefault(a, {"Re": [], "mu": [], "conv": []})
            d["Re"].append(float(r["Re"]))
            d["mu"].append(float(r["mu"]))
            d["conv"].append(r["converged"].strip() == "True")
    for a, d in by_amp.items():
        order = np.argsort(d["Re"])
        for k in ("Re", "mu"):
            d[k] = np.asarray(d[k], float)[order]
        d["conv"] = np.asarray(d["conv"], bool)[order]
    return by_amp


# ----------------------------------------------------------------------
# Figure 1: Re_c(amp)
# ----------------------------------------------------------------------

def plot_critical_curve(curve, width, xlabel, shade=True):
    fig, ax = plt.subplots(figsize=(width, 0.78 * width))

    found = [c for c in curve if c["found"]]
    amps = np.array([c["amp"] for c in found])
    recs = np.array([c["Re_c"] for c in found])

    # connecting line under the markers
    ax.plot(amps, recs, "-", color="0.55", lw=0.9, zorder=1)

    # unstable side of the primary branch
    if shade and len(amps) > 1:
        top = recs.max() + 0.12 * (recs.max() - recs.min() + 1)
        ax.fill_between(amps, recs, top, color="0.5", alpha=0.10,
                        lw=0, zorder=0)
        ax.text(amps.mean(), 0.5 * (recs.mean() + top), "unstable",
                ha="center", va="center", color="0.35", fontsize=8,
                style="italic")

    # markers by kind
    seen = []
    for c in found:
        m, fc, ec, lab = KIND_STYLE.get(c["kind"], KIND_STYLE[""])
        ax.plot(c["amp"], c["Re_c"], m, mfc=fc, mec=ec, mew=1.0,
                ls="none", zorder=3,
                label=lab if c["kind"] not in seen else None)
        seen.append(c["kind"])

    ax.axvline(0.0, color="0.8", lw=0.6, zorder=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$\mathrm{Re}_c$")
    ax.legend(loc="upper left", handlelength=1.0,
              borderaxespad=0.4, labelspacing=0.35)
    return fig


# ----------------------------------------------------------------------
# Figure 2: mu(Re) sweeps + zoom
# ----------------------------------------------------------------------

def plot_mu_sweeps(sweep, curve, width, xlabel, zoom):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(width, 1.45 * width),
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.32})

    amps = sorted(sweep)
    a_min, a_max = min(amps), max(amps)
    cmap = plt.get_cmap("coolwarm")
    norm = plt.Normalize(a_min, a_max) if a_max > a_min \
        else plt.Normalize(a_min - 1, a_max + 1)
    re_c = {c["amp"]: c for c in curve if c["found"]}

    for ax in (ax1, ax2):
        ax.axhline(0.0, color="0.4", lw=0.7, ls="--", zorder=1)

    for a in amps:
        d = sweep[a]
        col = cmap(norm(a))
        ok = np.isfinite(d["mu"])

        ax1.plot(d["Re"][ok], d["mu"][ok], "-", color=col, lw=1.0,
                 zorder=2)
        ax2.plot(d["Re"][ok], d["mu"][ok], "o-", color=col, lw=1.0,
                 ms=2.6, mew=0, zorder=2)

        # non-converged points: rug on the bottom axis of panel (a)
        bad = ~d["conv"]
        if bad.any():
            tr = blended_transform_factory(ax1.transData, ax1.transAxes)
            ax1.plot(d["Re"][bad], np.full(bad.sum(), 0.015), "|",
                     color=col, ms=5, mew=1.0, transform=tr,
                     zorder=3, clip_on=False)

        # located critical point on the zero line (zoom panel)
        if a in re_c:
            c = re_c[a]
            m, fc, ec, _ = KIND_STYLE.get(c["kind"], KIND_STYLE[""])
            ax2.plot(c["Re_c"], 0.0, m, mfc=fc if fc != "none" else "w",
                     mec=col, mew=1.1, ms=6, zorder=4)

    ax1.set_ylabel(r"$\mu$")
    ax1.text(0.02, 0.05, r"$\vert\,$non-converged", transform=ax1.transAxes,
             fontsize=7, color="0.4", va="bottom")

    # zoom limits: around the located critical points
    if re_c:
        lo = min(c["Re_c"] for c in re_c.values())
        hi = max(c["Re_c"] for c in re_c.values())
        pad = 0.12 * (hi - lo + 1)
        ax2.set_xlim(lo - pad, hi + pad)
    ax2.set_ylim(-zoom, zoom)
    ax2.set_ylabel(r"$\mu$")
    ax2.set_xlabel(r"$\mathrm{Re}$")

    for ax, tag in ((ax1, "(a)"), (ax2, "(b)")):
        ax.text(0.012, 0.96, tag, transform=ax.transAxes,
                va="top", fontsize=9)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=(ax1, ax2), pad=0.02, aspect=35)
    cb.set_label(xlabel)
    cb.ax.tick_params(labelsize=8)
    return fig


# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("curve_csv", nargs="?", default="critical_curve.csv")
    p.add_argument("sweep_csv", nargs="?", default="critical_sweep.csv")
    p.add_argument("--outdir", default="figures")
    p.add_argument("--width", default="single",
                help="preset ('single','double','linewidth','0.7linewidth') "
                        "or a width in inches, e.g. 5.79")
    p.add_argument("--xlabel", default=r"$a$",
                   help="control-parameter axis label (default: $a$)")
    p.add_argument("--zoom", type=float, default=0.05,
                   help="half-height of the mu zoom panel")
    p.add_argument("--no-shade", action="store_true",
                   help="disable the shaded unstable region")
    p.add_argument("--usetex", action="store_true")
    args = p.parse_args()

    set_style(args.usetex)
    os.makedirs(args.outdir, exist_ok=True)
    W = resolve_width(args.width)
    curve = read_curve(args.curve_csv)

    fig1 = plot_critical_curve(curve, W, args.xlabel,
                               shade=not args.no_shade)
    for ext in ("pdf", "png"):
        f = os.path.join(args.outdir, f"critical_curve.{ext}")
        fig1.savefig(f)
        print(f"wrote {f}")

    if os.path.exists(args.sweep_csv):
        sweep = read_sweep(args.sweep_csv)
        fig2 = plot_mu_sweeps(sweep, curve, W, args.xlabel, args.zoom)
        for ext in ("pdf", "png"):
            f = os.path.join(args.outdir, f"mu_sweeps.{ext}")
            fig2.savefig(f)
            print(f"wrote {f}")
    else:
        print(f"(no {args.sweep_csv}; skipped mu_sweeps figure)")


if __name__ == "__main__":
    main()
