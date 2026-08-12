#!/usr/bin/env python3
"""
fig_k_sweep.py -- fig:k_sweep

Error and cost against the number of clusters, at the nominal POD tolerance.
The companion to tab:k_sensitivity: the table gives the dimensions and the
tensor cost, this gives the accuracy those dimensions buy.

  (a) relative velocity error against Reynolds number, median over legs, one
      curve per K. Same construction as fig:local_vs_global.
  (b) the tradeoff the choice of K actually turns on: median and maximum error
      against the median reduced solve time, one marker per K. If a larger K is
      both faster and no less accurate, that is visible here and nowhere else.

RESTRICTION, AND WHY IT IS NOT OPTIONAL.
The K = 2 and K = 6 sweeps cover nine amplitude columns against twenty-one for
K = 1 and K = 4, and the columns they are missing are the large-|a| ones where
the pitchfork sits near the top of the box -- that is, the hard ones. Comparing
each run over its own amplitude set would hand the smaller sweeps an easier
problem and report the difference as a property of K. Every curve and every
marker here is therefore computed over the amplitude columns that ALL loaded
runs share, and the count is printed and written into a macro. Pass
--all-amps to defeat this only if you intend to caption the difference.

    python section_6_figures/fig_k_sweep.py --width 0.7linewidth
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

DEFAULT_RUNS = [(1, "E2_K1_tensor"), (2, "E5_K2_tensor"),
                (4, "E1_K4_tensor"), (6, "E6_K6_tensor")]

WORD = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six",
        7: "Seven", 8: "Eight", 9: "Nine", 10: "Ten"}


def tighten(plt, on=True):
    """setup_style is sized for a one-panel figure; a 2-across layout with a
    legend needs smaller text."""
    if not on:
        return
    plt.rcParams.update({
        "font.size": 7.5, "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "lines.linewidth": 1.1, "lines.markersize": 4,
        "axes.titlepad": 3.0, "axes.labelpad": 2.0,
    })


def kcolor(plt, i, n):
    return plt.get_cmap(C.seq_cmap())(0.12 + 0.76 * i / max(n - 1, 1))


def main():
    p = C.base_parser(__doc__)
    p.add_argument("--runs", nargs="*", default=None,
                   help="K=TAG pairs, e.g. 4=E1_K4_tensor")
    p.add_argument("--col", default="err_u", help="error column to plot")
    p.add_argument("--envelope", action="store_true",
                   help="add the min/max band per K; off by default because "
                        "four overlapping bands are unreadable")
    p.add_argument("--all-amps", dest="common_amps", action="store_false",
                   default=True,
                   help="do NOT restrict to the shared amplitude columns")
    p.add_argument("--no-tighten", dest="tighten", action="store_false")
    args = p.parse_args()
    plt = C.setup_style(args.usetex)
    tighten(plt, args.tighten)
    nu = C.norm_label(args.mass_meta, "M_u")

    runs = DEFAULT_RUNS
    if args.runs:
        runs = [(int(s.split("=")[0]), s.split("=")[1]) for s in args.runs]

    frames, amps = {}, None
    for K, tag in runs:
        if not os.path.isdir(C.state_dir(tag, args.states_root)):
            print(f"  [skip] {tag}: no such state dir")
            continue
        df = C.load_run(tag, args.states_root)
        df = C.drop_duplicate_legs(
            df, C.duplicate_leg_keys(df, audit_json=args.audit, tag=tag,
                                     verbose=False))
        df = df[df.converged.astype(bool)].copy()
        if args.col not in df or not df[args.col].notna().any():
            print(f"  [skip] {tag}: no '{args.col}' column -- was this run made "
                  f"with FOM_COMPUTE=1?")
            continue
        frames[K] = df
        a = set(df.amp.round(6))
        amps = a if amps is None else (amps & a)
    if not frames:
        raise SystemExit("nothing to plot -- check the K runs exist and carry "
                         f"'{args.col}'")

    print(f"  loaded K = {sorted(frames)}")
    print(f"  amplitude columns shared by all: {len(amps)}")
    sizes = {K: len(set(d.amp.round(6))) for K, d in frames.items()}
    if len(set(sizes.values())) > 1:
        print("  !! the runs do NOT share an amplitude grid: "
              + ", ".join(f"K={K}: {n}" for K, n in sorted(sizes.items())))
        print("     the missing columns are the large-|a| ones, so an "
              "unrestricted comparison would flatter the smaller sweeps")
        if not args.common_amps:
            print("     --all-amps given: plotting each run over its OWN set; "
                  "the caption MUST say so")
    if args.common_amps:
        for K in frames:
            frames[K] = frames[K][frames[K].amp.round(6).isin(amps)]

    fig, axes = plt.subplots(
        1, 2, figsize=C.figsize(args.width, args.aspect or 0.44),
        layout="constrained")
    ax, axp = axes
    n = len(frames)
    macros = {"KSweepCommonAmps": len(amps)}
    summary = []

    for i, K in enumerate(sorted(frames)):
        d = frames[K].dropna(subset=[args.col, "Re"])
        if d.empty:
            continue
        col = kcolor(plt, i, n)
        g = d.groupby("Re")[args.col]
        med = g.median()
        if args.envelope:
            ax.fill_between(med.index, g.min().values, g.max().values,
                            color=col, alpha=0.12, linewidth=0)
        ax.plot(med.index, med.values, "-", color=col, label=f"$K={K}$")

        t = d.t_rom.dropna() if "t_rom" in d else pd.Series(dtype=float)
        row = {"K": K, "n": len(d),
               "median": float(d[args.col].median()),
               "max": float(d[args.col].max()),
               "t_rom": 1e3 * float(t.median()) if len(t) else np.nan}
        summary.append(row)
        macros[f"KSweepErrK{WORD.get(K, K)}"] = C.fmt(row["median"], ".2e")
        if np.isfinite(row["t_rom"]):
            macros[f"KSweepTimeK{WORD.get(K, K)}"] = C.fmt(row["t_rom"], ".1f")

    ax.set_yscale("log")
    ax.set_xlabel(r"$\mathrm{Re}$")
    ax.set_ylabel(rf"$\|u-u_N\|_{{{nu}}}/\|u\|_{{{nu}}}$")
    ax.set_title("(a) error along the sweep", loc="left")
    ax.legend(loc="best", ncol=2)

    s = pd.DataFrame(summary).sort_values("K")
    if s.t_rom.notna().any():
        for i, (_, r) in enumerate(s.iterrows()):
            col = kcolor(plt, sorted(frames).index(int(r.K)), n)
            axp.plot([r.t_rom], [r["median"]], "o", color=col, ms=6)
            axp.plot([r.t_rom], [r["max"]], "s", color=col, ms=5, mfc="none")
            axp.annotate(f"$K={int(r.K)}$", (r.t_rom, r["median"]),
                         textcoords="offset points", xytext=(5, -9),
                         fontsize=7, color=C.P("C_SYM"))
        axp.plot(s.t_rom, s["median"], "-", color="0.75", lw=0.8, zorder=0)
        axp.plot(s.t_rom, s["max"], ":", color="0.75", lw=0.8, zorder=0)
        axp.set_yscale("log")
        axp.set_xlabel(r"median $t_{\mathrm{ROM}}$ (ms)")
        axp.set_ylabel("relative error")
        axp.set_title("(b) accuracy against cost", loc="left")
        from matplotlib.lines import Line2D
        axp.legend(handles=[
            Line2D([], [], marker="o", ls="none", color="0.35", label="median"),
            Line2D([], [], marker="s", ls="none", mfc="none", color="0.35",
                   label="max")], loc="best")
    else:
        axp.text(0.5, 0.5, "no t_rom column", ha="center", va="center",
                 transform=axp.transAxes, fontsize=7, color="0.4")

    fig.get_layout_engine().set(w_pad=0.045, h_pad=0.02, wspace=0.06)
    C.save(fig, args.outdir, "fig_k_sweep", png=args.png)

    print(f"\n  --- over the {len(amps)} shared amplitude columns ---")
    print("  " + s.to_string(index=False))
    best_err = int(s.loc[s["median"].idxmin()].K)
    if s.t_rom.notna().any():
        best_t = int(s.loc[s.t_rom.idxmin()].K)
        print(f"\n  lowest median error: K={best_err};  fastest: K={best_t}")
        if best_err == best_t and best_err != 4:
            print(f"  !! K={best_err} is both the most accurate and the fastest "
                  f"here. If the paper reports K=4 as the proposed setting, "
                  f"that needs a stated reason or a rerun -- a referee will "
                  f"read this panel exactly this way.")
    s.to_csv(os.path.join(args.outdir, "k_sweep.csv"), index=False)
    print(f"  wrote {os.path.join(args.outdir, 'k_sweep.csv')}")
    C.write_macros(os.path.join(args.outdir, "macros_k_sweep.tex"), macros)


if __name__ == "__main__":
    main()
