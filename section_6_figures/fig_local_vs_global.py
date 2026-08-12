#!/usr/bin/env python3
"""
fig_local_vs_global.py -- fig:local_vs_global

Panel (a) matched dimension:  E11_K4_matchdim vs E12_K1_matchdim
Panel (b) matched tolerance:  E1_K4_tensor    vs E2_K1_tensor

Coverage is NOT the story: E11 and E12 have nearly identical per-leg point
counts, so the capped global basis does not lose the asymmetric branches. The
ablation therefore has to rest on error magnitude, which is what is plotted --
median over legs with a min/max envelope, because these errors span orders of
magnitude and a mean just tracks the worst leg.

The script also prints the per-leg coverage difference, including any leg present
in one run and absent from the other. G6 says E11 is missing sym@amp-0.80 while
E12 has it: the local model failing where the global one succeeds is the wrong
direction for the argument, so it is reported loudly rather than plotted quietly.

    python section_6_figures/fig_local_vs_global.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402


def prep(tag, root, audit):
    df = C.load_run(tag, root)
    df = C.drop_duplicate_legs(
        df, C.duplicate_leg_keys(df, audit_json=audit, tag=tag, verbose=False))
    return df[df.converged.astype(bool)].copy()


def envelope(ax, df, color, label, col="err_u"):
    d = df.dropna(subset=[col, "Re"])
    if d.empty:
        return
    g = d.groupby("Re")[col]
    med, lo, hi = g.median(), g.min(), g.max()
    ax.fill_between(med.index, lo.values, hi.values, color=color, alpha=0.18,
                    linewidth=0)
    ax.plot(med.index, med.values, "-", color=color, lw=1.3, label=label)


def coverage_report(name, a, b, tag_a, tag_b):
    la = a.groupby("branch").size()
    lb = b.groupby("branch").size()
    all_legs = sorted(set(la.index) | set(lb.index))
    rows = []
    for leg in all_legs:
        na, nb = int(la.get(leg, 0)), int(lb.get(leg, 0))
        rows.append({"branch": leg, tag_a: na, tag_b: nb, "diff": na - nb})
    r = pd.DataFrame(rows)
    missing_a = r[r[tag_a] == 0]
    missing_b = r[r[tag_b] == 0]
    print(f"\n  --- {name}: per-leg coverage ---")
    print(f"  total points: {tag_a}={int(la.sum())}  {tag_b}={int(lb.sum())}")
    if len(missing_a):
        print(f"  !! legs present in {tag_b} but ABSENT from {tag_a}:")
        for _, row in missing_a.iterrows():
            print(f"       {row.branch}  ({row[tag_b]} points)")
        print("     the local model failing where the global one succeeds is the "
              "wrong direction for this ablation -- diagnose before submission")
    if len(missing_b):
        print(f"  legs present in {tag_a} but absent from {tag_b}:")
        for _, row in missing_b.iterrows():
            print(f"       {row.branch}  ({row[tag_a]} points)")
    both = r[(r[tag_a] > 0) & (r[tag_b] > 0)]
    if len(both):
        print(f"  legs in both: {len(both)}; median |diff| = "
              f"{both['diff'].abs().median():.1f} points")
    return r


def main():
    p = C.base_parser(__doc__)
    p.add_argument("--matchdim", nargs=2, default=["E11_K4_matchdim",
                                                   "E12_K1_matchdim"])
    p.add_argument("--matchtol", nargs=2, default=["E1_K4_tensor",
                                                   "E2_K1_tensor"])
    p.add_argument("--zoom", type=float, nargs=2, default=None,
                   help="Re window for the insets; default: around median Re_c")
    args = p.parse_args()
    plt = C.setup_style(args.usetex)
    nu = C.norm_label(args.mass_meta, "M_u")

    runs = {t: prep(t, args.states_root, args.audit)
            for t in args.matchdim + args.matchtol}

    fig, axes = plt.subplots(
        1, 2, figsize=C.figsize(args.width, args.aspect or 0.40), sharey=True)
    pairs = [("(a) matched dimension", args.matchdim),
             ("(b) matched tolerance", args.matchtol)]
    cols = [C.cluster_color(0), C.P("C_GLOBAL")]     # local, global

    zoom = args.zoom
    if zoom is None:
        cr = C.mu_crossings(runs[args.matchtol[0]]).dropna(subset=["Re_c"])
        mid = float(cr.Re_c.median()) if len(cr) else None
        zoom = (mid - 8, mid + 8) if mid else None

    for ax, (title, (tl, tg)) in zip(axes, pairs):
        envelope(ax, runs[tl], cols[0], f"local ({tl.split('_')[1]})")
        envelope(ax, runs[tg], cols[1], r"global ($K\!=\!1$)")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\mathrm{Re}$")
        ax.set_title(title, loc="left")
        ax.legend(loc="upper left")
        if zoom:
            ins = ax.inset_axes([0.55, 0.08, 0.42, 0.42])
            for t, c in ((tl, cols[0]), (tg, cols[1])):
                envelope(ins, runs[t][(runs[t].Re >= zoom[0])
                                      & (runs[t].Re <= zoom[1])], c, None)
            ins.set_yscale("log")
            ins.set_xlim(*zoom)
            ins.tick_params(labelsize=6)
            ins.set_title(r"near $\mathrm{Re}_c$", fontsize=6)
    axes[0].set_ylabel(rf"$\|u-u_N\|_{{{nu}}}/\|u\|_{{{nu}}}$")

    C.save(fig, args.outdir, "fig_local_vs_global", png=args.png)

    macros = {}
    for name, (tl, tg) in pairs:
        r = coverage_report(name, runs[tl], runs[tg], tl, tg)
        r.to_csv(os.path.join(args.outdir,
                              f"coverage_{tl}_vs_{tg}.csv"), index=False)
        el = runs[tl].err_u.dropna() if "err_u" in runs[tl] else pd.Series(dtype=float)
        eg = runs[tg].err_u.dropna() if "err_u" in runs[tg] else pd.Series(dtype=float)
        if len(el) and len(eg):
            ratio = float(eg.median() / el.median())
            key = "MatchDim" if "dimension" in name else "MatchTol"
            macros[f"{key}ErrLocal"] = C.fmt(float(el.median()), ".2e")
            macros[f"{key}ErrGlobal"] = C.fmt(float(eg.median()), ".2e")
            macros[f"{key}ErrRatio"] = C.fmt(ratio, ".2f")
            print(f"  {name}: median err_u local={el.median():.3e} "
                  f"global={eg.median():.3e}  ratio={ratio:.2f}x")
            if ratio < 1.0:
                print("     NOTE: the global model has the LOWER median error "
                      "here -- the ablation does not support the claim as stated")
    if macros:
        C.write_macros(os.path.join(args.outdir, "macros_local_vs_global.tex"),
                       macros)


if __name__ == "__main__":
    main()
