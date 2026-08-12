#!/usr/bin/env python3
"""
fig_rec_convergence.py -- fig:rec_convergence

Re_c at zero rotation against the local reduced dimension, over the E13
tolerance family (1e-4, 1e-6, 1e-8, 1e-10). N comes from the offline cache, so
the x axis is the dimension the run actually used.

WHAT THE REFERENCE IS, AND IS NOT.
The draft asks for the discrepancy against "the full-order critical Reynolds
number". The full-order generalised eigensweep (G1) does not exist, so that
number does not exist either, and this script will not invent it. What does
exist is the full-order lift departure at a = 0 from the one family member run
with FOM_COMPUTE=1. That is a genuine full-order quantity and, uniquely at
a = 0, a principled locator of the pitchfork: with no rotation the symmetric
branch carries no lift, so departure from zero IS the symmetry breaking. Away
from zero it is not, which is exactly why this figure is the a = 0 figure.

It is still a lift bracket and not an eigenvalue crossing, so it is labelled
that way on the axes, in the printed output and in the macros. Do not let the
caption call it the full-order eigenvalue. If G1 is ever run, pass the real
number through --rec-fom and the band is replaced.

The second panel is not decoration. At the loosest tolerance the reduced
indicator crosses zero TWICE and the continuation spawns a leg pair that does
not exist: an under-resolved basis does not merely misplace the bifurcation, it
manufactures one. That is the sharpest statement in the subsection about basis
adequacy for detection, and it is invisible in panel (a), which plots only the
first crossing.

    python section_6_figures/fig_rec_convergence.py --width 0.7linewidth
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

def tighten(plt, on=True):
    """setup_style is sized for a ONE-panel figure. A 3-across layout needs
    smaller text or the titles and tick labels collide. --no-tighten opts out."""
    if not on:
        return
    plt.rcParams.update({
        "font.size": 7, "axes.labelsize": 7.5, "axes.titlesize": 7.5,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
        "lines.linewidth": 1.0, "lines.markersize": 3.5,
        "axes.titlepad": 3.0, "axes.labelpad": 2.0,
    })


def thin_log_ticks(ax, axis="x", n=4):
    import matplotlib.ticker as mt
    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_major_locator(mt.LogLocator(numticks=n))
    a.set_minor_locator(mt.LogLocator(subs="auto", numticks=n * 3))
    a.set_minor_formatter(mt.NullFormatter())


def tol_axis(ax, d):
    """Tolerance maps one-to-one onto N, so it belongs on a second x axis, not
    as a per-point annotation -- the annotations collide as soon as two members
    land on the same Re_c, which is exactly what happens once N is converged."""
    ok = d[d.N.notna()]
    sec = ax.secondary_xaxis("top")
    sec.set_xticks(list(ok.N))
    sec.set_xticklabels([f"$10^{{{int(round(np.log10(t)))}}}$" for t in ok.tol])
    sec.tick_params(labelsize=6, pad=1)
    sec.set_xlabel(r"$\varepsilon_{\mathrm{POD}}$", fontsize=6.5, labelpad=1)
    return sec


DEFAULT_FAMILY = [
    ("E13_K4_a0_tol1e-4", 1e-4),
    ("E13_K4_a0_tol1e-6", 1e-6),
    ("E13_K4_a0_tol1e-8", 1e-8),
    ("E13_K4_a0_tol1e-10", 1e-10),
]


# ---------------------------------------------------------------------------

def leg_inventory(tag, df):
    """Legs and their point counts. A member with legs the nominal member does
    not have is a finding, not a nuisance -- surface it before anything is
    averaged away."""
    return {k: int(n) for k, n in df.groupby("kind").size().items()}


def member(tag, tol, args, dims, legs, nominal_legs):
    sd = C.state_dir(tag, args.states_root)
    if not os.path.isdir(sd):
        print(f"  [skip] {tag}: no such state dir")
        return None
    df = C.load_run(tag, args.states_root)
    inv = leg_inventory(tag, df)
    df = C.drop_duplicate_legs(df, legs)
    df = df[df.converged.astype(bool)].copy()

    r = dims.get((args.K, tol))
    if r is None or any(x is None for x in r):
        print(f"  [skip] {tag}: no reduced dimensions in cache for "
              f"K={args.K}, tol={tol:g}")
        return None

    cr = C.mu_crossings(df)
    row = cr[np.isclose(cr.amp, 0.0)]
    if row.empty:
        print(f"  [warn] {tag}: no a=0 column after the leg filter")
        return None
    row = row.iloc[0]

    sym = df[df.is_sym]
    n_dup_re = int(len(sym) - sym.Re.nunique())

    rec = {
        "tag": tag, "tol": tol, "r": r,
        "N": sum(r) if args.x == "sum" else max(r),
        "n_points": len(df), "n_sym": len(sym), "dup_Re_in_sym": n_dup_re,
        "n_legs": len(inv), "legs": inv,
        "n_crossings": int(row.get("n_crossings", 0) or 0),
        "Re_c": float(row.get("Re_c", np.nan)),
        "bracket": float(row.get("bracket", np.nan)),
    }
    extra = set(inv) - set(nominal_legs) if nominal_legs else set()
    rec["extra_legs"] = sorted(extra)
    return rec


def fom_reference(args, legs):
    """Full-order lift departure at a = 0, with the Re step as its bracket."""
    if args.rec_fom is not None:
        print(f"  reference: --rec-fom {args.rec_fom:g} "
              f"(bracket {args.rec_fom_bracket:g}) -- overriding the lift proxy")
        return {"Re_c": float(args.rec_fom),
                "bracket": float(args.rec_fom_bracket),
                "kind": "eigenvalue", "tag": "(supplied)"}

    sd = C.state_dir(args.fom_from, args.states_root)
    if not os.path.isdir(sd):
        print(f"  [warn] {args.fom_from} not found -- no full-order reference")
        return None
    df = C.load_run(args.fom_from, args.states_root)
    df = C.drop_duplicate_legs(df, legs)
    df = df[df.converged.astype(bool)]
    if "C_L_fom" not in df or not df.C_L_fom.notna().any():
        print(f"  [warn] {args.fom_from} has no C_L_fom -- needs a "
              f"FOM_COMPUTE=1 run; no full-order reference, and the caption "
              f"must not promise a discrepancy")
        return None
    pr = C.cl_fom_departure(df, args.cl_frac)
    row = pr[np.isclose(pr.amp, 0.0)]
    if row.empty or not np.isfinite(row.Re_c_proxy.iloc[0]):
        print(f"  [warn] {args.fom_from}: lift never clears {args.cl_frac:g} "
              f"of its own scale at a=0 -- no reference")
        return None
    re = float(row.Re_c_proxy.iloc[0])
    step = float(np.median(np.diff(np.sort(df.Re.unique())))) if \
        df.Re.nunique() > 1 else np.nan
    print(f"  reference: full-order lift departure at Re={re:.2f} "
          f"(step {step:g}) from {args.fom_from}")
    print("  NB this is a lift bracket, NOT the full-order eigenvalue crossing")
    return {"Re_c": re, "bracket": step, "kind": "lift departure",
            "tag": args.fom_from}


# ---------------------------------------------------------------------------

def panel_convergence(ax, d, ref, glob, args):
    ok = d[np.isfinite(d.Re_c)]
    ax.errorbar(ok.N, ok.Re_c, yerr=ok.bracket / 2.0, fmt="o-", ms=4,
                capsize=2, lw=1.1, color=C.cluster_color(0),
                label=r"local ROM, $\mu = 0$")

    bad = d[(d.n_crossings > 1) & np.isfinite(d.Re_c)]
    if len(bad):
        ax.plot(bad.N, bad.Re_c, "o", ms=9, mfc="none", mew=1.3,
                color=C.P("C_NEG"), zorder=5,
                label="spurious extra crossing")

    if glob is not None:
        ax.plot([glob["N"]], [glob["Re_c"]], "D", ms=5, color=C.P("C_GLOBAL"),
                label=r"global ROM ($K\!=\!1$)")

    if ref is not None:
        ax.axhline(ref["Re_c"], color=C.P("C_SYM"), lw=1.0, ls="--",
                   label=f"full order, {ref['kind']}")
        if np.isfinite(ref["bracket"]):
            ax.axhspan(ref["Re_c"] - ref["bracket"] / 2.0,
                       ref["Re_c"] + ref["bracket"] / 2.0,
                       color=C.P("C_SYM"), alpha=0.18, linewidth=0)

    ax.set_xscale("log")
    thin_log_ticks(ax, "x", 3)
    tol_axis(ax, d)
    ax.set_xlabel(r"$N$ " + (r"($\sum_k r_k$)" if args.x == "sum"
                             else r"($\max_k r_k$)"))
    ax.set_ylabel(r"$\mathrm{Re}_c(0)$")
    ax.set_title("(a) critical Reynolds number", loc="left")
    # four entries will not fit inside a panel this narrow; the caller lifts
    # them into a figure-level legend under all three panels
    return ax.get_legend_handles_labels()


def panel_discrepancy(ax, d, ref, args):
    """|Re_c(N) - reference| under refinement. Falls back to self-convergence
    against the finest member if no full-order reference exists, and says so in
    the axis label rather than passing one off as the other."""
    ok = d[np.isfinite(d.Re_c)]
    if len(ok) < 2:
        ax.text(0.5, 0.5, "too few members", ha="center", va="center",
                transform=ax.transAxes, fontsize=7, color="0.4")
        return
    if ref is not None:
        base = ref["Re_c"]
        lab = r"$|\mathrm{Re}_c - \mathrm{Re}_c^{h}|$"
        title = "(b) discrepancy"
    else:
        base = float(ok.iloc[-1].Re_c)
        lab = r"$|\mathrm{Re}_c(N) - \mathrm{Re}_c(N_{\max})|$"
        title = "(b) self-convergence"
    dv = (ok.Re_c - base).abs()
    floor = float(dv[dv > 0].min()) / 3.0 if (dv > 0).any() else 1e-3
    ax.plot(ok.N, dv.clip(lower=floor), "o-", ms=4, lw=1.1,
            color=C.cluster_color(0))
    if ref is not None and np.isfinite(ref["bracket"]):
        ax.axhline(ref["bracket"], color=C.P("C_SYM"), ls=":", lw=0.9)
        ax.text(0.03, ref["bracket"], "bracket", fontsize=6,
                color=C.P("C_SYM"), va="bottom",
                transform=ax.get_yaxis_transform())
    ax.set_xscale("log")
    ax.set_yscale("log")
    thin_log_ticks(ax, "x", 3)
    thin_log_ticks(ax, "y", 4)
    tol_axis(ax, ok)
    ax.set_xlabel(r"$N$")
    ax.set_ylabel(lab)
    ax.set_title(title, loc="left")


def panel_mu(ax, frames, d, ref, plt):
    """mu(Re) on the symmetric leg, one curve per tolerance. Every zero crossing
    beyond the first is marked where it occurs -- that is the spurious
    bifurcation, and it is a property of a location, not of a legend entry."""
    cmap = plt.get_cmap(C.seq_cmap())
    n = max(len(d) - 1, 1)
    for j, (_, r) in enumerate(d.iterrows()):
        df = frames.get(r.tag)
        if df is None:
            continue
        s = df[df.is_sym].dropna(subset=["mu"]).sort_values("Re")
        if s.empty:
            continue
        col = cmap(j / n)
        spurious = r.n_crossings > 1
        ax.plot(s.Re, s.mu, "--" if spurious else "-", lw=1.1, color=col,
                label=f"$10^{{{int(round(np.log10(r.tol)))}}}$")
        if spurious:
            mu = s.mu.values.astype(float)
            Re = s.Re.values.astype(float)
            sgn = np.sign(mu)
            flips = [i for i in np.where(np.diff(sgn) != 0)[0] if sgn[i] != 0]
            for i in flips[1:]:
                xc = 0.5 * (Re[i] + Re[i + 1])
                ax.plot([xc], [0.0], "o", ms=7, mfc="none", mew=1.3,
                        color=C.P("C_NEG"), zorder=6)
    ax.axhline(0.0, color=C.P("C_GLOBAL"), lw=0.8)
    if ref is not None:
        ax.axvline(ref["Re_c"], color=C.P("C_SYM"), ls="--", lw=0.9)
    # An under-resolved basis throws mu out by orders of magnitude, so a linear
    # axis renders every resolved curve as the zero line under a "1e7" offset.
    # symlog fixes that, with the linear band sized to hold the RESOLVED curves
    # exactly: those then read normally and only the blow-up enters the log
    # region. Ticks are placed by hand -- the default locator puts three of them
    # inside the compressed linear band, where they overlap.
    import matplotlib.ticker as mt
    res, allv = [], []
    for _, r in d.iterrows():
        f = frames.get(r.tag)
        if f is None:
            continue
        mv = np.abs(f[f.is_sym].mu.dropna().values.astype(float))
        mv = mv[np.isfinite(mv)]
        allv.append(mv)
        if r.n_crossings <= 1:
            res.append(mv)
    allv = np.concatenate(allv) if allv else np.array([])
    resv = np.concatenate(res) if res else np.array([])
    pos = allv[allv > 0]
    if len(pos) and len(resv) and pos.max() > 100 * max(resv.max(), 1e-30):
        lt = 10.0 ** np.ceil(np.log10(max(resv.max(), 1e-12)))
        ax.set_yscale("symlog", linthresh=lt)
        # NOT a tick at +/-linthresh: the linear band is ~one decade wide, so
        # those land within a few pixels of zero and overlap it
        top = 10.0 ** np.floor(np.log10(pos.max()))
        ticks = [0.0]
        mid = top / 1e3
        if mid > 10 * lt:
            ticks += [mid, -mid]
        if top > 10 * lt:
            ticks += [top, -top]
        ax.yaxis.set_major_locator(mt.FixedLocator(ticks))
        ax.yaxis.set_minor_locator(mt.NullLocator())
    ax.set_xlabel(r"$\mathrm{Re}$")
    ax.set_ylabel(r"$\mu$")
    ax.set_title(r"(c) $\mu(\mathrm{Re})$ at $a=0$", loc="left")
    # mu is negative at low Re and positive at high Re, so the upper left is
    # the one quadrant no curve enters
    ax.legend(loc="upper left", fontsize=6, labelspacing=0.25,
              handlelength=1.2, handletextpad=0.35, borderpad=0.25,
              title=r"$\varepsilon_{\mathrm{POD}}$", title_fontsize=6)


# ---------------------------------------------------------------------------

def main():
    p = C.base_parser(__doc__)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--family", nargs="*", default=None,
                   help="TAG=TOL pairs, e.g. E13_K4_a0_tol1e-8=1e-8")
    p.add_argument("--global-tag", default="E14_K1_a0_tol1e-8")
    p.add_argument("--global-tol", type=float, default=1e-8)
    p.add_argument("--nominal", default="E13_K4_a0_tol1e-8",
                   help="member whose leg set defines the reference inventory")
    p.add_argument("--fom-from", default="E13_K4_a0_tol1e-8",
                   help="member carrying C_L_fom for the full-order reference")
    p.add_argument("--rec-fom", type=float, default=None,
                   help="true full-order Re_c(0) from the G1 eigensweep, once "
                        "it exists; replaces the lift proxy")
    p.add_argument("--rec-fom-bracket", type=float, default=1.0)
    p.add_argument("--cl-frac", type=float, default=0.02)
    p.add_argument("--x", choices=["sum", "max"], default="sum")
    p.add_argument("--no-tighten", dest="tighten", action="store_false",
                   help="keep the house single-panel font sizes")
    args = p.parse_args()
    plt = C.setup_style(args.usetex)
    tighten(plt, args.tighten)

    family = DEFAULT_FAMILY
    if args.family:
        family = [(s.split("=")[0], float(s.split("=")[1])) for s in args.family]

    # Legs are classified ONCE, on the nominal member, and reused. Classifying
    # per member would let the loose-tolerance run's spurious pair be scored as
    # a duplicate of the symmetric leg and dropped -- deleting the finding.
    legs = set()
    if os.path.isdir(C.state_dir(args.nominal, args.states_root)):
        legs = C.legs_from_run(args.nominal, args.states_root,
                               audit_json=args.audit)
    else:
        print(f"  [warn] nominal member {args.nominal} not found -- no leg "
              f"filter applied")

    dims = C.cache_dims(args.cache_root)
    nominal_legs = None
    if os.path.isdir(C.state_dir(args.nominal, args.states_root)):
        nominal_legs = leg_inventory(
            args.nominal, C.load_run(args.nominal, args.states_root))

    rows, frames = [], {}
    for tag, tol in family:
        rec = member(tag, tol, args, dims, legs, nominal_legs)
        if rec is None:
            continue
        rows.append(rec)
        frames[tag] = C.load_run(tag, args.states_root)
    if not rows:
        raise SystemExit("nothing to plot -- check the E13 runs and the caches")
    d = pd.DataFrame(rows).sort_values("N").reset_index(drop=True)

    glob = None
    if os.path.isdir(C.state_dir(args.global_tag, args.states_root)):
        import argparse as _ap
        gargs = _ap.Namespace(**{**vars(args), "K": 1})
        glob = member(args.global_tag, args.global_tol, gargs, dims, legs,
                      nominal_legs)

    ref = fom_reference(args, legs)

    # ------------------------------------------------------------- diagnostics
    print("\n  --- the family ---")
    for _, r in d.iterrows():
        print(f"  {r.tag:22s} tol={r.tol:<8g} r={r.r} N={r.N:<5d} "
              f"pts={r.n_points:<5d} sym={r.n_sym:<4d} legs={r.n_legs} "
              f"crossings={r.n_crossings}  Re_c={C.fmt(r.Re_c, '.3f')} "
              f"bracket={C.fmt(r.bracket, '.2g')}")
        if r.dup_Re_in_sym:
            print(f"      !! {r.dup_Re_in_sym} repeated Re on the symmetric leg "
                  f"-- this member was written more than once; Re_c is not "
                  f"trustworthy until that is resolved")
        if r.extra_legs:
            print(f"      !! legs absent from {args.nominal}: {r.extra_legs}")
        if r.n_crossings > 1:
            print(f"      !! mu changes sign {r.n_crossings} times: the basis "
                  f"at this tolerance manufactures a bifurcation that the "
                  f"resolved bases do not have")

    fig, axes = plt.subplots(
        1, 3, figsize=C.figsize(args.width, args.aspect or 0.38),
        layout="constrained")
    h, l = panel_convergence(axes[0], d, ref, glob, args)
    panel_discrepancy(axes[1], d, ref, args)
    panel_mu(axes[2], frames, d, ref, plt)
    fig.get_layout_engine().set(w_pad=0.045, h_pad=0.02, wspace=0.07)
    if h:
        fig.legend(h, l, loc="outside lower center", ncol=min(len(h), 4),
                   handletextpad=0.4, columnspacing=1.4, borderpad=0.3)
    C.save(fig, args.outdir, "fig_rec_convergence", png=args.png)

    # ------------------------------------------------------------------ numbers
    macros = {"RecConvNMin": int(d.N.min()), "RecConvNMax": int(d.N.max()),
              "RecConvMembers": len(d)}
    nominal = d[d.tag == args.nominal]
    if len(nominal):
        r = nominal.iloc[0]
        macros["RecNominal"] = C.fmt(float(r.Re_c), ".2f")
        macros["RecNominalN"] = int(r.N)
        macros["RecNominalBracket"] = C.fmt(float(r.bracket), ".2g")

    if ref is not None:
        macros["RecFom"] = C.fmt(ref["Re_c"], ".2f")
        macros["RecFomBracket"] = C.fmt(ref["bracket"], ".2g")
        macros["RecFomKind"] = ref["kind"]
        print("\n  --- discrepancy against the full-order "
              f"{ref['kind']} at Re={ref['Re_c']:.2f} ---")
        for _, r in d.iterrows():
            if not np.isfinite(r.Re_c):
                continue
            dv = abs(r.Re_c - ref["Re_c"])
            flag = "  <- within the reference bracket" if (
                np.isfinite(ref["bracket"]) and dv <= ref["bracket"]) else ""
            print(f"  N={r.N:<5d} tol={r.tol:<8g} |dRe_c|={dv:6.3f}{flag}")
        if len(nominal):
            dv = abs(float(nominal.iloc[0].Re_c) - ref["Re_c"])
            macros["RecDiscrepancy"] = C.fmt(dv, ".2f")
        ok = d[np.isfinite(d.Re_c)]
        mono = bool((ok.Re_c - ref["Re_c"]).abs().is_monotonic_decreasing)
        macros["RecDiscrepancyMonotone"] = "yes" if mono else "no"
        if not mono:
            print("  the discrepancy is NOT monotone in N -- say 'converges to "
                  "within the bracket', not 'decreases monotonically'")
    else:
        print("\n  [note] no full-order reference: this figure shows the "
              "SELF-convergence of Re_c under refinement only, and the caption "
              "must not claim a discrepancy")

    if glob is not None:
        macros["RecGlobal"] = C.fmt(glob["Re_c"], ".2f")
        macros["RecGlobalN"] = int(glob["N"])
        print(f"\n  global (K=1) at N={glob['N']}: Re_c={glob['Re_c']:.3f}"
              + (f"   local at comparable N={nominal.iloc[0].N}: "
                 f"{nominal.iloc[0].Re_c:.3f}" if len(nominal) else ""))

    sp = d[d.n_crossings > 1]
    if len(sp):
        r = sp.iloc[0]
        macros["RecSpuriousTol"] = f"10^{{{int(round(np.log10(r.tol)))}}}"
        macros["RecSpuriousN"] = int(r.N)
        macros["RecSpuriousCrossings"] = int(r.n_crossings)
        macros["RecSpuriousExtraLegs"] = len(r.extra_legs)

    out_csv = os.path.join(args.outdir, "rec_convergence.csv")
    os.makedirs(args.outdir, exist_ok=True)
    d.drop(columns=["legs"]).to_csv(out_csv, index=False)
    print(f"\n  wrote {out_csv}")
    C.write_macros(os.path.join(args.outdir, "macros_rec_convergence.tex"),
                   macros)


if __name__ == "__main__":
    main()
