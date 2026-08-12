#!/usr/bin/env python3
"""
fig_critical_curve.py -- fig:critical_curve and fig:mu_sweeps

Both come from the same mu extraction, so one script emits both; --which selects.

fig:critical_curve  Re_c(a) from the mu zero crossings, one per amplitude column.
    The bracket is the Re step, so it is drawn as an error bar rather than
    implied by a smooth line: the resolution is part of the result. Markers are
    coded by crossing direction. The curve is NOT symmetric in a and should not
    be -- mirror-symmetric rotation makes +a and -a physically distinct -- so the
    script prints the asymmetry it measures rather than letting the eye judge.

fig:mu_sweeps  mu against Re at representative amplitudes, no extraction needed.

    python section_6_figures/fig_critical_curve.py --which both
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402


def plot_critical_curve(plt, args, df, crit, proxy):
    fig, ax = plt.subplots(
        figsize=C.figsize(args.width, args.aspect or 0.62))
    c = crit.dropna(subset=["Re_c"])

    for direction, mk, col, lbl in (
            ("neg_to_pos", "o", C.P("C_POS"), r"$\mu:-\!\to\!+$"),
            ("pos_to_neg", "s", C.P("C_NEG"), r"$\mu:+\!\to\!-$")):
        sub = c[c.direction == direction] if "direction" in c else c.iloc[0:0]
        if len(sub):
            ax.errorbar(sub.amp, sub.Re_c, yerr=sub.bracket / 2.0, fmt=mk,
                        ms=4, capsize=2, lw=0.9, color=col, label=lbl)
    ax.plot(c.amp, c.Re_c, "-", color=C.P("C_SYM"), lw=0.9, zorder=0)

    pr = proxy.dropna(subset=["Re_c_proxy"])
    if len(pr):
        ax.plot(pr.amp, pr.Re_c_proxy, "x", ms=4, color=C.P("C_GLOBAL"),
                label=r"$C_L^{\mathrm{FOM}}$ departure")

    ax.set_xlabel(r"rotation amplitude $a$")
    ax.set_ylabel(r"$\mathrm{Re}_c$")
    ax.legend(loc="best")
    C.save(fig, args.outdir, "fig_critical_curve", png=args.png)

    # asymmetry in a, stated rather than eyeballed
    macros = {}
    m = c.set_index(c.amp.round(6)).Re_c
    pairs = [(a, m[a], m[-a]) for a in m.index if a > 0 and -a in m.index]
    if pairs:
        d = [abs(p - n) for _, p, n in pairs]
        print("  Re_c(+a) vs Re_c(-a):")
        for a, pv, nv in pairs:
            print(f"    a=+-{a:.2f}: {pv:.2f} vs {nv:.2f}   diff={pv - nv:+.2f}")
        macros["RecAsymMax"] = C.fmt(max(d), ".2f")
        macros["RecAsymMedian"] = C.fmt(float(np.median(d)), ".2f")
    if len(c):
        macros["RecBracket"] = C.fmt(float(c.bracket.median()), ".2g")
        macros["RecMin"] = C.fmt(float(c.Re_c.min()), ".2f")
        macros["RecMax"] = C.fmt(float(c.Re_c.max()), ".2f")
        macros["NCritAmps"] = len(c)
    if len(pr) and len(c):
        j = c.merge(pr, on="amp")
        if len(j):
            dd = (j.Re_c - j.Re_c_proxy).abs()
            macros["RecVsProxyMax"] = C.fmt(float(dd.max()), ".2f")
            print(f"  |Re_c(mu) - Re_c(C_L_fom)|: median={dd.median():.2f} "
                  f"max={dd.max():.2f}  (bracket={c.bracket.median():.2g})")
    return macros


def plot_mu_sweeps(plt, args, df, crit, amps):
    fig, ax = plt.subplots(
        figsize=C.figsize(args.width, args.aspect or 0.62))
    cmap = plt.get_cmap(C.seq_cmap())
    for j, a in enumerate(amps):
        s = df[df.is_sym & np.isclose(df.amp, a)].dropna(subset=["mu"])
        s = s.sort_values("Re")
        if s.empty:
            continue
        ax.plot(s.Re, s.mu, "-", lw=1.1, color=cmap(j / max(len(amps) - 1, 1)),
                label=f"$a={a:+.2f}$")
        row = crit[np.isclose(crit.amp, a)]
        if len(row) and np.isfinite(row.Re_c.iloc[0]):
            ax.axvline(row.Re_c.iloc[0], color=cmap(j / max(len(amps) - 1, 1)),
                       ls=":", lw=0.8)
    ax.axhline(0.0, color=C.P("C_GLOBAL"), lw=0.8)
    ax.set_xlabel(r"$\mathrm{Re}$")
    ax.set_ylabel(r"reduced stability indicator $\mu$")
    ax.legend(loc="best", ncol=2)
    C.save(fig, args.outdir, "fig_mu_sweeps", png=args.png)


def main():
    p = C.base_parser(__doc__)
    p.add_argument("--tag", default="E1_K4_tensor")
    p.add_argument("--which", choices=["curve", "sweeps", "both"], default="both")
    p.add_argument("--sweep-amps", type=float, nargs="*", default=None)
    p.add_argument("--cl-frac", type=float, default=0.02)
    args = p.parse_args()
    plt = C.setup_style(args.usetex)

    df = C.load_run(args.tag, args.states_root)
    df = C.drop_duplicate_legs(
        df, C.duplicate_leg_keys(df, audit_json=args.audit, tag=args.tag))

    crit = C.mu_crossings(df)
    proxy = C.cl_fom_departure(df, args.cl_frac)

    n_no = int((crit.get("n_crossings", 0) == 0).sum()) if "n_crossings" in crit else 0
    if n_no:
        print(f"  [warn] {n_no} amplitude column(s) have no mu sign change -- "
              f"they will be absent from Re_c(a); check whether the leg simply "
              f"terminated above Re_c")

    macros = {}
    if args.which in ("curve", "both"):
        macros.update(plot_critical_curve(plt, args, df, crit, proxy))
    if args.which in ("sweeps", "both"):
        amps = args.sweep_amps
        if not amps:
            u = np.sort(df.amp.unique())
            amps = [u[i] for i in np.linspace(0, len(u) - 1,
                                              min(5, len(u))).astype(int)]
        plot_mu_sweeps(plt, args, df, crit, amps)

    crit.merge(proxy, on="amp", how="outer").to_csv(
        os.path.join(args.outdir, "critical_curve.csv"), index=False)
    print(f"  wrote {os.path.join(args.outdir, 'critical_curve.csv')}")
    if macros:
        C.write_macros(os.path.join(args.outdir, "macros_critical_curve.tex"),
                       macros)


if __name__ == "__main__":
    main()
