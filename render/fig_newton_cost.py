#!/usr/bin/env python3
"""
fig_newton_cost.py -- fig:newton_cost   (REPLACES fig:newton_histories)

The old figure carried the tensor-vs-DEIM contrast on three probe points. R3 and
R4 replay the SAME parameter grid from the SAME stored full-order seeds, one with
exact tensorial convection and one with DEIM, so the comparison can be made over
every point instead. Same seeds means the difference is the hyper-reduction and
nothing else: no continuation-path effect, no warm-start advantage either way.

Three panels, one per claim the subsection has to make:

  (a) Newton cost. A performance profile: the fraction of the paired points that
      converged within n iterations. The curve PLATEAUS at the convergence rate,
      so one curve carries both "DEIM needs more iterations" and "DEIM does not
      get there at all" without the censoring sleight of hand that an ECDF over
      converged-only points would commit.
  (b) Where DEIM fails, in (Re, a), split into diverged / converged-without-mu /
      converged-with-mu, with Re_c(a) from the tensor arm overlaid. This is the
      geometry that rem:deim_mechanism predicts: the bias term dominates where
      J_red is ill-conditioned, i.e. approaching the pitchfork.
  (c) Cost per solve and cost per Newton iterate. The second is the one the
      "DEIM is not even cheaper" paragraph actually needs -- a slower solve could
      just be more iterations, and dividing separates the two.

What this script refuses to do:

  * classify duplicate legs on the DEIM arm. C_L is NaN over most shared Re
    there, so no leg can be tested and the filter would silently pass everything
    through. Legs are classified once on the master run (--legs-from) and the
    same set is applied to both arms.
  * quote any ratio over sets that differ. Every number in panel (c) and every
    ratio macro comes from the points where BOTH arms converged. That restriction
    discards exactly the points DEIM found hardest, so it flatters DEIM; the
    printed output says so, and the claim survives it.
  * treat a non-converged point as a data point with a large iteration count.
    Those points hit the cap by construction and are handled in panel (a) as the
    height of the plateau, never as a number in a median.

    python render/fig_newton_cost.py --outdir render/out
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

KEYS = ["Re", "amp", "branch"]


def tighten(plt, on=True):
    """setup_style is sized for a ONE-panel figure: 10 pt titles and labels in a
    2-inch panel collide by construction. Scale the text down for a 3-across
    layout. Applied after set_style so it wins, and skippable with --no-tighten
    if the house style is ever resized for multi-panel use."""
    if not on:
        return
    plt.rcParams.update({
        "font.size": 7, "axes.labelsize": 7.5, "axes.titlesize": 7.5,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
        "lines.linewidth": 1.0, "lines.markersize": 3.5,
        "axes.titlepad": 3.0, "axes.labelpad": 2.0,
    })


def thin_log_ticks(ax, axis="x", n=4):
    """Log minor-tick labels are what turn an axis into a smear."""
    import matplotlib.ticker as mt
    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_major_locator(mt.LogLocator(numticks=n))
    a.set_minor_locator(mt.LogLocator(subs="auto", numticks=n * 3))
    a.set_minor_formatter(mt.NullFormatter())


# ---------------------------------------------------------------------------

def prep(tag, root, legs, verbose=True):
    df = C.load_run(tag, root)
    n0 = len(df)
    df = C.drop_duplicate_legs(df, legs)
    if verbose:
        print(f"  {tag:22s} {n0:5d} rows, {len(df):5d} after the leg filter")
    for c in ("converged", "iterations", "t_rom", "mu"):
        if c not in df.columns:
            print(f"  [warn] {tag}: no '{c}' column")
    df["_conv"] = df.converged.astype(bool) if "converged" in df else False
    df["_mu_ok"] = (df.mu.notna() if "mu" in df else False) & df["_conv"]
    return df


def pair(a, b, tag_a, tag_b):
    """Inner join on (Re, amp, branch). Report what does not pair."""
    def key(df):
        k = df.copy()
        k["Re"] = k.Re.round(6)
        k["amp"] = k.amp.round(6)
        return k

    cols = KEYS + ["_conv", "_mu_ok", "iterations", "t_rom"]
    cols_a = [c for c in cols if c in a.columns]
    cols_b = [c for c in cols if c in b.columns]
    m = key(a)[cols_a].merge(key(b)[cols_b], on=KEYS, how="inner",
                             suffixes=("_t", "_d"))
    n_dup = len(m) - len(m.drop_duplicates(subset=KEYS))
    if n_dup:
        print(f"  [warn] {n_dup} duplicated key(s) in the join -- (Re, amp, "
              f"branch) is not unique in one of the arms; taking the first")
        m = m.drop_duplicates(subset=KEYS)
    print(f"  paired on (Re, amp, branch): {len(m)} points "
          f"({tag_a}: {len(a)}, {tag_b}: {len(b)})")
    if len(m) < min(len(a), len(b)):
        print(f"  [note] {min(len(a), len(b)) - len(m)} point(s) of the smaller "
              f"arm did not pair -- the two runs are not on an identical grid")
    return m


# ---------------------------------------------------------------------------

def panel_profile(ax, m, cap, colors, labels):
    """Fraction of paired points converged within n Newton iterations."""
    if "iterations_t" not in m or "iterations_d" not in m:
        ax.text(0.5, 0.5, "no 'iterations' column", ha="center", va="center",
                transform=ax.transAxes, fontsize=7, color="0.4")
        return {}

    n = len(m)
    stats = {}
    for suf, col, lab in (("_t", colors[0], labels[0]),
                          ("_d", colors[1], labels[1])):
        conv = m[m["_conv" + suf]]
        v = pd.to_numeric(conv["iterations" + suf], errors="coerce").dropna()
        v = v[v < cap]                       # a point at the cap did not converge
        if v.empty:
            continue
        u, c = np.unique(v.values, return_counts=True)
        y = np.cumsum(c) / n
        ax.step(np.r_[0, u], np.r_[0, y], where="post", color=col, lw=1.4,
                label=lab)
        ax.plot([u[-1], cap], [y[-1], y[-1]], "-", color=col, lw=1.4)
        stats[suf] = {"median": float(np.median(v)),
                      "p90": float(np.quantile(v, 0.9)),
                      "rate": float(y[-1]), "max": float(u[-1])}
    ax.axhline(1.0, color=C.P("C_SYM"), lw=0.6, ls=":")
    ax.set_xscale("log")
    ax.set_xlim(1, cap)
    ax.set_ylim(0, 1.05)
    thin_log_ticks(ax, "x")
    ax.set_xlabel("Newton iterations")
    ax.set_ylabel("solved fraction")
    ax.set_title("(a) Newton cost", loc="left")
    ax.legend(loc="lower right")
    return stats


def panel_failmap(ax, d, crit, colors):
    """Where the DEIM arm fails, against Re_c(a) from the tensor arm."""
    ok = d[d["_mu_ok"]]
    nomu = d[d["_conv"] & ~d["_mu_ok"]]
    div = d[~d["_conv"]]
    ax.scatter(ok.Re, ok.amp, s=3, c="0.78", linewidths=0, rasterized=True,
               label=r"converged, $\mu$ available")
    ax.scatter(nomu.Re, nomu.amp, s=7, c=C.P("C_POS"), linewidths=0,
               marker="s", rasterized=True, label=r"converged, $\mu$ absent")
    ax.scatter(div.Re, div.amp, s=9, c=C.P("C_NEG"), linewidths=0, marker="x",
               rasterized=True, label="not converged")
    if crit is not None and len(crit) and "Re_c" in crit:
        c = crit.dropna(subset=["Re_c"])
        if len(c):
            ax.plot(c.Re_c, c.amp, "-", color="k", lw=1.3, zorder=6,
                    label=r"$\mathrm{Re}_c(a)$ (tensor)")
    ax.set_xlabel(r"$\mathrm{Re}$")
    ax.set_ylabel(r"amplitude $a$")
    ax.set_title("(b) DEIM failures", loc="left")
    # the (Re, a) plane is full everywhere, so an inset legend always covers
    # data; the caller lifts these into a figure-level legend
    return ax.get_legend_handles_labels()


def panel_cost(ax, m, colors, labels):
    """Per-solve and per-iterate cost, on the both-converged pairs only."""
    both = m[m["_conv_t"] & m["_conv_d"]]
    if "t_rom_t" not in both or "t_rom_d" not in both or both.empty:
        ax.text(0.5, 0.5, "no 't_rom' column", ha="center", va="center",
                transform=ax.transAxes, fontsize=7, color="0.4")
        return {}, both

    series, stats = [], {}
    for suf in ("_t", "_d"):
        t = 1e3 * pd.to_numeric(both["t_rom" + suf], errors="coerce")
        it = pd.to_numeric(both["iterations" + suf], errors="coerce")
        per = (t / it.where(it > 0)).dropna()
        series.append((t.dropna(), per))
        stats[suf] = {"t": float(t.median()), "per": float(per.median())}

    # four tick labels in a 2-inch panel overlap; the arms are keyed by colour
    # and the x axis names the two QUANTITIES instead
    data = [series[0][0], series[1][0], series[0][1], series[1][1]]
    bp = ax.boxplot(data, positions=[1, 1.8, 3.3, 4.1], widths=0.6,
                    showfliers=False, patch_artist=True,
                    medianprops=dict(color="k", lw=1.0))
    for i, box in enumerate(bp["boxes"]):
        box.set(facecolor=colors[i % 2], alpha=0.40, linewidth=0.7)
    ax.set_yscale("log")
    ax.set_xticks([1.4, 3.7])
    ax.set_xticklabels(["per solve", "per iterate"])
    ax.set_xlim(0.5, 4.6)
    ax.set_ylabel("ms")
    ax.set_title("(c) cost", loc="left")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=colors[i], alpha=0.40, edgecolor="0.3",
                             label=labels[i]) for i in (0, 1)],
              loc="lower left", handlelength=1.2, handletextpad=0.4,
              borderpad=0.25, labelspacing=0.25)
    return stats, both


# ---------------------------------------------------------------------------

def near_critical(d, crit, margin):
    """Fraction of DEIM failures at Re >= Re_c(a) - margin."""
    if crit is None or not len(crit) or "Re_c" not in crit:
        return None
    c = crit.dropna(subset=["Re_c"])
    if c.empty:
        return None
    rc = pd.Series(c.Re_c.values, index=c.amp.round(6).values)
    f = d[~d["_conv"]].copy()
    f["Re_c"] = f.amp.round(6).map(rc)
    known = f.dropna(subset=["Re_c"])
    if known.empty:
        return None
    return {"n_failures": len(f), "n_with_rec": len(known),
            "frac_near": float((known.Re >= known["Re_c"] - margin).mean()),
            "median_offset": float((known.Re - known["Re_c"]).median())}


def main():
    p = C.base_parser(__doc__)
    p.add_argument("--tensor", default="R3_tensor_fom")
    p.add_argument("--deim", default="R4_deim8_fom")
    p.add_argument("--legs-from", default="E1_K4_tensor",
                   help="master run for duplicate-leg classification; the DEIM "
                        "arm cannot classify its own legs")
    p.add_argument("--iter-cap", type=int, default=350,
                   help="Newton iteration cap; a point at the cap is treated as "
                        "not converged, never as an iteration count")
    p.add_argument("--margin", type=float, default=10.0,
                   help="Re window below Re_c(a) counted as 'near' in (b)")
    p.add_argument("--no-tighten", dest="tighten", action="store_false",
                   help="keep the house single-panel font sizes")
    args = p.parse_args()
    plt = C.setup_style(args.usetex)
    tighten(plt, args.tighten)

    # --- legs, classified once on the master run
    legs = set()
    if os.path.isdir(C.state_dir(args.legs_from, args.states_root)):
        legs = C.legs_from_run(args.legs_from, args.states_root,
                               audit_json=args.audit)
    else:
        print(f"  [warn] {args.legs_from} not found -- classifying legs on "
              f"{args.tensor} instead; do NOT do this on the DEIM arm")
        legs = C.duplicate_leg_keys(
            C.load_run(args.tensor, args.states_root),
            audit_json=args.audit, tag=args.tensor)

    t = prep(args.tensor, args.states_root, legs)
    d = prep(args.deim, args.states_root, legs)
    m = pair(t, d, args.tensor, args.deim)
    if m.empty:
        raise SystemExit("no paired points -- are the two runs on the same grid?")

    crit = C.mu_crossings(t)
    n_no = int((crit.get("n_crossings", 0) == 0).sum()) if "n_crossings" in crit else 0
    if n_no:
        print(f"  [note] {n_no} amplitude(s) have no mu sign change in the "
              f"tensor arm; those columns carry no Re_c in panel (b)")

    colors = [C.cluster_color(0), C.P("C_NEG")]
    labels = ["tensor", "DEIM"]

    fig, axes = plt.subplots(
        1, 3, figsize=C.figsize(args.width, args.aspect or 0.36),
        layout="constrained")
    prof = panel_profile(axes[0], m, args.iter_cap, colors, labels)
    h, l = panel_failmap(axes[1], d, crit, colors)
    cost, both = panel_cost(axes[2], m, colors, labels)
    # constrained layout packs panels tighter than tick labels allow; pad it
    fig.get_layout_engine().set(w_pad=0.045, h_pad=0.02, wspace=0.07)
    if h:
        fig.legend(h, l, loc="outside lower center", ncol=4, markerscale=1.6,
                   handletextpad=0.4, columnspacing=1.4, borderpad=0.3)

    C.save(fig, args.outdir, "fig_newton_cost", png=args.png)

    # ---------------------------------------------------------------- numbers
    macros = {"NewtonPaired": len(m), "NewtonIterCap": args.iter_cap,
              "NewtonBothConverged": int(len(both))}
    print("\n  --- Newton cost over the paired points ---")
    for suf, lab, key in (("_t", "tensor", "Tensor"), ("_d", "DEIM", "Deim")):
        conv = float(m[f"_conv{suf}"].mean())
        muok = float(m[f"_mu_ok{suf}"].mean())
        macros[f"Newton{key}ConvPct"] = C.fmt(100 * conv, ".1f")
        macros[f"Newton{key}MuPct"] = C.fmt(100 * muok, ".1f")
        s = prof.get(suf)
        if s:
            macros[f"Newton{key}IterMedian"] = C.fmt(s["median"], ".0f")
            macros[f"Newton{key}IterPninety"] = C.fmt(s["p90"], ".0f")
            macros[f"Newton{key}IterMax"] = C.fmt(s["max"], ".0f")
        print(f"  {lab:7s} converged {100 * conv:5.1f}%   mu available "
              f"{100 * muok:5.1f}%" + (f"   iterations median={s['median']:.0f} "
                                       f"p90={s['p90']:.0f} max={s['max']:.0f}"
                                       if s else ""))
    if len(prof) == 2:
        ratio = prof["_d"]["median"] / max(prof["_t"]["median"], 1e-12)
        macros["NewtonIterRatio"] = C.fmt(ratio, ".1f")
        print(f"  median iteration ratio DEIM/tensor: {ratio:.1f}x")

    if cost:
        macros["TSolveTensorMs"] = C.fmt(cost["_t"]["t"], ".1f")
        macros["TSolveDeimMs"] = C.fmt(cost["_d"]["t"], ".1f")
        macros["TIterTensorMs"] = C.fmt(cost["_t"]["per"], ".2f")
        macros["TIterDeimMs"] = C.fmt(cost["_d"]["per"], ".2f")
        rs = cost["_d"]["t"] / max(cost["_t"]["t"], 1e-12)
        ri = cost["_d"]["per"] / max(cost["_t"]["per"], 1e-12)
        macros["TSolveRatio"] = C.fmt(rs, ".1f")
        macros["TIterRatio"] = C.fmt(ri, ".1f")
        print(f"\n  --- cost, on the {len(both)} pairs where BOTH converged ---")
        print(f"  per solve   tensor {cost['_t']['t']:7.1f} ms   "
              f"DEIM {cost['_d']['t']:7.1f} ms   ratio {rs:.1f}x")
        print(f"  per iterate tensor {cost['_t']['per']:7.2f} ms   "
              f"DEIM {cost['_d']['per']:7.2f} ms   ratio {ri:.1f}x")
        print("  this subset excludes every point DEIM failed on, so it "
              "flatters DEIM;")
        print(f"  {'it is still slower on both measures' if min(rs, ri) > 1 else 'READ THE SIGNS -- one ratio is below 1'}.")

    nc = near_critical(d, crit, args.margin)
    if nc:
        macros["DeimFailNearRecPct"] = C.fmt(100 * nc["frac_near"], ".0f")
        macros["DeimFailMargin"] = C.fmt(args.margin, ".0f")
        print(f"\n  --- geography of the {nc['n_failures']} DEIM failures ---")
        print(f"  {nc['n_with_rec']} sit at an amplitude where the tensor arm "
              f"resolves Re_c")
        print(f"  {100 * nc['frac_near']:.0f}% of those are at "
              f"Re >= Re_c(a) - {args.margin:g}; median offset "
              f"{nc['median_offset']:+.1f}")
        if nc["frac_near"] < 0.5:
            print("  !! the failures are NOT concentrated near the bifurcation "
                  "-- rem:deim_mechanism does not explain them; say so or drop "
                  "the mechanism reading of this panel")

    # mirror symmetry: printed, not plotted -- R2 carries it more strongly
    sym_split(t, d, macros)

    os.makedirs(args.outdir, exist_ok=True)
    out_csv = os.path.join(args.outdir, "newton_cost_paired.csv")
    m.to_csv(out_csv, index=False)
    print(f"\n  wrote {out_csv}")
    C.write_macros(os.path.join(args.outdir, "macros_newton_cost.tex"), macros)


def sym_split(t, d, macros):
    """Does DEIM break the +/- mirror pair that the tensor path keeps equal?"""
    def split(df):
        a = (df[~df.is_sym & df["_conv"]].dropna(subset=["C_L"])
             if "C_L" in df else None)
        if a is None or a.empty:
            return None
        # kind can be 'asym+1', 'asym+1@amp-0.10' or 'asym+1@spine-', so the
        # sign must be read from the head, not from the whole string, and the
        # pair must be matched by sign rather than by column order
        head = a.kind.astype(str).str.split("@").str[0]
        a = a.assign(_mirror=np.where(head.str.contains(r"\+"), "+",
                                      np.where(head.str.contains("-"), "-", "")))
        a = a[a._mirror != ""]
        if a.empty:
            return None
        piv = a.pivot_table(index=["amp", "Re"], columns="_mirror",
                            values="C_L", aggfunc="first")
        if not {"+", "-"} <= set(piv.columns):
            return None
        g = (piv["+"].abs() - piv["-"].abs()).abs().dropna()
        scale = float(a.C_L.abs().max()) or 1.0
        return float(g.median() / scale), float(g.max() / scale), len(g)

    st, sd = split(t), split(d)
    if not st or not sd:
        return
    print("\n  --- mirror pair |C_L^+| vs |C_L^-|, relative to the C_L scale ---")
    print(f"  tensor  median {st[0]:.2e}  max {st[1]:.2e}  (n={st[2]})")
    print(f"  DEIM    median {sd[0]:.2e}  max {sd[1]:.2e}  (n={sd[2]})")
    macros["MirrorSplitTensor"] = C.fmt(st[0], ".1e")
    macros["MirrorSplitDeim"] = C.fmt(sd[0], ".1e")
    if sd[0] > 10 * st[0]:
        print("  DEIM breaks the mirror symmetry the tensor path preserves; "
              "this is a separate claim from accuracy and is worth one sentence")


if __name__ == "__main__":
    main()
