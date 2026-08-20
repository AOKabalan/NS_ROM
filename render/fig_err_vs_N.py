#!/usr/bin/env python3
"""
fig_err_vs_N.py -- fig:err_vs_N

Mean and max error against the local reduced dimension, one point per member of
the POD-tolerance family (1e-4, 1e-6, 1e-8 = E1, 1e-10). N comes from the offline
cache -- the leading dimension of each cluster's convection tensor -- so the x
axis is the dimension the run actually used, not the dimension it was asked for.

Two things this script refuses to fake:

  * the POD projection-error curve on the same axes needs the stored bases and
    E1's FOM fields, i.e. a computation, not a plot. Supply it as a CSV with
    columns (N, proj_err) via --projection, or the curve is simply absent and
    the caption must not promise it.
  * if the tolerance legs sit on different amplitude grids they are not
    comparable. The script compares point counts and amplitude sets across the
    family and refuses to plot mismatched members unless --allow-mixed-grids.

    python render/fig_err_vs_N.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

DEFAULT_FAMILY = [
    ("E7_K4_tensor_tol1e-4", 1e-4),
    ("E7_K4_tensor_tol1e-6", 1e-6),
    ("E1_K4_tensor", 1e-8),
    ("E7_K4_tensor_tol1e-10", 1e-10),
]


def main():
    p = C.base_parser(__doc__)
    p.add_argument("--K", type=int, default=4)
    p.add_argument("--family", nargs="*", default=None,
                   help="TAG=TOL pairs, e.g. E1_K4_tensor=1e-8")
    p.add_argument("--projection", default=None,
                   help="CSV with columns N,proj_err")
    p.add_argument("--allow-mixed-grids", action="store_true")
    p.add_argument("--common-points", action="store_true", default=True,
                   help="average only over points every member converged at "
                        "(default on; --all-points to disable)")
    p.add_argument("--all-points", dest="common_points", action="store_false")
    p.add_argument("--x", choices=["sum", "max"], default="sum",
                   help="N as sum over clusters (online storage) or max (per-solve)")
    args = p.parse_args()
    plt = C.setup_style(args.usetex)
    nu = C.norm_label(args.mass_meta, "M_u")

    family = DEFAULT_FAMILY
    if args.family:
        family = [(s.split("=")[0], float(s.split("=")[1])) for s in args.family]

    dims = C.cache_dims(args.cache_root)
    rows, grids, frames = [], {}, {}
    for tag, tol in family:
        sd = C.state_dir(tag, args.states_root)
        if not os.path.isdir(sd):
            print(f"  [skip] {tag}: no such state dir")
            continue
        df = C.load_run(tag, args.states_root)
        df = C.drop_duplicate_legs(
            df, C.duplicate_leg_keys(df, audit_json=args.audit, tag=tag,
                                     verbose=False))
        df = df[df.converged.astype(bool)]
        grids[tag] = (len(df), tuple(np.sort(df.amp.unique())))

        r = dims.get((args.K, tol))
        if r is None or any(x is None for x in r):
            print(f"  [skip] {tag}: no reduced dimensions in cache for "
                  f"K={args.K}, tol={tol:g}")
            continue
        frames[tag] = df
        rows.append({"tag": tag, "tol": tol,
                     "N": sum(r) if args.x == "sum" else max(r), "r": r})

    if not rows:
        raise SystemExit("nothing to plot -- check the tolerance caches exist")

    # A looser tolerance loses its HARDEST points to divergence, so averaging
    # each run over its own surviving set flatters it. Restrict to the points
    # every member converged at, and report how many that leaves.
    keys = None
    for tag, df in frames.items():
        k = set(zip(df.Re.round(6), df.amp.round(6), df.branch))
        keys = k if keys is None else (keys & k)
    print(f"  common converged points across the family: {len(keys)}")
    for row in rows:
        df = frames[row["tag"]]
        if args.common_points:
            m = [(re, am, br) in keys for re, am, br
                 in zip(df.Re.round(6), df.amp.round(6), df.branch)]
            df = df[pd.Series(m, index=df.index)]
        e = df.err_u.dropna() if "err_u" in df else pd.Series(dtype=float)
        row.update(n_points=len(df),
                   mean=float(e.mean()) if len(e) else np.nan,
                   median=float(e.median()) if len(e) else np.nan,
                   max=float(e.max()) if len(e) else np.nan)
    d = pd.DataFrame(rows).sort_values("N")

    # grid comparability
    sizes = {t: g[0] for t, g in grids.items()}
    ampsets = {t: g[1] for t, g in grids.items()}
    if len(set(ampsets.values())) > 1:
        print("\n  !! the tolerance family does NOT share one amplitude grid:")
        for t in grids:
            print(f"       {t:26s} n={sizes[t]:5d} n_amps={len(ampsets[t])}")
        if not args.allow_mixed_grids:
            raise SystemExit(
                "  refusing to plot: members on different grids are not "
                "comparable. Re-run the odd legs with NSROM_AMPS set, or pass "
                "--allow-mixed-grids if you intend to caption the difference.")

    fig, ax = plt.subplots(
        figsize=C.figsize(args.width, args.aspect or 0.62))
    ax.plot(d.N, d["mean"], "o-", label="mean", ms=4, color=C.cluster_color(0))
    ax.plot(d.N, d["max"], "s--", label="max", ms=4, color=C.P("C_NEG"))
    if args.projection and os.path.exists(args.projection):
        pj = pd.read_csv(args.projection).sort_values("N")
        ax.plot(pj.N, pj.proj_err, "^:", color=C.P("C_SYM"), ms=4,
                label="POD projection error")
    elif args.projection:
        print(f"  [warn] {args.projection} not found -- projection curve omitted")
    else:
        print("  [note] no --projection CSV: the projection-error curve is "
              "absent, so the caption must not promise it")

    for _, r in d.iterrows():
        ax.annotate(f"$10^{{{int(round(np.log10(r.tol)))}}}$",
                    (r.N, r["mean"]), textcoords="offset points",
                    xytext=(4, -9), fontsize=7, color=C.P("C_SYM"))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"reduced dimension $N$"
                  + (r" ($\sum_k r_k$)" if args.x == "sum" else r" ($\max_k r_k$)"))
    ax.set_ylabel(rf"relative error, $\|\cdot\|_{{{nu}}}$")
    ax.legend(loc="best")
    C.save(fig, args.outdir, "fig_err_vs_N", png=args.png)

    d.to_csv(os.path.join(args.outdir, "err_vs_N.csv"), index=False)
    print("\n  " + d.to_string(index=False))
    C.write_macros(os.path.join(args.outdir, "macros_err_vs_N.tex"), {
        "ErrVsNMinN": int(d.N.min()), "ErrVsNMaxN": int(d.N.max()),
        "ErrVsNBestMean": C.fmt(float(d["mean"].min()), ".2e"),
        "ErrVsNWorstMean": C.fmt(float(d["mean"].max()), ".2e"),
    })


if __name__ == "__main__":
    main()
