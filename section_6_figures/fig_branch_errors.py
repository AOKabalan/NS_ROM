#!/usr/bin/env python3
"""
fig_branch_errors.py -- fig:branch_errors  (PLOT ONLY)

The claim this figure carries is that the largest errors sit at the low-Re corner
at large |a|, NOT near Re_c -- which inverts the usual a-posteriori indicator.
That is an amplitude effect, so collapsing amplitudes with groupby('Re').mean()
destroys exactly the result being reported. Hence the (Re, a) plane, with the
critical curve overlaid so the reader can see that the hot corner is far from it.

Velocity errors come from the index (err_u, computed at run time in the M_u inner
product from float64 fields). Pressure errors come from compute_point_errors.py
(err_p_M, same convention, but recovered from float32-stored fields -- see the
noise-floor note that script prints).

    python section_6_figures/fig_branch_errors.py \
        --errors paper_data/point_errors_E1.csv --outdir section_6_figures/out
"""

import os
import sys

import numpy as np
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402


def panel(ax, d, col, label, crit=None, vmin=None, vmax=None):
    if col not in d.columns:
        ax.text(0.5, 0.5, f"{col} unavailable\n(run compute_point_errors.py)",
                ha="center", va="center", transform=ax.transAxes, fontsize=7,
                color="0.4")
        ax.set_xlabel(r"$\mathrm{Re}$")
        ax.set_title(label, loc="left", fontsize=9, pad=5)
        return None
    d = d.dropna(subset=[col, "Re", "amp"])
    if d.empty:
        ax.text(0.5, 0.5, f"no {col}", ha="center", va="center",
                transform=ax.transAxes)
        return None
    v = np.log10(np.clip(d[col].values, 1e-16, None))
    sc = ax.scatter(d.Re, d.amp, c=v, s=10, cmap=C.seq_cmap(), linewidths=0,
                    vmin=vmin, vmax=vmax, rasterized=True)
    if crit is not None and len(crit):
        c = crit.dropna(subset=["Re_c"])
        ax.plot(c.Re_c, c.amp, "-", color=C.P("C_NEG"), lw=1.4, zorder=5,
                label=r"$\mathrm{Re}_c(a)$")
        ax.legend(
            loc="lower left",
            fontsize=8,
            handlelength=1.6,
            borderpad=0.2,
        )
    ax.set_xlabel(r"$\mathrm{Re}$")
    ax.set_title(label, loc="left", fontsize=9, pad=5)
    return sc


def main():
    p = C.base_parser(__doc__)
    p.add_argument("--tag", default="E1_K4_tensor")
    p.add_argument("--errors", default="paper_data/point_errors_E1.csv")
    p.add_argument("--per-leg", action="store_true",
                   help="add a third panel with per-leg curves vs Re")
    args = p.parse_args()
    plt = C.setup_style(args.usetex)

    nu = C.norm_label(args.mass_meta, "M_u")
    npr = C.norm_label(args.mass_meta, "M_p")
    if r"\ast" in (nu, npr):
        print("  [warn] mass/mass_meta.json not found -- axis labels will show "
              "an unresolved norm. Run the dump before submitting.")

    df = C.load_run(args.tag, args.states_root)
    df = C.drop_duplicate_legs(
        df, C.duplicate_leg_keys(df, audit_json=args.audit, tag=args.tag))
    df = df[df.converged.astype(bool)].copy()
    df = C.attach_point_errors(df, args.errors)

    crit = C.mu_crossings(df)

    if args.per_leg:
        fig = plt.figure(figsize=C.figsize(args.width, args.aspect or 0.55),
                         constrained_layout=True)

        gs = GridSpec(
            2, 3,
            figure=fig,
            height_ratios=[20, 1.3],
            width_ratios=[1, 1, 1],
        )

        ax_u = fig.add_subplot(gs[0, 0])
        ax_p = fig.add_subplot(gs[0, 1], sharey=ax_u)
        ax_leg = fig.add_subplot(gs[0, 2])
        cax = fig.add_subplot(gs[1, 0:2])

        axes = [ax_u, ax_p, ax_leg]
    else:
        fig = plt.figure(figsize=C.figsize(args.width, args.aspect or 0.48),
                         constrained_layout=True)

        gs = GridSpec(
            2, 2,
            figure=fig,
            height_ratios=[20, 1.3],
        )

        ax_u = fig.add_subplot(gs[0, 0])
        ax_p = fig.add_subplot(gs[0, 1], sharey=ax_u)
        cax = fig.add_subplot(gs[1, :])

        axes = [ax_u, ax_p]

    have = [c for c in ("err_u", "err_p_M") if c in df and df[c].notna().any()]
    lo = min(np.log10(np.clip(df[c].min(), 1e-16, None)) for c in have) if have else -12
    hi = max(np.log10(np.clip(df[c].max(), 1e-16, None)) for c in have) if have else 0

    s1 = panel(axes[0], df, "err_u",
               rf"(a) velocity, $\|\cdot\|_{{{nu}}}$", crit, lo, hi)
    s2 = panel(axes[1], df, "err_p_M",
               rf"(b) pressure, $\|\cdot\|_{{{npr}}}$", crit, lo, hi)
    axes[0].set_ylabel(r"rotation amplitude $a$")

    sc = s1 or s2
    if sc is not None:
        cb = fig.colorbar(
            sc,
            cax=cax,
            orientation="horizontal",
        )
        cb.set_label(r"$\log_{10}$ relative error")

    if args.per_leg:
        ax = axes[2]
        seen = set()
        for (amp, kind), leg in df.groupby(["amp", "kind"]):
            leg = leg.dropna(subset=["err_u"]).sort_values("Re")
            if len(leg) < 3:
                continue
            sym = bool(leg.is_sym.iloc[0])
            ax.plot(leg.Re, leg.err_u, lw=0.8, color=C.branch_color(kind),
                    alpha=0.85 if not sym else 0.5, ls="-" if not sym else ":",
                    label=None if kind in seen else str(kind).split("@")[0])
            seen.add(kind)
        h, l = ax.get_legend_handles_labels()
        if h:
            ax.legend(
                h[:3], l[:3],
                loc="lower left",
                fontsize=8,
                frameon=False
            )
        ax.set_yscale("log")
        ax.set_xlabel(r"$\mathrm{Re}$")
        ax.set_ylabel(rf"$\|u-u_N\|_{{{nu}}}/\|u\|_{{{nu}}}$")
        ax.set_title("(c) per leg", loc="left", fontsize=9, pad=5)

    for ax in axes:
        ax.tick_params(direction="in", pad=2)

    if args.per_leg:
        axes[2].tick_params(axis="y", pad=2)

    # Turn off the y-axis numbers for panel (b) since they share the same axis scale
    axes[1].tick_params(labelleft=False)

    C.save(fig, args.outdir, "fig_branch_errors", png=args.png)

    # the numbers the accompanying remark asserts
    macros = {}
    for col, name in (("err_u", "ErrU"), ("err_p_M", "ErrP")):
        if col in df and df[col].notna().any():
            d = df.dropna(subset=[col])
            worst = d.loc[d[col].idxmax()]
            macros[f"{name}Median"] = C.fmt(float(d[col].median()), ".2e")
            macros[f"{name}Max"] = C.fmt(float(d[col].max()), ".2e")
            macros[f"{name}MaxRe"] = C.fmt(float(worst.Re), ".0f")
            macros[f"{name}MaxAmp"] = C.fmt(float(worst.amp), "+.2f")
            print(f"  {col}: median={d[col].median():.3e} "
                  f"max={d[col].max():.3e} at Re={worst.Re:.0f} a={worst.amp:+.2f}")
            # does the corner claim actually hold?
            near = d[(d.Re > d.Re.quantile(0.6))]
            corner = d[(d.Re < d.Re.quantile(0.4)) & (d.amp.abs() > 0.6)]
            if len(near) and len(corner):
                print(f"  corner/high-Re median ratio for {col}: "
                      f"{corner[col].median() / near[col].median():.2f}"
                      "  (>1 supports the remark)")
                macros[f"{name}CornerRatio"] = C.fmt(
                    float(corner[col].median() / near[col].median()), ".2f")
    C.write_macros(os.path.join(args.outdir, "macros_branch_errors.tex"), macros)


if __name__ == "__main__":
    main()