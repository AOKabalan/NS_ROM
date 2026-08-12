#!/usr/bin/env python3
"""
fig_rom_overlay.py -- fig:rom_overlay

Panel (a): the ROM bifurcation diagram drawn over the full-order one, at a few
representative amplitudes. FOM as a line, ROM as sparse markers, so agreement
reads as markers sitting on the line rather than as two indistinguishable curves.

Panel (b): parity. Every converged point in the sweep is one marker, with its
full-order lift on the abscissa and its reduced lift on the ordinate, coloured by
Reynolds number. An exact ROM would put every marker on the diagonal, so
agreement reads as the cloud collapsing onto the line, and any point that fails
-- on any branch, at any amplitude -- is visible without having to be looked for.
The diagram in (a) is what makes the result legible; the parity plot is what
makes it complete.

The signed-residual strip that used to sit under (b) is gone. It carried no
structure the prose does not already state as two numbers (the RMS and the
maximum, both written out as macros), it was documented as being plotted against
Re while the code plotted it against C_L^FOM, and it could not be aligned with
the panel above it because a parity plot needs an equal aspect and therefore a
narrower box than a full-width strip. --residual brings it back, against Re, for
anyone who wants to inspect the structure.

    python section_6_figures/fig_rom_overlay.py --outdir section_6_figures/out
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402


def main():
    p = C.base_parser(__doc__)
    p.add_argument("--tag", default="E1_K4_tensor")
    p.add_argument("--residual", action="store_true",
                   help="restore the signed-residual strip under panel (b), "
                        "plotted against Re")
    p.add_argument("--amps", type=float, nargs="*", default=None,
                   help="amplitudes for panel (a); default: a spread of 4")
    args = p.parse_args()
    plt = C.setup_style(args.usetex)

    df = C.load_run(args.tag, args.states_root)
    df = C.drop_duplicate_legs(
        df, C.duplicate_leg_keys(df, audit_json=args.audit, tag=args.tag))
    df = df[df.converged.astype(bool)].copy()

    cl_rom = "C_L_rom" if "C_L_rom" in df else "C_L"
    if "C_L_fom" not in df:
        raise SystemExit(f"{args.tag} has no C_L_fom -- needs a FOM_COMPUTE=1 run")

    amps = args.amps
    if not amps:
        u = np.sort(df.amp.unique())
        amps = [u[i] for i in np.linspace(0, len(u) - 1, min(4, len(u))).astype(int)]
    print(f"  panel (a) amplitudes: {amps}")

    p_res = args.residual
    if p_res:
        fig = plt.figure(figsize=C.figsize(args.width, args.aspect or 0.46),
                         layout="constrained")
        gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[1.15, 1])
        ax_d = fig.add_subplot(gs[:, 0])
        ax_p = fig.add_subplot(gs[0, 1])
        ax_r = fig.add_subplot(gs[1, 1])
    else:
        fig, (ax_d, ax_p) = plt.subplots(
            1, 2, figsize=C.figsize(args.width, args.aspect or 0.42),
            layout="constrained", width_ratios=[1.15, 1])
        ax_r = None

    # --- (a) diagram overlay
    cmap = plt.get_cmap(C.seq_cmap())
    for j, a in enumerate(amps):
        col = cmap(0.12 + 0.76 * j / max(len(amps) - 1, 1))
        blk = df[np.isclose(df.amp, a)]
        first = True
        for _, leg in blk.groupby("branch"):
            leg = leg.sort_values("Re")
            ax_d.plot(leg.Re, leg.C_L_fom, "-", color=col, lw=1.1,
                      label=f"$a={a:+.2f}$" if first else None)
            ax_d.plot(leg.Re[::4], leg[cl_rom][::4], "o", color=col, ms=2.6,
                      mfc="none", mew=0.7,
                      label="ROM" if (first and j == 0) else None)
            first = False
    ax_d.set_xlabel(r"$\mathrm{Re}$")
    ax_d.set_ylabel(r"$C_L$")
    ax_d.set_title("(a) reduced over full-order diagram", loc="left")
    ax_d.legend(loc="best", ncol=2, fontsize=6.5)

    # --- (b) parity over the whole domain
    d = df.dropna(subset=[cl_rom, "C_L_fom"])
    res = d[cl_rom].values - d.C_L_fom.values
    lim = 1.05 * float(np.nanmax(np.abs(np.r_[d[cl_rom].values,
                                              d.C_L_fom.values])))
    ax_p.plot([-lim, lim], [-lim, lim], "-", color=C.P("C_SYM"), lw=0.8, zorder=1)
    sc = ax_p.scatter(d.C_L_fom, d[cl_rom], c=d.Re, s=3, cmap=C.seq_cmap(),
                      linewidths=0, zorder=2, rasterized=True)
    ax_p.set_xlim(-lim, lim)
    ax_p.set_ylim(-lim, lim)
    ax_p.set_aspect("equal", adjustable="box")
    ax_p.set_xlabel(r"$C_L^{\mathrm{FOM}}$")
    ax_p.set_ylabel(r"$C_L^{\mathrm{ROM}}$")
    ax_p.set_title(f"(b) all {len(d)} converged points", loc="left")
    # the panel states its own accuracy, so it can be read without the caption
    ax_p.text(0.04, 0.96,
              "RMS " + C.fmt(float(np.sqrt(np.nanmean(res ** 2))), ".1e")
              + "\nmax " + C.fmt(float(np.nanmax(np.abs(res))), ".1e"),
              transform=ax_p.transAxes, ha="left", va="top", fontsize=6.5,
              color=C.P("C_SYM"))
    cb = fig.colorbar(sc, ax=ax_p, fraction=0.046, pad=0.02)
    cb.set_label(r"$\mathrm{Re}$")

    if ax_r is not None:
        ax_p.set_xlabel("")
        ax_r.axhline(0.0, color=C.P("C_SYM"), lw=0.8)
        ax_r.scatter(d.Re, res, s=3, c=C.P("C_NEG"), linewidths=0,
                     rasterized=True)
        ax_r.set_xlabel(r"$\mathrm{Re}$")
        ax_r.set_ylabel("residual")
        m = float(np.nanmax(np.abs(res))) * 1.15 or 1.0
        ax_r.set_ylim(-m, m)

    rms = float(np.sqrt(np.nanmean(res ** 2)))
    print(f"  residual: rms={rms:.3e} max={np.nanmax(np.abs(res)):.3e}")

    C.save(fig, args.outdir, "fig_rom_overlay", png=args.png)
    C.write_macros(os.path.join(args.outdir, "macros_rom_overlay.tex"), {
        "ClResidualRMS": C.fmt(rms, ".2e"),
        "ClResidualMax": C.fmt(float(np.nanmax(np.abs(res))), ".2e"),
        "NOverlayPoints": len(d),
    })


if __name__ == "__main__":
    main()
