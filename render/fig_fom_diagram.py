#!/usr/bin/env python3
"""
fig_fom_diagram.py -- fig:fom_diagram

The full-order bifurcation diagram at a = 0: C_L^FOM against Re on the symmetric
branch and the two mirror asymmetric branches, with the pitchfork located. This
is the reference every reduced quantity in Section 6 is measured against, so it
plots the FULL-ORDER lift only. No reduced curve appears here -- that overlay is
fig:rom_overlay, and mixing them would make the reference self-referential.

    python render/fig_fom_diagram.py --outdir render/out

WHERE Re_c(0) COMES FROM.
The full-order generalised eigensweep (G1) does not exist, so the crossing is
located by full-order lift departure: the first Re at which an asymmetric leg's
|C_L^FOM| clears a fixed fraction of its own scale. At a = 0 that is principled
rather than a convenience -- with no rotation the symmetric branch carries no
lift, so departure from zero IS the symmetry breaking. It is still a bracket of
one continuation step, and it is labelled as a lift departure everywhere it is
printed. Pass --rec-fom once G1 exists and the annotation switches.

The same proxy, at the same --cl-frac, backs fig:rec_convergence. Change one and
change both, or the two figures will quote different values for Re_c(0).

STABILITY STYLING.
Above Re_c the symmetric branch is drawn dashed. That is not a measurement: for
a supercritical pitchfork the symmetric state above the crossing is the unstable
spine by construction, which is why continuation along it is fragile (see the
robustness paragraph in 6.2). No stability claim is read off the reduced mu
here; doing so would import a reduced quantity into the full-order reference.

VELOCITY-FIELD INSETS.
This script does not render fields. common's design rule is that figure scripts
plot and anything needing stored fields is computed elsewhere; rendering a
Firedrake function needs the mesh and the solver environment, which a plotting
script has no business loading. Supply pre-rendered images:

    --inset-sym  paper_data/fields/sym_Re40.png
    --inset-asym paper_data/fields/asym_Re90.png

Without them the panel is drawn without insets and the script says so, because
the caption in the draft promises "representative velocity fields" and must not
if they are absent.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402


def rec_zero(df, args):
    """Re_c(0) from full-order lift departure, or the supplied G1 value."""
    if args.rec_fom is not None:
        print(f"  Re_c(0) = {args.rec_fom:g} (supplied via --rec-fom)")
        return {"Re_c": float(args.rec_fom), "bracket": float(args.rec_fom_bracket),
                "kind": "eigenvalue"}
    if "C_L_fom" not in df or not df.C_L_fom.notna().any():
        print("  [warn] no C_L_fom column -- this run was not made with "
              "FOM_COMPUTE=1, so there is no full-order diagram to draw")
        return None
    pr = C.cl_fom_departure(df, args.cl_frac)
    row = pr[np.isclose(pr.amp, 0.0)]
    if row.empty or not np.isfinite(row.Re_c_proxy.iloc[0]):
        print("  [warn] the asymmetric legs never clear the departure threshold "
              "-- Re_c(0) cannot be located from this run")
        return None
    step = (float(np.median(np.diff(np.sort(df.Re.unique()))))
            if df.Re.nunique() > 1 else np.nan)
    re = float(row.Re_c_proxy.iloc[0])
    print(f"  Re_c(0) = {re:.2f} by full-order lift departure "
          f"(frac={args.cl_frac:g}, step={step:g})")
    print("  NB a lift bracket, NOT the full-order eigenvalue crossing")
    return {"Re_c": re, "bracket": step, "kind": "lift departure"}


def add_inset(ax, path, rect, label):
    """Drop a pre-rendered field image into the axes."""
    if not path:
        return False
    if not os.path.exists(path):
        print(f"  [warn] inset image not found: {path}")
        return False
    import matplotlib.image as mpimg
    im = mpimg.imread(path)
    ins = ax.inset_axes(rect)
    ins.imshow(im)
    ins.set_xticks([])
    ins.set_yticks([])
    for sp in ins.spines.values():
        sp.set_linewidth(0.6)
        sp.set_color("0.4")
    if label:
        ins.set_title(label, fontsize=6, pad=1.5)
    return True


def main():
    p = C.base_parser(__doc__)
    p.add_argument("--tag", default="E13_K4_a0_tol1e-8")
    p.add_argument("--cl-frac", type=float, default=0.02,
                   help="departure threshold, as a fraction of the leg's own "
                        "scale; MUST match fig_rec_convergence.py")
    p.add_argument("--rec-fom", type=float, default=None,
                   help="Re_c(0) from the G1 eigensweep once it exists")
    p.add_argument("--rec-fom-bracket", type=float, default=1.0)
    p.add_argument("--inset-sym", default=None,
                   help="pre-rendered velocity field on the symmetric branch")
    p.add_argument("--inset-asym", default=None,
                   help="pre-rendered velocity field on an asymmetric branch")
    p.add_argument("--inset-sym-label", default=None)
    p.add_argument("--inset-asym-label", default=None)
    args = p.parse_args()
    plt = C.setup_style(args.usetex)

    df = C.load_run(args.tag, args.states_root)
    df = C.drop_duplicate_legs(
        df, C.duplicate_leg_keys(df, audit_json=args.audit, tag=args.tag))
    df = df[df.converged.astype(bool)].copy()
    df = df[np.isclose(df.amp, 0.0)]
    if df.empty:
        raise SystemExit(f"{args.tag} has no a=0 column -- this figure is the "
                         f"zero-rotation diagram; run the E13 a=0 family")
    if "C_L_fom" not in df:
        raise SystemExit(f"{args.tag} has no C_L_fom -- needs a FOM_COMPUTE=1 "
                         f"run; the full-order diagram cannot be drawn from "
                         f"reduced lift")

    ref = rec_zero(df, args)

    fig, ax = plt.subplots(figsize=C.figsize(args.width, args.aspect or 0.62))

    n_leg = 0
    for kind, leg in df.groupby("kind"):
        leg = leg.dropna(subset=["C_L_fom", "Re"]).sort_values("Re")
        if leg.empty:
            continue
        n_leg += 1
        col = C.branch_color(kind)
        sym = bool(leg.is_sym.iloc[0])
        if sym and ref is not None:
            # below the pitchfork the symmetric state is the stable solution,
            # above it the unstable spine -- a property of a supercritical
            # pitchfork, not something measured here
            lo = leg[leg.Re <= ref["Re_c"]]
            hi = leg[leg.Re >= ref["Re_c"]]
            ax.plot(lo.Re, lo.C_L_fom, "-", color=col, lw=1.4,
                    label="symmetric (stable)")
            if len(hi) > 1:
                ax.plot(hi.Re, hi.C_L_fom, "--", color=col, lw=1.1,
                        label="symmetric (unstable spine)")
        else:
            ax.plot(leg.Re, leg.C_L_fom, "-", color=col, lw=1.4,
                    label="symmetric" if sym else str(kind).split("@")[0])

    if ref is not None:
        ax.axvline(ref["Re_c"], color=C.P("C_SYM"), ls=":", lw=1.0, zorder=0)
        ax.annotate(rf"$\mathrm{{Re}}_c(0)={ref['Re_c']:.1f}$",
                    xy=(ref["Re_c"], 0.0), xycoords="data",
                    xytext=(6, 8), textcoords="offset points",
                    fontsize=7, color=C.P("C_SYM"))

    ax.axhline(0.0, color="0.85", lw=0.6, zorder=0)
    ax.set_xlabel(r"$\mathrm{Re}$")
    ax.set_ylabel(r"$C_L^{\mathrm{FOM}}$")
    ax.legend(loc="upper left", fontsize=7)

    got = 0
    got += add_inset(ax, args.inset_sym, [0.06, 0.06, 0.26, 0.26],
                     args.inset_sym_label or "symmetric")
    got += add_inset(ax, args.inset_asym, [0.68, 0.06, 0.26, 0.26],
                     args.inset_asym_label or "asymmetric")
    if got < 2:
        print("  [note] fewer than two field insets supplied: the draft caption "
              "promises 'representative velocity fields' and must be reworded, "
              "or render the fields and pass --inset-sym/--inset-asym")

    C.save(fig, args.outdir, "fig_fom_diagram", png=args.png)

    macros = {"NFomDiagramPoints": int(len(df)),
              "NFomDiagramLegs": int(n_leg)}
    if ref is not None:
        macros["RecZero"] = C.fmt(ref["Re_c"], ".1f")
        macros["RecZeroBracket"] = C.fmt(ref["bracket"], ".2g")
        macros["RecZeroKind"] = ref["kind"]
    cl = df.C_L_fom.abs()
    if cl.notna().any():
        macros["ClFomMax"] = C.fmt(float(cl.max()), ".3f")
        macros["ReMaxDiagram"] = C.fmt(float(df.Re.max()), ".0f")

    print(f"\n  {len(df)} converged a=0 points over {n_leg} legs; "
          f"max |C_L^FOM| = {cl.max():.4f} at Re <= {df.Re.max():.0f}")
    print("  [reminder] the draft still needs a literature comparison for "
          "Re_c(0): reference value, citation key, relative discrepancy and a "
          "one-clause attribution. That is not derivable from this run.")
    C.write_macros(os.path.join(args.outdir, "macros_fom_diagram.tex"), macros)


if __name__ == "__main__":
    main()
