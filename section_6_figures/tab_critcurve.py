#!/usr/bin/env python3
"""
tab_critcurve.py -- tab:critcurve

make_tables.py deliberately does not emit this table: it is blocked on G1 unless
the C_L^FOM proxy is used, and the proxy column is written by
fig_critical_curve.py. So this script reads that script's output rather than
recomputing anything, which also guarantees the table and fig:critical_curve
cannot disagree.

    python section_6_figures/fig_critical_curve.py --which both
    python section_6_figures/tab_critcurve.py

WHAT THIS TABLE VALIDATES AGAINST.
The draft says "full-order bisection". There is no bisection driver -- that is
gap G2, and it needs code that does not exist. What this table compares is the
reduced indicator's zero crossing against full-order lift departure. Those are
different objects and the caption says so. Do not let the word "bisection"
survive into the submitted caption unless someone writes the bisector.

THE ADJUDICATED CASE.
At some amplitudes the lift proxy does not locate the pitchfork at all. It is
principled only at a = 0, where the symmetric branch carries no lift; away from
zero the rotating cylinders generate lift on the symmetric branch too, so
departure from zero is not by itself a bifurcation. Where the disagreement
exceeds a few brackets the row is marked and excluded from the summary
statistic, and the exclusion is stated in the table note rather than performed
quietly. That marked row is the "one adjudicated near-degenerate case" the draft
asks for -- it just is not the reassuring kind.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

CHECK = r"\CHECK{}"


def pick_amps(d, n, always):
    """A spread of n amplitudes that carry both numbers, plus every flagged row
    -- an outlier must not be droppable by the sampling."""
    have = d[d.Re_c.notna() & d.Re_c_proxy.notna()]
    if have.empty:
        return d.iloc[0:0]
    idx = np.linspace(0, len(have) - 1, min(n, len(have))).astype(int)
    sel = set(have.iloc[idx].amp.round(6))
    sel |= set(np.round(always, 6))
    return d[d.amp.round(6).isin(sel)].sort_values("amp")


def main():
    p = C.base_parser(__doc__)
    p.add_argument("--csv", default="section_6_figures/out/critical_curve.csv")
    p.add_argument("--n-amps", type=int, default=5)
    p.add_argument("--flag-brackets", type=float, default=3.0,
                   help="disagreement above this many brackets marks the row "
                        "as one the proxy does not resolve")
    args = p.parse_args()

    if not os.path.exists(args.csv):
        raise SystemExit(
            f"{args.csv} not found -- run fig_critical_curve.py first; this "
            f"table is built from its output on purpose so the two cannot "
            f"disagree")
    d = pd.read_csv(args.csv)
    for col in ("amp", "Re_c", "bracket", "Re_c_proxy"):
        if col not in d.columns:
            raise SystemExit(f"{args.csv} has no '{col}' column -- regenerate "
                             f"it with the current fig_critical_curve.py")
    d = d.sort_values("amp").reset_index(drop=True)

    d["diff"] = (d.Re_c - d.Re_c_proxy).abs()
    br = float(np.nanmedian(d.bracket)) if d.bracket.notna().any() else np.nan
    thr = args.flag_brackets * (br if np.isfinite(br) else 1.0)
    d["flag"] = d["diff"] > thr

    flagged = d[d.flag]
    sel = pick_amps(d, args.n_amps, flagged.amp.values)

    print(f"  {len(d)} amplitudes in {args.csv}; "
          f"{int(d.Re_c.notna().sum())} with a mu crossing, "
          f"{int(d.Re_c_proxy.notna().sum())} with a lift proxy")
    print(f"  median bracket = {C.fmt(br, '.2g')}; flagging |diff| > "
          f"{thr:.2g} ({args.flag_brackets:g} brackets)")
    if len(flagged):
        print(f"  !! the lift proxy does not resolve {len(flagged)} "
              f"amplitude(s):")
        for _, r in flagged.iterrows():
            print(f"       a={r.amp:+.2f}: mu gives {r.Re_c:.2f}, proxy gives "
                  f"{r.Re_c_proxy:.2f}  (diff {r['diff']:.2f})")
        print("     these are excluded from the summary statistic and marked "
              "in the table; the note says so")

    rows = []
    for _, r in sel.iterrows():
        mark = r"$^{\dagger}$" if r.flag else ""
        rows.append([
            C.fmt(float(r.amp), "+.2f"),
            C.fmt(float(r.Re_c), ".2f") if np.isfinite(r.Re_c) else CHECK,
            C.fmt(float(r.bracket), ".2g") if np.isfinite(r.bracket) else CHECK,
            (C.fmt(float(r.Re_c_proxy), ".2f") + mark)
            if np.isfinite(r.Re_c_proxy) else CHECK,
            C.fmt(float(r["diff"]), ".2f") if np.isfinite(r["diff"]) else CHECK,
        ])

    clean = d[~d.flag & d["diff"].notna()]
    macros = {}
    if len(clean):
        macros["CritCurveMedianDiff"] = C.fmt(float(clean["diff"].median()), ".2f")
        macros["CritCurveMaxDiff"] = C.fmt(float(clean["diff"].max()), ".2f")
        macros["CritCurveNCompared"] = int(len(clean))
        print(f"\n  agreement over the {len(clean)} unflagged amplitudes: "
              f"median {clean['diff'].median():.2f}, max "
              f"{clean['diff'].max():.2f}, against a bracket of "
              f"{C.fmt(br, '.2g')}")
        if clean["diff"].median() <= (br if np.isfinite(br) else np.inf):
            print("  -> the reduced tracer resolves Re_c to within its own "
                  "continuation step; that is the claim the draft makes")
        else:
            print("  -> the median disagreement EXCEEDS one bracket; the draft "
                  "sentence 'comparable to the reported bracket widths' does "
                  "not hold as written")
    if np.isfinite(br):
        macros["CritCurveBracket"] = C.fmt(br, ".2g")
    macros["CritCurveNAmps"] = int(len(sel))
    macros["CritCurveNFlagged"] = int(len(flagged))
    if len(flagged):
        w = flagged.loc[flagged["diff"].idxmax()]
        macros["CritCurveOutlierAmp"] = C.fmt(float(w.amp), "+.2f")
        macros["CritCurveOutlierDiff"] = C.fmt(float(w["diff"]), ".2f")

    note = (r"$\Rec(\mu)$ is the zero crossing of the reduced stability "
            r"indicator, bracketed by one continuation step. The reference "
            r"column is full-order \emph{lift departure}, not bisection: the "
            r"first $\Rey$ at which an asymmetric leg's $|\CL^{\mathrm{FOM}}|$ "
            r"clears a fixed fraction of its own scale. ")
    if len(flagged):
        note += (r"Rows marked $\dagger$ are amplitudes at which the lift "
                 r"proxy does not locate the pitchfork -- away from "
                 r"$\arot = 0$ the rotating cylinders generate lift on the "
                 r"symmetric branch, so departure from zero is not by itself a "
                 r"bifurcation. They are shown for completeness and excluded "
                 r"from the quoted agreement. ")
    note += (r"A full-order eigensweep (and a bisection driver) would replace "
             r"this column outright; see \cref{sec:conclusions}.")

    body = C.latex_table(
        rows,
        [r"$\arot$", r"$\Rec(\mu)$", "bracket",
         r"$\Rec(\CL^{\mathrm{FOM}})$", r"$|\Delta|$"],
        r"Reduced critical Reynolds number against the full-order lift "
        r"departure at selected amplitudes.", "tab:critcurve",
        align="lrrrr", notes=note)

    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, "tab_critcurve.tex")
    with open(path, "w") as fh:
        fh.write(body)
    n_check = body.count(r"\CHECK")
    print(f"\n  wrote {path}" + (f"  ({n_check} \\CHECK cells)" if n_check else ""))
    C.write_macros(os.path.join(args.outdir, "macros_critcurve.tex"), macros)


if __name__ == "__main__":
    main()
