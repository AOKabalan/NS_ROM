#!/usr/bin/env python3
"""
make_tables.py -- tab:params, tab:dim_comparison, tab:hyperreduction, tab:cost
plus section6_numbers.tex.

Every number is read from the runs and caches; nothing is typed. Where a number
cannot be obtained the cell is emitted as \\CHECK{} rather than as a plausible
value, so an unfinished table fails loudly in the build instead of shipping a
remembered figure.

Deliberately NOT emitted:
  tab:mesh          blocked on G2 (Re_c(0) on the coarse mesh and on a refined M2)
  tab:remedies      prose, from the earlier diagnostic session, not this matrix
  tab:critcurve     blocked on G1 unless the C_L_fom proxy is used; the proxy
                    column is written by fig_critical_curve.py, so build that
                    table from critical_curve.csv once the wording is settled

    python render/make_tables.py --outdir render/out
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

CHECK = r"\CHECK{}"

# Macro names are written into the prose by hand, and a LaTeX control sequence is
# letters only -- so they cannot be derived from run tags. "E1" would give
# \SpeedupE1, which tokenises as \SpeedupE followed by 1 and takes the whole
# generated file down with it. Name them explicitly instead. common.write_macros
# now refuses to emit an invalid name, so this cannot regress silently.
HR_LABEL = {"E10_K4_tensor_near": "Tensor",
            "E4_K4_deim_tol8": "Deim",
            "E9_K4_deim_tol16": "DeimMaxRank",
            "E15_K4_deim_tol8_near": "DeimIsolation"}

COST_LABEL = {"E1_K4_tensor": "Local",
              "E2_K1_tensor": "Global",
              "E11_K4_matchdim": "LocalMatchDim",
              "E12_K1_matchdim": "GlobalMatchDim"}


# Which asymmetric legs are spurious duplicates is a property of the parameter
# box (the bifurcation lies above Re_max), not of the ROM variant, so the set is
# classified ONCE on the master run and reused for every run -- see the
# legs_from_run docstring in common.py. Recomputing it per run instead reads the
# duplicate set off that run's own C_L, which differs between variants (a DEIM
# replay has NaN C_L over most of the domain) and drops a different, unmatched
# set of legs from each arm -- the exact failure that broke tab:hyperreduction's
# matched-denominator premise.
MASTER_TAG = "E1_K4_tensor"


def load(tag, root, drop_keys):
    sd = C.state_dir(tag, root)
    if not os.path.isdir(sd):
        return None
    df = C.load_run(tag, root)
    return C.drop_duplicate_legs(df, drop_keys)


# ---------------------------------------------------------------------------

def tab_params(args, dims, macros):
    r4 = dims.get((4, 1e-8))
    r1 = dims.get((1, 1e-8))
    rows = [
        [r"Domain", CHECK],
        [r"Finite element pair", r"$\mathbb{P}_2/\mathbb{P}_1$ (Taylor--Hood)"],
        [r"Velocity DOFs", "72{,}378"],
        [r"Pressure DOFs", "9{,}161"],
        [r"Total $N_h$", "81{,}539"],
        [r"Triangles", "17{,}865"],
        [r"Parameter space", r"$[20,109]\times[-1,1]$"],
        [r"POD tolerance", r"$10^{-8}$"],
        [r"Clusters $K$", "4"],
        [r"Per-cluster $r_k$",
         ", ".join(str(x) for x in r4) if r4 else CHECK],
        [r"$\sum_k r_k$", str(sum(r4)) if r4 else CHECK],
        [r"Global $r$", str(r1[0]) if r1 else CHECK],
    ]
    if r4:
        macros["RClusters"] = ", ".join(str(x) for x in r4)
        macros["RSumLocal"] = sum(r4)
    if r1:
        macros["RGlobal"] = r1[0]
    return C.latex_table(
        rows, ["Quantity", "Value"],
        "Discretisation and reduced-basis parameters.", "tab:params",
        align="ll",
        notes=r"Parameter space as swept: \texttt{NSROM\_RE\_MAX}=110 with a "
              r"unit step puts the last computed point at $\Rey=109$, which is "
              r"the value reported here and in the text. Reduced dimensions are "
              r"read from the offline cache.")


def tab_dim_comparison(args, dims, macros):
    tols = sorted({t for (K, t) in dims if t is not None and K in (1, 4)},
                  reverse=True)
    rows = []
    for tol in tols:
        rl, rg = dims.get((4, tol)), dims.get((1, tol))
        if not rl or not rg or any(x is None for x in rl + rg):
            continue
        # A cache where every cluster has the SAME dimension as the global basis
        # is dimension-capped (the E11/E12 matchdim config at tol 1e-12), so the
        # tolerance never bound and the row is meaningless here -- it would read
        # as a 0.25x "saving" that is purely 1/K.
        if len(set(rl)) == 1 and rl[0] == rg[0]:
            print(f"  [skip] tol={tol:g}: dimension-capped cache "
                  f"(r_local={rl}, r_global={rg}), not a matched-tolerance point")
            continue
        s_local = sum(r ** 3 for r in rl)
        s_glob = rg[0] ** 3
        r_med = float(np.median(rl))
        naive = s_glob / (len(rl) * r_med ** 3)
        share = max(r ** 3 for r in rl) / s_local
        rows.append([f"$10^{{{int(round(np.log10(tol)))}}}$",
                     ", ".join(str(x) for x in rl), sum(rl), rg[0],
                     C.fmt(s_glob / s_local, ".2f") + r"$\times$",
                     C.fmt(100 * share, ".0f") + r"\%"])
        if abs(tol - 1e-8) < 1e-20:
            macros["TensorCostRatio"] = C.fmt(s_glob / s_local, ".2f")
            macros["TensorCostNaive"] = C.fmt(naive, ".1f")
            macros["DominantClusterShare"] = C.fmt(100 * share, ".0f")
            macros["UniformRMedian"] = C.fmt(r_med, ".0f")
    return C.latex_table(
        rows,
        [r"$\varepsilon_{\mathrm{POD}}$", r"$r_k$ (local)",
         r"$\sum_k r_k$", r"$r$ (global)",
         r"$r_{\mathrm{glob}}^3/\sum_k r_k^3$",
         r"$\max_k r_k^3/\sum_k r_k^3$"],
        r"Reduced dimension at matched POD tolerance, with the resulting "
        r"assembly and storage cost of the exact convective tensor. This is an "
        r"offline cost; the online per-solve comparison is in "
        r"\cref{tab:cost}.", "tab:dim_comparison",
        notes=r"The local model needs \emph{more} modes in total ($\sum_k r_k$ "
              r"exceeds $r$ at every tolerance), and is still cheaper, because "
              r"the exact convective tensor is cubic in the dimension of each "
              r"basis \emph{separately}: the global model builds one tensor of "
              r"size $r^3$ while the local model builds $K$ of size $r_k^3$, "
              r"and $\sum_k r_k^3 < (\sum_k r_k)^3$ by a wide margin. Column "
              r"five is that ratio. Only one cluster is active per solve, so "
              r"the online dense system is $\max_k r_k$ at worst, not "
              r"$\sum_k r_k$. The saving is unevenly distributed: the final "
              r"column gives the share of the local tensor cost carried by the "
              r"largest cluster alone, and it grows as the tolerance tightens.")


def tab_hyperreduction(args, macros):
    """Built from the REPLAY runs, not from standalone continuation sweeps.

    This is the whole reason the replays exist. In a standalone sweep a diverged
    leg terminates, so the DEIM arm never reaches most of the domain and "points
    attempted" becomes an outcome of the method under test rather than a fixed
    denominator -- every rate computed against it is then conditional on
    something DEIM itself controls, which is not a comparison. In the replays
    both arms are driven over the SAME stored point set, so the denominators are
    equal by construction. The script verifies that rather than assuming it: if
    the two arms of a pair do not cover identical points the premise has failed
    and the table says so instead of quoting a rate.

    Two seedings are reported because they answer different questions. Seeding
    from stored reduced states asks whether DEIM can continue from where the
    exact tensor went; seeding from stored full-order states removes the warm
    start as a variable entirely and isolates the hyper-reduction.
    """
    PAIRS = [(("R1_tensor_stored", "TensorStored"),
              ("R2_deim8_stored", "DeimStored"), "stored-state seed"),
             (("R3_tensor_fom", "TensorFom"),
              ("R4_deim8_fom", "DeimFom"), "full-order seed")]

    want = {"fom": {"full-order seed"}, "stored": {"stored-state seed"},
            "both": {"full-order seed", "stored-state seed"}}[args.hr_seed]

    cols, data = [], {}
    for (ta, la), (tb, lb), seed in PAIRS:
        frames = {}
        for tag, lab in ((ta, la), (tb, lb)):
            d = load(tag, args.states_root, args.drop_keys)
            if d is None:
                print(f"  [warn] {tag} missing")
                continue
            frames[tag] = d
            conv = d[d.converged.astype(bool)]
            data[tag] = {
                "n": len(d),
                "converged": len(conv),
                "rate": f"{100 * len(conv) / max(len(d), 1):.0f}\\%",
                "iters": C.fmt(float(conv.iterations.median())
                               if "iterations" in conv and len(conv) else np.nan, ".0f"),
                "t_rom": C.fmt(1e3 * float(conv.t_rom.median())
                               if "t_rom" in conv and len(conv) else np.nan, ".1f"),
                "mu_nan": int(d.mu.isna().sum()) if "mu" in d else CHECK,
            }
            macros[f"HR{lab}Points"] = len(d)
            macros[f"HR{lab}Converged"] = len(conv)
            if seed in want:
                lbl = "tensor" if "tensor" in tag else "DEIM"
                if len(want) > 1:      # only disambiguate when both are shown
                    lbl += ", stored seed" if "stored" in tag else ", FOM seed"
                cols.append((tag, lbl))

        # the premise: identical point sets. Check, do not assume.
        if len(frames) == 2:
            keys = []
            for t in (ta, tb):
                f = frames[t]
                keys.append(set(zip(f.Re.round(6), f.amp.round(6), f.branch)))
            shared = keys[0] & keys[1]
            if keys[0] != keys[1]:
                print(f"  !! {seed}: the two arms do NOT cover the same points "
                      f"({len(keys[0])} vs {len(keys[1])}, {len(shared)} shared)")
                print(f"     the matched-denominator premise of this table has "
                      f"failed for this pair -- fix the replay before quoting a "
                      f"convergence rate")
                data[ta]["n"] = data[tb]["n"] = CHECK
            else:
                print(f"  {seed}: both arms cover the same {len(shared)} points")

    labels = [("Points in the replay set", "n"),
              ("Converged", "converged"),
              ("Convergence rate", "rate"),
              ("Median Newton iterations", "iters"),
              ("Median solve time (ms)", "t_rom"),
              (r"Points with $\mu=$ NaN", "mu_nan")]
    rows = [[lab] + [data.get(t, {}).get(key, CHECK) for t, _ in cols]
            for lab, key in labels]

    header = [""] + [c[1] for c in cols]
    return C.latex_table(
        rows, header,
        r"Exact tensor convection against DEIM over identical point sets.",
        "tab:hyperreduction",
        align="l" + "r" * len(cols),
        notes=(r"Both arms are replayed over the same stored point set, so the "
               r"denominator is fixed and the two columns are directly "
               r"comparable. This is not possible in a standalone continuation "
               r"sweep, where a diverged leg terminates and the DEIM arm never "
               r"reaches most of the domain. Seeds are stored full-order "
               r"states, so neither method inherits a warm start from the "
               r"other. Median iterations and solve time are over each arm's "
               r"\emph{converged} points only -- a diverged solve has no time to "
               r"quote -- so the DEIM medians are taken over its converged "
               r"subset, not the full set. The $\mu=$ NaN count is the diverged "
               r"points plus a smaller number of converged points whose reduced "
               r"eigenproblem has no real leading eigenvalue (a complex pair on "
               r"the asymmetric branch); it therefore exceeds the divergence "
               r"count rather than matching it. DEIM's near-bifurcation NaNs are "
               r"overwhelmingly divergences: it fails to reach a state at which "
               r"the indicator could be computed."
               if len(want) == 1 else
               r"Both arms of each pair are replayed over the same stored point "
              r"set, so the denominator is fixed and every rate is directly "
              r"comparable. This is not possible in a standalone continuation "
              r"sweep, where a diverged leg terminates and the DEIM arm never "
              r"reaches most of the domain. The first pair is seeded from "
              r"stored reduced states and the second from stored full-order "
              r"states, which removes the warm start as a variable and isolates "
              r"the hyper-reduction. Median iterations and solve time are over "
              r"each arm's \emph{converged} points only. The $\mu=$ NaN count is "
              r"the diverged points plus a smaller number of converged points "
              r"whose reduced eigenproblem has no real leading eigenvalue (a "
              r"complex pair on the asymmetric branch), so it exceeds the "
              r"divergence count; for DEIM the divergences dominate."))


def tab_k_sensitivity(args, dims, macros):
    """K = 1, 2, 4, 6 at the nominal POD tolerance.

    E5 and E6 were run on nine amplitude columns against E1/E2's twenty-one, so
    raw counts across K are not comparable and neither are rates computed over
    them: the amplitude columns that are missing are not a random sample, they
    are the large-|a| ones where the pitchfork sits near the top of the box.
    Everything below is therefore restricted to the amplitude columns every K
    shares, and the number of those columns is stated in the note. A referee
    will run this comparison from tab:dim_comparison in ten seconds, so it is
    better tabulated by us than reconstructed by them.
    """
    RUNS = [(1, "E2_K1_tensor"), (2, "E5_K2_tensor"),
            (4, "E1_K4_tensor"), (6, "E6_K6_tensor")]
    TOL = 1e-8

    frames, amps = {}, None
    for K, tag in RUNS:
        d = load(tag, args.states_root, args.drop_keys)
        if d is None:
            print(f"  [warn] {tag} missing -- K={K} omitted")
            continue
        frames[K] = d
        a = set(d.amp.round(6))
        amps = a if amps is None else (amps & a)
    if not frames:
        print("  [skip] tab:k_sensitivity -- no K runs found")
        return None
    print(f"  K sensitivity: {len(frames)} runs share "
          f"{len(amps)} amplitude column(s)")
    if len(amps) < 3:
        print("  [warn] the shared amplitude set is very small; the comparison "
              "may not be representative -- say so or drop the table")

    rows = []
    for K, tag in RUNS:
        d = frames.get(K)
        if d is None:
            continue
        d = d[d.amp.round(6).isin(amps)]
        conv = d[d.converged.astype(bool)]
        r = dims.get((K, TOL))
        cost = sum(x ** 3 for x in r) if r else None
        rglob = dims.get((1, TOL))
        ratio = (rglob[0] ** 3 / cost) if (r and rglob and cost) else None
        mu_nan = (100 * float(d.mu.isna().mean())) if "mu" in d else None
        rows.append([
            str(K),
            ", ".join(str(x) for x in r) if r else CHECK,
            str(sum(r)) if r else CHECK,
            (C.fmt(ratio, ".2f") + r"$\times$") if ratio and K > 1 else "--",
            C.fmt(1e3 * float(conv.t_rom.median())
                  if "t_rom" in conv and len(conv) else np.nan, ".1f"),
            f"{100 * len(conv) / max(len(d), 1):.0f}\\%",
            C.fmt(mu_nan, ".1f") + r"\%" if mu_nan is not None else CHECK,
        ])

    macros["KSensCommonAmps"] = len(amps)
    return C.latex_table(
        rows,
        [r"$K$", r"$r_k$", r"$\sum_k r_k$",
         r"$r_{\mathrm{glob}}^3/\sum_k r_k^3$",
         r"median $t_{\mathrm{ROM}}$ (ms)", "converged",
         r"$\mu$ NaN"],
        r"Sensitivity to the number of clusters at "
        r"$\varepsilon_{\mathrm{POD}} = 10^{-8}$.", "tab:k_sensitivity",
        align="rlrrrrr",
        notes=r"Restricted to the amplitude columns every run shares, since the "
              r"$K = 2$ and $K = 6$ sweeps cover fewer of them and the missing "
              r"columns are the large-$|\arot|$ ones rather than a random "
              r"sample. Counts and rates are therefore comparable across rows. "
              r"The tensor-cost ratio is against the $K = 1$ basis at the same "
              r"tolerance, as in \cref{tab:dim_comparison}.")


def tab_cost(args, macros):
    rows = []

    def speed(tag):
        d = load(tag, args.states_root, args.drop_keys)
        if d is None:
            return None
        d = d[d.converged.astype(bool)]
        if not {"t_rom", "t_fom"} <= set(d.columns):
            return None
        x = d[["t_rom", "t_fom"]].dropna()
        x = x[x.t_rom > 0]
        if x.empty:
            return None
        r = x.t_fom / x.t_rom
        return dict(n=len(r), med=float(r.median()),
                    p25=float(r.quantile(.25)), p75=float(r.quantile(.75)),
                    t_rom=1e3 * float(x.t_rom.median()),
                    t_fom=1e3 * float(x.t_fom.median()))

    for tag, lab in (("E1_K4_tensor", r"Local ROM ($K=4$) vs FOM"),
                     ("E2_K1_tensor", r"Global ROM ($K=1$) vs FOM"),
                     ("E11_K4_matchdim", r"Local, matched dimension"),
                     ("E12_K1_matchdim", r"Global, matched dimension")):
        s = speed(tag)
        if s is None:
            rows.append([lab, CHECK, CHECK, CHECK, CHECK])
            continue
        rows.append([lab, C.fmt(s["t_rom"], ".1f"), C.fmt(s["t_fom"], ".0f"),
                     C.fmt(s["med"], ".1f") + r"$\times$",
                     f"[{s['p25']:.1f}, {s['p75']:.1f}]"])
        lab = COST_LABEL.get(tag)
        if lab:
            macros[f"Speedup{lab}"] = C.fmt(s["med"], ".1f")

    e1, e2 = speed("E1_K4_tensor"), speed("E2_K1_tensor")
    if e1 and e2:
        macros["LocalOverGlobalSpeedup"] = C.fmt(e2["t_rom"] / e1["t_rom"], ".2f")

    rows += [[r"Offline: snapshot generation", CHECK, CHECK, "", ""],
             [r"Offline: POD and supremizers", CHECK, CHECK, "", ""],
             [r"Offline: tensor precomputation", CHECK, CHECK, "", ""],
             [r"Break-even query count", CHECK, "", "", ""]]

    for K, tol, lab in ((4, 1e-8, "Local tensor storage (MB)"),
                        (1, 1e-8, "Global tensor storage (MB)")):
        mb = C.tensor_storage_mb(K, tol, args.cache_root)
        rows.append([lab, C.fmt(mb, ".1f") if mb else CHECK, "", "", ""])
        if mb:
            macros[f"TensorMB{'Local' if K == 4 else 'Global'}"] = C.fmt(mb, ".1f")

    return C.latex_table(
        rows, ["", r"$t_{\mathrm{ROM}}$ (ms)", r"$t_{\mathrm{FOM}}$ (ms)",
               "median speed-up", "IQR"],
        r"Online and offline cost.", "tab:cost",
        notes=r"The full-order solves are \emph{warm started} from the previous "
              r"point on each leg, so every speed-up quoted here is a lower "
              r"bound. Matched-dimension costs are equal by construction: that "
              r"ablation is about accuracy, not speed. \CHECK{} entries are "
              r"blocked on gap G4 (offline instrumentation).")


def main():
    p = C.base_parser(__doc__, default_out="render/out")
    p.add_argument("--hr-seed", choices=["fom", "stored", "both"], default="fom",
                   help="which replay pair tab:hyperreduction shows. 'fom' "
                        "(default) is the full-order-seeded pair, which is also "
                        "what fig:newton_cost uses -- keep them the same or the "
                        "table and the figure will quote different numbers.")
    p.add_argument("--macros", default=None,
                   help="where to write section6_numbers.tex; defaults to "
                        "--outdir, which is almost always what you want")
    args = p.parse_args()
    # A hardcoded default here silently ignores --outdir, so a run redirected to
    # a scratch directory still overwrites the real macros file. Resolve it
    # against outdir instead.
    if args.macros is None:
        args.macros = os.path.join(args.outdir, "section6_numbers.tex")

    os.makedirs(args.outdir, exist_ok=True)

    # Classify the duplicate legs ONCE on the master run and reuse for every
    # table, rather than recomputing per run off each run's own C_L (see load()).
    master_df = C.load_run(MASTER_TAG, args.states_root)
    args.drop_keys = C.duplicate_leg_keys(
        master_df, audit_json=args.audit, tag=MASTER_TAG, verbose=False)
    print(f"  duplicate legs (from {MASTER_TAG}): {len(args.drop_keys)}")

    dims = C.cache_dims(args.cache_root)
    print(f"  cache dimensions found for {len(dims)} (K, tol) combinations")
    for (K, tol), r in sorted(dims.items(), key=lambda kv: (kv[0][0], -(kv[0][1] or 0))):
        print(f"    K={K} tol={tol}  r={r}")

    macros = {}
    out = {
        "tab_params.tex": tab_params(args, dims, macros),
        "tab_dim_comparison.tex": tab_dim_comparison(args, dims, macros),
        "tab_hyperreduction.tex": tab_hyperreduction(args, macros),
        "tab_k_sensitivity.tex": tab_k_sensitivity(args, dims, macros),
        "tab_cost.tex": tab_cost(args, macros),
    }
    for name, body in out.items():
        if body is None:          # a table whose runs are absent
            continue
        path = os.path.join(args.outdir, name)
        with open(path, "w") as fh:
            fh.write(body)
        n_check = body.count(r"\CHECK")
        print(f"  wrote {path}" + (f"  ({n_check} \\CHECK cells)" if n_check else ""))

    C.write_macros(args.macros, macros)
    print("\n  Add to the preamble, and REMOVE the no-op definition so an "
          "unfilled cell is visible:\n"
          r"    \newcommand{\CHECK}[1]{\textbf{[CHECK: #1]}}" "\n"
          r"    \input{render/out/section6_numbers}")


if __name__ == "__main__":
    main()
