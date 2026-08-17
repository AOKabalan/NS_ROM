#!/usr/bin/env python3
"""
compute_point_errors.py -- per-point field errors, computed once, written to CSV.

Computation only: no plotting. Figure scripts read the CSV this produces, so a
label tweak never re-reads a gigabyte of stored fields.

What it does, per stored point that has a FOM field:

  err_p_M      relative pressure error in the M_p norm, with the gauge constant
               removed in a mass-weighted sense (NOT the nodal mean -- with a
               5.5x graded mesh those differ, and they differ most where the
               pressure structure is).

  err_u_recomp relative velocity error in the M_u norm, recomputed from the
               *stored* fields. The run already recorded err_u from the live
               float64 Functions; comparing the two measures how much the
               float32 field storage costs. That number decides whether the
               pressure panel needs a stated noise floor.

  err_p_nodal  the naive np.linalg.norm/np.mean version, for reference only, so
               the size of the convention error is visible rather than assumed.

Source of truth is index.jsonl (via StateStore), never sweep_results.csv: the
index carries idx, cluster, iterations and err_u, and the CSV has no idx column
at all, so any CSV join relies on row order matching write order.

Usage
-----
    python compute_point_errors.py \
        --state-dir states/E1_K4_tensor \
        --local-rom local_rom/K4_H1_tol1e-08 \
        --mass-p mass/M_p.npz --mass-u mass/M_u.npz \
        --out paper_data/point_errors_E1.csv

The mass matrices must be the SAME ones main_local.py passes as M_p / M_u --
dump them once from the driver (see --help-mass) rather than rebuilding, so the
norm in the paper is the norm the code used.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.append("scripts")
from nsrom.io.state_store import StateStore  # noqa: E402


HELP_MASS = """
To dump the mass matrices from main_local.py, next to where M_u/M_p are built:

    import scipy.sparse as sp
    from scipy.sparse import save_npz
    def _dump(M, path):
        h = M.M.handle if hasattr(M, "M") else M          # Firedrake -> PETSc
        ai, aj, av = h.getValuesCSR()
        save_npz(path, sp.csr_matrix((av, aj, ai), shape=h.getSize()))
    os.makedirs("mass", exist_ok=True)
    _dump(M_u, "mass/M_u.npz"); _dump(M_p, "mass/M_p.npz")

Do this once. Rebuilding the matrices here would risk quoting a norm that is
not the norm the errors were computed in.
"""


def load_mass(path):
    """Load a mass matrix as anything supporting M @ v."""
    if path is None:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} (see --help-mass)")
    try:
        import scipy.sparse as sp
        M = sp.load_npz(path)
        return M.tocsr()
    except Exception:
        pass
    with np.load(path) as d:
        keys = list(d.files)
        if len(keys) == 1:
            return d[keys[0]]
        for k in ("M", "mass", "A", "arr_0"):
            if k in d:
                return d[k]
        raise ValueError(f"{path}: ambiguous keys {keys}; expected one array")


def mnorm(v, M):
    if M is None:
        return float(np.linalg.norm(v))
    return float(np.sqrt(max(float(v @ (M @ v)), 0.0)))


def mass_gauge(p, M):
    """Remove the constant in the M-inner-product sense: p - <p,1>/<1,1>."""
    one = np.ones_like(p)
    if M is None:
        return p - p.mean()
    Mone = M @ one
    return p - (float(p @ Mone) / float(one @ Mone))

def load_bases(local_rom_dir, cluster):
    """Load the reduced velocity and pressure bases."""
    cdir = os.path.join(
        local_rom_dir,
        f"cluster_{cluster}",
        "operators",
    )

    path = os.path.join(cdir, "operators.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with np.load(path) as d:
        return d["Z_u"], d["Z_p"]



def gauge_columns(V, M):
    """
    Gauge-remove every basis column. The gauge map is linear in the field, so
    gauge(V @ c) == gauge_columns(V) @ c -- reconstruction is unchanged. But the
    projection error MUST be measured against the gauged basis: subtracting the
    constant from p_fom moves it out of span(V) (the constant vector is not in
    the basis), which would make the "floor" exceed the reconstruction error.
    """
    if M is None:
        return V - V.mean(axis=0, keepdims=True)
    one = np.ones(V.shape[0])
    Mone = M @ one
    denom = float(one @ Mone)
    return V - np.outer(one, (V.T @ Mone) / denom)


def projection_error(field, V, M):
    """
    Best possible error in span(V): ||f - V (V'MV)^-1 V'M f||_M / ||f||_M.
    A reconstruction error BELOW this is impossible, so it is the check that
    the reconstruction convention is right -- and it is also the POD projection
    error curve that fig:err_vs_N wants.
    """
    MV = M @ V if M is not None else V
    G = V.T @ MV
    c = np.linalg.solve(G, MV.T @ field if M is not None else V.T @ field)
    return mnorm(field - V @ c, M) / mnorm(field, M), c


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--state-dir", default="states/E1_K4_tensor")
    p.add_argument("--local-rom", default="local_rom/K4_H1_tol1e-08")
    p.add_argument("--mass-p", default="mass/M_p_L2.npz")
    p.add_argument("--mass-u", default="mass/M_u_H1.npz")
    p.add_argument("--out", default="paper_data/point_errors_E1.csv")
    p.add_argument("--limit", type=int, default=None, help="first N points (smoke test)")
    p.add_argument("--calibrate", type=int, default=12,
                   help="points used to identify the velocity block order")
    p.add_argument("--tol-match", type=float, default=0.05,
                   help="max relative discrepancy against the recorded err_u "
                        "for a reconstruction to be accepted")
    p.add_argument("--help-mass", action="store_true")
    args = p.parse_args(argv)

    if args.help_mass:
        print(HELP_MASS)
        return 0

    M_p = load_mass(args.mass_p)
    M_u = load_mass(args.mass_u)

    store = StateStore(args.state_dir)
    store.summary()
    pts = store.filter(has_fom_field=True, converged=True)
    if args.limit:
        pts = pts[:args.limit]
    print(f"  {len(pts)} points with stored FOM fields")
    bases, vbasis, pbasis = {}, {}, {}

    def get_bases(k):
        if k not in bases:
            bases[k] = load_bases(args.local_rom, k)
        return bases[k]

    # ---- main pass ---------------------------------------------------------
    rows = []
    for n, pt in enumerate(pts):
        rec = pt.record
        row = {k: rec.get(k) for k in
               ("idx", "Re", "amp", "branch", "cluster", "iterations", "mode",
                "t_rom", "t_fom", "mu", "C_L_rom", "C_L_fom", "fom_iters")}
        row["err_u_recorded"] = rec.get("err_u")
        k = rec["cluster"]

        p_fom = pt.get("p_fom")
        p_out = pt.get("p_out")
        if p_fom is not None and p_out is not None:
            p_fom = np.asarray(p_fom, dtype=np.float64)
            p_out = np.asarray(p_out, dtype=np.float64)
            key = (k, p_out.size)

            if key not in pbasis:
                _, Z_p = get_bases(k)
                V_p = Z_p[:, :p_out.size]
                pbasis[key] = (V_p, gauge_columns(V_p, M_p))

            V_p, V_pg = pbasis[key]
            pf = mass_gauge(p_fom, M_p)
            den = mnorm(pf, M_p)
            row["err_p_M"] = (mnorm(pf - V_pg @ p_out, M_p) / den
                              if den > 0 else np.nan)
            # impossible-to-beat floor, in the same gauge-quotient space;
            # also the pressure projection-error curve fig:err_vs_N wants
            row["proj_err_p"], _ = projection_error(pf, V_pg, M_p)
            row["pressure_basis"] = "Z_p"
            pn, pnr = p_fom - p_fom.mean(), (V_p @ p_out) - (V_p @ p_out).mean()
            dn = np.linalg.norm(pn)
            row["err_p_nodal"] = float(np.linalg.norm(pn - pnr) / dn) if dn else np.nan

        u_fom, u_out = pt.get("u_fom"), pt.get("u_out")
        if u_fom is not None and u_out is not None:
            u_fom = np.asarray(u_fom, dtype=np.float64)
            u_out = np.asarray(u_out, dtype=np.float64)
            key = (k, u_out.size)

        if key not in vbasis:
            Z_u, _ = get_bases(k)
            vbasis[key] = Z_u[:, :u_out.size]

        V_u = vbasis[key]

        den = mnorm(u_fom, M_u)
        row["err_u_recomp"] = (
            mnorm(u_fom - V_u @ u_out, M_u) / den
            if den > 0 else np.nan
        )
        row["proj_err_u"], _ = projection_error(u_fom, V_u, M_u)

        rows.append(row)
        if (n + 1) % 500 == 0:
            print(f"    {n + 1}/{len(pts)}")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\n  wrote {args.out}  ({len(df)} rows)")

    for col in ("err_p_M", "proj_err_p", "err_p_nodal", "err_u_recomp"):
        if col in df and df[col].notna().any():
            s = df[col].dropna()
            print(f"  {col:14s} median={s.median():.3e}  max={s.max():.3e}")

    # a reconstruction error below the projection error is impossible
    if {"err_p_M", "proj_err_p"} <= set(df.columns):
        d = df[["err_p_M", "proj_err_p"]].dropna()
        bad = d[d.err_p_M < d.proj_err_p * 0.999]
        if len(bad):
            print(f"\n  !! {len(bad)} of {len(d)} points have err_p_M BELOW the "
                  f"projection error.\n     That is impossible, so the pressure "
                  f"reconstruction convention is wrong -- most likely a\n     "
                  f"pressure lifting term, exactly as for velocity. Do not use "
                  f"these numbers.")
        else:
            r = (d.err_p_M / d.proj_err_p).replace([np.inf, -np.inf], np.nan).dropna()
            print(f"\n  err_p_M / projection error: median={r.median():.2f} "
                  f"(>=1 always, near 1 means the ROM is at its basis limit)")



    if "err_p_M" not in df.columns or df.err_p_M.isna().all():
        print("\n  No pressure errors were produced.")
    print("""
    NSROM_REPLAY_REF=states/E1_K4_tensor NSROM_REPLAY_SEED=stored \\
    NSROM_MODE=tensor NSROM_K=4 NSROM_POD_TOL=1e-8 \\
    NSROM_RUN_TAG=R5_tensor_errp NSROM_STATE_DIR=states/R5_tensor_errp \\
        python scripts/main_local.py
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
