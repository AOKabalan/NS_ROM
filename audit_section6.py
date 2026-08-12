#!/usr/bin/env python3
"""
audit_section6.py -- inventory every number Section 6 needs, straight from disk.

Deliberately dependency-free apart from numpy/pandas: it must never fail on an
import so that it can be run before anything else. It computes nothing that
belongs in a figure; it only reports what exists and what the derived numbers
actually are, so the paper text and the figure scripts can be written against
measured values instead of remembered ones.

Usage (from the repo root):

    python audit_section6.py
    python audit_section6.py --repo-root . --out paper_data/audit.json
    python audit_section6.py --states E1_K4_tensor E2_K1_tensor   # subset

Writes:
    paper_data/audit.json        machine-readable, the input to figure scripts
    paper_data/audit_report.txt  same content, human-readable
and prints the report to stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import zipfile
from collections import Counter, OrderedDict

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

BRANCH_RE = re.compile(r"^(?P<kind>[^@]+)@amp(?P<amp>[-+]?\d*\.?\d+)\s*$")
CACHE_RE = re.compile(r"^K(?P<K>\d+)_H1(?:_tol(?P<tol>.+))?$")


def jsonable(obj):
    """numpy -> plain python, recursively, so json.dump never chokes."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        return None if not np.isfinite(f) else f
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def q(series, p):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(np.percentile(s, p)) if len(s) else None


def med(series):
    return q(series, 50)


def parse_branch(name):
    m = BRANCH_RE.match(str(name))
    if not m:
        return {"kind": str(name), "amp": None}
    return {"kind": m.group("kind"), "amp": float(m.group("amp"))}


def npy_shape(path):
    try:
        a = np.load(path, mmap_mode="r", allow_pickle=False)
        return {"shape": list(a.shape), "dtype": str(a.dtype)}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


def npz_shapes(path):
    """Read .npy headers inside a .npz without decompressing the payloads."""
    out = {}
    try:
        with zipfile.ZipFile(path) as z:
            for name in z.namelist():
                key = name[:-4] if name.endswith(".npy") else name
                try:
                    with z.open(name) as fh:
                        version = np.lib.format.read_magic(fh)
                        shape, _fortran, dtype = np.lib.format._read_array_header(fh, version)
                    out[key] = {"shape": list(shape), "dtype": str(dtype)}
                except Exception:  # noqa: BLE001
                    # fall back to a real load for this member only
                    try:
                        with np.load(path, allow_pickle=False) as f:
                            arr = f[key]
                            out[key] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
                    except Exception as exc:  # noqa: BLE001
                        out[key] = {"error": f"{type(exc).__name__}: {exc}"}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}
    return out


def dir_inventory(path, max_entries=40):
    """Names + sizes only. Used where the on-disk layout is not yet known."""
    if not os.path.isdir(path):
        return None
    entries = []
    for name in sorted(os.listdir(path))[:max_entries]:
        full = os.path.join(path, name)
        rec = {"name": name, "is_dir": os.path.isdir(full)}
        if os.path.isfile(full):
            rec["size_mb"] = round(os.path.getsize(full) / 1e6, 4)
            if name.endswith(".npy"):
                rec.update(npy_shape(full))
            elif name.endswith(".npz"):
                rec["members"] = npz_shapes(full)
        entries.append(rec)
    total = len(os.listdir(path))
    return {"n_entries": total, "truncated": total > max_entries, "entries": entries}


# ---------------------------------------------------------------------------
# A. sweep runs
# ---------------------------------------------------------------------------

def audit_state(state_dir):
    tag = os.path.basename(state_dir.rstrip("/"))
    rec = {"tag": tag, "path": state_dir}

    meta_path = os.path.join(state_dir, "run_meta.json")
    if os.path.isfile(meta_path):
        try:
            with open(meta_path) as fh:
                rec["run_meta"] = json.load(fh)
        except Exception as exc:  # noqa: BLE001
            rec["run_meta_error"] = str(exc)

    csv_path = os.path.join(state_dir, "sweep_results.csv")
    if not os.path.isfile(csv_path):
        rec["error"] = "no sweep_results.csv"
        return rec

    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:  # noqa: BLE001
        rec["error"] = f"read_csv failed: {type(exc).__name__}: {exc}"
        return rec

    rec["n_rows"] = int(len(df))
    rec["columns"] = list(df.columns)

    idx = os.path.join(state_dir, "index.jsonl")
    if os.path.isfile(idx):
        with open(idx) as fh:
            rec["n_index_lines"] = sum(1 for _ in fh)
    pts = os.path.join(state_dir, "points")
    if os.path.isdir(pts):
        rec["n_point_files"] = len(os.listdir(pts))

    # --- parameter coverage
    if "Re" in df:
        Re = pd.to_numeric(df.Re, errors="coerce")
        rec["Re"] = {"min": float(Re.min()), "max": float(Re.max()),
                     "n_unique": int(Re.nunique())}
        u = np.sort(Re.dropna().unique())
        if len(u) > 1:
            rec["Re"]["step_mode"] = float(pd.Series(np.diff(u)).round(6).mode().iloc[0])
    if "amp" in df:
        amps = np.sort(pd.to_numeric(df.amp, errors="coerce").dropna().unique())
        rec["amps"] = {"n": int(len(amps)), "values": [round(float(a), 4) for a in amps]}

    # --- convergence / FOM presence
    for col in ("fom_converged", "converged"):
        if col in df:
            v = df[col]
            try:
                v = v.astype(bool)
            except Exception:  # noqa: BLE001
                v = pd.to_numeric(v, errors="coerce").fillna(0).astype(bool)
            rec[f"n_{col}_true"] = int(v.sum())
    if "C_L_fom" in df:
        rec["n_C_L_fom_present"] = int(pd.to_numeric(df.C_L_fom, errors="coerce").notna().sum())

    # --- mu availability: the DEIM claim rests on this
    if "mu" in df:
        mu = pd.to_numeric(df.mu, errors="coerce")
        rec["mu"] = {"n_finite": int(np.isfinite(mu).sum()),
                     "n_nan": int(mu.isna().sum()),
                     "min": q(mu, 0), "max": q(mu, 100)}

    # --- timings
    timings = {}
    for col in ("t_rom", "t_fom"):
        if col in df:
            timings[col] = {"median": med(df[col]), "p25": q(df[col], 25),
                            "p75": q(df[col], 75), "n": int(pd.to_numeric(df[col], errors="coerce").notna().sum())}
    if "t_rom" in df and "t_fom" in df:
        both = df[["t_rom", "t_fom"]].apply(pd.to_numeric, errors="coerce").dropna()
        both = both[both.t_rom > 0]
        if len(both):
            ratio = both.t_fom / both.t_rom
            timings["speedup_per_solve"] = {
                "median": float(ratio.median()), "p25": float(ratio.quantile(.25)),
                "p75": float(ratio.quantile(.75)), "n": int(len(ratio))}
    for col in ("newton_iters", "iters", "fom_iters"):
        if col in df:
            timings[col] = {"median": med(df[col]), "max": q(df[col], 100)}
    rec["timings"] = timings

    # --- per-leg breakdown
    if "branch" in df:
        legs = []
        for name, s in df.groupby("branch"):
            info = parse_branch(name)
            leg = {"branch": str(name), "kind": info["kind"], "amp": info["amp"],
                   "n": int(len(s))}
            if "Re" in s:
                Re = pd.to_numeric(s.Re, errors="coerce")
                leg["Re_min"] = float(Re.min())
                leg["Re_max"] = float(Re.max())
            for col in ("C_L", "C_L_fom"):
                if col in s:
                    v = pd.to_numeric(s[col], errors="coerce").abs()
                    leg[f"abs_{col}_max"] = float(v.max()) if v.notna().any() else None
                    leg[f"abs_{col}_median"] = float(v.median()) if v.notna().any() else None
            legs.append(leg)
        rec["legs"] = sorted(legs, key=lambda d: (d["kind"], d["amp"] if d["amp"] is not None else 0))
        rec["leg_kind_counts"] = dict(Counter(l["kind"] for l in rec["legs"]))

    return rec, df


# ---------------------------------------------------------------------------
# B. duplicate-leg detection (the |a| >= 0.8 relabelling question)
# ---------------------------------------------------------------------------

def duplicate_legs(df, abs_tol=1e-3, rel_tol=1e-2):
    """
    Two independent tests per asymmetric leg:
      (1) max |C_L| over the leg  -- the 'flat leg' test
      (2) max |C_L_asym - C_L_sym| over shared Re, normalised by the global
          |C_L| scale -- the 'same solution' test
    A leg is called a duplicate only where both agree; disagreements are
    reported so the threshold can be chosen from data.
    """
    if not {"branch", "Re", "C_L"} <= set(df.columns):
        return {"error": "needs branch, Re, C_L"}

    d = df.copy()
    d["_p"] = d.branch.map(parse_branch)
    d["_kind"] = d._p.map(lambda x: x["kind"])
    d["_amp"] = d._p.map(lambda x: x["amp"])
    d["C_L"] = pd.to_numeric(d.C_L, errors="coerce")
    d["Re"] = pd.to_numeric(d.Re, errors="coerce")

    scale = float(d.C_L.abs().max())
    out = {"C_L_scale": scale, "abs_tol": abs_tol, "rel_tol": rel_tol, "legs": []}

    sym_kinds = [k for k in d._kind.unique() if str(k).startswith("sym")]
    for amp, block in d.groupby("_amp"):
        sym = block[block._kind.isin(sym_kinds)]
        sym_map = (sym.dropna(subset=["Re"]).groupby("Re").C_L.mean()
                   if len(sym) else pd.Series(dtype=float))
        for kind, leg in block[~block._kind.isin(sym_kinds)].groupby("_kind"):
            leg = leg.dropna(subset=["Re"]).sort_values("Re")
            flat = float(leg.C_L.abs().max()) if leg.C_L.notna().any() else None
            shared = leg[leg.Re.isin(sym_map.index)]
            dist = None
            if len(shared):
                dist = float((shared.C_L.values - sym_map.reindex(shared.Re).values).__abs__().max())
            rec = {
                "amp": amp, "kind": kind, "n": int(len(leg)),
                "Re_min": float(leg.Re.min()) if len(leg) else None,
                "abs_C_L_max": flat,
                "n_shared_Re_with_sym": int(len(shared)),
                "max_dist_to_sym": dist,
                "flat_test_duplicate": (flat is not None and flat < abs_tol),
                "dist_test_duplicate": (dist is not None and dist < rel_tol * scale),
            }
            rec["tests_agree"] = rec["flat_test_duplicate"] == rec["dist_test_duplicate"]
            out["legs"].append(rec)

    out["duplicates_both_tests"] = [
        {"amp": l["amp"], "kind": l["kind"]}
        for l in out["legs"] if l["flat_test_duplicate"] and l["dist_test_duplicate"]]
    out["disagreements"] = [l for l in out["legs"] if not l["tests_agree"]]
    return out


# ---------------------------------------------------------------------------
# C. mu zero crossings and the C_L_fom proxy (critical curve, Re_c)
# ---------------------------------------------------------------------------

def crossings(df, cl_departure_frac=0.02):
    if not {"branch", "Re", "mu"} <= set(df.columns):
        return {"error": "needs branch, Re, mu"}

    d = df.copy()
    d["_p"] = d.branch.map(parse_branch)
    d["_kind"] = d._p.map(lambda x: x["kind"])
    d["_amp"] = d._p.map(lambda x: x["amp"])
    d["Re"] = pd.to_numeric(d.Re, errors="coerce")
    d["mu"] = pd.to_numeric(d.mu, errors="coerce")
    has_fom = "C_L_fom" in d.columns
    if has_fom:
        d["C_L_fom"] = pd.to_numeric(d.C_L_fom, errors="coerce")
        fom_scale = float(d.C_L_fom.abs().max())

    res = []
    for amp, block in d.groupby("_amp"):
        sym = block[block._kind.astype(str).str.startswith("sym")]
        sym = sym.dropna(subset=["Re"]).sort_values("Re")
        rec = {"amp": amp, "n_sym_points": int(len(sym))}

        s = sym.dropna(subset=["mu"])
        rec["n_mu_finite"] = int(len(s))
        if len(s) >= 2:
            mu = s.mu.values
            Re = s.Re.values
            sgn = np.sign(mu)
            flips = np.where(np.diff(sgn) != 0)[0]
            flips = [i for i in flips if sgn[i] != 0]
            rec["n_sign_changes"] = int(len(flips))
            if flips:
                i = flips[0]
                lo, hi = float(Re[i]), float(Re[i + 1])
                mlo, mhi = float(mu[i]), float(mu[i + 1])
                rec["mu_bracket"] = [lo, hi]
                rec["bracket_width"] = hi - lo
                rec["Re_c_mu_interp"] = (lo - mlo * (hi - lo) / (mhi - mlo)
                                         if mhi != mlo else 0.5 * (lo + hi))
                rec["mu_direction"] = "neg_to_pos" if mlo < 0 else "pos_to_neg"

        if has_fom and np.isfinite(fom_scale) and fom_scale > 0:
            asym = block[~block._kind.astype(str).str.startswith("sym")]
            asym = asym.dropna(subset=["Re", "C_L_fom"]).sort_values("Re")
            thr = cl_departure_frac * fom_scale
            over = asym[asym.C_L_fom.abs() > thr]
            rec["C_L_fom_threshold"] = thr
            rec["Re_c_C_L_fom_proxy"] = float(over.Re.min()) if len(over) else None
        res.append(rec)

    out = {"per_amp": res}
    both = [(r["amp"], r["Re_c_mu_interp"], r.get("Re_c_C_L_fom_proxy"))
            for r in res if "Re_c_mu_interp" in r and r.get("Re_c_C_L_fom_proxy") is not None]
    if both:
        diffs = [abs(a - b) for _, a, b in both]
        out["mu_vs_C_L_fom_agreement"] = {
            "n": len(both), "max_abs_diff": max(diffs),
            "median_abs_diff": float(np.median(diffs))}
    return out


# ---------------------------------------------------------------------------
# D. offline caches -> reduced dimensions and tensor cost
# ---------------------------------------------------------------------------

def audit_cache(cache_dir):
    name = os.path.basename(cache_dir.rstrip("/"))
    m = CACHE_RE.match(name)
    rec = {"name": name, "path": cache_dir}
    if m:
        rec["K"] = int(m.group("K"))
        raw = m.group("tol")
        rec["tol_raw"] = raw
        if raw is not None:
            try:
                rec["tol"] = float(raw)          # normalises 0.0001 and 1e-06 alike
            except ValueError:
                rec["tol"] = None

    rj = os.path.join(cache_dir, "run.json")
    if os.path.isfile(rj):
        try:
            with open(rj) as fh:
                rec["run_json"] = json.load(fh)
        except Exception as exc:  # noqa: BLE001
            rec["run_json_error"] = str(exc)

    clusters = sorted(d for d in os.listdir(cache_dir)
                      if d.startswith("cluster_") and os.path.isdir(os.path.join(cache_dir, d)))
    rec["n_cluster_dirs"] = len(clusters)
    rec["clusters"] = []
    for c in clusters:
        cdir = os.path.join(cache_dir, c)
        cinfo = {"name": c}
        cinfo["pod_basis"] = dir_inventory(os.path.join(cdir, "pod_basis"))
        cinfo["operators"] = dir_inventory(os.path.join(cdir, "operators"))
        ct = os.path.join(cdir, "conv_tensor.npz")
        if os.path.isfile(ct):
            cinfo["conv_tensor"] = {"size_mb": round(os.path.getsize(ct) / 1e6, 4),
                                    "members": npz_shapes(ct)}
            # the tensor's leading dimension is the reduced dimension
            dims = set()
            for k, v in cinfo["conv_tensor"]["members"].items():
                sh = v.get("shape")
                if sh and len(sh) == 3 and sh[0] == sh[1] == sh[2]:
                    dims.add(sh[0])
            if len(dims) == 1:
                cinfo["r_from_tensor"] = int(dims.pop())
        rec["clusters"].append(cinfo)

    rs = [c.get("r_from_tensor") for c in rec["clusters"]]
    if rs and all(isinstance(x, int) for x in rs):
        rec["r_per_cluster"] = rs
        rec["sum_r_cubed"] = int(sum(r ** 3 for r in rs))
        rec["max_r"] = max(rs)
        rec["cluster_share_of_tensor_cost"] = [
            round(r ** 3 / sum(x ** 3 for x in rs), 4) for r in rs]
    tot = 0.0
    for c in rec["clusters"]:
        if "conv_tensor" in c:
            tot += c["conv_tensor"]["size_mb"]
    rec["tensor_storage_mb_total"] = round(tot, 4)
    return rec


def tensor_cost_table(caches):
    """N_local(tol) vs N_global(tol), and the honest cost ratio."""
    by_tol = {}
    for c in caches:
        if c.get("tol") is None or "sum_r_cubed" not in c:
            continue
        key = f"{c['tol']:g}"
        slot = by_tol.setdefault(key, {})
        if c.get("K") == 1:
            slot["global"] = c
        else:
            slot.setdefault("local", {})[c["K"]] = c

    rows = []
    for tol, slot in sorted(by_tol.items(), key=lambda kv: float(kv[0]), reverse=True):
        g = slot.get("global")
        for K, l in sorted(slot.get("local", {}).items()):
            row = {"tol": float(tol), "K": K,
                   "r_local": l.get("r_per_cluster"),
                   "sum_r_local_cubed": l.get("sum_r_cubed"),
                   "max_r_local": l.get("max_r"),
                   "local_storage_mb": l.get("tensor_storage_mb_total")}
            if g and g.get("r_per_cluster"):
                r_glob = g["r_per_cluster"][0]
                row["r_global"] = r_glob
                row["global_storage_mb"] = g.get("tensor_storage_mb_total")
                if l.get("sum_r_cubed"):
                    row["tensor_cost_ratio_global_over_local"] = round(
                        r_glob ** 3 / l["sum_r_cubed"], 3)
                    row["naive_ratio_if_uniform_max_r"] = round(
                        r_glob ** 3 / (K * l["max_r"] ** 3), 3)
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# E. logs: enough structure to write a real parser next round
# ---------------------------------------------------------------------------

LOG_PATTERNS = OrderedDict([
    ("diverged", re.compile(r"diverg", re.I)),
    ("mu_nan", re.compile(r"mu\s*=\s*nan", re.I)),
    ("converged", re.compile(r"converg(?!e[sd]?\s*not)", re.I)),
    ("newton", re.compile(r"newton", re.I)),
    ("spine", re.compile(r"spine", re.I)),
    ("attempt", re.compile(r"attempt", re.I)),
    ("traceback", re.compile(r"Traceback", re.I)),
    ("offline_pod", re.compile(r"\bPOD\b", re.I)),
    ("offline_tensor", re.compile(r"tensor.*(precompute|assembl|built|cache)", re.I)),
    ("timing_seconds", re.compile(r"\b\d+\.\d+\s*s\b")),
])


def audit_log(path, n_examples=4, tail=25):
    rec = {"path": path, "size_mb": round(os.path.getsize(path) / 1e6, 4)}
    counts = Counter()
    examples = {k: [] for k in LOG_PATTERNS}
    lines = 0
    last = []
    try:
        with open(path, errors="replace") as fh:
            for line in fh:
                lines += 1
                last.append(line.rstrip("\n"))
                if len(last) > tail:
                    last.pop(0)
                for key, pat in LOG_PATTERNS.items():
                    if pat.search(line):
                        counts[key] += 1
                        if len(examples[key]) < n_examples:
                            examples[key].append(line.rstrip("\n")[:300])
    except Exception as exc:  # noqa: BLE001
        rec["error"] = f"{type(exc).__name__}: {exc}"
        return rec
    rec["n_lines"] = lines
    rec["pattern_counts"] = dict(counts)
    rec["pattern_examples"] = {k: v for k, v in examples.items() if v}
    rec["tail"] = last
    return rec


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def format_report(audit):
    L = []
    A = L.append
    A("=" * 78)
    A("SECTION 6 AUDIT")
    A("=" * 78)

    A("")
    A("--- SWEEP RUNS " + "-" * 62)
    A(f"{'tag':24s} {'rows':>6s} {'idx':>6s} {'pts':>6s} {'Re':>14s} {'amps':>5s} "
      f"{'muNaN':>6s} {'C_Lfom':>7s}")
    for r in audit["states"]:
        if "n_rows" not in r:
            A(f"{r['tag']:24s} ERROR: {r.get('error')}")
            continue
        Re = r.get("Re", {})
        A(f"{r['tag']:24s} {r['n_rows']:6d} {r.get('n_index_lines', -1):6d} "
          f"{r.get('n_point_files', -1):6d} "
          f"{Re.get('min', float('nan')):6.1f}-{Re.get('max', float('nan')):<7.1f} "
          f"{r.get('amps', {}).get('n', -1):5d} "
          f"{r.get('mu', {}).get('n_nan', -1):6d} "
          f"{r.get('n_C_L_fom_present', 0):7d}")

    A("")
    A("--- TIMINGS " + "-" * 65)
    for r in audit["states"]:
        t = r.get("timings") or {}
        sp = t.get("speedup_per_solve")
        if not t:
            continue
        bits = []
        if t.get("t_rom"):
            bits.append(f"t_rom med={t['t_rom']['median']:.4g}")
        if t.get("t_fom"):
            bits.append(f"t_fom med={t['t_fom']['median']:.4g}")
        if sp:
            bits.append(f"speedup med={sp['median']:.2f} IQR[{sp['p25']:.2f},{sp['p75']:.2f}] n={sp['n']}")
        A(f"  {r['tag']:24s} " + "  ".join(bits))

    A("")
    A("--- OFFLINE CACHES " + "-" * 58)
    A(f"{'cache':22s} {'K':>3s} {'tol':>9s} {'r per cluster':>26s} {'sum r^3':>12s} {'MB':>8s}")
    for c in audit["caches"]:
        tol_s = "none" if c.get("tol") is None else format(c["tol"], "g")
        A(f"{c['name']:22s} {c.get('K', -1):3d} "
          f"{tol_s:>9s} "
          f"{str(c.get('r_per_cluster', '?')):>26s} "
          f"{str(c.get('sum_r_cubed', '?')):>12s} "
          f"{c.get('tensor_storage_mb_total', 0):8.2f}")

    if audit.get("tensor_cost"):
        A("")
        A("--- N_local(tol) vs N_global(tol), tensor cost " + "-" * 31)
        A(f"{'tol':>8s} {'K':>3s} {'r_local':>26s} {'r_glob':>7s} {'ratio':>8s} {'naive':>8s}")
        for row in audit["tensor_cost"]:
            A(f"{row['tol']:8.0e} {row['K']:3d} {str(row.get('r_local')):>26s} "
              f"{str(row.get('r_global', '?')):>7s} "
              f"{str(row.get('tensor_cost_ratio_global_over_local', '?')):>8s} "
              f"{str(row.get('naive_ratio_if_uniform_max_r', '?')):>8s}")

    for tag, dup in audit.get("duplicate_legs", {}).items():
        A("")
        A(f"--- DUPLICATE LEGS ({tag}) " + "-" * 46)
        if "error" in dup:
            A(f"  {dup['error']}")
            continue
        A(f"  |C_L| scale = {dup['C_L_scale']:.4g}")
        A(f"  {'amp':>6s} {'kind':>10s} {'n':>4s} {'max|C_L|':>11s} {'dist2sym':>11s} {'flat':>5s} {'dist':>5s}")
        for l in dup["legs"]:
            A(f"  {l['amp'] if l['amp'] is not None else 0:6.2f} {str(l['kind']):>10s} {l['n']:4d} "
              f"{(l['abs_C_L_max'] or 0):11.3e} "
              f"{(l['max_dist_to_sym'] if l['max_dist_to_sym'] is not None else float('nan')):11.3e} "
              f"{str(l['flat_test_duplicate']):>5s} {str(l['dist_test_duplicate']):>5s}")
        A(f"  duplicates (both tests agree): {dup['duplicates_both_tests']}")
        if dup["disagreements"]:
            A(f"  !! tests disagree on {len(dup['disagreements'])} leg(s) -- pick the threshold from these")

    for tag, cr in audit.get("crossings", {}).items():
        A("")
        A(f"--- Re_c PER AMPLITUDE ({tag}) " + "-" * 42)
        if "error" in cr:
            A(f"  {cr['error']}")
            continue
        A(f"  {'amp':>6s} {'n_sym':>6s} {'muFin':>6s} {'flips':>5s} {'bracket':>17s} "
          f"{'Re_c(mu)':>9s} {'Re_c(C_Lfom)':>13s}")
        for r in cr["per_amp"]:
            br = r.get("mu_bracket")
            A(f"  {r['amp'] if r['amp'] is not None else 0:6.2f} {r['n_sym_points']:6d} "
              f"{r.get('n_mu_finite', 0):6d} {r.get('n_sign_changes', 0):5d} "
              f"{(f'[{br[0]:.1f},{br[1]:.1f}]' if br else '-'):>17s} "
              f"{(r.get('Re_c_mu_interp') or float('nan')):9.3f} "
              f"{(r.get('Re_c_C_L_fom_proxy') or float('nan')):13.3f}")
        ag = cr.get("mu_vs_C_L_fom_agreement")
        if ag:
            A(f"  mu vs C_L_fom proxy: n={ag['n']} median|diff|={ag['median_abs_diff']:.3f} "
              f"max|diff|={ag['max_abs_diff']:.3f}")

    A("")
    A("--- LOGS " + "-" * 68)
    for r in audit["logs"]:
        c = r.get("pattern_counts", {})
        A(f"  {os.path.basename(r['path']):34s} lines={r.get('n_lines', -1):6d} "
          + " ".join(f"{k}={v}" for k, v in sorted(c.items())))

    A("")
    A("=" * 78)
    return "\n".join(L)


# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-root", default=".")
    p.add_argument("--out", default="paper_data/audit.json")
    p.add_argument("--report", default="paper_data/audit_report.txt")
    p.add_argument("--states", nargs="*", default=None,
                   help="subset of state tags (default: all with a sweep_results.csv)")
    p.add_argument("--deep-tags", nargs="*",
                   default=["E1_K4_tensor", "E2_K1_tensor", "E11_K4_matchdim",
                            "E12_K1_matchdim", "R1_tensor_stored", "R2_deim8_stored"],
                   help="tags to run leg-duplication and Re_c extraction on")
    p.add_argument("--abs-tol", type=float, default=1e-3)
    p.add_argument("--rel-tol", type=float, default=1e-2)
    p.add_argument("--cl-departure-frac", type=float, default=0.02)
    args = p.parse_args(argv)

    root = os.path.abspath(args.repo_root)
    states_root = os.path.join(root, "states")
    cache_root = os.path.join(root, "local_rom")
    logs_root = os.path.join(root, "logs")

    audit = {"repo_root": root, "argv": sys.argv[1:],
             "thresholds": {"abs_tol": args.abs_tol, "rel_tol": args.rel_tol,
                            "cl_departure_frac": args.cl_departure_frac},
             "states": [], "caches": [], "logs": [],
             "duplicate_legs": {}, "crossings": {}}

    # --- runs
    if os.path.isdir(states_root):
        tags = args.states or sorted(
            d for d in os.listdir(states_root)
            if os.path.isfile(os.path.join(states_root, d, "sweep_results.csv")))
        for tag in tags:
            sd = os.path.join(states_root, tag)
            got = audit_state(sd)
            if isinstance(got, tuple):
                rec, df = got
            else:
                rec, df = got, None
            audit["states"].append(rec)
            if df is not None and tag in args.deep_tags:
                audit["duplicate_legs"][tag] = jsonable(
                    duplicate_legs(df, args.abs_tol, args.rel_tol))
                audit["crossings"][tag] = jsonable(
                    crossings(df, args.cl_departure_frac))
    else:
        audit["states_error"] = f"no such directory: {states_root}"

    # --- caches
    if os.path.isdir(cache_root):
        for d in sorted(os.listdir(cache_root)):
            full = os.path.join(cache_root, d)
            if os.path.isdir(full):
                audit["caches"].append(audit_cache(full))
        audit["tensor_cost"] = tensor_cost_table(audit["caches"])
    else:
        audit["caches_error"] = f"no such directory: {cache_root}"

    # --- logs
    if os.path.isdir(logs_root):
        for f in sorted(os.listdir(logs_root)):
            if f.endswith(".log") or f.startswith(("summary_", "replay_summary_")):
                audit["logs"].append(audit_log(os.path.join(logs_root, f)))

    audit = jsonable(audit)

    for path in (args.out, args.report):
        d = os.path.dirname(os.path.join(root, path))
        if d:
            os.makedirs(d, exist_ok=True)

    with open(os.path.join(root, args.out), "w") as fh:
        json.dump(audit, fh, indent=2, sort_keys=False)

    report = format_report(audit)
    with open(os.path.join(root, args.report), "w") as fh:
        fh.write(report + "\n")
    print(report)
    print(f"\nwrote {args.out} and {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
