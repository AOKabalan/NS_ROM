"""
common.py -- shared data access, styling and LaTeX emission for the paper
figure scripts.

Design rules, matching the handoff note:
  * figure scripts PLOT; they do not compute field errors. Anything requiring
    stored fields comes from paper_data/point_errors_*.csv.
  * index.jsonl is the single source of truth, never sweep_results.csv: the
    index carries idx, cluster, iterations and err_u, and the CSV has no idx
    column, so any CSV-based join relies on row order matching write order.
  * the duplicate-leg filter is DERIVED from the data (or read from the audit
    JSON), never hardcoded to |a| >= 0.8.
  * the error norm in axis labels is read from mass/mass_meta.json, so labels
    cannot silently disagree with the matrix the errors were computed in.
"""

from __future__ import annotations

import json
import os
import re
import sys

import numpy as np
import pandas as pd

# renderers are invoked from the repo root; put it on the path so an in-tree
# nsrom resolves as readily as an installed one
_ROOT = os.getcwd()
if os.path.isdir(_ROOT) and _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

BRANCH_RE = re.compile(r"^(?P<kind>[^@]+)@amp(?P<amp>[-+]?\d*\.?\d+)\s*$")

# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def state_dir(tag, root="states"):
    return tag if os.path.isdir(tag) else os.path.join(root, tag)


def load_run(tag, root="states", converged_only=False):
    """StateStore index -> DataFrame with parsed branch kind and amplitude."""
    from nsrom.io.state_store import StateStore

    sd = state_dir(tag, root)
    store = StateStore(sd)
    df = pd.DataFrame(store.records)
    if df.empty:
        raise RuntimeError(f"{sd}: empty index.jsonl")

    parsed = df.branch.map(lambda b: BRANCH_RE.match(str(b)))
    df["kind"] = [m.group("kind") if m else str(b)
                  for m, b in zip(parsed, df.branch)]
    df["amp_leg"] = [float(m.group("amp")) if m else np.nan for m in parsed]
    if "amp" not in df:
        df["amp"] = df["amp_leg"]
    df["is_sym"] = df.kind.astype(str).str.startswith("sym")
    df["tag"] = os.path.basename(sd.rstrip("/"))
    df["_meta"] = None
    df.attrs["meta"] = store.meta
    df.attrs["state_dir"] = sd
    if converged_only:
        df = df[df.converged.astype(bool)].copy()
    return df


def load_point_errors(path):
    """The output of compute_point_errors.py, keyed by idx."""
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def attach_point_errors(df, errors_csv):
    """Merge per-point field errors on idx -- an exact integer key."""
    e = load_point_errors(errors_csv)
    if e is None:
        return df
    cols = [c for c in ("idx", "err_p_M", "err_p_nodal", "err_u_recomp")
            if c in e.columns]
    return df.merge(e[cols], on="idx", how="left", validate="one_to_one")


# ---------------------------------------------------------------------------
# duplicate legs
# ---------------------------------------------------------------------------

def legs_from_run(tag, root="states", **kw):
    """Classify duplicate legs ONCE on the master run and reuse everywhere.

    Which legs are spurious is a property of the parameter box (the bifurcation
    lies above Re_max), not of the ROM variant, so every figure must apply the
    same set or panels stop being comparable. It also avoids classifying on a
    run whose symmetric legs diverged: for a DEIM replay C_L is NaN over most
    shared Re, so no leg can be tested at all.
    """
    return duplicate_leg_keys(load_run(tag, root), tag=tag, **kw)


def duplicate_leg_keys(df, abs_tol=1e-3, rel_tol=1e-2, audit_json=None,
                       tag=None, verbose=True):
    """
    Return {(amp, kind)} for asymmetric legs that are the symmetric solution
    relabelled. Prefers the audit JSON if given; otherwise recomputes with the
    same two-test criterion (flat |C_L|, and distance to the symmetric leg over
    shared Re) and reports disagreements rather than silently choosing.
    """
    if audit_json and os.path.exists(audit_json):
        with open(audit_json) as fh:
            a = json.load(fh)
        blk = (a.get("duplicate_legs") or {}).get(tag or df.tag.iloc[0])
        if blk and "duplicates_both_tests" in blk:
            keys = {(round(float(d["amp"]), 6), d["kind"])
                    for d in blk["duplicates_both_tests"]}
            if verbose:
                print(f"  [legs] {len(keys)} duplicate leg(s) from {audit_json}")
            return keys

    cl = "C_L_rom" if "C_L_rom" in df else "C_L"
    if cl not in df:
        return set()
    scale = float(pd.to_numeric(df[cl], errors="coerce").abs().max())
    keys, disagree = set(), []
    for amp, blk in df.groupby("amp"):
        sym = blk[blk.is_sym]
        smap = sym.groupby("Re")[cl].mean() if len(sym) else pd.Series(dtype=float)
        for kind, leg in blk[~blk.is_sym].groupby("kind"):
            v = pd.to_numeric(leg[cl], errors="coerce")
            flat = float(v.abs().max()) if v.notna().any() else None
            shared = leg[leg.Re.isin(smap.index)]
            dist = None
            if len(shared):
                gap = np.abs(pd.to_numeric(shared[cl], errors="coerce").values
                             - smap.reindex(shared.Re).values)
                if np.isfinite(gap).any():
                    dist = float(np.nanmax(gap))
            t1 = flat is not None and flat < abs_tol
            t2 = dist is not None and dist < rel_tol * scale
            # the distance test asks the actual question ("is this leg the
            # symmetric solution?") and is scale-relative, so it decides when
            # it is available; the flat test only stands in when no Re is shared
            dup = t2 if dist is not None else t1
            if dup:
                keys.add((round(float(amp), 6), kind))
            if dist is not None and t1 != t2:
                disagree.append((amp, kind, flat, dist))
    if verbose:
        print(f"  [legs] {len(keys)} duplicate leg(s) detected "
              f"(abs_tol={abs_tol:g}, rel_tol={rel_tol:g})")
        for amp, kind, flat, dist in disagree:
            print(f"  [legs] !! tests disagree at a={amp:+.2f} {kind}: "
                  f"max|C_L|={flat:.2e} dist2sym={dist}")
    return keys


def drop_duplicate_legs(df, keys):
    if not keys:
        return df
    mask = pd.Series(
        [(round(float(a), 6), k) in keys for a, k in zip(df.amp, df.kind)],
        index=df.index)
    return df[~mask].copy()


# ---------------------------------------------------------------------------
# Re_c extraction
# ---------------------------------------------------------------------------

def mu_crossings(df, min_points=2):
    """
    Per amplitude: first sign change of mu along the symmetric leg. Returns a
    DataFrame with the bracket, the linear-interpolated Re_c and the crossing
    direction. The bracket width IS the resolution -- quote it, do not hide it.
    """
    out = []
    for amp, blk in df[df.is_sym].groupby("amp"):
        s = blk.dropna(subset=["Re", "mu"]).sort_values("Re")
        rec = {"amp": float(amp), "n": len(s)}
        if len(s) >= min_points:
            mu, Re = s.mu.values.astype(float), s.Re.values.astype(float)
            sgn = np.sign(mu)
            flips = [i for i in np.where(np.diff(sgn) != 0)[0] if sgn[i] != 0]
            rec["n_crossings"] = len(flips)
            if flips:
                i = flips[0]
                lo, hi, mlo, mhi = Re[i], Re[i + 1], mu[i], mu[i + 1]
                rec.update(Re_lo=float(lo), Re_hi=float(hi),
                           bracket=float(hi - lo),
                           Re_c=float(lo - mlo * (hi - lo) / (mhi - mlo))
                           if mhi != mlo else float(0.5 * (lo + hi)),
                           direction="neg_to_pos" if mlo < 0 else "pos_to_neg")
        out.append(rec)
    return pd.DataFrame(out).sort_values("amp").reset_index(drop=True)


def cl_fom_departure(df, frac=0.02, per_amp_scale=True):
    """
    Re_c proxy: first Re where an asymmetric leg's |C_L_fom| clears frac * scale.

    The scale MUST be per amplitude. Against a global scale the proxy fails
    wherever a leg's own lift is small compared with the largest leg in the
    dataset -- at a = -0.40 the leg peaks at 5.8e-3 against a global 0.17, so a
    global threshold is not cleared until Re = 97 while mu crosses at 53.8.

    Note this proxy is only meaningful at a = 0 in principle: for a != 0 the
    rotating cylinders generate lift on the symmetric branch too, so departure
    from zero does not by itself locate a bifurcation. It is reported per
    amplitude for cross-checking against mu, not as an independent reference.
    """
    if "C_L_fom" not in df:
        return pd.DataFrame(columns=["amp", "Re_c_proxy"])
    gscale = float(pd.to_numeric(df.C_L_fom, errors="coerce").abs().max())
    out = []
    for amp, blk in df.groupby("amp"):
        a = blk[~blk.is_sym].dropna(subset=["Re", "C_L_fom"])
        scale = (float(a.C_L_fom.abs().max()) if (per_amp_scale and len(a))
                 else gscale)
        over = a[a.C_L_fom.abs() > frac * scale] if scale > 0 else a.iloc[0:0]
        out.append({"amp": float(amp), "proxy_scale": scale,
                    "Re_c_proxy": float(over.Re.min()) if len(over) else np.nan})
    return pd.DataFrame(out).sort_values("amp").reset_index(drop=True)


# ---------------------------------------------------------------------------
# reduced dimensions from the offline cache
# ---------------------------------------------------------------------------

CACHE_RE = re.compile(r"^K(?P<K>\d+)_H1(?:_tol(?P<tol>.+))?$")


def cache_dims(cache_root="local_rom"):
    """
    {(K, tol): [r_0, ...]} read from each cluster's conv_tensor.npz leading
    dimension. Tolerances are normalised through float(), so 'tol0.0001' and
    'tol1e-04' land on the same key -- the naming is inconsistent on disk and
    globbing on the string silently drops the loosest point.
    """
    import zipfile

    def _header(fh, version):
        """np.lib.format._read_array_header vanished in numpy 2."""
        major = version[0]
        for name in (f"read_array_header_{major}_0", "_read_array_header"):
            fn = getattr(np.lib.format, name, None)
            if fn is None:
                continue
            try:
                return fn(fh) if name.startswith("read_") else fn(fh, version)
            except TypeError:
                return fn(fh, version)
        raise RuntimeError(f"cannot read npy header version {version}")

    def shapes(p):
        out = {}
        with zipfile.ZipFile(p) as z:
            for nm in z.namelist():
                try:
                    with z.open(nm) as fh:
                        ver = np.lib.format.read_magic(fh)
                        sh, _f, _d = _header(fh, ver)
                    out[nm] = list(sh)
                except Exception:  # noqa: BLE001
                    with np.load(p, allow_pickle=False) as f:
                        key = nm[:-4] if nm.endswith(".npy") else nm
                        out[nm] = list(f[key].shape)
        return out

    dims = {}
    if not os.path.isdir(cache_root):
        return dims
    for d in sorted(os.listdir(cache_root)):
        m = CACHE_RE.match(d)
        if not m:
            continue
        try:
            tol = float(m.group("tol")) if m.group("tol") else None
        except ValueError:
            tol = None
        K = int(m.group("K"))
        rs = []
        k = 0
        while True:
            ct = os.path.join(cache_root, d, f"cluster_{k}", "conv_tensor.npz")
            if not os.path.exists(ct):
                break
            cube = {tuple(s) for s in shapes(ct).values()
                    if len(s) == 3 and s[0] == s[1] == s[2]}
            rs.append(int(next(iter(cube))[0]) if len(cube) == 1 else None)
            k += 1
        if rs:
            dims[(K, tol)] = rs
    return dims


def tensor_storage_mb(K, tol, cache_root="local_rom"):
    tag = "" if tol is None else f"_tol{tol:g}"
    for cand in (f"K{K}_H1{tag}", f"K{K}_H1_tol{tol}"):
        d = os.path.join(cache_root, cand)
        if os.path.isdir(d):
            tot = 0.0
            for k in range(64):
                p = os.path.join(d, f"cluster_{k}", "conv_tensor.npz")
                if not os.path.exists(p):
                    break
                tot += os.path.getsize(p) / 1e6
            return round(tot, 3)
    return None


# ---------------------------------------------------------------------------
# style and output
# ---------------------------------------------------------------------------

def norm_label(mass_meta="mass/mass_meta.json", which="M_u"):
    r"""'H^1' or 'L^2' for the matrix the errors were actually measured in.

    The direct comparison against an assembled L2 mass matrix is conclusive, so
    it is consulted first; the scaling verdict is only a fallback. Returns an
    obviously-wrong '\ast' if neither resolves, so an unlabelled norm cannot
    reach a draft unnoticed.
    """
    if os.path.exists(mass_meta):
        try:
            with open(mass_meta) as fh:
                m = json.load(fh)
            rec = (m.get("matrices", {}) or {}).get(which, {}) or {}
            lc = rec.get("l2_comparison")
            if isinstance(lc, dict) and "is_l2_mass" in lc:
                return "L^2" if lc["is_l2_mass"] else "H^1"
            v = str(rec.get("verdict", "")).lower()
            if "gradient" in v:
                return "H^1"
            if "mass" in v:
                return "L^2"
        except Exception:  # noqa: BLE001
            pass
    return r"\ast"      # deliberately wrong-looking: forces the norm to be resolved


# ---------------------------------------------------------------------------
# style -- every knob lives in render/style.py; nothing is mirrored here
# ---------------------------------------------------------------------------

import style  # noqa: E402  (render/ is on sys.path via the caller's insert)


def P(name):
    """Palette / geometry constant from render/style.py."""
    return getattr(style, name)


def branch_color(kind):
    """Semantic colour: the two asymmetric branches by sign, symmetric grey."""
    k = str(kind)
    if k.startswith("sym"):
        return style.C_SYM
    if "+" in k:
        return style.C_POS
    if "-" in k:
        return style.C_NEG
    return style.C_GLOBAL


def cluster_color(k):
    return style.CLUSTER_COLORS[int(k) % len(style.CLUSTER_COLORS)]


def seq_cmap():
    return style.CMAP_SEQ


def div_cmap():
    return style.CMAP_DIV


def resolve_width(spec):
    return float(style.resolve_width(spec))


def figsize(width_spec, aspect=0.62):
    """(w, h) in inches. Size to the width the figure is INCLUDED at and never
    let \\includegraphics rescale, or font sizes drift between figures."""
    w = resolve_width(width_spec)
    return (w, w * float(aspect))


def setup_style(usetex=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    style.set_style(usetex=usetex)
    return plt


def save(fig, outdir, name, png=False):
    """PDF is the deliverable; dpi and bbox come from the shared rcParams."""
    os.makedirs(outdir, exist_ok=True)
    exts = ["pdf"] + (["png"] if png else [])
    for ext in exts:
        p = os.path.join(outdir, f"{name}.{ext}")
        fig.savefig(p)
        print(f"  wrote {p}")


def base_parser(description, default_out="render/out"):
    import argparse
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--outdir", default=default_out)
    p.add_argument("--width", default="linewidth",
                   help="preset ('single', 'double', 'linewidth', "
                        "'0.7linewidth') or inches")
    p.add_argument("--aspect", type=float, default=None,
                   help="override the figure's height/width ratio")
    p.add_argument("--usetex", action="store_true")
    p.add_argument("--png", action="store_true", help="also write a PNG preview")
    p.add_argument("--states-root", default="states")
    p.add_argument("--cache-root", default="local_rom")
    p.add_argument("--audit", default="paper_data/audit.json")
    p.add_argument("--mass-meta", default="mass/mass_meta.json")
    return p


# ---------------------------------------------------------------------------
# LaTeX
# ---------------------------------------------------------------------------

def latex_table(rows, header, caption, label, align=None, notes=None):
    align = align or ("l" + "r" * (len(header) - 1))
    L = [r"\begin{table}[tb]", r"  \centering",
         f"  \\caption{{{caption}}}", f"  \\label{{{label}}}",
         f"  \\begin{{tabular}}{{{align}}}", r"    \toprule",
         "    " + " & ".join(header) + r" \\", r"    \midrule"]
    for r in rows:
        L.append("    " + " & ".join("--" if c is None else str(c) for c in r) + r" \\")
    L += [r"    \bottomrule", r"  \end{tabular}"]
    if notes:
        L.append(r"  \\[2pt] \footnotesize " + notes)
    L += [r"\end{table}", ""]
    return "\n".join(L)


def fmt(x, spec=".3g", dash="--"):
    if x is None:
        return dash
    try:
        if isinstance(x, float) and not np.isfinite(x):
            return dash
        return format(x, spec)
    except (TypeError, ValueError):
        return str(x)


def write_macros(path, macros):
    r"""Emit \newcommand definitions so no number is typed into the .tex.

    A LaTeX control sequence is letters only. ``\SpeedupE1`` tokenises as
    ``\SpeedupE`` followed by ``1``, so \newcommand reads the digit as an
    argument count and the file fails to read -- taking every macro after it
    down with it, which makes the error look like it comes from somewhere else
    entirely. Deriving a macro name from a run tag (E1, E10, K4) produces this
    every time, so it is checked here rather than discovered in LaTeX.

    The check raises instead of renaming: a silent rename would desynchronise
    the prose, which spells the macro out by hand.
    """
    bad = sorted(k for k in macros if not k.isalpha())
    if bad:
        raise ValueError(
            "these macro names are not valid LaTeX control sequences "
            "(letters only, no digits or underscores):\n  "
            + "\n  ".join(bad)
            + "\n\nRename them at the call site with an explicit label map. "
              "Do not derive a macro name from a run tag."
        )
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        fh.write("% generated -- do not edit by hand\n")
        for k, v in macros.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print(f"  wrote {path} ({len(macros)} macros)")
