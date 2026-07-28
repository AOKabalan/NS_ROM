"""
Publication figures for the local-ROM parameter sweep (fluidic pinball).

Everything pulls fonts, sizes, widths and the palette from
`nsrom.plotting.style`, so the paper stays single-source-of-truth.

Input CSV header:
    branch, Re, amp, cluster, converged, iterations, t_rom, mu,
    C_D, C_L, C_L_rom, C_L_fom, err_u, fom_converged, fom_iters

BRANCH LABELS
-------------
The raw `branch` column encodes the *continuation path*, not the physical
branch: 'asym+1@spine+', 'asym-1@amp-0.20', 'sym', ... Physically these live on
three sheets, so canonical_branch() collapses them to 'sym' / 'asym 1' /
'asym 2'. The raw label stays in `branch` for per-path diagnostics; the
collapsed one lands in `branch_base` and is what every figure legend uses.

3D LAYOUT
---------
matplotlib's tight_layout is unreliable for 3D axes, and style.py's
minor-tick setting (correct for 2D) puts hundreds of ticks on 3D axes. Both are
handled in _finish_3d(), which every 3D figure calls -- do not add tight_layout
to a 3D figure here or the labels will collide again.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator, NullLocator
from matplotlib.colors import LogNorm

from nsrom.plotting.style import (
    set_style,
    resolve_width,
    C_POS,      # asym 1  (deep blue)
    C_NEG,      # asym 2  (burnt orange)
    C_SYM,      # symmetric (mid grey)
    C_GLOBAL,   # near-black reference, reserved for the global-ROM overlay
    CLUSTER_COLORS,
)

# =============================================================================
# canonical branches
# =============================================================================

SYM, ASYM1, ASYM2 = "sym", "asym 1", "asym 2"

CANON_COLORS = {SYM: C_SYM, ASYM1: C_POS, ASYM2: C_NEG}
CANON_ORDER = [SYM, ASYM1, ASYM2]          # legend / table row order


def canonical_branch(label):
    """'asym+1@spine+' -> 'asym 1';  'asym-1@amp-0.20' -> 'asym 2';  'sym' -> 'sym'.

    Returns None when the stem is unrecognised, in which case load_sweep()
    falls back to sign(C_L_fom). If your +1/-1 convention is the opposite of
    C_L's sign, swap ASYM1/ASYM2 here -- nothing else needs to change.
    """
    stem = str(label).split("@")[0].strip().lower()
    if stem.startswith("sym"):
        return SYM
    if "+1" in stem:
        return ASYM1
    if "-1" in stem:
        return ASYM2
    return None


# =============================================================================
# loading
# =============================================================================

def load_sweep(csv_path, converged_only=True):
    """Read the sweep CSV, add err_cl and the canonical `branch_base` column."""
    df = pd.read_csv(csv_path)
    df["err_cl"] = (df["C_L_rom"] - df["C_L_fom"]).abs()

    base = df["branch"].map(canonical_branch).astype(object)
    missing = base.isna()
    if missing.any():
        cl = df.loc[missing, "C_L_fom"].to_numpy()
        base.loc[missing] = np.where(cl > 1e-9, ASYM1,
                                     np.where(cl < -1e-9, ASYM2, SYM))
    df["branch_base"] = base

    if converged_only and "converged" in df:
        df = df[df["converged"].astype(bool)].copy()
    return df


def unique_branches(df):
    """Report raw paths and how they collapsed -- sanity check, run once."""
    raw = sorted(df["branch"].unique(), key=str)
    print(f"{len(raw)} raw continuation paths ->", end=" ")
    print({b: int((df["branch_base"] == b).sum()) for b in CANON_ORDER
           if (df["branch_base"] == b).any()})
    return raw


def _iter_branches(df):
    """Yield (canonical_label, sub_df, colour) in CANON_ORDER."""
    for label in CANON_ORDER:
        sub = df[df["branch_base"] == label]
        if not sub.empty:
            yield label, sub, CANON_COLORS[label]


def _save(fig, out, tight_bbox=True):
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        # 3D figures are laid out by hand; a tight bbox would undo that
        fig.savefig(out, bbox_inches="tight" if tight_bbox else None)
        print("wrote", out)
    return fig


# =============================================================================
# 3D layout helper  -- the fix for the crowding
# =============================================================================

def _finish_3d(fig, ax, nticks=4, box_aspect=(1, 1, 0.62)):
    """Make a 3D axes publication-legible.

    - kills the minor ticks style.py switches on (right for 2D, ruinous in 3D)
    - thins the major ticks
    - pads the axis labels away from the tick labels
    - lightens the panes/grid so the data carries the figure
    """
    ax.set_box_aspect(box_aspect)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_minor_locator(NullLocator())
        axis.set_major_locator(MaxNLocator(nticks))
        axis.pane.set_facecolor("white")
        axis.pane.set_edgecolor("0.88")
        axis.pane.set_alpha(1.0)
        axis._axinfo["grid"].update(color="0.90", linewidth=0.5)
        axis._axinfo["tick"].update(inward_factor=0.0, outward_factor=0.3)

    ax.xaxis.labelpad = 10
    ax.yaxis.labelpad = 10
    ax.zaxis.labelpad = 8
    ax.tick_params(pad=2)


# =============================================================================
# error colour scale
# =============================================================================

# Edit this once if err_u is a relative error -- e.g.
#   ERR_LABEL = r"$\|u_{\mathrm{ROM}}-u_{\mathrm{FOM}}\|/\|u_{\mathrm{FOM}}\|$"
ERR_LABEL = r"$\|u_{\mathrm{ROM}}-u_{\mathrm{FOM}}\|$"


def error_norm(df, err_col="err_u", vmin=None, vmax=None):
    """LogNorm over the error column, so colourbars tick as 10^k directly.

    Shared by the manifold and the faceted map so both figures use identical
    limits -- important if you place them side by side in the paper.

    Non-positive entries (exact zeros, or diverged rows if you passed
    converged_only=False) are excluded from the limits; LogNorm cannot map
    them and matplotlib will draw them with the 'under' colour.
    """
    v = df[err_col].to_numpy(dtype=float)
    pos = v[np.isfinite(v) & (v > 0)]
    if pos.size == 0:
        raise ValueError(f"{err_col} has no positive finite values")
    return LogNorm(vmin=vmin or pos.min(), vmax=vmax or pos.max())


def _cbar_3d(fig, mappable, label, rect=(0.87, 0.28, 0.022, 0.44)):
    """Colorbar in its own axes so it never collides with the z-label."""
    cax = fig.add_axes(rect)
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label(label, labelpad=6)
    cbar.outline.set_linewidth(0.6)
    cbar.ax.tick_params(width=0.6, length=2.5)
    return cbar


# =============================================================================
# 1. 3D error scatter
# =============================================================================

def plot_error_3d(df, width="linewidth", elev=22, azim=-58, out=None):
    """(Re, amp, log10 err_u), one colour per canonical branch.

    Default width is the full text width: a 3D axes at single-column (3.4in)
    cannot carry three labelled axes plus a legend without collisions.
    """
    set_style()
    w = resolve_width(width)
    fig = plt.figure(figsize=(w, w * 0.62))
    ax = fig.add_subplot(111, projection="3d")

    for label, sub, col in _iter_branches(df):
        ax.scatter(sub["Re"], sub["amp"], np.log10(sub["err_u"]),
                   s=9, color=col, depthshade=False, alpha=0.85,
                   edgecolors="none", label=label)

    ax.set_xlabel(r"$Re$")
    ax.set_ylabel(r"amplitude $A$")
    ax.set_zlabel(r"$\|u_{\mathrm{ROM}}-u_{\mathrm{FOM}}\|$")
    ax.zaxis.set_major_formatter(FuncFormatter(lambda v, _: r"$10^{%d}$" % v))
    ax.view_init(elev=elev, azim=azim)
    _finish_3d(fig, ax)

    # legend below the axes, horizontal -- never overlaps the data
    fig.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.01),
               handletextpad=0.3, columnspacing=1.4, markerscale=1.6)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.12, top=0.98)
    return _save(fig, out, tight_bbox=False)


# =============================================================================
# 2. 3D solution manifold  (the hero figure)
# =============================================================================

def plot_solution_manifold_3d(df, err_col="err_u", width="linewidth",
                              elev=22, azim=-58, out=None, as_lines=False,
                              norm=None):
    """(Re, amp, C_L) coloured by log10 error -- the pitchfork unfolded.

    as_lines=True draws each continuation path as a thin grey guide underneath
    the coloured points, which reads far better than a bare cloud when the
    sweep is made of continuation curves (as yours is).
    """
    set_style()
    w = resolve_width(width)
    fig = plt.figure(figsize=(w, w * 0.62))
    ax = fig.add_subplot(111, projection="3d")

    if as_lines:
        for _, path in df.groupby("branch"):
            path = path.sort_values("Re")
            ax.plot(path["Re"], path["amp"], path["C_L_fom"],
                    color="0.80", lw=0.5, zorder=1)

    p = ax.scatter(df["Re"], df["amp"], df["C_L_fom"],
                   c=df[err_col], cmap="viridis",
                   norm=norm or error_norm(df, err_col), s=7,
                   depthshade=False, edgecolors="none", zorder=2)

    ax.set_xlabel(r"$Re$")
    ax.set_ylabel(r"amplitude $A$")
    ax.set_zlabel(r"$C_L$")
    ax.view_init(elev=elev, azim=azim)
    _finish_3d(fig, ax)

    _cbar_3d(fig, p, ERR_LABEL)
    fig.subplots_adjust(left=0.0, right=0.83, bottom=0.04, top=0.98)
    return _save(fig, out, tight_bbox=False)


# =============================================================================
# 3. 2D faceted error map  -- the most readable error figure
# =============================================================================

def plot_error_map(df, err_col="err_u", width="linewidth", out=None,
                   norm=None):
    """One (Re, amp) panel per branch, colour = log10 error, shared colourbar.

    This is the version a reviewer can actually read numbers off: no depth
    ambiguity, no occlusion, and the three sheets sit side by side. Strongly
    consider this over the 3D scatter for the accuracy claim.
    """
    set_style()
    w = resolve_width(width)
    branches = [(l, s, c) for l, s, c in _iter_branches(df)]
    n = len(branches)

    fig, axes = plt.subplots(1, n, figsize=(w, w / n * 0.95),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)

    nrm = norm or error_norm(df, err_col)

    for ax, (label, sub, _) in zip(axes, branches):
        p = ax.scatter(sub["Re"], sub["amp"], c=sub[err_col],
                       cmap="viridis", s=7, norm=nrm,
                       edgecolors="none")
        ax.set_title(label)
        ax.set_xlabel(r"$Re$")
        ax.xaxis.set_major_locator(MaxNLocator(5))
    axes[0].set_ylabel(r"amplitude $A$")

    fig.subplots_adjust(left=0.08, right=0.87, bottom=0.18, top=0.90, wspace=0.12)
    cax = fig.add_axes([0.89, 0.18, 0.018, 0.72])
    cbar = fig.colorbar(p, cax=cax)
    cbar.set_label(ERR_LABEL)
    cbar.outline.set_linewidth(0.6)
    return _save(fig, out, tight_bbox=False)


# =============================================================================
# 4. parity plot
# =============================================================================

def plot_parity(df, width="single", out=None, inset_residual=True):
    """C_L_rom vs C_L_fom against y=x, coloured by branch."""
    set_style()
    w = resolve_width(width)
    fig, ax = plt.subplots(figsize=(w, w))

    lo = min(df["C_L_fom"].min(), df["C_L_rom"].min())
    hi = max(df["C_L_fom"].max(), df["C_L_rom"].max())
    pad = 0.06 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
            color="0.65", lw=0.8, zorder=0)

    for label, sub, col in _iter_branches(df):
        ax.scatter(sub["C_L_fom"], sub["C_L_rom"], s=10, color=col,
                   edgecolors="none", alpha=0.85, label=label)

    ax.set_xlabel(r"$C_L^{\mathrm{FOM}}$")
    ax.set_ylabel(r"$C_L^{\mathrm{ROM}}$")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", markerscale=1.6, handletextpad=0.3,
              borderpad=0.3)
    fig.tight_layout()
    return _save(fig, out)


# =============================================================================
# 5. bifurcation slice
# =============================================================================

def plot_bifurcation(df, amp_value, width="single", atol=1e-8, out=None):
    """C_L vs Re at fixed amplitude: FOM open markers, ROM filled.

    Two-part legend -- colour encodes the branch, marker style encodes
    FOM vs ROM. Avoids the 2*n_branches legend entries of the naive version.
    """
    set_style()
    w = resolve_width(width)
    fig, ax = plt.subplots(figsize=(w, w * 0.75))

    sub = df[np.isclose(df["amp"], amp_value, atol=atol)]
    if sub.empty:
        raise ValueError(f"no rows near amp={amp_value}; available: "
                         f"{np.unique(np.round(df['amp'], 4))}")

    for label, bsub, col in _iter_branches(sub):
        bsub = bsub.sort_values("Re")
        ax.plot(bsub["Re"], bsub["C_L_fom"], "o", mfc="none", mec=col,
                mew=0.8, ms=5, ls="none")
        ax.plot(bsub["Re"], bsub["C_L_rom"], ".", color=col, ms=5, ls="none")

    branch_h = [Line2D([], [], color=CANON_COLORS[l], lw=2.0, label=l)
                for l, _, _ in _iter_branches(sub)]
    style_h = [Line2D([], [], color="0.35", marker="o", mfc="none",
                      ls="none", ms=5, label="FOM"),
               Line2D([], [], color="0.35", marker=".", ls="none",
                      ms=6, label="ROM")]
    leg = ax.legend(handles=branch_h, loc="upper left", borderpad=0.3,
                    handletextpad=0.5, fontsize="small")
    ax.add_artist(leg)
    ax.legend(handles=style_h, loc="lower right", borderpad=0.3,
              handletextpad=0.4, fontsize="small")

    ax.set_xlabel(r"$Re$")
    ax.set_ylabel(r"$C_L$")
    ax.set_title(rf"$A={amp_value:g}$")
    fig.tight_layout()
    return _save(fig, out)


# =============================================================================
# 6. cluster partition map
# =============================================================================

def plot_cluster_map(df, width="single", out=None):
    """(Re, amp) coloured by cluster id. Legend sits outside the data area."""
    set_style()
    w = resolve_width(width)
    fig, ax = plt.subplots(figsize=(w, w * 0.8))

    cids = sorted(df["cluster"].unique())
    for k, cid in enumerate(cids):
        sub = df[df["cluster"] == cid]
        ax.scatter(sub["Re"], sub["amp"], s=9, edgecolors="none",
                   color=CLUSTER_COLORS[k % len(CLUSTER_COLORS)],
                   label=f"{cid}")

    ax.set_xlabel(r"$Re$")
    ax.set_ylabel(r"amplitude $A$")
    ncol = min(len(cids), 4)
    ax.legend(title="cluster", loc="upper center", bbox_to_anchor=(0.5, -0.22),
              ncol=ncol, fontsize="small", handletextpad=0.3,
              columnspacing=1.0, markerscale=1.8)
    fig.tight_layout()
    return _save(fig, out)


# =============================================================================
# 7. error summary table  (booktabs, written by hand -- no jinja2 needed)
# =============================================================================

_TABLE_SPEC = [
    # (column key, header, formatter)
    ("N",           r"$N$",                        "{:d}".format),
    ("conv_rate",   r"conv.",                      "{:.2f}".format),
    ("mean_err_u",  r"mean $\varepsilon_u$",       "{:.2e}".format),
    ("max_err_u",   r"max $\varepsilon_u$",        "{:.2e}".format),
    ("mean_err_CL", r"mean $\varepsilon_{C_L}$",   "{:.2e}".format),
    ("max_err_CL",  r"max $\varepsilon_{C_L}$",    "{:.2e}".format),
    ("mean_t_rom",  r"$\bar{t}$ [s]",              "{:.3f}".format),
    ("mean_speedup", r"speed-up",                  "{:.0f}".format),
]


def _sci(s):
    """1.23e-04 -> 1.23\\times 10^{-4} in math mode."""
    mant, exp = s.split("e")
    return rf"${mant}\times 10^{{{int(exp)}}}$"


def error_table(results, out=None, t_fom_col=None, caption=None, label=None):
    """Per-branch error/convergence summary; DataFrame + optional booktabs .tex.

    `results` is {method_name: df}. Pass both local and global ROM and they
    appear as two row groups in one table:
        error_table({"Local ROM": local_df, "Global ROM": global_df})

    Speed-up needs a per-point FOM wall-time; your CSV has t_rom only, so add a
    t_fom column to the sweep and pass t_fom_col="t_fom". Without it the
    speed-up column is simply omitted.
    """
    rows = []
    for method, df in results.items():
        for label_b, sub, _ in _iter_branches(df):
            row = {
                "method": method,
                "branch": label_b,
                "N": len(sub),
                "conv_rate": float(sub["converged"].mean())
                             if "converged" in sub else np.nan,
                "mean_err_u": sub["err_u"].mean(),
                "max_err_u": sub["err_u"].max(),
                "mean_err_CL": sub["err_cl"].mean(),
                "max_err_CL": sub["err_cl"].max(),
                "mean_t_rom": sub["t_rom"].mean(),
            }
            if t_fom_col and t_fom_col in sub:
                row["mean_speedup"] = (sub[t_fom_col] / sub["t_rom"]).mean()
            rows.append(row)
    table = pd.DataFrame(rows)

    if out is None:
        return table

    cols = [(k, h, f) for k, h, f in _TABLE_SPEC if k in table.columns]
    ncol = len(cols)

    L = [r"\begin{table}[tb]", r"  \centering"]
    if caption:
        L.append(rf"  \caption{{{caption}}}")
    if label:
        L.append(rf"  \label{{{label}}}")
    L += [rf"  \begin{{tabular}}{{ll{'r' * ncol}}}",
          r"    \toprule",
          "    method & branch & " + " & ".join(h for _, h, _ in cols)
          + r" \\",
          r"    \midrule"]

    for method, grp in table.groupby("method", sort=False):
        for i, (_, r) in enumerate(grp.iterrows()):
            cells = []
            for k, _, f in cols:
                v = r[k]
                s = f(int(v) if k == "N" else v)
                cells.append(_sci(s) if "e" in s and k != "N" else s)
            head = rf"\multirow{{{len(grp)}}}{{*}}{{{method}}}" if i == 0 else ""
            L.append(f"    {head} & {r['branch']} & " + " & ".join(cells)
                     + r" \\")
        L.append(r"    \midrule" if method != table["method"].iloc[-1]
                 else r"    \bottomrule")

    L += [r"  \end{tabular}", r"\end{table}"]

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print("wrote", out, "(needs \\usepackage{booktabs,multirow})")
    return table


# =============================================================================
# driver
# =============================================================================

def make_all(csv_path, outdir="figures", width="single", amp_for_bif=None):
    outdir = Path(outdir)
    df = load_sweep(csv_path)
    unique_branches(df)

    # one shared colour scale so the manifold and the facets are comparable
    nrm = error_norm(df)

    # 3D and faceted figures want the full text width
    plot_error_3d(df, out=outdir / "err3d.pdf")
    plot_solution_manifold_3d(df, as_lines=True, norm=nrm,
                              out=outdir / "manifold3d.pdf")
    plot_error_map(df, norm=nrm, out=outdir / "errmap.pdf")

    plot_parity(df, width=width, out=outdir / "parity.pdf")
    plot_cluster_map(df, width=width, out=outdir / "clusters.pdf")

    if amp_for_bif is None:
        amps = np.unique(df["amp"])
        amp_for_bif = float(amps[np.argmin(np.abs(amps - np.median(amps)))])
    plot_bifurcation(df, amp_for_bif, width=width,
                     out=outdir / "bifurcation.pdf")

    tab = error_table({"Local ROM": df}, out=outdir / "error_table.tex",
                      caption="Local-ROM accuracy and cost, per branch.",
                      label="tab:local_rom_error")
    print(tab.to_string(index=False))
    plt.close("all")
    return df, tab


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--width", default="single")
    ap.add_argument("--amp", type=float, default=None)
    a = ap.parse_args()
    make_all(a.csv, outdir=a.outdir, width=a.width, amp_for_bif=a.amp)
