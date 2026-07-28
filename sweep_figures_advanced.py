"""
Candidate designs for the manifold figure -- render all, pick one.

Every candidate shows the same information (the solution manifold in
(Re, amp, C_L), coloured by ROM error) but makes a different trade between
structural readability and quantitative readability. Run compare_all() to
produce the whole set, then keep the winner and delete the rest.

    from nsrom.plotting.sweep_figures import load_sweep
    from nsrom.plotting.sweep_figures_advanced import compare_all
    df = load_sweep("local_rom/K4_H1/sweep_results.csv")
    compare_all(df, outdir="figures/manifold_candidates")

All candidates share the colour scale via error_norm(), so they are directly
comparable to each other and to the faceted map in sweep_figures.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from nsrom.plotting.style import (
    set_style, resolve_width, C_POS, C_NEG, C_SYM, CLUSTER_COLORS,
)
from sweep_figures import (
    SYM, ASYM1, ASYM2, CANON_COLORS, CANON_ORDER,
    ERR_LABEL, error_norm, _iter_branches, _finish_3d, _cbar_3d, _save,
)

CMAP = "viridis"


def robust_norm(df, err_col="err_u", plo=2, phi=98):
    """LogNorm clipped to a percentile range instead of min/max.

    Why you usually want this: a sweep that spans ~6 decades typically has most
    of its points inside 2 of them, with the rest stretched thin over outliers
    near the bifurcation and in diverged corners. Scaling to the full range
    then spends the whole colormap on a handful of points and renders the bulk
    of the manifold in one flat colour.

    Clipping to p2-p98 restores contrast where the data actually lives.
    Everything outside saturates at the end colours, so nothing is hidden --
    it is just no longer allowed to set the scale. State the clipping in the
    caption if you use it.
    """
    v = df[err_col].to_numpy(dtype=float)
    pos = v[np.isfinite(v) & (v > 0)]
    if pos.size == 0:
        raise ValueError(f"{err_col} has no positive finite values")
    lo, hi = np.percentile(pos, [plo, phi])
    return LogNorm(vmin=max(lo, pos.min()), vmax=min(hi, pos.max()))


# =============================================================================
# shared helpers
# =============================================================================

def _paths(df, sort_by="Re"):
    """Yield (raw_branch_label, path_df) for each continuation path.

    Grouping on the RAW branch column keeps each continuation trajectory
    intact, which is what you want when drawing lines -- grouping on
    branch_base would connect points from different paths and produce
    zig-zags across the sheet.
    """
    for label, path in df.groupby("branch", sort=False):
        yield label, path.sort_values(sort_by)


def _tri_mask(x, y, tri, frac=2.5):
    """Mask triangles whose longest edge is an outlier.

    Delaunay over scattered (Re, amp) points will happily bridge the gap where
    an asymmetric sheet terminates at the critical curve, inventing surface
    that does not exist. Dropping long-edged triangles removes those bridges.
    Raise frac to fill more, lower it to be more conservative.
    """
    t = tri.triangles
    xt, yt = x[t], y[t]
    # longest edge of each triangle, in normalised coordinates
    sx = np.ptp(x) or 1.0
    sy = np.ptp(y) or 1.0
    e = np.stack([
        np.hypot((xt[:, i] - xt[:, j]) / sx, (yt[:, i] - yt[:, j]) / sy)
        for i, j in ((0, 1), (1, 2), (2, 0))
    ])
    longest = e.max(axis=0)
    return longest > frac * np.median(longest)


def _branch_legend(fig, labels=None, **kw):
    labels = labels or CANON_ORDER
    h = [Line2D([], [], color=CANON_COLORS[l], lw=2.2, label=l)
         for l in labels]
    return fig.legend(handles=h, **kw)


# =============================================================================
# A. ribbons -- continuation paths as error-coloured lines
# =============================================================================

def manifold_ribbons(df, err_col="err_u", width="linewidth", norm=None,
                     elev=22, azim=-58, lw=2.2, color_by="branch", out=None):
    """Each continuation path drawn as a thick line.

    color_by="branch" (default) -- each sheet in its own palette colour, with
        a three-entry legend and no colourbar. This is the figure that shows
        the STRUCTURE of the solution manifold: where the pitchfork opens,
        which sheets coexist, how they fold. Nothing competes with that
        reading. Pair it with a separate error figure (errmap / composite)
        rather than trying to make one figure do both jobs.

    color_by="error" -- colour by err_col under a log norm, as before.
    """
    set_style()
    w = resolve_width(width)

    fig = plt.figure(figsize=(w, w * 0.62))
    # room for the labels a 3D axes draws outside its own rect (see composite)
    rect = [0.02, 0.10, 0.62, 0.84] if color_by == "error" else \
           [0.04, 0.10, 0.86, 0.84]
    ax = fig.add_axes(rect, projection="3d")

    if color_by == "branch":
        # one Line3DCollection per branch: fast, and gives a clean legend
        for label, sub, col in _iter_branches(df):
            segs = []
            for _, path in _paths(sub):
                p = path[["Re", "amp", "C_L_fom"]].to_numpy(dtype=float)
                if len(p) >= 2:
                    segs.append(np.stack([p[:-1], p[1:]], axis=1))
            if not segs:
                continue
            ax.add_collection3d(Line3DCollection(
                np.concatenate(segs), colors=col, linewidths=lw,
                capstyle="round"))
        mappable = None
    elif color_by == "error":
        nrm = norm or error_norm(df, err_col)
        segs, vals = [], []
        for _, path in _paths(df):
            p = path[["Re", "amp", "C_L_fom"]].to_numpy(dtype=float)
            if len(p) < 2:
                continue
            e = path[err_col].to_numpy(dtype=float)
            segs.append(np.stack([p[:-1], p[1:]], axis=1))
            vals.append(0.5 * (e[:-1] + e[1:]))
        mappable = Line3DCollection(np.concatenate(segs), cmap=CMAP, norm=nrm,
                                    linewidths=lw, capstyle="round")
        mappable.set_array(np.concatenate(vals))
        ax.add_collection3d(mappable)
    else:
        raise ValueError("color_by must be 'branch' or 'error'")

    ax.set_xlim(df["Re"].min(), df["Re"].max())
    ax.set_ylim(df["amp"].min(), df["amp"].max())
    ax.set_zlim(df["C_L_fom"].min(), df["C_L_fom"].max())
    ax.set_xlabel(r"$Re$")
    ax.set_ylabel(r"amplitude $A$")
    ax.set_zlabel(r"$C_L$")
    ax.view_init(elev=elev, azim=azim)
    _finish_3d(fig, ax)
    ax.xaxis.labelpad = ax.yaxis.labelpad = 6
    ax.zaxis.labelpad = 6

    if color_by == "branch":
        present = [l for l, _, _ in _iter_branches(df)]
        _branch_legend(fig, present, loc="upper right",
                       bbox_to_anchor=(0.99, 0.95), handlelength=2.2,
                       handletextpad=0.6, borderpad=0.4)
    else:
        _cbar_3d(fig, mappable, ERR_LABEL, rect=(0.80, 0.24, 0.022, 0.50))
    return _save(fig, out, tight_bbox=False)


# =============================================================================
# B. surface -- each sheet triangulated into a real surface
# =============================================================================

def manifold_surface(df, err_col="err_u", width="linewidth", norm=None,
                     elev=22, azim=-58, frac=2.5, alpha=0.95, out=None):
    """Each branch triangulated in (Re, amp) and lifted to C_L.

    The most literally "manifold" reading: three sheets, not dots. Facecolour
    is the per-triangle mean error. Long-edged triangles are masked so the
    asymmetric sheets terminate at the critical curve instead of being bridged
    across it -- tune `frac` if the sheets look over- or under-filled.

    Caveat for the paper: this INTERPOLATES between sampled points. It is a
    visualisation of the sheet, not raw data. If a referee is likely to read
    the surface as computed rather than interpolated, prefer ribbons.
    """
    set_style()
    w = resolve_width(width)
    nrm = norm or error_norm(df, err_col)

    fig = plt.figure(figsize=(w, w * 0.62))
    ax = fig.add_subplot(111, projection="3d")

    for label, sub, _ in _iter_branches(df):
        x = sub["Re"].to_numpy(float)
        y = sub["amp"].to_numpy(float)
        z = sub["C_L_fom"].to_numpy(float)
        e = sub[err_col].to_numpy(float)
        if len(x) < 3:
            continue
        try:
            tri = mtri.Triangulation(x, y)
        except (ValueError, RuntimeError):
            continue                     # degenerate (e.g. collinear) sheet
        tri.set_mask(_tri_mask(x, y, tri, frac=frac))

        s = ax.plot_trisurf(tri, z, cmap=CMAP, norm=nrm,
                            linewidth=0, antialiased=True, alpha=alpha)
        # colour each facet by its mean error rather than by height
        s.set_array(e[tri.triangles[~tri.mask]].mean(axis=1))

    ax.set_xlabel(r"$Re$")
    ax.set_ylabel(r"amplitude $A$")
    ax.set_zlabel(r"$C_L$")
    ax.view_init(elev=elev, azim=azim)
    _finish_3d(fig, ax)

    sm = plt.cm.ScalarMappable(cmap=CMAP, norm=nrm)
    sm.set_array([])
    _cbar_3d(fig, sm, ERR_LABEL)
    fig.subplots_adjust(left=0.0, right=0.83, bottom=0.04, top=0.98)
    return _save(fig, out, tight_bbox=False)


# =============================================================================
# C. manifold + floor projection
# =============================================================================

def manifold_with_floor(df, err_col="err_u", width="linewidth", norm=None,
                        elev=24, azim=-58, out=None):
    """Ribbons above, error shadow projected flat on the (Re, amp) floor.

    The floor projection recovers the thing 3D costs you: an undistorted view
    of where in parameter space the error lives. One figure, both readings.
    """
    set_style()
    w = resolve_width(width)
    nrm = norm or error_norm(df, err_col)

    fig = plt.figure(figsize=(w, w * 0.68))
    ax = fig.add_subplot(111, projection="3d")

    zmin = df["C_L_fom"].min()
    zspan = np.ptp(df["C_L_fom"]) or 1.0
    floor = zmin - 0.28 * zspan

    # shadow first so the manifold draws over it
    ax.scatter(df["Re"], df["amp"], np.full(len(df), floor),
               c=df[err_col], cmap=CMAP, norm=nrm, s=4,
               depthshade=False, edgecolors="none", alpha=0.75)

    segs, vals = [], []
    for _, path in _paths(df):
        p = path[["Re", "amp", "C_L_fom"]].to_numpy(dtype=float)
        if len(p) < 2:
            continue
        e = path[err_col].to_numpy(dtype=float)
        segs.append(np.stack([p[:-1], p[1:]], axis=1))
        vals.append(0.5 * (e[:-1] + e[1:]))
    lc = Line3DCollection(np.concatenate(segs), cmap=CMAP, norm=nrm,
                          linewidths=1.8, capstyle="round")
    lc.set_array(np.concatenate(vals))
    ax.add_collection3d(lc)

    ax.set_xlim(df["Re"].min(), df["Re"].max())
    ax.set_ylim(df["amp"].min(), df["amp"].max())
    ax.set_zlim(floor, df["C_L_fom"].max())
    ax.set_xlabel(r"$Re$")
    ax.set_ylabel(r"amplitude $A$")
    ax.set_zlabel(r"$C_L$")
    ax.view_init(elev=elev, azim=azim)
    _finish_3d(fig, ax, box_aspect=(1, 1, 0.72))
    _cbar_3d(fig, lc, ERR_LABEL)
    fig.subplots_adjust(left=0.0, right=0.83, bottom=0.04, top=0.98)
    return _save(fig, out, tight_bbox=False)


# =============================================================================
# D. multi-view triptych
# =============================================================================

def manifold_multiview(df, err_col="err_u", width="linewidth", norm=None,
                       views=((22, -58), (18, -110), (78, -90)), out=None):
    """The same manifold from several angles -- kills depth ambiguity.

    The third default view is near top-down, which reads as the (Re, amp)
    plane and shows the branch footprint. Costs vertical space; earns it if
    reviewers keep misreading the single view.
    """
    set_style()
    w = resolve_width(width)
    nrm = norm or error_norm(df, err_col)
    n = len(views)

    fig = plt.figure(figsize=(w, w / n * 1.02))
    for k, (elev, azim) in enumerate(views, start=1):
        ax = fig.add_subplot(1, n, k, projection="3d")
        segs, vals = [], []
        for _, path in _paths(df):
            p = path[["Re", "amp", "C_L_fom"]].to_numpy(dtype=float)
            if len(p) < 2:
                continue
            e = path[err_col].to_numpy(dtype=float)
            segs.append(np.stack([p[:-1], p[1:]], axis=1))
            vals.append(0.5 * (e[:-1] + e[1:]))
        lc = Line3DCollection(np.concatenate(segs), cmap=CMAP, norm=nrm,
                              linewidths=1.4, capstyle="round")
        lc.set_array(np.concatenate(vals))
        ax.add_collection3d(lc)
        ax.set_xlim(df["Re"].min(), df["Re"].max())
        ax.set_ylim(df["amp"].min(), df["amp"].max())
        ax.set_zlim(df["C_L_fom"].min(), df["C_L_fom"].max())
        ax.set_xlabel(r"$Re$", labelpad=-2)
        ax.set_ylabel(r"$A$", labelpad=-2)
        ax.set_zlabel(r"$C_L$", labelpad=-4)
        ax.view_init(elev=elev, azim=azim)
        _finish_3d(fig, ax, nticks=3)
        ax.tick_params(labelsize=6, pad=0)

    fig.subplots_adjust(left=0.01, right=0.88, bottom=0.02, top=0.98,
                        wspace=0.02)
    cax = fig.add_axes([0.90, 0.20, 0.016, 0.60])
    cb = fig.colorbar(lc, cax=cax)
    cb.set_label(ERR_LABEL)
    cb.outline.set_linewidth(0.6)
    return _save(fig, out, tight_bbox=False)


# =============================================================================
# E. 2D carpet -- pitchfork slices, no 3D at all
# =============================================================================

def manifold_carpet(df, err_col="err_u", width="linewidth", norm=None,
                    n_amp=6, ncols=3, out=None):
    """Small multiples: C_L vs Re at selected amplitudes, coloured by error.

    This is the 3D manifold cut into readable slices. No depth ambiguity, no
    occlusion, and every point sits at a position the reader can read off both
    axes. For a two-column journal this is usually the most defensible choice
    -- the 3D view is prettier, this one is checkable.
    """
    set_style()
    w = resolve_width(width)
    nrm = norm or error_norm(df, err_col)

    amps = np.unique(np.round(df["amp"].to_numpy(), 6))
    if len(amps) > n_amp:                      # evenly spaced subset
        amps = amps[np.linspace(0, len(amps) - 1, n_amp).astype(int)]
    nrows = int(np.ceil(len(amps) / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(w, w / ncols * nrows * 0.85),
                             sharex=True, sharey=True, squeeze=False)
    axes = axes.ravel()

    tol = 0.02 * (np.ptp(df["amp"]) or 1.0)
    for ax, a in zip(axes, amps):
        sub = df[np.abs(df["amp"] - a) <= tol]
        for _, path in _paths(sub):
            ax.plot(path["Re"], path["C_L_fom"], color="0.85", lw=0.6, zorder=1)
        p = ax.scatter(sub["Re"], sub["C_L_fom"], c=sub[err_col], cmap=CMAP,
                       norm=nrm, s=5, edgecolors="none", zorder=2)
        ax.set_title(rf"$A={a:.2f}$", pad=3)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
    for ax in axes[len(amps):]:
        ax.set_visible(False)
    for ax in axes[len(amps) - ncols:len(amps)]:
        ax.set_xlabel(r"$Re$")
    for k in range(0, len(amps), ncols):
        axes[k].set_ylabel(r"$C_L$")

    fig.subplots_adjust(left=0.09, right=0.87, bottom=0.10, top=0.94,
                        wspace=0.10, hspace=0.30)
    cax = fig.add_axes([0.89, 0.10, 0.016, 0.84])
    cb = fig.colorbar(p, cax=cax)
    cb.set_label(ERR_LABEL)
    cb.outline.set_linewidth(0.6)
    return _save(fig, out, tight_bbox=False)


# =============================================================================
# F. composite hero -- 3D structure + 2D quantitative panel
# =============================================================================

def manifold_composite(df, err_col="err_u", width="linewidth", norm=None,
                       elev=22, azim=-58, band=(10, 90), out=None):
    """Left: 3D ribbons (structure). Right: error vs Re per branch (numbers).

    Layout notes:
    - The colourbar sits ABOVE the 3D panel, horizontal, with its label on top
      and ticks underneath. A bar below or beside the 3D axes lands on the Re
      / amplitude labels, because a 3D axes draws its labels well outside its
      nominal rectangle.
    - Mirror branches (asym 1 / asym 2) carry near-identical error, so on the
      right panel one line hides the other exactly. They are drawn with
      different dash patterns and decreasing width so both stay visible.
    - `band` is a percentile pair for the spread envelope. Min-max (0, 100) is
      extremely spiky on real sweeps; (10, 90) shows the same story calmly.
      Pass band=None for a median line only.
    """
    set_style()
    w = resolve_width(width)
    nrm = norm or error_norm(df, err_col)

    fig = plt.figure(figsize=(w, w * 0.46))
    # A 3D axes draws its axis labels well OUTSIDE its own rectangle, and this
    # figure is saved at a fixed size (no tight bbox, so the paper gets the
    # exact width style.py asks for). The rect therefore has to leave room:
    # left>0 and a generous bottom margin, or Re / amplitude get cut off at
    # the canvas edge. Verified with a clipping check -- if you enlarge the
    # panel, re-check.
    ax3 = fig.add_axes([0.02, 0.13, 0.40, 0.70], projection="3d")
    ax2 = fig.add_axes([0.62, 0.17, 0.35, 0.72])

    # --- left: ribbons
    segs, vals = [], []
    for _, path in _paths(df):
        p = path[["Re", "amp", "C_L_fom"]].to_numpy(dtype=float)
        if len(p) < 2:
            continue
        e = path[err_col].to_numpy(dtype=float)
        segs.append(np.stack([p[:-1], p[1:]], axis=1))
        vals.append(0.5 * (e[:-1] + e[1:]))
    lc = Line3DCollection(np.concatenate(segs), cmap=CMAP, norm=nrm,
                          linewidths=1.5, capstyle="round")
    lc.set_array(np.concatenate(vals))
    ax3.add_collection3d(lc)
    ax3.set_xlim(df["Re"].min(), df["Re"].max())
    ax3.set_ylim(df["amp"].min(), df["amp"].max())
    ax3.set_zlim(df["C_L_fom"].min(), df["C_L_fom"].max())
    ax3.set_xlabel(r"$Re$")
    ax3.set_ylabel(r"amplitude $A$")
    ax3.set_zlabel(r"$C_L$")
    ax3.view_init(elev=elev, azim=azim)
    _finish_3d(fig, ax3, nticks=4)
    ax3.xaxis.labelpad = ax3.yaxis.labelpad = 6   # tighter than the default
    ax3.zaxis.labelpad = 6

    # --- colourbar ABOVE the 3D panel, label on top, ticks below the bar
    cax = fig.add_axes([0.06, 0.88, 0.32, 0.030])
    cb = fig.colorbar(lc, cax=cax, orientation="horizontal", extend="both")
    cb.set_label(ERR_LABEL, labelpad=4)
    cb.ax.xaxis.set_label_position("top")
    cb.ax.xaxis.set_ticks_position("bottom")
    cb.ax.tick_params(labelsize=6, width=0.5, length=2, pad=1)
    cb.outline.set_linewidth(0.5)

    # --- right: error envelope per branch
    dashes = {SYM: (None, None), ASYM1: (None, None), ASYM2: (3.2, 1.6)}
    widths = {SYM: 1.6, ASYM1: 1.5, ASYM2: 1.1}
    for label, sub, col in _iter_branches(df):
        g = sub.groupby(np.round(sub["Re"], 3))[err_col]
        re = g.median().index.to_numpy(float)
        med = g.median().to_numpy()
        o = np.argsort(re)
        if band is not None:
            lo = g.quantile(band[0] / 100).to_numpy()
            hi = g.quantile(band[1] / 100).to_numpy()
            ax2.fill_between(re[o], lo[o], hi[o], color=col, alpha=0.16, lw=0)
        ln, = ax2.plot(re[o], med[o], color=col, lw=widths[label], label=label)
        if dashes[label][0] is not None:
            ln.set_dashes(dashes[label])

    ax2.set_yscale("log")
    ax2.set_xlabel(r"$Re$")
    ax2.set_ylabel(ERR_LABEL)
    ax2.legend(loc="best", fontsize="small", handletextpad=0.6,
               borderpad=0.3, handlelength=2.6)
    if band is not None:
        ax2.text(0.02, 0.02, rf"median, p{band[0]}--p{band[1]}",
                 transform=ax2.transAxes, fontsize=6, color="0.4")
    return _save(fig, out, tight_bbox=False)


# =============================================================================
# driver
# =============================================================================

CANDIDATES = {
    "A_ribbons":    manifold_ribbons,
    "B_surface":    manifold_surface,
    "C_floor":      manifold_with_floor,
    "D_multiview":  manifold_multiview,
    "E_carpet":     manifold_carpet,
    "F_composite":  manifold_composite,
}


def compare_all(df, outdir="figures/manifold_candidates", err_col="err_u",
                ext="pdf", norm=None):
    """Render every candidate with a shared colour scale."""
    outdir = Path(outdir)
    nrm = norm or error_norm(df, err_col)
    made = {}
    for name, fn in CANDIDATES.items():
        try:
            fn(df, err_col=err_col, norm=nrm, out=outdir / f"{name}.{ext}")
            made[name] = "ok"
        except Exception as exc:                       # keep going
            made[name] = f"FAILED: {type(exc).__name__}: {exc}"
            print(f"  !! {name}: {made[name]}")
        plt.close("all")
    return made


if __name__ == "__main__":
    import argparse
    from sweep_figures import load_sweep

    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--outdir", default="figures/manifold_candidates")
    ap.add_argument("--ext", default="pdf")
    a = ap.parse_args()
    compare_all(load_sweep(a.csv), outdir=a.outdir, ext=a.ext)
