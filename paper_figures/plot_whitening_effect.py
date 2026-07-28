#!/usr/bin/env python3
"""
plot_whitening_effect.py
========================

ONE publication figure showing what Mahalanobis whitening does, with no
statistics reported: the snapshot cloud in coefficient space, BEFORE and AFTER
whitening, projected onto the plane of

    x = the dominant-energy POD mode        (default: mode 1)
    y = the branch-discriminating POD mode  (default: chosen by Fisher ratio)

coloured by branch, drawn with EQUAL ASPECT.

Why equal aspect matters
------------------------
k-means clusters by plain Euclidean distance in coefficient space, and
whitening only rescales each axis to unit variance. Drawing the two axes on a
common scale is therefore the faithful picture: in the "before" panel the
low-energy discriminating mode is a thin COMPRESSED band -- the branches are
nearly coincident in raw distance, so k-means cannot separate them. In the
"after" panel that axis is stretched to unit variance and the branches open
into SEPARATED clusters. If either axis were autoscaled independently the
effect would be hidden, so equal aspect is enforced here and must not be
removed.

The y-mode is picked by the Fisher ratio purely to choose which plane to draw;
that number is never displayed. Styling comes from nsrom.plotting.style.

Output
------
    figures/whitening_effect.pdf

Usage
-----
    python plot_whitening_effect.py                 # draft (mathtext)
    python plot_whitening_effect.py --usetex        # final (LaTeX fonts)
    python plot_whitening_effect.py --x-mode 1 --y-mode 6 --width double
"""

import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

from nsrom.navier_stokes import setup_navier_stokes_problem
from nsrom.snapshot_collection import load_snapshot_dofs
from nsrom.lifting_functions import load_lifting_functions
from nsrom.cluster_building import (
    build_energy_snapshots,
    get_pod_coefficients,
    mahalanobis_whitening,
)

from nsrom.plotting.style import (
    set_style, resolve_width, BASE_FONTSIZE,
    C_POS, C_NEG, C_SYM, C_GLOBAL, CLUSTER_COLORS,
)


# =============================================================================
# CONFIGURATION  (mirrors main_local.py)
# =============================================================================

SNAPSHOT_DIR       = "multi_param_multi_branch"
LIFTING_DIR        = "lifting"
MESH_FILE          = "mesh/mid_pinball.msh"
REYNOLDS_INIT      = 100.0
AMPLITUDE_INIT     = 0.0
INNER_PRODUCT_TYPE = "H1"

FIG_DIR      = "figures"
RANK         = 8             # POD modes computed / whitened (search space)
X_MODE       = 1             # 1-based; dominant-energy mode on the x-axis
Y_MODE       = None          # 1-based; None -> auto-pick by Fisher ratio
LIFT_TOL_REL = 0.02          # unforced |C_L| < this*max|C_L| -> symmetric branch
BRANCH_LABELS = {}           # optional {branch_id: name}, else onset-sign naming
DRAW_ELLIPSES = True         # 1-sigma per-branch ellipse to guide the eye


# =============================================================================
# DATA
# =============================================================================

def compute_coefficients(rank):
    problem = setup_navier_stokes_problem(MESH_FILE, REYNOLDS_INIT, AMPLITUDE_INIT)
    snap    = load_snapshot_dofs(SNAPSHOT_DIR)
    lifting = load_lifting_functions(LIFTING_DIR, problem)

    S_homo, _S_tilde, M_u, _M_p, _L, _P = build_energy_snapshots(
        snap["velocity"], snap["parameters"], lifting, problem,
        INNER_PRODUCT_TYPE, homogenize=True,
    )
    rank = min(rank, S_homo.shape[1])
    A, eigenvalues, _modes = get_pod_coefficients(S_homo, rank=rank, M_u=M_u)
    A_white = mahalanobis_whitening(A, eigenvalues, rank)
    return A, A_white, eigenvalues, snap


# =============================================================================
# MODE SELECTION (Fisher ratio, used ONLY to choose the plane)
# =============================================================================

def fisher_ratio(a, labels):
    """Between-class / within-class variance of a 1-D coefficient row."""
    grand = a.mean()
    between = within = 0.0
    for c in np.unique(labels):
        m = labels == c
        if np.sum(m) < 2:
            continue
        mu = a[m].mean()
        between += np.sum(m) * (mu - grand) ** 2
        within  += np.sum((a[m] - mu) ** 2)
    return between / (within + 1e-30)


def pick_y_mode(A, snap, rank, x_mode0):
    """Return a 0-based mode index that best separates the branches."""
    branch_ids = snap.get("branch_ids")
    if branch_ids is None:
        return 1 if x_mode0 == 0 else 0
    branch_ids = np.asarray(branch_ids).ravel()
    scores = np.array([fisher_ratio(A[i], branch_ids) if i != x_mode0 else -np.inf
                       for i in range(rank)])
    return int(np.argmax(scores))


# =============================================================================
# BRANCH GROUPING  (colour by branch; sign from the UNFORCED representative)
# =============================================================================

def resolve_branch_groups(snap, n_snap):
    lift = snap.get("lift_coefficients")
    lift = np.asarray(lift, float).ravel() if lift is not None else None
    if lift is not None and lift.size != n_snap:
        lift = None

    branch_ids = snap.get("branch_ids")
    if branch_ids is None:
        return [("snapshots", np.ones(n_snap, bool), C_GLOBAL)]
    branch_ids = np.asarray(branch_ids).ravel()

    params = np.asarray(snap.get("parameters"), float) if snap.get("parameters") is not None else None
    amp = params[:, 1] if params is not None and params.ndim == 2 else None
    tol = LIFT_TOL_REL * (np.max(np.abs(lift)) if lift is not None else 1.0)

    used, groups = set(), []
    for bid in np.unique(branch_ids):
        mask = branch_ids == bid
        color = label = None
        key = float(bid)
        if lift is not None:
            idx = np.where(mask)[0]
            rep = idx[np.argmin(np.abs(amp[idx]))] if amp is not None else idx[0]
            s = float(lift[rep]); key = s
            if   s >  tol: cand_c, cand_l = C_POS, r"$C_L>0$ branch"
            elif s < -tol: cand_c, cand_l = C_NEG, r"$C_L<0$ branch"
            else:          cand_c, cand_l = C_SYM, "symmetric branch"
            if cand_c not in used:
                color, label = cand_c, cand_l; used.add(cand_c)
        if color is None:
            for c in CLUSTER_COLORS:
                if c not in used:
                    color = c; used.add(c); break
            if label is None:
                label = f"branch {int(bid)}"
        label = BRANCH_LABELS.get(int(bid), label)
        groups.append((label, mask, color, key))

    groups.sort(key=lambda g: g[3], reverse=True)
    return [(l, m, c) for (l, m, c, _k) in groups]


# =============================================================================
# FIGURE
# =============================================================================

def _ellipse(ax, x, y, color, nsig=1.0):
    if x.size < 3:
        return
    cov = np.cov(np.vstack([x, y]))
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    ang = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    w, h = 2.0 * nsig * np.sqrt(np.maximum(vals, 0.0))
    ax.add_patch(Ellipse((x.mean(), y.mean()), w, h, angle=ang,
                         facecolor=color, edgecolor=color, lw=0.7, alpha=0.12, zorder=1))


def _panel(ax, Ax, Ay, groups, xlabel, ylabel, title):
    ax.axhline(0, color="0.85", lw=0.6, zorder=0)
    ax.axvline(0, color="0.85", lw=0.6, zorder=0)
    for label, mask, color in groups:
        if not np.any(mask):
            continue
        ax.scatter(Ax[mask], Ay[mask], s=12, c=color,
                   edgecolors="none", alpha=0.80, zorder=3, label=label)
        if DRAW_ELLIPSES:
            _ellipse(ax, Ax[mask], Ay[mask], color)
        ax.scatter([Ax[mask].mean()], [Ay[mask].mean()], s=48, marker="o",
                   facecolors="none", edgecolors=color, linewidths=1.1, zorder=4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=BASE_FONTSIZE + 1)
    ax.set_aspect("equal", adjustable="datalim")   # <-- the honest bit; do not remove
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))


def make_figure(A, A_white, groups, ix, iy, width, out):
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(width, 0.54 * width),
                                   constrained_layout=True)

    _panel(ax0, A[ix], A[iy], groups,
           rf"$a_{{{ix + 1}}}$", rf"$a_{{{iy + 1}}}$", "before whitening")
    _panel(ax1, A_white[ix], A_white[iy], groups,
           rf"$\tilde a_{{{ix + 1}}}$", rf"$\tilde a_{{{iy + 1}}}$", "after whitening")

    seen, handles, labels = set(), [], []
    for label, mask, color in groups:
        if not np.any(mask) or label in seen:
            continue
        seen.add(label)
        handles.append(Line2D([0], [0], marker="o", linestyle="none",
                              markersize=5, markerfacecolor=color, markeredgecolor="none"))
        labels.append(label)
    if len(labels) > 1:
        fig.legend(handles, labels, loc="upper center", ncol=len(labels),
                   bbox_to_anchor=(0.5, 1.04), frameon=False,
                   handletextpad=0.3, columnspacing=1.2)

    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}   (x=mode {ix + 1}, y=mode {iy + 1})")


# =============================================================================
# MAIN
# =============================================================================

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--usetex", action="store_true")
    p.add_argument("--width", default="double",
                   help="preset (single/double/linewidth/0.7linewidth) or inches")
    p.add_argument("--rank", type=int, default=RANK)
    p.add_argument("--x-mode", type=int, default=X_MODE, help="1-based x-axis mode")
    p.add_argument("--y-mode", type=int, default=Y_MODE,
                   help="1-based y-axis mode; omit to auto-pick by Fisher ratio")
    p.add_argument("--out-dir", default=FIG_DIR)
    args = p.parse_args()

    set_style(usetex=args.usetex)
    os.makedirs(args.out_dir, exist_ok=True)

    A, A_white, _eig, snap = compute_coefficients(args.rank)
    rank   = A.shape[0]
    n_snap = A.shape[1]

    ix = args.x_mode - 1
    if not (0 <= ix < rank):
        raise SystemExit(f"--x-mode must be in 1..{rank}")
    iy = (args.y_mode - 1) if args.y_mode is not None else pick_y_mode(A, snap, rank, ix)
    if not (0 <= iy < rank) or iy == ix:
        raise SystemExit(f"--y-mode must be in 1..{rank} and differ from --x-mode")

    groups = resolve_branch_groups(snap, n_snap)
    width  = resolve_width(args.width)
    make_figure(A, A_white, groups, ix, iy, width,
                os.path.join(args.out_dir, "whitening_effect.pdf"))


if __name__ == "__main__":
    main()
