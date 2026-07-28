#!/usr/bin/env python3
"""
plot_pod_coefficients.py
========================

Publication figures for the first N POD-coefficient modes over the snapshot
ensemble. Produces TWO figures, both written into the figures folder:

    figures/pod_coefficients.pdf            raw coefficients  a_i
    figures/pod_coefficients_whitened.pdf   whitened          a_i / sqrt(lambda_i)

Both reproduce the ``pod_whitened`` clustering route and stop at the
coefficient matrix (no k-means):

  1. homogenise the velocity snapshots (subtract the lifting),
  2. build the energy-norm POD basis on the homogenised snapshots,
  3. project the snapshots onto that basis            ->  A            (raw)
  4. divide each row by its per-mode std sqrt(lambda_i) ->  A_white    (whitened)

Step 3 is exactly ``nsrom.cluster_building.get_pod_coefficients`` and step 4 is
``nsrom.cluster_building.mahalanobis_whitening`` -- the same calls
``cluster_kmeans_pod_whitened`` makes before k-means. Raw rows carry the modal
energy (mode 1 large, mode 8 small); whitened rows each have unit variance, so
a low-energy branch-splitting mode shows up as clearly as a high-energy one.

Every visual choice (fonts, palette, dpi, embedded fonts) is inherited from
``nsrom.plotting.style``. POD modes are defined up to sign, so the sign of each
coefficient is arbitrary -- only its structure across the ensemble matters.

Usage
-----
    python plot_pod_coefficients.py                       # draft (mathtext)
    python plot_pod_coefficients.py --usetex              # final (LaTeX fonts)
    python plot_pod_coefficients.py --width double --xaxis Re --color-by branch
"""

import os
import math
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D

from nsrom.navier_stokes import setup_navier_stokes_problem
from nsrom.snapshot_collection import load_snapshot_dofs
from nsrom.lifting_functions import load_lifting_functions
from nsrom.cluster_building import (
    build_energy_snapshots,
    get_pod_coefficients,
    mahalanobis_whitening,
)

from nsrom.plotting.style import (
    set_style,
    resolve_width,
    BASE_FONTSIZE,
    C_POS, C_NEG, C_SYM, C_GLOBAL,
    CLUSTER_COLORS,
)


# =============================================================================
# CONFIGURATION  (mirrors main_local.py so this is drop-in)
# =============================================================================

SNAPSHOT_DIR       = "multi_param_multi_branch"
LIFTING_DIR        = "lifting"
MESH_FILE          = "mesh/mid_pinball.msh"
REYNOLDS_INIT      = 100.0
AMPLITUDE_INIT     = 0.0
INNER_PRODUCT_TYPE = "H1"      # energy norm used as the POD inner product

# --- figure ---
FIG_DIR      = "figures"     # all output goes here
N_MODES      = 8             # number of POD modes (panels) to plot
NCOLS        = 2             # columns in the panel grid; rows are derived
ROW_HEIGHT   = 1.5           # inches per panel row
COLOR_BY     = "branch"      # "branch" | "lift" | "none"
LIFT_TOL_REL = 0.02          # |C_L| < this * max|C_L|  ->  symmetric (near onset)

# Optional {branch_id: legend label}. If empty, each branch is named from its
# UNFORCED (amp~0) lift sign -> "$C_L>0$ branch" / "$C_L<0$ branch" / symmetric,
# matching the convention in style.py. Override here with your own physical
# names, e.g. {0: "symmetric", 1: "upper", 2: "lower"}.
BRANCH_LABELS = {}


# =============================================================================
# DATA:  snapshots -> POD basis -> coefficients (raw + whitened)
# =============================================================================

def compute_coefficients(n_modes):
    """Return (A, A_white, eigenvalues, snapshot_data), A* shape (rank, n_snap)."""
    problem = setup_navier_stokes_problem(
        mesh_file=MESH_FILE,
        reynolds_init=REYNOLDS_INIT,
        amplitude_init=AMPLITUDE_INIT,
    )
    snap    = load_snapshot_dofs(SNAPSHOT_DIR)
    lifting = load_lifting_functions(LIFTING_DIR, problem)

    S_homo, _S_tilde, M_u, _M_p, _L, _P = build_energy_snapshots(
        snap["velocity"], snap["parameters"], lifting, problem,
        INNER_PRODUCT_TYPE, homogenize=True,
    )

    rank = min(n_modes, S_homo.shape[1])
    A, eigenvalues, _modes = get_pod_coefficients(S_homo, rank=rank, M_u=M_u)
    A_white = mahalanobis_whitening(A, eigenvalues, rank)   # a_i / sqrt(lambda_i)
    return A, A_white, eigenvalues, snap


# =============================================================================
# BRANCH GROUPING (for colour)
# =============================================================================

def _lift_vector(snap, n_snap):
    lift = snap.get("lift_coefficients")
    if lift is None:
        return None
    lift = np.asarray(lift, dtype=float).ravel()
    return lift if lift.size == n_snap else None


def resolve_groups(snap, n_snap, color_by):
    """Return an ordered list of (label, mask, color) tuples for the legend."""
    lift = _lift_vector(snap, n_snap)

    if color_by == "none" or (color_by == "lift" and lift is None):
        return [("snapshots", np.ones(n_snap, bool), C_GLOBAL)]

    tol = LIFT_TOL_REL * (np.max(np.abs(lift)) if lift is not None else 1.0)

    if color_by == "lift":
        pos = lift > tol
        neg = lift < -tol
        sym = ~(pos | neg)
        groups = [(r"$C_L>0$", pos, C_POS),
                  ("symmetric", sym, C_SYM),
                  (r"$C_L<0$", neg, C_NEG)]
        return [g for g in groups if np.any(g[1])]

    # color_by == "branch": group by branch id (a continuation object). A branch
    # spans many amplitudes and its instantaneous C_L flips sign under forcing,
    # so we do NOT label/colour it by the mean or per-snapshot lift. Instead each
    # branch's sign is read from its UNFORCED representative -- the snapshot with
    # the smallest |amp| on that branch -- which is the branch's intrinsic
    # (onset) deflection and is stable. Colours follow style.py's branch palette;
    # labels are branch NAMES ("$C_L>0$ branch"), overridable via BRANCH_LABELS.
    branch_ids = snap.get("branch_ids")
    if branch_ids is None:
        return resolve_groups(snap, n_snap, "lift")

    branch_ids = np.asarray(branch_ids).ravel()
    params = np.asarray(snap.get("parameters"), dtype=float) if snap.get("parameters") is not None else None
    amp = params[:, 1] if params is not None and params.ndim == 2 else None

    used_colors = set()
    groups = []
    for i, bid in enumerate(np.unique(branch_ids)):
        mask = branch_ids == bid
        color = label = None
        key = i

        if lift is not None:
            # unforced representative: least-forced snapshot on this branch
            idx = np.where(mask)[0]
            rep = idx[np.argmin(np.abs(amp[idx]))] if amp is not None else idx[0]
            s = float(lift[rep])
            key = s
            if   s >  tol:  cand_c, cand_l = C_POS, r"1st asymm branch"
            elif s < -tol:  cand_c, cand_l = C_NEG, r"2nd asymm branch"
            else:           cand_c, cand_l = C_SYM, "symmetric branch"
            if cand_c not in used_colors:      # keep semantic colour if free
                color, label = cand_c, cand_l
                used_colors.add(cand_c)

        if color is None:                      # no lift, or colour collision
            for c in CLUSTER_COLORS:
                if c not in used_colors:
                    color = c
                    used_colors.add(c)
                    break
            if label is None:
                label = f"branch {int(bid)}"

        label = BRANCH_LABELS.get(int(bid), label)   # user override wins
        groups.append((label, mask, color, key))

    groups.sort(key=lambda g: g[3], reverse=True)     # C_L>0 ... symmetric ... C_L<0
    return [(label, mask, color) for (label, mask, color, _key) in groups]


# =============================================================================
# FIGURE
# =============================================================================

def make_figure(A, x, xlabel, groups, width, out, ylabel_fn, annot_fn=None):
    n_modes = A.shape[0]
    nrows   = math.ceil(n_modes / NCOLS)

    fig, axes = plt.subplots(
        nrows, NCOLS,
        figsize=(width, ROW_HEIGHT * nrows),
        constrained_layout=True,
        squeeze=False,
    )
    axf = axes.ravel()

    for k in range(n_modes):
        ax = axf[k]
        for label, mask, color in groups:
            if not np.any(mask):
                continue
            ax.scatter(x[mask], A[k, mask], s=14, c=color,
                       edgecolors="none", alpha=0.85, label=label)
        ax.set_ylabel(ylabel_fn(k))
        if annot_fn is not None:
            txt = annot_fn(k)
            if txt:
                ax.text(0.97, 0.92, txt, transform=ax.transAxes,
                        ha="right", va="top",
                        fontsize=BASE_FONTSIZE - 1, color="0.35")
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))

    # x-label only on the lowest populated panel of each column
    last_in_col = {}
    for k in range(n_modes):
        last_in_col[k % NCOLS] = k
    for k in last_in_col.values():
        axf[k].set_xlabel(xlabel)

    # hide any empty cells in the grid
    for k in range(n_modes, nrows * NCOLS):
        axf[k].set_visible(False)

    # single shared legend above the grid (captured by savefig bbox='tight')
    seen, handles, labels = set(), [], []
    for label, mask, color in groups:
        if not np.any(mask) or label in seen:
            continue
        seen.add(label)
        handles.append(Line2D([0], [0], marker="o", linestyle="none",
                              markersize=5, markerfacecolor=color,
                              markeredgecolor="none"))
        labels.append(label)
    if len(labels) > 1:
        fig.legend(handles, labels, loc="upper center", ncol=len(labels),
                   bbox_to_anchor=(0.5, 1.03), frameon=False,
                   handletextpad=0.3, columnspacing=1.2)

    fig.savefig(out)   # dpi / bbox='tight' / pad / embedded fonts come from style
    plt.close(fig)
    print(f"wrote {out}  ({n_modes} modes, {A.shape[1]} snapshots)")


# =============================================================================
# MAIN
# =============================================================================

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--usetex", action="store_true",
                   help="render text through LaTeX (fonts match the manuscript)")
    p.add_argument("--width", default="double",
                   help="figure width: preset (single/double/linewidth/0.7linewidth) or inches")
    p.add_argument("--xaxis", choices=["Re", "amp", "index"], default="Re",
                   help="what to put on the horizontal axis")
    p.add_argument("--color-by", choices=["branch", "lift", "none"], default=COLOR_BY,
                   help="how to colour the snapshots")
    p.add_argument("--n-modes", type=int, default=N_MODES,
                   help="number of POD modes (panels) to plot")
    p.add_argument("--out-dir", default=FIG_DIR,
                   help="folder for the output figures")
    args = p.parse_args()

    set_style(usetex=args.usetex)
    os.makedirs(args.out_dir, exist_ok=True)

    A, A_white, eigenvalues, snap = compute_coefficients(args.n_modes)
    rank   = A.shape[0]
    n_snap = A.shape[1]
    frac   = eigenvalues[:rank] / float(np.sum(eigenvalues))   # captured energy

    params = np.asarray(snap["parameters"], dtype=float)       # (n_snap, 2) -> (Re, amp)
    if args.xaxis == "Re":
        x, xlabel = params[:, 0], r"$Re$"
    elif args.xaxis == "amp":
        x, xlabel = params[:, 1], "control amplitude"
    else:
        x, xlabel = np.arange(n_snap, dtype=float), "snapshot index"

    groups = resolve_groups(snap, n_snap, args.color_by)
    width  = resolve_width(args.width)

    # raw coefficients: annotate each panel with its captured-energy fraction
    make_figure(
        A, x, xlabel, groups, width,
        os.path.join(args.out_dir, "pod_coefficients.pdf"),
        ylabel_fn=lambda k: rf"$a_{{{k + 1}}}$",
        annot_fn=lambda k: rf"${frac[k] * 100:.1f}\%$",
    )

    # whitened coefficients: annotate with the empirical std (should be ~1)
    make_figure(
        A_white, x, xlabel, groups, width,
        os.path.join(args.out_dir, "pod_coefficients_whitened.pdf"),
        ylabel_fn=lambda k: rf"$\tilde a_{{{k + 1}}}$",
        annot_fn=lambda k: rf"$\sigma\!=\!{np.std(A_white[k]):.2f}$",
    )


if __name__ == "__main__":
    main()
