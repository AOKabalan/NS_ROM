#!/usr/bin/env python3
"""
plot_clusters_compare.py
========================

The figure that actually shows the effect of whitening: the k-means CLUSTER
ASSIGNMENT in the (Re, lift) plane, energy-norm metric vs whitened metric,
side by side. Points are coloured by the cluster they were assigned to; lift
on the vertical axis acts as a branch proxy (~0 symmetric, +/- asymmetric).

Raw metric  -> clusters are vertical Re/amplitude slabs that cut across the
               lift branches (the partition follows the forcing, not the branch).
Whitened    -> clusters line up with the lift bands (the partition follows the
               branch).

This shows WHY whitening is needed in a way the branch-coloured coefficient
scatter cannot: the k-means failure is about the full-dimensional distance
being dominated by the within-branch spread of the high-energy modes, which is
only visible once you look at the assignment itself.

Output
------
    figures/clusters_whitened_vs_plain.pdf
"""

import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans

from nsrom.navier_stokes import setup_navier_stokes_problem
from nsrom.snapshot_collection import load_snapshot_dofs
from nsrom.lifting_functions import load_lifting_functions
from nsrom.cluster_building import (
    build_energy_snapshots, get_pod_coefficients, mahalanobis_whitening,
)
from nsrom.plotting.style import set_style, resolve_width, BASE_FONTSIZE, CLUSTER_COLORS


SNAPSHOT_DIR       = "multi_param_multi_branch"
LIFTING_DIR        = "lifting"
MESH_FILE          = "mesh/mid_pinball.msh"
REYNOLDS_INIT      = 100.0
AMPLITUDE_INIT     = 0.0
INNER_PRODUCT_TYPE = "H1"

FIG_DIR   = "figures"
RANK      = 5          # POD modes clustered
K         = 4           # clusters
SEED      = 0


def compute(rank):
    problem = setup_navier_stokes_problem(MESH_FILE, REYNOLDS_INIT, AMPLITUDE_INIT)
    snap    = load_snapshot_dofs(SNAPSHOT_DIR)
    lifting = load_lifting_functions(LIFTING_DIR, problem)
    S_homo, _St, M_u, _Mp, _L, _P = build_energy_snapshots(
        snap["velocity"], snap["parameters"], lifting, problem,
        INNER_PRODUCT_TYPE, homogenize=True)
    rank = min(rank, S_homo.shape[1])
    A, eig, _m = get_pod_coefficients(S_homo, rank=rank, M_u=M_u)
    A_white = mahalanobis_whitening(A, eig, rank)
    return A, A_white, snap


def kmeans_labels(X, k):
    return KMeans(n_clusters=k, init="k-means++", n_init=10,
                  random_state=SEED).fit_predict(X)


def panel(ax, Re, lift, labels, k, title):
    for c in range(k):
        m = labels == c
        ax.scatter(Re[m], lift[m], s=14, c=CLUSTER_COLORS[c % len(CLUSTER_COLORS)],
                   edgecolors="none", alpha=0.85)
    ax.axhline(0, color="0.85", lw=0.6, zorder=0)
    ax.set_xlabel(r"$Re$")
    ax.set_title(title, fontsize=BASE_FONTSIZE + 1)
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--usetex", action="store_true")
    p.add_argument("--width", default="double")
    p.add_argument("--rank", type=int, default=RANK)
    p.add_argument("--k", type=int, default=K)
    p.add_argument("--out-dir", default=FIG_DIR)
    args = p.parse_args()

    set_style(usetex=args.usetex)
    os.makedirs(args.out_dir, exist_ok=True)

    A, A_white, snap = compute(args.rank)
    params = np.asarray(snap["parameters"], float)
    Re = params[:, 0]
    lift = np.asarray(snap["lift_coefficients"], float).ravel()

    raw_labels   = kmeans_labels(A.T, args.k)
    white_labels = kmeans_labels(A_white.T, args.k)

    width = resolve_width(args.width)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(width, 0.46 * width),
                                   sharey=True, constrained_layout=True)
    panel(ax0, Re, lift, raw_labels,   args.k, "energy-norm $k$-means")
    panel(ax1, Re, lift, white_labels, args.k, "whitened $k$-means")
    ax0.set_ylabel(r"$C_L$")

    handles = [Line2D([0], [0], marker="o", linestyle="none", markersize=5,
                      markerfacecolor=CLUSTER_COLORS[c % len(CLUSTER_COLORS)],
                      markeredgecolor="none") for c in range(args.k)]
    labels  = [f"cluster {c}" for c in range(args.k)]
    fig.legend(handles, labels, loc="upper center", ncol=args.k,
               bbox_to_anchor=(0.5, 1.05), frameon=False,
               handletextpad=0.3, columnspacing=1.2)

    out = os.path.join(args.out_dir, "clusters_whitened_vs_plain.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
