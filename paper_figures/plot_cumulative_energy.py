#!/usr/bin/env python
"""
Cumulative POD energy vs number of modes: global basis vs per-cluster local
bases. Supports the local-ROM argument -- each cluster's spectrum decays
faster, so a local basis captures the same energy fraction with fewer modes.

Reads pod_basis.npz files written by save_pod_basis (keys:
    velocity_modes, pressure_modes, supremizer_modes (optional),
    velocity_eigenvalues, pressure_eigenvalues, supremizer_eigenvalues),
and the sibling metadata.json when present (used only to detect whether the
stored eigenvalues are the truncated retained set vs the full spectrum).

Layout expected (defaults match the NS_ROM tree):
    global/pod_basis_multi_branch/pod_basis.npz
    local_rom/K4_H1/cluster_<k>/pod_basis/pod_basis.npz

Produces (PDF + PNG in --outdir):
    cumulative_energy_<field>.{pdf,png}   (metric=cumulative, default)
    neglected_energy_<field>.{pdf,png}    (metric=neglected)

Usage:
    python plot_cumulative_energy.py
    python plot_cumulative_energy.py --field velocity --threshold 0.999
    python plot_cumulative_energy.py --metric neglected --logx
    python plot_cumulative_energy.py --local-root local_rom/K8_H1 \
        --global-basis global/pod_basis_multi_branch/pod_basis.npz \
        --width linewidth --usetex
"""

import argparse
import glob
import json
import os
import re

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nsrom.plotting.style import (
    set_style, resolve_width, C_GLOBAL, CLUSTER_COLORS,
)


# ----------------------------------------------------------------------
# IO
# ----------------------------------------------------------------------

def _cluster_index(path):
    """Extract the integer k from a .../cluster_<k>/... path (for sorting)."""
    m = re.search(r"cluster_(\d+)", path)
    return int(m.group(1)) if m else 1 << 30


def find_cluster_bases(local_root):
    """-> [(k, npz_path), ...] sorted by cluster index."""
    pat = os.path.join(local_root, "cluster_*", "pod_basis", "pod_basis.npz")
    hits = sorted(glob.glob(pat), key=_cluster_index)
    return [(_cluster_index(p), p) for p in hits]


def load_eigenvalues(npz_path, field):
    """-> descending, positive eigenvalues for `field` from a pod_basis.npz."""
    d = np.load(npz_path)
    key = f"{field}_eigenvalues"
    if key not in d.files:
        raise KeyError(f"{npz_path} has no '{key}' "
                       f"(available: {', '.join(d.files)})")
    ev = np.asarray(d[key], dtype=float)
    ev = ev[np.isfinite(ev)]
    ev = ev[ev > 0.0]                       # drop round-off negatives/zeros
    ev = np.sort(ev)[::-1]                  # ensure descending
    return ev


def cumulative_energy(npz_path, field, use_metadata_total=True):
    """-> (modes[1..N], cumulative_fraction).

    Normalises by the full-spectrum energy. If the stored eigenvalues are the
    truncated *retained* set (len == n_<field>_modes) and metadata reports an
    energy fraction < 1, the true total is reconstructed from that fraction so
    the curve tops out honestly at the captured fraction rather than 1.0.
    """
    ev = load_eigenvalues(npz_path, field)
    total = ev.sum()

    if use_metadata_total:
        meta_path = os.path.join(os.path.dirname(npz_path), "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            n_modes = meta.get(f"n_{field}_modes")
            frac = meta.get(f"{field}_energy_fraction")
            if (n_modes is not None and frac is not None
                    and len(ev) <= n_modes and 0.0 < frac < 0.999999):
                total = total / frac       # reconstruct full-spectrum total

    cum = np.cumsum(ev) / total
    modes = np.arange(1, len(ev) + 1)
    return modes, cum


def modes_to_threshold(cum, thr):
    """First mode count reaching cumulative fraction >= thr (None if never)."""
    hit = np.where(cum >= thr)[0]
    return int(hit[0] + 1) if hit.size else None


# ----------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------

def plot_energy(global_curve, cluster_curves, field, metric, thr, width,
                logx=False):
    fig, ax = plt.subplots(figsize=(width, 0.66 * width))

    def transform(cum):
        return (1.0 - cum) if metric == "neglected" else cum

    # threshold guide line
    y_thr = transform(np.array([thr]))[0]
    ax.axhline(y_thr, ls=":", color="0.5", lw=0.9, zorder=1)

    def curve_label(name, cum):
        m = modes_to_threshold(cum, thr)
        tag = f"{m}" if m is not None else fr"$>${len(cum)}"
        return f"{name} ({tag})"

    # global (reference)
    gm, gc = global_curve
    ax.plot(gm, transform(gc), "-", color=C_GLOBAL, lw=1.7, zorder=5,
            label=curve_label("global", gc))
    m = modes_to_threshold(gc, thr)
    if m is not None:
        ax.plot(m, transform(gc[m - 1:m])[0], "o", ms=3.5,
                color=C_GLOBAL, zorder=6)

    # clusters
    for (k, (cm, cc)) in cluster_curves:
        col = CLUSTER_COLORS[k % len(CLUSTER_COLORS)]
        ax.plot(cm, transform(cc), "-", color=col, lw=1.1, zorder=4,
                label=curve_label(f"cluster {k}", cc))
        m = modes_to_threshold(cc, thr)
        if m is not None:
            ax.plot(m, transform(cc[m - 1:m])[0], "o", ms=3.0,
                    color=col, zorder=6)

    if logx:
        ax.set_xscale("log")

    ax.set_xlabel(r"number of POD modes $n$")
    if metric == "neglected":
        ax.set_yscale("log")
        ax.set_ylabel(r"neglected energy $1-\sum_{i\leq n}\lambda_i/"
                      r"\sum_i\lambda_i$")
        thr_txt = rf"${100*(1-thr):.2g}\%$ neglected"
    else:
        ax.set_ylabel(r"cumulative energy $\sum_{i\leq n}\lambda_i/"
                      r"\sum_i\lambda_i$")
        thr_txt = rf"${100*thr:.4g}\%$ captured"

    # threshold annotation, parked at the right edge on the guide line
    ax.text(0.985, y_thr, thr_txt, transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=7, color="0.4")

    field_txt = {"velocity": "velocity", "pressure": "pressure",
                 "supremizer": "supremizer"}.get(field, field)
    ax.legend(loc="best", handlelength=1.4, borderaxespad=0.4,
              labelspacing=0.3, title=f"{field_txt}  ($n$ to threshold)",
              title_fontsize=7)

    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--global-basis",
                   default="global/pod_basis_multi_branch/pod_basis.npz")
    p.add_argument("--local-root", default="local_rom/K4_H1",
                   help="dir containing cluster_<k>/pod_basis/pod_basis.npz")
    p.add_argument("--field", default="velocity",
                   choices=["velocity", "pressure", "supremizer"])
    p.add_argument("--metric", default="cumulative",
                   choices=["cumulative", "neglected"])
    p.add_argument("--threshold", type=float, default=0.999,
                   help="energy fraction for the modes-to-threshold count")
    p.add_argument("--logx", action="store_true")
    p.add_argument("--no-metadata-total", action="store_true",
                   help="normalise by the stored eigenvalue sum only")
    p.add_argument("--outdir", default="figures")
    p.add_argument("--width", default="linewidth",
                   help="preset ('single','double','linewidth','0.7linewidth')"
                        " or a width in inches, e.g. 6.22")
    p.add_argument("--usetex", action="store_true")
    args = p.parse_args()

    set_style(args.usetex)
    os.makedirs(args.outdir, exist_ok=True)
    W = resolve_width(args.width)
    use_meta = not args.no_metadata_total

    # global
    global_curve = cumulative_energy(args.global_basis, args.field, use_meta)
    print(f"global: {len(global_curve[0])} {args.field} modes, "
          f"reaches {global_curve[1][-1]:.6f} of total")

    # clusters
    bases = find_cluster_bases(args.local_root)
    if not bases:
        raise SystemExit(f"no cluster bases under {args.local_root} "
                         f"(looked for cluster_*/pod_basis/pod_basis.npz)")
    cluster_curves = []
    for k, path in bases:
        curve = cumulative_energy(path, args.field, use_meta)
        cluster_curves.append((k, curve))
        m = modes_to_threshold(curve[1], args.threshold)
        print(f"cluster {k}: {len(curve[0])} modes, "
              f"{args.threshold:.3%} at n={m}")

    m_g = modes_to_threshold(global_curve[1], args.threshold)
    print(f"global:    {args.threshold:.3%} at n={m_g}")

    fig = plot_energy(global_curve, cluster_curves, args.field, args.metric,
                      args.threshold, W, logx=args.logx)

    stem = ("neglected_energy" if args.metric == "neglected"
            else "cumulative_energy") + f"_{args.field}"
    for ext in ("pdf", "png"):
        f = os.path.join(args.outdir, f"{stem}.{ext}")
        fig.savefig(f)
        print(f"wrote {f}")


if __name__ == "__main__":
    main()
