#!/usr/bin/env python
"""
Publication-quality figure of the fluidic pinball geometry and boundary
conditions, built directly from a Gmsh .msh file (exact geometry, no
hand-drawn circles).

Style (fonts, sizes, palette, column widths) comes from nsrom.plotting.style,
shared with every other figure script.

Produces (PDF + PNG in --outdir):

    pinball_geometry.{pdf,png} : schematic domain + BC callouts in the
                                 main panel, real mesh detail in a zoomed
                                 inset over the empty downstream region.

Usage:
    python pinball_geometry_figure.py pinball.msh
    python pinball_geometry_figure.py pinball.msh --width double --outdir figures
    python pinball_geometry_figure.py pinball.msh --usetex --xlabel '$a$'

Requires numpy + matplotlib + meshio. --usetex needs a LaTeX install;
without it, Computer Modern mathtext is used (matches the other figures).
"""

import argparse
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
from matplotlib.collections import LineCollection

import meshio

from nsrom.plotting.style import set_style, resolve_width, SINGLE_COL, DOUBLE_COL, C_POS, C_NEG

# ----------------------------------------------------------------------
# Figure-local colours (semantic colours come from the shared palette)
# ----------------------------------------------------------------------

C_INFLOW = C_POS     # inflow arrows / left-edge accent (shared blue)
C_ROT    = C_NEG     # cylinder rotation arrows        (shared rust)
C_EDGE   = "0.15"    # boundary / cylinder edges
C_FILL   = "0.88"    # cylinder fill
C_LEAD   = "0.35"    # leader lines / callout text
C_MESH   = "0.45"    # inset triangulation


# ----------------------------------------------------------------------
# Mesh IO
# ----------------------------------------------------------------------

def read_mesh(path):
    """-> (points[N,2], Triangulation, {phys_name: [edges]}) from a .msh.

    Works for Gmsh 2.2 and 4.1, linear or quadratic triangles."""
    mesh = meshio.read(path)
    pts = mesh.points[:, :2]

    tris = None
    for key in ("triangle", "triangle6"):
        if key in mesh.cells_dict:
            tris = mesh.cells_dict[key][:, :3]   # first 3 nodes -> linear T
            break
    if tris is None:
        sys.exit("no triangle cells found in mesh")
    triang = Triangulation(pts[:, 0], pts[:, 1], tris)

    tag2name = {tag: name for name, (tag, dim) in mesh.field_data.items()}
    phys = mesh.cell_data.get("gmsh:physical", [None] * len(mesh.cells))
    edges_by_name = {}
    for cb, data in zip(mesh.cells, phys):
        if cb.type != "line":
            continue
        tags = data if data is not None else [None] * len(cb.data)
        for edge, tag in zip(cb.data, tags):
            name = tag2name.get(tag, f"tag_{tag}")
            edges_by_name.setdefault(name, []).append(edge)

    return pts, triang, edges_by_name


def detect_cylinders(pts, edges_by_name):
    """-> list of (name, center[2], radius), sorted front (small x) first."""
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    L, H = xmax - xmin, ymax - ymin
    cyl = []
    for name, edges in edges_by_name.items():
        nodes = np.unique(np.array(edges))
        c = pts[nodes].mean(axis=0)
        span = np.ptp(pts[nodes], axis=0)
        if span[0] < 0.4 * L and span[1] < 0.4 * H:
            cyl.append((name, c, 0.5 * span.max()))
    return sorted(cyl, key=lambda t: t[1][0])


# ----------------------------------------------------------------------
# Figure: domain + BCs + mesh inset
# ----------------------------------------------------------------------

# axis-limit margins (fractions of domain L, H) -> also fix the aspect
MX_L, MX_R, MY = 0.17, 0.10, 0.32


def plot_geometry(pts, triang, edges_by_name, cyl, width, xlabel,
                  box=None, inset=True):
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    L, H = xmax - xmin, ymax - ymin

    # figure height follows the true (equal-aspect) domain proportions
    xspan = L * (1 + MX_L + MX_R)
    yspan = H * (1 + 2 * MY)
    fig, ax = plt.subplots(figsize=(width, width * yspan / xspan))

    # domain outline (axes.linewidth-weight to match the other figures)
    ax.add_patch(Rectangle((xmin, ymin), L, H, fill=False,
                           ec="0.2", lw=0.8))

    # exact boundaries from the mesh
    for edges in edges_by_name.values():
        segs = pts[np.array(edges)]
        ax.add_collection(LineCollection(segs, colors=C_EDGE, linewidths=1.0))

    # cylinders
    for name, c, r in cyl:
        ax.add_patch(Circle(c, r, facecolor=C_FILL, edgecolor=C_EDGE,
                            lw=0.8, zorder=3))

    # inflow arrows: start just outside the left edge and cross into the
    # domain, so it reads as flow entering through that boundary
    for yy in np.linspace(ymin + 0.14 * H, ymax - 0.14 * H, 5):
        ax.annotate("", xy=(xmin + 0.08 * L, yy), xytext=(xmin - 0.035 * L, yy),
                    arrowprops=dict(arrowstyle="-|>", color=C_INFLOW, lw=1.1))

    # ---- boundary-condition labels: each label hugs its own edge and is
    #      tied to it by an explicit leader arrow (name <-> boundary) ----
    def bc_label(text, edge_xy, text_xy, rotation=0, ha="center", va="center"):
        ax.annotate(text, xy=edge_xy, xytext=text_xy, rotation=rotation,
                    ha=ha, va=va, fontsize=8, color="0.12", zorder=6,
                    arrowprops=dict(arrowstyle="-|>", lw=0.7, color="0.35",
                                    shrinkA=3, shrinkB=1))

    # left edge -> inflow  (label reads up the edge, arrow points onto it)
    bc_label(r"inflow: $\mathbf{u}=U_\infty\mathbf{e}_x$",
             (xmin, ymin + 0.30 * H), (xmin - 0.055 * L, ymin + 0.30 * H),
             rotation=90, va="center", ha="center")
    # right edge -> outflow
    bc_label(r"outflow: $\partial_n\mathbf{u}=\mathbf{0}$",
             (xmax, 0.0), (xmax + 0.05 * L, 0.0),
             rotation=90, va="center", ha="center")
    # top edge -> free-stream (upper wall)
    bc_label(r"free-stream $\mathbf{u}=U_\infty\mathbf{e}_x$ (upper wall)",
             (xmin + 0.20 * L, ymax), (xmin + 0.20 * L, ymax + 0.09 * H),
             va="bottom", ha="center")
    # bottom edge -> free-stream (lower wall)
    bc_label(r"free-stream $\mathbf{u}=U_\infty\mathbf{e}_x$ (lower wall)",
             (xmin + 0.20 * L, ymin), (xmin + 0.20 * L, ymin - 0.09 * H),
             va="top", ha="center")

    # dimension bars (length sits fully below the lower-wall label)
    ax.annotate("", xy=(xmax, ymin - 0.24 * H), xytext=(xmin, ymin - 0.24 * H),
                arrowprops=dict(arrowstyle="<->", lw=0.7))
    ax.text((xmin + xmax) / 2, ymin - 0.255 * H, rf"${L:.0f}\,D$",
            ha="center", va="top", fontsize=8)
    ax.annotate("", xy=(xmin - 0.11 * L, ymax), xytext=(xmin - 0.11 * L, ymin),
                arrowprops=dict(arrowstyle="<->", lw=0.7))
    ax.text(xmin - 0.135 * L, (ymin + ymax) / 2, rf"${H:.0f}\,D$",
            ha="center", va="center", rotation=90, fontsize=8)

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(xmin - MX_L * L, xmax + MX_R * L)
    ax.set_ylim(ymin - MY * H, ymax + MY * H)

    # ---- mesh-detail inset in the TOP-RIGHT corner; dashed, tilted zoom
    #      lines tie it to the cylinder region. Also carries the cylinder
    #      BCs (no-slip / rotation) where the zoom leaves room. ----
    if inset:
        if box is None:
            fc = cyl[0][1]
            box = (fc[0] - 1.6, fc[0] + 4.4, -2.6, 2.6)
        bx0, bx1, by0, by1 = box
        axins = ax.inset_axes([0.54, 0.54, 0.34, 0.44])
        axins.triplot(triang, color=C_MESH, lw=0.15)
        for name, c, r in cyl:
            axins.add_patch(Circle(c, r, facecolor=C_FILL,
                                  edgecolor=C_EDGE, lw=0.7, zorder=3))

        # no-slip on the front cylinder
        fcx, fcy = cyl[0][1]
        fr = cyl[0][2]
        axins.annotate(
            "no-slip",
            xy=(fcx, fcy + fr),
            xytext=(bx0 + 0.16 * (bx1 - bx0), by1 - 0.10 * (by1 - by0)),
            fontsize=7,
            color="0.15",
            ha="center",
            va="top",
            zorder=5,
            bbox=dict(
                facecolor="white",
                edgecolor="0.5",
                boxstyle="round,pad=0.2",
                linewidth=0.5,
            ),
            arrowprops=dict(
                arrowstyle="-",
                lw=0.6,
                color=C_LEAD,
            ),
        )

        # rotation arrows + label on the rear cylinders
        for name, c, r in cyl[1:]:
            s = 1.0 if c[1] >= 0 else -1.0        # tweak direction as needed
            axins.add_patch(FancyArrowPatch(
                (c[0] + 1.5 * r, c[1] - 1.1 * r * s),
                (c[0] + 1.5 * r, c[1] + 1.1 * r * s),
                connectionstyle=f"arc3,rad={0.7 * s}", arrowstyle="-|>",
                mutation_scale=8, color=C_ROT, lw=1.1, zorder=4))

        # rotation label, centred between the two rear (rotating) cylinders
        c1 = cyl[1][1]
        c2 = cyl[2][1]
        mid = 0.5 * (c1 + c2)
        axins.text(
            mid[0] + 1.0,      # tweak horizontally to taste
            mid[1],            # vertically centred between the cylinders
            rf"rotation {xlabel}",
            fontsize=7,
            color="0.15",
            ha="left",
            va="center",
            zorder=5,
            bbox=dict(
                facecolor="white",
                edgecolor="0.5",
                boxstyle="round,pad=0.2",
                linewidth=0.5,
            ),
        )

        axins.set_xlim(bx0, bx1)
        axins.set_ylim(by0, by1)
        axins.set_aspect("equal")
        axins.set_xticks([])
        axins.set_yticks([])
        for sp in axins.spines.values():
            sp.set_linewidth(0.8)
            sp.set_color("0.4")

        # dashed zoom rectangle + tilted connector lines
        ind = ax.indicate_inset_zoom(axins, edgecolor="0.35", alpha=1.0)
        dash = (0, (4, 3))
        ind.rectangle.set(linestyle=dash, linewidth=0.8, edgecolor="0.35")
        for cpatch in ind.connectors:
            cpatch.set(linestyle=dash, linewidth=0.8, color="0.35")

    return fig


# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("msh", nargs="?", default="pinball.msh")
    p.add_argument("--outdir", default="figures")
    p.add_argument("--width", default="single",
                help="preset ('single','double','linewidth','0.7linewidth') "
                        "or a width in inches, e.g. 5.79")
    p.add_argument("--xlabel", default=r"$a$",
                   help="rotation-control symbol (default: $a$, "
                        "match the critical-curve figures)")
    p.add_argument("--box", type=float, nargs=4, default=None,
                   metavar=("X0", "X1", "Y0", "Y1"),
                   help="inset zoom box; default auto-frames the cylinders")
    p.add_argument("--no-inset", action="store_true")
    p.add_argument("--usetex", action="store_true")
    args = p.parse_args()

    set_style(args.usetex)
    os.makedirs(args.outdir, exist_ok=True)
    W = resolve_width(args.width)
    pts, triang, edges_by_name = read_mesh(args.msh)
    cyl = detect_cylinders(pts, edges_by_name)
    print("detected cylinders:",
          [(n, tuple(np.round(c, 2))) for n, c, _ in cyl])
    if len(cyl) != 3:
        print(f"  warning: expected 3 cylinders, found {len(cyl)} "
              f"(check boundary physical groups / detection heuristic)")

    fig = plot_geometry(pts, triang, edges_by_name, cyl, W, args.xlabel,
                        box=args.box, inset=not args.no_inset)
    for ext in ("pdf", "png"):
        f = os.path.join(args.outdir, f"pinball_geometry.{ext}")
        fig.savefig(f)
        print(f"wrote {f}")


if __name__ == "__main__":
    main()
