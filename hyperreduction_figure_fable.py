r"""
Hyper-reduction pipeline figure for the fluidic pinball.

Panels (one figure, no subfigure labels):
  left top:    DEIM sampling of the convective residual vector,
  left bottom: MDEIM sampling of the convective Jacobian (CSR entries),
  right:       the ACTUAL reduced integration domain K on the pinball mesh.

The mesh panel is not schematic: it reconstructs the same reduced cell set that
`SubMeshDEIM` (hyper_reduction.py) assembles on, using the identical
index -> node -> cell bijections. Cells are two-toned by origin:
residual-support cells K_r vs Jacobian-only cells K_J \ K_r.

All styling comes from style.py so the figure matches the manuscript.
Generate at the width it is included at and include WITHOUT rescaling:
    python hyperreduction_figure.py --width linewidth
    \\includegraphics{figures/hyperreduction_pinball.pdf}

Two ways to supply the cells, in order of preference:

  (A) FIREDRAKE path (exact, recommended): run inside the firedrake venv.
      Rebuilds the problem, loads the DEIM indices, derives the reduced
      cells with the SAME logic as SubMeshDEIM, and caches them to
      REDUCED_CELLS_NPZ for path (B). Set --firedrake (default).

  (B) NUMPY path (no firedrake): reads the cache written by (A) plus the
      .msh via meshio. Use --no-firedrake. Beware: meshio cell ordering
      must match firedrake's; if the highlighted set looks wrong, rerun (A).
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D

from nsrom.plotting.style import set_style, resolve_width, C_POS, C_NEG

# ====================== CONFIG ======================
MESH_FILE         = "mesh/mid_pinball.msh"
DEIM_OPS_NPZ      = "global/deim_ops.npz"     # indices_F / indices_J_rows / _cols
REDUCED_CELLS_NPZ = "reduced_cells.npz"       # cache written by the firedrake path
REYNOLDS_INIT     = 100.0
AMPLITUDE_INIT    = 0.0
OUT_STEM          = "hyperreduction_pinball"
CYL_MARKERS       = [4, 5, 6]                 # physical tags of the cylinder walls
CYLINDERS_OVERRIDE = None   # [(cx,cy,R)]*3 to skip detection if it ever fails
# ====================================================

# Roles mapped onto the shared paper palette:
C_RES  = C_NEG    # residual / DEIM accent  (burnt orange)
C_JAC  = C_POS    # Jacobian / MDEIM accent (deep blue)
C_MESH = "0.80"   # background triangulation
C_DARK = "0.15"   # annotation ink


# ---------------------------------------------------------------------------
# (A) Firedrake path: derive the exact reduced cells the submesh uses.
# ---------------------------------------------------------------------------
def get_cells_and_mesh_firedrake():
    from firedrake import DirichletBC, Constant
    from nsrom.navier_stokes import setup_navier_stokes_problem

    problem = setup_navier_stokes_problem(
        mesh_file=MESH_FILE, reynolds_init=REYNOLDS_INIT,
        amplitude_init=AMPLITUDE_INIT,
    )
    V = problem.velocity_space
    mesh = V.mesh()
    value_size = V.value_size
    cell_node = V.cell_node_map().values        # (n_cells, nodes_per_cell)

    d = np.load(DEIM_OPS_NPZ, allow_pickle=True)
    print("deim_ops.npz keys:", list(d.keys()))
    indices_F      = np.asarray(d["indices_F"]).ravel()
    indices_J_rows = np.asarray(d["indices_J_rows"]).ravel()
    indices_J_cols = np.asarray(d["indices_J_cols"]).ravel()

    # --- SAME bijections as SubMeshDEIM, vectorized via node->cell adjacency ---
    from collections import defaultdict
    node_to_cells = defaultdict(set)
    for c in range(cell_node.shape[0]):
        for n in cell_node[c]:
            node_to_cells[int(n)].add(c)

    res_nodes = np.unique(indices_F // value_size)
    cells_res = set()
    for n in res_nodes:
        cells_res |= node_to_cells[int(n)]

    jr = indices_J_rows // value_size
    jc = indices_J_cols // value_size
    cells_jac = set()
    for rn, cn in zip(jr, jc):
        cells_jac |= (node_to_cells[int(rn)] & node_to_cells[int(cn)])

    n_cells = cell_node.shape[0]
    cells_res = np.array(sorted(cells_res), dtype=np.int64)
    cells_jac = np.array(sorted(cells_jac), dtype=np.int64)
    n_union = len(set(cells_res) | set(cells_jac))
    print(f"reduced cells: res={len(cells_res)} jac={len(cells_jac)} "
          f"union={n_union} / {n_cells} ({100*n_union/n_cells:.1f}%)")

    # --- mesh geometry straight from firedrake (guaranteed consistent order) ---
    coords = mesh.coordinates.dat.data_ro
    Vc = mesh.coordinates.function_space()
    tri = Vc.cell_node_map().values[:, :3].astype(np.int64)
    pts = np.asarray(coords)[:, :2]

    # --- cylinders from physical boundary tags (most reliable) ---
    cyl = []
    try:
        for mrk in CYL_MARKERS:
            nodes = DirichletBC(V, Constant((0.0,) * value_size), mrk).nodes
            P = pts[np.unique(nodes // value_size)]
            if len(P) < 6:
                continue
            c = P.mean(0)
            r = float(np.median(np.hypot(P[:, 0] - c[0], P[:, 1] - c[1])))
            cyl.append((float(c[0]), float(c[1]), r))
        print(f"cylinders from tags {CYL_MARKERS}: "
              f"{[tuple(round(v, 3) for v in c) for c in cyl]}")
    except Exception as e:
        print("tag-based cylinder extraction failed:", e)
        cyl = []

    # cache everything the numpy path (and the paper repo) needs
    np.savez(REDUCED_CELLS_NPZ, cells_res=cells_res, cells_jac=cells_jac,
             pts=pts, tri=tri, cyl=np.array(cyl, dtype=float))
    print(f"cached geometry + cells to {REDUCED_CELLS_NPZ}")
    return pts, tri, cells_res, cells_jac, cyl


# ---------------------------------------------------------------------------
# (B) No-firedrake path: everything comes from the cache written by (A).
#     Falls back to meshio + geometric detection only if the cache is absent.
# ---------------------------------------------------------------------------
def get_cells_and_mesh_numpy():
    try:
        d = np.load(REDUCED_CELLS_NPZ)
        pts, tri = d["pts"], d["tri"]
        cells_res, cells_jac = d["cells_res"], d["cells_jac"]
        cyl = [tuple(c) for c in d["cyl"]] if d["cyl"].size else []
        n_union = len(set(cells_res) | set(cells_jac))
        print(f"loaded cache: {n_union} reduced cells of {len(tri)} "
              f"({100*n_union/len(tri):.1f}%)")
        return pts, tri, cells_res, cells_jac, cyl
    except FileNotFoundError:
        pass

    import meshio
    m = meshio.read(MESH_FILE)
    pts = np.asarray(m.points[:, :2], float)
    tri = None
    for cb in m.cells:
        if cb.type in ("triangle", "triangle6"):
            tri = np.asarray(cb.data[:, :3], np.int64)
            break
    if tri is None:
        raise RuntimeError("No triangle cells in mesh.")
    # legacy single-array cache
    union = np.load("reduced_cells.npy").astype(np.int64)
    print("WARNING: legacy .npy cache; cells shown in one colour, and meshio "
          "cell ordering may differ from firedrake's. Prefer the firedrake path.")
    return pts, tri, union, np.array([], dtype=np.int64), []


def detect_cylinders(pts, tris):
    """Fallback: find cylinders as closed interior boundary loops."""
    if CYLINDERS_OVERRIDE is not None:
        return [tuple(map(float, c)) for c in CYLINDERS_OVERRIDE]

    from collections import defaultdict
    edge_count = defaultdict(int)
    for t in tris:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            edge_count[tuple(sorted((int(a), int(b))))] += 1
    bnd_edges = [e for e, c in edge_count.items() if c == 1]
    if not bnd_edges:
        return []

    adj = defaultdict(set)
    for a, b in bnd_edges:
        adj[a].add(b); adj[b].add(a)

    unseen, loops = set(adj), []
    while unseen:
        start = next(iter(unseen))
        stack, comp = [start], set()
        while stack:
            n = stack.pop()
            if n in comp:
                continue
            comp.add(n); unseen.discard(n)
            stack.extend(adj[n] - comp)
        loops.append(np.array(sorted(comp)))

    xmin, ymin = pts.min(0); xmax, ymax = pts.max(0)
    span = max(xmax - xmin, ymax - ymin)
    tol = 0.02 * span

    cyl = []
    for loop in loops:
        P = pts[loop]
        if (P[:, 0].min() < xmin + tol or P[:, 0].max() > xmax - tol or
                P[:, 1].min() < ymin + tol or P[:, 1].max() > ymax - tol):
            continue                       # outer channel walls
        c = P.mean(0)
        dist = np.hypot(P[:, 0] - c[0], P[:, 1] - c[1])
        r = float(np.median(dist))
        if not (0 < r < 0.25 * span):
            continue
        if len(dist) >= 6 and np.std(dist) / (r + 1e-30) < 0.15:
            cyl.append((float(c[0]), float(c[1]), r))
    return cyl


# ---------------------------------------------------------------------------
# Didactic panels
# ---------------------------------------------------------------------------
def draw_residual_panel(ax):
    """(a) DEIM: a few entries of the residual vector are sampled."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("Residual (DEIM)", fontsize=plt.rcParams["font.size"], pad=3)

    nb = 24
    sel = {3, 8, 12, 17, 21}
    x0, w = 0.26, 0.16
    h = 0.72 / nb
    for i in range(nb):
        y0 = 0.88 - i * h
        ax.add_patch(Rectangle((x0, y0), w, h * 0.82,
                     fc=C_RES if i in sel else "0.85",
                     ec="white", lw=0.3))
        if i in sel:
            ax.plot(x0 + w + 0.05, y0 + 0.4 * h, marker=4, ms=3.5,
                    color=C_RES, mec="none")
    ax.text(x0 + w / 2, 0.03,
            r"$\mathbf{N}(\mathbf{u}_h)\in\mathbb{R}^{N_h^u}$",
            ha="center", va="bottom",
            fontsize=plt.rcParams["font.size"] - 1)
    ax.text(x0 + w + 0.12, 0.52, r"$\mathbf{P}^{T}\mathbf{N}$" + "\n"
            + r"$m_F$ rows", fontsize=plt.rcParams["font.size"] - 1.5,
            color=C_RES, va="center")


def draw_jacobian_panel(ax):
    """(b) MDEIM: a few nonzero entries of the sparse Jacobian are sampled."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("Jacobian (MDEIM)", fontsize=plt.rcParams["font.size"], pad=3)

    ng, x0, y0, side = 13, 0.22, 0.14, 0.62
    rng = np.random.default_rng(1)
    occ = {(i, j) for i in range(ng) for j in range(ng)
           if abs(i - j) <= 1 or rng.random() < 0.06}
    sel = set(sorted(occ)[::9][:7])
    step = side / ng
    for (i, j) in occ:
        xx = x0 + (j + 0.5) * step
        yy = y0 + (ng - 0.5 - i) * step
        if (i, j) in sel:
            ax.plot(xx, yy, "s", color=C_JAC, ms=3.6, mec="none")
        else:
            ax.plot(xx, yy, ".", color="0.72", ms=2.2)
    ax.add_patch(Rectangle((x0, y0), side, side, fill=False, ec=C_DARK, lw=0.6))
    ax.text(x0 + side / 2, y0 + side + 0.045,
            r"nonzeros of $\mathbf{J}(\mathbf{u}_h)$", ha="center",
            fontsize=plt.rcParams["font.size"] - 1)
    ax.text(x0 + side / 2, 0.035, r"$m_J$ entries $(i,j)$", ha="center",
            fontsize=plt.rcParams["font.size"] - 1.5, color=C_JAC)


# ---------------------------------------------------------------------------
# Mesh panel with real reduced cells + zoom inset
# ---------------------------------------------------------------------------
def draw_mesh_panel(ax, pts, tris, cells_res, cells_jac, cyl):
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("Reduced Integration Domain $\\mathcal{K}$",
                 fontsize=plt.rcParams["font.size"], pad=3)

    set_res = set(int(c) for c in cells_res)
    set_jac = set(int(c) for c in cells_jac)
    jac_only = sorted(set_jac - set_res)
    union = sorted(set_res | set_jac)

    triang = mtri.Triangulation(pts[:, 0], pts[:, 1], tris)

    def paint(a, lw_mesh):
        a.triplot(triang, color=C_MESH, lw=lw_mesh, alpha=0.9)
        if set_res:
            a.add_collection(PolyCollection(
                [pts[tris[c]] for c in sorted(set_res)],
                facecolors=C_RES, edgecolors=C_RES, lw=0.2, alpha=0.95))
        if jac_only:
            a.add_collection(PolyCollection(
                [pts[tris[c]] for c in jac_only],
                facecolors=C_JAC, edgecolors=C_JAC, lw=0.2, alpha=0.95))
        for (cx, cy, R) in cyl:
            a.add_patch(Circle((cx, cy), R, fc="white", ec=C_DARK,
                               lw=0.8, zorder=5))

    paint(ax, 0.15)
    ax.set_xlim(pts[:, 0].min(), pts[:, 0].max())
    ax.set_ylim(pts[:, 1].min(), pts[:, 1].max())

    # ---- legend + count ----
    handles = [Line2D([], [], marker="s", ls="none", ms=5, mec="none",
                      mfc=C_RES, label=r"$\mathcal{K}_F$ (residual)")]
    if jac_only:
        handles.append(Line2D([], [], marker="s", ls="none", ms=5, mec="none",
                              mfc=C_JAC, label=r"$\mathcal{K}_J\setminus\mathcal{K}_F$"))
    ax.legend(handles=handles, loc="lower left", handletextpad=0.4,
              borderaxespad=0.0, fontsize=plt.rcParams["font.size"] - 1.5)

    frac = 100 * len(union) / len(tris)
    ax.text(0.5, -0.06,
            rf"$|\mathcal{{K}}| = {len(union)}$ of "
            rf"${len(tris)}$ cells $\;(\approx {frac:.1f}\%)$",
            transform=ax.transAxes, ha="center",
            fontsize=plt.rcParams["font.size"] - 1, color=C_DARK)


# ---------------------------------------------------------------------------
# Figure assembly
# ---------------------------------------------------------------------------
def build_figure(pts, tris, cells_res, cells_jac, cyl, width):
    fig = plt.figure(figsize=(width, 0.44 * width))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 2.55],
                          height_ratios=[1, 1], wspace=0.36, hspace=0.55,
                          left=0.015, right=0.985, top=0.90, bottom=0.075)

    ax_r = fig.add_subplot(gs[0, 0])
    ax_j = fig.add_subplot(gs[1, 0])
    ax_m = fig.add_subplot(gs[:, 1])

    draw_residual_panel(ax_r)
    draw_jacobian_panel(ax_j)
    draw_mesh_panel(ax_m, pts, tris, cells_res, cells_jac, cyl)

    # arrows carrying the index -> cell bijections
    fs = plt.rcParams["font.size"] - 1.5

    def connect(y, col, label):
        fig.add_artist(FancyArrowPatch(
            (0.265, y), (0.315, y), transform=fig.transFigure,
            arrowstyle="-|>", mutation_scale=9, lw=0.9, color=col, zorder=20))
        fig.text(0.29, y + 0.035, label, ha="center", va="bottom",
                 fontsize=fs, color=col)

    connect(0.68, C_RES, "row $\\to$ DOF $\\to$\nsupport cells")
    connect(0.28, C_JAC, "$(i,j)\\to$\nshared cells")

    return fig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--width", default="linewidth",
                   help="preset from style.COLUMN_WIDTHS or inches")
    p.add_argument("--usetex", action="store_true")
    fd = p.add_mutually_exclusive_group()
    fd.add_argument("--firedrake", dest="firedrake", action="store_true",
                    default=True)
    fd.add_argument("--no-firedrake", dest="firedrake", action="store_false")
    args = p.parse_args()

    set_style(usetex=args.usetex)
    width = resolve_width(args.width)

    if args.firedrake:
        pts, tris, cells_res, cells_jac, cyl = get_cells_and_mesh_firedrake()
    else:
        pts, tris, cells_res, cells_jac, cyl = get_cells_and_mesh_numpy()

    if not cyl:
        cyl = detect_cylinders(pts, tris)
        print("geometric cylinder detection:", cyl)
    if not cyl:
        print("!! no cylinders found; set CYLINDERS_OVERRIDE at top of file.")

    fig = build_figure(pts, tris, cells_res, cells_jac, cyl, width)
    fig.savefig(f"{OUT_STEM}.pdf")
    fig.savefig(f"{OUT_STEM}.png", dpi=300)
    print(f"saved {OUT_STEM}.pdf / .png at width {width:.2f} in")


if __name__ == "__main__":
    main()
