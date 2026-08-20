"""
Hyper-reduction pipeline figure for the fluidic pinball, highlighting the
ACTUAL reduced integration cells used by the DEIM/MDEIM submesh.

The right panel is not schematic: it reconstructs the same reduced cell set
that `SubMeshDEIM` (hyper_reduction.py) assembles on, using the identical
index -> node -> cell bijections. Left/middle panels remain didactic.

Two ways to supply the cells, in order of preference:

  (A) FIRE_DRAKE path (exact, recommended): run inside your firedrake venv.
      Rebuilds the problem, loads the DEIM indices, and derives the reduced
      cells with the SAME logic as SubMeshDEIM. Set USE_FIREDRAKE = True.

  (B) NUMPY path (no firedrake): if you have already saved the reduced cell
      indices to a .npy (e.g. np.save("reduced_cells.npy", reduced_cells)
      inside SubMeshDEIM.__init__), point REDUCED_CELLS_NPY at it and set
      USE_FIREDRAKE = False. The mesh is still read for plotting.

Requires (both paths): meshio? no -- see MESH LOADING note. matplotlib, numpy.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from matplotlib.collections import PolyCollection

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import style  # noqa: E402

style.set_style()

# Schematic accents, deliberately NOT the branch palette: nothing in this
# figure is a solution branch, so reusing C_POS / C_NEG would imply a meaning
# the panels do not carry.
ACCENT, ACCENT2, GREY, DARK = "#c0392b", "#2471a3", "#c2c2c2", "#2b2b2b"

# ====================== CONFIG ======================
USE_FIREDRAKE     = True                      # (A) exact path if True, else (B)
MESH_FILE         = "mesh/mid_pinball.msh"
DEIM_OPS_NPZ      = "global/deim_ops.npz"      # holds indices_F / indices_J_rows / _cols
REDUCED_CELLS_NPY = "reduced_cells.npy"        # used only if USE_FIREDRAKE = False
REYNOLDS_INIT     = 100.0
AMPLITUDE_INIT    = 0.0
OUT_STEM          = "hyperreduction_pinball"
OUTDIR            = "render/out"
# ====================================================


# ---------------------------------------------------------------------------
# (A) Firedrake path: derive the exact reduced cells the submesh uses.
# ---------------------------------------------------------------------------
def get_cells_and_mesh_firedrake():
    from firedrake import SpatialCoordinate, Function
    # --- adapt these imports to your package layout if needed ---
    from nsrom.navier_stokes import setup_navier_stokes_problem

    problem = setup_navier_stokes_problem(
        mesh_file=MESH_FILE, reynolds_init=REYNOLDS_INIT,
        amplitude_init=AMPLITUDE_INIT,
    )
    V = problem.velocity_space
    mesh = V.mesh()
    value_size = V.value_size
    cell_node = V.cell_node_map().values        # (n_cells, nodes_per_cell)

    # --- load DEIM indices (print keys so you can rename if they differ) ---
    d = np.load(DEIM_OPS_NPZ, allow_pickle=True)
    print("deim_ops.npz keys:", list(d.keys()))
    indices_F      = np.asarray(d["indices_F"]).ravel()
    indices_J_rows = np.asarray(d["indices_J_rows"]).ravel()
    indices_J_cols = np.asarray(d["indices_J_cols"]).ravel()

    # --- SAME bijections as SubMeshDEIM, but vectorized via adjacency ---
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

    reduced_cells = np.array(sorted(cells_res | cells_jac), dtype=np.int64)
    print(f"reduced cells: res={len(cells_res)} jac={len(cells_jac)} "
          f"union={len(reduced_cells)} / {cell_node.shape[0]} "
          f"({100*len(reduced_cells)/cell_node.shape[0]:.1f}%)")
    np.save("reduced_cells.npy", reduced_cells)   # cache for path (B) / the paper

    # --- mesh geometry for plotting, straight from firedrake ---
    coords = mesh.coordinates.dat.data_ro
    # triangle connectivity of the *geometry* (P1 cell->vertex map)
    Vc = mesh.coordinates.function_space()
    tri = Vc.cell_node_map().values.astype(np.int64)
    pts = np.asarray(coords)[:, :2]

    # --- cylinders from physical boundary tags (most reliable) ---
    # Pinball cylinder wall markers; adjust to your .msh physical groups.
    # (Your supremizer BCs used [1,3,4,5,6]; the cylinders are a subset.)
    CYL_MARKERS = [4, 5, 6]
    cyl = []
    try:
        from firedrake import DirichletBC, Constant
        for mrk in CYL_MARKERS:
            nodes = DirichletBC(V, Constant((0.0,) * value_size), mrk).nodes
            P = pts[np.unique(nodes // value_size)]
            if len(P) < 6:
                continue
            c = P.mean(0)
            r = float(np.median(np.hypot(P[:,0]-c[0], P[:,1]-c[1])))
            cyl.append((float(c[0]), float(c[1]), r))
        print(f"cylinders from tags {CYL_MARKERS}: "
              f"{[tuple(round(v,3) for v in c) for c in cyl]}")
    except Exception as e:
        print("tag-based cylinder extraction failed:", e)
        cyl = []
    return pts, tri, reduced_cells, cyl


# ---------------------------------------------------------------------------
# (B) No-firedrake path: read mesh with meshio, load cached reduced cells.
# ---------------------------------------------------------------------------
def get_cells_and_mesh_numpy():
    import meshio
    m = meshio.read(MESH_FILE)
    pts = np.asarray(m.points[:, :2], float)
    tri = None
    for cb in m.cells:
        if cb.type in ("triangle", "triangle6"):
            tri = np.asarray(cb.data[:, :3], np.int64)   # first 3 nodes = vertices
            break
    if tri is None:
        raise RuntimeError("No triangle cells in mesh.")
    reduced_cells = np.load(REDUCED_CELLS_NPY).astype(np.int64)
    print(f"loaded {len(reduced_cells)} reduced cells (of {len(tri)}, "
          f"{100*len(reduced_cells)/len(tri):.1f}%)")
    # NOTE: meshio cell ordering must match the ordering used to build the
    # submesh. Firedrake may reorder cells; if the highlighted set looks wrong,
    # use USE_FIREDRAKE = True which guarantees consistent ordering.
    return pts, tri, reduced_cells, []   # cyl filled by detect_cylinders below


# ---------------------------------------------------------------------------
# If auto-detection is ever wrong, set the three cylinders explicitly here
# as [(cx, cy, R), (cx, cy, R), (cx, cy, R)] and detection is skipped.
CYLINDERS_OVERRIDE = None


def detect_cylinders(pts, tris):
    """
    Find the pinball cylinders as *closed interior boundary loops*.

    Boundary edges (belonging to one triangle) are grouped into connected
    loops; each loop touching neither the domain bbox is treated as a
    cylinder, with center = loop centroid and R = median vertex distance.
    Robust to the outer channel walls, which form the one loop we discard.
    """
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

    # adjacency among boundary vertices, then split into connected loops
    adj = defaultdict(set)
    for a, b in bnd_edges:
        adj[a].add(b); adj[b].add(a)

    unseen = set(adj)
    loops = []
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
        touches_bbox = (P[:,0].min() < xmin + tol or P[:,0].max() > xmax - tol or
                        P[:,1].min() < ymin + tol or P[:,1].max() > ymax - tol)
        if touches_bbox:
            continue                      # outer channel walls -> skip
        c = P.mean(0)
        d = np.hypot(P[:,0]-c[0], P[:,1]-c[1])
        r = float(np.median(d))
        if not (0 < r < 0.25 * span):
            continue
        # circularity: real cylinders have near-constant radius; mesh gaps
        # and ragged internal boundaries do not. Reject high radius variance.
        if len(d) >= 6 and np.std(d) / (r + 1e-30) < 0.15:
            cyl.append((float(c[0]), float(c[1]), r))
    return cyl


def build_figure(pts, tris, cyl, selected):
    fig = plt.figure(figsize=(13.5, 6.0))
    gs = fig.add_gridspec(2, 3, width_ratios=[0.95, 1.05, 2.5],
                          height_ratios=[1, 1], wspace=0.10, hspace=0.30,
                          left=0.02, right=0.99, top=0.86, bottom=0.05)

    # LEFT TOP: residual DEIM vector (didactic)
    axr = fig.add_subplot(gs[0, 0]); axr.set_xlim(0,1); axr.set_ylim(0,1); axr.axis('off')
    axr.set_title("Residual  (DEIM)", fontsize=10, pad=4)
    nb, sel_rows = 22, {3,7,8,14,18}
    for i in range(nb):
        y0 = 0.88 - i*(0.78/nb)
        axr.add_patch(Rectangle((0.34, y0), 0.15, 0.78/nb*0.85,
                      fc=ACCENT if i in sel_rows else GREY, ec='white', lw=0.4))
    axr.text(0.415, 0.95, r"$\mathbf{r}\in\mathbb{R}^{N_h}$", ha='center', fontsize=9)
    axr.annotate("", xy=(0.49,0.55), xytext=(0.66,0.60),
                 arrowprops=dict(arrowstyle='->', color=ACCENT, lw=0.9))
    axr.text(0.70, 0.585, r"$m_r$"+"\nindices", fontsize=7.5, color=ACCENT, va='center')
    axr.text(0.03, 0.22, r"$\mathbf{P}^{T}\mathbf{r}$", color=ACCENT, fontsize=9)

    # LEFT BOTTOM: jacobian MDEIM sparse matrix (didactic)
    axj = fig.add_subplot(gs[1, 0]); axj.set_xlim(0,1); axj.set_ylim(0,1); axj.axis('off')
    axj.set_title("Jacobian  (MDEIM)", fontsize=10, pad=4)
    ng, x0, y0m, side = 12, 0.24, 0.14, 0.60
    rngm = np.random.default_rng(1); occ = set()
    for i in range(ng):
        for j in range(ng):
            if abs(i-j) <= 1 or rngm.random() < 0.07: occ.add((i,j))
    sel_entries = set(list(occ)[::6][:7])
    for (i,j) in occ:
        xx = x0 + j*(side/ng); yy = y0m + (ng-1-i)*(side/ng)
        if (i,j) in sel_entries:
            axj.plot(xx+side/ng/2, yy+side/ng/2, 's', color=ACCENT2, ms=5)
        else:
            axj.plot(xx+side/ng/2, yy+side/ng/2, '.', color=GREY, ms=3)
    axj.add_patch(Rectangle((x0,y0m), side, side, fill=False, ec=DARK, lw=0.7))
    axj.text(x0+side/2, y0m+side+0.05, r"$\mathbf{J}$  nonzeros", ha='center', fontsize=9)
    axj.text(0.5, 0.03, r"entries $(i,j)$", ha='center', color=ACCENT2, fontsize=8)

    # MIDDLE: bijection (didactic)
    axb = fig.add_subplot(gs[:, 1]); axb.set_xlim(0,1); axb.set_ylim(0,1); axb.axis('off')
    axb.set_title("entry $\\to$ index $\\to$ cells", fontsize=10, pad=4)
    def tri_patch(ax, ox, oy, scale, support_idx, col):
        ang = np.linspace(0, 2*np.pi, 7)[:-1]
        ring = np.column_stack([ox+scale*np.cos(ang), oy+scale*np.sin(ang)])
        hubp = np.array([ox, oy]); polys, cols = [], []
        for t in range(6):
            polys.append([hubp, ring[t], ring[(t+1)%6]])
            cols.append(col if t in support_idx else 'white')
        ax.add_collection(PolyCollection(polys, facecolors=cols, edgecolors=DARK, lw=0.5, alpha=0.9))
        ax.plot(ox, oy, 'o', color=col, ms=4)
    tri_patch(axb, 0.5, 0.70, 0.17, {0,1,2,3,4,5}, ACCENT)
    axb.text(0.5, 0.92, "row $i\\,\\to$ DOF\n$\\to$ all support cells", ha='center', fontsize=8, color=ACCENT)
    tri_patch(axb, 0.5, 0.24, 0.17, {1,2}, ACCENT2)
    axb.text(0.5, 0.45, "entry $(i,j)\\,\\to$\nshared cells only", ha='center', fontsize=8, color=ACCENT2)

    # RIGHT: REAL mesh + REAL submesh
    axm = fig.add_subplot(gs[:, 2]); axm.set_aspect('equal'); axm.axis('off')
    axm.set_title("Reduced integration domain on the pinball mesh", fontsize=10, pad=4)
    triang = mtri.Triangulation(pts[:,0], pts[:,1], tris)
    axm.triplot(triang, color=GREY, lw=0.18, alpha=0.85)
    axm.add_collection(PolyCollection([pts[tris[c]] for c in selected],
                       facecolors=ACCENT, edgecolors=ACCENT, lw=0.3, alpha=0.92))
    for (ccx, ccy, R) in cyl:
        axm.add_patch(Circle((ccx, ccy), R, fc='white', ec=DARK, lw=1.1, zorder=5))
    axm.set_xlim(pts[:,0].min(), pts[:,0].max())
    axm.set_ylim(pts[:,1].min(), pts[:,1].max())
    frac = len(selected)/len(tris)*100
    axm.text(0.5, -0.04,
             r"assembly restricted to $|\mathcal{K}|\ll N_{\mathrm{cells}}$"
             rf"   ($|\mathcal{{K}}|={len(selected)}$, $\approx{frac:.1f}\%$ of cells)",
             transform=axm.transAxes, ha='center', fontsize=9, color=ACCENT)

    def connect(a,b,c,dd,col):
        fig.add_artist(FancyArrowPatch((a,b),(c,dd), transform=fig.transFigure,
            arrowstyle='-|>', mutation_scale=11, lw=1.0, color=col, zorder=20))
    connect(0.165, 0.64, 0.235, 0.64, ACCENT)
    connect(0.165, 0.28, 0.235, 0.28, ACCENT2)
    connect(0.415, 0.5, 0.48, 0.5, DARK)

    fig.text(0.5, 0.95,
        "Online assembly of residual and Jacobian is evaluated on the submesh "
        r"$\mathcal{K}$ only $\;-\;$ cost independent of $N_h$",
        ha='center', fontsize=10.5, color=DARK)
    return fig


if __name__ == "__main__":
    if USE_FIREDRAKE:
        pts, tris, selected, cyl = get_cells_and_mesh_firedrake()
    else:
        pts, tris, selected, cyl = get_cells_and_mesh_numpy()

    if not cyl:                       # tag-based failed or numpy path
        cyl = detect_cylinders(pts, tris)
        print("geometric cylinder detection:", cyl)
    if not cyl:
        print("!! no cylinders found; set CYLINDERS_OVERRIDE at top of file.")

    fig = build_figure(pts, tris, cyl, selected)
    os.makedirs(OUTDIR, exist_ok=True)
    stem = os.path.join(OUTDIR, OUT_STEM)
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=160, bbox_inches="tight")
    print(f"  wrote {stem}.pdf / .png")
