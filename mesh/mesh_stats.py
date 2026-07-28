#!/usr/bin/env python3
"""Report mesh statistics from a Gmsh .msh file for the paper's mesh table.

Usage:  python mesh_stats.py pinball_coarse.msh [--cyl 5 6 7] [--D 1.0]
"""
import argparse
import numpy as np
import meshio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mshfile")
    ap.add_argument("--cyl", type=int, nargs="*", default=None,
                    help="physical tags of the cylinder boundaries")
    ap.add_argument("--D", type=float, default=1.0, help="cylinder diameter")
    args = ap.parse_args()

    m = meshio.read(args.mshfile)
    P = m.points[:, :2]

    print(f"=== {args.mshfile} ===")
    if m.field_data:
        print("\nphysical groups (name -> tag, dim):")
        for name, (tag, dim) in sorted(m.field_data.items(), key=lambda kv: kv[1][0]):
            print(f"  {name:<20s} tag={tag:<4d} dim={dim}")

    # ---------------------------------------------------------------- cells
    tri = m.cells_dict["triangle"]
    nv = np.unique(tri).size

    E = np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    E = np.unique(np.sort(E, axis=1), axis=0)
    L = np.linalg.norm(P[E[:, 0]] - P[E[:, 1]], axis=1)

    v0, v1, v2 = P[tri[:, 0]], P[tri[:, 1]], P[tri[:, 2]]
    e1, e2 = v1 - v0, v2 - v0
    A = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    hcell = np.max(np.linalg.norm(
        np.stack([v1 - v0, v2 - v1, v0 - v2]), axis=2), axis=0)  # cell diameter

    print("\n--- global ---")
    print(f"  triangles          {tri.shape[0]:>10d}")
    print(f"  vertices           {nv:>10d}")
    print(f"  edges              {E.shape[0]:>10d}")
    print(f"  Taylor-Hood dofs   {2 * (nv + E.shape[0]) + nv:>10d}"
          f"   (P2 vel: {2 * (nv + E.shape[0])}, P1 pres: {nv})")
    print(f"  h_min / h_max      {hcell.min():.4e} / {hcell.max():.4e}"
          f"   (grading {hcell.max() / hcell.min():.1f}x)")
    print(f"  edge len mean      {L.mean():.4e}")
    print(f"  area min / total   {A.min():.4e} / {A.sum():.4e}")

    # ------------------------------------------------------- near-cylinder
    if args.cyl is None:
        print("\n(no --cyl tags given; skipping near-wall resolution)")
        return

    lin = m.cells_dict["line"]
    tag = m.cell_data_dict["gmsh:physical"]["line"]

    print("\n--- cylinder boundaries ---")
    for t in args.cyl:
        seg = lin[tag == t]
        if seg.size == 0:
            print(f"  tag {t}: no line elements found")
            continue
        ls = np.linalg.norm(P[seg[:, 0]] - P[seg[:, 1]], axis=1)
        perim = ls.sum()
        print(f"  tag {t}: {len(seg):>4d} segments   "
              f"h_wall = {ls.mean():.4e} (min {ls.min():.4e}, max {ls.max():.4e})   "
              f"h/D = {ls.mean() / args.D:.4e}   perimeter = {perim:.4f}")

    # cells touching any cylinder
    wall_nodes = np.unique(lin[np.isin(tag, args.cyl)])
    touching = np.isin(tri, wall_nodes).any(axis=1)
    print(f"\n  cells adjacent to cylinders: {touching.sum()}")
    print(f"  their h: mean {hcell[touching].mean():.4e}, "
          f"min {hcell[touching].min():.4e}")
    print(f"  refinement ratio h_far/h_wall: "
          f"{hcell[~touching].mean() / hcell[touching].mean():.2f}")
    from scipy.spatial import cKDTree
    d, _ = cKDTree(P).query(P * [1, -1])
    print(f"mesh asymmetry under y -> -y: max node mismatch {d.max():.3e}")


if __name__ == "__main__":
    main()
