"""
Reverse-engineer the cluster assignment.

For every FOM snapshot:
  1. project it onto EVERY local ROM velocity basis  -> M_u-orthogonal
     representation error per cluster  -> proj-argmin ("best representer")
  2. compute the whitened global-POD distance to every cluster centroid
     -> metric-argmin (this is exactly the k-means nearest-centroid rule)
  3. line both up against the true branch label.

The interesting region is the near-onset asymmetric snapshots, where
"best representer" (energy) and "branch identity" (whitened) disagree.
Everything is done in the SAME homogeneous space + M_u inner product the
bases were built in.
"""

from firedrake import *
import numpy as np

from nsrom.navier_stokes import setup_navier_stokes_problem, load_solution
from nsrom.snapshot_collection import load_snapshot_dofs
from nsrom.lifting_functions import load_lifting_functions
from nsrom.cluster_building import (
    load_clustering_result,
    build_energy_snapshots,
)
from nsrom.rom import homogenize_snapshots
from nsrom.layout import LocalROMLayout
from nsrom.config import LocalROMConfig
from nsrom.local_rom import build_local_roms

SNAPSHOT_DIR      = "multi_param_multi_branch"
DEIM_SNAPSHOT_DIR = "multi_param_multi_branch"
LIFTING_DIR       = "lifting"
MESH_FILE         = "mesh/mid_pinball.msh"
FOM_CHECKPOINT    = "initial_states/velocity_checkpoint_symm"
REYNOLDS_INIT  = 100.0
AMPLITUDE_INIT = 0.0
OUT_NPZ  = "cluster_reverse_engineering.npz"


# ---------------------------------------------------------------------------
# core numerics
# ---------------------------------------------------------------------------
def projection_errors(S_homo, operators, M_u, K):
    """
    Relative M_u-orthogonal representation error of each homogenized snapshot
    (columns of S_homo, shape (n_dofs, n_snap)) onto every cluster's velocity
    basis Z_u.  Returns err[k, i]  (K x n_snap).

    Orthogonal projection identity used:
        ||u - Z a||_M^2 = ||u||_M^2 - a^T (Z^T M u),   Gram a = Z^T M u
    so no field reconstruction is needed.
    """
    n_snap = S_homo.shape[1]
    Mu_S   = M_u @ S_homo                                # (n_dofs, n_snap)
    norm2  = np.einsum('ij,ij->j', S_homo, Mu_S)          # (n_snap,) ||u||_M^2
    norm2  = np.maximum(norm2, 1e-30)

    err = np.empty((K, n_snap))
    for k in range(K):
        Z    = operators[k].Z_u                           # (n_dofs, n_modes_k)
        MZ   = M_u @ Z                                     # (n_dofs, n_modes_k)
        Gram = Z.T @ MZ                                    # (n_modes_k, n_modes_k)
        B    = MZ.T @ S_homo                              # (n_modes_k, n_snap) = Z^T M u
        A    = np.linalg.solve(Gram, B)                   # projection coeffs
        proj2 = np.einsum('ij,ij->j', A, B)               # ||proj||_M^2 = a^T B
        err[k] = np.sqrt(np.maximum(norm2 - proj2, 0.0) / norm2)
    return err


def whitened_metric_distances(S_homo, clustering, M_u):
    """
    Whitened global-POD distance from each snapshot to every cluster centroid
    -- identical rule to the pod_whitened k-means assignment.
    Returns dist[k, i]  (K x n_snap).
    """
    for attr in ("pod_modes", "pod_sigma", "centers_pod"):
        if not hasattr(clustering, attr):
            raise AttributeError(
                f"clustering.{attr} missing -- need the pod_whitened artifacts.")

    Phi   = clustering.pod_modes                 # (n_dofs, rank), M_u-orthonormal
    sigma = clustering.pod_sigma                 # (rank,)
    cw    = clustering.centers_pod / sigma[None, :]        # (K, rank) whitened centroids

    G = (Phi.T @ (M_u @ S_homo)) / sigma[:, None]         # (rank, n_snap) whitened coords
    # dist[k,i] = || cw[k] - G[:,i] ||
    dist = np.sqrt(((cw[:, :, None] - G[None, :, :]) ** 2).sum(axis=1))   # (K, n_snap)
    return dist


# ---------------------------------------------------------------------------
def analyze(S_homo, params, C_L, branch_ids, clustering, operators, M_u):
    K      = clustering.n_clusters
    Re     = np.asarray(params)[:, 0]
    amp    = np.asarray(params)[:, 1]
    C_L    = np.asarray(C_L)
    branch = np.asarray(branch_ids)

    perr = projection_errors(S_homo, operators, M_u, K)          # (K, n_snap)
    mdist = whitened_metric_distances(S_homo, clustering, M_u)   # (K, n_snap)

    proj_best   = perr.argmin(axis=0)
    metric_best = mdist.argmin(axis=0)

    # ---- summary -------------------------------------------------------
    n = S_homo.shape[1]
    print(f"\n=== {n} snapshots, K={K} clusters ===")
    print(f"  metric-argmin == clustering label : "
          f"{(metric_best == clustering.labels).mean()*100:5.1f}%  "
          f"(sanity check, expect ~100%)")
    print(f"  proj-argmin   == metric-argmin     : "
          f"{(proj_best == metric_best).mean()*100:5.1f}%")

    # where do representer and branch-metric disagree?  -> the interesting set
    disagree = proj_best != metric_best
    print(f"  disagreement snapshots            : {disagree.sum()} / {n}")
    if disagree.any():
        print("  |C_L| range of disagreement       : "
              f"[{np.abs(C_L[disagree]).min():.4f}, "
              f"{np.abs(C_L[disagree]).max():.4f}]   "
              f"(low |C_L| = near onset)")

    # ---- per-snapshot table, on-axis asymmetric branches, sorted by Re --
    amps_unique = np.unique(amp)
    for a in amps_unique:
        for br in sorted(set(branch.tolist())):
            # sel = (branch == br) & (np.abs(amp) < 1e-9)
            sel = (branch == br) & (np.abs(amp - a) < 1e-9)
            if sel.sum() == 0:
                continue
            order = np.argsort(Re[sel])
            idx   = np.where(sel)[0][order]
            print(f"\n--- branch {br}, amp={a} : Re | C_L | proj_best | metric_best "
                  f"| perr[each k] ---")
            for i in idx:

                pe = " ".join(f"{perr[k, i]:.4f}" for k in range(K))
                flag = "  <-- differ" if proj_best[i] != metric_best[i] else ""
                print(f"  Re={Re[i]:6.2f}  C_L={C_L[i]:+.4f}  "
                    f"proj={proj_best[i]}  metric={metric_best[i]}  "
                    f"[{pe}]{flag}")

    # ---- crossover Re*: cluster-2 vs pure cluster, per on-axis branch ---
    # pure cluster for a branch = the cluster that owns most of that branch's
    # well-developed (high-|C_L|) snapshots.
    print("\n=== representation crossover (proj err: cluster-2 vs pure) ===")
    for br in sorted(set(branch.tolist())):
        for a in amps_unique:
            sel = (branch == br) & (np.abs(amp - a) < 1e-9)
            if sel.sum() == 0:
                continue
            idx   = np.where(sel)[0]
            hi    = idx[np.abs(C_L[idx]) > 0.9 * np.abs(C_L[idx]).max()]
            pure  = int(np.bincount(metric_best[hi], minlength=K).argmax())
            if pure == 2:
                continue  # this branch already lives in the symmetric carrier
            order = idx[np.argsort(Re[idx])]
            crossed = None
            for i in order:
                if perr[pure, i] < perr[2, i] and crossed is None:
                    crossed = Re[i]
        print(f"  branch {br}: pure cluster = {pure};  "
              f"pure overtakes cluster-2 at Re* ~ "
              f"{crossed if crossed is not None else 'n/a'}")

    np.savez(OUT_NPZ,
             Re=Re, amp=amp, C_L=C_L, branch=branch,
             proj_err=perr, metric_dist=mdist,
             proj_best=proj_best, metric_best=metric_best,
             clustering_labels=clustering.labels)
    print(f"\nsaved -> {OUT_NPZ}")
    return perr, mdist, proj_best, metric_best


# ---------------------------------------------------------------------------
def main():
    cfg = LocalROMConfig(
        n_clusters=4,
        inner_product_type="H1",
        pod_energy_tol=1e-8,
        n_velocity_max=50, n_pressure_max=50, n_supremizer_max=50,
        boundary_markers=(1, 3, 4, 5, 6),
        use_deim=True,
        compute_affine_convection=True,
        deim_energy_tol=1e-10,
        m_F_max=80, m_J_max=80,
        n_modes_F=None, n_modes_J=None,
        recompute_clustering=False, recompute_pod=False, recompute_deim=False,
    )
    problem = setup_navier_stokes_problem(
        mesh_file=MESH_FILE,
        reynolds_init=REYNOLDS_INIT,
        amplitude_init=AMPLITUDE_INIT,
    )

    print("\n--- Load snapshots and lifting ---")
    snapshot_data = load_snapshot_dofs(SNAPSHOT_DIR)
    lifting       = load_lifting_functions(LIFTING_DIR, problem)

    velocity_snapshots = snapshot_data['velocity']
    pressure_snapshots = snapshot_data['pressure']
    parameters         = snapshot_data['parameters']
    lift_coefficients  = snapshot_data['lift_coefficients']
    branch_ids         = snapshot_data['branch_ids']
    print(f"  Snapshots:  {velocity_snapshots.shape[0]}")
    print(f"  Parameters: {len(parameters)} entries")

    layout = LocalROMLayout(
        base_dir="local_rom",
        n_clusters=cfg.n_clusters,
        inner_product_type=cfg.inner_product_type,
    )
    clustering = load_clustering_result(layout.clustering_dir)
    deim_snapshot_data = load_snapshot_dofs(DEIM_SNAPSHOT_DIR)

    # M_u / M_p in the SAME space the bases use (homogenize=True)
    _S_homo, _S_tilde, M_u, M_p, _L, _P = build_energy_snapshots(
        velocity_snapshots, parameters, lifting, problem,
        cfg.inner_product_type, homogenize=True,
    )

    bases, operators, deim_ops_dict = build_local_roms(
        clustering=clustering,
        velocity_snapshots=velocity_snapshots,
        pressure_snapshots=pressure_snapshots,
        deim_snapshot_data=deim_snapshot_data,
        deim_snapshot_dir=DEIM_SNAPSHOT_DIR,
        lifting=lifting, problem=problem, cfg=cfg, layout=layout,
    )

    # homogenize snapshots exactly as build_energy_snapshots does
    u_base    = lifting.u_base.dat.data_ro.flatten()
    u_control = lifting.u_control.dat.data_ro.flatten()
    S_homo_trans= homogenize_snapshots(velocity_snapshots, parameters, u_base, u_control)
    S_homo = S_homo_trans.T
    #   S_homo: (n_dofs, n_snap)   -- columns are homogeneous snapshots

    analyze(S_homo, parameters, lift_coefficients, branch_ids,
            clustering, operators, M_u)


if __name__ == "__main__":
    main()
