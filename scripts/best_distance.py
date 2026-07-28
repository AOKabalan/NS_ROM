from firedrake import *
import numpy as np
from nsrom.navier_stokes import setup_navier_stokes_problem, load_solution
from nsrom.snapshot_collection import load_snapshot_dofs
from nsrom.lifting_functions import load_lifting_functions
from nsrom.cluster_building import load_clustering_result
from nsrom.layout import LocalROMLayout
from nsrom.config import LocalROMConfig, RunManifest, save_run
from nsrom.local_rom import build_local_roms
from nsrom.rom import solve_rom, reconstruct_solution, project_to_rom_coefficients, SubMeshDEIM

# I want to load snapshots
# project on different local ROMs
# calculate the errors of each
# decide which local ROM is best for each snapshot point

SNAPSHOT_DIR      = "multi_param_multi_branch"
DEIM_SNAPSHOT_DIR = "multi_param_multi_branch"
LIFTING_DIR       = "lifting"
MESH_FILE         = "mesh/mid_pinball.msh"
FOM_CHECKPOINT    = "initial_states/velocity_checkpoint_symm"
REYNOLDS_INIT  = 100.0
AMPLITUDE_INIT = 0.0



def project_on_all_local_roms(vel_data, pres_data, clustering, bases, operators):
    for i in range(len(vel_data)):
        for k in range(clustering.n_clusters):
            src_vel = vel_data[i]
            src_pres = pres_data[i]
            u_coeffs, p_coeffs = project_to_rom_coefficients(
                src_vel, src_pres, operators[k], amp=amp, M_u=M_u, M_p=M_p,
            )



def main():
    cfg = LocalROMConfig(
        n_clusters=4,
        inner_product_type="H1",# can only use semi when no need for cholesky aka use pod before clustering
        pod_energy_tol=1e-8,
        n_velocity_max=50,
        n_pressure_max=50,
        n_supremizer_max=50,
        boundary_markers=(1, 3, 4, 5, 6),
        use_deim= True,
        compute_affine_convection=True,
        # deim_energy_tol=1e-12,
        deim_energy_tol=1e-10,
        m_F_max=80,
        m_J_max=80,
        n_modes_F=None,
        n_modes_J=None,
        recompute_clustering=False,
        recompute_pod=False,
        recompute_deim=False,
    )
    problem = setup_navier_stokes_problem(
        mesh_file=MESH_FILE,
        reynolds_init=REYNOLDS_INIT,
        amplitude_init=AMPLITUDE_INIT,
    )
    # --- Load snapshots and lifting ---
    print("\n--- Load snapshots and lifting ---")
    snapshot_data = load_snapshot_dofs(SNAPSHOT_DIR)
    lifting       = load_lifting_functions(LIFTING_DIR, problem)

    velocity_snapshots = snapshot_data['velocity']
    pressure_snapshots = snapshot_data['pressure']
    parameters         = snapshot_data['parameters']
    lift_coefficients  = snapshot_data['lift_coefficients']
    branch_ids = snapshot_data['branch_ids']

    print(f"  Snapshots:  {velocity_snapshots.shape[0]}")
    print(f"  Parameters: {len(parameters)} entries")
    layout = LocalROMLayout(
        base_dir="local_rom",
        n_clusters=cfg.n_clusters,
        inner_product_type=cfg.inner_product_type,
    )

    clustering = load_clustering_result(layout.clustering_dir)
    deim_snapshot_data = load_snapshot_dofs(DEIM_SNAPSHOT_DIR)

    bases, operators, deim_ops_dict = build_local_roms(
        clustering=clustering,
        velocity_snapshots=velocity_snapshots,
        pressure_snapshots=pressure_snapshots,
        deim_snapshot_data=deim_snapshot_data,
        deim_snapshot_dir=DEIM_SNAPSHOT_DIR,
        lifting=lifting,
        problem=problem,
        cfg=cfg,
        layout=layout,
    )

if __name__ == "__main__":
    main()
