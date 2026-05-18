"""
Local ROM pipeline with optional DEIM support.

Configuration is held in `cfg` (LocalROMConfig).
Paths, test parameters, and sweep ranges remain as module-level constants.

File structure:
    local_rom/K{cfg.n_clusters}_{cfg.inner_product_type}/
        clustering/
        cluster_0/
            pod_basis/
            operators/
            deim_basis.npz      (if cfg.use_deim)
            deim_ops.npz        (if cfg.use_deim)
        cluster_1/
            ...
"""

import numpy as np

from nsrom.navier_stokes import setup_navier_stokes_problem, load_solution
from nsrom.snapshot_collection import load_snapshot_dofs
from nsrom.lifting_functions import load_lifting_functions
from nsrom.cluster_building import build_energy_snapshots, compute_or_load_clustering
from nsrom.local_rom import (
    build_local_roms,
    solve_at_parameter,
    compare_rom_vs_fom,
    precompute_fast_selection,
    precompute_change_of_basis,
)
from nsrom.layout import LocalROMLayout
from nsrom.config import LocalROMConfig, RunManifest, save_run

from sweep import build_snake_path, run_sweep, save_sweep_results
from plot import plot_lift_3d


# =============================================================================
# CONFIGURATION
# =============================================================================

cfg = LocalROMConfig(
    n_clusters=5,
    inner_product_type="H1",
    pod_energy_tol=1e-8,
    n_velocity_max=50,
    n_pressure_max=50,
    n_supremizer_max=50,
    boundary_markers=(1, 3, 4, 5, 6),
    use_deim=True,
    compute_affine_convection=True,
    deim_energy_tol=1e-8,
    m_F_max=80,
    m_J_max=80,
    n_modes_F=None,
    n_modes_J=None,
    recompute_clustering=False,
    recompute_pod=False,
    recompute_deim=False,
)

# --- Paths ---
SNAPSHOT_DIR      = "multi_param_multi_branch"
DEIM_SNAPSHOT_DIR = "multi_param_multi_branch"
LIFTING_DIR       = "lifting"
MESH_FILE         = "mesh/mid_pinball.msh"
FOM_CHECKPOINT    = "initial_states/velocity_checkpoint_down"

REYNOLDS_INIT  = 100.0
AMPLITUDE_INIT = 0.0

# --- Test parameters ---
TEST_RE  = 80.0
TEST_AMP = 0.1

# --- Sweep ---
RE_VALUES  = np.arange(30.0, 90.0, 10.0)
AMP_VALUES = np.arange(-1.0, 1.0, 0.1)
FOM_EVERY  = 0       # 0 = skip FOM, N = compare every Nth point
RUN_SWEEP  = False


# =============================================================================
# MANIFEST and LAYOUT
# =============================================================================

manifest = RunManifest(
    snapshot_dir=SNAPSHOT_DIR,
    deim_snapshot_dir=DEIM_SNAPSHOT_DIR if cfg.use_deim else None,
    lifting_dir=LIFTING_DIR,
    mesh_file=MESH_FILE,
    fom_checkpoint=FOM_CHECKPOINT,
    reynolds_init=REYNOLDS_INIT,
    amplitude_init=AMPLITUDE_INIT,
    test_re=TEST_RE,
    test_amp=TEST_AMP,
    run_sweep=RUN_SWEEP,
    fom_every=FOM_EVERY,
    re_values=RE_VALUES.tolist(),
    amp_values=AMP_VALUES.tolist(),
)

layout = LocalROMLayout(
    base_dir="local_rom",
    n_clusters=cfg.n_clusters,
    inner_product_type=cfg.inner_product_type,
)


# =============================================================================
# MAIN
# =============================================================================

def main():
    save_run(cfg, manifest, layout.run_json())

    mode_str   = "DEIM" if cfg.use_deim else "Exact"
    affine_str = "Affine" if cfg.compute_affine_convection else "Full"

    print("=" * 80)
    print(f"LOCAL ROM — {mode_str} solver, {affine_str} convection, K={cfg.n_clusters}")
    print("=" * 80)
    print(f"  USE_DEIM:    {cfg.use_deim}")
    print(f"  AFFINE:      {cfg.compute_affine_convection}")
    print(f"  ROM modes:   tol={cfg.pod_energy_tol}, max={cfg.n_velocity_max}")
    print(f"  Inner prod:  {cfg.inner_product_type}")
    print(f"  Test params: Re={TEST_RE}, amp={TEST_AMP}")
    if cfg.use_deim:
        print(f"  DEIM:        tol={cfg.deim_energy_tol}, max_F={cfg.m_F_max}, max_J={cfg.m_J_max}")
    print()

    # --- Setup problem ---
    print("--- Setup problem ---")
    problem = setup_navier_stokes_problem(
        mesh_file=MESH_FILE,
        reynolds_init=REYNOLDS_INIT,
        amplitude_init=AMPLITUDE_INIT,
    )
    initial_solution = load_solution(FOM_CHECKPOINT, problem)

    # --- Load snapshots and lifting ---
    print("\n--- Load snapshots and lifting ---")
    snapshot_data = load_snapshot_dofs(SNAPSHOT_DIR)
    lifting       = load_lifting_functions(LIFTING_DIR, problem)

    velocity_snapshots = snapshot_data['velocity']
    pressure_snapshots = snapshot_data['pressure']
    parameters         = snapshot_data['parameters']
    lift_coefficients  = snapshot_data['lift_coefficients']

    print(f"  Snapshots:  {velocity_snapshots.shape[0]}")
    print(f"  Parameters: {len(parameters)} entries")

    # --- Energy-norm transform ---
    print("\n--- Energy-norm transform ---")

    S_homo, S_tilde, M_u, M_p, L, P = build_energy_snapshots(
        velocity_snapshots, parameters, lifting, problem, cfg.inner_product_type,
        homogenize = False
    )
    print(f"  S_tilde: {S_tilde.shape}")
    method = "pod_whitened" #"pod_whitened" or "cholesky"

    # --- Clustering ---
    print("\n--- Clustering ---")
    clustering, reason = compute_or_load_clustering(
        S_tilde, S_homo, M_u,
        parameters, L, P,
        save_dir=layout.clustering_dir,
        n_clusters=cfg.n_clusters,
        method=method,   # or "pod_whitened" or "cholesky" with rank=...
        rank=10,           # or e.g. 15
        recompute=cfg.recompute_clustering,
    )

    print(f"  {reason}")
    clustering.summary()
    plot_lift_3d(
        parameters,
        lift_coefficients,
        clustering.labels,
        title=f"Clusters (method={method}, K={clustering.n_clusters})",
    )
    # --- Local ROMs ---
    print("\n--- Local ROMs ---")
    deim_snapshot_data = None
    if cfg.use_deim:
        deim_snapshot_data = load_snapshot_dofs(DEIM_SNAPSHOT_DIR)
        print(f"  DEIM snapshots: {deim_snapshot_data['velocity'].shape[0]}")

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

    for k in range(clustering.n_clusters):
        basis_k = bases[k]
        print(f"  Cluster {k}: {basis_k.n_velocity_modes} vel + "
              f"{basis_k.n_supremizer_modes} sup + "
              f"{basis_k.n_pressure_modes} pres")
        if cfg.use_deim:
            print(f"             DEIM m_F={deim_ops_dict[k].m_F}, m_J={deim_ops_dict[k].m_J}")

    # --- Online: single point or sweep ---
    if RUN_SWEEP:
        print("\n--- Precompute fast cluster selection ---")
        precompute_fast_selection(clustering, operators, M_u)

        print("\n--- Precompute change-of-basis ---")
        precompute_change_of_basis(clustering, operators, M_u, M_p)

        print("\n--- Parameter sweep ---")
        path = build_snake_path(RE_VALUES, AMP_VALUES)
        print(f"  Grid: {len(RE_VALUES)} Re × {len(AMP_VALUES)} amp = {len(path)} points")
        print(f"  FOM comparison: every {FOM_EVERY} points" if FOM_EVERY > 0 else "  FOM comparison: off")

        results = run_sweep(
            path, clustering, operators, deim_ops_dict,
            problem, initial_solution, M_u, M_p,
            use_deim=cfg.use_deim,
            fom_every=FOM_EVERY,
        )

        n_conv = sum(1 for r in results if r['converged'])
        print(f"\n  Converged:      {n_conv}/{len(results)}")
        print(f"  Total ROM time: {sum(r['t_rom'] for r in results):.1f}s")

        if FOM_EVERY > 0:
            fom_pts = [r for r in results if 'u_err' in r]
            if fom_pts:
                u_errs = [r['u_err'] for r in fom_pts]
                print(f"  Velocity error: mean={np.mean(u_errs)*100:.3f}%, max={np.max(u_errs)*100:.3f}%")

        save_sweep_results(results, layout.sweep_csv())

    else:
        print(f"\n--- Online solve (Re={TEST_RE}, amp={TEST_AMP}) ---")
        rom_result = solve_at_parameter(
            Re=TEST_RE,
            amp=TEST_AMP,
            clustering=clustering,
            operators=operators,
            deim_ops_dict=deim_ops_dict,
            problem=problem,
            initial_solution=initial_solution,
            M_u=M_u,
            M_p=M_p,
            use_deim=cfg.use_deim,
        )
        rom_result.summary()

        print("\n--- FOM comparison ---")
        comp = compare_rom_vs_fom(rom_result, problem, initial_solution, M_u, M_p)

        k          = rom_result.cluster
        solution   = rom_result
        deim_ops_k = deim_ops_dict[k]

        print(f"\n{'=' * 80}")
        print(f"LOCAL ROM — {mode_str} solver, {affine_str} convection, K={cfg.n_clusters}")
        print(f"{'=' * 80}")
        print(f"  Re={TEST_RE}, amp={TEST_AMP}")
        print(f"  Cluster selected:  {k}")
        print(f"  ROM modes:         {operators[k].n_velocity_modes} vel × {operators[k].n_pressure_modes} pres")
        print(f"  Affine convection: {operators[k].has_affine_convection}")
        if cfg.use_deim:
            print(f"  DEIM:              m_F={deim_ops_k.m_F}, m_J={deim_ops_k.m_J}")
        print(f"  Converged:         {solution.converged} ({solution.iterations} iterations)")
        comp.summary(u_norm_label=cfg.inner_product_type)
        print(f"{'=' * 80}")


if __name__ == '__main__':
    main()