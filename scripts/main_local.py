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

from nsrom.helper_functions import dofs_to_function
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
from nsrom.bifurcation.detection import (
    build_reduced_velocity_l2_mass_matrices,
    find_critical_curve,
    save_critical_curve,
)

from sweep import (
    build_fixed_amp_path,
    build_snake_path,
    build_hubspoke_path,
    run_sweep,
    run_bifurcation_sweep,
    save_sweep_results,
    run_sweep_diagnostic,
    run_sweep_simple,
    build_bifurcation_diagram,
    build_full_diagram,
    find_critical_curve_continuation,
)
from plot import plot_lift_3d

# =============================================================================
# CONFIGURATION
# =============================================================================

cfg = LocalROMConfig(
    n_clusters=1,
    inner_product_type="H1",# can only use semi when no need for cholesky aka use pod before clustering
    pod_energy_tol=1e-8,
    n_velocity_max=70,
    n_pressure_max=50,
    n_supremizer_max=50,
    boundary_markers=(1, 3, 4, 5, 6),
    use_deim= False,
    use_tensor= True,
    compute_affine_convection=True,
    # deim_energy_tol=1e-12,
    deim_energy_tol_F=1e-16,
    deim_energy_tol_J=1e-16,
    m_F_max=400,
    m_J_max=400,
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
FOM_CHECKPOINT    = "initial_states/base_velocity"

REYNOLDS_INIT  = 100.0
AMPLITUDE_INIT = 0.0

# --- Test parameters ---
TEST_RE  = 80.0
TEST_AMP = 0.1

# --- Sweep ---
RE_VALUES  = np.arange(50.0, 100.0, 1.0)
AMP_VALUES = np.arange(-1.0, 1.0, 0.1)
AMP_FIXED = 0.0
RE_FIXED = 80
FOM_EVERY  = 0       # 0 = skip FOM, N = compare every Nth point
RUN_SWEEP  = True


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
    if cfg.use_deim and cfg.use_tensor:

        raise ValueError("USE_DEIM and USE_TENSOR cannot both be True — choose one (or set both to False).")
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
        print(f"  DEIM:        tol_F={cfg.deim_energy_tol_F}, tol_J={cfg.deim_energy_tol_J}, max_F={cfg.m_F_max}, max_J={cfg.m_J_max}")
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
    branch_ids = snapshot_data['branch_ids']

    print(f"  Snapshots:  {velocity_snapshots.shape[0]}")
    print(f"  Parameters: {len(parameters)} entries")







    # --- Energy-norm transform ---
    print("\n--- Energy-norm transform ---")

    S_homo, S_tilde, M_u, M_p, L, P = build_energy_snapshots(
        velocity_snapshots, parameters, lifting, problem, cfg.inner_product_type,
        homogenize = True
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
    unique_ids = np.unique(branch_ids)
    count = np.zeros((clustering.n_clusters, len(unique_ids)), dtype=int)

    # for i in range(clustering.n_clusters):
    #     mask = clustering.labels == i

    #     for col, k in enumerate(unique_ids):
    #         count[i, col] = np.sum(branch_ids[mask] == k)
    #         print(f"Cluster {i} has {count[i, col]} of branch {k}")

    # for i in clustering.cluster_indices[1]:
    #     Re_i, amp_i = parameters[i][0], parameters[i][1]
    #     if Re_i >= 95 and amp_i <= -0.3:
    #         print(f"Re={Re_i:.1f} amp={amp_i:+.2f} C_L={lift_coefficients[i]:+.4f}")
    # for i,label in enumerate(clustering.labels):
    #     branch= branch_ids[i]
    #     count[label,branch] += 1
    # print(count)




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
        # for k in (1, 3):
        #     idx = deim_ops_dict[k].indices_J
        #     ndup = len(idx) - len(np.unique(idx))
        #     # condition of the sampled-mode matrix is the other tell:
        #     print(f"cluster {k}: m_J={len(idx)}, duplicate_indices={ndup}, "
        #         f"U_J_P_inv finite={np.isfinite(deim_ops_dict[k].U_J_P_inv).all()}")
        # for k in (1, 3):
        #     print(f"cluster {k}: cond(U_J_P) = {np.linalg.cond(deim_ops_dict[k].U_J_P_inv):.2e}")

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
        M_u = M_u,
    )
    #--------------------------------------------------------------------------------------------------------
    from scripts.run_hyperreduction_study import report_deim_ceiling, run_study

    # STEP 0 -- run this alone first, read the output, then continue
    report_deim_ceiling(operators, layout, cfg)


    # --- Build exact convection tensors (once) ---
    from nsrom.tensor_convection import (build_convection_tensor, save_convection_tensor,
                                TensorConvection, verify_against_exact)
    import os
    if cfg.use_tensor:

        tensor_conv_dict = {}
        for k in range(clustering.n_clusters):
            Z_u_k = operators[k].Z_u                       # (n_dofs, N_u), includes supremizers
            tpath = os.path.join(layout.cluster_dir(k), "conv_tensor.npz")
            if cfg.recompute_deim or not os.path.exists(tpath):
                N = build_convection_tensor(Z_u_k, problem.velocity_space)
                save_convection_tensor(N, tpath)
            else:
                from nsrom.tensor_convection import load_convection_tensor
                N = load_convection_tensor(tpath)
                if N.shape[0] != Z_u_k.shape[1]:      # basis changed under a stale tensor
                    N = build_convection_tensor(Z_u_k, problem.velocity_space)
                    save_convection_tensor(N, tpath)
            tc = TensorConvection(N)
            verify_against_exact(tc, Z_u_k, problem.velocity_space,
                                np.random.randn(Z_u_k.shape[1]))   # must print ~1e-12
            tensor_conv_dict[k] = tc

    #--------------------------------------------------------------------------------------------------------

    #--------------------------------------------------------------------------------------------------------
    # for k in range(clustering.n_clusters):
    #     if deim_ops_dict.get(k) is not None:
    #         deim_ops_dict[k].tensor_convection = tensor_conv_dict[k]
    for k in range(clustering.n_clusters):
        operators[k].tensor_convection = tensor_conv_dict[k]
    #--------------------------------------------------------------------------------------------------------
    # --- Get Reduced Mass Matrix ---
    M_uu_red = build_reduced_velocity_l2_mass_matrices(
        problem = problem,
        clustering=clustering,
        operators=operators,
    )

    #####################################################
    C      = clustering.centroids_original   # (n_dofs, K)
    labels = clustering.labels               # whitened k-means assignment
    K, n   = clustering.n_clusters, S_homo.shape[1]

    energy_labels = np.empty(n, dtype=int)
    for s in range(n):
        Diff = C - S_homo[:, [s]]                 # (n_dofs, K)
        d2   = np.einsum('ij,ij->j', Diff, M_u @ Diff)   # energy-norm dist^2 to each centroid
        energy_labels[s] = int(np.argmin(d2))

    mism = int(np.sum(energy_labels != labels))
    print(f"energy-norm vs whitened k-means: {mism}/{n} snapshots disagree ({100*mism/n:.1f}%)")
    conf = np.array([[np.sum((labels==a) & (energy_labels==b)) for b in range(K)] for a in range(K)])
    print(conf)
    print((conf / conf.sum(axis=1, keepdims=True)).round(2))   # row-normalized
    #####################################################

    # rom_result = solve_at_parameter(
    #     Re=91,
    #     amp=0.0,
    #     clustering=clustering,
    #     operators=operators,
    #     deim_ops_dict=deim_ops_dict,
    #     problem=problem,
    #     initial_solution=initial_solution,
    #     M_u=M_u,
    #     M_p=M_p,
    #     use_deim=cfg.use_deim,
    #     cluster_forced=2,
    #     verbose = True,
    # )
    # --- Online: single point or sweep ---
    if RUN_SWEEP:
        print("\n--- Precompute fast cluster selection ---")
        precompute_fast_selection(clustering, operators, M_u)

        print("\n--- Precompute change-of-basis ---")
        precompute_change_of_basis(clustering, operators, M_u, M_p)

        print("\n--- Parameter sweep ---")
        path = build_fixed_amp_path(RE_VALUES,AMP_FIXED)
        # path = build_snake_path(RE_VALUES, AMP_VALUES)
        # path = build_hubspoke_path(RE_VALUES, AMP_VALUES)

        print(f"  Grid: {len(RE_VALUES)} Re × {len(AMP_VALUES)} amp = {len(path)} points")
        print(f"  FOM comparison: every {FOM_EVERY} points" if FOM_EVERY > 0 else "  FOM comparison: off")
        # results = run_sweep_simple(
        #     path, clustering, operators, deim_ops_dict,
        #     problem, initial_solution, M_u, M_p, M_uu_red,
        #     use_deim=cfg.use_deim,
        # )
        sol_asym_plus = load_solution('solution_rom_Re99_amp0_asym+1', problem)
        sol_asym_minus = load_solution('solution_rom_Re99_amp0_asym-1', problem)
        from conv_test import continue_amp_bare
        for k in range(clustering.n_clusters):
            print(f"cluster {k}: tensor attached = "
                f"{getattr(operators[k], 'tensor_convection', None) is not None}")
        # records = continue_amp_bare(
        #     Re=99.0,
        #     amp_values=np.linspace(0.0, -1.0, 11),   # 0, -0.1, ..., -1.0
        #     clustering=clustering,
        #     operators=operators,
        #     deim_ops_dict=deim_ops_dict,
        #     problem=problem,
        #     initial_solution=high_re,       # full solution at (Re=99, amp=0)
        #     M_u=M_u, M_p=M_p,
        #     start_cluster=1,                          # set to the branch's cluster
        # )
        # run_study(clustering, operators, problem, initial_solution,
        #         M_u, M_p, M_uu_red, layout, cfg,
        #         Re_list=RE_VALUES.tolist(), amp_values=AMP_VALUES.tolist(),
        #         saved_asym={'asym+': sol_asym_plus, 'asym-': sol_asym_minus},
        #         Re_saved_asym=99.0,
        #         diagram_kwargs=dict(max_iter=40),   # only if the builder takes it
        #         run_diagram=True)

        # asd
        from build_diagram_bare_with_sym import build_full_diagram_bare
        results = build_full_diagram_bare(
            Re_range=np.arange(20.0, 110.0, 1.0),
            amp_values=[-1,-0.9,-0.8,-0.7,-0.6,-0.5, -0.4, -0.3,-0.25, -0.2, -0.1,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],  # 0 handled by phase 1; both signs
            clustering=clustering, operators=operators, deim_ops_dict=deim_ops_dict,
            problem=problem, initial_solution=initial_solution,
            M_u=M_u, M_p=M_p, M_uu_red=M_uu_red,
            use_deim=cfg.use_deim, fom_compute = True ,
        )
        from speedup_summary import speedup_summary
        stats = speedup_summary(results)      # prints the table, returns the dict
        # save the sweep results to CSV
        save_sweep_results(results, layout.sweep_csv())




        # ===================================
        #this is the one
        # results = build_full_diagram(
        #     Re_range=np.arange(50.0, 100.0, 1.0),       # ascending; [0]=Re_min, [-1]=Re_max
        #     # amp_values=[-0.5, -0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4, 0.5],  # 0 handled by phase 1; both signs
        #     amp_values=[ -0.3, -0.2, -0.1, 0.1, 0.2, 0.3],  # 0 handled by phase 1; both signs
        #     clustering=clustering, operators=operators, deim_ops_dict=deim_ops_dict,
        #     problem=problem, initial_solution=initial_solution,
        #     M_u=M_u, M_p=M_p, M_uu_red=M_uu_red,
        #     use_deim=cfg.use_deim,
        #     tau=0.5, proj_tol=0.01,
        # )
        ################################### this ^^^^^
        # results = build_bifurcation_diagram(
        #     path, clustering, operators, deim_ops_dict,
        #     problem, initial_solution, M_u, M_p, M_uu_red,
        #     use_deim=cfg.use_deim,
        #     tau=0.8, proj_tol=0.01,
        #     offaxis_amps=(-0.3,),   # walk each asym branch to amp=-0.3
        #     re_min=50.0,            # then sweep Re 99 -> 50 at that amp
        #     n_amp_steps=10,         # 10 amp increments in leg 1
        #     re_step=1.0,
        # )
        # results = build_bifurcation_diagram(
        #     path, clustering, operators, deim_ops_dict,
        #     problem, initial_solution, M_u, M_p, M_uu_red,
        #     use_deim=cfg.use_deim, eps=0.8, tau=1.2,
        # )

        # results, critical = run_bifurcation_sweep(
        #     path, clustering, operators, deim_ops_dict,
        #     problem, initial_solution, M_u, M_p, M_uu_red,
        #     use_deim=cfg.use_deim, fom_every= FOM_EVERY,
        # )
        # if critical:
        #     print(f"Re_c at amp=0: {critical[0]['Re_c_linear']:.6f}")
        # results = run_sweep(
        #     path, clustering, operators, deim_ops_dict,
        #     problem, initial_solution, M_u, M_p,
        #     use_deim=cfg.use_deim,
        #     fom_every=FOM_EVERY,
        # )

        # n_conv = sum(1 for r in results if r['converged'])
        # print(f"\n  Converged:      {n_conv}/{len(results)}")
        # print(f"  Total ROM time: {sum(r['t_rom'] for r in results):.1f}s")

        # if FOM_EVERY > 0:
        #     fom_pts = [r for r in results if 'u_err' in r]
        #     if fom_pts:
        #         u_errs = [r['u_err'] for r in fom_pts]
        #         print(f"  Velocity error: mean={np.mean(u_errs)*100:.3f}%, max={np.max(u_errs)*100:.3f}%")


        ##############FOR CRITICAL CURVE CONTINUATION######################
        # curve = find_critical_curve_continuation(
        #     amp_values=[-1,-0.8,-0.6,-0.4,-0.2, 0.0, 0.2, 0.4,0.6,0.8,1.0],
        #     # amp_values=[0.0, 0.1, 0.2, 0.4,0.6,0.8,1.0],
        #     Re_scan=np.arange(65.0, 150.0, 1.0),
        #     clustering=clustering, operators=operators, deim_ops_dict=deim_ops_dict,
        #     problem=problem, initial_solution=initial_solution,
        #     M_u=M_u, M_p=M_p, M_uu_red=M_uu_red, use_deim=False,
        # )
        # save_critical_curve(curve, "critical_curve.csv", sweep_path="critical_sweep.csv")



    else:
        print('\n')
        print('#'*80)
        print('#'*80)
        print('#'*80)
        print(f"--- Online solve (Re={TEST_RE}, amp={TEST_AMP}) ---")
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

        print('#'*80)
        print('#'*80)
        print('#'*80)
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