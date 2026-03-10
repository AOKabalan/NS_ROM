"""
Local ROM pipeline with optional DEIM support.

Configuration flags:
    USE_DEIM:    True → DEIM/MDEIM for convection,  False → exact projection
    COMPUTE_AFFINE_CONVECTION:  True → precompute affine terms

File structure:
    local_rom/K{N_CLUSTERS}_{INNER_PRODUCT_TYPE}/
        clustering/
        cluster_0/
            pod_basis/
            operators/
            deim_basis.npz      (if USE_DEIM)
            deim_ops.npz        (if USE_DEIM)
        cluster_1/
            ...
"""

import numpy as np
import os
import time

from firedrake import norm, Function, TestFunction, TrialFunction, inner, grad, dx, assemble
import scipy.sparse as sp

from Navier_Stokes import (
    NavierStokesSolution,
    setup_navier_stokes_problem,
    solve_steady_navier_stokes,
    load_solution,
    compute_forces,
)
from snapshot_collection import load_snapshot_dofs
from lifting_functions import load_lifting_functions
from helper_functions import dofs_to_functions

from rom import (
    compute_pod_basis,
    save_pod_basis,
    load_pod_basis,
    assemble_inner_product_matrix,
    build_reduced_operators,
    save_reduced_operators,
    load_reduced_operators,
    solve_rom,
    reconstruct_solution,
    project_to_rom_coefficients,
    homogenize_snapshots,
    SubMeshDEIM,
)

from cluster_building import (
    apply_energy_transform,
    cluster_kmeans,
    build_clustering_result,
    save_clustering_result,
    load_clustering_result,
    clustering_is_valid,
    select_cluster,
)

from deim import compute_and_save_basis, load_basis_metadata
from online_operators import (
    DEIMOnlineOperators,
    build_deim_online_operators,
    load_ops_metadata,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

# --- Mode flags ---
USE_DEIM                  = True
COMPUTE_AFFINE_CONVECTION = True

# --- Paths ---
SNAPSHOT_DIR   = "two_param_snapshots_pod_dofs"
LIFTING_DIR    = "lifting"
MESH_FILE      = "mesh/mid_pinball.msh"
FOM_CHECKPOINT = "initial_states/velocity_checkpoint_down"

REYNOLDS_INIT  = 100.0
AMPLITUDE_INIT = 0.0

INNER_PRODUCT_TYPE = "H1"

# --- Clustering ---
N_CLUSTERS           = 3
RECOMPUTE_CLUSTERING = False

# --- ROM truncation ---
N_VELOCITY_MODES   = 20
N_PRESSURE_MODES   = 20
N_SUPREMIZER_MODES = 20

BOUNDARY_MARKERS = [1, 3, 4, 5, 6]

# --- DEIM config (only used if USE_DEIM=True) ---
DEIM_SNAPSHOT_DIR = "two_param_snapshots_pod_dofs"
N_MODES_F         = None   # None = full
N_MODES_J         = None
M_F               = 30    # online truncation
M_J               = 30

# --- Force recomputation ---
RECOMPUTE_POD  = False
RECOMPUTE_DEIM = False

# --- Test parameters ---
TEST_RE  = 92.0
TEST_AMP = -0.15

# =============================================================================
# FILE STRUCTURE
# =============================================================================

_CFG          = f"K{N_CLUSTERS}_{INNER_PRODUCT_TYPE}"
LOCAL_ROM_DIR = os.path.join("local_rom", _CFG)
CLUSTER_DIR   = os.path.join(LOCAL_ROM_DIR, "clustering")


def pod_dir(k):
    return os.path.join(LOCAL_ROM_DIR, f"cluster_{k}", "pod_basis")

def ops_dir(k):
    return os.path.join(LOCAL_ROM_DIR, f"cluster_{k}", "operators")

def deim_basis_path(k):
    return os.path.join(LOCAL_ROM_DIR, f"cluster_{k}", "deim_basis.npz")

def deim_ops_path(k):
    return os.path.join(LOCAL_ROM_DIR, f"cluster_{k}", "deim_ops.npz")

def pod_exists(k):
    return os.path.isdir(pod_dir(k)) and os.path.isdir(ops_dir(k))


# =============================================================================
# DEIM Cache Helpers
# =============================================================================

def needs_basis_recompute(basis_path, snapshot_dir, n_modes_F, n_modes_J):
    """Check if DEIM basis needs recomputation."""
    if not os.path.exists(basis_path):
        return True, "Basis file not found"
    try:
        meta = load_basis_metadata(basis_path)
    except Exception as e:
        return True, f"Failed to load metadata: {e}"

    if meta['snapshot_dir'] != snapshot_dir:
        return True, f"Snapshot dir changed: '{meta['snapshot_dir']}' → '{snapshot_dir}'"

    stored_F = meta['stored_n_modes_F']
    stored_J = meta['stored_n_modes_J']

    if n_modes_F is not None and stored_F != -1 and n_modes_F > stored_F:
        return True, f"Need more F modes: stored={stored_F}, need={n_modes_F}"
    if n_modes_J is not None and stored_J != -1 and n_modes_J > stored_J:
        return True, f"Need more J modes: stored={stored_J}, need={n_modes_J}"

    return False, "Up-to-date"


def needs_ops_rebuild(ops_path, basis_path, m_F, m_J, n_rom_modes):
    """Check if DEIM online operators need rebuild."""
    if not os.path.exists(ops_path):
        return True, "Operators file not found"
    try:
        meta = load_ops_metadata(ops_path)
    except Exception as e:
        return True, f"Failed to load metadata: {e}"

    if meta['basis_path'] != basis_path:
        return True, "Basis path changed"
    if meta['m_F'] != m_F:
        return True, f"m_F changed: {meta['m_F']} → {m_F}"
    if meta['m_J'] != m_J:
        return True, f"m_J changed: {meta['m_J']} → {m_J}"
    if meta['n_rom_modes'] != n_rom_modes:
        return True, f"ROM modes changed: {meta['n_rom_modes']} → {n_rom_modes}"

    return False, "Up-to-date"


# =============================================================================
# DEIM Debug
# =============================================================================

def deim_debug(operators, deim_ops, problem, u_test):
    """Quick sanity check: compare exact vs DEIM projection."""
    Z_u = operators.Z_u
    V_full = problem.velocity_space
    v_test = TestFunction(V_full)

    # Residual check
    F_form = inner(grad(u_test) * u_test, v_test) * dx
    F_vec = assemble(F_form).dat.data_ro.flatten()
    F_exact = Z_u.T @ F_vec
    F_deim = deim_ops.reduced_convective(F_vec)
    err_F = np.linalg.norm(F_exact - F_deim) / np.linalg.norm(F_exact)

    # Jacobian check
    u_trial = TrialFunction(V_full)
    J_form = (
        inner(grad(u_trial) * u_test, v_test) * dx
        + inner(grad(u_test) * u_trial, v_test) * dx
    )
    J_petsc = assemble(J_form, mat_type='aij').M.handle
    indptr, indices, J_data = J_petsc.getValuesCSR()
    J_scipy = sp.csr_matrix((J_data, indices, indptr), shape=J_petsc.getSize())
    J_exact = Z_u.T @ (J_scipy @ Z_u)
    J_mdeim = deim_ops.reduced_jacobian(J_data)
    err_J = np.linalg.norm(J_exact - J_mdeim) / np.linalg.norm(J_exact)

    print(f"    DEIM sanity check:")
    print(f"      Residual error:  {err_F:.6e}")
    print(f"      Jacobian error:  {err_J:.6e}")

    return err_F, err_J


# =============================================================================
# MAIN
# =============================================================================

def main():

    mode_str = "DEIM" if USE_DEIM else "Exact"
    affine_str = "Affine" if COMPUTE_AFFINE_CONVECTION else "Full"

    print("=" * 80)
    print(f"LOCAL ROM — {mode_str} solver, {affine_str} convection, K={N_CLUSTERS}")
    print("=" * 80)
    print(f"  USE_DEIM:     {USE_DEIM}")
    print(f"  AFFINE:       {COMPUTE_AFFINE_CONVECTION}")
    print(f"  ROM modes:    vel={N_VELOCITY_MODES}, pres={N_PRESSURE_MODES}, sup={N_SUPREMIZER_MODES}")
    print(f"  Inner prod:   {INNER_PRODUCT_TYPE}")
    print(f"  Test params:  Re={TEST_RE}, amp={TEST_AMP}")
    if USE_DEIM:
        print(f"  DEIM modes:   m_F={M_F}, m_J={M_J}")
    print()

    # -------------------------------------------------------------------------
    print("--- Step 1: Setup problem ---")
    problem = setup_navier_stokes_problem(
        mesh_file=MESH_FILE,
        reynolds_init=REYNOLDS_INIT,
        amplitude_init=AMPLITUDE_INIT,
    )

    initial_solution = load_solution(FOM_CHECKPOINT, problem)

    # -------------------------------------------------------------------------
    print("\n--- Step 2: Load snapshots and lifting ---")
    snapshot_data = load_snapshot_dofs(SNAPSHOT_DIR)
    lifting       = load_lifting_functions(LIFTING_DIR, problem)

    velocity_snapshots = snapshot_data['velocity']
    pressure_snapshots = snapshot_data['pressure']
    parameters         = snapshot_data['parameters']

    u_base    = lifting.u_base.dat.data_ro.flatten()
    u_control = lifting.u_control.dat.data_ro.flatten()

    print(f"  Snapshots: {velocity_snapshots.shape[0]}")
    print(f"  Parameters: {len(parameters)} entries")

    # -------------------------------------------------------------------------
    print("\n--- Step 3: Energy-norm transform ---")
    M_u = assemble_inner_product_matrix(problem.velocity_space, INNER_PRODUCT_TYPE)
    M_p = assemble_inner_product_matrix(problem.pressure_space, 'L2')
    L, P, S_tilde = apply_energy_transform(velocity_snapshots, M_u)

    print(f"  S_tilde: {S_tilde.shape}")

    # -------------------------------------------------------------------------
    print("\n--- Step 4: Clustering ---")
    valid, reason = clustering_is_valid(CLUSTER_DIR, S_tilde, N_CLUSTERS)

    if RECOMPUTE_CLUSTERING or not valid:
        print(f"  {reason} — recomputing ...")
        labels, centers, inertia = cluster_kmeans(S_tilde, N_CLUSTERS)
        clustering = build_clustering_result(S_tilde, labels, centers, L, P, parameters)
        save_clustering_result(clustering, CLUSTER_DIR, S_tilde, N_CLUSTERS)
    else:
        print(f"  {reason} — loading from {CLUSTER_DIR}/")
        clustering = load_clustering_result(CLUSTER_DIR)

    clustering.summary()

    # -------------------------------------------------------------------------
    # Load DEIM snapshots once (shared across clusters)
    # -------------------------------------------------------------------------
    deim_snapshot_data = None
    if USE_DEIM:
        print("\n--- Loading DEIM snapshots ---")
        deim_snapshot_data = load_snapshot_dofs(DEIM_SNAPSHOT_DIR)
        print(f"  DEIM snapshots: {deim_snapshot_data['velocity'].shape[0]}")

    # -------------------------------------------------------------------------
    print("\n--- Step 5: Local ROMs ---")
    bases         = {}
    operators     = {}
    deim_ops_dict = {}

    for k in range(clustering.n_clusters):
        print(f"\n{'─'*60}")
        print(f"  Cluster {k}")
        print(f"{'─'*60}")

        idx      = clustering.cluster_indices[k]
        params_k = clustering.cluster_parameters(k)
        n_snap_k = len(idx)

        # =====================================================================
        # 5a: POD basis + reduced operators
        # =====================================================================
        if RECOMPUTE_POD or not pod_exists(k):
            print(f"  Building POD + operators ...")
            S_u_k = velocity_snapshots[idx, :]
            S_p_k = pressure_snapshots[idx, :]

            basis_k = compute_pod_basis(
                velocity_snapshots=S_u_k,
                pressure_snapshots=S_p_k,
                problem=problem,
                inner_product_type=INNER_PRODUCT_TYPE,
                compute_supremizer=True,
                parameters=params_k,
                u_base=u_base,
                u_control=u_control,
                amplitude_index=1,
                boundary_markers=BOUNDARY_MARKERS,
            ).truncate(
                n_velocity=min(N_VELOCITY_MODES, n_snap_k - 1),
                n_pressure=min(N_PRESSURE_MODES, n_snap_k - 1),
                n_supremizer=min(N_SUPREMIZER_MODES, n_snap_k - 1),
            )

            operators_k = build_reduced_operators(
                problem,
                basis_k,
                lifting,
                inner_product_type=INNER_PRODUCT_TYPE,
                compute_affine_convection=COMPUTE_AFFINE_CONVECTION,
            )

            save_pod_basis(basis_k, pod_dir(k))
            save_reduced_operators(operators_k, ops_dir(k))
            print(f"  Saved to {LOCAL_ROM_DIR}/cluster_{k}/")
        else:
            print(f"  Loading POD + operators ...")
            basis_k = load_pod_basis(pod_dir(k)).truncate(
                n_velocity=N_VELOCITY_MODES,
                n_pressure=N_PRESSURE_MODES,
                n_supremizer=N_SUPREMIZER_MODES,
            )
            operators_k = load_reduced_operators(ops_dir(k))

        print(f"    {basis_k.n_velocity_modes} vel + "
              f"{basis_k.n_supremizer_modes} sup + "
              f"{basis_k.n_pressure_modes} pres")

        bases[k]     = basis_k
        operators[k] = operators_k

        # =====================================================================
        # 5b: DEIM basis + online operators (per cluster)
        # =====================================================================
        if USE_DEIM:
            print(f"  DEIM for cluster {k} ...")
            Z_u_k = operators_k.Z_u
            n_rom_modes_k = Z_u_k.shape[1]

            bp = deim_basis_path(k)
            op = deim_ops_path(k)

            # --- DEIM basis ---
            if RECOMPUTE_DEIM or RECOMPUTE_POD:
                recompute_basis = True
                reason = "forced recompute"
            else:
                recompute_basis, reason = needs_basis_recompute(
                    bp, DEIM_SNAPSHOT_DIR, N_MODES_F, N_MODES_J
                )

            if recompute_basis:
                print(f"    Computing DEIM basis: {reason}")

                # Use cluster snapshots for DEIM basis
                deim_vel_k = deim_snapshot_data['velocity'][idx, :]
                deim_params_k = np.array([deim_snapshot_data['parameters'][i] for i in idx])
                if COMPUTE_AFFINE_CONVECTION:
                    deim_vel_k = homogenize_snapshots(
                        deim_vel_k, deim_params_k, u_base, u_control,
                    )

                deim_vel_functions_k = dofs_to_functions(
                    deim_vel_k, problem.velocity_space
                )
                os.makedirs(os.path.dirname(bp), exist_ok=True)
                # Number of DEIM snapshots in this cluster
                n_deim_k = deim_vel_k.shape[0]

                # Cap modes to available snapshots
                # n_modes_F_k = min(N_MODES_F, n_deim_k) if N_MODES_F is not None else None
                # n_modes_J_k = min(N_MODES_J, n_deim_k) if N_MODES_J is not None else None
                # m_F_k = min(M_F, n_modes_F_k or n_deim_k)
                # m_J_k = min(M_J, n_modes_J_k or n_deim_k)
                compute_and_save_basis(
                    deim_vel_functions_k,
                    problem.velocity_space,
                    output_path=bp,
                    n_modes_F=N_MODES_F,
                    n_modes_J=N_MODES_J,
                    snapshot_dir=DEIM_SNAPSHOT_DIR,
                )
                del deim_vel_functions_k
            else:
                print(f"    Cached DEIM basis: {reason}")

            # --- DEIM online operators ---
            if RECOMPUTE_DEIM or RECOMPUTE_POD:
                rebuild_ops = True
                reason = "forced recompute"
            else:
                rebuild_ops, reason = needs_ops_rebuild(
                    op, bp, M_F, M_J, n_rom_modes_k
                )

            if rebuild_ops:
                print(f"    Building DEIM operators: {reason}")
                deim_ops_k = build_deim_online_operators(bp, Z_u_k, M_F, M_J)
                deim_ops_k.save(op, basis_path=bp)
            else:
                print(f"    Cached DEIM operators: {reason}")
                deim_ops_k = DEIMOnlineOperators.load(op)
                print(f"      m_F={deim_ops_k.m_F}, m_J={deim_ops_k.m_J}")

            deim_ops_dict[k] = deim_ops_k
        else:
            deim_ops_dict[k] = None
    # -------------------------------------------------------------------------
    # ADD THIS HERE BECASUE RECONSTRUCT USES IT
    problem.reynolds.assign(TEST_RE)
    problem.amplitude.assign(TEST_AMP)



    # -------------------------------------------------------------------------
    print(f"\n--- Step 6: Online solve (Re={TEST_RE}, amp={TEST_AMP}) ---")
    nu = 1.0 / TEST_RE

    k = select_cluster(clustering, TEST_RE, TEST_AMP)
    print(f"  Selected cluster: {k}")

    # Initial guess via projection
    velocity_dofs = initial_solution.velocity.dat.data_ro.flatten()
    pressure_dofs = initial_solution.pressure.dat.data_ro.flatten()
    u_coeffs, p_coeffs = project_to_rom_coefficients(
        velocity_dofs, pressure_dofs,
        operators[k],
        amp=TEST_AMP,
        M_u=M_u,
        M_p=M_p,
    )

    # Build SubMeshDEIM if needed
    deim_ops_k = deim_ops_dict[k]
    submesh_deim = None
    if USE_DEIM and deim_ops_k is not None:
        submesh_deim = SubMeshDEIM(problem.velocity_space, deim_ops_k)

        # Optional sanity check
        u_test = Function(problem.velocity_space)
        u_test.dat.data[:] = velocity_dofs.reshape(u_test.dat.data.shape)
        deim_debug(operators[k], deim_ops_k, problem, u_test)

    t0 = time.perf_counter()
    solution = solve_rom(
        operators=operators[k],
        nu=nu,
        amp=TEST_AMP,
        problem=problem,
        deim_ops=deim_ops_k,
        u_initial_guess=u_coeffs,
        p_initial_guess =p_coeffs,
        submesh_deim=submesh_deim,
    )
    t_rom = time.perf_counter() - t0

    solution_rom = reconstruct_solution(solution, operators[k], problem, amp = TEST_AMP)


    print(f"  Amplitude:  {solution.amplitude}")
    print(f"  Converged:  {solution.converged}")
    print(f"  Iterations: {solution.iterations}")
    print(f"  ROM time:   {t_rom:.3f}s")

    # -------------------------------------------------------------------------
    print("\n--- Step 7: FOM comparison ---")
    t0 = time.perf_counter()
    hf_solution = solve_steady_navier_stokes(problem, initial_solution.solution)
    t_fom = time.perf_counter() - t0

    def norm_M(v, M):
        return np.sqrt(float(v @ (M @ v)))

    u_fom      = hf_solution.velocity.dat.data_ro.flatten()
    u_rom_dofs = solution_rom.velocity.dat.data_ro.flatten()
    p_fom      = hf_solution.pressure.dat.data_ro.flatten()
    p_rom_dofs = solution_rom.pressure.dat.data_ro.flatten()

    u_err = norm_M(u_fom - u_rom_dofs, M_u) / norm_M(u_fom, M_u)
    p_err = norm_M(p_fom - p_rom_dofs, M_p) / norm_M(p_fom, M_p)

    print(f"\n{'='*80}")
    print(f"LOCAL ROM — {mode_str} solver, {affine_str} convection, K={N_CLUSTERS}")
    print(f"{'='*80}")
    print(f"  Re={TEST_RE}, amp={TEST_AMP}")
    print(f"  Cluster selected: {k}")
    print(f"  ROM modes:  {operators[k].n_velocity_modes} vel × {operators[k].n_pressure_modes} pres")
    print(f"  Affine convection: {operators[k].has_affine_convection}")
    if USE_DEIM:
        print(f"  DEIM: m_F={deim_ops_k.m_F}, m_J={deim_ops_k.m_J}")
    print(f"  Converged: {solution.converged} ({solution.iterations} iterations)")
    print(f"  Velocity error ({INNER_PRODUCT_TYPE}): {u_err*100:.4f}%")
    print(f"  Pressure error (L2):                   {p_err*100:.4f}%")
    print(f"  ROM time:  {t_rom:.3f}s")
    print(f"  FOM time:  {t_fom:.3f}s")
    print(f"  Speedup:   {t_fom/t_rom:.1f}x")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
