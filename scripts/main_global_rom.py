"""
Unified ROM test script.

Tests: POD → Operators → (optional DEIM) → Solve → Reconstruct → Compare with FOM

Configuration flags:
    USE_DEIM:    True → DEIM/MDEIM for convection,  False → exact projection
    USE_AFFINE:  True → precompute affine terms (quadratic-only DEIM),  False → full assembly

Combinations:
    DEIM=F, AFFINE=F  →  Exact solver, full convection in callbacks
    DEIM=F, AFFINE=T  →  Exact solver, quadratic-only in callbacks (affine constant+linear precomputed)
    DEIM=T, AFFINE=F  →  DEIM on full convection
    DEIM=T, AFFINE=T  →  DEIM on quadratic part only (best accuracy for same DEIM modes)
"""

import numpy as np
import os
from firedrake import norm, Function, TestFunction, TrialFunction, inner, grad, dx, assemble
import scipy.sparse as sp
import time

from nsrom.navier_stokes import (
    setup_navier_stokes_problem,
    solve_steady_navier_stokes,
    load_solution,
)
from nsrom.snapshot_collection import load_snapshot_dofs
from nsrom.lifting_functions import load_lifting_functions
from nsrom.helper_functions import dofs_to_functions

from rom import (
    # POD
    compute_pod_basis,
    save_pod_basis,
    load_pod_basis,
    assemble_inner_product_matrix,
    # Operators
    build_reduced_operators,
    save_reduced_operators,
    # Solver (unified)
    solve_rom,
    reconstruct_solution,
    project_to_rom_coefficients,
    homogenize_snapshots,
    SubMeshDEIM,
)

from nsrom.deim import compute_and_save_basis, load_basis_metadata
from nsrom.online_operators import (
    DEIMOnlineOperators,
    build_deim_online_operators,
    load_ops_metadata,
)
# =============================================================================
# CONFIGURATION
# =============================================================================

# --- Mode flags ---
USE_DEIM = False          # True → DEIM, False → exact projection
USE_AFFINE = True          # True → precompute affine convection terms

# --- Paths ---
SNAPSHOT_DIR = "two_param_snapshots_pod_dofs"
POD_DIR = "pod_basis_two_param"
OPS_DIR = "operators_two_param"
LIFTING_DIR = "lifting"
MESH_FILE = "mesh/mid_pinball.msh"
FOM_CHECKPOINT = "newlygenerated_high"

# --- DEIM paths (only used if USE_DEIM=True) ---
DEIM_SNAPSHOT_DIR = "two_param_snapshots_rom_deim_dofs"
DEIM_BASIS_PATH = "deim_basis.npz"
DEIM_OPS_PATH = "deim_ops.npz"

# --- DEIM basis computation (max modes to store) ---
N_MODES_F = None              # None = full
N_MODES_J = 200

# --- DEIM online truncation (modes to use, must be ≤ stored) ---
M_F = 100
M_J = 100

# --- ROM truncation ---
N_VELOCITY_MODES = 35
N_PRESSURE_MODES = 35
N_SUPREMIZER_MODES = 35
INNER_PRODUCT_TYPE = 'H1'

# --- Test parameters ---
TEST_RE = 90.0
TEST_AMP = -0.1

# --- Force recomputation ---
RECOMPUTE_POD = False
RECOMPUTE_DEIM = False

# =============================================================================


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
        return True, f"Basis path changed"
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

    print(f"\n  DEIM sanity check:")
    print(f"    Residual error:  {err_F:.6e}")
    print(f"    Jacobian error:  {err_J:.6e}")

    return err_F, err_J


# =============================================================================
# Main
# =============================================================================

def main():
    # Print configuration
    mode_str = "DEIM" if USE_DEIM else "Exact"
    affine_str = "Affine" if USE_AFFINE else "Full"

    print("=" * 80)
    print(f"ROM TEST — {mode_str} solver, {affine_str} convection")
    print("=" * 80)
    print(f"  USE_DEIM:     {USE_DEIM}")
    print(f"  USE_AFFINE:   {USE_AFFINE}")
    print(f"  ROM modes:    vel={N_VELOCITY_MODES}, pres={N_PRESSURE_MODES}, sup={N_SUPREMIZER_MODES}")
    print(f"  Inner prod:   {INNER_PRODUCT_TYPE}")
    print(f"  Test params:  Re={TEST_RE}, amp={TEST_AMP}")
    if USE_DEIM:
        print(f"  DEIM modes:   m_F={M_F}, m_J={M_J}")
    print()

    # =========================================================================
    # STEP 1: Setup problem
    # =========================================================================
    print("--- Step 1: Setup problem ---")
    problem = setup_navier_stokes_problem(
        mesh_file=MESH_FILE,
        reynolds_init=100.0,
        amplitude_init=0.0,
    )

    # =========================================================================
    # STEP 2: Load data
    # =========================================================================
    print("\n--- Step 2: Load snapshots and lifting functions ---")
    snapshot_data = load_snapshot_dofs(SNAPSHOT_DIR)
    lifting = load_lifting_functions(LIFTING_DIR, problem)

    velocity_snapshots = snapshot_data['velocity']
    pressure_snapshots = snapshot_data['pressure']
    parameters = snapshot_data['parameters']

    u_base = lifting.u_base.dat.data_ro.flatten()
    u_control = lifting.u_control.dat.data_ro.flatten()

    print(f"  Snapshots: {velocity_snapshots.shape[0]}")

    # =========================================================================
    # STEP 3: POD basis
    # =========================================================================
    print("\n--- Step 3: POD basis ---")

    pod_exists = os.path.exists(os.path.join(POD_DIR, "pod_basis.npz"))

    if RECOMPUTE_POD or not pod_exists:
        reason = "RECOMPUTE_POD=True" if RECOMPUTE_POD else "not found"
        print(f"  Computing POD basis ({reason})...")
        basis_full = compute_pod_basis(
            velocity_snapshots=velocity_snapshots,
            pressure_snapshots=pressure_snapshots,
            problem=problem,
            inner_product_type=INNER_PRODUCT_TYPE,
            compute_supremizer=True,
            parameters=parameters,
            u_base=u_base,
            u_control=u_control,
            amplitude_index=1,
            boundary_markers=[1, 3, 4, 5, 6],
        )
        save_pod_basis(basis_full, POD_DIR)
    else:
        print(f"  Loading existing POD basis from {POD_DIR}/")
        basis_full = load_pod_basis(POD_DIR)

    basis = basis_full.truncate(
        n_velocity=N_VELOCITY_MODES,
        n_pressure=N_PRESSURE_MODES,
        n_supremizer=N_SUPREMIZER_MODES,
    )
    print(f"  Truncated: {basis.n_velocity_modes} vel + "
          f"{basis.n_supremizer_modes} sup + {basis.n_pressure_modes} pres")

    # =========================================================================
    # STEP 4: Build reduced operators
    #
    # Supremizer enrichment happens internally.
    # Affine convection precomputed if USE_AFFINE=True.
    # =========================================================================
    print("\n--- Step 4: Build reduced operators ---")
    operators = build_reduced_operators(
        problem,
        basis,
        lifting,
        inner_product_type=INNER_PRODUCT_TYPE,
        compute_affine_convection=USE_AFFINE,
    )
    save_reduced_operators(operators, OPS_DIR)

    # =========================================================================
    # STEP 5: DEIM (only if USE_DEIM=True)
    # =========================================================================
    deim_ops = None

    if USE_DEIM:
        print("\n--- Step 5: DEIM basis ---")

        deim_snapshot_data = load_snapshot_dofs(DEIM_SNAPSHOT_DIR)
        print(f"  DEIM snapshots: {deim_snapshot_data['velocity'].shape[0]}")

        # --- 5a: DEIM basis ---
        if RECOMPUTE_DEIM:
            recompute = True
            reason = "RECOMPUTE_DEIM=True"
        else:
            recompute, reason = needs_basis_recompute(
                DEIM_BASIS_PATH, DEIM_SNAPSHOT_DIR, N_MODES_F, N_MODES_J
            )

        if recompute:
            print(f"  Computing DEIM basis: {reason}")
            if USE_AFFINE:
                print("Using homogeneous snaps")
                deim_velocity_snapshots = homogenize_snapshots(
                    deim_snapshot_data['velocity'],
                    deim_snapshot_data['parameters'],
                    u_base,
                    u_control,
                )
            else:
                deim_velocity_snapshots = deim_snapshot_data['velocity']

            deim_velocity_functions = dofs_to_functions(
                deim_velocity_snapshots, problem.velocity_space
            )
            compute_and_save_basis(
                deim_velocity_functions,
                problem.velocity_space,
                output_path=DEIM_BASIS_PATH,
                n_modes_F=N_MODES_F,
                n_modes_J=N_MODES_J,
                snapshot_dir=DEIM_SNAPSHOT_DIR,
            )
            del deim_velocity_functions
        else:
            print(f"  Using cached DEIM basis: {reason}")
            meta = load_basis_metadata(DEIM_BASIS_PATH)
            print(f"    Stored: F={meta['stored_n_modes_F']}, J={meta['stored_n_modes_J']}")

        # --- 5b: DEIM online operators ---
        print("\n--- Step 5b: DEIM online operators ---")

        Z_u = operators.Z_u
        n_rom_modes = Z_u.shape[1]

        if RECOMPUTE_DEIM or RECOMPUTE_POD:
            rebuild = True
            reason = "forced recompute"
        else:
            rebuild, reason = needs_ops_rebuild(
                DEIM_OPS_PATH, DEIM_BASIS_PATH, M_F, M_J, n_rom_modes
            )

        if rebuild:
            print(f"  Building DEIM operators: {reason}")
            deim_ops = build_deim_online_operators(
                DEIM_BASIS_PATH, Z_u, M_F, M_J
            )
            deim_ops.save(DEIM_OPS_PATH, basis_path=DEIM_BASIS_PATH)
        else:
            print(f"  Using cached DEIM operators: {reason}")
            deim_ops = DEIMOnlineOperators.load(DEIM_OPS_PATH)
            print(f"    m_F={deim_ops.m_F}, m_J={deim_ops.m_J}")

        submesh_deim = SubMeshDEIM(problem.velocity_space, deim_ops)

    else:
        print("\n--- Step 5: DEIM skipped (USE_DEIM=False) ---")
        submesh_deim = None

    # =========================================================================
    # STEP 6: Solve ROM
    # =========================================================================
    print("\n--- Step 6: Solve ROM ---")

    Re = TEST_RE
    nu = 1.0 / Re
    amp = TEST_AMP

    # Load FOM solution for initial guess
    fom_solution = load_solution(FOM_CHECKPOINT, problem)
    velocity_dofs = fom_solution.velocity.dat.data_ro.flatten()
    pressure_dofs = fom_solution.pressure.dat.data_ro.flatten()

    M_u = assemble_inner_product_matrix(problem.velocity_space, INNER_PRODUCT_TYPE)
    M_p = assemble_inner_product_matrix(problem.pressure_space, 'L2')

    u_coeffs, p_coeffs = project_to_rom_coefficients(
        velocity_dofs, pressure_dofs,
        operators,
        amp=amp,
        M_u=M_u,
        M_p=M_p,
    )

    # Optional DEIM sanity check
    if USE_DEIM:
        u_test = Function(problem.velocity_space)
        u_test.dat.data[:] = velocity_dofs.reshape(u_test.dat.data.shape)
        deim_debug(operators, deim_ops, problem, u_test)



    # Solve — deim_ops=None → exact, deim_ops=... → DEIM
    t0 = time.perf_counter()
    solution = solve_rom(
        operators=operators,
        nu=nu,
        amp=amp,
        problem=problem,
        deim_ops=deim_ops,
        u_initial_guess=u_coeffs,
        p_initial_guess=p_coeffs,
        submesh_deim = submesh_deim,
    )
    t_rom = time.perf_counter() - t0
    print(f"\n  Solution:")
    print(f"    Re = {solution.reynolds}, amp = {solution.amplitude}")
    print(f"    converged = {solution.converged}, iterations = {solution.iterations}")
    print(f"    ||alpha|| = {np.linalg.norm(solution.velocity_coeffs):.6e}")
    print(f"    ||beta||  = {np.linalg.norm(solution.pressure_coeffs):.6e}")
    print(f"    N_u = {len(solution.velocity_coeffs)}, N_p = {len(solution.pressure_coeffs)}")

    # =========================================================================
    # STEP 7: Reconstruct
    # =========================================================================
    print("\n--- Step 7: Reconstruct solution ---")
    u_rom, p_rom = reconstruct_solution(solution, operators, problem, amp= TEST_AMP)
    print(f"  ||u_rom|| = {norm(u_rom):.6e}")
    print(f"  ||p_rom|| = {norm(p_rom):.6e}")

    # =========================================================================
    # STEP 8: Compare with FOM
    # =========================================================================
    print("\n--- Step 8: Compare with FOM ---")

    problem.reynolds.assign(Re)
    problem.amplitude.assign(amp)

    print("  Solving FOM...")
    # FOM solve
    t0 = time.perf_counter()
    HF_solution = solve_steady_navier_stokes(
        problem, initial_guess=fom_solution.solution
    )
    t_fom = time.perf_counter() - t0


    u_fom = HF_solution.velocity
    p_fom = HF_solution.pressure

    # Compute errors
    def norm_M(v, M):
        return np.sqrt(v.T @ (M @ v))

    u_fom_dofs = u_fom.dat.data_ro.flatten()
    u_rom_dofs = u_rom.dat.data_ro.flatten()
    p_fom_dofs = p_fom.dat.data_ro.flatten()
    p_rom_dofs = p_rom.dat.data_ro.flatten()

    u_err = norm_M(u_fom_dofs - u_rom_dofs, M_u) / norm_M(u_fom_dofs, M_u)
    p_err = norm_M(p_fom_dofs - p_rom_dofs, M_p) / norm_M(p_fom_dofs, M_p)

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"SUMMARY — {mode_str} solver, {affine_str} convection")
    print(f"{'='*80}")
    print(f"  Re = {Re}, amp = {amp}")
    print(f"  ROM:  {operators.n_velocity_modes} velocity × {operators.n_pressure_modes} pressure")
    print(f"  Affine convection: {operators.has_affine_convection}")
    if USE_DEIM:
        print(f"  DEIM: m_F={deim_ops.m_F}, m_J={deim_ops.m_J}")
    print(f"  Converged: {solution.converged} ({solution.iterations} iterations)")
    print(f"  Velocity error ({INNER_PRODUCT_TYPE}): {u_err*100:.4f}%")
    print(f"  Pressure error (L2):                   {p_err*100:.4f}%")
    print(f"{'='*80}")
    print(f"=================================== SPEEDUP ====================================")
    print(f"  ROM solve: {t_rom:.4f} s")
    print(f"  FOM solve: {t_fom:.4f} s")
    print(f"  Speedup:   {t_fom / t_rom:.1f}x")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
