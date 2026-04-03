"""
Unified ROM Solver.

Single entry point for solving the reduced Navier-Stokes system.
Handles exact and DEIM modes transparently.

Usage:
    # Exact (no DEIM)
    solution = solve_rom(operators, nu, amp, problem)

    # DEIM
    solution = solve_rom(operators, nu, amp, problem, deim_ops=my_deim)

    # Affine convection is auto-detected from operators.has_affine_convection
"""

import numpy as np
import traceback
from firedrake import (
    Function, TestFunctions, TrialFunctions,
    split, inner, dot, dx,
    Constant, UnitIntervalMesh, VectorFunctionSpace,
    NonlinearVariationalProblem, NonlinearVariationalSolver,
)
from ..navier_stokes import NavierStokesSolution
from .data_structures import ReducedOperators, ROMSolution
from .rom_callbacks import make_callbacks

__all__ = [
    'solve_rom',
    'solve_rom_parameter_sweep',
    'reconstruct_velocity',
    'reconstruct_pressure',
    'reconstruct_solution',
    'project_to_rom_coefficients',
    'compute_rom_error',
]


# =============================================================================
# SOLVER SETUP
# =============================================================================

def _setup_rom_solver(
    operators: ReducedOperators,
    nu: float,
    amp: float,
    problem,
    deim_ops=None,
    solver_parameters: dict = None,
    submesh_deim=None,
):
    """
    Set up the ROM solver (internal).

    Automatically selects the right mode:
        - Exact vs DEIM (based on deim_ops)
        - Full vs quadratic-only (based on operators.has_affine_convection)

    Returns:
        solver, w_rom, callback_data
    """
    N_u = operators.n_velocity_modes
    N_p = operators.n_pressure_modes

    # ---- Determine mode ----
    use_deim = deim_ops is not None
    quadratic_only = operators.has_affine_convection
    thetas = operators._default_thetas(amp)

    mode_str = ("DEIM" if use_deim else "Exact")
    conv_str = ("quadratic-only" if quadratic_only else "full convection")

    print(f"\n  Setting up ROM solver...")
    print(f"    Mode: {mode_str}, {conv_str}")
    print(f"    N_velocity: {N_u}")
    print(f"    N_pressure: {N_p}")
    print(f"    nu: {nu} (Re = {1/nu:.1f})")
    print(f"    amp: {amp}")
    print(f"    thetas: {list(thetas)}")
    if use_deim:
        print(f"    DEIM modes (residual): {deim_ops.m_F}")
        print(f"    MDEIM modes (Jacobian): {deim_ops.m_J}")

    # ---- Online assembly ----
    if quadratic_only:
        system = operators.assemble_online(nu, list(thetas))
        A_eff = system.A_eff       # nu*A_N + C_lin(thetas)
        f_eff = system.f_eff       # f_N - lifting - c_0
        g_eff = system.g_eff       # g_N - lifting
    else:
        A_eff = nu * operators.A_N
        f_eff = operators.get_adjusted_forcing(nu, amp)
        g_eff = operators.get_adjusted_divergence_rhs(amp)

    # ---- Build submesh for DEIM (one-time setup) ----
    if use_deim and submesh_deim is None:
        from .hyper_reduction import SubMeshDEIM
        V_full = problem.velocity_space
        submesh_deim = SubMeshDEIM(V_full, deim_ops)
    elif use_deim:
        print(f"    Using pre-built SubMeshDEIM "
              f"({submesh_deim.V_sub.dim()} DOFs)")

    # ---- ROM mesh and function spaces ----
    rom_mesh = UnitIntervalMesh(1)
    V_rom = VectorFunctionSpace(rom_mesh, "DG", 0, dim=N_u)
    Q_rom = VectorFunctionSpace(rom_mesh, "DG", 0, dim=N_p)
    W_rom = V_rom * Q_rom

    w_rom = Function(W_rom)
    u_rom, p_rom = split(w_rom)
    v_rom, q_rom = TestFunctions(W_rom)
    du_rom, dp_rom = TrialFunctions(W_rom)

    # ---- Firedrake Constants ----
    A_eff_const = Constant(A_eff, rom_mesh)
    B_const = Constant(operators.B_N, rom_mesh)
    f_eff_const = Constant(f_eff, rom_mesh)
    g_eff_const = Constant(g_eff, rom_mesh)

    conv_res_const = Constant(np.zeros(N_u), rom_mesh)
    conv_J_const = Constant(np.zeros((N_u, N_u)), rom_mesh)

    # ---- Callback data ----
    V_full = problem.velocity_space

    callback_data = {
        # Basis and dimensions
        'Z_u': operators.Z_u,
        'N_u': N_u,
        'N_p': N_p,
        # Full-order workspace
        'V_full': V_full,
        'u_work': Function(V_full, name="u_work"),
        # Mode flags
        'use_deim': use_deim,
        'quadratic_only': quadratic_only,
        # Lifting (for full mode)
        'thetas': list(thetas),
        'lifting_dofs': operators.lifting_dofs,
        # DEIM (may be None)
        'deim_ops': deim_ops,
        'submesh_deim': submesh_deim,    # ← NEW: SubMeshDEIM or None
        # Firedrake Constants (updated by callbacks)
        'conv_res_const': conv_res_const,
        'conv_J_const': conv_J_const,
        # Mutable constants (for parameter updates)
        'A_eff_const': A_eff_const,
        'f_eff_const': f_eff_const,
        'g_eff_const': g_eff_const,
        # Parameters (for reference / updates)
        'operators': operators,
        'nu': nu,
        'amp': amp,
        # Tracking
        'iteration_count': 0,
    }

    # ---- Create callbacks ----
    res_cb, jac_cb = make_callbacks(callback_data)

    # ---- Variational forms (identical for all modes) ----

    # Residual
    F = inner(dot(A_eff_const, u_rom), v_rom) * dx       # diffusion + linear convection
    F += inner(conv_res_const, v_rom) * dx                 # nonlinear convection (callback)
    F += inner(dot(B_const.T, p_rom), v_rom) * dx          # pressure gradient
    F -= inner(f_eff_const, v_rom) * dx                    # forcing + lifting
    F += inner(dot(B_const, u_rom), q_rom) * dx            # divergence
    F -= inner(g_eff_const, q_rom) * dx                    # divergence RHS

    # Jacobian
    J = inner(dot(A_eff_const, du_rom), v_rom) * dx        # diffusion + linear convection
    J += inner(dot(conv_J_const, du_rom), v_rom) * dx      # nonlinear convection (callback)
    J += inner(dot(B_const.T, dp_rom), v_rom) * dx         # pressure gradient
    J += inner(dot(B_const, du_rom), q_rom) * dx           # divergence

    # ---- PETSc solver ----
    if solver_parameters is None:
        # solver_parameters = {
        #     'snes_type': 'newtonls',
        #     'snes_linesearch_type': 'bt',
        #     'snes_monitor': None,
        #     'snes_converged_reason': None,
        #     'snes_max_it': 100,
        #     'snes_rtol': 1e-6,
        #     'snes_atol': 1e-7,
        #     'snes_stol': 1e-8,
        #     'ksp_type': 'preonly',
        #     'pc_type': 'lu',
        #     'pc_factor_mat_solver_type': 'mumps',
        #     'mat_type': 'aij',
        # }
        solver_parameters = {
            'snes_type': 'newtonls',
            'snes_linesearch_type': 'bt',
            'snes_monitor': None,
            'snes_converged_reason': None,
            'snes_max_it': 150,          # bump from 100 — bifurcation points need more room
            'snes_rtol': 1e-7,           # tighter — you want well-converged snapshots for POD
            'snes_atol': 1e-8,          # tighter — matters for continuation where initial residual is small
            'snes_stol': 1e-12,          # very tight — don't let stagnation fake convergence
            'ksp_type': 'preonly',
            'pc_type': 'lu',
            'pc_factor_mat_solver_type': 'mumps',
            'mat_type': 'aij',
        }
    rom_problem = NonlinearVariationalProblem(F, w_rom, J=J)
    solver = NonlinearVariationalSolver(
        rom_problem,
        solver_parameters=solver_parameters,
        pre_function_callback=res_cb,
        pre_jacobian_callback=jac_cb,
    )

    print("    ✓ ROM solver created")

    return solver, w_rom, callback_data


def _update_parameters(callback_data, nu_new, amp_new):
    """
    Update ROM parameters without recreating the solver.

    Updates Constants in-place so the existing variational forms
    pick up the new values automatically.
    """
    operators = callback_data['operators']
    thetas_new = operators._default_thetas(amp_new)

    # Update callback state
    callback_data['nu'] = nu_new
    callback_data['amp'] = amp_new
    callback_data['thetas'] = list(thetas_new)
    callback_data['iteration_count'] = 0

    # Recompute online system
    if callback_data['quadratic_only']:
        system = operators.assemble_online(nu_new, list(thetas_new))
        callback_data['A_eff_const'].assign(system.A_eff)
        callback_data['f_eff_const'].assign(system.f_eff)
        callback_data['g_eff_const'].assign(system.g_eff)
    else:
        callback_data['A_eff_const'].assign(nu_new * operators.A_N)
        callback_data['f_eff_const'].assign(
            operators.get_adjusted_forcing(nu_new, amp_new)
        )
        callback_data['g_eff_const'].assign(
            operators.get_adjusted_divergence_rhs(amp_new)
        )


# =============================================================================
# MAIN SOLVE FUNCTION
# =============================================================================

def solve_rom(
    operators: ReducedOperators,
    nu: float,
    amp: float,
    problem,
    deim_ops=None,
    submesh_deim=None,
    u_initial_guess: np.ndarray = None,
    p_initial_guess: np.ndarray = None,
    solver_parameters: dict = None,
    store_full_dofs: bool = True,
) -> ROMSolution:
    """
    Solve the reduced Navier-Stokes system.

    Automatically selects the right mode:
        - deim_ops=None  → exact projection
        - deim_ops=...   → DEIM/MDEIM projection (with submesh hyper-reduction)
        - operators.has_affine_convection → quadratic-only callbacks

    Args:
        operators: ReducedOperators
        nu: Viscosity (1/Re)
        amp: Amplitude parameter
        problem: NavierStokesProblem (for FOM assembly in callbacks)
        deim_ops: DEIMOnlineOperators or None
        submesh_deim: Pre-built SubMeshDEIM or None. If None and deim_ops
                      is provided, it will be built internally. Pass a
                      pre-built one to avoid rebuilding across multiple calls.
        initial_guess: (N_u,) initial ROM velocity coefficients
        solver_parameters: PETSc solver parameters
        store_full_dofs: If True, reconstruct and store full-order DOFs

    Returns:
        ROMSolution
    """
    print("=" * 60)
    print("SOLVING ROM")
    print("=" * 60)
    print(f"  nu = {nu} (Re = {1/nu:.1f})")
    print(f"  amp = {amp}")

    # Setup
    solver, w_rom, callback_data = _setup_rom_solver(
        operators, nu, amp, problem, deim_ops, solver_parameters,
        submesh_deim=submesh_deim,
    )

    # Apply initial guess
    u_sub, p_sub = w_rom.subfunctions
    if u_initial_guess is not None and p_initial_guess is not None:
        u_sub.dat.data[:] = u_initial_guess.reshape(u_sub.dat.data.shape)
        p_sub.dat.data[:] = p_initial_guess.reshape(p_sub.dat.data.shape)
        print(f"  Using initial guess: ||alpha_0|| = {np.linalg.norm(u_initial_guess):.6e}")

    # Solve
    converged = True
    try:
        solver.solve()
        iterations = callback_data['iteration_count']
        print(f"\n  ✓ Converged in {iterations} iterations")
    except Exception as e:
        converged = False
        iterations = callback_data['iteration_count']
        print(f"\n  ✗ Solver failed after {iterations} iterations: {e}")
        traceback.print_exc()

    # Extract solution
    velocity_coeffs = u_sub.dat.data_ro.flatten().copy()
    pressure_coeffs = p_sub.dat.data_ro.flatten().copy()

    # Reconstruct full-order DOFs
    velocity_dofs = None
    pressure_dofs = None
    if store_full_dofs:
        thetas = operators._default_thetas(amp)
        u_lift = np.einsum('i,ij->j', thetas, operators.lifting_dofs)
        velocity_dofs = u_lift + operators.Z_u @ velocity_coeffs
        pressure_dofs = operators.Z_p @ pressure_coeffs

    # Build metadata
    meta = {
        'nu': nu,
        'n_velocity_modes': operators.n_velocity_modes,
        'n_pressure_modes': operators.n_pressure_modes,
        'mode': 'deim' if deim_ops is not None else 'exact',
        'quadratic_only': operators.has_affine_convection,
    }
    if deim_ops is not None:
        meta['deim_modes'] = deim_ops.m_F
        meta['mdeim_modes'] = deim_ops.m_J

    solution = ROMSolution(
        velocity_coeffs=velocity_coeffs,
        pressure_coeffs=pressure_coeffs,
        reynolds=1.0 / nu,
        amplitude=amp,
        velocity_dofs=velocity_dofs,
        pressure_dofs=pressure_dofs,
        iterations=iterations,
        converged=converged,
        metadata=meta,
    )

    print(f"  ||alpha|| = {np.linalg.norm(velocity_coeffs):.6e}")
    print(f"  ||beta||  = {np.linalg.norm(pressure_coeffs):.6e}")
    print("")

    print("#"*100)
    print("")

    return solution


# =============================================================================
# PARAMETER SWEEP
# =============================================================================

def solve_rom_parameter_sweep(
    operators: ReducedOperators,
    problem,
    reynolds_values: np.ndarray = None,
    amplitude_values: np.ndarray = None,
    deim_ops=None,
    submesh_deim=None,
    use_continuation: bool = True,
    store_full_dofs: bool = False,
    solver_parameters: dict = None,
) -> list:
    """
    Solve ROM for multiple parameter values.

    Reuses solver across parameter values (only updates Constants).

    Args:
        operators: ReducedOperators
        problem: NavierStokesProblem
        reynolds_values: Array of Reynolds numbers
        amplitude_values: Array of amplitude values
        deim_ops: DEIMOnlineOperators or None
        submesh_deim: Pre-built SubMeshDEIM or None
        use_continuation: Use previous solution as initial guess
        store_full_dofs: Store full-order DOFs for each solution
        solver_parameters: PETSc solver parameters

    Returns:
        List of ROMSolution
    """
    if reynolds_values is None:
        reynolds_values = np.array([100.0])
    if amplitude_values is None:
        amplitude_values = np.array([0.0])

    n_total = len(reynolds_values) * len(amplitude_values)

    print("=" * 60)
    print("ROM PARAMETER SWEEP")
    print("=" * 60)
    print(f"  Reynolds:   {reynolds_values}")
    print(f"  Amplitude:  {amplitude_values}")
    print(f"  Total:      {n_total} solves")
    print(f"  Mode:       {'DEIM' if deim_ops else 'Exact'}")
    print(f"  Quadratic:  {operators.has_affine_convection}")

    # Create solver once with first parameter values
    nu_0 = 1.0 / reynolds_values[0]
    amp_0 = amplitude_values[0]

    solver, w_rom, callback_data = _setup_rom_solver(
        operators, nu_0, amp_0, problem, deim_ops, solver_parameters,
        submesh_deim=submesh_deim,
    )

    solutions = []
    prev_guess = None

    for amp in amplitude_values:
        for Re in reynolds_values:
            nu = 1.0 / Re
            print(f"\n{'─'*40}")
            print(f"  Re = {Re:.1f}, amp = {amp:.3f}")
            print(f"{'─'*40}")

            # Update parameters (no solver recreation)
            _update_parameters(callback_data, nu, amp)

            # Apply initial guess
            u_sub, p_sub = w_rom.subfunctions
            if prev_guess is not None and use_continuation:
                u_sub.dat.data[:] = prev_guess.reshape(u_sub.dat.data.shape)
            else:
                # Reset to zero
                w_rom.assign(0)

            # Solve
            converged = True
            try:
                solver.solve()
                iterations = callback_data['iteration_count']
                print(f"  ✓ Converged in {iterations} iterations")
            except Exception as e:
                converged = False
                iterations = callback_data['iteration_count']
                print(f"  ✗ Failed after {iterations} iterations: {e}")

            # Extract solution
            velocity_coeffs = u_sub.dat.data_ro.flatten().copy()
            pressure_coeffs = p_sub.dat.data_ro.flatten().copy()

            velocity_dofs = None
            pressure_dofs = None
            if store_full_dofs:
                thetas = operators._default_thetas(amp)
                u_lift = np.einsum('i,ij->j', thetas, operators.lifting_dofs)
                velocity_dofs = u_lift + operators.Z_u @ velocity_coeffs
                pressure_dofs = operators.Z_p @ pressure_coeffs

            solution = ROMSolution(
                velocity_coeffs=velocity_coeffs,
                pressure_coeffs=pressure_coeffs,
                reynolds=Re,
                amplitude=amp,
                velocity_dofs=velocity_dofs,
                pressure_dofs=pressure_dofs,
                iterations=iterations,
                converged=converged,
                metadata={
                    'nu': nu,
                    'mode': 'deim' if deim_ops else 'exact',
                },
            )
            solutions.append(solution)

            if use_continuation:
                prev_guess = velocity_coeffs

    # Summary
    n_converged = sum(s.converged for s in solutions)
    print(f"\n{'='*60}")
    print(f"SWEEP COMPLETE: {n_converged}/{n_total} converged")
    print(f"{'='*60}")

    return solutions


# =============================================================================
# RECONSTRUCTION
# =============================================================================

def reconstruct_velocity(
    solution: ROMSolution,
    operators: ReducedOperators,
    problem,
    amp: float = None,
) -> "Function":
    """Reconstruct full-order velocity from ROM solution."""
    if amp is None:
        amp = solution.amplitude

    # if solution.velocity_dofs is not None:
    #     velocity_dofs = solution.velocity_dofs
    # else:
    thetas = operators._default_thetas(amp)
    u_lift = np.einsum('i,ij->j', thetas, operators.lifting_dofs)
    velocity_dofs = u_lift + operators.Z_u @ solution.velocity_coeffs

    u_full = Function(problem.velocity_space, name="Velocity")
    u_full.dat.data[:] = velocity_dofs.reshape(u_full.dat.data.shape)
    return u_full


def reconstruct_pressure(
    solution: ROMSolution,
    operators: ReducedOperators,
    problem,
) -> "Function":
    """Reconstruct full-order pressure from ROM solution."""
    if solution.pressure_dofs is not None:
        pressure_dofs = solution.pressure_dofs
    else:
        pressure_dofs = operators.Z_p @ solution.pressure_coeffs

    p_full = Function(problem.pressure_space, name="Pressure")
    p_full.dat.data[:] = pressure_dofs.reshape(p_full.dat.data.shape)
    return p_full

def reconstruct_solution(
    solution: ROMSolution,
    operators: ReducedOperators,
    problem,
    amp: float,
) -> NavierStokesSolution:
    """Reconstruct both velocity and pressure, returning a NavierStokesSolution."""
    u_full = reconstruct_velocity(solution, operators, problem, amp)
    p_full = reconstruct_pressure(solution, operators, problem)

    mixed = Function(problem.mixed_space, name="Solution")
    mixed.sub(0).assign(u_full)
    mixed.sub(1).assign(p_full)

    return NavierStokesSolution(
        solution=mixed,
        velocity=u_full,
        pressure=p_full,
        reynolds=solution.reynolds,
        amplitude=solution.amplitude if amp is None else amp,
    )


# =============================================================================
# PROJECTION (FOM → ROM coefficients)
# =============================================================================

def project_to_rom_coefficients(
    velocity_dofs: np.ndarray,
    pressure_dofs: np.ndarray,
    operators: ReducedOperators,
    amp: float,
    M_u,
    M_p,
) -> tuple:
    """
    Project full-order DOFs onto the ROM basis.

    Computes:
        u_hom = u_dofs - sum(theta_k * u_k)
        alpha = Z_u^T @ M_u @ u_hom
        beta  = Z_p^T @ M_p @ p_dofs

    Args:
        velocity_dofs: (n_dofs_u,) FOM velocity DOFs
        pressure_dofs: (n_dofs_p,) FOM pressure DOFs
        operators: ReducedOperators
        amp: Amplitude parameter
        M_u: Velocity inner product matrix (sparse)
        M_p: Pressure inner product matrix (sparse)

    Returns:
        (alpha, beta): ROM coefficient vectors
    """
    thetas = operators._default_thetas(amp)
    u_lift = np.einsum('i,ij->j', thetas, operators.lifting_dofs)
    u_hom = velocity_dofs - u_lift

    alpha = operators.Z_u.T @ (M_u @ u_hom)
    beta = operators.Z_p.T @ (M_p @ pressure_dofs)

    return alpha, beta


# =============================================================================
# ERROR COMPUTATION
# =============================================================================

def compute_rom_error(u_fom, u_rom, M_u):
    """
    Compute relative error in the inner product norm.

    ||u_fom - u_rom||_M / ||u_fom||_M
    """
    u_fom_dofs = u_fom.dat.data_ro.flatten()
    u_rom_dofs = u_rom.dat.data_ro.flatten()
    error = u_fom_dofs - u_rom_dofs

    error_norm = np.sqrt(error.T @ (M_u @ error))
    fom_norm = np.sqrt(u_fom_dofs.T @ (M_u @ u_fom_dofs))

    return error_norm / fom_norm