"""
ROM Solver.

Solves reduced-order Navier-Stokes system with Newton iteration.
Handles nonlinear convection via callbacks.
"""

import numpy as np
import scipy.sparse as sp
from firedrake import (
    Function, TestFunction, TrialFunction, TestFunctions, TrialFunctions,
    VectorFunctionSpace,
    UnitIntervalMesh, Constant,
    inner, dot, grad, div, dx, split,
    NonlinearVariationalProblem, NonlinearVariationalSolver,
    assemble
)

from .data_structures import ReducedOperators, ROMSolution

__all__ = [
    'setup_rom_solver',
    'solve_rom',
    'reconstruct_solution',
    'reconstruct_velocity',
    'reconstruct_pressure',
    'solve_rom_parameter_sweep',
]


# =============================================================================
# ROM SOLVER SETUP
# =============================================================================

def setup_rom_solver(
    operators: ReducedOperators,
    nu: float,
    amp: float,
    problem,
    solver_parameters: dict = None
):
    """
    Set up ROM solver with Newton iteration and convection callbacks.

    Args:
        operators: ReducedOperators from build_reduced_operators
        nu: Viscosity (1/Re)
        amp: Amplitude parameter
        problem: NavierStokesProblem (for full-order assembly in callbacks)
        solver_parameters: Optional PETSc solver parameters

    Returns:
        solver: NonlinearVariationalSolver
        w_rom: Solution Function
        callback_data: Dict with callback state for inspection
    """
    # Extract dimensions
    N_u = operators.n_velocity_modes
    N_p = operators.n_pressure_modes

    print(f"\n  Setting up ROM solver...")
    print(f"    N_velocity: {N_u}")
    print(f"    N_pressure: {N_p}")
    print(f"    nu: {nu}")
    print(f"    amp: {amp}")

    # Create ROM mesh (dummy 1D mesh)
    rom_mesh = UnitIntervalMesh(1)

    # ROM function spaces (vector DG0 to store coefficients)
    V_rom = VectorFunctionSpace(rom_mesh, "DG", 0, dim=N_u)
    Q_rom = VectorFunctionSpace(rom_mesh, "DG", 0, dim=N_p)
    W_rom = V_rom * Q_rom

    # Solution and test functions
    w_rom = Function(W_rom)
    u_rom, p_rom = split(w_rom)
    v_rom, q_rom = TestFunctions(W_rom)

    # Trial functions for Jacobian
    du_rom, dp_rom = TrialFunctions(W_rom)

    # Convert operators to Firedrake Constants
    A_const = Constant(operators.A_N, rom_mesh)
    B_const = Constant(operators.B_N, rom_mesh)

    # Adjusted forcing (includes lifting contributions)
    f_adjusted = operators.get_adjusted_forcing(nu, amp)
    f_const = Constant(f_adjusted, rom_mesh)

    # Adjusted divergence RHS
    g_adjusted = operators.get_adjusted_divergence_rhs(amp)
    g_const = Constant(g_adjusted, rom_mesh)

    # Convection matrices (updated by callbacks)

    N_N = np.zeros((N_u, N_u))
    N_res = np.zeros((N_u, N_u))
    N_jac = np.zeros((N_u, N_u))
    N_const_res = Constant(N_res, rom_mesh)
    N_const_jac = Constant(N_jac, rom_mesh)

    M_N = np.zeros((N_u, N_u))
    M_const_jac = Constant(M_N, rom_mesh)

    conv_residual = np.zeros(N_u)
    conv_res_const = Constant(conv_residual, rom_mesh)

    # Full-order workspace
    V_full = problem.velocity_space

    # Callback data storage
    callback_data = {
        # Operators and basis
        'operators': operators,
        'Z_u': operators.Z_u,
        'Z_p': operators.Z_p,
        # Parameters
        'nu': nu,
        'amp': amp,
        'N_u': N_u,
        'N_p': N_p,
        # Full-order workspace
        'V_full': V_full,
        'u_full_total': Function(V_full, name="u_total"),
        'u_full_reduced': Function(V_full, name="u_reduced"),
        # Lifting data
        'u_base': operators.u_base,
        'u_control': operators.u_control,
        # Convection constants (updated by callbacks)
        'N_const_res': N_const_res,
        'N_const_jac': N_const_jac,
        'M_const_jac': M_const_jac,
        'conv_res_const': conv_res_const,  # Vector for residual
        # Iteration tracking
        'iteration_count': 0,
    }

    # =========================================================================
    # CALLBACKS
    # =========================================================================

    def update_convection_residual(current_solution):
        """Update convection matrix for residual evaluation."""
        Z_u = callback_data['Z_u']
        u_base = callback_data['u_base']
        u_control = callback_data['u_control']

        # Extract coefficients using array_r
        u_rom_coeffs = current_solution.array_r[:N_u]

        # Reconstruct total velocity
        u_reduced_part = Z_u @ u_rom_coeffs
        u_total_np = u_base + amp * u_control + u_reduced_part

        # Update function
        callback_data['u_full_total'].dat.data[:] = u_total_np.reshape(
            callback_data['u_full_total'].dat.data.shape
        )

        # Assemble convection matrix
        u_trial = TrialFunction(V_full)
        v_test = TestFunction(V_full)
        N_form = inner(dot(grad(u_trial), callback_data['u_full_total']), v_test) * dx
        # N_form = inner(dot( callback_data['u_full_total'], grad(u_trial)), v_test) * dx

        N_full = assemble(N_form)

        # Convert to scipy and project
        N_petsc = N_full.M.handle
        indptr, indices, data = N_petsc.getValuesCSR()
        N_scipy = sp.csr_matrix((data, indices, indptr), shape=N_petsc.getSize())
        N_N = Z_u.T @ (N_scipy @ Z_u)
        # Convection contribution to residual
        conv_residual = Z_u.T @ (N_scipy @ u_total_np)

    # Store this vector, not a matrix!
        callback_data['conv_res_const'].assign(conv_residual)
        callback_data['N_const_res'].assign(N_N)

    def update_jacobian_matrices(current_solution):
        """Update N_N and M_N for Jacobian."""
        callback_data['iteration_count'] += 1

        Z_u = callback_data['Z_u']
        u_base = callback_data['u_base']
        u_control = callback_data['u_control']

        # Extract coefficients using array_r
        u_rom_coeffs = current_solution.array_r[:N_u]

        # Reconstruct velocities
        u_reduced_np = Z_u @ u_rom_coeffs
        u_total_np = u_base + amp * u_control + u_reduced_np

        # Update functions
        callback_data['u_full_total'].dat.data[:] = u_total_np.reshape(
            callback_data['u_full_total'].dat.data.shape
        )
        callback_data['u_full_reduced'].dat.data[:] = u_reduced_np.reshape(
            callback_data['u_full_reduced'].dat.data.shape
        )

        u_total = callback_data['u_full_total']

        u_trial = TrialFunction(V_full)
        v_test = TestFunction(V_full)

        # N: (u_total · ∇) u_trial
        N_form = inner(dot(grad(u_trial), u_total), v_test) * dx
        # N_form = inner(dot(u_total, grad(u_trial)), v_test) * dx  #
        N_full = assemble(N_form)

        N_petsc = N_full.M.handle
        indptr_N, indices_N, data_N = N_petsc.getValuesCSR()
        N_scipy = sp.csr_matrix((data_N, indices_N, indptr_N), shape=N_petsc.getSize())
        N_N = Z_u.T @ (N_scipy @ Z_u)

        # M: (u_trial · ∇) u_total
        M_form = inner(dot(grad(u_total), u_trial), v_test) * dx

        # M_form = inner(dot(u_trial, grad(u_total)), v_test) * dx  # wrong
        M_full = assemble(M_form)

        M_petsc = M_full.M.handle
        indptr_M, indices_M, data_M = M_petsc.getValuesCSR()
        M_scipy = sp.csr_matrix((data_M, indices_M, indptr_M), shape=M_petsc.getSize())
        M_N = Z_u.T @ (M_scipy @ Z_u)

        callback_data['N_const_jac'].assign(N_N)
        callback_data['M_const_jac'].assign(M_N)

        if callback_data['iteration_count'] % 1 == 0:
            print(f"      Iteration {callback_data['iteration_count']}: "
                  f"||N_N|| = {np.linalg.norm(N_N):.6e}, "
                  f"||M_N|| = {np.linalg.norm(M_N):.6e}")

    # =========================================================================
    # VARIATIONAL FORMS - Use dot() not @
    # =========================================================================

    # Residual form
    F = nu * inner(dot(A_const, u_rom), v_rom) * dx
    # F += inner(dot(N_const_res, u_rom), v_rom) * dx
    F += inner(conv_res_const, v_rom) * dx  # CORRECT
    F += inner(dot(B_const.T, p_rom), v_rom) * dx
    F -= inner(f_const, v_rom) * dx
    F += inner(dot(B_const, u_rom), q_rom) * dx
    F -= inner(g_const, q_rom) * dx

    # Jacobian form
    J = nu * inner(dot(A_const, du_rom), v_rom) * dx
    J += inner(dot(N_const_jac, du_rom), v_rom) * dx
    J += inner(dot(M_const_jac, du_rom), v_rom) * dx
    J += inner(dot(B_const.T, dp_rom), v_rom) * dx
    J += inner(dot(B_const, du_rom), q_rom) * dx

    # =========================================================================
    # SOLVER SETUP
    # =========================================================================

    if solver_parameters is None:
        solver_parameters = {
            'snes_type': 'newtonls',
            'snes_linesearch_type': 'bt',#or put 'l2'
            'snes_monitor': None,
            'snes_converged_reason': None,
            'snes_max_it': 100,
            'snes_rtol': 1e-6,
            'snes_atol': 1e-7,
            'snes_stol': 1e-8,
            'ksp_type': 'preonly',
            'pc_type': 'lu',
            'pc_factor_mat_solver_type': 'mumps',
            'mat_type': 'aij',
        }

    # Create problem and solver
    rom_problem = NonlinearVariationalProblem(F, w_rom, J=J)

    solver = NonlinearVariationalSolver(
        rom_problem,
        solver_parameters=solver_parameters,
        pre_function_callback=update_convection_residual,
        pre_jacobian_callback=update_jacobian_matrices
    )

    print("    ✓ ROM solver created")

    return solver, w_rom, callback_data


# =============================================================================
# ROM SOLVE
# =============================================================================

def solve_rom(
    operators: ReducedOperators,
    nu: float,
    amp: float,
    problem,
    initial_guess: np.ndarray = None,
    solver_parameters: dict = None,
    store_full_dofs: bool = True,
    return_callback_data: bool = False
) -> ROMSolution:
    """
    Solve ROM for given parameters.

    Args:
        operators: ReducedOperators
        nu: Viscosity (1/Re)
        amp: Amplitude parameter
        problem: NavierStokesProblem
        initial_guess: Optional (N_u + N_p,) initial guess
        solver_parameters: Optional PETSc parameters
        store_full_dofs: If True, store reconstructed full-order DOFs
        return_callback_data: If True, also return callback_data dict

    Returns:
        ROMSolution (or tuple (ROMSolution, callback_data) if return_callback_data)
    """
    print("=" * 60)
    print("SOLVING ROM")
    print("=" * 60)
    print(f"  nu = {nu} (Re = {1/nu:.1f})")
    print(f"  amp = {amp}")

    # Setup solver
    solver, w_rom, callback_data = setup_rom_solver(
        operators, nu, amp, problem, solver_parameters
    )
    u_rom, p_rom = w_rom.subfunctions
    # Set initial guess
    if initial_guess is not None:
        u_rom.dat.data[:] = initial_guess
        print(f"  Using provided initial guess")

    # Solve
    callback_data['iteration_count'] = 0
    converged = True
    residual_norm = 0.0

    try:
        solver.solve()
        iterations = callback_data['iteration_count']
        print(f"\n  ✓ Converged in {iterations} iterations")
    except Exception as e:
        converged = False
        iterations = callback_data['iteration_count']
        print(f"\n  ✗ Solver failed after {iterations} iterations: {e}")

    # Extract coefficients using subfunctions
    N_u = operators.n_velocity_modes
    N_p = operators.n_pressure_modes


    velocity_coeffs = u_rom.dat.data_ro.flatten().copy()
    pressure_coeffs = p_rom.dat.data_ro.flatten().copy()

    # Optionally reconstruct full DOFs
    velocity_dofs = None
    pressure_dofs = None

    if store_full_dofs:
        velocity_dofs = (
            operators.u_base +
            amp * operators.u_control +
            operators.Z_u @ velocity_coeffs
        )
        pressure_dofs = operators.Z_p @ pressure_coeffs

    # Create solution
    solution = ROMSolution(
        velocity_coeffs=velocity_coeffs,
        pressure_coeffs=pressure_coeffs,
        reynolds=1.0 / nu,
        amplitude=amp,
        velocity_dofs=velocity_dofs,
        pressure_dofs=pressure_dofs,
        iterations=iterations,
        converged=converged,
        residual_norm=residual_norm,
        metadata={
            'nu': nu,
            'n_velocity_modes': N_u,
            'n_pressure_modes': N_p,
        }
    )

    print(f"\n  ||u_rom|| = {np.linalg.norm(velocity_coeffs):.6e}")
    print(f"  ||p_rom|| = {np.linalg.norm(pressure_coeffs):.6e}")

    if return_callback_data:
        return solution, callback_data
    return solution


# =============================================================================
# RECONSTRUCTION
# =============================================================================

def reconstruct_velocity(
    solution: ROMSolution,
    operators: ReducedOperators,
    problem,
    amp: float = None
) -> Function:
    """
    Reconstruct full-order velocity from ROM solution.

    u_full = u_base + amp * u_control + Z_u @ u_rom
    """
    if amp is None:
        amp = solution.amplitude

    if solution.velocity_dofs is not None:
        velocity_dofs = solution.velocity_dofs
    else:
        velocity_dofs = (
            operators.u_base +
            amp * operators.u_control +
            operators.Z_u @ solution.velocity_coeffs
        )

    u_full = Function(problem.velocity_space, name="Velocity")
    u_full.dat.data[:] = velocity_dofs.reshape(u_full.dat.data.shape)

    return u_full


def reconstruct_pressure(
    solution: ROMSolution,
    operators: ReducedOperators,
    problem
) -> Function:
    """
    Reconstruct full-order pressure from ROM solution.

    p_full = Z_p @ p_rom
    """
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
    amp: float = None
):
    """
    Reconstruct full-order velocity and pressure.

    Returns:
        (u_full, p_full): Tuple of Firedrake Functions
    """
    u_full = reconstruct_velocity(solution, operators, problem, amp)
    p_full = reconstruct_pressure(solution, operators, problem)

    return u_full, p_full


# =============================================================================
# PARAMETER SWEEP
# =============================================================================

def solve_rom_parameter_sweep(
    operators: ReducedOperators,
    problem,
    reynolds_values: np.ndarray = None,
    amplitude_values: np.ndarray = None,
    use_continuation: bool = True,
    store_full_dofs: bool = False,
    solver_parameters: dict = None
):
    """
    Solve ROM for multiple parameter values.

    Args:
        operators: ReducedOperators
        problem: NavierStokesProblem
        reynolds_values: Array of Reynolds numbers (default: [100])
        amplitude_values: Array of amplitudes (default: [0])
        use_continuation: Use previous solution as initial guess
        store_full_dofs: If True, reconstruct full DOFs for each solution
        solver_parameters: Optional PETSc parameters

    Returns:
        List of ROMSolution
    """
    if reynolds_values is None:
        reynolds_values = np.array([100.0])
    if amplitude_values is None:
        amplitude_values = np.array([0.0])

    print("=" * 60)
    print("ROM PARAMETER SWEEP")
    print("=" * 60)
    print(f"  Reynolds: {reynolds_values}")
    print(f"  Amplitude: {amplitude_values}")
    print(f"  Total solves: {len(reynolds_values) * len(amplitude_values)}")

    solutions = []
    prev_guess = None

    for amp in amplitude_values:
        for Re in reynolds_values:
            nu = 1.0 / Re

            print(f"\n--- Re = {Re:.1f}, amp = {amp:.3f} ---")

            solution = solve_rom(
                operators=operators,
                nu=nu,
                amp=amp,
                problem=problem,
                initial_guess=prev_guess,
                solver_parameters=solver_parameters,
                store_full_dofs=store_full_dofs
            )

            solutions.append(solution)

            # Use as initial guess for next solve
            if use_continuation:
                prev_guess = solution.velocity_coeffs



    print("\n" + "=" * 60)
    print("PARAMETER SWEEP COMPLETE")
    print("=" * 60)
    print(f"  Total solutions: {len(solutions)}")
    print(f"  Converged: {sum(s.converged for s in solutions)}/{len(solutions)}")

    return solutions

def calculate_error(rom_solution: ROMSolution, branch: str):
    Re = rom_solution.reynolds
    amp = rom_solution.amplitude
    # if branch == # can be "symm", "up", or "down"
    # full_solution
    return None



def project_to_rom_coefficients(
    velocity_dofs: np.ndarray,
    pressure_dofs: np.ndarray,
    operators: ReducedOperators,
    amp: float,
    M_u,  # Need this!
    M_p   # Need this!
) -> np.ndarray:
    """Project full-order solution to ROM coefficients."""
    u_homog = velocity_dofs - operators.u_base - amp * operators.u_control

    # Correct projection for M-orthonormal basis
    u_coeffs = operators.Z_u.T @ (M_u @ u_homog)
    p_coeffs = operators.Z_p.T @ (M_p @ pressure_dofs)

    return u_coeffs,p_coeffs


# ERROR COMPUTATION — needs M for consistent norm
def compute_rom_error(u_fom, u_rom, M_u):
    """Compute error in the same norm used for POD."""
    u_fom_dofs = u_fom.dat.data_ro.flatten()
    u_rom_dofs = u_rom.dat.data_ro.flatten()

    error = u_fom_dofs - u_rom_dofs

    error_norm = np.sqrt(error.T @ (M_u @ error))
    fom_norm = np.sqrt(u_fom_dofs.T @ (M_u @ u_fom_dofs))

    return error_norm / fom_norm
