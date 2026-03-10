

import numpy as np
import scipy.sparse as sp
from firedrake import *
import scipy.sparse as sp
from .data_structures import ReducedOperators, ROMSolution
from online_operators import DEIMOnlineOperators  # Changed import
import traceback
__all__ = [
    'setup_rom_solver_deim',
    'solve_rom_deim',
    'reconstruct_solution',
    'reconstruct_velocity',
    'reconstruct_pressure',
    'solve_rom_parameter_sweep_deim',
]


def setup_rom_solver_deim(
    operators: ReducedOperators,
    nu: float,
    amp: float,
    problem,
    deim_ops: DEIMOnlineOperators,  # Renamed for clarity
    solver_parameters: dict = None,
    debug: bool = False,  # Add flag
):
    """Set up ROM solver with DEIM for nonlinear terms."""
    N_u = operators.n_velocity_modes
    N_p = operators.n_pressure_modes

    print(f"\n  Setting up ROM solver with DEIM...")
    print(f"    N_velocity: {N_u}")
    print(f"    N_pressure: {N_p}")
    print(f"    DEIM modes (convective): {deim_ops.m_F}")
    print(f"    MDEIM modes (Jacobian): {deim_ops.m_J}")
    print(f"    nu: {nu}")
    print(f"    amp: {amp}")

    # Create ROM mesh
    rom_mesh = UnitIntervalMesh(1)

    # ROM function spaces
    V_rom = VectorFunctionSpace(rom_mesh, "DG", 0, dim=N_u)
    Q_rom = VectorFunctionSpace(rom_mesh, "DG", 0, dim=N_p)
    W_rom = V_rom * Q_rom

    # Solution and test functions
    w_rom = Function(W_rom)
    u_rom, p_rom = split(w_rom)
    v_rom, q_rom = TestFunctions(W_rom)
    du_rom, dp_rom = TrialFunctions(W_rom)

    # Convert operators to Constants
    A_const = Constant(operators.A_N, rom_mesh)
    B_const = Constant(operators.B_N, rom_mesh)
    f_const = Constant(operators.get_adjusted_forcing(nu, amp), rom_mesh)
    g_const = Constant(operators.get_adjusted_divergence_rhs(amp), rom_mesh)

    # DEIM terms (updated by callbacks)
    conv_res_const = Constant(np.zeros(N_u), rom_mesh)
    conv_J_const = Constant(np.zeros((N_u, N_u)), rom_mesh)

    # Full-order workspace
    V_full = problem.velocity_space
    error_res = 0.0

    # Callback data
    callback_data = {
        'operators': operators,
        'Z_u': operators.Z_u,
        'Z_p': operators.Z_p,
        'nu': nu,
        'amp': amp,
        'N_u': N_u,
        'N_p': N_p,
        'deim_ops': deim_ops,
        'V_full': V_full,
        'u_full_total': Function(V_full, name="u_total"),
        'u_base': operators.u_base,
        'u_control': operators.u_control,
        'conv_res_const': conv_res_const,
        'conv_J_const': conv_J_const,
        'iteration_count': 0,
        'debug': debug,
        'print_debug': False,
        'error_res': error_res,
    }


    def update_convection_residual(current_solution):
        """Update convection residual using DEIM."""
        Z_u = callback_data['Z_u']
        u_base = callback_data['u_base']
        u_control = callback_data['u_control']
        amp_val = callback_data['amp']
        deim_ops = callback_data['deim_ops']

        # u_rom_coeffs = current_solution.dat.data_ro[:N_u].flatten()
        # u_sub, p_sub = current_solution.subfunctions
        # u_rom_coeffs = u_sub.dat.data_ro.flatten()
        # Extract directly from PETSc Vec
        coeffs = current_solution.array_r  # Read-only array
        u_rom_coeffs = coeffs[:N_u]
        # Reconstruct total velocity
        u_total_np = u_base + amp_val * u_control + Z_u @ u_rom_coeffs
        callback_data['u_full_total'].dat.data[:] = u_total_np.reshape(
            callback_data['u_full_total'].dat.data.shape
        )

        # Assemble convection residual
        u_total = callback_data['u_full_total']
        v_test = TestFunction(V_full)
        F_form = inner(grad(u_total) * u_total, v_test) * dx
        F_vec = assemble(F_form).dat.data_ro.flatten()
        F_vec_red = Z_u.T @ F_vec

        # DEIM approximation (pass full vector)
        conv_residual = deim_ops.reduced_convective(F_vec)
        callback_data['conv_res_const'].assign(conv_residual)
        # callback_data['conv_res_const'].assign(F_vec_red)
        callback_data['error_res'] = np.linalg.norm(F_vec_red - conv_residual) / np.linalg.norm(F_vec_red)
        print(f'The error inside convection callback is :{np.linalg.norm(F_vec_red - conv_residual) / np.linalg.norm(F_vec_red)}')


    def update_jacobian_matrices(current_solution):
        """Update convection Jacobian using MDEIM."""
        callback_data['iteration_count'] += 1
        callback_data['print_debug'] = True  # Enable debug print for this iteration

        Z_u = callback_data['Z_u']
        u_base = callback_data['u_base']
        u_control = callback_data['u_control']
        amp_val = callback_data['amp']
        deim_ops = callback_data['deim_ops']

        # u_rom_coeffs = current_solution.dat.data_ro[:N_u].flatten()
        # u_sub, p_sub = current_solution.subfunctions
        # u_rom_coeffs = u_sub.dat.data_ro.flatten()
        coeffs = current_solution.array_r
        u_rom_coeffs = coeffs[:N_u]
          # Reconstruct total velocity
        u_total_np = u_base + amp_val * u_control + Z_u @ u_rom_coeffs
        callback_data['u_full_total'].dat.data[:] = u_total_np.reshape(
            callback_data['u_full_total'].dat.data.shape
        )

        # Assemble Jacobian
        u_total = callback_data['u_full_total']
        u_trial = TrialFunction(V_full)
        v_test = TestFunction(V_full)
        J_form = (
            inner(grad(u_trial) * u_total, v_test) * dx
            + inner(grad(u_total) * u_trial, v_test) * dx
        )
        J_petsc = assemble(J_form).M.handle
        J_i, J_j, J_data = J_petsc.getValuesCSR()
        J_sci = sp.csr_matrix((J_data, J_j, J_i), shape=J_petsc.getSize())
        # MDEIM approximation (pass full CSR data)
        J_reduced = deim_ops.reduced_jacobian(J_data)
        callback_data['conv_J_const'].assign(J_reduced)
        # J_anal = Z_u.T @ J_sci@ Z_u
        # callback_data['conv_J_const'].assign(J_anal)

        if callback_data['iteration_count'] % 1 == 0:
            print(f"      Iteration {callback_data['iteration_count']}: "
                  f"||J_c|| = {np.linalg.norm(J_reduced):.6e}")

    def debug_callback(current_solution, residual):
        """Compare exact vs DEIM reduced terms."""
        if not callback_data['debug']:
            return

        # Only print once per Newton iteration (after Jacobian update)
        if not callback_data.get('print_debug', False):
            return
        callback_data['print_debug'] = False  # Reset flag

        Z_u = callback_data['Z_u']
        u_base = callback_data['u_base']
        u_control = callback_data['u_control']
        amp_val = callback_data['amp']
        nu_val = callback_data['nu']
        ops = callback_data['operators']
        deim = callback_data['deim_ops']
        V_full = callback_data['V_full']
        N_u = callback_data['N_u']

        # Get current ROM coefficients
        coeffs = current_solution.array_r
        a = coeffs[:N_u]

        # Reconstruct velocity
        u_total_np = u_base + amp_val * u_control + Z_u @ a
        u_func = callback_data['u_full_total']
        u_func.dat.data[:] = u_total_np.reshape(u_func.dat.data.shape)

        # Assemble full convective term
        v_test = TestFunction(V_full)
        F_form = inner(grad(u_func) * u_func, v_test) * dx
        F_full = assemble(F_form).dat.data_ro.flatten()

        # Exact vs DEIM residual
        F_r_exact = Z_u.T @  F_full
        F_r_deim = deim.reduced_convective(F_full)

        # Assemble Jacobian
        u_trial = TrialFunction(V_full)
        J_form = (
            inner(grad(u_trial) * u_func, v_test) * dx
            + inner(grad(u_func) * u_trial, v_test) * dx
        )
        J_petsc = assemble(J_form).M.handle
        indptr, indices, J_data = J_petsc.getValuesCSR()

        # Exact vs MDEIM Jacobian
        J_scipy = sp.csr_matrix((J_data, indices, indptr), shape=J_petsc.getSize())
        J_r_exact = Z_u.T @ (J_scipy @ Z_u)
        J_r_mdeim = deim.reduced_jacobian(J_data)

        # Full ROM residual (momentum only, ignoring pressure for simplicity)
        f_adj = ops.get_adjusted_forcing(nu_val, amp_val)
        res_exact = nu_val * ops.A_N @ a + F_r_exact - f_adj
        res_deim = nu_val * ops.A_N @ a + F_r_deim - f_adj

        print(f"      [DEBUG] ||F_r exact - F_r DEIM|| / ||F_r exact|| = {np.linalg.norm(F_r_exact - F_r_deim) / np.linalg.norm(F_r_exact):.2e}")
        print(f"      [DEBUG] ||J_r exact - J_r MDEIM|| / ||J_r exact|| = {np.linalg.norm(J_r_exact - J_r_mdeim) / np.linalg.norm(J_r_exact):.2e}")
        print(f"      [DEBUG] ||ROM res (exact)|| = {np.linalg.norm(res_exact):.6e}")
        print(f"      [DEBUG] ||ROM res (DEIM) || = {np.linalg.norm(res_deim):.6e}")
        print(f"      [DEBUG] ||F_r exact|| = {np.linalg.norm(F_r_exact)}")
        print(f"      [DEBUG] ||F_r DEIM|| = {np.linalg.norm(F_r_deim)}")
        print(f"      [DEBUG]  ERROR before = {callback_data['error_res']}")

    # =========================================================================
    # VARIATIONAL FORMS
    # =========================================================================

    F  = nu * inner(dot(A_const, u_rom), v_rom) * dx
    F += inner(conv_res_const, v_rom) * dx
    F += inner(dot(B_const.T, p_rom), v_rom) * dx
    F -= inner(f_const, v_rom) * dx
    F += inner(dot(B_const, u_rom), q_rom) * dx
    F -= inner(g_const, q_rom) * dx

    J  = nu * inner(dot(A_const, du_rom), v_rom) * dx
    J += inner(dot(conv_J_const, du_rom), v_rom) * dx
    J += inner(dot(B_const.T, dp_rom), v_rom) * dx
    J += inner(dot(B_const, du_rom), q_rom) * dx

    # =========================================================================
    # SOLVER
    # =========================================================================

    if solver_parameters is None:
        solver_parameters = {
            'snes_type': 'newtonls',
            'snes_linesearch_type': 'bt',
            'snes_monitor': None,
            'snes_converged_reason': None,
            'snes_max_it': 100,
            'snes_rtol': 1e-7,
            'snes_atol': 1e-7,
            'snes_stol': 1e-8,
            'ksp_type': 'preonly',
            'pc_type': 'lu',
            'pc_factor_mat_solver_type': 'mumps',
            # 'mat_type': 'aij',
            # 'snes_view': None,              # Print SNES config
            # 'ksp_monitor': None,            # Monitor linear solve
            # 'ksp_converged_reason': None,   # Why KSP stopped
            # 'snes_linesearch_monitor': None,  # Monitor line search
        }

    rom_problem = NonlinearVariationalProblem(F, w_rom, J=J)
    solver = NonlinearVariationalSolver(
        rom_problem,
        solver_parameters=solver_parameters,
        pre_function_callback=update_convection_residual,
        pre_jacobian_callback=update_jacobian_matrices,
        post_function_callback=debug_callback,
    )

    print("    ✓ ROM solver with DEIM created")
    return solver, w_rom, callback_data


def solve_rom_deim(
    operators: ReducedOperators,
    nu: float,
    amp: float,
    problem,
    deim_ops: DEIMOnlineOperators,
    initial_guess: np.ndarray = None,
    solver_parameters: dict = None,
    store_full_dofs: bool = True,
    return_callback_data: bool = False
) -> ROMSolution:
    """Solve ROM using DEIM for nonlinear terms."""
    print("=" * 60)
    print("SOLVING ROM WITH DEIM")
    print("=" * 60)
    print(f"  nu = {nu} (Re = {1/nu:.1f})")
    print(f"  amp = {amp}")

    solver, w_rom, callback_data = setup_rom_solver_deim(
        operators, nu, amp, problem, deim_ops, solver_parameters,debug = False
    )

    u_rom, p_rom = w_rom.subfunctions

    if initial_guess is not None:
        u_rom.dat.data[:] = initial_guess.reshape(u_rom.dat.data.shape)
        print(f"  Using provided initial guess")

    callback_data['iteration_count'] = 0
    converged = True
    residual_norm = 0.0
    # print(f' The initial sol is : {u_rom.dat.data[:][:10]}')
    try:
        solver.solve()
        iterations = callback_data['iteration_count']
    except Exception as e:
        iterations = callback_data['iteration_count']
        print("\n🔥 Exception during SNES solve:")
        traceback.print_exc()
        raise
    # try:
    #     solver.solve()
    #     iterations = callback_data['iteration_count']
    #     print(f"\n  ✓ Converged in {iterations} iterations")
    # except Exception as e:
    #     converged = False
    #     iterations = callback_data['iteration_count']
    #     print(f"\n  ✗ Solver failed after {iterations} iterations: {e}")

    N_u = operators.n_velocity_modes

    velocity_coeffs = u_rom.dat.data_ro.flatten().copy()
    pressure_coeffs = p_rom.dat.data_ro.flatten().copy()

    velocity_dofs = None
    pressure_dofs = None

    if store_full_dofs:
        velocity_dofs = (
            operators.u_base
            + amp * operators.u_control
            + operators.Z_u @ velocity_coeffs
        )
        pressure_dofs = operators.Z_p @ pressure_coeffs
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
            'n_velocity_modes': operators.n_velocity_modes,
            'n_pressure_modes': operators.n_pressure_modes,
            'deim_modes': deim_ops.m_F,
            'mdeim_modes': deim_ops.m_J,
        }
    )

    print(f"\n  ||u_rom|| = {np.linalg.norm(velocity_coeffs):.6e}")
    print(f"  ||p_rom|| = {np.linalg.norm(pressure_coeffs):.6e}")

    if return_callback_data:
        return solution, callback_data
    return solution


def solve_rom_parameter_sweep_deim(
    operators: ReducedOperators,
    problem,
    deim_ops: DEIMOnlineOperators,
    reynolds_values: np.ndarray = None,
    amplitude_values: np.ndarray = None,
    use_continuation: bool = True,
    store_full_dofs: bool = False,
    solver_parameters: dict = None
):
    """Solve ROM with DEIM for multiple parameter values."""
    if reynolds_values is None:
        reynolds_values = np.array([100.0])
    if amplitude_values is None:
        amplitude_values = np.array([0.0])

    print("=" * 60)
    print("ROM PARAMETER SWEEP WITH DEIM")
    print("=" * 60)
    print(f"  Reynolds: {reynolds_values}")
    print(f"  Amplitude: {amplitude_values}")
    print(f"  Total solves: {len(reynolds_values) * len(amplitude_values)}")
    print(f"  DEIM modes: {deim_ops.m_F}")
    print(f"  MDEIM modes: {deim_ops.m_J}")

    solutions = []
    prev_guess = None

    for amp in amplitude_values:
        for Re in reynolds_values:
            nu = 1.0 / Re
            print(f"\n--- Re = {Re:.1f}, amp = {amp:.3f} ---")

            solution = solve_rom_deim(
                operators=operators,
                nu=nu,
                amp=amp,
                problem=problem,
                deim_ops=deim_ops,
                initial_guess=prev_guess,
                solver_parameters=solver_parameters,
                store_full_dofs=store_full_dofs
            )

            solutions.append(solution)

            if use_continuation:
                prev_guess = solution.velocity_coeffs

    print("\n" + "=" * 60)
    print("PARAMETER SWEEP COMPLETE")
    print("=" * 60)
    print(f"  Total solutions: {len(solutions)}")
    print(f"  Converged: {sum(s.converged for s in solutions)}/{len(solutions)}")

    return solutions


# Reconstruction functions unchanged
def reconstruct_velocity(solution: ROMSolution, operators: ReducedOperators, problem, amp: float = None) -> Function:
    if amp is None:
        amp = solution.amplitude

    if solution.velocity_dofs is not None:
        velocity_dofs = solution.velocity_dofs
    else:
        velocity_dofs = operators.u_base + amp * operators.u_control + operators.Z_u @ solution.velocity_coeffs

    u_full = Function(problem.velocity_space, name="Velocity")
    u_full.dat.data[:] = velocity_dofs.reshape(u_full.dat.data.shape)
    return u_full


def reconstruct_pressure(solution: ROMSolution, operators: ReducedOperators, problem) -> Function:
    if solution.pressure_dofs is not None:
        pressure_dofs = solution.pressure_dofs
    else:
        pressure_dofs = operators.Z_p @ solution.pressure_coeffs

    p_full = Function(problem.pressure_space, name="Pressure")
    p_full.dat.data[:] = pressure_dofs.reshape(p_full.dat.data.shape)
    return p_full


def reconstruct_solution(solution: ROMSolution, operators: ReducedOperators, problem, amp: float = None):
    return reconstruct_velocity(solution, operators, problem, amp), reconstruct_pressure(solution, operators, problem)

def compute_rom_error(u_fom, u_rom, M_u):
    """Compute error in the same norm used for POD."""
    u_fom_dofs = u_fom.dat.data_ro.flatten()
    u_rom_dofs = u_rom.dat.data_ro.flatten()

    error = u_fom_dofs - u_rom_dofs

    error_norm = np.sqrt(error.T @ (M_u @ error))
    fom_norm = np.sqrt(u_fom_dofs.T @ (M_u @ u_fom_dofs))

    return error_norm / fom_norm