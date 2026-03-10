# reduction.py
import firedrake
from firedrake import *
import numpy as np
import scipy.sparse as sp

def full_order_operators(problem):
    """Assemble full-order A, B, f, g matrices"""
    V = problem.velocity_space
    Q = problem.pressure_space

    print("\n  Assembling full-order operators...")

    u_trial = TrialFunction(V)
    v_test = TestFunction(V)
    p_trial = TrialFunction(Q)
    q_test = TestFunction(Q)

    # Stiffness matrix A
    A_form = inner(grad(u_trial), grad(v_test)) * dx
    A = assemble(A_form)

    # Divergence matrix B
    B_form = -q_test * div(u_trial) * dx
    B = assemble(B_form)

    # Force vectors
    f_form = inner(Constant((0.0, 0.0)), v_test) * dx
    f = assemble(f_form)

    g_form = Constant(0.0) * q_test * dx
    g = assemble(g_form)

    print("  ✓ Full-order operators assembled")

    return A, B, f, g


def build_reduced_operators_with_lifting(problem, Z_enriched, Z_p, lifting):
    """
    Build reduced operators accounting for lifting functions

    Args:
        problem: FOM problem object
        Z_enriched: Reduced basis (from homogenized snapshots)
        Z_p: Pressure basis
        lifting: LiftingFunctions object with u_base, u_control

    Returns:
        operators: Dictionary with all ROM operators including lifting contributions
    """
    # Get standard operators
    A, B, f, g = full_order_operators(problem)

    print("\nBuilding reduced operators with lifting functions...")
    N_vel = Z_enriched.shape[1]
    N_p = Z_p.shape[1]

    print(f"  Reduced velocity DOFs: {N_vel}")
    print(f"  Reduced pressure DOFs: {N_p}")

    # Convert to scipy sparse
    A_petsc = A.M.handle
    B_petsc = B.M.handle

    indptr_A, indices_A, data_A = A_petsc.getValuesCSR()
    A_scipy = sp.csr_matrix((data_A, indices_A, indptr_A),
                             shape=A_petsc.getSize())

    indptr_B, indices_B, data_B = B_petsc.getValuesCSR()
    B_scipy = sp.csr_matrix((data_B, indices_B, indptr_B),
                             shape=B_petsc.getSize())

    # Get vectors
    f_np = f.dat.data.flatten()
    g_np = g.dat.data[:]

    print("  Projecting to reduced space...")

    # Standard reduced operators
    A_N = Z_enriched.T @ (A_scipy @ Z_enriched)
    B_N = Z_p.T @ (B_scipy @ Z_enriched)
    f_N = Z_enriched.T @ f_np
    g_N = Z_p.T @ g_np

    # Lifting function contributions
    u_base_np = lifting.u_base.dat.data[:].flatten()
    u_control_np = lifting.u_control.dat.data[:].flatten()

    # Compute A_full @ u_base and A_full @ u_control
    A_u_base = A_scipy @ u_base_np
    A_u_control = A_scipy @ u_control_np

    # Project to reduced space
    A_N_u_base = Z_enriched.T @ A_u_base
    A_N_u_control = Z_enriched.T @ A_u_control

    # Divergence of lifting functions
    B_u_base = B_scipy @ u_base_np
    B_u_control = B_scipy @ u_control_np

    B_N_u_base = Z_p.T @ B_u_base
    B_N_u_control = Z_p.T @ B_u_control

    print(f"  A_N shape: {A_N.shape}")
    print(f"  B_N shape: {B_N.shape}")
    print(f"  Lifting contributions computed")

    operators = {
        'A_N': A_N,
        'B_N': B_N,
        'f_N': f_N,
        'g_N': g_N,
        'Z_enriched': Z_enriched,
        'Z_p': Z_p,
        # Lifting function data
        'u_base': u_base_np,
        'u_control': u_control_np,
        'A_N_u_base': A_N_u_base,
        'A_N_u_control': A_N_u_control,
        'B_N_u_base': B_N_u_base,
        'B_N_u_control': B_N_u_control
    }

    print("  ✓ Reduced operators with lifting built")

    return operators


def setup_rom_problem_with_lifting(operators, nu, problem, amp, rom_mesh=None):
    """
    Set up ROM with lifting functions and convection

    Args:
        operators: Dictionary from build_reduced_operators_with_lifting
        nu: Viscosity
        problem: FOM problem object
        amp: Amplitude parameter for control lifting function
        rom_mesh: Optional ROM mesh

    Returns:
        solver: NonlinearVariationalSolver
        w_rom: Solution Function
        callback_data: Dictionary with callback state
    """
    # Extract operators
    A_N = operators['A_N']
    B_N = operators['B_N']
    f_N = operators['f_N']
    g_N = operators['g_N']
    Z_enriched = operators['Z_enriched']
    Z_p = operators['Z_p']

    # Lifting function data
    u_base = operators['u_base']
    u_control = operators['u_control']
    A_N_u_base = operators['A_N_u_base']
    A_N_u_control = operators['A_N_u_control']
    B_N_u_base = operators['B_N_u_base']
    B_N_u_control = operators['B_N_u_control']

    N_vel = A_N.shape[0]
    N_p = B_N.shape[0]

    print(f"\n  Setting up ROM with lifting (amp={amp}, N_vel={N_vel}, N_p={N_p})...")

    # Create ROM mesh
    if rom_mesh is None:
        rom_mesh = UnitIntervalMesh(1)

    # ROM function spaces
    V_rom = VectorFunctionSpace(rom_mesh, "DG", 0, dim=N_vel)
    Q_rom = VectorFunctionSpace(rom_mesh, "DG", 0, dim=N_p)
    W_rom = V_rom * Q_rom

    # Solution function
    w_rom = Function(W_rom)
    u_rom, p_rom = split(w_rom)

    #Trial Functions for Jacobian
    du_rom, dp_rom = TrialFunctions(W_rom)


    # Test functions
    v_rom, q_rom = TestFunctions(W_rom)

    # Convert operators to Constants
    A_const = Constant(A_N, rom_mesh)
    B_const = Constant(B_N, rom_mesh)

    # Adjusted forcing term
    f_adjusted = f_N - nu * (A_N_u_base + amp * A_N_u_control)
    f_const = Constant(f_adjusted, rom_mesh)

    # Adjusted divergence RHS
    g_adjusted = g_N - (B_N_u_base + amp * B_N_u_control)
    g_const = Constant(g_adjusted, rom_mesh)

    # Convection matrix (updated by callback)
    N_N = np.zeros((N_vel, N_vel))
    N_const = Constant(N_N, rom_mesh)

    # M_N for Jacobian correction
    M_N = np.zeros((N_vel, N_vel))
    M_const = Constant(M_N, rom_mesh)

    # Full-order space
    V_full = problem.velocity_space
    DC_N = np.zeros((N_vel, N_vel))
    DC_const = Constant(DC_N, rom_mesh)


    # Storage for callback
    callback_data = {
        'Z_enriched': Z_enriched,
        'V_full': V_full,
        'N_const': N_const,
        'DC_const': DC_const,
        'M_const': M_const,
        'N_vel': N_vel,
        'N_p': N_p,
        'u_full_total': Function(V_full, name="u_total"),
        'u_full_reduced': Function(V_full, name="u_reduced"),
        'u_base': u_base,
        'u_control': u_control,
        'amp': amp,
        'iteration_count': 0,
        'A_N': A_N,                   # <-- ADD for debugging
        'B_N': B_N,                   # <-- ADD for debugging
        'nu': nu,                     # <-- ADD for debugging
    }
# Create separate callbacks for residual and Jacobian
    def update_convection_residual(current_solution):
        """Update convection matrix for residual evaluation"""
        u_rom_coeffs = current_solution.array_r[:N_vel]
        u_reduced_part = Z_enriched @ u_rom_coeffs
        u_total = u_base + amp * u_control + u_reduced_part

        callback_data['u_full_total'].dat.data[:] = u_total.reshape(
            callback_data['u_full_total'].dat.data.shape
        )

        # Assemble convection matrix
        u_trial = TrialFunction(V_full)
        v_test = TestFunction(V_full)
        N_form = inner(dot(grad(u_trial), callback_data['u_full_total']), v_test) * dx
        N_full = assemble(N_form)

        # Convert and project
        N_petsc = N_full.M.handle
        indptr, indices, data = N_petsc.getValuesCSR()
        N_scipy = sp.csr_matrix((data, indices, indptr), shape=N_petsc.getSize())
        N_N = Z_enriched.T @ (N_scipy @ Z_enriched)

        callback_data['N_const'].assign(N_N)

    def update_jacobian_matrices(current_solution):
        """Update N_N and M_N for Jacobian"""
        callback_data['iteration_count'] += 1

        # Get current ROM coefficients
        u_rom_coeffs = current_solution.array_r[:N_vel]

        # Reconstruct full-order velocities
        u_reduced_np = Z_enriched @ u_rom_coeffs  # Just Z @ u_r (without lifting)
        u_total_np = u_base + amp * u_control + u_reduced_np

        # Update Functions
        callback_data['u_full_total'].dat.data[:] = u_total_np.reshape(
            callback_data['u_full_total'].dat.data.shape
        )
        callback_data['u_full_reduced'].dat.data[:] = u_reduced_np.reshape(
            callback_data['u_full_reduced'].dat.data.shape
        )

        u_total = callback_data['u_full_total']
        u_reduced = callback_data['u_full_reduced']

        V_full = callback_data['V_full']
        u_trial = TrialFunction(V_full)
        v_test = TestFunction(V_full)

        # ------------------------------------------------------------------
        # N_full: (u_total · ∇) u_trial  -->  advection BY u_total
        # # ------------------------------------------------------------------
        N_form = inner(dot(grad(u_trial), u_total), v_test) * dx
        N_full = assemble(N_form)

        N_petsc = N_full.M.handle
        indptr_N, indices_N, data_N = N_petsc.getValuesCSR()
        N_scipy = sp.csr_matrix((data_N, indices_N, indptr_N),
                                shape=N_petsc.getSize())
        N_N = Z_enriched.T @ (N_scipy @ Z_enriched)

        # ------------------------------------------------------------------
        # M_full: (u_trial · ∇) u_total  -->  u_trial advects u_total
        # ------------------------------------------------------------------
        M_form = inner(dot(grad(u_total), u_trial), v_test) * dx
        M_full = assemble(M_form)

        M_petsc = M_full.M.handle
        indptr_M, indices_M, data_M = M_petsc.getValuesCSR()
        M_scipy = sp.csr_matrix((data_M, indices_M, indptr_M),
                                shape=M_petsc.getSize())
        M_N = Z_enriched.T @ (M_scipy @ Z_enriched)

        # ------------------------------------------------------------------
        # Total convection Jacobian: N_N + M_N
        # ------------------------------------------------------------------
        # J_conv = N_N + M_N

        callback_data['N_const'].assign(N_N)
        callback_data['M_const'].assign(M_N)

        # c_form = inner(dot(grad(u_total), u_total), v_test) * dx
        # c_assembled = assemble(c_form)
        # c_np = c_assembled.dat.data[:].flatten()

        # # Project to reduced space: c_N = Z^T @ c
        # c_N = Z_enriched.T @ c_np
        # dc_form = derivative(c_form, u_total, u_trial)
        # DC_full = assemble(dc_form)

        # DC_petsc = DC_full.M.handle
        # indptr_dc, indices_dc, data_dc = DC_petsc.getValuesCSR()
        # DC_scipy = sp.csr_matrix((data_dc, indices_dc, indptr_dc), shape=DC_petsc.getSize())
        # DC_N_new = Z_enriched.T @ (DC_scipy @ Z_enriched)  # <-- Z^T @ DC @ Z

        # Update Constants
        # callback_data['N_const'].assign(N_N_new)
        # callback_data['DC_const'].assign(DC_N_new)

        # print(f"    Iter {callback_data['iteration_count']}: "
        #     #   f"||N_N|| = {np.linalg.norm(N_N_new):.6e}, "
        #       f"||DC_N|| = {np.linalg.norm(DC_N_new):.6e}")

        if callback_data['iteration_count'] % 1 == 0:
            print(f"    Iteration {callback_data['iteration_count']}: "
                f"||N_N|| = {np.linalg.norm(N_N):.6e}, "
                f"||M_N|| = {np.linalg.norm(M_N):.6e}")

    # Build residual form
    F = nu * inner(dot(A_const, u_rom), v_rom) * dx
    F += inner(dot(N_const, u_rom), v_rom) * dx
    F += inner(dot(B_const.T, p_rom), v_rom) * dx
    F -= inner(f_const, v_rom) * dx
    F += inner(dot(B_const, u_rom), q_rom) * dx
    F -= inner(g_const, q_rom) * dx

    # Build Jacobian form (including convection contribution)
    J = nu * inner(dot(A_const, du_rom), v_rom) * dx
    J += inner(dot(N_const, du_rom), v_rom) * dx
    J += inner(dot(M_const, du_rom), v_rom) * dx
    # J += inner(dot(DC_const, du_rom), v_rom) * dx
    J += inner(dot(B_const.T, dp_rom), v_rom) * dx
    J += inner(dot(B_const, du_rom), q_rom) * dx
    eps_p = Constant(1e-10)

    # J += eps_p * inner(dp_rom, q_rom) * dx
    # F += eps_p * inner(p_rom, q_rom) * dx
    # Create problem with explicit Jacobian
    problem_nl = NonlinearVariationalProblem(F, w_rom, J=J)
    solver_parameters = {
        'snes_type': 'newtonls',
        'snes_linesearch_type': 'l2',
        'snes_monitor': None,
        'snes_converged_reason': None,
        'snes_max_it': 100,
        'snes_rtol': 1e-6,
        'snes_atol': 1e-7,
        'snes_stol': 1e-8,
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        'pc_factor_mat_solver_type': 'mumps',
        'mat_type': 'aij'
        # 'ksp_monitor': None,           # <-- ADD: see linear solver
        # 'ksp_converged_reason': None,  # <-- ADD: see why it failed
        # 'ksp_monitor_true_residual': None,
        # 'ksp_view': None
    }
    # solver_parameters = {
    #     'snes_type': 'newtonls',
    #     'snes_monitor': None,
    #     'snes_max_it': 50,
    #     'snes_rtol': 1e-8,
    #     'snes_atol': 1e-10,
    #     'ksp_type': 'preonly',
    #     'pc_type': 'lu',
    #     'mat_type': 'aij'
    # }

    solver = NonlinearVariationalSolver(
        problem_nl,
        solver_parameters=solver_parameters,
        pre_function_callback=update_convection_residual,      # ← For residual
        pre_jacobian_callback=update_jacobian_matrices    # ← For Jacobian
    )

    print("  ✓ ROM solver with correct Jacobian created")


    return solver, w_rom, callback_data



def solve_rom_with_lifting(operators, nu, problem, amp, initial_guess=None):
    """
    Complete workflow to solve ROM

    Args:
        operators: Dictionary from build_reduced_operators_with_lifting
        nu: Viscosity
        problem: FOM problem object
        amp: Amplitude parameter
        initial_guess: Optional initial guess for ROM coefficients

    Returns:
        w_rom: Solution Function with ROM coefficients
        callback_data: Dictionary with solve information
    """
    solver, w_rom, callback_data = setup_rom_problem_with_lifting(
        operators, nu, problem, amp
    )

    if initial_guess is not None:
        w_rom.dat.data[:] = initial_guess

    print("\n  Solving ROM with lifting functions...")
    callback_data['iteration_count'] = 0

    try:
        solver.solve()
        print(f"  ✓ Converged in {callback_data['iteration_count']} iterations")
    except Exception as e:
        print(f"  ✗ Solver failed: {e}")
        raise

    return w_rom, callback_data


def reconstruct_full_solution_with_lifting(w_rom, operators, problem, amp):
    """
    Reconstruct full solution including lifting functions

    u_full = u_base + amp * u_control + Z_enriched @ u_rom
    p_full = Z_p @ p_rom

    Args:
        w_rom: ROM solution Function
        operators: Dictionary with basis and lifting data
        problem: FOM problem object
        amp: Amplitude parameter

    Returns:
        u_full: Full-order velocity Function
        p_full: Full-order pressure Function
    """
    Z_enriched = operators['Z_enriched']
    Z_p = operators['Z_p']
    u_base = operators['u_base']
    u_control = operators['u_control']
    N_vel = Z_enriched.shape[1]
    u_rom, p_rom = w_rom.subfunctions
    u_rom_coeffs = u_rom.dat.data.flatten()
    p_rom_coeffs = p_rom.dat.data.flatten()

    # # Extract ROM coefficients
    # u_rom_coeffs = w_rom.dat.data[:N_vel]
    # p_rom_coeffs = w_rom.dat.data[N_vel:]

    # Create full-order Functions
    u_full = Function(problem.velocity_space, name="Velocity")
    p_full = Function(problem.pressure_space, name="Pressure")

    # Reconstruct with lifting
    u_full.dat.data[:] = (u_base + amp * u_control + Z_enriched @ u_rom_coeffs).reshape(
            u_full.dat.data.shape
        )
    p_full.dat.data[:] = Z_p @ p_rom_coeffs

    return u_full, p_full
