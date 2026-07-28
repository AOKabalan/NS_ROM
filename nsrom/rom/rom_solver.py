"""
Unified ROM Solver.

Single entry point for solving the reduced Navier-Stokes system.
The convection-evaluation strategy is chosen explicitly via `mode`:

    'exact'  : Z_u^T F(u)      -- full-mesh assembly, no hyper-reduction
    'deim'   : deim(F(u))      -- submesh assembly    (requires deim_ops)
    'tensor' : tensor(alpha)   -- no assembly, exact  (requires quadratic_only
                                  and a precomputed operators.tensor_convection)

Usage:
    solution = solve_rom(operators, nu, amp, problem, mode='exact')

    solution = solve_rom(operators, nu, amp, problem,
                         mode='deim', deim_ops=my_deim)

    solution = solve_rom(operators, nu, amp, problem, mode='tensor')

    # With Firedrake internals (e.g. for bifurcation indicator):
    solution, w_rom, callback_data = solve_rom(
        operators, nu, amp, problem, mode='tensor', return_internals=True,
    )

`mode` is a required keyword argument: a forgotten mode raises TypeError at
the call site rather than silently falling back to a different solver path.
Quadratic-only assembly is still auto-detected from
operators.has_affine_convection.
"""

import numpy as np

from firedrake import (
    Function, TestFunctions, TrialFunctions,
    split, inner, dot, dx,
    Constant, UnitIntervalMesh, VectorFunctionSpace,
    NonlinearVariationalProblem, NonlinearVariationalSolver,
)
from firedrake.exceptions import ConvergenceError

from ..navier_stokes import NavierStokesSolution
from .data_structures import ReducedOperators, ROMSolution
from .rom_callbacks import make_callbacks


__all__ = [
    'solve_rom',
    'solve_rom_with_internals',
    'solve_rom_parameter_sweep',
    'reconstruct_velocity',
    'reconstruct_pressure',
    'reconstruct_solution',
    'project_to_rom_coefficients',
    'compute_rom_error',
    'VALID_MODES',
]


VALID_MODES = ('exact', 'deim', 'tensor')


# Default PETSc SNES parameters for the ROM Newton system.
# Tight tolerances matter for bifurcation/continuation work where the
# initial residual at a continuation step can already be small.
_DEFAULT_SOLVER_PARAMETERS = {
    'snes_type': 'newtonls',
    'snes_linesearch_type': 'basic',
    'snes_max_it': 350,
    'snes_rtol': 1e-7,
    'snes_atol': 1e-8,
    'snes_stol': 1e-12,
    'ksp_type': 'preonly',
    'pc_type': 'lu',
    'pc_factor_mat_solver_type': 'mumps',
    'mat_type': 'aij',
}


_PETSC_OUTPUT_OPTIONS = {
    'snes_monitor',
    'snes_converged_reason',
    'snes_linesearch_monitor',
    'ksp_monitor',
    'ksp_monitor_true_residual',
    'ksp_converged_reason',
}


_MODE_LABELS = {
    'exact': 'Exact',
    'deim': 'DEIM',
    'tensor': 'Tensor(exact)',
}


def _prepare_solver_parameters(solver_parameters=None, verbose=True):
    """
    Return PETSc solver parameters consistent with the Python verbose flag.

    Important: PETSc options such as 'snes_monitor': None enable output.
    They must be absent, not set to None, when silence is desired.
    """
    if solver_parameters is None:
        params = dict(_DEFAULT_SOLVER_PARAMETERS)
    else:
        params = dict(solver_parameters)

    if verbose:
        params.setdefault('snes_monitor', None)
        params.setdefault('snes_converged_reason', None)
    else:
        for key in list(params):
            if key in _PETSC_OUTPUT_OPTIONS or key.endswith('_monitor'):
                params.pop(key, None)

    return params


# =============================================================================
# MODE RESOLUTION
# =============================================================================

def _resolve_mode(operators, mode, deim_ops=None):
    """
    Validate the requested convection mode and return the handles it needs.

        'exact'  : Z_u^T F(u), full-mesh assembly
        'deim'   : DEIM/MDEIM, submesh assembly     (requires deim_ops)
        'tensor' : precomputed exact tensor         (requires quadratic_only)

    Returns
    -------
    (mode, tensor_conv, deim_ops)
        Handles not used by the selected mode are returned as None, so a
        stray `deim_ops` passed alongside mode='tensor' cannot leak into
        callback_data or the solution metadata.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"Unknown mode {mode!r}; expected one of {VALID_MODES}.")

    if mode == 'tensor':
        if not operators.has_affine_convection:
            raise ValueError(
                "mode='tensor' requires operators.has_affine_convection=True "
                "(the tensor represents the pure quadratic term C(u',u') only)."
            )
        tensor_conv = getattr(operators, 'tensor_convection', None)
        if tensor_conv is None:
            raise ValueError(
                "mode='tensor' but operators.tensor_convection is None. "
                "Precompute and attach the convection tensor offline first."
            )
        return 'tensor', tensor_conv, None

    if mode == 'deim':
        if deim_ops is None:
            raise ValueError("mode='deim' requires deim_ops=...")
        return 'deim', None, deim_ops

    return 'exact', None, None


# =============================================================================
# SOLVER SETUP
# =============================================================================

def _setup_rom_solver(
    operators: ReducedOperators,
    nu: float,
    amp: float,
    problem,
    *,
    mode: str,
    deim_ops=None,
    solver_parameters: dict = None,
    submesh_deim=None,
    verbose: bool = True,
):
    """
    Build the ROM NonlinearVariationalSolver (internal).

    `mode` selects the convection path ('exact' | 'deim' | 'tensor').
    Quadratic-only assembly is auto-detected from
    operators.has_affine_convection.

    Returns
    -------
    solver, w_rom, callback_data
    """
    N_u = operators.n_velocity_modes
    N_p = operators.n_pressure_modes
    quadratic_only = operators.has_affine_convection
    thetas = operators._default_thetas(amp)

    mode, tensor_conv, deim_ops = _resolve_mode(operators, mode, deim_ops)
    use_tensor = (mode == 'tensor')
    use_deim = (mode == 'deim')
    if not use_deim:
        submesh_deim = None      # never assembled in tensor/exact mode

    if verbose:
        conv_str = "quadratic-only" if quadratic_only else "full convection"
        print(f"\n  Setting up ROM solver...")
        print(f"    Mode: {_MODE_LABELS[mode]}, {conv_str}")
        print(f"    N_velocity: {N_u}")
        print(f"    N_pressure: {N_p}")
        print(f"    nu: {nu} (Re = {1/nu:.1f})")
        print(f"    amp: {amp}")
        print(f"    n_thetas: {len(thetas)}")
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

    # ---- Submesh for DEIM (one-time setup) ----
    if use_deim and submesh_deim is None:
        from .hyper_reduction import SubMeshDEIM
        submesh_deim = SubMeshDEIM(problem.velocity_space, deim_ops)
    elif use_deim and verbose:
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
        # Mode: 'mode' is the resolved public value; use_tensor/use_deim are
        # derived internals that make_callbacks branches on.
        'mode': mode,
        'use_tensor': use_tensor,
        'use_deim': use_deim,
        'quadratic_only': quadratic_only,
        # Lifting (for full mode)
        'thetas': list(thetas),
        'lifting_dofs': operators.lifting_dofs,
        # Convection handles (only the one for the active mode is non-None)
        'deim_ops': deim_ops,
        'submesh_deim': submesh_deim,
        'tensor_convection': tensor_conv,
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
        # For bifurcation indicator and post-solve work
        'rom_mesh': rom_mesh,
        'W_rom': W_rom,
        'B_const': B_const,
    }

    # ---- Callbacks ----
    res_cb, jac_cb = make_callbacks(callback_data)

    # ---- Variational forms ----
    # Residual
    F = inner(dot(A_eff_const, u_rom), v_rom) * dx       # diffusion + linear convection
    F += inner(conv_res_const, v_rom) * dx                # nonlinear convection (callback)
    F += inner(dot(B_const.T, p_rom), v_rom) * dx         # pressure gradient
    F -= inner(f_eff_const, v_rom) * dx                   # forcing + lifting
    F += inner(dot(B_const, u_rom), q_rom) * dx           # divergence
    F -= inner(g_eff_const, q_rom) * dx                   # divergence RHS

    # Jacobian
    J = inner(dot(A_eff_const, du_rom), v_rom) * dx       # diffusion + linear convection
    J += inner(dot(conv_J_const, du_rom), v_rom) * dx     # nonlinear convection (callback)
    J += inner(dot(B_const.T, dp_rom), v_rom) * dx        # pressure gradient
    J += inner(dot(B_const, du_rom), q_rom) * dx          # divergence

    # ---- PETSc solver ----
    solver_parameters = _prepare_solver_parameters(
        solver_parameters=solver_parameters,
        verbose=verbose,
    )

    rom_problem = NonlinearVariationalProblem(F, w_rom, J=J)
    solver = NonlinearVariationalSolver(
        rom_problem,
        solver_parameters=solver_parameters,
        pre_function_callback=res_cb,
        pre_jacobian_callback=jac_cb,
    )

    if verbose:
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

    callback_data['nu'] = nu_new
    callback_data['amp'] = amp_new
    callback_data['thetas'] = list(thetas_new)
    callback_data['iteration_count'] = 0

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
# SOLVE HELPERS (shared by all entry points)
# =============================================================================

def _apply_initial_guess(w_rom, u_initial_guess, p_initial_guess, verbose=True):
    """
    Write initial guess into the mixed Function. Raises if exactly one of
    velocity/pressure is provided.

    Returns True if a guess was applied, False otherwise.
    """
    u_provided = u_initial_guess is not None
    p_provided = p_initial_guess is not None

    if u_provided != p_provided:
        raise ValueError(
            "Initial guess must be provided for both velocity and pressure, "
            "or for neither (got velocity={}, pressure={}).".format(
                "yes" if u_provided else "no",
                "yes" if p_provided else "no",
            )
        )

    if not u_provided:
        return False

    u_sub, p_sub = w_rom.subfunctions
    u_sub.dat.data[:] = u_initial_guess.reshape(u_sub.dat.data.shape)
    p_sub.dat.data[:] = p_initial_guess.reshape(p_sub.dat.data.shape)

    if verbose:
        print(f"  Using initial guess: ||alpha_0|| = "
              f"{np.linalg.norm(u_initial_guess):.6e}")
    return True


def _run_solve(solver, callback_data, verbose=True):
    """
    Run the Newton solve. Only Firedrake's ConvergenceError is caught —
    every other exception propagates so real bugs surface immediately.

    Returns (converged, iterations).
    """
    try:
        solver.solve()
        iterations = callback_data['iteration_count']
        # if verbose:
        print(f"##### ✓ Converged in {iterations} iterations #####")
        return True, iterations
    except ConvergenceError as e:
        iterations = callback_data['iteration_count']
        # if verbose:
        print(f"##### ✗ SNES did not converge after {iterations} "
              f"iterations: {e} #####")
        return False, iterations


def _extract_solution(
    w_rom, operators, nu, amp, deim_ops,
    converged, iterations, store_full_dofs,
    *,
    mode: str,
    verbose=True,
) -> ROMSolution:
    """
    Build a ROMSolution from the current state of w_rom.

    `mode` is recorded verbatim in the metadata: it is the mode that was
    actually used, not one inferred from whatever attributes `operators`
    happens to carry.
    """
    u_sub, p_sub = w_rom.subfunctions
    velocity_coeffs = u_sub.dat.data_ro.flatten().copy()
    pressure_coeffs = p_sub.dat.data_ro.flatten().copy()

    velocity_dofs = None
    pressure_dofs = None
    if store_full_dofs:
        thetas = operators._default_thetas(amp)
        u_lift = np.einsum('i,ij->j', thetas, operators.lifting_dofs)
        velocity_dofs = u_lift + operators.Z_u @ velocity_coeffs
        pressure_dofs = operators.Z_p @ pressure_coeffs

    meta = {
        'nu': nu,
        'n_velocity_modes': operators.n_velocity_modes,
        'n_pressure_modes': operators.n_pressure_modes,
        'mode': mode,
        'quadratic_only': operators.has_affine_convection,
    }
    if mode == 'deim':
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

    if verbose:
        print(f"  ||alpha|| = {np.linalg.norm(velocity_coeffs):.6e}")
        print(f"  ||beta||  = {np.linalg.norm(pressure_coeffs):.6e}")
        print("")

    return solution


# =============================================================================
# MAIN SOLVE FUNCTION
# =============================================================================

def solve_rom(
    operators: ReducedOperators,
    nu: float,
    amp: float,
    problem,
    *,
    mode: str,
    deim_ops=None,
    submesh_deim=None,
    u_initial_guess: np.ndarray = None,
    p_initial_guess: np.ndarray = None,
    solver_parameters: dict = None,
    store_full_dofs: bool = True,
    return_internals: bool = False,
    verbose: bool = False,
):
    """
    Solve the reduced Navier-Stokes system.

    Parameters
    ----------
    operators : ReducedOperators
    nu : float
        Viscosity (1/Re).
    amp : float
        Amplitude parameter.
    problem : NavierStokesProblem
        Used by callbacks for FOM-side workspace assembly.
    mode : {'exact', 'deim', 'tensor'}
        Convection evaluation strategy. Required.
            'exact'  -- Z_u^T F(u), full-mesh assembly.
            'deim'   -- DEIM/MDEIM with submesh assembly; requires deim_ops.
            'tensor' -- precomputed exact tensor; requires
                        operators.has_affine_convection and
                        operators.tensor_convection.
    deim_ops : DEIMOnlineOperators or None
        Required for mode='deim'; ignored otherwise.
    submesh_deim : SubMeshDEIM or None
        Pre-built SubMeshDEIM for mode='deim'. Built internally if None.
    u_initial_guess, p_initial_guess : np.ndarray or None
        Initial reduced coordinates. Must be provided as a pair or not at all.
    solver_parameters : dict or None
        PETSc SNES parameters. Defaults to a tight-tolerance Newton/LU setup.
    store_full_dofs : bool, default True
        If True, reconstruct and store full-order DOFs on the returned
        ROMSolution. Turn off for large sweeps where you'll reconstruct
        on demand.
    return_internals : bool, default False
        If True, return (solution, w_rom, callback_data) for post-solve
        operations like the bifurcation indicator.
    verbose : bool, default False
        Print setup, progress, and norms. Set False inside sweep loops.

    Returns
    -------
    ROMSolution
        Or (ROMSolution, w_rom, callback_data) if return_internals=True.
    """
    if verbose:
        print("=" * 60)
        print("SOLVING ROM")
        print("=" * 60)
        print(f"  nu = {nu} (Re = {1/nu:.1f})")
        print(f"  amp = {amp}")

    solver, w_rom, callback_data = _setup_rom_solver(
        operators, nu, amp, problem,
        mode=mode,
        deim_ops=deim_ops,
        solver_parameters=solver_parameters,
        submesh_deim=submesh_deim,
        verbose=verbose,
    )

    _apply_initial_guess(w_rom, u_initial_guess, p_initial_guess, verbose=verbose)

    converged, iterations = _run_solve(solver, callback_data, verbose=verbose)

    solution = _extract_solution(
        w_rom, operators, nu, amp, callback_data['deim_ops'],
        converged, iterations, store_full_dofs,
        mode=callback_data['mode'],
        verbose=verbose,
    )

    if verbose:
        print("#" * 100)
        print("")

    if return_internals:
        return solution, w_rom, callback_data
    return solution


def solve_rom_with_internals(*args, **kwargs):
    """
    Deprecated. Prefer `solve_rom(..., return_internals=True)`.

    Kept as a backward-compatible thin shim while call sites migrate.
    """
    kwargs['return_internals'] = True
    return solve_rom(*args, **kwargs)


# =============================================================================
# PARAMETER SWEEP
# =============================================================================

def solve_rom_parameter_sweep(
    operators: ReducedOperators,
    problem,
    *,
    mode: str,
    reynolds_values: np.ndarray = None,
    amplitude_values: np.ndarray = None,
    deim_ops=None,
    submesh_deim=None,
    use_continuation: bool = True,
    store_full_dofs: bool = False,
    solver_parameters: dict = None,
    verbose: bool = True,
) -> list:
    """
    Solve the ROM at every (Re, amp) pair.

    The solver is built once; only Constants are updated between solves.
    Continuation carries both velocity and pressure coefficients explicitly.

    See `solve_rom` for the meaning of `mode`.
    """
    if reynolds_values is None:
        reynolds_values = np.array([100.0])
    if amplitude_values is None:
        amplitude_values = np.array([0.0])

    # Resolve up front so the banner reports the mode that will actually run.
    mode, _, deim_ops = _resolve_mode(operators, mode, deim_ops)

    n_total = len(reynolds_values) * len(amplitude_values)

    if verbose:
        print("=" * 60)
        print("ROM PARAMETER SWEEP")
        print("=" * 60)
        print(f"  Reynolds:   {reynolds_values}")
        print(f"  Amplitude:  {amplitude_values}")
        print(f"  Total:      {n_total} solves")
        print(f"  Mode:       {_MODE_LABELS[mode]}")
        print(f"  Quadratic:  {operators.has_affine_convection}")

    # Build solver once with the first parameter pair.
    nu_0 = 1.0 / reynolds_values[0]
    amp_0 = amplitude_values[0]

    solver, w_rom, callback_data = _setup_rom_solver(
        operators, nu_0, amp_0, problem,
        mode=mode,
        deim_ops=deim_ops,
        solver_parameters=solver_parameters,
        submesh_deim=submesh_deim,
        verbose=verbose,
    )

    solutions = []
    prev_alpha = None
    prev_beta = None

    for amp in amplitude_values:
        for Re in reynolds_values:
            nu = 1.0 / Re
            if verbose:
                print(f"\n{'─' * 40}")
                print(f"  Re = {Re:.1f}, amp = {amp:.3f}")
                print(f"{'─' * 40}")

            _update_parameters(callback_data, nu, amp)

            # Continuation guess (both fields), or reset
            if use_continuation and prev_alpha is not None:
                _apply_initial_guess(
                    w_rom, prev_alpha, prev_beta, verbose=False,
                )
            else:
                w_rom.assign(0)

            converged, iterations = _run_solve(
                solver, callback_data, verbose=verbose,
            )

            solution = _extract_solution(
                w_rom, operators, nu, amp, callback_data['deim_ops'],
                converged, iterations, store_full_dofs,
                mode=mode,
                verbose=verbose,
            )
            solutions.append(solution)

            if use_continuation:
                prev_alpha = solution.velocity_coeffs
                prev_beta = solution.pressure_coeffs

    if verbose:
        n_converged = sum(s.converged for s in solutions)
        print(f"\n{'=' * 60}")
        print(f"SWEEP COMPLETE: {n_converged}/{n_total} converged")
        print(f"{'=' * 60}")

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
    """Reconstruct full-order velocity from a ROM solution."""
    if amp is None:
        amp = solution.amplitude

    # Always recompute from coefficients: a stored `velocity_dofs` was
    # built at the original amplitude and would be wrong if the caller
    # passes a different `amp` here.
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
    """Reconstruct full-order pressure from a ROM solution."""
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
    """Reconstruct both velocity and pressure as a NavierStokesSolution."""
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
    Relative error in the M-inner-product norm:
        ||u_fom - u_rom||_M / ||u_fom||_M
    """
    u_fom_dofs = u_fom.dat.data_ro.flatten()
    u_rom_dofs = u_rom.dat.data_ro.flatten()
    error = u_fom_dofs - u_rom_dofs

    error_norm = np.sqrt(error.T @ (M_u @ error))
    fom_norm = np.sqrt(u_fom_dofs.T @ (M_u @ u_fom_dofs))

    return error_norm / fom_norm
