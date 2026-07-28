from firedrake import (
    TestFunction, TrialFunction, split,
    Constant, derivative, inner, dx, assemble,Function,
)
import numpy as np
from firedrake.preconditioners.patch import bcdofs
from nsrom.helper_functions import dofs_to_functions
from nsrom.navier_stokes import setup_navier_stokes_problem, load_solution, navier_stokes_residual
from .eigen_solver import solve_leftmost_real_eigenpairs, vec_to_function
SNAPSHOT_DIR      = "multi_param_multi_branch"
DEIM_SNAPSHOT_DIR = "multi_param_multi_branch"
LIFTING_DIR       = "lifting"
MESH_FILE         = "mesh/mid_pinball.msh"
FOM_CHECKPOINT    = "initial_states/velocity_checkpoint_symm"

REYNOLDS_INIT  = 100.0
AMPLITUDE_INIT = 0.0

def build_state_jacobian_pencil(problem, base_solution, mat_type="aij"):
    """
    Build (J, M) for the generalized eigenproblem  J δw = μ M δw
    linearized at base_solution.

    BC handling (textbook hydrodynamic stability): default Firedrake BC
    application leaves J_ii = M_ii = 1 at constrained velocity DOFs,
    giving spurious μ = 1. We then explicitly zero M's rows/cols/diag
    at those DOFs (diag=0), which sends spurious eigenvalues to ∞ —
    far from the physical leftmost real μ near 0. Symmetry of M is
    preserved (both row and column are zeroed).

    Bifurcation criterion: smallest real μ crosses zero. On the stable
    symmetric branch all real μ > 0; the leading one ↓ 0 as Re ↑ Re_c.
    (Linearized stability eigenvalues σ = −μ; we stay with μ.)

    Parameters
    ----------
    problem : NavierStokesProblem
    base_solution : NavierStokesSolution
        Sets problem.reynolds, problem.amplitude internally.
    mat_type : str
        PETSc format. "aij" for SLEPc + MUMPS shift-invert.

    Returns
    -------
    J, M : PETSc.Mat
    """
    Z = problem.mixed_space

    problem.reynolds.assign(base_solution.reynolds)
    problem.amplitude.assign(base_solution.amplitude)

    # State Jacobian via UFL derivative.
    w = base_solution.solution
    v = TestFunction(Z)
    F = navier_stokes_residual(w, v, problem.reynolds)
    J_form = derivative(F, w)

    # Mass on velocity component only — pressure block naturally zero.
    trial = TrialFunction(Z)
    u_t, _ = split(trial)
    v_u, _ = split(v)
    M_form = inner(u_t, v_u) * dx

    # Pass the problem BCs as-is. For matrix assembly only, BC *values*
    # don't enter the matrix; only the row/col zeroing pattern matters.
    bcs = problem.boundary_conditions

    J = assemble(J_form, bcs=bcs, mat_type=mat_type)
    M = assemble(M_form, bcs=bcs, mat_type=mat_type)

    # Push spurious BC-DOF eigenvalues to ∞ by zeroing M's BC rows/cols/diag.
    for bc in bcs:
        M.M.handle.zeroRowsColumns(bcdofs(bc), diag=0.0)

    return J.M.handle, M.M.handle


