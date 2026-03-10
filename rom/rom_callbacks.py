"""
ROM Callback Factory.

Creates residual and Jacobian callbacks for the ROM solver.
Handles 4 modes transparently:

    ┌─────────────────┬──────────────────┬─────────────────────┐
    │                 │ Exact projection │ DEIM projection     │
    ├─────────────────┼──────────────────┼─────────────────────┤
    │ Full u_total    │ Z_u^T @ F(u_tot) │ deim(F(u_tot))     │
    │ Quadratic only  │ Z_u^T @ F(u_red) │ deim(F(u_red))     │
    └─────────────────┴──────────────────┴─────────────────────┘

The solver doesn't know which mode is active — it just calls the callbacks.
"""

import numpy as np
import scipy.sparse as sp
from firedrake import (
    Function, TrialFunction, TestFunction,
    inner, grad, dx, assemble,
)

__all__ = ['make_callbacks']


def make_callbacks(callback_data: dict):
    """
    Create residual and Jacobian callbacks for the ROM solver.

    The mode is determined by callback_data:
        - quadratic_only: True if affine convection is precomputed
                          (callbacks assemble only the (Phi*alpha . grad) Phi*alpha part)
        - use_deim: True if DEIM/MDEIM is used for projection

    Args:
        callback_data: dict containing:
            Z_u:             (n_dofs, N_u) velocity basis
            N_u:             int, number of velocity modes
            V_full:          FOM velocity function space
            u_work:          Function(V_full) workspace
            use_deim:        bool
            quadratic_only:  bool
            thetas:          list of floats [theta_0, theta_1, ...]
            lifting_dofs:    (K, n_dofs) lifting function DOFs
            deim_ops:        DEIMOnlineOperators or None
            submesh_deim:    SubMeshDEIM or None (set when use_deim=True)
            conv_res_const:  Firedrake Constant (N_u,) to update
            conv_J_const:    Firedrake Constant (N_u, N_u) to update
            iteration_count: int (mutable counter)

    Returns:
        (update_residual, update_jacobian): Pair of callback functions
    """
    # Unpack (references, not copies)
    Z_u = callback_data['Z_u']
    N_u = callback_data['N_u']
    V_full = callback_data['V_full']
    u_work = callback_data['u_work']
    use_deim = callback_data['use_deim']
    quadratic_only = callback_data['quadratic_only']

    # Submesh infrastructure (only when use_deim=True)
    sm = callback_data.get('submesh_deim', None)

    # Firedrake forms on full mesh (for exact mode)
    v_test = TestFunction(V_full)
    u_trial = TrialFunction(V_full)

    # -----------------------------------------------------------------
    # Shared helper
    # -----------------------------------------------------------------

    def _reconstruct_velocity(current_solution):
        """
        Build the velocity field for FOM assembly.

        If quadratic_only: u = Z_u @ alpha  (reduced part only)
        If full:           u = sum(theta_k * u_k) + Z_u @ alpha  (total)
        """
        coeffs = current_solution.array_r
        u_rom_coeffs = coeffs[:N_u]

        if quadratic_only:
            u_np = Z_u @ u_rom_coeffs
        else:
            thetas = np.array(callback_data['thetas'])
            lifting = callback_data['lifting_dofs']
            u_np = np.einsum('i,ij->j', thetas, lifting) + Z_u @ u_rom_coeffs

        u_work.dat.data[:] = u_np.reshape(u_work.dat.data.shape)

    # -----------------------------------------------------------------
    # Residual callback
    # -----------------------------------------------------------------

    def update_convection_residual(current_solution):
        """
        Update the convection contribution to the residual.

        Assembles ((u . grad) u, v) then projects to reduced space.

        DEIM mode: assemble on submesh (281 cells) instead of full mesh (17k cells).
        Exact mode: assemble on full mesh, project with Z_u^T.
        """
        _reconstruct_velocity(current_solution)

        if use_deim:
            # Transfer to submesh and assemble there
            sm.transfer_to_submesh(u_work)
            F_sub = assemble(
                inner(grad(sm.u_sub) * sm.u_sub, sm.v_test) * dx
            )
            conv_res = sm.extract_residual(F_sub)
        else:
            # Full assembly + exact projection
            F_form = inner(grad(u_work) * u_work, v_test) * dx
            F_vec = assemble(F_form).dat.data_ro.flatten()
            conv_res = Z_u.T @ F_vec

        callback_data['conv_res_const'].assign(conv_res)

    # -----------------------------------------------------------------
    # Jacobian callback
    # -----------------------------------------------------------------

    def update_convection_jacobian(current_solution):
        """
        Update the convection contribution to the Jacobian.

        Assembles both product-rule terms:
            N: (u . grad) du    — u advects perturbation
            M: (du . grad) u   — perturbation advects u

        DEIM mode: assemble on submesh, extract via MDEIM.
        Exact mode: assemble on full mesh, project with Z_u^T @ J @ Z_u.
        """
        callback_data['iteration_count'] += 1

        _reconstruct_velocity(current_solution)

        if use_deim:
            # Transfer to submesh and assemble there
            sm.transfer_to_submesh(u_work)
            J_sub = assemble(
                (inner(grad(sm.u_trial) * sm.u_sub, sm.v_test)
                 + inner(grad(sm.u_sub) * sm.u_trial, sm.v_test)) * dx,
                mat_type='aij'
            )
            _, _, J_data_sub = J_sub.M.handle.getValuesCSR()
            conv_J = sm.extract_jacobian(J_data_sub)
        else:
            # Full assembly + exact projection
            J_form = (
                inner(grad(u_trial) * u_work, v_test) * dx
                + inner(grad(u_work) * u_trial, v_test) * dx
            )
            J_assembled = assemble(J_form, mat_type='aij')
            J_petsc = J_assembled.M.handle
            indptr, indices, J_data = J_petsc.getValuesCSR()
            J_scipy = sp.csr_matrix(
                (J_data, indices, indptr), shape=J_petsc.getSize()
            )
            conv_J = Z_u.T @ (J_scipy @ Z_u)

        callback_data['conv_J_const'].assign(conv_J)

        print(f"      Newton iteration {callback_data['iteration_count']}: "
              f"||J_conv|| = {np.linalg.norm(conv_J):.6e}")

    return update_convection_residual, update_convection_jacobian