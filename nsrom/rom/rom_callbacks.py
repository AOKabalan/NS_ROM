"""
ROM Callback Factory (with exact tensorial convection branch).

Projection modes:
    Exact projection : Z_u^T @ F(u)        (full-mesh assembly, no speedup)
    DEIM projection  : deim(F(u))          (submesh assembly)
    Tensor (exact)   : tensor(alpha)       (NO assembly; exact; quadratic_only only)

Tensor mode replaces DEIM/MDEIM for the pure quadratic convection C(u',u') with
an exact tensorial evaluation (tensor_convection.py). Valid ONLY in
quadratic_only mode: the tensor is built from Z_u alone, so lifting cross-terms
and the constant term must still be supplied affinely by the solver -- the same
contract the DEIM/exact quadratic_only path already relies on.
"""

import numpy as np
import scipy.sparse as sp
from firedrake import (
    Function, TrialFunction, TestFunction,
    inner, grad, dx, assemble,
)

__all__ = ['make_callbacks']


def make_callbacks(callback_data: dict):
    Z_u = callback_data['Z_u']
    N_u = callback_data['N_u']
    V_full = callback_data['V_full']
    u_work = callback_data['u_work']
    use_deim = callback_data['use_deim']
    quadratic_only = callback_data['quadratic_only']

    use_tensor = callback_data.get('use_tensor', False)
    tc = callback_data.get('tensor_convection', None)
    if use_tensor:
        if not quadratic_only:
            raise ValueError("use_tensor=True requires quadratic_only=True "
                             "(tensor represents C(u',u') only).")
        if tc is None:
            raise ValueError("use_tensor=True but 'tensor_convection' is None.")
        if use_deim:
            raise ValueError("use_tensor and use_deim are mutually exclusive.")

    sm = callback_data.get('submesh_deim', None)
    v_test = TestFunction(V_full)
    u_trial = TrialFunction(V_full)

    def _reconstruct_velocity(current_solution):
        coeffs = current_solution.array_r
        u_rom_coeffs = coeffs[:N_u]
        if quadratic_only:
            u_np = Z_u @ u_rom_coeffs
        else:
            thetas = np.array(callback_data['thetas'])
            lifting = callback_data['lifting_dofs']
            u_np = np.einsum('i,ij->j', thetas, lifting) + Z_u @ u_rom_coeffs
        u_work.dat.data[:] = u_np.reshape(u_work.dat.data.shape)

    def update_convection_residual(current_solution):
        if use_tensor:
            alpha = current_solution.array_r[:N_u]
            callback_data['conv_res_const'].assign(tc.residual(alpha))
            return
        _reconstruct_velocity(current_solution)
        if use_deim:
            sm.transfer_to_submesh(u_work)
            F_sub = assemble(inner(grad(sm.u_sub) * sm.u_sub, sm.v_test) * dx)
            conv_res = sm.extract_residual(F_sub)
        else:
            F_form = inner(grad(u_work) * u_work, v_test) * dx
            F_vec = assemble(F_form).dat.data_ro.flatten()
            conv_res = Z_u.T @ F_vec
        callback_data['conv_res_const'].assign(conv_res)

    def update_convection_jacobian(current_solution):
        callback_data['iteration_count'] += 1
        if use_tensor:
            alpha = current_solution.array_r[:N_u]
            callback_data['conv_J_const'].assign(tc.jacobian(alpha))
            return
        _reconstruct_velocity(current_solution)
        if use_deim:
            sm.transfer_to_submesh(u_work)
            J_sub = assemble(
                (inner(grad(sm.u_trial) * sm.u_sub, sm.v_test)
                 + inner(grad(sm.u_sub) * sm.u_trial, sm.v_test)) * dx,
                mat_type='aij'
            )
            _, _, J_data_sub = J_sub.M.handle.getValuesCSR()
            conv_J = sm.extract_jacobian(J_data_sub)
        else:
            J_form = (inner(grad(u_trial) * u_work, v_test) * dx
                      + inner(grad(u_work) * u_trial, v_test) * dx)
            J_assembled = assemble(J_form, mat_type='aij')
            J_petsc = J_assembled.M.handle
            indptr, indices, J_data = J_petsc.getValuesCSR()
            J_scipy = sp.csr_matrix((J_data, indices, indptr),
                                    shape=J_petsc.getSize())
            conv_J = Z_u.T @ (J_scipy @ Z_u)
        callback_data['conv_J_const'].assign(conv_J)

    return update_convection_residual, update_convection_jacobian
