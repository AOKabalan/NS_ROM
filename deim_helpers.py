"""DEIM cache validation and debug utilities."""

import numpy as np
import os
import scipy.sparse as sp

from firedrake import TestFunction, TrialFunction, inner, grad, dx, assemble
from deim import load_basis_metadata
from online_operators import load_ops_metadata


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


def deim_debug(operators, deim_ops, problem, u_test):
    """Quick sanity check: compare exact vs DEIM projection."""
    Z_u = operators.Z_u
    V_full = problem.velocity_space
    v_test = TestFunction(V_full)

    F_form = inner(grad(u_test) * u_test, v_test) * dx
    F_vec = assemble(F_form).dat.data_ro.flatten()
    F_exact = Z_u.T @ F_vec
    F_deim = deim_ops.reduced_convective(F_vec)
    err_F = np.linalg.norm(F_exact - F_deim) / np.linalg.norm(F_exact)

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
