"""
Exact tensorial reduced convection for the quadratic Navier-Stokes term.

Replaces DEIM/MDEIM for the (u'.grad)u' part in quadratic_only mode.
Exploits that the convective Jacobian is LINEAR in the state:

    J(Z a) = sum_j a_j J(phi_j)          (phi_j = columns of Z_u)
    B(u,u) = 1/2 J(u) u                  (quadratic residual via polarization)

so with N_j = Z^T J(phi_j) Z precomputed offline (r assemblies per cluster),

    F_r(a)      = 1/2 * sum_j a_j (N_j a)                    exact
    dF_r/da_k   = 1/2 * [ N_k a + (sum_j a_j N_j) e_k ]      exact

i.e.  J_r(a) = 1/2 * [ T2(a) + M(a) ]  with
    M(a)  = sum_j a_j N_j                (contraction over first index)
    T2(a) = matrix whose k-th column is N_k a

This is exact for every a in R^r -- supremizer directions included --
because the "training set" IS the basis. Storage r^3 floats
(41^3 = 0.5 MB, 80^3 = 4 MB). Online cost ~r^3 flops per evaluation.

Offline build reuses the same form as assemble_jacobian_snapshots, so the
sign/convention matches the DEIM path and the exact callbacks verbatim.
"""

import numpy as np
from pathlib import Path
from firedrake import (
    Function, TestFunction, TrialFunction, inner, grad, dx, assemble,
)
import scipy.sparse as sp


# =============================================================================
# Offline
# =============================================================================

def build_convection_tensor(Z_u, velocity_space, log_interval=10):
    """
    Precompute N_j = Z^T J(phi_j) Z for every column phi_j of Z_u.

    Args:
        Z_u: (n_dofs, r) velocity basis INCLUDING supremizer columns
             (must be the same array the solver puts in callback_data['Z_u'])
        velocity_space: full-mesh VectorFunctionSpace

    Returns:
        N: (r, r, r) array, N[j] = Z^T J(phi_j) Z
    """
    n_dofs, r = Z_u.shape
    v = TestFunction(velocity_space)
    u_trial = TrialFunction(velocity_space)
    phi = Function(velocity_space)

    N = np.zeros((r, r, r))
    print(f"Building convection tensor: {r} Jacobian assemblies...")
    for j in range(r):
        if (j + 1) % log_interval == 0 or j == 0:
            print(f"  column {j+1}/{r}")
        phi.dat.data[:] = Z_u[:, j].reshape(phi.dat.data.shape)
        # SAME form as assemble_jacobian_snapshots / exact-mode callback:
        J_form = (inner(grad(u_trial) * phi, v) * dx
                  + inner(grad(phi) * u_trial, v) * dx)
        indptr, cols, data = assemble(J_form, mat_type='aij').M.handle.getValuesCSR()
        J_sp = sp.csr_matrix((data, cols, indptr), shape=(n_dofs, n_dofs))
        N[j] = Z_u.T @ (J_sp @ Z_u)
    print(f"  tensor shape {N.shape}, {N.nbytes/1e6:.1f} MB")
    return N


def save_convection_tensor(N, path):
    np.savez_compressed(path, N=N)
    print(f"Saved convection tensor to {path}")


def load_convection_tensor(path):
    return np.load(path)['N']


# =============================================================================
# Online (drop-in for the DEIM branch of the callbacks)
# =============================================================================

class TensorConvection:
    """
    Exact reduced convection evaluation. Same outputs as the DEIM branch of
    update_convection_residual / update_convection_jacobian, but exact and
    assembly-free.

    Usage inside make_callbacks (replacing the use_deim branch):

        tc = callback_data['tensor_convection']       # TensorConvection
        alpha = current_solution.array_r[:N_u]
        conv_res = tc.residual(alpha)                 # (r,)
        conv_J   = tc.jacobian(alpha)                 # (r, r)
    """

    def __init__(self, N):
        self.N = np.ascontiguousarray(N)      # (r, r, r)
        self.r = N.shape[0]

    def _M(self, alpha):
        """sum_j alpha_j N_j -> (r, r).  This equals Z^T J(Z a) Z exactly."""
        return np.tensordot(alpha, self.N, axes=([0], [0]))

    def residual(self, alpha):
        """Exact Z^T B(Za, Za) = 1/2 (Z^T J(Za) Z) a."""
        return 0.5 * (self._M(alpha) @ alpha)

    def jacobian(self, alpha):
        """
        Exact d/da [ Z^T B(Za,Za) ] = Z^T J(Za) Z  -- because
        d/da_k 1/2 (M(a) a) = 1/2 [ N_k a + M(a) e_k ] and, by the symmetry
        B(u,w)+B(w,u) = J(u)w = J(w)u for this trilinear form,
        T2(a) (columns N_k a) equals M(a). We compute both and average to
        stay exact even if the assembled form breaks that symmetry at
        roundoff / boundary-term level.
        """
        M = self._M(alpha)
        T2 = np.tensordot(self.N, alpha, axes=([2], [0])).T  # T2[i,k] = (N_k a)_i
        return 0.5 * (M + T2)


# =============================================================================
# Verification helper
# =============================================================================

def verify_against_exact(tc, Z_u, velocity_space, alpha, atol=1e-9):
    """
    Compare tensor evaluation with a fresh full-mesh assembly at state
    u' = Z a  (the exact-mode callback path). Run once per cluster on a
    random alpha and on a real solution's alpha.
    """
    v = TestFunction(velocity_space)
    u_trial = TrialFunction(velocity_space)
    u_work = Function(velocity_space)
    u_work.dat.data[:] = (Z_u @ alpha).reshape(u_work.dat.data.shape)

    F_vec = assemble(inner(grad(u_work) * u_work, v) * dx).dat.data_ro.flatten()
    red_exact = Z_u.T @ F_vec
    eF = (np.linalg.norm(tc.residual(alpha) - red_exact)
          / max(np.linalg.norm(red_exact), 1e-30))

    J_form = (inner(grad(u_trial) * u_work, v) * dx
              + inner(grad(u_work) * u_trial, v) * dx)
    indptr, cols, data = assemble(J_form, mat_type='aij').M.handle.getValuesCSR()
    J_sp = sp.csr_matrix((data, cols, indptr),
                         shape=(Z_u.shape[0], Z_u.shape[0]))
    redJ_exact = Z_u.T @ (J_sp @ Z_u)
    eJ = (np.linalg.norm(tc.jacobian(alpha) - redJ_exact)
          / np.linalg.norm(redJ_exact))

    ok = eF < atol and eJ < atol
    print(f"[tensor verify] rel err F = {eF:.2e}, J = {eJ:.2e} "
          f"{'OK' if ok else '<-- MISMATCH'}")
    return eF, eJ
