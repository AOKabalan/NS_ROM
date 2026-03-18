"""
Reduced Operator Construction.

Builds reduced-order operators from POD basis and lifting functions.
All operators stored as numpy arrays for portability.
"""

import numpy as np
import scipy.sparse as sp
from firedrake import (
    Function, TrialFunction, TestFunction,
    inner, grad, div, dx, assemble, Constant
)

from .data_structures import PODBasis, ReducedOperators
from .pod import (
    assemble_inner_product_matrix,
    supremizer_enrichment,
    enrich_velocity_basis,
)

__all__ = [
    'assemble_full_order_operators',
    'build_reduced_operators',
    'assemble_convection_matrix',
    'assemble_convection_jacobian',
]


# =============================================================================
# FULL-ORDER OPERATOR ASSEMBLY
# =============================================================================

def assemble_full_order_operators(problem) -> dict:
    """
    Assemble full-order Navier-Stokes operators.

    Returns sparse matrices:
        A: Stiffness (viscous term)
        B: Divergence (pressure-velocity coupling)
        f: Forcing vector
        g: Divergence RHS (typically zero)
    """
    V = problem.velocity_space
    Q = problem.pressure_space

    print("  Assembling full-order operators...")

    u_trial = TrialFunction(V)
    v_test = TestFunction(V)
    p_trial = TrialFunction(Q)
    q_test = TestFunction(Q)

    # Stiffness matrix: (grad(u), grad(v))
    A_form = inner(grad(u_trial), grad(v_test)) * dx
    A_assembled = assemble(A_form, mat_type='aij')

    # Divergence matrix: -(q, div(u))
    B_form = -q_test * div(u_trial) * dx
    B_assembled = assemble(B_form, mat_type='aij')

    # Forcing vector (zero for now)
    f_form = inner(Constant((0.0, 0.0)), v_test) * dx
    f_assembled = assemble(f_form)

    # Divergence RHS (zero)
    g_form = Constant(0.0) * q_test * dx
    g_assembled = assemble(g_form)

    # Convert to scipy sparse
    A_petsc = A_assembled.M.handle
    indptr_A, indices_A, data_A = A_petsc.getValuesCSR()
    A_scipy = sp.csr_matrix((data_A, indices_A, indptr_A), shape=A_petsc.getSize())

    B_petsc = B_assembled.M.handle
    indptr_B, indices_B, data_B = B_petsc.getValuesCSR()
    B_scipy = sp.csr_matrix((data_B, indices_B, indptr_B), shape=B_petsc.getSize())

    f_np = f_assembled.dat.data.flatten()
    g_np = g_assembled.dat.data.flatten()

    print(f"    A: {A_scipy.shape}")
    print(f"    B: {B_scipy.shape}")
    print(f"    f: {f_np.shape}")
    print(f"    g: {g_np.shape}")

    return {'A': A_scipy, 'B': B_scipy, 'f': f_np, 'g': g_np}


# =============================================================================
# AFFINE CONVECTION ASSEMBLY (offline)
# =============================================================================

def _assemble_affine_convection(
    lifting_dofs: np.ndarray,
    problem,
    Z_u: np.ndarray,
) -> dict:
    """
    Precompute affine convection cross-terms for K lifting components.

    For each lifting component u_k, assembles:
        c_conv[k,l] = Z_u^T @ assemble( (u_k . grad) u_l )    -- vectors
        N_conv[k]   = Z_u^T @ N(u_k) @ Z_u                    -- matrices
        M_conv[k]   = Z_u^T @ M(u_k) @ Z_u                    -- matrices

    Args:
        lifting_dofs: (K, n_dofs) lifting function DOFs
        problem: NavierStokesProblem
        Z_u: (n_dofs, N_u) velocity basis

    Returns:
        dict with 'c_conv', 'N_conv', 'M_conv'
    """
    K = lifting_dofs.shape[0]
    N_u = Z_u.shape[1]
    V = problem.velocity_space

    u_trial = TrialFunction(V)
    v_test = TestFunction(V)

    # Workspace functions
    u_k_func = Function(V, name="u_k")
    u_l_func = Function(V, name="u_l")

    c_conv = np.zeros((K, K, N_u))
    N_conv = np.zeros((K, N_u, N_u))
    M_conv = np.zeros((K, N_u, N_u))

    print(f"\n  Computing affine convection terms (K={K})...")

    # --- Constant vectors: c_conv[k,l] = Z_u^T @ assemble((u_k . grad) u_l) ---
    for k in range(K):
        u_k_func.dat.data[:] = lifting_dofs[k].reshape(u_k_func.dat.data.shape)
        for l in range(K):
            u_l_func.dat.data[:] = lifting_dofs[l].reshape(u_l_func.dat.data.shape)

            # (u_k . grad) u_l
            F_form = inner(grad(u_l_func) * u_k_func, v_test) * dx
            F_vec = assemble(F_form).dat.data_ro.flatten()
            c_conv[k, l] = Z_u.T @ F_vec

            print(f"    c_conv[{k},{l}]: ||.|| = {np.linalg.norm(c_conv[k,l]):.6e}")

    # --- Linear matrices: N_conv[k], M_conv[k] ---
    for k in range(K):
        u_k_func.dat.data[:] = lifting_dofs[k].reshape(u_k_func.dat.data.shape)

        # N_conv[k]: (u_k . grad) Phi_j
        N_form = inner(grad(u_trial) * u_k_func, v_test) * dx
        N_assembled = assemble(N_form, mat_type='aij')
        N_petsc = N_assembled.M.handle
        indptr, indices, data = N_petsc.getValuesCSR()
        N_scipy = sp.csr_matrix((data, indices, indptr), shape=N_petsc.getSize())
        N_conv[k] = Z_u.T @ (N_scipy @ Z_u)

        # M_conv[k]: (Phi_j . grad) u_k
        M_form = inner(grad(u_k_func) * u_trial, v_test) * dx
        M_assembled = assemble(M_form, mat_type='aij')
        M_petsc = M_assembled.M.handle
        indptr, indices, data = M_petsc.getValuesCSR()
        M_scipy = sp.csr_matrix((data, indices, indptr), shape=M_petsc.getSize())
        M_conv[k] = Z_u.T @ (M_scipy @ Z_u)

        print(f"    N_conv[{k}]: ||.|| = {np.linalg.norm(N_conv[k]):.6e}")
        print(f"    M_conv[{k}]: ||.|| = {np.linalg.norm(M_conv[k]):.6e}")

    return {'c_conv': c_conv, 'N_conv': N_conv, 'M_conv': M_conv}


# =============================================================================
# REDUCED OPERATOR CONSTRUCTION
# =============================================================================

def build_reduced_operators(
    problem,
    basis: PODBasis,
    lifting,                          # LiftingFunctions
    inner_product_type: str = 'H1',   # For supremizer enrichment
    compute_affine_convection=False,
) -> ReducedOperators:
    """
    Build reduced-order operators from POD basis and lifting functions.

    Internally performs supremizer enrichment (if supremizer modes exist
    in the basis) via Gram-Schmidt orthonormalization.

    Computes:
        A_N = Z_u^T @ A @ Z_u
        B_N = Z_p^T @ B @ Z_u
        f_N = Z_u^T @ f
        g_N = Z_p^T @ g

    Plus K-indexed lifting contributions and optionally affine
    convection cross-terms for efficient online assembly.

    Args:
        problem: NavierStokesProblem
        basis: PODBasis with velocity_modes, pressure_modes, supremizer_modes
        lifting: LiftingFunctions with u_base, u_control
        inner_product_type: Inner product for Gram-Schmidt ('L2', 'H1', 'H1_semi')
        compute_affine_convection: If True, precompute c_conv, N_conv, M_conv

    Returns:
        ReducedOperators dataclass
    """
    print("=" * 60)
    print("BUILDING REDUCED OPERATORS")
    print("=" * 60)

    # ---- Supremizer enrichment ----
    Z_v = basis.velocity_modes
    Z_p = basis.pressure_modes

    if basis.supremizer_modes is not None:
        print("\n  Enriching velocity basis with supremizers...")

        # Compute supremizers from pressure modes
        # Orthonormalize against velocity modes
        M_u = assemble_inner_product_matrix(problem.velocity_space, inner_product_type)
        Z_u = enrich_velocity_basis(Z_v, basis.supremizer_modes, M_u)
    else:
        print("\n  No supremizer modes — using velocity basis as-is")
        Z_u = Z_v

    N_u = Z_u.shape[1]
    N_p = Z_p.shape[1]

    print(f"  Velocity basis: {Z_u.shape} ({N_u} modes)")
    print(f"    ({Z_v.shape[1]} velocity + {N_u - Z_v.shape[1]} supremizer)")
    print(f"  Pressure basis: {Z_p.shape} ({N_p} modes)")

    # ---- Full-order operators ----
    fom = assemble_full_order_operators(problem)
    A = fom['A']
    B = fom['B']
    f = fom['f']
    g = fom['g']

    # ---- Core projection ----
    print("\n  Projecting to reduced space...")

    A_N = Z_u.T @ (A @ Z_u)
    B_N = Z_p.T @ (B @ Z_u)
    f_N = Z_u.T @ f
    g_N = Z_p.T @ g

    print(f"    A_N: {A_N.shape}")
    print(f"    B_N: {B_N.shape}")
    print(f"    f_N: {f_N.shape}")
    print(f"    g_N: {g_N.shape}")

    # ---- K-indexed lifting ----
    #
    # Collect all lifting functions. Currently:
    #   k=0: u_base   (theta_0 = 1.0)
    #   k=1: u_control (theta_1 = amp)
    #
    # To add more, just extend this list.

    lifting_list = [
        lifting.u_base.dat.data_ro.flatten(),
        lifting.u_control.dat.data_ro.flatten(),
    ]
    K = len(lifting_list)
    lifting_dofs_array = np.array(lifting_list)  # (K, n_dofs)

    print(f"\n  Computing lifting contributions (K={K})...")

    A_lift = np.zeros((K, N_u))
    B_lift = np.zeros((K, N_p))

    for k in range(K):
        A_lift[k] = Z_u.T @ (A @ lifting_dofs_array[k])
        B_lift[k] = Z_p.T @ (B @ lifting_dofs_array[k])
        print(f"    Lifting[{k}]: ||A_lift|| = {np.linalg.norm(A_lift[k]):.6e}, "
              f"||B_lift|| = {np.linalg.norm(B_lift[k]):.6e}")

    # ---- Affine convection (optional) ----
    conv_kwargs = {}
    if compute_affine_convection:
        conv_terms = _assemble_affine_convection(lifting_dofs_array, problem, Z_u)
        conv_kwargs = conv_terms

    # ---- Build ReducedOperators ----
    operators = ReducedOperators(
        A_N=A_N,
        B_N=B_N,
        f_N=f_N,
        g_N=g_N,
        Z_u=Z_u,
        Z_p=Z_p,
        n_lifting=K,
        lifting_dofs=lifting_dofs_array,
        A_lift=A_lift,
        B_lift=B_lift,
        metadata={
            'n_velocity_modes': basis.n_velocity_modes,
            'n_pressure_modes': basis.n_pressure_modes,
            'n_supremizer_modes': N_u - basis.n_velocity_modes,
            'n_enriched_modes': N_u,
            'inner_product_type': inner_product_type,
            'reynolds_ref': lifting.reynolds_ref,
        },
        **conv_kwargs,
    )

    print("\n" + "=" * 60)
    print("REDUCED OPERATORS COMPLETE")
    print("=" * 60)
    print(f"  N_velocity (enriched): {operators.n_velocity_modes}")
    print(f"  N_pressure: {operators.n_pressure_modes}")
    print(f"  N_lifting: {K}")
    print(f"  Affine convection: {operators.has_affine_convection}")

    return operators


# =============================================================================
# CONVECTION OPERATORS (for online / exact callbacks)
# =============================================================================

def assemble_convection_matrix(
    u_advecting: np.ndarray,
    problem,
    Z_u: np.ndarray,
) -> np.ndarray:
    """
    Assemble reduced convection matrix: N(u) = Z_u^T @ N_full(u) @ Z_u

    N_full corresponds to: (u . grad) u_trial
    """
    V = problem.velocity_space
    u_func = Function(V)
    u_func.dat.data[:] = u_advecting.reshape(u_func.dat.data.shape)

    u_trial = TrialFunction(V)
    v_test = TestFunction(V)

    N_form = inner(grad(u_trial) * u_func, v_test) * dx
    N_assembled = assemble(N_form, mat_type='aij')

    N_petsc = N_assembled.M.handle
    indptr, indices, data = N_petsc.getValuesCSR()
    N_scipy = sp.csr_matrix((data, indices, indptr), shape=N_petsc.getSize())

    return Z_u.T @ (N_scipy @ Z_u)


def assemble_convection_jacobian(
    u_advecting: np.ndarray,
    problem,
    Z_u: np.ndarray,
) -> tuple:
    """
    Assemble reduced convection Jacobian matrices.

    Returns (N_N, M_N) where:
        N_N: (u_total . grad) Phi_j   — advection BY u_total
        M_N: (Phi_j . grad) u_total   — Phi_j advects u_total

    Total Jacobian contribution = N_N + M_N
    """
    V = problem.velocity_space
    u_func = Function(V)
    u_func.dat.data[:] = u_advecting.reshape(u_func.dat.data.shape)

    u_trial = TrialFunction(V)
    v_test = TestFunction(V)

    # N: (u . grad) u_trial
    N_form = inner(grad(u_trial) * u_func, v_test) * dx
    N_assembled = assemble(N_form, mat_type='aij')
    N_petsc = N_assembled.M.handle
    indptr_N, indices_N, data_N = N_petsc.getValuesCSR()
    N_scipy = sp.csr_matrix((data_N, indices_N, indptr_N), shape=N_petsc.getSize())

    # M: (u_trial . grad) u
    M_form = inner(grad(u_func) * u_trial, v_test) * dx
    M_assembled = assemble(M_form, mat_type='aij')
    M_petsc = M_assembled.M.handle
    indptr_M, indices_M, data_M = M_petsc.getValuesCSR()
    M_scipy = sp.csr_matrix((data_M, indices_M, indptr_M), shape=M_petsc.getSize())

    N_N = Z_u.T @ (N_scipy @ Z_u)
    M_N = Z_u.T @ (M_scipy @ Z_u)

    return N_N, M_N