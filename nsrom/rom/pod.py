"""
POD Basis Computation.

Provides functions for:
- Computing POD modes via method of snapshots
- Supremizer enrichment for inf-sup stability
- Snapshot homogenization (lifting function subtraction)
- Inner product assembly

All functions work with numpy arrays internally.
Firedrake is only needed for inner product assembly and supremizer computation.
"""
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from firedrake import (
TrialFunction, TestFunction, assemble, div, dx,
DirichletBC, Constant,
)
from typing import Literal, Tuple, List
import numpy as np
from scipy import linalg as la

from scipy.sparse import csr_matrix
from firedrake import (
    Function, TestFunction, TrialFunction, FunctionSpace,
    inner, grad, div, dx, assemble, solve, Constant, DirichletBC
)

from .data_structures import PODBasis
from helper_functions import dofs_to_function, dofs_to_functions, functions_to_dofs

__all__ = [
    # Inner products
    'assemble_inner_product_matrix',
    'assemble_inner_product',
    # Snapshot processing
    'homogenize_snapshots',
    # POD computation
    'build_correlation_matrix',
    'solve_eigenvalue_problem',
    'compute_pod_modes',
    'compute_pod_basis',
    'truncate_modes',
    # Supremizer
    'compute_supremizer_snapshots',
    'compute_supremizer_modes',
    'supremizer_enrichment',
    'supremizer_enrichment_fd',
    # Utilities
    'project_to_basis',
    'reconstruct_from_basis',
    'compute_reconstruction_error',
    'check_orthonormality',
    'enrich_velocity_basis',
]


# =============================================================================
# INNER PRODUCT MATRICES
# =============================================================================

def assemble_inner_product(
    space: FunctionSpace,
    inner_product_type: Literal['L2', 'H1', 'H1_semi'] = 'L2',
    bc: bool = False
) -> csr_matrix:

    u = TrialFunction(space)
    v = TestFunction(space)

    if inner_product_type == 'L2':
        form = inner(u, v) * dx
    elif inner_product_type == 'H1':
        form = inner(u, v) * dx + inner(grad(u), grad(v)) * dx
    elif inner_product_type == 'H1_semi':
        form = inner(grad(u), grad(v)) * dx
    else:
        raise ValueError(f"Unknown inner_product_type: {inner_product_type}")

    boundary_markers = [1, 3, 4, 5, 6]

    bcs = [DirichletBC(space, Constant((0.0, 0.0)), marker) for marker in boundary_markers]



    if bc==False:
        return assemble(form, mat_type='aij')
    else:
        M_petsc = assemble(form, mat_type='aij', bcs = bcs).M.handle
        indptr, indices, data = M_petsc.getValuesCSR()

        return csr_matrix((data, indices, indptr), shape=M_petsc.getSize())




def assemble_inner_product_matrix(
    space: FunctionSpace,
    inner_product_type: Literal['L2', 'H1', 'H1_semi'] = 'L2'
) -> csr_matrix:
    """
    Assemble inner product matrix for a function space.

    Args:
        space: Firedrake FunctionSpace (scalar or vector)
        inner_product_type: Type of inner product
            - 'L2': (u, v)
            - 'H1': (u, v) + (grad(u), grad(v))
            - 'H1_semi': (grad(u), grad(v))

    Returns:
        scipy csr_matrix of shape (n_dofs, n_dofs)
    """
    u = TrialFunction(space)
    v = TestFunction(space)

    if inner_product_type == 'L2':
        form = inner(u, v) * dx
    elif inner_product_type == 'H1':
        form = inner(u, v) * dx + inner(grad(u), grad(v)) * dx
    elif inner_product_type == 'H1_semi':
        form = inner(grad(u), grad(v)) * dx
    else:
        raise ValueError(f"Unknown inner_product_type: {inner_product_type}")

    M_petsc = assemble(form, mat_type='aij').M.handle
    indptr, indices, data = M_petsc.getValuesCSR()

    return csr_matrix((data, indices, indptr), shape=M_petsc.getSize())


# =============================================================================
# SNAPSHOT PROCESSING
# =============================================================================

def homogenize_snapshots(
    velocity_snapshots: np.ndarray,
    parameters: np.ndarray,
    u_base: np.ndarray,
    u_control: np.ndarray,
    amplitude_index: int = 1
) -> np.ndarray:
    """
    Subtract lifting functions from velocity snapshots.

    u_homogenized = u_snapshot - u_base - amplitude * u_control

    Args:
        velocity_snapshots: (n_snapshots, n_dofs) velocity DOFs
        parameters: (n_snapshots, n_params) parameter values
        u_base: (n_dofs,) base lifting function DOFs
        u_control: (n_dofs,) control lifting function DOFs
        amplitude_index: Index of amplitude in parameters array

    Returns:
        (n_snapshots, n_dofs) homogenized velocity snapshots
    """
    n_snapshots = velocity_snapshots.shape[0]
    u_homogenized = np.zeros_like(velocity_snapshots)

    for i in range(n_snapshots):
        amp = parameters[i, amplitude_index]
        u_homogenized[i, :] = velocity_snapshots[i, :] - u_base - amp * u_control

    return u_homogenized


# =============================================================================
# POD COMPUTATION
# =============================================================================

def build_correlation_matrix(
    snapshots: np.ndarray,
    inner_product: csr_matrix = None
) -> np.ndarray:
    """
    Build correlation matrix for method of snapshots.

    C = S^T @ M @ S

    where S is (n_dofs, n_snapshots) and M is the inner product matrix.

    Args:
        snapshots: (n_dofs, n_snapshots) snapshot matrix (columns are snapshots)
        inner_product: (n_dofs, n_dofs) inner product matrix

    Returns:
        (n_snapshots, n_snapshots) correlation matrix
    """
    if inner_product is not None:
        assert snapshots.ndim == 2, "snapshots must be 2D"
        n_dofs, n_snapshots = snapshots.shape

        assert inner_product.shape == (n_dofs, n_dofs), \
            f"Inner product shape {inner_product.shape} != ({n_dofs}, {n_dofs})"

        return snapshots.T @ (inner_product @ snapshots)
    else:
        return snapshots.T @ snapshots


# def solve_eigenvalue_problem(
#     snapshots: np.ndarray,
#     correlation: np.ndarray,
#     inner_product: csr_matrix = None
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     Solve eigenvalue problem and compute POD modes.

#     Args:
#         snapshots: (n_dofs, n_snapshots) snapshot matrix
#         correlation: (n_snapshots, n_snapshots) correlation matrix
#         inner_product: (n_dofs, n_dofs) inner product matrix

#     Returns:
#         modes: (n_dofs, n_snapshots) orthonormal POD modes
#         eigenvalues: (n_snapshots,) eigenvalues in descending order
#     """
#     # Solve symmetric eigenvalue problem
#     eigenvalues, eigenvectors = la.eigh(correlation)

#     # Sort in descending order
#     idx = np.argsort(eigenvalues)[::-1]
#     eigenvalues = eigenvalues[idx]
#     eigenvectors = eigenvectors[:, idx]

#     # Compute and normalize modes
#     n_dofs, n_snapshots = snapshots.shape
#     modes = np.zeros((n_dofs, n_snapshots))


#     if inner_product is not None:
#         for i in range(n_snapshots):
#             if eigenvalues[i] > 1e-15:
#                 mode = snapshots @ eigenvectors[:, i]
#                 norm = np.sqrt(mode.T @ (inner_product @ mode))
#                 if norm > 1e-15:
#                     modes[:, i] = mode / norm
#     else:
#         for i in range(n_snapshots):
#             if eigenvalues[i] > 1e-15:
#                 mode = snapshots @ eigenvectors[:, i]
#                 norm = np.sqrt(mode.T @ mode)
#                 if norm > 1e-15:
#                     modes[:, i] = mode / norm

#     return modes, eigenvalues

def solve_eigenvalue_problem(
    snapshots: np.ndarray,
    correlation: np.ndarray,
    inner_product: csr_matrix = None,
    n_modes: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve eigenvalue problem and compute POD modes.

    Args:
        snapshots: (n_dofs, n_snapshots) snapshot matrix
        correlation: (n_snapshots, n_snapshots) correlation matrix
        inner_product: (n_dofs, n_dofs) inner product matrix (optional)
        n_modes: number of modes to compute (optional, default: all)
                 Setting this saves memory for large problems.

    Returns:
        modes: (n_dofs, n_modes) orthonormal POD modes
        eigenvalues: (n_snapshots,) eigenvalues in descending order (all of them for error estimation)
    """
    # Solve symmetric eigenvalue problem
    eigenvalues, eigenvectors = la.eigh(correlation)

    # Sort in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Determine number of modes to compute
    n_dofs, n_snapshots = snapshots.shape
    if n_modes is None:
        n_modes = n_snapshots
    else:
        n_modes = min(n_modes, n_snapshots)

    # Compute and normalize only n_modes modes (memory efficient)
    modes = np.zeros((n_dofs, n_modes))

    if inner_product is not None:
        for i in range(n_modes):
            if eigenvalues[i] > 1e-15:
                mode = snapshots @ eigenvectors[:, i]
                norm = np.sqrt(mode.T @ (inner_product @ mode))
                if norm > 1e-15:
                    modes[:, i] = mode / norm
    else:
        for i in range(n_modes):
            if eigenvalues[i] > 1e-15:
                mode = snapshots @ eigenvectors[:, i]
                norm = np.linalg.norm(mode)
                if norm > 1e-15:
                    modes[:, i] = mode / norm

    return modes, eigenvalues

def truncate_modes(
    modes: np.ndarray,
    eigenvalues: np.ndarray,
    n_modes: int = None,
    energy_threshold: float = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Truncate modes by number or energy threshold.

    Args:
        modes: (n_dofs, n_total) all modes
        eigenvalues: (n_total,) all eigenvalues
        n_modes: Maximum number of modes to keep
        energy_threshold: Minimum energy fraction to retain (e.g., 0.9999)

    Returns:
        modes_truncated: (n_dofs, n_retained) truncated modes
        eigenvalues_truncated: (n_retained,) truncated eigenvalues
    """
    n_total = len(eigenvalues)
    total_energy = np.sum(eigenvalues)

    if energy_threshold is not None and total_energy > 0:
        cumulative = np.cumsum(eigenvalues) / total_energy
        n_energy = np.searchsorted(cumulative, energy_threshold) + 1

        if n_modes is not None:
            n_truncated = min(n_modes, n_energy)
        else:
            n_truncated = n_energy
    elif n_modes is not None:
        n_truncated = min(n_modes, n_total)
    else:
        n_truncated = n_total

    modes_truncated = modes[:, :n_truncated]
    eigenvalues_truncated = eigenvalues[:n_truncated]

    energy_retained = np.sum(eigenvalues_truncated) / total_energy * 100
    print(f"  Truncated: {n_total} → {n_truncated} modes ({energy_retained:.10f}% energy)")

    return modes_truncated, eigenvalues_truncated


def compute_pod_modes(
    snapshots: np.ndarray,
    inner_product: csr_matrix,
    n_modes: int = None,
    energy_threshold: float = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute POD modes from snapshots.

    Args:
        snapshots: (n_dofs, n_snapshots) snapshot matrix
        inner_product: (n_dofs, n_dofs) inner product matrix
        n_modes: Number of modes to retain (optional)
        energy_threshold: Energy fraction to retain, e.g., 0.9999 (optional)

    Returns:
        modes: (n_dofs, n_retained) POD modes
        eigenvalues: (n_retained,) eigenvalues
    """
    C = build_correlation_matrix(snapshots, inner_product)
    modes, eigenvalues = solve_eigenvalue_problem(snapshots, C, inner_product)

    return truncate_modes(
        modes, eigenvalues,
        n_modes=n_modes,
        energy_threshold=energy_threshold
    )


# =============================================================================
# SUPREMIZER COMPUTATION
# =============================================================================

def compute_supremizer_snapshots(
    pressure_snapshots: np.ndarray,
    problem,
    boundary_markers: List[int] = None
) -> np.ndarray:
    """
    Compute supremizer snapshots for inf-sup stability.

    For each pressure snapshot p, solve:
        (grad(s), grad(v)) = -(p, div(v))  for all v
    with homogeneous Dirichlet BCs.

    Args:
        pressure_snapshots: (n_dofs_p, n_snapshots) pressure modes or snapshots
        problem: NavierStokesProblem
        boundary_markers: List of boundary markers for homogeneous BCs
                         Default: [1, 3, 4, 5, 6] (pinball mesh)

    Returns:
        (n_dofs_u, n_snapshots) supremizer snapshots
    """
    assert pressure_snapshots.ndim == 2, "Input must be 2D"
    n_dofs_p, n_snapshots = pressure_snapshots.shape

    V = problem.velocity_space
    Q = problem.pressure_space

    print(f"  Computing {n_snapshots} supremizer snapshots...")

    # Setup variational problem
    u_trial = TrialFunction(V)
    v_test = TestFunction(V)
    a_form = inner(grad(u_trial), grad(v_test)) * dx

    # Homogeneous BCs
    if boundary_markers is None:
        boundary_markers = [1, 3, 4, 5, 6]  # Default for pinball

    bcs = [DirichletBC(V, Constant((0.0, 0.0)), marker) for marker in boundary_markers]

    # Assemble matrix once

    A = assemble(a_form, bcs=bcs)
    # Convert pressure snapshots to functions
    p_functions = dofs_to_functions(pressure_snapshots.T, Q, name_prefix="p")

    # Solve for each pressure
    s = Function(V)
    supremizer_list = []

    for i, p_func in enumerate(p_functions):
        rhs = assemble(-p_func * div(v_test) * dx)
        solve(A, s, rhs)
        supremizer_list.append(s.copy(deepcopy=True))

        if (i + 1) % 10 == 0 or i == n_snapshots - 1:
            print(f"    Computed {i + 1}/{n_snapshots}")

    # Convert to numpy array
    return functions_to_dofs(supremizer_list)


def compute_supremizer_modes(
    pressure_snapshots: np.ndarray,
    problem,
    inner_product: csr_matrix,
    n_modes: int = None,
    energy_threshold: float = None,
    boundary_markers: List[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute supremizer modes via POD of supremizer snapshots.

    Args:
        pressure_modes: (n_dofs_p, n_pressure_modes) pressure POD modes
        problem: NavierStokesProblem
        inner_product: (n_dofs_u, n_dofs_u) velocity inner product
        n_modes: Number of supremizer modes to retain
        energy_threshold: Energy threshold for truncation
        boundary_markers: Boundary markers for homogeneous BCs

    Returns:
        supremizer_modes: (n_dofs_u, n_modes) supremizer POD modes
        supremizer_eigenvalues: (n_modes,) eigenvalues
    """
    # Compute supremizer snapshots
    sup_snapshots = compute_supremizer_snapshots(
        pressure_snapshots, problem, boundary_markers
    )

    # POD of supremizer snapshots
    print("  Computing POD of supremizer snapshots...")
    return compute_pod_modes(
        sup_snapshots, inner_product,
        n_modes=n_modes,
        energy_threshold=energy_threshold
    )

def supremizer_enrichment(
    problem,
    pressure_modes: np.ndarray,
    inner_product_type: str,
) -> np.ndarray:

    import petsc4py.PETSc as PETSc
    from firedrake import TrialFunction, TestFunction, assemble, div, dx

    V = problem.velocity_space
    Q = problem.pressure_space

    # Assemble matrices
    u_trial = TrialFunction(V)
    v_test = TestFunction(V)
    p_trial = TrialFunction(Q)

    a_form = inner(grad(u_trial), grad(v_test)) * dx

    # Homogeneous BCs
    boundary_markers = [1, 3, 4, 5, 6]  # Default for pinball

    bcs = [DirichletBC(V, Constant((0.0, 0.0)), marker) for marker in boundary_markers]
    boundary_dofs = np.unique(np.concatenate([
        DirichletBC(V, Constant((0.0, 0.0)), m).nodes
        for m in [1, 3, 4, 5, 6]
    ]))

    # Assemble matrix once

    A = assemble(a_form, bcs=bcs)
    B_T = assemble(-p_trial * div(v_test) * dx, mat_type="aij")
    p_basis_functions = dofs_to_functions(pressure_modes.T, Q)
    s = Function(V)
    supremizer_list = []

    for i, p_func in enumerate(p_basis_functions):
        rhs = assemble(-p_func * div(v_test) * dx)
        solve(A, s, rhs)
        supremizer_list.append(s.copy(deepcopy=True))

    # Convert to numpy array
    return functions_to_dofs(supremizer_list)

# The one below works but uses the scipy solver
def supremizer_enrichment_scipy(problem, pressure_modes, inner_product_type):


    V = problem.velocity_space
    Q = problem.pressure_space
    u_trial = TrialFunction(V)

    v_test = TestFunction(V)
    p_trial = TrialFunction(Q)
    a_form = inner(grad(u_trial), grad(v_test)) * dx

    # Homogeneous BCs
    boundary_markers = [1, 3, 4, 5, 6]  # Default for pinball

    bcs = [DirichletBC(V, Constant((0.0, 0.0)), marker) for marker in boundary_markers]


    boundary_nodes = np.unique(np.concatenate([
        DirichletBC(V, Constant((0.0, 0.0)), m).nodes
        for m in [1, 3, 4, 5, 6]
    ]))
    boundary_dofs = np.sort(np.concatenate([2 * boundary_nodes, 2 * boundary_nodes + 1]))

    # Assemble matrix once

    M = assemble(a_form,mat_type="aij")

    B_T = assemble(-p_trial * div(v_test) * dx, mat_type="aij")
    X = csr_matrix(M.M.handle.getValuesCSR()[::-1], shape=M.M.handle.getSize())
    Bt = csr_matrix(B_T.M.handle.getValuesCSR()[::-1], shape=B_T.M.handle.getSize())

    X = X.tolil()
    X[boundary_dofs, :] = 0
    X[boundary_dofs, boundary_dofs] = 1.0
    X = X.tocsr()

    rhs = Bt @ pressure_modes  # (N_u, n_p)
    rhs[boundary_dofs, :] = 0.0  # enforce homogeneous Dirichlet

    n_modes = pressure_modes.shape[1]
    supremizers = np.zeros_like(rhs)

    for i in range(n_modes):
        supremizers[:, i] = spsolve(X, rhs[:, i])
    return supremizers



# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def compute_pod_basis(
    velocity_snapshots: np.ndarray,
    pressure_snapshots: np.ndarray,
    problem,
    # n_velocity_modes: int = None,
    # n_pressure_modes: int = None,
    # n_supremizer_modes: int = None,
    velocity_energy_threshold: float = None,
    pressure_energy_threshold: float = None,
    inner_product_type: Literal['L2', 'H1', 'H1_semi'] = 'H1_semi',
    compute_supremizer: bool = True,
    parameters: np.ndarray = None,
    u_base: np.ndarray = None,
    u_control: np.ndarray = None,
    amplitude_index: int = 1,
    boundary_markers: List[int] = None
) -> PODBasis:
    """
    Compute complete POD basis from snapshots.

    Main entry point for POD computation. Handles:
    - Snapshot homogenization (if lifting functions provided)
    - Inner product assembly
    - Velocity and pressure POD
    - Optional supremizer enrichment

    Args:
        velocity_snapshots: (n_snapshots, n_dofs_u) raw velocity DOFs
        pressure_snapshots: (n_snapshots, n_dofs_p) pressure DOFs
        problem: NavierStokesProblem
        n_velocity_modes: Number of velocity modes to retain
        n_pressure_modes: Number of pressure modes to retain
        n_supremizer_modes: Number of supremizer modes to retain
        velocity_energy_threshold: Energy threshold for velocity
        pressure_energy_threshold: Energy threshold for pressure
        supremizer_energy_threshold: Energy threshold for supremizer
        inner_product_type: 'L2', 'H1', or 'H1_semi'
        compute_supremizer: Whether to compute supremizer enrichment
        parameters: (n_snapshots, n_params) for homogenization
        u_base: (n_dofs_u,) base lifting function
        u_control: (n_dofs_u,) control lifting function
        amplitude_index: Index of amplitude in parameters
        boundary_markers: Boundary markers for supremizer BCs

    Returns:
        PODBasis with all modes and eigenvalues
    """
    print("=" * 60)
    print("POD BASIS COMPUTATION")
    print("=" * 60)

    n_snapshots = velocity_snapshots.shape[0]
    print(f"  Snapshots: {n_snapshots}")
    print(f"  Velocity DOFs: {velocity_snapshots.shape[1]}")
    print(f"  Pressure DOFs: {pressure_snapshots.shape[1]}")
    print(f"  Inner product: {inner_product_type}")

    # Homogenize if lifting functions provided
    if u_base is not None and u_control is not None and parameters is not None:
        print("\nHomogenizing velocity snapshots...")
        velocity_snapshots = homogenize_snapshots(
            velocity_snapshots, parameters, u_base, u_control, amplitude_index
        )

    # Transpose to (n_dofs, n_snapshots) for POD
    U = velocity_snapshots.T
    P = pressure_snapshots.T

    # Assemble inner product matrices
    print("\nAssembling inner product matrices...")
    M_u = assemble_inner_product_matrix(problem.velocity_space, inner_product_type)
    M_p = assemble_inner_product_matrix(problem.pressure_space, 'L2')
    print(f"  M_u: {M_u.shape}")
    print(f"  M_p: {M_p.shape}")

    # Velocity POD
    print("\nComputing velocity POD...")
    velocity_modes, velocity_eigenvalues = compute_pod_modes(
        U, M_u,
        # n_modes=n_velocity_modes,
        energy_threshold=velocity_energy_threshold
    )

    # Pressure POD
    print("\nComputing pressure POD...")
    pressure_modes, pressure_eigenvalues = compute_pod_modes(
        P, M_p,
        # n_modes=n_pressure_modes,
        energy_threshold=pressure_energy_threshold
    )

    # Supremizer
    supremizer_modes = None
    supremizer_eigenvalues = None
    total_supremizer_energy = 0.0
    if compute_supremizer:
        print("\nComputing supremizer enrichment...")
        # supremizer_modes, supremizer_eigenvalues = compute_supremizer_modes(
        #     P,
        #     problem,
        #     M_u,
        #     # n_modes=n_supremizer_modes,
        #     energy_threshold=supremizer_energy_threshold,
        #     boundary_markers=boundary_markers
        # )
        # total_supremizer_energy = np.sum(supremizer_eigenvalues)
        supremizer_modes =  supremizer_enrichment(
            problem,
            pressure_modes,
            inner_product_type=inner_product_type
        )
        supremizer_eigenvalues = None
        total_supremizer_energy = None

    print(f"  The lenght of vel eigenvalue is: {len(velocity_eigenvalues)}")
    total_velocity_energy = np.sum(velocity_eigenvalues)
    total_pressure_energy = np.sum(pressure_eigenvalues)
    # Build basis
    basis = PODBasis(
        velocity_modes=velocity_modes,
        pressure_modes=pressure_modes,
        velocity_eigenvalues=velocity_eigenvalues,
        pressure_eigenvalues=pressure_eigenvalues,
        supremizer_modes=supremizer_modes,
        supremizer_eigenvalues=supremizer_eigenvalues,
        inner_product_type=inner_product_type,
        metadata={
            'n_snapshots': n_snapshots,
            'homogenized': u_base is not None,
            'total_velocity_energy': total_velocity_energy,
            'total_pressure_energy': total_pressure_energy,
            'total_supremizer_energy': total_supremizer_energy,
        }
    )
    # truncated_basis = basis.truncate(n_velocity = n_velocity_modes, n_pressure = n_pressure_modes, n_supremizer = n_supremizer_modes)

    # Summary

    print("\n" + "=" * 60)
    print("POD BASIS COMPLETE")
    print("=" * 60)

    if basis.supremizer_modes is not None:
        print(f"  Supremizer modes: {basis.n_supremizer_modes}")
    print(f"  Enriched velocity: {basis.n_enriched_velocity_modes} modes")

    return basis


# =============================================================================
# UTILITIES
# =============================================================================

def project_to_basis(
    snapshots: np.ndarray,
    modes: np.ndarray,
    inner_product: csr_matrix
) -> np.ndarray:
    """
    Project snapshots onto POD basis.

    coeffs = modes^T @ M @ snapshots

    Args:
        snapshots: (n_dofs, n_snapshots) snapshots to project
        modes: (n_dofs, n_modes) POD basis
        inner_product: (n_dofs, n_dofs) inner product matrix

    Returns:
        (n_modes, n_snapshots) projection coefficients
    """
    return modes.T @ (inner_product @ snapshots)


def reconstruct_from_basis(
    coeffs: np.ndarray,
    modes: np.ndarray
) -> np.ndarray:
    """
    Reconstruct snapshots from POD coefficients.

    reconstruction = modes @ coeffs

    Args:
        coeffs: (n_modes, n_snapshots) or (n_modes,) coefficients
        modes: (n_dofs, n_modes) POD basis

    Returns:
        (n_dofs, n_snapshots) or (n_dofs,) reconstructed snapshots
    """
    return modes @ coeffs


def check_orthonormality(
    modes: np.ndarray,
    inner_product: csr_matrix,
    tol: float = 1e-7
) -> Tuple[bool, float]:
    """
    Check if modes are orthonormal with respect to inner product.

    Args:
        modes: (n_dofs, n_modes) mode matrix
        inner_product: (n_dofs, n_dofs) inner product matrix
        tol: Tolerance for deviation from identity

    Returns:
        is_orthonormal: True if orthonormal within tolerance
        max_error: Maximum deviation from identity
    """
    G = modes.T @ (inner_product @ modes)
    I = np.eye(G.shape[0])
    max_error = np.max(np.abs(G - I))
    print(f"Number of modes: {modes.shape[1]}")
    print(f"Max diagonal deviation: {np.max(np.abs(np.diag(G) - 1)):.2e}")
    print(f"Max off-diagonal: {np.max(np.abs(G - np.diag(np.diag(G)))):.2e}")
    print(f"Max error from identity: {max_error:.2e}")
    print(f"Orthonormal: {max_error < tol}")

    return max_error < tol, max_error


def compute_reconstruction_error(
    snapshots: np.ndarray,
    modes: np.ndarray,
    inner_product: csr_matrix,
    n_modes_list: List[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute reconstruction error for varying number of modes.

    Args:
        snapshots: (n_dofs, n_snapshots) original snapshots
        modes: (n_dofs, n_total_modes) POD basis
        inner_product: (n_dofs, n_dofs) inner product matrix
        n_modes_list: List of mode counts to test (default: 1 to n_total)

    Returns:
        n_modes_array: Array of mode counts
        errors: Mean relative reconstruction error for each count
    """
    n_total = modes.shape[1]

    if n_modes_list is None:
        n_modes_list = list(range(1, n_total + 1))

    # Compute original norms
    original_norm_sq = np.sum(snapshots * (inner_product @ snapshots), axis=0)

    errors = []

    for n in n_modes_list:
        coeffs = project_to_basis(snapshots, modes[:, :n], inner_product)
        recon = reconstruct_from_basis(coeffs, modes[:, :n])

        error = snapshots - recon
        error_norm_sq = np.sum(error * (inner_product @ error), axis=0)

        valid = original_norm_sq > 1e-14
        rel_error = np.sqrt(error_norm_sq[valid] / original_norm_sq[valid])
        errors.append(np.mean(rel_error))

    return np.array(n_modes_list), np.array(errors)
# =============================================================================
# SUPREMIZER ENRICHMENT (Gram-Schmidt)
# =============================================================================

def _gram_schmidt_orthonormalize(new_modes, existing_modes, M):
    """
    Orthonormalize new_modes against existing_modes w.r.t. inner product M.

    Uses modified Gram-Schmidt with two passes for numerical stability.

    Args:
        new_modes:      (n_dofs, n_new) modes to orthonormalize
        existing_modes: (n_dofs, n_existing) already orthonormal modes
        M:              (n_dofs, n_dofs) inner product matrix (sparse)

    Returns:
        (n_dofs, n_kept) orthonormalized modes (zero modes dropped)
    """
    all_modes = existing_modes.copy()
    result = []

    for i in range(new_modes.shape[1]):
        v = new_modes[:, i].copy()

        # Orthogonalize against all existing + previously added modes
        # Two passes for stability
        for _ in range(2):
            for j in range(all_modes.shape[1]):
                coeff = float(all_modes[:, j] @ (M @ v))
                v -= coeff * all_modes[:, j]

        # Normalize
        norm = np.sqrt(float(v @ (M @ v)))
        if norm > 1e-10:
            v /= norm
            result.append(v)
            all_modes = np.column_stack([all_modes, v])

    if len(result) == 0:
        return np.zeros((new_modes.shape[0], 0))

    return np.column_stack(result)


def enrich_velocity_basis(
    velocity_modes: np.ndarray,
    supremizer_modes: np.ndarray,
    inner_product: csr_matrix,
) -> np.ndarray:
    """
    Enrich velocity basis with supremizer modes via Gram-Schmidt.

    Orthonormalizes supremizer_modes against velocity_modes, then
    appends the non-zero result.

    Args:
        velocity_modes:   (n_dofs, N_v) existing velocity POD modes
        supremizer_modes: (n_dofs, N_s) raw supremizer modes
        inner_product:    (n_dofs, n_dofs) inner product matrix

    Returns:
        (n_dofs, N_v + N_kept) enriched velocity basis
    """
    s_ortho = _gram_schmidt_orthonormalize(
        supremizer_modes, velocity_modes, inner_product
    )

    n_kept = s_ortho.shape[1]
    n_vel = velocity_modes.shape[1]

    print(f"  Supremizer enrichment: {n_vel} velocity + {n_kept} supremizer = {n_vel + n_kept} total")

    return np.hstack([velocity_modes, s_ortho])
