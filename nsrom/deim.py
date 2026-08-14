# deim.py
"""
DEIM (Discrete Empirical Interpolation Method) for ROM acceleration.

Provides:
- Core algorithms: POD basis, DEIM index selection, interpolation matrices
- Snapshot assembly: convective term and Jacobian snapshots
- Basis I/O with metadata for cache validation
- Testing utilities
"""

import numpy as np
from scipy import linalg as la
from pathlib import Path
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
from firedrake import Function, TestFunction, TrialFunction, inner, grad, dx, assemble


# =============================================================================
# Core Algorithms
# =============================================================================

def compute_deim_pod_basis(
    snapshots: np.ndarray,
    n_modes: Optional[int] = None,
    tol: float = 1e-15
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a plain (Euclidean) POD basis via the method of snapshots.

    DEIM-internal helper for the nonlinear-term (F/J) bases: single field,
    NO inner-product weighting, NO supremizer, NO homogenization. Distinct
    from nsrom.rom.pod.compute_pod_basis, which builds the full inner-product-
    weighted, supremizer-enriched ROM basis and returns a PODBasis object.

    Args:
        snapshots: (n_dofs, n_snapshots) array
        n_modes: Number of modes to compute (None = all)
        tol: Tolerance for eigenvalue cutoff

    Returns:
        modes: (n_dofs, n_modes) orthonormal basis
        eigenvalues: (n_modes,) sorted descending
    """
    n_dofs, n_snapshots = snapshots.shape
    n_modes = n_modes or n_snapshots

    C = snapshots.T @ snapshots
    eigenvalues, eigenvectors = la.eigh(C)

    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Compute modes
    modes = np.zeros((n_dofs, n_modes))
    for i in range(n_modes):
        if eigenvalues[i] > tol:
            U = snapshots @ eigenvectors[:, i]
            norm = np.linalg.norm(U)
            if norm > tol:
                modes[:, i] = U / norm

    return modes, eigenvalues[:n_modes]

# def deim_algorithm(U_m: np.ndarray, tol: float = 1e-12) -> np.ndarray:
#     """
#     Apply DEIM algorithm to select interpolation indices.

#     Args:
#         U_m: (n_dofs, m) POD basis matrix
#         tol: Tolerance for detecting singular matrices

#     Returns:
#         indices: (m,) interpolation indices
#     """
#     n_dofs, m = U_m.shape
#     indices = np.zeros(m, dtype=int)

#     # First index: maximum of first mode
#     indices[0] = np.argmax(np.abs(U_m[:, 0]))

#     for i in range(1, m):
#         U_i = U_m[:, :i]

#         # Get the submatrix at selected indices
#         U_sub = U_i[indices[:i], :]
#         rhs = U_m[indices[:i], i]

#         # Check condition number before solving
#         cond = np.linalg.cond(U_sub)
#         if cond > 1.0 / tol:
#             print(f"  Warning: DEIM iteration {i}, condition number = {cond:.2e}")
#             # Use least squares instead of direct solve
#             c, _, _, _ = np.linalg.lstsq(U_sub, rhs, rcond=tol)
#         else:
#             c = np.linalg.solve(U_sub, rhs)

#         r = U_m[:, i] - U_i @ c

#         # Find maximum that's not already selected
#         r_abs = np.abs(r)
#         for idx in indices[:i]:
#             r_abs[idx] = -1  # Exclude already selected indices

#         new_idx = np.argmax(r_abs)

#         # Check if residual is too small (mode is in span of previous modes)
#         if r_abs[new_idx] < tol:
#             print(f"  Warning: DEIM iteration {i}, residual too small ({r_abs[new_idx]:.2e})")
#             print(f"           Mode {i} may be linearly dependent. Selecting largest unused DOF.")
#             # Fall back to selecting the DOF with largest value in this mode
#             mode_abs = np.abs(U_m[:, i])
#             for idx in indices[:i]:
#                 mode_abs[idx] = -1
#             new_idx = np.argmax(mode_abs)

#         indices[i] = new_idx

#     # Check for duplicates
#     unique_indices = np.unique(indices)
#     if len(unique_indices) != len(indices):
#         n_duplicates = len(indices) - len(unique_indices)
#         print(f"  Warning: {n_duplicates} duplicate DEIM indices detected!")

#     return indices
def deim_algorithm(U_m: np.ndarray) -> np.ndarray:
    """
    Apply DEIM algorithm to select interpolation indices.

    Args:
        U_m: (n_dofs, m) POD basis matrix

    Returns:
        indices: (m,) interpolation indices
    """
    n_dofs, m = U_m.shape
    indices = np.zeros(m, dtype=int)
    indices[0] = np.argmax(np.abs(U_m[:, 0]))

    for i in range(1, m):
        U_i = U_m[:, :i]
        c = np.linalg.solve(U_i[indices[:i], :], U_m[indices[:i], i])
        r = U_m[:, i] - U_i @ c
        indices[i] = np.argmax(np.abs(r))

    return indices


def build_deim_interpolation_matrix(
    U_m: np.ndarray,
    indices: np.ndarray
) -> np.ndarray:
    """
    Build DEIM interpolation matrix B = U_m @ inv(U_m[indices, :]).

    Args:
        U_m: (n_dofs, m) POD basis
        indices: (m,) interpolation indices

    Returns:
        B: (n_dofs, m) interpolation matrix
    """
    U_p = U_m[indices, :]
    return U_m @ np.linalg.inv(U_p)


# =============================================================================
# Snapshot Assembly
# =============================================================================

def assemble_convective_snapshots(
    velocity_functions: List[Function],
    velocity_space,
    log_interval: int = 50
) -> np.ndarray:
    """
    Assemble convective term snapshots: F(u) = (u · ∇)u.

    Args:
        velocity_functions: List of velocity Function objects
        velocity_space: Firedrake velocity function space
        log_interval: Print progress every N snapshots

    Returns:
        F_snapshots: (n_dofs, n_snapshots) array
    """
    v = TestFunction(velocity_space)
    n_snapshots = len(velocity_functions)
    n_dofs = velocity_space.dim()

    F_snapshots = np.zeros((n_dofs, n_snapshots))
    print(f"Assembling {n_snapshots} convective snapshots...")

    for i, u in enumerate(velocity_functions):
        if (i + 1) % log_interval == 0 or i == 0:
            print(f"  Snapshot {i+1}/{n_snapshots}")
        F_conv = inner(grad(u) * u, v) * dx
        F_snapshots[:, i] = assemble(F_conv).dat.data_ro.flatten().copy()

    return F_snapshots


def assemble_jacobian_snapshots(
    velocity_functions: List[Function],
    velocity_space,
    log_interval: int = 50
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Assemble Jacobian snapshots as flattened CSR data arrays.

    Args:
        velocity_functions: List of velocity Function objects
        velocity_space: Firedrake velocity function space
        log_interval: Print progress every N snapshots

    Returns:
        J_snapshots: (nnz, n_snapshots) array of CSR data values
        csr_pattern: (indptr, col_indices) tuple for reconstructing CSR matrices
    """
    v = TestFunction(velocity_space)
    u_trial = TrialFunction(velocity_space)
    n_snapshots = len(velocity_functions)

    # Get sparsity pattern from reference
    u0 = velocity_functions[0]
    J_form = inner(grad(u_trial) * u0, v) * dx + inner(grad(u0) * u_trial, v) * dx
    J_ref = assemble(J_form).M.handle
    indptr, col_indices, data_ref = J_ref.getValuesCSR()
    nnz = len(data_ref)

    print(f"Assembling {n_snapshots} Jacobian snapshots (nnz={nnz})...")
    J_snapshots = np.zeros((nnz, n_snapshots))

    for i, u in enumerate(velocity_functions):
        if (i + 1) % log_interval == 0 or i == 0:
            print(f"  Snapshot {i+1}/{n_snapshots}")
        J_form = inner(grad(u_trial) * u, v) * dx + inner(grad(u) * u_trial, v) * dx
        _, _, data = assemble(J_form).M.handle.getValuesCSR()
        J_snapshots[:, i] = data

    return J_snapshots, (indptr, col_indices)


# =============================================================================
# Basis I/O with Metadata
# =============================================================================

def compute_and_save_basis(
    velocity_functions: List[Function],
    velocity_space,
    output_path: str | Path,
    n_modes_F: Optional[int] = None,
    n_modes_J: Optional[int] = None,
    snapshot_dir: Optional[str] = None,
):
    """
    Compute DEIM basis for convective and Jacobian terms, then save.

    Args:
        velocity_functions: List of velocity Function objects
        velocity_space: Firedrake velocity function space
        output_path: Path to save the basis (.npz)
        n_modes_F: Number of convective modes (None = full)
        n_modes_J: Number of Jacobian modes (None = full)
        snapshot_dir: Source snapshot directory (stored as metadata)
    """
    print("=" * 60)
    print("DEIM BASIS COMPUTATION")
    print("=" * 60)
    print(f"Number of snapshots: {len(velocity_functions)}")
    print(f"Output: {output_path}")

    # Convective
    print("\n--- Convective Term ---")
    F_snapshots = assemble_convective_snapshots(velocity_functions, velocity_space)
    modes_F, eigenvalues_F = compute_deim_pod_basis(F_snapshots, n_modes_F)
    actual_n_modes_F = modes_F.shape[1]
    print(f"Convective: {actual_n_modes_F} modes")
    del F_snapshots

    # Jacobian
    print("\n--- Jacobian Term ---")
    J_snapshots, (indptr, col_indices) = assemble_jacobian_snapshots(
        velocity_functions, velocity_space
    )
    modes_J, eigenvalues_J = compute_deim_pod_basis(J_snapshots, n_modes_J)
    actual_n_modes_J = modes_J.shape[1]
    n_dofs = velocity_space.dim()
    print(f"Jacobian: {actual_n_modes_J} modes, nnz={len(col_indices)}")
    del J_snapshots

    # Save with metadata
    np.savez_compressed(
        output_path,
        # Basis data
        modes_F=modes_F,
        eigenvalues_F=eigenvalues_F,
        modes_J=modes_J,
        eigenvalues_J=eigenvalues_J,
        indptr=indptr,
        col_indices=col_indices,
        n_dofs=n_dofs,
        # Metadata for cache validation
        snapshot_dir=snapshot_dir if snapshot_dir else "",
        stored_n_modes_F=actual_n_modes_F if n_modes_F is not None else -1,  # -1 means full
        stored_n_modes_J=actual_n_modes_J if n_modes_J is not None else -1,
    )

    print(f"\nSaved DEIM basis to {output_path}")
    print(f"  Convective modes: {actual_n_modes_F} {'(full)' if n_modes_F is None else ''}")
    print(f"  Jacobian modes: {actual_n_modes_J} {'(full)' if n_modes_J is None else ''}")


def load_basis(path: str | Path) -> dict:
    """
    Load saved DEIM basis.

    Args:
        path: Path to saved basis (.npz)

    Returns:
        Dictionary with keys:
        - modes_F, eigenvalues_F: Convective POD basis
        - modes_J, eigenvalues_J: Jacobian POD basis
        - indptr, col_indices: CSR sparsity pattern
        - n_dofs: Number of velocity DOFs
        - metadata: dict with snapshot_dir, stored_n_modes_F, stored_n_modes_J
    """
    data = np.load(path)

    return {
        'modes_F': data['modes_F'],
        'eigenvalues_F': data['eigenvalues_F'],
        'modes_J': data['modes_J'],
        'eigenvalues_J': data['eigenvalues_J'],
        'indptr': data['indptr'],
        'col_indices': data['col_indices'],
        'n_dofs': int(data['n_dofs']),
        'metadata': {
            'snapshot_dir': str(data['snapshot_dir']),
            'stored_n_modes_F': int(data['stored_n_modes_F']),
            'stored_n_modes_J': int(data['stored_n_modes_J']),
        }
    }


def load_basis_metadata(path: str | Path) -> dict:
    """
    Load only metadata from saved basis (for quick cache checks).

    Args:
        path: Path to saved basis (.npz)

    Returns:
        Dictionary with snapshot_dir, stored_n_modes_F, stored_n_modes_J
    """
    data = np.load(path)

    return {
        'snapshot_dir': str(data['snapshot_dir']),
        'stored_n_modes_F': int(data['stored_n_modes_F']),
        'stored_n_modes_J': int(data['stored_n_modes_J']),
    }


# =============================================================================
# Testing & Analysis Utilities
# =============================================================================

def test_deim_reconstruction(
    snapshots: np.ndarray,
    B: np.ndarray,
    indices: np.ndarray,
    name: str = ""
) -> np.ndarray:
    """
    Test DEIM reconstruction accuracy on snapshots.

    Args:
        snapshots: (n_dofs, n_snapshots) original data
        B: (n_dofs, m) DEIM interpolation matrix
        indices: (m,) interpolation indices
        name: Label for printing

    Returns:
        rel_errors: (n_snapshots,) relative errors
    """
    n_snapshots = snapshots.shape[1]
    rel_errors = np.zeros(n_snapshots)

    for i in range(n_snapshots):
        f_true = snapshots[:, i]
        f_approx = B @ f_true[indices]
        norm_true = np.linalg.norm(f_true)
        if norm_true > 1e-15:
            rel_errors[i] = np.linalg.norm(f_true - f_approx) / norm_true

    print(f"DEIM Reconstruction {name} (m={len(indices)}):")
    print(f"  Mean relative error: {np.mean(rel_errors):.6e}")
    print(f"  Max relative error:  {np.max(rel_errors):.6e}")

    return rel_errors


def test_on_new_field(
    f_true: np.ndarray,
    B: np.ndarray,
    indices: np.ndarray,
    name: str = ""
) -> float:
    """
    Test DEIM approximation on a new field.

    Args:
        f_true: (n_dofs,) true field values
        B: (n_dofs, m) DEIM interpolation matrix
        indices: (m,) interpolation indices
        name: Label for printing

    Returns:
        rel_error: Relative error
    """
    f_approx = B @ f_true[indices]
    error = np.linalg.norm(f_true - f_approx)
    rel_error = error / np.linalg.norm(f_true)
    print(f"Test {name}: abs={error:.6e}, rel={rel_error:.6e}")
    return rel_error


def check_interpolation_property(B: np.ndarray, indices: np.ndarray) -> float:
    """
    Verify B[indices, :] ≈ I (interpolation property).

    Args:
        B: (n_dofs, m) DEIM interpolation matrix
        indices: (m,) interpolation indices

    Returns:
        error: ||B[indices, :] - I||
    """
    error = np.linalg.norm(B[indices, :] - np.eye(len(indices)))
    status = "✓" if error < 1e-10 else "✗"
    print(f"Interpolation property: ||B[p,:] - I|| = {error:.6e} {status}")
    return error


def analyze_deim_indices(
    indices: np.ndarray,
    n_dofs: int,
    save_path: Optional[str] = None
):
    """
    Analyze and optionally visualize DEIM index distribution.

    Args:
        indices: (m,) interpolation indices
        n_dofs: Total number of DOFs
        save_path: Optional path to save histogram plot
    """
    n_unique = len(np.unique(indices))
    print(f"\nDEIM Index Analysis:")
    print(f"  Count: {len(indices)}, Unique: {n_unique}")
    print(f"  Range: [{indices.min()}, {indices.max()}] / {n_dofs}")
    print(f"  Coverage: {100 * len(indices) / n_dofs:.2f}%")

    if n_unique != len(indices):
        print(f"  ⚠ WARNING: {len(indices) - n_unique} duplicate indices!")

    if save_path:
        plt.figure(figsize=(10, 3))
        plt.hist(indices, bins=50, edgecolor='black')
        plt.xlabel('DOF index')
        plt.ylabel('Count')
        plt.title('DEIM interpolation indices distribution')
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  Saved histogram to {save_path}")


def build_deim_approximation(
    snapshots: np.ndarray,
    n_modes: int,
    name: str = ""
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Complete DEIM pipeline: POD → DEIM → interpolation matrix.

    Args:
        snapshots: (n_dofs, n_snapshots) data matrix
        n_modes: Number of POD modes / DEIM points
        name: Label for printing

    Returns:
        modes: (n_dofs, n_modes) POD basis
        indices: (n_modes,) DEIM indices
        B: (n_dofs, n_modes) interpolation matrix
        eigenvalues: (n_modes,) POD eigenvalues
    """
    print(f"\n{'='*50}")
    print(f"Building DEIM approximation for {name}")
    print(f"{'='*50}")

    modes, eigenvalues = compute_deim_pod_basis(snapshots, n_modes)
    print(f"POD: {n_modes} modes computed")

    indices = deim_algorithm(modes)
    B = build_deim_interpolation_matrix(modes, indices)
    print(f"DEIM: {len(indices)} interpolation points selected")

    check_interpolation_property(B, indices)

    return modes, indices, B, eigenvalues


# =============================================================================
# Main (for standalone testing)
# =============================================================================

def main():
    """Standalone test of DEIM functionality."""
    from .navier_stokes import setup_navier_stokes_problem, load_solution
    from .snapshot_collection import load_snapshot_dofs
    from .helper_functions import dofs_to_functions

    # Setup
    problem = setup_navier_stokes_problem(
        mesh_file='mesh/mid_pinball.msh',
        reynolds_init=100.0,
        amplitude_init=0.0
    )

    # Load snapshots
    snapshot_dir = "one_branch_dofs"
    snapshot_data = load_snapshot_dofs(snapshot_dir)
    velocity_snapshots = snapshot_data['velocity']
    velocity_functions = dofs_to_functions(velocity_snapshots, problem.velocity_space)
    print(f"Loaded {len(velocity_functions)} velocity snapshots from {snapshot_dir}")

    # Compute and save basis
    compute_and_save_basis(
        velocity_functions,
        problem.velocity_space,
        output_path="deim_basis.npz",
        n_modes_F=None,  # full
        n_modes_J=200,   # truncated
        snapshot_dir=snapshot_dir,
    )

    # Test loading
    print("\n--- Testing load functions ---")
    meta = load_basis_metadata("deim_basis.npz")
    print(f"Metadata: {meta}")

    basis = load_basis("deim_basis.npz")
    print(f"Loaded basis: F modes = {basis['modes_F'].shape}, J modes = {basis['modes_J'].shape}")


if __name__ == "__main__":
    main()