"""
Matrix Definiteness Diagnostics.

Quick diagnostic for sparse inner-product matrices:
  - symmetry check
  - eigenvalue spectrum (smallest / largest)
  - definiteness classification
  - kernel (null-space) detection
  - Cholesky feasibility

Usage:
    from matrix_diagnostics import diagnose_matrix
    diagnose_matrix(M, name="H1_semi")
"""
import numpy as np
from scipy.sparse import issparse
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigvalsh


def diagnose_matrix(M, name="M", full_spectrum=False, tol=1e-12):
    """
    Diagnose a matrix for symmetry, definiteness, and kernel.

    Args:
        M:              scipy sparse or dense matrix
        name:           label for printout
        full_spectrum:  if True, compute ALL eigenvalues (slow for large M,
                        but gives exact rank). If False, only compute the
                        6 smallest and 6 largest via ARPACK.
        tol:            threshold below which an eigenvalue is considered zero
    """
    print(f"\n{'='*60}")
    print(f"  MATRIX DIAGNOSTICS: {name}")
    print(f"{'='*60}")

    n = M.shape[0]
    print(f"  Shape:        {M.shape}")
    if issparse(M):
        print(f"  Format:       sparse ({M.format}), nnz = {M.nnz}")
        print(f"  Density:      {M.nnz / (n * n) * 100:.2f}%")
    else:
        print(f"  Format:       dense")

    # --- Symmetry check ---
    if issparse(M):
        diff = M - M.T
        asym = abs(diff).max()
    else:
        asym = np.max(np.abs(M - M.T))
    print(f"\n  Symmetry error (max|M - M^T|): {asym:.2e}")
    is_symmetric = asym < tol
    print(f"  Symmetric:    {'YES' if is_symmetric else 'NO'}")

    if not is_symmetric:
        print("  WARNING: matrix is not symmetric — eigenvalue analysis "
              "assumes symmetry. Results may be misleading.")

    # --- Eigenvalue spectrum ---
    print(f"\n  --- Eigenvalue spectrum ---")

    if full_spectrum or n <= 2000:
        # Dense eigenvalue computation — exact but O(n^3)
        if issparse(M):
            M_dense = M.toarray()
        else:
            M_dense = np.asarray(M)
        eigs = eigvalsh(M_dense)  # sorted ascending
        print(f"  Method:       full (eigvalsh), all {len(eigs)} eigenvalues")
    else:
        # Sparse: compute only extreme eigenvalues via ARPACK
        k = min(20, n - 2)
        try:
            eigs_small = eigsh(M, k=k, which='SM', return_eigenvectors=False)
        except Exception as e:
            print(f"  ARPACK SM failed: {e}")
            print("  Falling back to dense computation...")
            M_dense = M.toarray() if issparse(M) else np.asarray(M)
            eigs = eigvalsh(M_dense)
            eigs_small = eigs[:k]

        try:
            eigs_large = eigsh(M, k=min(6, n - 2), which='LM',
                               return_eigenvectors=False)
        except Exception:
            eigs_large = np.array([])

        eigs = np.sort(np.concatenate([eigs_small, eigs_large]))
        print(f"  Method:       ARPACK (partial), {len(eigs)} eigenvalues")

    # Sort and display
    eigs_sorted = np.sort(eigs)
    lambda_min = eigs_sorted[0]
    lambda_max = eigs_sorted[-1]

    print(f"\n  λ_min:        {lambda_min:.6e}")
    print(f"  λ_max:        {lambda_max:.6e}")
    if lambda_max > 0:
        print(f"  Condition:    {lambda_max / max(abs(lambda_min), 1e-30):.2e}")

    # Count zero / negative / positive eigenvalues
    n_negative = np.sum(eigs_sorted < -tol)
    n_zero     = np.sum(np.abs(eigs_sorted) <= tol)
    n_positive = np.sum(eigs_sorted > tol)

    print(f"\n  Negative:     {n_negative}")
    print(f"  Zero (|λ| ≤ {tol:.0e}):  {n_zero}")
    print(f"  Positive:     {n_positive}")

    if full_spectrum or n <= 2000:
        print(f"  Rank:         {n - n_zero}  (out of {n})")
        print(f"  Null dim:     {n_zero}")

    # --- Classification ---
    print(f"\n  --- Classification ---")
    if n_negative == 0 and n_zero == 0:
        classification = "POSITIVE DEFINITE"
        cholesky_ok = True
    elif n_negative == 0 and n_zero > 0:
        classification = "POSITIVE SEMI-DEFINITE (singular)"
        cholesky_ok = False
    elif n_positive == 0 and n_zero == 0:
        classification = "NEGATIVE DEFINITE"
        cholesky_ok = False
    elif n_positive == 0 and n_zero > 0:
        classification = "NEGATIVE SEMI-DEFINITE (singular)"
        cholesky_ok = False
    else:
        classification = "INDEFINITE"
        cholesky_ok = False

    print(f"  Result:       {classification}")
    print(f"  Cholesky OK:  {'YES' if cholesky_ok else 'NO'}")

    # --- Smallest eigenvalues detail ---
    n_show = min(10, len(eigs_sorted))
    print(f"\n  Smallest {n_show} eigenvalues:")
    for i in range(n_show):
        marker = " <-- ZERO" if abs(eigs_sorted[i]) <= tol else ""
        marker = " <-- NEGATIVE" if eigs_sorted[i] < -tol else marker
        print(f"    λ_{i+1:3d} = {eigs_sorted[i]:+.6e}{marker}")

    print(f"\n  Largest {min(5, len(eigs_sorted))} eigenvalues:")
    for i in range(min(5, len(eigs_sorted))):
        idx = len(eigs_sorted) - 1 - i
        print(f"    λ_{idx+1:3d} = {eigs_sorted[idx]:+.6e}")

    print(f"\n{'='*60}\n")

    return {
        'name': name,
        'shape': M.shape,
        'symmetric': is_symmetric,
        'lambda_min': lambda_min,
        'lambda_max': lambda_max,
        'n_negative': n_negative,
        'n_zero': n_zero,
        'n_positive': n_positive,
        'classification': classification,
        'cholesky_ok': cholesky_ok,
        'eigenvalues': eigs_sorted,
    }


def compare_matrices(matrices_dict, full_spectrum=False, tol=1e-12):
    """
    Diagnose and compare several matrices side by side.

    Args:
        matrices_dict: dict of {name: matrix}, e.g.
                       {"L2": M_l2, "H1": M_h1, "H1_semi": M_h1s}

    Returns:
        dict of {name: diagnostics}
    """
    results = {}
    for name, M in matrices_dict.items():
        results[name] = diagnose_matrix(M, name=name,
                                         full_spectrum=full_spectrum, tol=tol)

    # Summary table
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  {'Name':<12} {'λ_min':>12} {'λ_max':>12} {'Zero':>6} "
          f"{'Classification':<30} {'Chol?'}")
    print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*6} {'-'*30} {'-'*5}")

    for name, r in results.items():
        print(f"  {name:<12} {r['lambda_min']:>12.2e} {r['lambda_max']:>12.2e} "
              f"{r['n_zero']:>6} {r['classification']:<30} "
              f"{'YES' if r['cholesky_ok'] else 'NO'}")

    print()
    return results


# =============================================================================
# Example usage
# =============================================================================
if __name__ == "__main__":
    # Example with your setup:
    #
    # from rom import assemble_inner_product_matrix
    # from Navier_Stokes import setup_navier_stokes_problem
    #
    # problem = setup_navier_stokes_problem(...)
    # V = problem.velocity_space
    #
    # M_l2   = assemble_inner_product_matrix(V, 'L2')
    # M_h1   = assemble_inner_product_matrix(V, 'H1')
    # M_h1s  = assemble_inner_product_matrix(V, 'H1_semi')
    #
    # # Diagnose one:
    # diagnose_matrix(M_h1s, name="H1_semi")
    #
    # # Or compare all three:
    # compare_matrices({"L2": M_l2, "H1": M_h1, "H1_semi": M_h1s})

    # Quick demo with a toy matrix
    from scipy.sparse import diags

    # Positive definite
    A = diags([2, -1, -1], [0, -1, 1], shape=(5, 5), format='csc')
    diagnose_matrix(A, name="tridiag (PD)")

    # Positive semi-definite (graph Laplacian)
    L = diags([2, -1, -1], [0, -1, 1], shape=(5, 5), format='csc')
    L_lap = L.toarray()
    L_lap[0, 0] = 1; L_lap[-1, -1] = 1  # make it singular-ish
    from scipy.sparse import csc_matrix
    diagnose_matrix(csc_matrix(L_lap), name="laplacian-like (PSD)")
