import numpy as np
from slepc4py import SLEPc
from firedrake import COMM_WORLD, Function

def vec_to_function(vec, space, name="eigenfunction"):
    """Wrap a PETSc Vec as a Firedrake Function on `space`. Serial-safe."""
    f = Function(space, name=name)
    with f.dat.vec_wo as f_vec:
        vec.copy(f_vec)
    return f
def solve_leftmost_real_eigenpairs(
    J, M, *,
    n_eigenvalues=6,
    target=0.0,
    tolerance=1e-8,
    max_it=200,
    monitor=False,
):
    """
    Generalized eigenproblem  J δw = μ M δw, returning the eigenvalues
    whose real part is closest to `target` (default 0).

    Method: Krylov-Schur + shift-invert (σ = target) with MUMPS LU as the
    inner solver. Suitable for non-Hermitian J, possibly-singular M (the
    BC-DOF zeros sent in via `build_state_jacobian_pencil` push spurious
    eigenvalues to ∞, well outside the leftmost real cluster near 0).

    Convention (as in build_state_jacobian_pencil): μ are eigenvalues of
    the Newton-residual Jacobian. Linearized stability eigenvalues σ = −μ.
    Bifurcation indicator: smallest Re(μ) crosses zero from above.

    Returns
    -------
    eigenvalues : np.ndarray, shape (n_converged,), complex
        Sorted by ascending Re(μ).
    eigenvectors_real : list[PETSc.Vec]
    eigenvectors_imag : list[PETSc.Vec]
        Real and imaginary parts of the right eigenvectors, same order.
        For a real μ the imaginary part is numerically zero.
    n_converged : int
    """
    eps = SLEPc.EPS().create(comm=COMM_WORLD)
    eps.setOperators(J, M)
    eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
    eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
    eps.setDimensions(nev=n_eigenvalues)
    eps.setTolerances(tol=tolerance, max_it=max_it)
    eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)
    eps.setTarget(target)

    st = eps.getST()
    st.setType(SLEPc.ST.Type.SINVERT)
    st.setShift(target)
    ksp = st.getKSP()
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("lu")
    pc.setFactorSolverType("mumps")

    if monitor:
        eps.setMonitor(
            lambda eps_, it, nconv, eig, err:
                print(f"  iter={it:3d}  nconv={nconv}")
        )
    n = J.getSize()[0]
    nev = max(1, min(n_eigenvalues, n - 1))
    ncv = min(n, max(2 * nev + 1, nev + 10))
    eps.setDimensions(nev=nev, ncv=ncv)
    eps.solve()
    nconv = eps.getConverged()

    if nconv == 0:
        eps.destroy()
        return np.array([], dtype=complex), [], [], 0

    vr = J.createVecRight()
    vi = J.createVecRight()
    eigvals = np.empty(nconv, dtype=complex)
    vrs, vis = [], []
    for i in range(nconv):
        eigvals[i] = eps.getEigenpair(i, vr, vi)
        vrs.append(vr.copy())
        vis.append(vi.copy())

    eps.destroy()

    order = np.argsort(eigvals.real)
    eigvals = eigvals[order]
    vrs = [vrs[k] for k in order]
    vis = [vis[k] for k in order]
    return eigvals, vrs, vis, nconv