import numpy as np
import matplotlib.pyplot as plt
from firedrake import Function

from nsrom.navier_stokes import NavierStokesSolution, solve_steady_navier_stokes
from .jacobian import build_state_jacobian_pencil
from .eigen_solver import solve_leftmost_real_eigenpairs


def snapshot_to_solution(velocity_dofs, pressure_dofs, Re, amp, problem):
    """Wrap raw DOF arrays as a NavierStokesSolution on problem.mixed_space."""
    w = Function(problem.mixed_space)
    u, p = w.subfunctions
    u.dat.data[:] = velocity_dofs.reshape(u.dat.data.shape)
    p.dat.data[:] = pressure_dofs.reshape(p.dat.data.shape)
    return NavierStokesSolution(
        solution=w, velocity=u, pressure=p,
        reynolds=float(Re), amplitude=float(amp),
    )


def sweep_symmetric_branch(
    velocity_snapshots,
    pressure_snapshots,
    parameters,
    branch_ids,
    problem,
    n_eigenvalues=6,
    amp_tol=1e-8,
):
    """
    Run the eigensolver at every symmetric-branch (branch_id == 0), A = 0
    snapshot. Returns one record per converged Re, sorted ascending.
    """
    mask = (branch_ids == 0) & (np.abs(parameters[:, 1]) < amp_tol)
    indices = np.where(mask)[0]
    Re_values = parameters[indices, 0]
    order = np.argsort(Re_values)
    indices = indices[order]

    records = []
    for k, i in enumerate(indices):
        Re = float(parameters[i, 0])
        amp = float(parameters[i, 1])
        sol = snapshot_to_solution(
            velocity_snapshots[i], pressure_snapshots[i], Re, amp, problem
        )
        try:
            J, M = build_state_jacobian_pencil(problem, sol)
            eigvals, _, _, nconv = solve_leftmost_real_eigenpairs(
                J, M, n_eigenvalues=n_eigenvalues
            )
            print(f'We are getting {np.shape(eigvals)[0]} eigenvalues')
        except Exception as e:
            print(f"  [{k+1}/{len(indices)}] Re={Re:7.2f}  FAILED: {e}")
            continue
        is_real = np.abs(eigvals.imag) < 1e-8
        n_real = int(np.sum(is_real))

        if n_real > 0:
            real_re = eigvals.real[is_real]
            leading_real = real_re[np.argmin(np.abs(real_re))]
        else:
            leading_real = np.nan
        # n_real = int(np.sum(np.abs(eigvals.imag) < 1e-8))
        # leading_real = (
        #     np.min(eigvals.real[np.abs(eigvals.imag) < 1e-8])
        #     if n_real > 0 else np.nan
        # )
        print(
            f"  [{k+1}/{len(indices)}] Re={Re:7.2f}  "
            f"nconv={nconv}  n_real={n_real}  "
            f"leading_real_μ={leading_real:+.4e}"
        )
        records.append({"Re": Re, "eigenvalues": eigvals})

    return records


def plot_spectrum_vs_Re(records, savepath=None):
    fig, ax = plt.subplots(figsize=(8, 5))
    for rec in records:
        Re = rec["Re"]
        ev = rec["eigenvalues"]
        is_real = np.abs(ev.imag) < 1e-8
        ax.scatter(np.full(np.sum(is_real), Re), ev.real[is_real],
                   c="tab:blue", s=22, label="real" if Re == records[0]["Re"] else None)
        ax.scatter(np.full(np.sum(~is_real), Re), ev.real[~is_real],
                   c="tab:orange", s=22, marker="x",
                   label="complex (Re part)" if Re == records[0]["Re"] else None)
    ax.axhline(0.0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("Re")
    ax.set_ylabel(r"Re$(\mu)$")
    ax.set_title("Leftmost spectrum on symmetric branch (A = 0)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
        print(f"  Plot saved: {savepath}")
    plt.show()
    return fig


def sweep_symmetric_branch_all_amp(
    velocity_snapshots,
    pressure_snapshots,
    parameters,
    branch_ids,
    problem,
    n_eigenvalues=6,
):
    """
    Run the eigensolver at every symmetric-branch snapshot (branch_id == 0).
    Symmetric forcing preserves the Z₂ symmetry of the problem, so the
    symmetric branch and its pitchfork exist at every A in the snapshot set;
    Re_c is a function of A.

    Returns records sorted by (A, Re).
    """
    mask = (branch_ids == 0)
    indices = np.where(mask)[0]
    A_values = parameters[indices, 1]
    Re_values = parameters[indices, 0]
    order = np.lexsort((Re_values, A_values))   # primary key: A, secondary: Re
    indices = indices[order]

    records = []
    for k, i in enumerate(indices):
        Re = float(parameters[i, 0])
        amp = float(parameters[i, 1])
        sol = snapshot_to_solution(
            velocity_snapshots[i], pressure_snapshots[i], Re, amp, problem
        )
        try:
            J, M = build_state_jacobian_pencil(problem, sol)
            eigvals, _, _, nconv = solve_leftmost_real_eigenpairs(
                J, M, n_eigenvalues=n_eigenvalues
            )
        except Exception as e:
            print(f"  [{k+1}/{len(indices)}] Re={Re:7.2f} A={amp:+.3f}  FAILED: {e}")
            continue
        is_real = np.abs(eigvals.imag) < 1e-8
        n_real = int(np.sum(is_real))

        if n_real > 0:
            real_re = eigvals.real[is_real]
            leading_real = real_re[np.argmin(np.abs(real_re))]
        else:
            leading_real = np.nan

        print(
            f"  [{k+1}/{len(indices)}] Re={Re:7.2f} A={amp:+.3f}  "
            f"nconv={nconv}  n_real={n_real}  "
            f"leading_real_μ={leading_real:+.4e}"
        )
        records.append({"Re": Re, "A": amp, "eigenvalues": eigvals})

    return records

def plot_spectrum_vs_Re_by_A(records, savepath=None):
    """One subplot per distinct A value, blue = real eigenvalues, orange = Re(complex)."""
    A_unique = sorted({rec["A"] for rec in records})
    n = len(A_unique)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             squeeze=False, sharex=True, sharey=True)

    for ax, A in zip(axes.flat, A_unique):
        recs_A = [r for r in records if r["A"] == A]
        for r in recs_A:
            Re = r["Re"]
            ev = r["eigenvalues"]
            is_real = np.abs(ev.imag) < 1e-8
            ax.scatter(np.full(np.sum(is_real), Re), ev.real[is_real],
                       c="tab:blue", s=22)
            ax.scatter(np.full(np.sum(~is_real), Re), ev.real[~is_real],
                       c="tab:orange", s=22, marker="x")
        ax.axhline(0.0, color="k", lw=0.8, ls="--")
        ax.set_title(f"A = {A:+.3f}")
        ax.grid(True, alpha=0.3)

    for ax in axes[-1, :]:
        ax.set_xlabel("Re")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"Re$(\mu)$")

    # Hide unused panels.
    for ax in axes.flat[n:]:
        ax.set_visible(False)

    fig.suptitle("Leftmost spectrum on symmetric branch — by A", y=1.02)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
        print(f"  Plot saved: {savepath}")
    plt.show()
    return fig

def leading_real_mu(problem, sol, n_eigenvalues=6):
    """
    Real eigenvalue closest to zero (the pitchfork-candidate mode), returned
    with its signed real part. The signed value crosses zero at Re_c, suitable
    for bisection.
    """
    J, M = build_state_jacobian_pencil(problem, sol)
    eigvals, vrs, vis, _ = solve_leftmost_real_eigenpairs(
        J, M, n_eigenvalues=n_eigenvalues
    )
    is_real = np.abs(eigvals.imag) < 1e-8
    if not is_real.any():
        raise RuntimeError("No purely-real eigenvalue in leftmost cluster.")

    real_idx = np.where(is_real)[0]
    real_re = eigvals.real[real_idx]
    j = int(np.argmin(np.abs(real_re)))   # pick by |Re|
    i = int(real_idx[j])
    return float(eigvals[i].real), vrs[i].copy()  # return signed Re


def solve_symmetric_at_Re(problem, Re, warm_start_solution):
    """Solve the steady NS at a given Re on the symmetric branch (A = 0)."""
    problem.reynolds.assign(Re)
    problem.amplitude.assign(0.0)
    return solve_steady_navier_stokes(problem, warm_start_solution.solution)


def bisect_pitchfork(
    problem,
    Re_left, mu_left, sol_left,
    Re_right, mu_right, sol_right,
    Re_tol=1e-2,
    newton_safe_mu=1e-4,
    max_iter=20,
):
    history = [(Re_left, mu_left), (Re_right, mu_right)]

    # Compute endpoint eigenvectors once, so we always have one to return.
    _, eigvec_left = leading_real_mu(problem, sol_left)
    _, eigvec_right = leading_real_mu(problem, sol_right)

    for it in range(max_iter):
        # Switch to secant once we're inside Newton-unsafe territory.
        if min(abs(mu_left), abs(mu_right)) < newton_safe_mu:
            Re_c = Re_left - mu_left * (Re_right - Re_left) / (mu_right - mu_left)

            print(
                f"  [secant] |μ| < {newton_safe_mu:.0e}, "
                f"Re_c ≈ {Re_c:.4f} by inverse linear interpolation"
            )

            # Pick the endpoint closest to criticality as base/eigenvector.
            if abs(mu_left) < abs(mu_right):
                return Re_c, sol_left, eigvec_left, mu_left, history
            else:
                return Re_c, sol_right, eigvec_right, mu_right, history

        Re_mid = 0.5 * (Re_left + Re_right)

        warm = sol_left if abs(mu_left) < abs(mu_right) else sol_right
        sol_mid = solve_symmetric_at_Re(problem, Re_mid, warm)

        mu_mid, eigvec_mid = leading_real_mu(problem, sol_mid)
        history.append((Re_mid, mu_mid))

        print(f"  [it {it+1:2d}] Re={Re_mid:7.4f}  μ={mu_mid:+.4e}")

        if mu_mid > 0:
            Re_left, mu_left, sol_left = Re_mid, mu_mid, sol_mid
            eigvec_left = eigvec_mid
        else:
            Re_right, mu_right, sol_right = Re_mid, mu_mid, sol_mid
            eigvec_right = eigvec_mid

        if Re_right - Re_left < Re_tol:
            break

    Re_c = 0.5 * (Re_left + Re_right)

    if abs(mu_left) < abs(mu_right):
        return Re_c, sol_left, eigvec_left, mu_left, history
    else:
        return Re_c, sol_right, eigvec_right, mu_right, history
