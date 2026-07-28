"""
Bifurcation diagram: FOM.

Reads:
    - sweep_results.npz

Writes:
    - bifurcation_diagram.png
    - bifurcation_diagram.pdf
    - bifurcation_diagram.npz
"""

import numpy as np
import matplotlib.pyplot as plt

from nsrom.navier_stokes import (
    setup_navier_stokes_problem,
    load_solution,
    solve_steady_navier_stokes,
    compute_forces,
)

from nsrom.bifurcation.eigen_solver import solve_leftmost_real_eigenpairs
from nsrom.bifurcation.jacobian import build_state_jacobian_pencil
from nsrom.bifurcation.branch_jump import (
    branch_jump_pitchfork,
    continue_asymmetric_branch,
)


# =============================================================================
# CONFIG
# =============================================================================

MESH_FILE = "mesh/mid_pinball.msh"
FOM_CHECKPOINT = "initial_states/velocity_checkpoint_symm"

REYNOLDS_INIT = 100.0
AMPLITUDE_INIT = 0.0
AMP_PARAM = 0.0

SWEEP_RESULTS = "sweep_results.npz"

RE_MAX = 90.0

DRE_NEAR = 0.5
DRE_FAR = 1.0
NEAR_STEPS = 4
CL_COLLAPSE = 0.5

FOM_EPS_VALUES = (0.2,)
FOM_AMP_VALUES = (0.5,)


# ============================================================================
# HELPERS
# =============================================================================

def detect_sign_change(re_values, mu_values):
    """Return linearly interpolated first zero crossing."""
    re = np.asarray(re_values, dtype=float)
    mu = np.asarray(mu_values, dtype=float)

    mask = ~np.isnan(mu)
    re = re[mask]
    mu = mu[mask]

    if len(mu) < 2:
        return None

    idx = np.where(np.diff(np.sign(mu)) != 0)[0]
    if len(idx) == 0:
        return None

    i = int(idx[0])
    t = -mu[i] / (mu[i + 1] - mu[i])
    return float(re[i] + t * (re[i + 1] - re[i]))


def fom_pitchfork_eigvec(problem, fom_solution, n_eig=6, target=0.0):
    """
    Return FOM pitchfork eigenvector.

    We select the real eigenvalue closest to zero, not the most negative one.
    """
    J, M = build_state_jacobian_pencil(problem, fom_solution)

    eigvals, vrs, _, nconv = solve_leftmost_real_eigenpairs(
        J,
        M,
        n_eigenvalues=n_eig,
        target=target,
    )

    if nconv == 0:
        raise RuntimeError("FOM eigensolver converged no eigenpairs.")

    scale = max(1.0, max(abs(e) for e in eigvals))
    imag_tol = 1e-8 * scale

    real_idx = np.where(np.abs(eigvals.imag) < imag_tol)[0]

    if len(real_idx) == 0:
        raise RuntimeError("No real FOM eigenvalue found near target.")

    i = int(real_idx[np.argmin(np.abs(eigvals.real[real_idx]))])

    return vrs[i].copy(), float(eigvals[i].real)


def select_seed_by_lift(seeds, problem):
    """
    Select one negative-lift and one positive-lift seed.

    This is better than selecting by kick sign because eigenvector sign is arbitrary.
    """
    infos = []

    for seed in seeds:
        forces = compute_forces(seed["solution"], problem)

        infos.append({
            "seed": seed,
            "C_L": float(forces.lift),
            "C_D": float(forces.drag),
            "epsilon": float(seed["epsilon"]),
            "amplitude": float(seed["amplitude"]),
            "kick_sign": int(seed["sign"]),
            "Re": float(seed["Re"]),
        })

    if not infos:
        return None, None

    print("\nSuccessful FOM seeds:")
    for info in infos:
        print(
            f"  kick={info['kick_sign']:+d} | "
            f"Re={info['Re']:.4f} | "
            f"eps={info['epsilon']:.3e} | "
            f"amp={info['amplitude']:.3e} | "
            f"C_L={info['C_L']:+.6e}"
        )

    # Prefer closest seed to bifurcation.
    eps_min = min(info["epsilon"] for info in infos)
    near = [info for info in infos if info["epsilon"] == eps_min]

    neg = next((info for info in near if info["C_L"] < 0.0), None)
    pos = next((info for info in near if info["C_L"] > 0.0), None)

    # Fall back to all seeds if one sign is missing at eps_min.
    if neg is None:
        neg = next((info for info in infos if info["C_L"] < 0.0), None)

    if pos is None:
        pos = next((info for info in infos if info["C_L"] > 0.0), None)

    return neg, pos


def continue_fom_branch(label, seed_info, problem):
    """Continue one FOM branch from a selected seed."""
    if seed_info is None:
        print(f"\n--- FOM branch {label}: missing seed ---")
        return []

    seed = seed_info["seed"]

    print(
        f"\n--- FOM branch {label} ---\n"
        f"  seed Re={seed_info['Re']:.4f}, "
        f"C_L={seed_info['C_L']:+.6e}, "
        f"kick={seed_info['kick_sign']:+d}, "
        f"amp={seed_info['amplitude']:.3e}"
    )

    _, diagnostics = continue_asymmetric_branch(
        problem=problem,
        sol_start=seed["solution"],
        Re_max=RE_MAX,
        dRe_near=DRE_NEAR,
        dRe_far=DRE_FAR,
        near_steps=NEAR_STEPS,
        cl_collapse_factor=CL_COLLAPSE,
    )

    return diagnostics


def branch_arrays(records):
    if not records:
        return np.array([]), np.array([])

    return (
        np.array([r["Re"] for r in records]),
        np.array([r["C_L"] for r in records]),
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("BIFURCATION DIAGRAM — FOM")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Load sweep.
    # -------------------------------------------------------------------------

    sweep = np.load(SWEEP_RESULTS)

    re_sym = np.asarray(sweep["Re"])
    mu_fom = np.asarray(sweep["mu_fom"])

    re_c_fom = detect_sign_change(re_sym, mu_fom)

    if re_c_fom is None:
        raise RuntimeError("Could not detect Re_c^FOM.")

    print(f"Re_c^FOM = {re_c_fom:.6f}")

    # -------------------------------------------------------------------------
    # Build problem.
    # -------------------------------------------------------------------------

    problem = setup_navier_stokes_problem(
        mesh_file=MESH_FILE,
        reynolds_init=REYNOLDS_INIT,
        amplitude_init=AMPLITUDE_INIT,
    )

    initial_solution = load_solution(FOM_CHECKPOINT, problem)

    # -------------------------------------------------------------------------
    # FOM critical solution and branch jump.
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("FOM ASYMMETRIC BRANCHES")
    print("=" * 80)

    problem.reynolds.assign(float(re_c_fom))
    problem.amplitude.assign(float(AMP_PARAM))

    sol_c_fom = solve_steady_navier_stokes(
        problem,
        initial_solution.solution,
    )

    eigvec_c, mu_c = fom_pitchfork_eigvec(problem, sol_c_fom)

    print(
        f"FOM symmetric solution near Re_c:\n"
        f"  Re_c^FOM = {re_c_fom:.6f}\n"
        f"  mu_c     = {mu_c:+.6e}"
    )

    seeds_fom = branch_jump_pitchfork(
        problem=problem,
        sol_c=sol_c_fom,
        eigvec_c=eigvec_c,
        Re_c=re_c_fom,
        eps_values=FOM_EPS_VALUES,
        amp_values=FOM_AMP_VALUES,
    )

    neg_seed, pos_seed = select_seed_by_lift(seeds_fom, problem)

    if neg_seed is None or pos_seed is None:
        print("\nWARNING: did not find both FOM lift-sign branches.")

    fom_neg = continue_fom_branch("C_L < 0", neg_seed, problem)
    fom_pos = continue_fom_branch("C_L > 0", pos_seed, problem)

    # -------------------------------------------------------------------------
    # Prepare arrays.
    # -------------------------------------------------------------------------

    fom_neg_Re, fom_neg_CL = branch_arrays(fom_neg)
    fom_pos_Re, fom_pos_CL = branch_arrays(fom_pos)

    # -------------------------------------------------------------------------
    # Plot.
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("PLOTTING")
    print("=" * 80)

    fig, ax = plt.subplots(figsize=(8, 5.5))

    ax.axhline(
        0.0,
        color="0.7",
        lw=1.0,
        label="symmetric branch",
    )

    if len(fom_neg_Re):
        ax.plot(fom_neg_Re, fom_neg_CL, "--", lw=1.8, label="FOM, C_L < 0")

    if len(fom_pos_Re):
        ax.plot(fom_pos_Re, fom_pos_CL, "--", lw=1.8, label="FOM, C_L > 0")

    ax.axvline(
        re_c_fom,
        color="black",
        ls=":",
        lw=1.2,
        label=f"Re_c^FOM = {re_c_fom:.3f}",
    )

    ax.set_xlabel("Re")
    ax.set_ylabel(r"$C_L$")
    ax.set_title("Pitchfork bifurcation (FOM)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()

    fig.savefig("bifurcation_diagram.png", dpi=150)
    fig.savefig("bifurcation_diagram.pdf")

    print("Saved bifurcation_diagram.png")
    print("Saved bifurcation_diagram.pdf")

    # -------------------------------------------------------------------------
    # Save.
    # -------------------------------------------------------------------------

    np.savez(
        "bifurcation_diagram.npz",
        re_c_fom=re_c_fom,

        sym_Re=re_sym,
        sym_mu_fom=mu_fom,

        fom_neg_Re=fom_neg_Re,
        fom_neg_CL=fom_neg_CL,
        fom_pos_Re=fom_pos_Re,
        fom_pos_CL=fom_pos_CL,
    )

    print("Saved bifurcation_diagram.npz")


if __name__ == "__main__":
    main()
