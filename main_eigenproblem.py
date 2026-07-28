from firedrake import Function
import numpy as np
from pathlib import Path

from nsrom.navier_stokes import setup_navier_stokes_problem, load_solution, save_solution, compute_forces
from nsrom.snapshot_collection import load_snapshot_dofs
from nsrom.lifting_functions import load_lifting_functions
from nsrom.bifurcation.jacobian import build_state_jacobian_pencil
from nsrom.bifurcation.eigen_solver import solve_leftmost_real_eigenpairs
from nsrom.bifurcation.sweep_study import sweep_symmetric_branch, plot_spectrum_vs_Re, sweep_symmetric_branch_all_amp, plot_spectrum_vs_Re_by_A, bisect_pitchfork, snapshot_to_solution, solve_symmetric_at_Re, leading_real_mu
from nsrom.bifurcation.branch_jump import branch_jump_pitchfork, continue_asymmetric_branch
SNAPSHOT_DIR = "multi_param_multi_branch"
DEIM_SNAPSHOT_DIR = "multi_param_multi_branch"
LIFTING_DIR = "lifting"
MESH_FILE = "mesh/mid_pinball.msh"
FOM_CHECKPOINT = "initial_states/velocity_checkpoint_symm"

REYNOLDS_INIT = 100.0
AMPLITUDE_INIT = 0.0
RE_CRITICAL_SOL = 'nsrom/bifurcation/critical_Re_sol'

def antisym_score(arr):
    """
    Score in [-1, +1].

    +1 means perfectly antisymmetric.
    -1 means symmetric.
    """
    return -float(np.dot(arr, arr[::-1]) / (np.dot(arr, arr) + 1e-30))


def main():
    print("--- Setup problem ---")

    problem = setup_navier_stokes_problem(
        mesh_file=MESH_FILE,
        reynolds_init=REYNOLDS_INIT,
        amplitude_init=AMPLITUDE_INIT,
    )

    initial_solution = load_solution(FOM_CHECKPOINT, problem)
    # --- Load snapshots and lifting ---
    print("\n--- Load snapshots and lifting ---")
    snapshot_data = load_snapshot_dofs(SNAPSHOT_DIR)
    lifting       = load_lifting_functions(LIFTING_DIR, problem)

    velocity_snapshots = snapshot_data['velocity']
    pressure_snapshots = snapshot_data['pressure']
    parameters         = snapshot_data['parameters']
    lift_coefficients  = snapshot_data['lift_coefficients']

    print(f"  Snapshots:  {velocity_snapshots.shape[0]}")
    print(f"  Parameters: {len(parameters)} entries")
    print("\n--- Build Jacobian pencil at initial_solution ---")

    J, M = build_state_jacobian_pencil(problem, initial_solution)

    n, _ = J.getSize()

    print(f"  System size: {n}")
    print(f"  J nnz:       {J.getInfo()['nz_used']:.0f}")
    print(f"  M nnz:       {M.getInfo()['nz_used']:.0f}")
    print(f"  M symmetric: {M.isSymmetric(1e-10)}  (expected True)")
    print(f"  J symmetric: {J.isSymmetric(1e-10)}  (expected False)")
    print(f"  Re:          {float(problem.reynolds)}")
    print(f"  Amplitude:   {float(problem.amplitude)}")
    path = Path(f'{RE_CRITICAL_SOL}.h5')
    if path.is_file():
        meta = np.load(RE_CRITICAL_SOL + "_metadata.npz",allow_pickle = True)

        sol_c = load_solution(
            RE_CRITICAL_SOL,
            problem,
            with_pressure=True,
        )
        sol_c.reynolds=float(meta["reynolds"])
        sol_c.amplitude=float(meta["amplitude"])
        Re_c = sol_c.reynolds
        print(f"Loaded sol_c.reynolds = {sol_c.reynolds}")
        print(f"Loaded sol_c.amplitude = {sol_c.amplitude}")
        print(f"Re_c used for branch jump = {Re_c}")

        mu_c, eigvec_c = leading_real_mu(problem, sol_c)
        print(f"Loaded critical mu_c = {mu_c:+.6e}")
        records = meta["records"]
    else:

        print("\n--- Sweep symmetric branch ---")
        branch_ids = snapshot_data["branch_ids"]
        records = sweep_symmetric_branch(
            velocity_snapshots, pressure_snapshots, parameters, branch_ids,
            problem, n_eigenvalues=6,
        )

        print("\n--- Plot spectrum vs Re ---")
        # plot_spectrum_vs_Re(records, savepath="bifurcation/spectrum_symmetric.png")

        # print("\n--- Sweep symmetric branch (all A) ---")
        # branch_ids = snapshot_data["branch_ids"]
        # records = sweep_symmetric_branch_all_amp(
        #     velocity_snapshots, pressure_snapshots, parameters, branch_ids, problem,
        #     n_eigenvalues=6,
        # )

        # print("\n--- Plot spectrum vs Re, grouped by A ---")
        # plot_spectrum_vs_Re_by_A(records, savepath="bifurcation/spectrum_symmetric.png")

        print("\n--- Bisect Re_c (pitchfork on symmetric branch) ---")

        # Find tightest bracket from the sweep records.
        Re_arr = np.array([r["Re"] for r in records])
        mu_arr = np.array([
            np.min(r["eigenvalues"].real[np.abs(r["eigenvalues"].imag) < 1e-8])
            for r in records
        ])
        sign_change = np.where(np.diff(np.sign(mu_arr)) < 0)[0]
        if len(sign_change) == 0:
            raise RuntimeError("No sign change in leading real μ across sweep.")
        i_left = int(sign_change[0])
        i_right = i_left + 1

        Re_left, mu_left = float(Re_arr[i_left]), float(mu_arr[i_left])
        Re_right, mu_right = float(Re_arr[i_right]), float(mu_arr[i_right])
        print(f"  Initial bracket: Re ∈ [{Re_left:.3f}, {Re_right:.3f}], μ ∈ [{mu_left:+.3e}, {mu_right:+.3e}]")



        # Reconstruct snapshot solutions at the bracket endpoints for warm-starting.
        def snap_idx(Re_target):
            return int(np.where(
                (branch_ids == 0)
                & (np.abs(parameters[:, 0] - Re_target) < 1e-2)
                & (np.abs(parameters[:, 1]) < 1e-8)
            )[0][0])

        i_snap_left  = snap_idx(Re_left)
        i_snap_right = snap_idx(Re_right)
        sol_left  = snapshot_to_solution(velocity_snapshots[i_snap_left],  pressure_snapshots[i_snap_left],
                                        Re_left,  0.0, problem)
        sol_right = snapshot_to_solution(velocity_snapshots[i_snap_right], pressure_snapshots[i_snap_right],
                                        Re_right, 0.0, problem)

        Re_c, sol_c, eigvec_c, mu_c, history = bisect_pitchfork(
            problem,
            Re_left,  mu_left,  sol_left,
            Re_right, mu_right, sol_right,
            Re_tol=1e-2,
        )
        print(f"  Bracket history: {len(history)} points")
        problem.reynolds.assign(float(Re_c))
        problem.amplitude.assign(0.0)

        # Re-solve the symmetric state exactly at Re_c
        sol_c = solve_symmetric_at_Re(problem, Re_c, sol_c)

        # Recompute eigenvector at the actual critical state
        mu_c, eigvec_c = leading_real_mu(problem, sol_c)

        # Force metadata to match
        sol_c.reynolds = float(Re_c)
        sol_c.amplitude = 0.0

        print(f"  Critical state before saving:")
        print(f"    Re_c = {Re_c:.8f}")
        print(f"    mu_c = {mu_c:+.6e}")

        save_solution(sol_c, RE_CRITICAL_SOL)
        np.savez(
            RE_CRITICAL_SOL + "_metadata.npz",
            reynolds=float(Re_c),
            amplitude=0.0,
            records = records,
        )



    print("\n--- Branch jump from pitchfork ---")

    jump_results = branch_jump_pitchfork(
        problem,
        sol_c,
        eigvec_c,
        Re_c,
        eps_values=(0.5,),
        amp_values=(3e-2, 1e-1),
        perturb_pressure=False,
        asymmetry_tol=1e-3,
    )

    print(f"\n  Successful jumps: {len(jump_results)}")

    for j, r in enumerate(jump_results):
        print(
            f"  jump {j}: "
            f"sign={r['sign']:+d}, "
            f"Re={r['Re']:.6f}, "
            f"eps={r['epsilon']:.3e}, "
            f"amp={r['amplitude']:.3e}, "
            f"ux_score={r['scores']['ux_antisym_score']:+.3f}, "
            f"uy_score={r['scores']['uy_antisym_score']:+.3f}"
        )
    print("\n--- Lift/drag check for jumped branches ---")

    for j, r in enumerate(jump_results):
        forces = compute_forces(r["solution"], problem)

        print(
            f"  jump {j}: "
            f"sign={r['sign']:+d}, "
            f"Re={r['Re']:.6f}, "
            f"amp={r['amplitude']:.3e}, "
            f"C_D={forces.drag:+.8e}, "
            f"C_L={forces.lift:+.8e}"
        )

    print("\n--- Continue along asymmetric branches ---")

    sol_pos = next(r["solution"] for r in jump_results if r["sign"] == +1)
    sol_neg = next(r["solution"] for r in jump_results if r["sign"] == -1)

    print("\n  Upper branch (C_L < 0):")
    upper_sols, upper_diag = continue_asymmetric_branch(
        problem, sol_pos, Re_max=100.0,
    )

    print("\n  Lower branch (C_L > 0):")
    lower_sols, lower_diag = continue_asymmetric_branch(
        problem, sol_neg, Re_max=100.0,
    )

    print(f"\n  Upper branch: {len(upper_sols)} solutions, "
        f"Re ∈ [{upper_diag[0]['Re']:.2f}, {upper_diag[-1]['Re']:.2f}]")
    print(f"  Lower branch: {len(lower_sols)} solutions, "
        f"Re ∈ [{lower_diag[0]['Re']:.2f}, {lower_diag[-1]['Re']:.2f}]")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))

    # Symmetric branch (from your records, A=0)
    sym_records = [r for r in records ]
    Re_sym = [r["Re"] for r in sym_records]
    ax.plot(Re_sym, [0.0] * len(Re_sym), "k-", label="symmetric (C_L = 0)")

    # Asymmetric branches
    Re_upper = [d["Re"] for d in upper_diag]
    CL_upper = [d["C_L"] for d in upper_diag]
    Re_lower = [d["Re"] for d in lower_diag]
    CL_lower = [d["C_L"] for d in lower_diag]

    ax.plot(Re_upper, CL_upper, "b.-", label="upper asymmetric")
    ax.plot(Re_lower, CL_lower, "r.-", label="lower asymmetric")
    ax.axvline(Re_c, color="gray", ls="--", alpha=0.5, label=f"Re_c = {Re_c:.2f}")
    ax.axhline(0.0, color="k", lw=0.5)
    ax.set_xlabel("Re")
    ax.set_ylabel("C_L")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title("Pinball pitchfork bifurcation diagram")
    plt.show()
if __name__ == "__main__":
    main()