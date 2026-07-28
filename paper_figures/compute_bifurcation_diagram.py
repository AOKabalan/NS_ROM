#!/usr/bin/env python
"""
Compute the FOM pitchfork bifurcation diagram for the fluidic pinball.

Pipeline (FOM only):
  1. sweep the symmetric branch and locate Re_c   (cached)
  2. compute the symmetric branch by continuation from RE_SYM_MIN to RE_MAX
  3. branch-jump onto the two asymmetric branches from the critical eigenmode
  4. continue both asymmetric branches in Re

The expensive sweep + bisection runs only once: its result is cached at
RE_CRITICAL_SOL and reloaded on subsequent runs.

This script does the numerics only and writes (in OUTDIR):
    bifurcation_diagram.npz

Render the figure with:
    python plot_bifurcation_diagram.py
"""

import os
from pathlib import Path

import numpy as np

from nsrom.navier_stokes import (
    setup_navier_stokes_problem,
    load_solution,
    save_solution,
    compute_forces,
)
from nsrom.snapshot_collection import load_snapshot_dofs
from nsrom.lifting_functions import load_lifting_functions
from nsrom.bifurcation.sweep_study import (
    sweep_symmetric_branch,
    bisect_pitchfork,
    snapshot_to_solution,
    solve_symmetric_at_Re,
    leading_real_mu,
)
from nsrom.bifurcation.branch_jump import (
    branch_jump_pitchfork,
    continue_asymmetric_branch,
)


# =============================================================================
# CONFIG
# =============================================================================

SNAPSHOT_DIR = "multi_param_multi_branch"
LIFTING_DIR = "lifting"
MESH_FILE = "mesh/mid_pinball.msh"
FOM_CHECKPOINT = "initial_states/velocity_checkpoint_symm"
RE_CRITICAL_SOL = "nsrom/bifurcation/critical_Re_sol"

REYNOLDS_INIT = 100.0
AMPLITUDE_INIT = 0.0

# Symmetric branch continuation (computed, not assumed C_L = 0).
RE_SYM_MIN = 30.0
DRE_SYM = 2.5

# Branch-jump grid that separates both pitchfork branches:
# small eps -> Re_jump near Re_c (branches close to symmetric line),
# large amp -> kick big enough to straddle onto opposite branches.
FOM_EPS_VALUES = (0.2,)
FOM_AMP_VALUES = (0.5,)

RE_MAX = 100.0

OUTDIR = "figures"
NPZ_NAME = "bifurcation_diagram.npz"


# =============================================================================
# CRITICAL POINT  (sweep + bisect, cached)
# =============================================================================

def load_or_compute_critical(problem, snapshot_data):
    """
    Return (Re_c, sol_c, eigvec_c, mu_c, records).

    Loads the cached critical solution if present; otherwise runs the
    symmetric sweep + pitchfork bisection and caches the result.
    """
    velocity_snapshots = snapshot_data["velocity"]
    pressure_snapshots = snapshot_data["pressure"]
    parameters = snapshot_data["parameters"]

    path = Path(f"{RE_CRITICAL_SOL}.h5")

    if path.is_file():
        print("\n--- Load cached critical solution ---")
        meta = np.load(RE_CRITICAL_SOL + "_metadata.npz", allow_pickle=True)

        sol_c = load_solution(RE_CRITICAL_SOL, problem, with_pressure=True)
        sol_c.reynolds = float(meta["reynolds"])
        sol_c.amplitude = float(meta["amplitude"])
        Re_c = sol_c.reynolds

        mu_c, eigvec_c = leading_real_mu(problem, sol_c)
        records = meta["records"]

        print(f"  Re_c = {Re_c:.6f}")
        print(f"  mu_c = {mu_c:+.6e}")
        return Re_c, sol_c, eigvec_c, mu_c, records

    print("\n--- Sweep symmetric branch ---")
    branch_ids = snapshot_data["branch_ids"]
    records = sweep_symmetric_branch(
        velocity_snapshots, pressure_snapshots, parameters, branch_ids,
        problem, n_eigenvalues=6,
    )

    print("\n--- Bisect Re_c (pitchfork on symmetric branch) ---")
    Re_arr = np.array([r["Re"] for r in records])
    mu_arr = np.array([
        np.min(r["eigenvalues"].real[np.abs(r["eigenvalues"].imag) < 1e-8])
        for r in records
    ])

    sign_change = np.where(np.diff(np.sign(mu_arr)) < 0)[0]
    if len(sign_change) == 0:
        raise RuntimeError("No sign change in leading real mu across sweep.")

    i_left = int(sign_change[0])
    i_right = i_left + 1
    Re_left, mu_left = float(Re_arr[i_left]), float(mu_arr[i_left])
    Re_right, mu_right = float(Re_arr[i_right]), float(mu_arr[i_right])
    print(f"  bracket: Re in [{Re_left:.3f}, {Re_right:.3f}]")

    def snap_idx(Re_target):
        return int(np.where(
            (branch_ids == 0)
            & (np.abs(parameters[:, 0] - Re_target) < 1e-2)
            & (np.abs(parameters[:, 1]) < 1e-8)
        )[0][0])

    sol_left = snapshot_to_solution(
        velocity_snapshots[snap_idx(Re_left)],
        pressure_snapshots[snap_idx(Re_left)],
        Re_left, 0.0, problem,
    )
    sol_right = snapshot_to_solution(
        velocity_snapshots[snap_idx(Re_right)],
        pressure_snapshots[snap_idx(Re_right)],
        Re_right, 0.0, problem,
    )

    Re_c, sol_c, eigvec_c, mu_c, history = bisect_pitchfork(
        problem,
        Re_left, mu_left, sol_left,
        Re_right, mu_right, sol_right,
        Re_tol=1e-2,
    )
    print(f"  bracket history: {len(history)} points")

    problem.reynolds.assign(float(Re_c))
    problem.amplitude.assign(0.0)

    # Re-solve the symmetric state exactly at Re_c, recompute eigenvector.
    sol_c = solve_symmetric_at_Re(problem, Re_c, sol_c)
    mu_c, eigvec_c = leading_real_mu(problem, sol_c)
    sol_c.reynolds = float(Re_c)
    sol_c.amplitude = 0.0

    print(f"  Re_c = {Re_c:.8f}")
    print(f"  mu_c = {mu_c:+.6e}")

    save_solution(sol_c, RE_CRITICAL_SOL)
    np.savez(
        RE_CRITICAL_SOL + "_metadata.npz",
        reynolds=float(Re_c),
        amplitude=0.0,
        records=records,
    )

    return Re_c, sol_c, eigvec_c, mu_c, records


# =============================================================================
# SYMMETRIC BRANCH  (computed by continuation)
# =============================================================================

def compute_symmetric_branch(problem, sol_seed, Re_min, Re_max, dRe):
    """
    Continue the symmetric (amplitude = 0) steady branch from Re_min to Re_max,
    solving the FOM at each step and recording C_L, C_D.

    Newton stays on the symmetric branch even above Re_c (where it is
    unstable): a symmetric guess never leaves the symmetric subspace.
    """
    problem.amplitude.assign(0.0)

    Re_values = list(np.arange(Re_min, Re_max + 1e-9, dRe))
    if Re_values[-1] < Re_max - 1e-9:
        Re_values.append(float(Re_max))

    records = []
    sol_prev = sol_seed

    for Re in Re_values:
        Re = float(Re)
        problem.reynolds.assign(Re)
        try:
            sol = solve_symmetric_at_Re(problem, Re, sol_prev)
        except Exception as e:
            print(f"  symmetric solve failed at Re={Re:.2f}: "
                  f"{type(e).__name__}: {e}")
            break

        forces = compute_forces(sol, problem)
        records.append({
            "Re": Re,
            "C_L": float(forces.lift),
            "C_D": float(forces.drag),
        })
        sol_prev = sol

    return records


# =============================================================================
# HELPERS
# =============================================================================

def diag_arrays(diag):
    if not diag:
        return np.array([]), np.array([])
    return (
        np.array([d["Re"] for d in diag], dtype=float),
        np.array([d["C_L"] for d in diag], dtype=float),
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    os.makedirs(OUTDIR, exist_ok=True)

    print("--- Setup problem ---")
    problem = setup_navier_stokes_problem(
        mesh_file=MESH_FILE,
        reynolds_init=REYNOLDS_INIT,
        amplitude_init=AMPLITUDE_INIT,
    )

    initial_solution = load_solution(FOM_CHECKPOINT, problem)

    print("\n--- Load snapshots and lifting ---")
    snapshot_data = load_snapshot_dofs(SNAPSHOT_DIR)
    load_lifting_functions(LIFTING_DIR, problem)
    print(f"  Snapshots:  {snapshot_data['velocity'].shape[0]}")
    print(f"  Parameters: {len(snapshot_data['parameters'])} entries")

    Re_c, sol_c, eigvec_c, mu_c, _ = load_or_compute_critical(
        problem, snapshot_data,
    )

    # -------------------------------------------------------------------------
    # Symmetric branch (computed from RE_SYM_MIN to RE_MAX).
    # -------------------------------------------------------------------------

    print("\n--- Compute symmetric branch ---")
    sym_records = compute_symmetric_branch(
        problem, initial_solution, RE_SYM_MIN, RE_MAX, DRE_SYM,
    )
    if not sym_records:
        raise RuntimeError("Symmetric branch continuation produced no points.")
    print(f"  {len(sym_records)} points, "
          f"Re in [{sym_records[0]['Re']:.1f}, {sym_records[-1]['Re']:.1f}], "
          f"max |C_L| = {max(abs(r['C_L']) for r in sym_records):.2e}")

    # -------------------------------------------------------------------------
    # Branch jump.
    # -------------------------------------------------------------------------

    print("\n--- Branch jump from pitchfork ---")
    jump_results = branch_jump_pitchfork(
        problem,
        sol_c,
        eigvec_c,
        Re_c,
        eps_values=FOM_EPS_VALUES,
        amp_values=FOM_AMP_VALUES,
        perturb_pressure=False,
        asymmetry_tol=1e-3,
    )

    print(f"\n  Successful jumps: {len(jump_results)}")
    for j, r in enumerate(jump_results):
        print(
            f"  jump {j}: sign={r['sign']:+d}, Re={r['Re']:.6f}, "
            f"eps={r['epsilon']:.3e}, amp={r['amplitude']:.3e}, "
            f"C_L={r['C_L']:+.6e}"
        )

    # Select by lift sign (robust to arbitrary eigenvector sign).
    pos_seeds = [r for r in jump_results if r["C_L"] > 0.0]
    neg_seeds = [r for r in jump_results if r["C_L"] < 0.0]

    # -------------------------------------------------------------------------
    # Continue asymmetric branches.
    # -------------------------------------------------------------------------

    print("\n--- Continue asymmetric branches ---")
    pos_diag, neg_diag = [], []

    if pos_seeds:
        print("\n  C_L > 0 branch:")
        _, pos_diag = continue_asymmetric_branch(
            problem, pos_seeds[0]["solution"], Re_max=RE_MAX,
        )

    if neg_seeds:
        print("\n  C_L < 0 branch:")
        _, neg_diag = continue_asymmetric_branch(
            problem, neg_seeds[0]["solution"], Re_max=RE_MAX,
        )

    # Fallback: mirror the found branch if the solver only got one
    # (exact pitchfork symmetry: C_L -> -C_L at the same Re).
    pos_mirror = neg_mirror = False

    if pos_diag and not neg_diag:
        neg_diag = [
            {"Re": d["Re"], "C_L": -d["C_L"], "C_D": d.get("C_D", np.nan)}
            for d in pos_diag
        ]
        neg_mirror = True
        print("\n  WARNING: C_L<0 branch not found by solver; "
              "mirroring C_L>0 branch.")
    elif neg_diag and not pos_diag:
        pos_diag = [
            {"Re": d["Re"], "C_L": -d["C_L"], "C_D": d.get("C_D", np.nan)}
            for d in neg_diag
        ]
        pos_mirror = True
        print("\n  WARNING: C_L>0 branch not found by solver; "
              "mirroring C_L<0 branch.")
    elif not pos_diag and not neg_diag:
        raise RuntimeError("No asymmetric branch found; check jump settings.")

    if pos_diag:
        print(f"  C_L>0 branch: {len(pos_diag)} points, "
              f"Re in [{pos_diag[0]['Re']:.2f}, {pos_diag[-1]['Re']:.2f}]")
    if neg_diag:
        print(f"  C_L<0 branch: {len(neg_diag)} points, "
              f"Re in [{neg_diag[0]['Re']:.2f}, {neg_diag[-1]['Re']:.2f}]")

    # -------------------------------------------------------------------------
    # Save data.
    # -------------------------------------------------------------------------

    print("\n--- Save data ---")
    sym_Re, sym_CL = diag_arrays(sym_records)
    pos_Re, pos_CL = diag_arrays(pos_diag)
    neg_Re, neg_CL = diag_arrays(neg_diag)

    npz_path = os.path.join(OUTDIR, NPZ_NAME)
    np.savez(
        npz_path,
        Re_c=Re_c,
        mu_c=mu_c,
        sym_Re=sym_Re,
        sym_CL=sym_CL,
        pos_Re=pos_Re,
        pos_CL=pos_CL,
        neg_Re=neg_Re,
        neg_CL=neg_CL,
        pos_mirror=pos_mirror,
        neg_mirror=neg_mirror,
    )
    print(f"  wrote {npz_path}")
    print("\nDone. Plot with: python plot_bifurcation_diagram.py "
          f"{npz_path}")


if __name__ == "__main__":
    main()
