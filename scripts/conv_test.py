"""Bare amp-continuation for the fluidic pinball local ROM.

Start from a full-space solution that already sits on one branch at
(Re, amp=0) and walk amp downward (0 -> -1.0) at fixed Re by continuation.

Selection is bare nearest-whitened-centroid. Warm start is the previous
converged reduced state; on a cluster switch we re-project the previous
full-space solution into the new cluster's basis (projection warm start),
which avoids depending on the change-of-basis path.
"""

import time
import numpy as np

from nsrom.navier_stokes import compute_forces
from nsrom.rom import (
    solve_rom,
    reconstruct_solution,
    project_to_rom_coefficients,
    SubMeshDEIM,
)
from nsrom.local_rom import select_cluster, _whitened_pod_distances


# ---------------------------------------------------------------------------
# bare selector: nearest whitened centroid, nothing else
# ---------------------------------------------------------------------------
def select_nearest_cluster(clustering, a_prev, k_prev, verbose=False):
    """argmin over clusters of the whitened global-POD distance. No tau, no veto."""
    d = _whitened_pod_distances(clustering, a_prev, k_prev)      # (K,)
    k = int(np.argmin(d))
    if verbose:
        marks = " ".join(f"k{j}={d[j]:.4f}" + ("*" if j == k else "")
                         for j in range(len(d)))
        print(f"    [select] {marks}")
    return k


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------
def continue_amp_bare(
    Re, amp_values,
    clustering, operators, deim_ops_dict,
    problem, initial_solution, M_u, M_p,
    use_deim=True,
    *,
    start_cluster=None,
    verbose=True,
):
    """Continuation in amp at fixed Re, seeded from ``initial_solution``.

    Parameters
    ----------
    Re : float
        Fixed Reynolds number for the whole sweep (e.g. 99).
    amp_values : iterable of float
        Amplitudes to solve, e.g. ``np.linspace(0.0, -1.0, 11)``. They are
        sorted descending internally so the walk always goes 0 -> -1.0.
    initial_solution :
        Full-space solution already on the desired branch at (Re, amp=0).
    start_cluster : int or None
        Cluster whose basis the initial solution is first projected into.
        If None, picked from the parameters via ``select_cluster(clustering, Re, 0)``.
        NOTE: at (Re=99, amp=0) up to three branches coexist and the
        parameter-based default cannot tell them apart -- if the initial
        solution is on a specific asymmetric branch, pass start_cluster
        explicitly so the seed lands in the right basis.

    Returns
    -------
    list of dict, one record per amp.
    """
    nu = 1.0 / Re
    problem.reynolds.assign(Re)

    # amp schedule: always walk downward, 0 -> -1.0
    amps = sorted((float(a) for a in amp_values), reverse=True)

    # build the submesh DEIM helpers once per cluster
    submesh_deims = {}
    if use_deim:
        for k in range(clustering.n_clusters):
            if deim_ops_dict.get(k) is not None:
                submesh_deims[k] = SubMeshDEIM(problem.velocity_space, deim_ops_dict[k])

    # --- seed the continuation from the initial full-space solution ---
    if start_cluster is None:
        start_cluster = select_cluster(clustering, Re, 0.0)

    init_vel  = initial_solution.velocity.dat.data_ro.flatten()
    init_pres = initial_solution.pressure.dat.data_ro.flatten()
    prev_u_coeffs, prev_p_coeffs = project_to_rom_coefficients(
        init_vel, init_pres, operators[start_cluster], amp=0.0, M_u=M_u, M_p=M_p)

    prev_k            = start_cluster
    prev_amp          = 0.0
    prev_solution_rom = initial_solution     # full-space state to re-project on a switch

    records = []

    for amp in amps:
        # bare selection from the previous reduced state
        k = select_nearest_cluster(clustering, prev_u_coeffs, prev_k, verbose=verbose)

        # warm start: carry coefficients within a cluster; re-project on a switch
        if k == prev_k:
            u_guess, p_guess = prev_u_coeffs, prev_p_coeffs
        else:
            pv = prev_solution_rom.velocity.dat.data_ro.flatten()
            pp = prev_solution_rom.pressure.dat.data_ro.flatten()
            u_guess, p_guess = project_to_rom_coefficients(
                pv, pp, operators[k], amp=prev_amp, M_u=M_u, M_p=M_p)
            if verbose:
                print(f"    [switch] cluster {prev_k} -> {k} (re-projected warm start)")

        # --- ROM solve at this amp ---
        t0 = time.perf_counter()
        # solution_no_deim, w_rom_no_deim,callback_data_no_deim= solve_rom(
        #     operators=operators[k], nu=nu, amp=amp, problem=problem,
        #     deim_ops=None,
        #     u_initial_guess=u_guess, p_initial_guess=p_guess,
        #     submesh_deim=None, return_internals=True)

        solution, w_rom, callback_data = solve_rom(
            operators=operators[k], nu=nu, amp=amp, problem=problem,
            deim_ops=deim_ops_dict[k],
            u_initial_guess=u_guess, p_initial_guess=p_guess,
            submesh_deim=submesh_deims.get(k), return_internals=True, verbose=True,)
        t_rom = time.perf_counter() - t0
        print(f"use_tensor={callback_data['use_tensor']}  use_deim={callback_data['use_deim']}")
        print(f"mode={solution.metadata['mode']}")     # 'tensor' | 'deim' | 'exact'

        solution_rom = reconstruct_solution(solution, operators[k], problem, amp=amp)

        record = {
            "Re": Re, "amp": amp, "cluster": k,
            "converged": bool(solution.converged), "t_rom": t_rom,
        }

        if solution.converged:
            problem.amplitude.assign(amp)
            forces = compute_forces(solution_rom, problem)
            record["C_D"]     = float(forces.drag)
            record["C_L"]     = float(forces.lift)
            record["C_L_rom"] = float(forces.lift)

            # advance the continuation ONLY from a converged point
            prev_k            = k
            prev_u_coeffs     = solution.velocity_coeffs
            prev_p_coeffs     = solution.pressure_coeffs
            prev_amp          = amp
            prev_solution_rom = solution_rom
            if verbose:
                print(f"  amp={amp:+.3f}  k={k}  C_L={forces.lift:+.4f}  ({t_rom:.3f}s)")
        else:
            if verbose:
                print(f"  amp={amp:+.3f}  k={k}  NOT CONVERGED -- keeping previous state")

        records.append(record)

    return records


# ---------------------------------------------------------------------------
# example call
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # records = continue_amp_bare(
    #     Re=99.0,
    #     amp_values=np.linspace(0.0, -1.0, 11),   # 0, -0.1, ..., -1.0
    #     clustering=clustering,
    #     operators=operators,
    #     deim_ops_dict=deim_ops_dict,
    #     problem=problem,
    #     initial_solution=initial_solution,       # full solution at (Re=99, amp=0)
    #     M_u=M_u, M_p=M_p,
    #     start_cluster=1,                          # set to the branch's cluster
    # )
    pass
