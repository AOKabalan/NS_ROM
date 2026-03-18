"""Parameter sweep with continuation for local ROM."""

import numpy as np
import time

from Navier_Stokes import solve_steady_navier_stokes, compute_forces
from rom import solve_rom, reconstruct_solution, project_to_rom_coefficients, SubMeshDEIM
from cluster_building import select_cluster, select_cluster_reduced, apply_change_of_basis


def build_snake_path(Re_values, amp_values):
    """Snake-ordered sweep: alternates amp direction each Re row."""
    path = []
    for i, Re in enumerate(Re_values):
        amps = amp_values if i % 2 == 0 else amp_values[::-1]
        for amp in amps:
            path.append((float(Re), float(amp)))
    return path


def make_thetas(amp, n_lifting):
    """Build thetas from amplitude. For pinball: [1.0, amp]."""
    if n_lifting == 2:
        return [1.0, amp]
    elif n_lifting == 1:
        return [1.0]
    else:
        raise ValueError(f"make_thetas not defined for n_lifting={n_lifting}")


def run_sweep(path, clustering, operators, deim_ops_dict,
              problem, initial_solution, M_u, M_p,
              use_deim=True, fom_every=0, use_change_of_basis = True):

    results = []
    prev_k = None
    prev_u_coeffs = None
    prev_p_coeffs = None
    prev_amp = None
    prev_solution_rom = None

    n_lifting = operators[0].n_lifting

    submesh_deims = {}
    if use_deim:
        for k in range(clustering.n_clusters):
            if deim_ops_dict.get(k) is not None:
                submesh_deims[k] = SubMeshDEIM(problem.velocity_space, deim_ops_dict[k])

    n_total = len(path)

    for i, (Re, amp) in enumerate(path):
        nu = 1.0 / Re

        # --- Cluster selection ---
        if prev_u_coeffs is not None and clustering.fast_selection is not None:
            k = select_cluster_reduced(clustering, prev_u_coeffs, prev_k)
        else:
            k = select_cluster(clustering, Re, amp)
        cluster_switched = (k != prev_k)

        # --- Initial guess ---
        if prev_u_coeffs is None:
            src_vel  = initial_solution.velocity.dat.data_ro.flatten()
            src_pres = initial_solution.pressure.dat.data_ro.flatten()
            u_coeffs, p_coeffs = project_to_rom_coefficients(
                src_vel, src_pres, operators[k], amp=amp, M_u=M_u, M_p=M_p,
            )
        elif cluster_switched and use_change_of_basis and clustering.change_of_basis is not None:
            thetas_old = make_thetas(prev_amp, n_lifting)
            thetas_new = make_thetas(amp, n_lifting)
            u_coeffs, p_coeffs = apply_change_of_basis(
                clustering, prev_k, k, prev_u_coeffs, prev_p_coeffs,
                thetas_old, thetas_new,
            )
        elif cluster_switched:
            src_vel  = prev_solution_rom.velocity.dat.data_ro.flatten()
            src_pres = prev_solution_rom.pressure.dat.data_ro.flatten()
            u_coeffs, p_coeffs = project_to_rom_coefficients(
                src_vel, src_pres, operators[k], amp=amp, M_u=M_u, M_p=M_p,
            )
        else:
            u_coeffs = prev_u_coeffs
            p_coeffs = prev_p_coeffs

        # --- ROM solve ---
        t0 = time.perf_counter()
        solution = solve_rom(
            operators=operators[k],
            nu=nu,
            amp=amp,
            problem=problem,
            deim_ops=deim_ops_dict[k],
            u_initial_guess=u_coeffs,
            p_initial_guess=p_coeffs,
            submesh_deim=submesh_deims.get(k),
        )
        t_rom = time.perf_counter() - t0

        solution_rom = reconstruct_solution(solution, operators[k], problem, amp=amp)

        prev_k = k
        # if solution.converged :
        prev_u_coeffs = solution.velocity_coeffs
        prev_p_coeffs = solution.pressure_coeffs
        prev_amp = amp
        prev_solution_rom = solution_rom

        record = {
            'Re': Re, 'amp': amp, 'cluster': k,
            'converged': solution.converged,
            'iterations': solution.iterations,
            't_rom': t_rom,
            'cluster_switched': cluster_switched,
        }

        # --- Forces (for every converged point) ---
        if solution.converged:
            problem.reynolds.assign(Re)
            problem.amplitude.assign(amp)
            forces = compute_forces(solution_rom, problem)
            record['C_D'] = forces.drag
            record['C_L'] = forces.lift

        # --- Optional FOM comparison ---
        if fom_every > 0 and (i % fom_every == 0):
            problem.reynolds.assign(Re)
            problem.amplitude.assign(amp)

            t0 = time.perf_counter()
            hf = solve_steady_navier_stokes(problem, initial_solution.solution)
            t_fom = time.perf_counter() - t0

            u_fom = hf.velocity.dat.data_ro.flatten()
            p_fom = hf.pressure.dat.data_ro.flatten()
            u_rom_dofs = solution_rom.velocity.dat.data_ro.flatten()
            p_rom_dofs = solution_rom.pressure.dat.data_ro.flatten()

            norm_M = lambda v, M: np.sqrt(float(v @ (M @ v)))
            record['u_err']   = norm_M(u_fom - u_rom_dofs, M_u) / norm_M(u_fom, M_u)
            record['p_err']   = norm_M(p_fom - p_rom_dofs, M_p) / norm_M(p_fom, M_p)
            record['t_fom']   = t_fom
            record['speedup'] = t_fom / t_rom if t_rom > 0 else float('inf')

        results.append(record)

        status = "✓" if solution.converged else "✗"
        err_str = f"u_err={record['u_err']*100:.3f}%" if 'u_err' in record else ""
        print(f"  [{i+1:3d}/{n_total}] Re={Re:6.1f} amp={amp:+.2f} "
              f"k={k} {status} it={solution.iterations:2d} "
              f"t={t_rom:.3f}s {'SW' if cluster_switched else '  '} {err_str}")

    return results


def save_sweep_results(results, save_path):
    import csv
    all_keys = list(dict.fromkeys(k for r in results for k in r.keys()))
    with open(save_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(results)
    print(f"  Sweep saved to {save_path}")

