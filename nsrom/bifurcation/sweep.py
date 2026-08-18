"""Parameter sweep with continuation for local ROM."""

import numpy as np
import time

from nsrom.navier_stokes import solve_steady_navier_stokes, compute_forces
from nsrom.rom import solve_rom, reconstruct_solution, project_to_rom_coefficients, SubMeshDEIM
from nsrom.rom.local import (
    select_cluster,
    select_cluster_reduced,
    select_cluster_reduced_hysteresis,
    select_the_cluster,
    apply_change_of_basis,
    select_cluster_predicted,
    accept_switch_by_projection,
    diagnose_clusters,
    select_cluster_reduced_projection,
    classify_state_for_seed,
)
from nsrom.bifurcation.branch_jump import compute_rom_indicator,in_sweep_branch_jump_rom
from nsrom.bifurcation.detection import find_mu_bracket


def build_fixed_amp_path(Re_values, amp):
    """Sweep Re at a fixed amplitude."""
    return [(float(Re), float(amp)) for Re in Re_values]

def build_fixed_re_path(amp_values, Re):
    """Sweep amp at a fixed Reynolds number."""
    return [(float(Re), float(amp)) for amp in amp_values]

def build_hubspoke_path(Re_values, amp_values):
    """
    For each Re, anchor at amp=0 first (using previous Re's amp=0 solution),
    then sweep positive amps, then negative amps.
    amp_values should include 0 or be split around it.
    """
    pos_amps = sorted([a for a in amp_values if a >= 0])
    neg_amps = sorted([a for a in amp_values if a < 0], reverse=True)  # 0 → -1

    path = []
    for Re in Re_values:
        # positive sweep (includes 0 as anchor)
        for amp in pos_amps:
            path.append((float(Re), float(amp)))
        # negative sweep from 0 downward
        for amp in neg_amps:
            path.append((float(Re), float(amp)))
    return path

def build_snake_path(Re_values, amp_values):
    """Snake-ordered sweep: alternates amp direction each Re row."""
    path = []
    for i, Re in enumerate(Re_values):
        amps = amp_values if i % 2 == 0 else amp_values[::-1]
        for amp in amps:
            path.append((float(Re), float(amp)))
    return path

def simple_path(Re_values, amp_values):
    path = []


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
    prev_Re = None
    # anchor_coeffs[Re] = (u_coeffs, p_coeffs) at amp=0 for that Re
    anchor_coeffs = {}

    n_lifting = operators[0].n_lifting

    submesh_deims = {}
    if use_deim:
        for k in range(clustering.n_clusters):
            if deim_ops_dict.get(k) is not None:
                submesh_deims[k] = SubMeshDEIM(problem.velocity_space, deim_ops_dict[k])

    n_total = len(path)

    for i, (Re, amp) in enumerate(path):
        nu = 1.0 / Re
        new_re_row = (Re != prev_Re)
        # --- Cluster selection ---
        if prev_u_coeffs is not None and clustering.fast_selection is not None:
            k = select_cluster_reduced(clustering, prev_u_coeffs, prev_k)
        else:
            # k,_ = select_cluster(clustering, Re, amp)
            k = select_the_cluster(clustering, Re, amp, initial_solution, M_u)

        cluster_switched = (k != prev_k)

        # --- Initial guess ---
        if prev_u_coeffs is None:
            src_vel  = initial_solution.velocity.dat.data_ro.flatten()
            src_pres = initial_solution.pressure.dat.data_ro.flatten()
            u_coeffs, p_coeffs = project_to_rom_coefficients(
                src_vel, src_pres, operators[k], amp=amp, M_u=M_u, M_p=M_p,
            )
        elif new_re_row and Re in anchor_coeffs:
            # restart from amp=0 solution of this Re
            u_coeffs, p_coeffs = anchor_coeffs[Re]
            if cluster_switched and use_change_of_basis and clustering.change_of_basis is not None:
                thetas_old = make_thetas(0.0, n_lifting)
                thetas_new = make_thetas(amp, n_lifting)
                u_coeffs, p_coeffs = apply_change_of_basis(
                    clustering, prev_k, k, u_coeffs, p_coeffs,
                    thetas_old, thetas_new,
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
        # --- Store amp=0 anchor for this Re ---
        if amp == 0.0 and solution.converged:
            anchor_coeffs[Re] = (solution.velocity_coeffs, solution.pressure_coeffs)

        prev_k = k
        prev_u_coeffs = solution.velocity_coeffs
        prev_p_coeffs = solution.pressure_coeffs
        prev_amp = amp
        prev_Re = Re
        prev_solution_rom = solution_rom

        record = {
            'Re': Re, 'amp': amp, 'cluster': k,
            'converged': solution.converged,
            'iterations': solution.iterations,
            't_rom': t_rom,
            'cluster_switched': cluster_switched,
            'new_re_row': new_re_row,
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

        status  = "✓" if solution.converged else "✗"
        err_str = f"u_err={record['u_err']*100:.3f}%" if 'u_err' in record else ""
        anchor_str = " [anchor]" if amp == 0.0 else ""
        print(f"  [{i+1:3d}/{n_total}] Re={Re:6.1f} amp={amp:+.2f} "
              f"k={k} {status} it={solution.iterations:2d} "
              f"t={t_rom:.3f}s {'SW' if cluster_switched else '  '}{anchor_str} {err_str}")

    return results


def save_sweep_results(results, save_path):
    import csv
    all_keys = list(dict.fromkeys(k for r in results for k in r.keys()))
    with open(save_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(results)
    print(f"  Sweep saved to {save_path}")

def run_bifurcation_sweep(
    path,
    clustering,
    operators,
    deim_ops_dict,
    problem,
    initial_solution,
    M_u,
    M_p,
    M_uu_red,
    use_deim=True,
    fom_every=0,
    use_change_of_basis=True,
    *,
    n_eigenvalues=6,
    target=0.0,
    tau=0.8,
    eps_tie=1e-2,
    select_verbose=False,
):

    results = []
    prev_k = None
    prev_u_coeffs = None
    prev_p_coeffs = None
    prev_amp = None
    prev_solution_rom = None
    prev_Re = None

    # two-back state for the secant predictor
    prev_prev_k = None
    prev_prev_u_coeffs = None
    last_step_restarted = False

    # anchor_coeffs[Re] = (u_coeffs, p_coeffs) at amp=0 for that Re
    anchor_coeffs = {}

    n_lifting = operators[0].n_lifting

    submesh_deims = {}
    if use_deim:
        for k in range(clustering.n_clusters):
            if deim_ops_dict.get(k) is not None:
                submesh_deims[k] = SubMeshDEIM(problem.velocity_space, deim_ops_dict[k])

    n_total = len(path)


    for i, (Re, amp) in enumerate(path):
        nu = 1.0 / Re
        new_re_row = (Re != prev_Re)

        # --- Cluster selection ---
        can_predict = (
            prev_u_coeffs is not None
            and prev_prev_u_coeffs is not None
            and not last_step_restarted
            and clustering.change_of_basis is not None
        )

        if can_predict:
            k = select_cluster_predicted(
                clustering,
                a_cur=prev_u_coeffs, k_cur=prev_k,
                a_old=prev_prev_u_coeffs, k_old=prev_prev_k,
                Re=Re, amp=amp,
                tau=tau, eps_tie=eps_tie,
                verbose=select_verbose,
            )

        elif prev_u_coeffs is not None:
            # one solved state only (or just restarted): stay put
            k = prev_k
        else:
            # cold start: parameter+ centroid -space selection
            k = select_the_cluster(clustering, Re, amp, initial_solution, M_u)
        cluster_switched = (k != prev_k)

        # --- Initial guess ---
        restarted = False
        if prev_u_coeffs is None:
            src_vel = initial_solution.velocity.dat.data_ro.flatten()
            src_pres = initial_solution.pressure.dat.data_ro.flatten()
            u_coeffs, p_coeffs = project_to_rom_coefficients(
                src_vel, src_pres, operators[k], amp=amp, M_u=M_u, M_p=M_p,
            )
        elif new_re_row and Re in anchor_coeffs:
            # restart from amp=0 solution of this Re (breaks the secant)
            restarted = True
            u_coeffs, p_coeffs = anchor_coeffs[Re]
            if cluster_switched and use_change_of_basis and clustering.change_of_basis is not None:
                thetas_old = make_thetas(0.0, n_lifting)
                thetas_new = make_thetas(amp, n_lifting)
                u_coeffs, p_coeffs = apply_change_of_basis(
                    clustering, prev_k, k, u_coeffs, p_coeffs,
                    thetas_old, thetas_new,
                )
        elif cluster_switched and use_change_of_basis and clustering.change_of_basis is not None:
            thetas_old = make_thetas(prev_amp, n_lifting)
            thetas_new = make_thetas(amp, n_lifting)
            u_coeffs, p_coeffs = apply_change_of_basis(
                clustering, prev_k, k, prev_u_coeffs, prev_p_coeffs,
                thetas_old, thetas_new,
            )
        elif cluster_switched:
            src_vel = prev_solution_rom.velocity.dat.data_ro.flatten()
            src_pres = prev_solution_rom.pressure.dat.data_ro.flatten()
            u_coeffs, p_coeffs = project_to_rom_coefficients(
                src_vel, src_pres, operators[k], amp=amp, M_u=M_u, M_p=M_p,
            )

        else:
            u_coeffs = prev_u_coeffs
            p_coeffs = prev_p_coeffs

        # --- ROM solve (with internals for the indicator) ---
        t0 = time.perf_counter()
        rom_output = solve_rom(
            operators=operators[k],
            nu=nu,
            amp=amp,
            problem=problem,
            deim_ops=deim_ops_dict[k],
            u_initial_guess=u_coeffs,
            p_initial_guess=p_coeffs,
            submesh_deim=submesh_deims.get(k),
            return_internals=True,
        )
        t_rom = time.perf_counter() - t0
        solution, w_rom, callback_data = rom_output

        solution_rom = reconstruct_solution(solution, operators[k], problem, amp=amp)

        # --- ROM bifurcation indicator ---
        if solution.converged:
            rom_ind = compute_rom_indicator(
                w_rom, callback_data,
                operators[k], M_uu_red[k],
                n_eigenvalues=n_eigenvalues,
                target=target,
            )
            mu = rom_ind['mu'] if rom_ind is not None else float('nan')
        else:
            mu = float('nan')

        # --- Store amp=0 anchor for this Re ---
        if amp == 0.0 and solution.converged:
            anchor_coeffs[Re] = (solution.velocity_coeffs, solution.pressure_coeffs)

        # --- History update (shift one-back into two-back) ---
        prev_prev_k = prev_k
        prev_prev_u_coeffs = prev_u_coeffs
        prev_k = k
        prev_u_coeffs = solution.velocity_coeffs
        prev_p_coeffs = solution.pressure_coeffs
        prev_amp = amp
        prev_Re = Re
        prev_solution_rom = solution_rom
        last_step_restarted = restarted

        record = {
            'Re': Re, 'amp': amp, 'cluster': k,
            'converged': solution.converged,
            'iterations': solution.iterations,
            't_rom': t_rom,
            'cluster_switched': cluster_switched,
            'new_re_row': new_re_row,
            'mu': mu,
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
        anchor_str = " [anchor]" if amp == 0.0 else ""
        mu_str = f"mu={mu:+.3e}" if np.isfinite(mu) else "mu=    nan   "
        print(f"  [{i+1:3d}/{n_total}] Re={Re:6.1f} amp={amp:+.2f} "
              f"k={k} {status} it={solution.iterations:2d} "
              f"{mu_str} t={t_rom:.3f}s "
              f"{'SW' if cluster_switched else '  '}{anchor_str} {err_str}")

    # -----------------------------------------------------------------
    # Sign-change detection (per amplitude)
    # -----------------------------------------------------------------
    crossings = []
    amps_in_sweep = sorted({r['amp'] for r in results})
    for amp_target in amps_in_sweep:
        amp_records = [r for r in results if r['amp'] == amp_target]
        left, right = find_mu_bracket(amp_records, mu_key='mu')
        if left is None:
            continue

        Re_l = float(left['Re'])
        mu_l = float(left['mu'])
        Re_r = float(right['Re'])
        mu_r = float(right['mu'])

        if left is right:
            Re_c_linear = Re_l
        else:
            Re_c_linear = Re_l - mu_l * (Re_r - Re_l) / (mu_r - mu_l)

        crossings.append({
            'amp':         float(amp_target),
            'Re_left':     Re_l,
            'mu_left':     mu_l,
            'Re_right':    Re_r,
            'mu_right':    mu_r,
            'Re_c_linear': float(Re_c_linear),
        })

    if crossings:
        print("\n" + "=" * 70)
        print("EIGENVALUE CROSSINGS  (linear interpolation, no refinement)")
        print("=" * 70)
        for c in crossings:
            print(f"  amp={c['amp']:+.3f}  "
                  f"bracket [{c['Re_left']:.3f}, {c['Re_right']:.3f}]  "
                  f"mu [{c['mu_left']:+.3e}, {c['mu_right']:+.3e}]  "
                  f"Re_c ≈ {c['Re_c_linear']:.4f}")

    return results, crossings

"""
Diagnostic continuation sweep.

Same continuation logic as run_bifurcation_sweep, but at each step with a
previous state it (a) force-solves every cluster to get the per-cluster
eigenvalue mu, then (b) calls diagnose_clusters to print the table.

The solving lives HERE, not in diagnose_clusters (which is pure).

"""


def run_sweep_diagnostic(
    path,
    clustering,
    operators,
    deim_ops_dict,
    problem,
    initial_solution,
    M_u,
    M_p,
    M_uu_red,
    use_deim=True,
    fom_every=0,
    use_change_of_basis=True,
    *,
    n_eigenvalues=6,
    target=0.0,
    tau=0.8,
    eps_tie=1e-2,
    select_verbose=False,
    diag_eigenvalues=True,
):
    results = []
    prev_k = None
    prev_u_coeffs = None
    prev_p_coeffs = None
    prev_amp = None
    prev_solution_rom = None
    prev_Re = None
    # two-back state for the secant predictor
    prev_prev_k = None
    prev_prev_u_coeffs = None
    last_step_restarted = False
    n_lifting = operators[0].n_lifting
    submesh_deims = {}
    if use_deim:
        for k in range(clustering.n_clusters):
            if deim_ops_dict.get(k) is not None:
                submesh_deims[k] = SubMeshDEIM(problem.velocity_space, deim_ops_dict[k])
    n_total = len(path)
    K = clustering.n_clusters

    for i, (Re, amp) in enumerate(path):
        nu = 1.0 / Re

        can_predict = (
            prev_u_coeffs is not None
            and prev_prev_u_coeffs is not None
            and not last_step_restarted
            and clustering.change_of_basis is not None
        )

        # --- diagnostics (whenever we have a current state) ---
        if prev_u_coeffs is not None:
            print(f"\n[{i + 1}/{n_total}] Re={Re:.3f}, amp={amp:.3f}")

            # per-cluster eigenvalues: force-solve each cluster HERE.
            mu_diag   = np.full(K, np.nan)
            conv_diag = np.zeros(K, dtype=bool)
            iter_diag = np.full(K, -1, dtype=int)
            if diag_eigenvalues:
                src_vel  = initial_solution.velocity.dat.data_ro.flatten()
                src_pres = initial_solution.pressure.dat.data_ro.flatten()
                for kk in range(K):
                    u0, p0 = project_to_rom_coefficients(
                        src_vel, src_pres, operators[kk], amp=amp, M_u=M_u, M_p=M_p,
                    )
                    out_kk = solve_rom(
                        operators=operators[kk],
                        nu=nu, amp=amp, problem=problem,
                        deim_ops=deim_ops_dict[kk],
                        u_initial_guess=u0, p_initial_guess=p0,
                        submesh_deim=submesh_deims.get(kk),
                        return_internals=True,
                    )
                    sol_kk, w_kk, cb_kk = out_kk
                    conv_diag[kk] = bool(sol_kk.converged)
                    iter_diag[kk] = int(sol_kk.iterations)
                    if sol_kk.converged:
                        ind = compute_rom_indicator(
                            w_kk, cb_kk, operators[kk], M_uu_red[kk],
                            n_eigenvalues=n_eigenvalues, target=target,
                        )
                        if ind is not None:
                            mu_diag[kk] = float(ind['mu'])


            diagnose_clusters(
                clustering,
                a_cur=prev_u_coeffs, k_cur=prev_k,
                a_old=prev_prev_u_coeffs if can_predict else None,
                k_old=prev_prev_k if can_predict else None,
                mu=(mu_diag if diag_eigenvalues else None),
                converged=(conv_diag if diag_eigenvalues else None),
                iterations=(iter_diag if diag_eigenvalues else None),
                Re=Re, amp=amp, verbose=True,
            )

        # --- cluster selection ---
        if can_predict:
            k= select_cluster_reduced_hysteresis(clustering, prev_u_coeffs, prev_k, tau=tau, verbose=True)
            # k = select_cluster_predicted(
            #     clustering,
            #     prev_u_coeffs, prev_k,
            #     prev_prev_u_coeffs, prev_prev_k,
            #     Re=Re, amp=amp, tau=tau, eps_tie=eps_tie,
            #     verbose=select_verbose,
            # )
        elif prev_u_coeffs is not None:
            k = prev_k
        else:
            k = select_the_cluster(clustering, Re, amp, initial_solution, M_u)
        cluster_switched = (k != prev_k)

        # --- initial guess ---
        if prev_u_coeffs is None:
            src_vel = initial_solution.velocity.dat.data_ro.flatten()
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
            src_vel = prev_solution_rom.velocity.dat.data_ro.flatten()
            src_pres = prev_solution_rom.pressure.dat.data_ro.flatten()
            u_coeffs, p_coeffs = project_to_rom_coefficients(
                src_vel, src_pres, operators[k], amp=amp, M_u=M_u, M_p=M_p,
            )
        else:
            u_coeffs = prev_u_coeffs
            p_coeffs = prev_p_coeffs

        # --- ROM solve (continuation; with internals for the indicator) ---
        t0 = time.perf_counter()
        rom_output = solve_rom(
            operators=operators[k],
            nu=nu,
            amp=amp,
            problem=problem,
            deim_ops=deim_ops_dict[k],
            u_initial_guess=u_coeffs,
            p_initial_guess=p_coeffs,
            submesh_deim=submesh_deims.get(k),
            return_internals=True,
        )
        t_rom = time.perf_counter() - t0
        solution, w_rom, callback_data = rom_output

        solution_rom = reconstruct_solution(solution, operators[k], problem, amp=amp)

        # --- ROM bifurcation indicator (chosen cluster) ---
        if solution.converged:
            rom_ind = compute_rom_indicator(
                w_rom, callback_data,
                operators[k], M_uu_red[k],
                n_eigenvalues=n_eigenvalues, target=target,
            )
            mu = rom_ind['mu'] if rom_ind is not None else float('nan')
        else:
            mu = float('nan')

        record = {
            'Re': Re, 'amp': amp, 'cluster': k,
            'converged': solution.converged,
            'iterations': solution.iterations,
            't_rom': t_rom,
            'cluster_switched': cluster_switched,
            'mu': mu,
        }
        if solution.converged:
            problem.reynolds.assign(Re)
            problem.amplitude.assign(amp)
            forces = compute_forces(solution_rom, problem)
            record['C_D'] = forces.drag
            record['C_L'] = forces.lift
        results.append(record)

        # --- history update ---
        prev_prev_k = prev_k
        prev_prev_u_coeffs = prev_u_coeffs
        prev_k = k
        prev_u_coeffs = solution.velocity_coeffs
        prev_p_coeffs = solution.pressure_coeffs
        prev_amp = amp
        prev_Re = Re
        prev_solution_rom = solution_rom
        last_step_restarted = False

    return results

def run_sweep_simple(
    path,
    clustering,
    operators,
    deim_ops_dict,
    problem,
    initial_solution,
    M_u,
    M_p,
    M_uu_red,
    use_deim=True,
    use_change_of_basis=True,
    *,
    n_eigenvalues=6,
    target=0.0,
    tau=0.8,
    verbose=True,
):
    results = []
    prev_k = None
    prev_u_coeffs = None
    prev_p_coeffs = None
    prev_amp = None
    prev_solution_rom = None
    prev_mu = None

    n_lifting = operators[0].n_lifting
    submesh_deims = {}
    if use_deim:
        for k in range(clustering.n_clusters):
            if deim_ops_dict.get(k) is not None:
                submesh_deims[k] = SubMeshDEIM(problem.velocity_space, deim_ops_dict[k])

    for i, (Re, amp) in enumerate(path):
        nu = 1.0 / Re

        # --- cluster selection ---
        if prev_u_coeffs is None:
            k = select_the_cluster(clustering, Re, amp, initial_solution, M_u)
        else:
            k = select_cluster_reduced_hysteresis(clustering, prev_u_coeffs, prev_k, tau=tau)
        cluster_switched = (k != prev_k)

        # --- initial guess ---
        if prev_u_coeffs is None:
            src_vel = initial_solution.velocity.dat.data_ro.flatten()
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
            src_vel = prev_solution_rom.velocity.dat.data_ro.flatten()
            src_pres = prev_solution_rom.pressure.dat.data_ro.flatten()
            u_coeffs, p_coeffs = project_to_rom_coefficients(
                src_vel, src_pres, operators[k], amp=amp, M_u=M_u, M_p=M_p,
            )
        else:
            u_coeffs = prev_u_coeffs
            p_coeffs = prev_p_coeffs

        # --- ROM solve ---
        t0 = time.perf_counter()
        solution, w_rom, callback_data = solve_rom(
            operators=operators[k], nu=nu, amp=amp, problem=problem,
            deim_ops=deim_ops_dict[k],
            u_initial_guess=u_coeffs, p_initial_guess=p_coeffs,
            submesh_deim=submesh_deims.get(k),
            return_internals=True,
        )
        t_rom = time.perf_counter() - t0
        solution_rom = reconstruct_solution(solution, operators[k], problem, amp=amp)

        # --- bifurcation indicator ---
        mu = float('nan')
        nu_red = None
        if solution.converged:
            rom_ind = compute_rom_indicator(
                w_rom, callback_data, operators[k], M_uu_red[k],
                n_eigenvalues=n_eigenvalues, target=target,
            )
            if rom_ind is not None:
                mu = float(rom_ind['mu'])
                nu_red = rom_ind['nu_red']

        # --- branch-jump hook ---------------------------------------------
        # mu crossed from positive to <= 0  => symmetric branch (this cluster)
        # just lost stability. This is where the branch jump goes next.
        crossed = (
            prev_mu is not None and np.isfinite(prev_mu)
            and np.isfinite(mu) and prev_mu > 0.0 >= mu
        )
        if crossed:
            print(f"  [bifurcation] mu crossed zero at Re={Re:.3f}  "
                  f"({prev_mu:.3e} -> {mu:.3e})")
            EPS = 0.8

            for s in (+1, -1):
                res = in_sweep_branch_jump_rom(
                    u_rom       = solution.velocity_coeffs,
                    p_rom       = solution.pressure_coeffs,
                    Re          = Re,
                    amp         = amp,
                    operators_k = operators[k],
                    M_uu_red_k  = M_uu_red[k],
                    problem     = problem,
                    eps         = EPS,
                    sign        = s,
                    indicator   = rom_ind,
                    offset      = 0,
                    deim_ops    = deim_ops_dict[k],
                    submesh     = submesh_deims.get(k),
                )
            # TODO (next): branch jump. Everything needed is in scope:
            #   N_u    = operators[k].Z_u.shape[1]
            #   nu_v   = nu_red[:N_u]                      # critical mode (reduced)
            #   a_sym  = solution.velocity_coeffs          # symmetric state
            #   a_kick = a_sym + eps * nu_v                # +/- for the two branches
            # then re-solve from a_kick (in this cluster, or switch to an
            # asymmetric cluster via apply_change_of_basis) and continue.
        # -------------------------------------------------------------------

        if verbose:
            status = "✓" if solution.converged else "✗"
            print(f"[{i+1}/{len(path)}] Re={Re:.2f} amp={amp:.2f} "
                  f"k={k} {status} iters={solution.iterations} mu={mu:.3e}")

        record = {
            'Re': Re, 'amp': amp, 'cluster': k,
            'converged': solution.converged,
            'iterations': solution.iterations,
            't_rom': t_rom,
            'cluster_switched': cluster_switched,
            'mu': mu,
        }
        if solution.converged:
            problem.reynolds.assign(Re)
            problem.amplitude.assign(amp)
            forces = compute_forces(solution_rom, problem)
            record['C_D'] = forces.drag
            record['C_L'] = forces.lift
        results.append(record)

        # --- history update ---
        prev_k = k
        prev_u_coeffs = solution.velocity_coeffs
        prev_p_coeffs = solution.pressure_coeffs
        prev_amp = amp
        prev_solution_rom = solution_rom
        prev_mu = mu

    return results


def continue_branch(
    path,
    clustering, operators, deim_ops_dict, submesh_deims,
    problem, initial_solution, M_u, M_p, M_uu_red,
    *,
    seed=None,                # dict(u_coeffs,p_coeffs,k,amp,solution_rom) or None
    can_bifurcate=True,
    branch_id="sym",
    eps=0.8, jump_offset=0, cl_floor=1e-3,
    n_eigenvalues=6, target=0.0, tau=0.8,
    use_change_of_basis=True, verbose=True,
):
    records, spawns = [], []
    last_converged = None
    converged_states = []
    n_lifting = operators[0].n_lifting
    prev_prev_k = None
    prev_prev_u_coeffs = None
    last_step_restarted = False
    if seed is None:
        prev_k = prev_u_coeffs = prev_p_coeffs = None
        prev_amp = prev_solution_rom = prev_mu = None
    else:
        prev_k            = seed['k']
        prev_u_coeffs     = seed['u_coeffs']
        prev_p_coeffs     = seed['p_coeffs']
        prev_amp          = seed['amp']
        prev_solution_rom = seed['solution_rom']
        prev_mu           = None          # children don't re-trigger

    for i, (Re, amp) in enumerate(path):
        nu = 1.0 / Re

        # --- cluster selection ---
        if prev_u_coeffs is None:
            k = select_the_cluster(clustering, Re, amp, initial_solution, M_u)
        else:
            # k = select_cluster_reduced_hysteresis(clustering, prev_u_coeffs, prev_k, tau=tau, verbose = True)

            k = select_cluster_reduced_projection(clustering, prev_u_coeffs, prev_k,a_old=prev_prev_u_coeffs,k_old= prev_prev_k, tau=tau, verbose=True)
        cluster_switched = (k != prev_k)

        # --- initial guess ---
        if prev_u_coeffs is None:
            src_vel  = initial_solution.velocity.dat.data_ro.flatten()
            src_pres = initial_solution.pressure.dat.data_ro.flatten()
            u_coeffs, p_coeffs = project_to_rom_coefficients(
                src_vel, src_pres, operators[k], amp=amp, M_u=M_u, M_p=M_p)
        elif cluster_switched and use_change_of_basis and clustering.change_of_basis is not None:
            thetas_old = make_thetas(prev_amp, n_lifting)
            thetas_new = make_thetas(amp, n_lifting)
            u_coeffs, p_coeffs = apply_change_of_basis(
                clustering, prev_k, k, prev_u_coeffs, prev_p_coeffs, thetas_old, thetas_new)
        elif cluster_switched:
            src_vel  = prev_solution_rom.velocity.dat.data_ro.flatten()
            src_pres = prev_solution_rom.pressure.dat.data_ro.flatten()
            u_coeffs, p_coeffs = project_to_rom_coefficients(
                src_vel, src_pres, operators[k], amp=amp, M_u=M_u, M_p=M_p)
        else:
            u_coeffs, p_coeffs = prev_u_coeffs, prev_p_coeffs

        # --- solve ---
        t0 = time.perf_counter()
        solution, w_rom, callback_data = solve_rom(
            operators=operators[k], nu=nu, amp=amp, problem=problem,
            deim_ops=deim_ops_dict[k],
            u_initial_guess=u_coeffs, p_initial_guess=p_coeffs,
            submesh_deim=submesh_deims.get(k), return_internals=True)
        t_rom = time.perf_counter() - t0
        solution_rom = reconstruct_solution(solution, operators[k], problem, amp=amp)

        # --- indicator ---
        mu, rom_ind = float('nan'), None
        if solution.converged:
            rom_ind = compute_rom_indicator(
                w_rom, callback_data, operators[k], M_uu_red[k],
                n_eigenvalues=n_eigenvalues, target=target)
            if rom_ind is not None:
                mu = float(rom_ind['mu'])

        # --- branch jump (trunk only) ---
        if can_bifurcate and solution.converged:
            crossed = (prev_mu is not None and np.isfinite(prev_mu)
                       and np.isfinite(mu) and prev_mu > 0.0 >= mu)
            if crossed:
                if verbose:
                    print(f"  [bifurcation] {branch_id}: mu crossed at "
                          f"Re={Re:.3f} ({prev_mu:.2e} -> {mu:.2e})")
                for s in (+1, -1):
                    res = in_sweep_branch_jump_rom(
                        u_rom=solution.velocity_coeffs, p_rom=solution.pressure_coeffs,
                        Re=Re, amp=amp, operators_k=operators[k], M_uu_red_k=M_uu_red[k],
                        problem=problem, eps=eps, sign=s, indicator=rom_ind,
                        offset=jump_offset, deim_ops=deim_ops_dict[k],
                        submesh=submesh_deims.get(k))
                    if res['converged'] and abs(res['C_L']) < cl_floor:   # fell back to symmetric
                        print(f"DOUBLED EPS")
                        res = in_sweep_branch_jump_rom(
                            u_rom=solution.velocity_coeffs, p_rom=solution.pressure_coeffs,
                            Re=Re, amp=amp, operators_k=operators[k], M_uu_red_k=M_uu_red[k],
                            problem=problem, eps=2.0 * eps, sign=s, indicator=rom_ind,
                            offset=jump_offset, deim_ops=deim_ops_dict[k],
                            submesh=submesh_deims.get(k))
                    if res['converged']:
                        res['cluster'] = k
                        res['spawn_index'] = i
                        spawns.append(res)

        if verbose:
            status = "OK " if solution.converged else "DIV"
            print(f"[{branch_id}] Re={Re:.2f} amp={amp:.2f} k={k} {status} "
                  f"it={solution.iterations} mu={mu:.2e}")

        record = {'branch': branch_id, 'Re': Re, 'amp': amp, 'cluster': k,
                  'converged': solution.converged, 'iterations': solution.iterations,
                  't_rom': t_rom, 'mu': mu}
        if solution.converged:
            problem.reynolds.assign(Re)
            problem.amplitude.assign(amp)
            forces = compute_forces(solution_rom, problem)
            record['C_D'] = forces.drag
            record['C_L'] = forces.lift
            print(f"  [forces] C_D={forces.drag:.4f} C_L={forces.lift:.4f}")
            print('#' * 70)
            last_converged = {
                'u_coeffs': solution.velocity_coeffs,
                'p_coeffs': solution.pressure_coeffs,
                'k': k, 'amp': amp, 'Re': Re,
                'solution_rom': solution_rom,
            }
            converged_states.append(last_converged)
        records.append(record)

        # --- history update ---
        if prev_k and prev_u_coeffs is not None:
            prev_prev_u_coeffs = prev_u_coeffs
            prev_prev_k = prev_k
        prev_k, prev_u_coeffs, prev_p_coeffs = k, solution.velocity_coeffs, solution.pressure_coeffs
        prev_amp, prev_solution_rom, prev_mu = amp, solution_rom, mu

    return records, spawns, last_converged, converged_states


def build_offaxis_branch(
    anchor,
    clustering, operators, deim_ops_dict, submesh_deims,
    problem, initial_solution, M_u, M_p, M_uu_red,
    *,
    re_anchor, amp_target, re_min,
    n_amp_steps=10, re_step=1.0,
    branch_id="offaxis", tau=0.8, verbose=True,
):
    """
    Continue an amp=0 asymmetric branch into amp != 0.

    At amp=0 the asymmetric branches are reachable from the symmetric trunk by
    the eigenvector kick at the pitchfork. At amp != 0 the reflection symmetry is
    already broken, so those branches are disconnected from the trunk and the
    kick cannot reach them. Instead we walk an existing amp=0 asymmetric solution
    off-axis in two legs:

      Leg 1 (amp-walk):  hold Re = re_anchor, step amp  0 -> amp_target.
      Leg 2 (Re-down):   hold amp = amp_target, sweep Re  re_anchor -> re_min.

    The HIGH-Re anchor is used (not the onset edge) so the branch is strongly
    developed: the cluster selector then keeps the asymmetric cluster throughout
    the amp-walk. The Re-down leg terminates naturally where the asymmetric
    solution ceases to exist (continuation stops converging / folds back).

    No new solver: this is pure path construction + seeding on top of
    continue_branch, which is parameter-agnostic (it walks any (Re, amp) list and
    fetches nu=1/Re and the amp-dependent lifting fresh at each step).

    Parameters
    ----------
    anchor : dict(u_coeffs, p_coeffs, k, amp, solution_rom)
        Converged amp=0 asymmetric state at Re = re_anchor (the seed).
    re_anchor : float   Re of the anchor (top of both legs).
    amp_target : float  off-axis amplitude to reach (e.g. -0.3).
    re_min : float      lowest Re for the down-sweep (set == re_anchor to skip leg 2).
    n_amp_steps : int   number of amp increments in leg 1.
    re_step : float     Re decrement magnitude in leg 2.

    Returns
    -------
    records : list   per-step continuation records along the off-axis path.
    final   : dict    last converged state (or None).
    """
    amp0 = anchor['amp']

    # Leg 1: amp0 -> amp_target at fixed Re (drop the anchor point itself)
    amp_leg = np.linspace(amp0, amp_target, n_amp_steps + 1)[1:]
    path_amp = [(float(re_anchor), float(a)) for a in amp_leg]

    # Leg 2: Re re_anchor -> re_min at fixed amp_target
    re_leg = np.arange(re_anchor - re_step, re_min - 1e-9, -re_step)
    path_re = [(float(r), float(amp_target)) for r in re_leg]

    path = path_amp + path_re
    if not path:
        return [], anchor

    if verbose:
        print(f"\n  [offaxis] {branch_id}: anchor (Re={re_anchor:.1f}, amp={amp0:.2f}) "
              f"k={anchor['k']} -> amp-walk to {amp_target:+.2f} ({len(path_amp)} steps), "
              f"then Re down to {re_min:.1f} ({len(path_re)} steps)")

    records, _, final, _ = continue_branch(
        path, clustering, operators, deim_ops_dict, submesh_deims,
        problem, initial_solution, M_u, M_p, M_uu_red,
        seed=anchor, can_bifurcate=False, branch_id=branch_id,
        tau=tau, verbose=verbose)
    return records, final


def build_bifurcation_diagram(
    path,
    clustering, operators, deim_ops_dict,
    problem, initial_solution, M_u, M_p, M_uu_red,
    use_deim=True,
    *,
    eps=0.8, jump_offset=0, tau=0.8, proj_tol=0.01,
    offaxis_amps=(), re_min=None, n_amp_steps=10, re_step=1.0,
    submesh_deims=None, return_anchors=False,
    verbose=True,
):
    if submesh_deims is None:
        submesh_deims = {}
        if use_deim:
            for k in range(clustering.n_clusters):
                if deim_ops_dict.get(k) is not None:
                    submesh_deims[k] = SubMeshDEIM(problem.velocity_space, deim_ops_dict[k])

    all_records = []

    # --- trunk: symmetric branch, allowed to bifurcate ---
    trunk_records, spawns, _, _ = continue_branch(
        path, clustering, operators, deim_ops_dict, submesh_deims,
        problem, initial_solution, M_u, M_p, M_uu_red,
        seed=None, can_bifurcate=True, branch_id="sym",
        eps=eps, jump_offset=jump_offset, tau=tau, verbose=verbose)
    all_records += trunk_records

    anchors = []   # (branch_id, final converged amp=0 state) per asym branch, for off-axis seeding

    # --- children: each spawn continued forward, no further bifurcation ---
    for sp in spawns:
        re_c = path[sp['spawn_index']][0]
        child_path = [pt for pt in path if pt[0] > re_c]   # remaining Re, ascending
        if not child_path:
            continue

        k_src = sp['cluster']                  # trunk cluster the kick was computed in
        amp_c = sp['amp']
        a_kick = sp['sol'].velocity_coeffs     # kicked state, in k_src's frame
        p_kick = sp['sol'].pressure_coeffs

        # classify the newborn state onto the cluster whose basis best holds it.
        # the source frame is excluded (self-projection is tautologically 0), so the
        # ranking is over genuinely distinct clusters -- no sign->cluster table.

        # I will make k stay the same
        k_child = k_src
        # k_child = classify_state_for_seed(
        #     clustering, a_kick, k_src, proj_tol=proj_tol, verbose=verbose)

        if k_child != k_src:
            thetas = make_thetas(amp_c, n_lifting=operators[0].n_lifting)
            u_seed, p_seed = apply_change_of_basis(
                clustering, k_src, k_child, a_kick, p_kick, thetas, thetas)
        else:
            u_seed, p_seed = a_kick, p_kick

        seed = {
            'u_coeffs': u_seed,
            'p_coeffs': p_seed,
            'k':        k_child,
            'amp':      amp_c,
            # DOF-space field for reference is reconstructed in the frame it was solved in
            'solution_rom': reconstruct_solution(sp['sol'], operators[k_src], problem, amp=amp_c),
        }
        child_records, _, child_final, _ = continue_branch(
            child_path, clustering, operators, deim_ops_dict, submesh_deims,
            problem, initial_solution, M_u, M_p, M_uu_red,
            seed=seed, can_bifurcate=False, branch_id=f"asym{sp['sign']:+d}",
            eps=eps, jump_offset=jump_offset, tau=tau, verbose=verbose)
        all_records += child_records
        if child_final is not None:
            anchors.append((f"asym{sp['sign']:+d}", child_final))

    # --- off-axis branches: walk each amp=0 asymmetric anchor into amp != 0 ---
    for base_id, anchor in anchors:
        for amp_t in offaxis_amps:
            re_a = anchor['Re']
            off_records, _ = build_offaxis_branch(
                anchor, clustering, operators, deim_ops_dict, submesh_deims,
                problem, initial_solution, M_u, M_p, M_uu_red,
                re_anchor=re_a, amp_target=amp_t,
                re_min=(re_min if re_min is not None else re_a),
                n_amp_steps=n_amp_steps, re_step=re_step,
                branch_id=f"{base_id}@amp{amp_t:+.2f}",
                tau=tau, verbose=verbose)
            all_records += off_records

    if return_anchors:
        return all_records, anchors
    return all_records

def build_full_diagram(
    Re_range, amp_values,
    clustering, operators, deim_ops_dict,
    problem, initial_solution, M_u, M_p, M_uu_red,
    use_deim=True,
    *,
    eps=0.8, jump_offset=0, tau=0.8, proj_tol=0.01,
    verbose=True,
):
    """
    Full (Re, amp) bifurcation diagram, composed from the existing pieces.

    Phase 1 -- on-axis (amp = 0):
        symmetric trunk + eigenvector-kick pitchfork -> asym+ and asym- branches,
        traced up to Re_max. Their Re_max endpoints become the off-axis anchors.

    Phase 2 -- off-axis grid (amp != 0):
        For each asymmetric anchor and each amp SIGN present in amp_values:
          SPINE   : hold Re = Re_max and walk amp 0 -> extreme through every
                    requested amp value, capturing the converged state at each amp.
          Re-DOWN : from each spine state, sweep Re across the whole range at that
                    fixed amp. Each column terminates/merges at its own (amp-shifted)
                    critical Reynolds.
        Positive and negative amps are walked independently -- the rotation sign is
        physical, not a mirror -- so all four quarter-families are produced:
        {asym+, asym-} x {+amp, -amp}.

    No new solver: every step is continue_branch on a constructed (Re, amp) path,
    seeded so the cluster selector stays on the asymmetric basis until the merge.

    Parameters
    ----------
    Re_range   : ascending sequence of Re values (Re_range[0]=Re_min, [-1]=Re_max).
    amp_values : sequence of amp values to sample; 0 is handled by Phase 1.

    Returns
    -------
    all_records : list of step records, each tagged by record['branch']:
        "sym", "asym+1", "asym-1"          (phase 1, amp = 0)
        "asym{+/-}1@spine{+/-}"            (amp-walk at Re_max)
        "asym{+/-}1@amp{+.2f}"             (Re-sweep at fixed amp)
      Slice the surface by grouping on this tag.
    """
    Re_list = [float(r) for r in Re_range]
    Re_min, Re_max = Re_list[0], Re_list[-1]
    Re_down = Re_list[-2::-1]                       # Re_max-step ... Re_min (descending)

    neg = sorted([float(a) for a in amp_values if a < 0], reverse=True)  # -0.1,-0.2,...
    pos = sorted([float(a) for a in amp_values if a > 0])                #  0.1, 0.2,...

    # shared hyper-reduction submeshes (built once, reused by every phase)
    submesh_deims = {}
    if use_deim:
        for k in range(clustering.n_clusters):
            if deim_ops_dict.get(k) is not None:
                submesh_deims[k] = SubMeshDEIM(problem.velocity_space, deim_ops_dict[k])

    all_records = []

    # =========================== Phase 1: amp = 0 ===========================
    path0 = [(r, 0.0) for r in Re_list]
    base_records, anchors = build_bifurcation_diagram(
        path0, clustering, operators, deim_ops_dict,
        problem, initial_solution, M_u, M_p, M_uu_red, use_deim=use_deim,
        eps=eps, jump_offset=jump_offset, tau=tau, proj_tol=proj_tol,
        offaxis_amps=(), submesh_deims=submesh_deims, return_anchors=True,
        verbose=verbose)
    all_records += base_records

    if not anchors:
        if verbose:
            print("  [full] no asymmetric anchors at amp=0; returning on-axis diagram only")
        return all_records

    # =========================== Phase 2: off-axis grid =====================
    for base_id, anchor in anchors:                 # ("asym+1", state), ("asym-1", state)
        for side_name, amp_list in (("-", neg), ("+", pos)):
            if not amp_list:
                continue

            # ---- SPINE: amp-walk at Re_max through every requested amp ----
            spine_path = [(Re_max, a) for a in amp_list]
            if verbose:
                print(f"\n  [full] {base_id} spine{side_name}: Re={Re_max:.1f}, "
                      f"amp 0 -> {amp_list[-1]:+.2f} ({len(amp_list)} stops)")
            spine_records, _, _, spine_states = continue_branch(
                spine_path, clustering, operators, deim_ops_dict, submesh_deims,
                problem, initial_solution, M_u, M_p, M_uu_red,
                seed=anchor, can_bifurcate=False,
                branch_id=f"{base_id}@spine{side_name}", tau=tau, verbose=verbose)
            all_records += spine_records

            # ---- Re-DOWN sweep at each amp stop, seeded by its spine state ----
            for st in spine_states:
                a = st['amp']
                redown_path = [(r, a) for r in Re_down]
                if not redown_path:
                    continue
                if verbose:
                    print(f"\n  [full] {base_id} Re-sweep at amp={a:+.2f}: "
                          f"Re {Re_max:.1f} -> {Re_min:.1f}")
                sweep_records, _, _, _ = continue_branch(
                    redown_path, clustering, operators, deim_ops_dict, submesh_deims,
                    problem, initial_solution, M_u, M_p, M_uu_red,
                    seed=st, can_bifurcate=False,
                    branch_id=f"{base_id}@amp{a:+.2f}", tau=tau, verbose=verbose)
                all_records += sweep_records

    return all_records


def find_critical_curve_continuation(
    amp_values,
    Re_scan,
    clustering,
    operators,
    deim_ops_dict,
    problem,
    initial_solution,
    M_u,
    M_p,
    M_uu_red,
    use_deim=False,
    *,
    refine_step=0.15,
    turn_tol=1e-2,
    rebound_tol=5e-4,
    confirm_steps=2,
    back_margin=1.0,
    near_tol=5e-2,
    max_dead_run=4,
    cl_slack=2e-3,
    tau=0.8,
    submesh_deims=None,
    verbose=True,
):
    """
    Reduced critical curve Re_c(amp): warm-started continuation with
    windowing, built from three primitives.

    WALK  -- march Re upward on the coarse grid (starting a little below
             the previous amp's Re_c: windowing), tracking the finite-mu
             history. Stops on a sign change (bracket), a confirmed
             rebound of |mu| after a small minimum (rounded), or too many
             consecutive Newton failures.
    PIN   -- when Newton fails and the last two mu's project a zero
             within ~2 coarse steps ahead (fallback: |last mu| <
             near_tol), the crossing is probably between the last good
             Re and the failure: halve toward the frontier until either
             a sign change appears (-> bracket) or the interval is
             refine_step wide. If the walk then dies for good with the
             projected zero still just ahead, the crossing is reported
             as EXTRAPOLATED from the last two mu's.
    BISECT-- shrink a sign-change bracket to width <= refine_step with
             warm starts; dead midpoints are retried at 0.25/0.75, and if
             the bracket interior is dead the tightest bracket is
             linearly interpolated.

    kind is 'crossing', 'rounded' (connected-branch closest approach, NOT
    the disconnected-branch fold), 'extrapolated' (Newton died at the
    crossing even at fine steps; estimate from the last two mu's --
    verify against the FOM), or None.

    mu = nan with a CONVERGED solve (eigensolver missed the real mode)
    is stepped through silently; only non-convergence counts as failure.

    Assumes Re_c is non-decreasing in amp (else pass amp_values reversed;
    the started-past fallback keeps it correct either way, just slower).

    Returns a list of dicts sorted by amp:
        {'amp','Re_c','mu_c','found','kind','bracket','n_solves','sweep'}
    """

    if submesh_deims is None:
        submesh_deims = {}
    if use_deim:
        for k in range(clustering.n_clusters):
            if deim_ops_dict.get(k) is not None:
                submesh_deims[k] = SubMeshDEIM(
                    problem.velocity_space, deim_ops_dict[k])

    Re_grid = [float(r) for r in Re_scan]
    step0 = Re_grid[1] - Re_grid[0] if len(Re_grid) > 1 else 1.0
    n_solves = [0]

    def _mu(rec):
        m = rec.get("mu")
        return m if (m is not None and np.isfinite(m)) else None

    def _step(Re, amp, seed):
        n_solves[0] += 1
        recs, _, _, states = continue_branch(
            [(float(Re), float(amp))],
            clustering, operators, deim_ops_dict, submesh_deims,
            problem, initial_solution, M_u, M_p, M_uu_red,
            seed=seed, can_bifurcate=False,
            branch_id=f"crit@amp{amp:+.2f}", tau=tau, verbose=verbose)
        rec = recs[0] if recs else {"Re": float(Re), "amp": float(amp),
                                    "mu": None, "cluster": None,
                                    "converged": False}
        if verbose:
            m = _mu(rec)
            print(f" Re={Re:8.4f} mu=" +
                  (f"{m:+.4e}" if m is not None else " nan "))
        return rec, (states[-1] if states else None)

    def _cl_thr(*recs):
        """Anchor-relative C_L acceptance threshold: the largest |C_L|
        among the anchor records plus cl_slack. Self-adapting: near a
        symmetric crossing (anchors ~1e-4) it rejects sub-threshold
        drift that inflates mu and biases Re_c right; near a genuine
        rounding the anchors already carry the physical deflection, so
        the same rule leaves the refinement untouched."""
        if cl_slack is None:
            return None
        base = max((abs(r["C_L"]) for r in recs
                    if r.get("C_L") is not None), default=0.0)
        return base + cl_slack

    def _clean(rec, thr):
        """mu for FINE solves: rejects points whose |C_L| exceeds the
        anchor-relative threshold (Newton captured by / drifting onto
        the other sheet). The coarse walk keeps every converged point
        so rebound detection still sees the drift."""
        m = _mu(rec)
        if m is None:
            return None
        cl = rec.get("C_L")
        if thr is not None and cl is not None and abs(cl) > thr:
            return None
        return m

    def _fit_crossing(ctx, L, R, thr):
        """Weighted-fit crossing estimate: straight-line LS fit of
        mu(Re) over ALL clean points near the bracket. Uses every
        accepted solve instead of two endpoints, damps the ~1e-3
        eigen jitter, and the fit residual yields a 1-sigma
        uncertainty on Re_c. Returns (Re_c, err) or None."""
        w = max(2.0 * refine_step, step0)
        sL = 1.0 if L["mu"] > 0 else -1.0
        pts = {}
        for r in ctx.sweep:
            if (_clean(r, thr) is not None
                    and L["Re"] - w <= r["Re"] <= R["Re"] + w):
                # sign-consistency with a single crossing in [L, R]:
                # L-side sign only left of R, opposite sign only
                # right of L (guards against curvature / double dips)
                if ((r["mu"] * sL > 0 and r["Re"] <= R["Re"])
                        or (r["mu"] * sL <= 0 and r["Re"] >= L["Re"])):
                    pts[round(r["Re"], 9)] = r
        pts = list(pts.values())
        if len(pts) < 3:
            return None
        x = np.array([r["Re"] for r in pts])
        y = np.array([r["mu"] for r in pts])
        xm = x - x.mean()
        a, b = np.polyfit(xm, y, 1)
        if a >= 0.0:
            return None
        Re_c = x.mean() - b / a
        # model-mismatch guard: a linear fit that lands far from the
        # bracket interpolation means curvature dominates -> distrust
        if abs(Re_c - _interp(L, R)) > refine_step:
            return None
        if not (L["Re"] - w <= Re_c <= R["Re"] + w):
            return None
        dof = len(pts) - 2
        s2 = float(((y - (a * xm + b)) ** 2).sum()) / dof if dof else 0.0
        Sxx = float((xm ** 2).sum())
        err = abs(1.0 / a) * np.sqrt(
            s2 / len(pts) + (b ** 2) * s2 / (a ** 2 * Sxx))
        return float(Re_c), float(err)

    def _interp(a, b):
        if a["mu"] == b["mu"]:
            return 0.5 * (a["Re"] + b["Re"])
        return a["Re"] - a["mu"] * (b["Re"] - a["Re"]) / (b["mu"] - a["mu"])

    def _nearest(states, Re):
        return (min(states, key=lambda s: abs(s["Re"] - Re))
                if states else None)

    class Ctx:
        """Mutable per-amp sweep context shared by walk/pin/bisect."""

        def __init__(self, seed):
            self.seed, self.states, self.sweep = seed, [], []
            self.vals = []            # finite-mu records, in order
            self.imin = None          # index of min |mu| in vals

        def solve(self, Re, amp, near=None):
            seed = (_nearest(self.states, near) if near is not None
                    else None) or self.seed
            rec, st = _step(Re, amp, seed)
            self.sweep.append(rec)
            if st is not None:
                self.seed = st
                self.states.append(st)
            return rec

        def push(self, rec):
            """Record a finite-mu point; return 'bracket'/'rounded'/None."""
            v = self.vals
            if v and v[-1]["mu"] * rec["mu"] <= 0.0:
                v.append(rec)
                return "bracket"
            v.append(rec)
            i = len(v) - 1
            if self.imin is None or abs(rec["mu"]) < abs(v[self.imin]["mu"]):
                self.imin = i
            elif (abs(v[self.imin]["mu"]) < turn_tol
                  and i - self.imin >= confirm_steps
                  and abs(rec["mu"]) > abs(v[self.imin]["mu"]) + rebound_tol):
                return "rounded"
            return None

    def _pin(ctx, amp, Re_bad):
        """Newton died at Re_bad with small |mu| behind: halve toward the
        frontier. Returns 'bracket' if a sign change appears, else None."""
        while Re_bad - ctx.vals[-1]["Re"] > refine_step:
            Re_m = 0.5 * (ctx.vals[-1]["Re"] + Re_bad)
            rec = ctx.solve(Re_m, amp, near=Re_m)
            if _mu(rec) is None:
                Re_bad = Re_m
            elif ctx.push(rec) == "bracket":
                return "bracket"
        return None

    def _bisect(ctx, amp, L, R):
        """Shrink bracket [L,R] to width <= refine_step, accepting only
        points clean relative to the bracket anchors; final Re_c from
        the weighted fit over all clean nearby points (fallback: linear
        interp of the bracket). Returns (Re_c, mu_c, bracket, err)."""
        thr = _cl_thr(L, R)
        for _ in range(12):
            if R["Re"] - L["Re"] <= refine_step:
                break
            for frac in (0.5, 0.25, 0.75):
                Re_m = L["Re"] + frac * (R["Re"] - L["Re"])
                rec = ctx.solve(Re_m, amp, near=L["Re"])
                if _clean(rec, thr) is not None:
                    if rec["mu"] * L["mu"] > 0.0:
                        L = rec
                    else:
                        R = rec
                    break
            else:
                break  # bracket interior dead/off-branch: stop
        mu_c = L["mu"] if abs(L["mu"]) <= abs(R["mu"]) else R["mu"]
        fit = _fit_crossing(ctx, L, R, thr)
        if fit is not None:
            Re_c, err = fit
        else:
            Re_c, err = _interp(L, R), None
        return Re_c, mu_c, (float(L["Re"]), float(R["Re"])), err

    def _proj(ctx):
        """Zero of the line through the last two mu's, if mu is heading
        down toward it; else None."""
        v = ctx.vals
        if (len(v) >= 2 and v[-1]["mu"] < v[-2]["mu"]):
            z = _interp(v[-2], v[-1])
            if z > v[-1]["Re"]:
                return z
        return None

    def _walk(amp, Re_start, seed0):
        """Coarse march; returns (ctx, status) with status in
        {'bracket','rounded',None}."""
        ctx = Ctx(seed0)
        dead = 0
        Re = Re_start
        while Re <= Re_grid[-1] + 1e-9:
            rec = ctx.solve(Re, amp)
            if _mu(rec) is None:
                if not rec.get("converged", False):
                    z = _proj(ctx)
                    suspicious = dead == 0 and ctx.vals and (
                        (z is not None and z <= Re + 2.0 * step0)
                        or abs(ctx.vals[-1]["mu"]) < near_tol)
                    if suspicious and _pin(ctx, amp, Re) == "bracket":
                        return ctx, "bracket"
                    dead += 1
                    if dead > max_dead_run:
                        break
            else:
                dead = 0
                status = ctx.push(rec)
                if status is not None:
                    return ctx, status
            Re += step0
        return ctx, None

    curve = []
    prev_Re_c, prev_states = None, []

    for amp in sorted(amp_values):
        n_solves[0] = 0
        if verbose:
            print("\n" + "=" * 80)
            print(f"CRITICAL CURVE amp = {amp:+.4f}")
            print("=" * 80)

        Re_start = (Re_grid[0] if prev_Re_c is None
                    else max(Re_grid[0], prev_Re_c - back_margin))
        ctx, status = _walk(amp, Re_start, _nearest(prev_states, Re_start))

        # Windowed start landed past the crossing -> full sweep.
        if status is None and ctx.vals and ctx.vals[0]["mu"] <= 0.0:
            if verbose:
                print(" [window started past crossing -> full sweep]")
            ctx, status = _walk(amp, Re_grid[0],
                                min(prev_states, key=lambda s: s["Re"])
                                if prev_states else None)

        Re_c = mu_c = kind = bracket = err = None

        if status == "bracket":
            Re_c, mu_c, bracket, err = _bisect(ctx, amp,
                                               ctx.vals[-2], ctx.vals[-1])
            kind = "crossing"
        elif status == "rounded":
            # --------------------------------------------------------
            # Refine the minimum before believing it: mu decreasing
            # then increasing on the coarse grid can hide a narrow
            # double crossing between grid points. Probe the wider
            # half around the running minimum; a sign flip -> hidden
            # crossing (bisected); otherwise -> genuine rounded, with
            # Re_c from the REFINED minimum. Off-branch fine points
            # (anchor-relative C_L check) stop the refinement, never corrupt it.
            # --------------------------------------------------------
            i = ctx.imin
            pts = sorted(
                {r["Re"]: r for r in ctx.vals[max(i - 1, 0):i + 2]}.values(),
                key=lambda r: r["Re"])
            thr = _cl_thr(*pts)

            for _ in range(8):
                b = min(pts, key=lambda r: abs(r["mu"]))
                ib = pts.index(b)
                lo = pts[max(ib - 1, 0)]
                hi = pts[min(ib + 1, len(pts) - 1)]
                if hi["Re"] - lo["Re"] <= refine_step:
                    break
                # probe the wider flank of the current minimum
                Re_m = (0.5 * (lo["Re"] + b["Re"])
                        if b["Re"] - lo["Re"] >= hi["Re"] - b["Re"]
                        else 0.5 * (b["Re"] + hi["Re"]))
                rec = ctx.solve(Re_m, amp, near=Re_m)
                if _clean(rec, thr) is None:
                    break  # dead or off-branch at fine scale: stop here
                if rec["mu"] * b["mu"] <= 0.0:
                    # hidden crossing: bracket against the nearest
                    # refined point (which carries the opposite sign)
                    other = min(pts, key=lambda r: abs(r["Re"] - Re_m))
                    Lb, Rb = sorted((rec, other), key=lambda r: r["Re"])
                    Re_c, mu_c, bracket, err = _bisect(ctx, amp, Lb, Rb)
                    kind = "crossing"
                    break
                pts = sorted(pts + [rec], key=lambda r: r["Re"])

            if kind is None:
                # genuine rounded: parabolic vertex of the refined min
                b = min(pts, key=lambda r: abs(r["mu"]))
                ib = pts.index(b)
                Re_c, mu_c = float(b["Re"]), float(b["mu"])
                if 0 < ib < len(pts) - 1:
                    (x0, a), (x1, m), (x2, c) = [
                        (p["Re"], p["mu"])
                        for p in pts[ib - 1:ib + 2]]
                    # exact quadratic vertex on a NON-uniform grid
                    # (refined points are not equally spaced)
                    d1 = (m - a) / (x1 - x0)
                    d2 = (c - m) / (x2 - x1)
                    f2 = (d2 - d1) / (x2 - x0)
                    if f2 != 0:
                        Re_c = 0.5 * (x0 + x1) - d1 / (2.0 * f2)
                kind = "rounded"

        else:
            z = _proj(ctx)
            if (status is None and z is not None
                    and z <= Re_grid[-1] + 1e-9
                    and z - ctx.vals[-1]["Re"] <= 2.0 * step0):
                # Newton kept dying right where mu was heading to zero:
                # report the projected crossing honestly. Verify vs FOM.
                Re_c, mu_c = z, ctx.vals[-1]["mu"]
                kind = "extrapolated"

        if verbose:
            if Re_c is None:
                print(f" -> no critical point for amp={amp:+.4f}")
            else:
                es = f" +/- {err:.3f}" if err is not None else ""
                print(f" -> Re_c({amp:+.4f}) = {Re_c:.4f}{es} "
                      f"[{kind}, {n_solves[0]} solves]")

        curve.append({"amp": float(amp),
                      "Re_c": None if Re_c is None else float(Re_c),
                      "mu_c": None if mu_c is None else float(mu_c),
                      "found": Re_c is not None,
                      "kind": kind,
                      "Re_c_err": None if err is None else float(err),
                      "bracket": bracket,
                      "n_solves": int(n_solves[0]),
                      "sweep": ctx.sweep})

        if Re_c is not None:
            prev_Re_c = Re_c
        if ctx.states:
            prev_states = ctx.states

    return sorted(curve, key=lambda c: c["amp"])
