"""
Bare-metric bifurcation diagram.

Cluster selection = nearest whitened global-POD centroid, full stop.
No hysteresis (tau), no projection-loss veto, no seed classification.
The reverse-engineering study (737/740 agreement between whitened-metric
argmin and best-representer, misses only at near-zero-C_L ties) is what
licenses this: the metric is correct on its own, so we stop damping it.

Drop-in replacement for build_full_diagram.

Usage:

    from build_diagram_bare import build_full_diagram_bare
    results = build_full_diagram_bare(
        Re_range=np.arange(50.0, 100.0, 1.0),
        amp_values=[-0.3, -0.2, -0.1, 0.1, 0.2, 0.3],
        clustering=clustering, operators=operators, deim_ops_dict=deim_ops_dict,
        problem=problem, initial_solution=initial_solution,
        M_u=M_u, M_p=M_p, M_uu_red=M_uu_red,
        use_deim=cfg.use_deim,
        fom_compute=True,          # <-- also solve the FOM at each point, record error + C_L
        trace_symmetric=True,      # <-- also carry the symmetric branch off-axis
        sym_start='low',           # spine at Re_min, then sweep Re up at each amp
    )

trace_symmetric=True adds a third phase: the symmetric trunk is continued in
amp from 0 to each extreme and then swept in Re at every amp, over the FULL
Re_range, with no collapse test and no early break -- so the symmetric sheet
covers the whole (Re, amp) grid, converged or not. Its records carry
branch='sym@spine{+,-}' and branch='sym@amp{+/-x.xx}'.

lock_children=True traces each asym branch in a single cluster (kink-free):
the symmetric trunk still selects freely, but each asymmetric child and every
off-axis leg is pinned to the cluster its seed lands in.

FOM comparison (fom_compute=True): at every converged ROM point a steady FOM
solve is run and the record gains C_L_rom, C_L_fom, err_u (relative velocity
error in the M_u norm), fom_converged, fom_iters. No fields are stored. To
save time, each continuation leg seeds its FIRST FOM point from the ROM
reconstruction (which fixes the branch) and every subsequent point from the
previous FOM solution (plain continuation) -- which also keeps the FOM on the
true branch through the near-merge rows where the ROM starts to collapse.
"""

import numpy as np
import time

from firedrake import Function

from nsrom.navier_stokes import compute_forces, solve_steady_navier_stokes, save_solution
from nsrom.rom import solve_rom, reconstruct_solution, project_to_rom_coefficients, SubMeshDEIM
from nsrom.local_rom import (
    select_the_cluster,
    select_cluster,
    apply_change_of_basis,
    _whitened_pod_distances,
)
from nsrom.bifurcation.branch_jump import compute_rom_indicator, in_sweep_branch_jump_rom

# make_thetas lives in sweep.py; import it from there so we reuse the same one
from cluster_diagnostics import test_fj_consistency
from sweep import make_thetas


# ---------------------------------------------------------------------------
# bare selector: nearest whitened centroid, nothing else
# ---------------------------------------------------------------------------
def select_nearest_cluster(clustering, a_prev, k_prev, verbose=False):
    """argmin over clusters of the whitened global-POD distance. No tau, no veto."""
    d = _whitened_pod_distances(clustering, a_prev, k_prev)   # (K,)
    k = int(np.argmin(d))
    if verbose:
        marks = " ".join(f"k{j}={d[j]:.4f}" + ("*" if j == k else "")
                         for j in range(len(d)))
        print(f"    [select] {marks}")
    return k


# ---------------------------------------------------------------------------
# FOM helpers
# ---------------------------------------------------------------------------
def _assemble_mixed(problem, velocity, pressure):
    """Pack separate velocity/pressure Functions into a mixed-space Function
    suitable as an initial guess for solve_steady_navier_stokes."""
    w = Function(problem.mixed_space)
    u, p = w.subfunctions
    u.assign(velocity)
    p.assign(pressure)
    return w


def _norm_M(vec, M):
    """sqrt(vec^T M vec) -- same convention as compare_rom_vs_fom."""
    return float(np.sqrt(vec @ (M @ vec)))


def _fom_compare(problem, solution_rom, guess, M_u):
    """Run one steady FOM solve from `guess`, compare to the ROM solution.
    Returns (fom_solution, C_L_fom, err_u, t_fom) with err_u the relative
    velocity error in the M_u inner product and t_fom the wall time of the FOM
    solve alone (same convention as t_rom -- solve only, nothing else).
    On non-convergence returns (fom, nan, nan, t_fom)."""
    t0 = time.perf_counter()
    fom = solve_steady_navier_stokes(problem, initial_guess=guess)
    t_fom = time.perf_counter() - t0
    if not fom.converged:
        return fom, float('nan'), float('nan'), t_fom
    forces = compute_forces(fom, problem)
    u_rom = solution_rom.velocity.dat.data_ro.flatten()
    u_fom = fom.velocity.dat.data_ro.flatten()
    d = u_fom - u_rom
    denom = _norm_M(u_fom, M_u)
    err_u = _norm_M(d, M_u) / denom if denom > 0 else float('nan')
    return fom, float(forces.lift), err_u, t_fom


# ---------------------------------------------------------------------------
# continuation along a (Re, amp) path -- bare selection
# ---------------------------------------------------------------------------
def _continue(
    path,
    clustering, operators, deim_ops_dict, submesh_deims,
    problem, initial_solution, M_u, M_p, M_uu_red,
    *,
    seed=None, can_bifurcate=True, branch_id="sym",
    eps=0.8, jump_offset=0, cl_floor=1e-3,
    n_eigenvalues=6, target=0.0,
    lock_cluster=None,           # int -> pin this cluster; None -> select each step
    terminate_on_collapse=False, # asym legs: stop at the merge into symmetric
    collapse_frac=0.0001,           # break if |C_L| < collapse_frac * running peak
    fom_compute=False,           # also solve the FOM at each point (error + C_L)
    keep_last_converged_guess=False,  # don't seed from a diverged point
    debug_snapshot=None,         # (Re, amp) -> save the reconstructed field there
    use_change_of_basis=True, verbose=True,
):
    records, spawns = [], []
    last_converged = None
    converged_states = []
    n_lifting = operators[0].n_lifting
    cl_peak = 0.0                # running max |C_L| seen on this branch
    prev_fom = None              # previous FOM solution on THIS leg (continuation)

    if seed is None:
        prev_k = prev_u_coeffs = prev_p_coeffs = None
        prev_amp = prev_solution_rom = prev_mu = None
    else:
        prev_k            = seed['k']
        prev_u_coeffs     = seed['u_coeffs']
        prev_p_coeffs     = seed['p_coeffs']
        prev_amp          = seed['amp']
        prev_solution_rom = seed['solution_rom']
        prev_mu           = None

    for (Re, amp) in path:
        nu = 1.0 / Re

        # --- cluster selection: bare nearest-centroid (or pinned) ---
        if lock_cluster is not None:
            k = lock_cluster
        elif prev_u_coeffs is None:
            # k = select_the_cluster(clustering, Re, amp, initial_solution, M_u,verbose=True)
            k= select_cluster(clustering, Re, amp)
        else:
            k = select_nearest_cluster(clustering, prev_u_coeffs, prev_k, verbose=verbose)
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

        # --- ROM solve ---
        t0 = time.perf_counter()
        solution, w_rom, callback_data = solve_rom(
            operators=operators[k], nu=nu, amp=amp, problem=problem,
            deim_ops=deim_ops_dict[k],
            u_initial_guess=u_coeffs, p_initial_guess=p_coeffs,
            submesh_deim=submesh_deims.get(k), return_internals=True,)
        t_rom = time.perf_counter() - t0
        solution_rom = reconstruct_solution(solution, operators[k], problem, amp=amp)

        # --- optional debug snapshot (OFF by default) ----------------------
        # This used to be unconditional and it *aborted* every amp=0 leg at
        # Re=99 before the point was ever recorded. Keep it opt-in.
        if debug_snapshot is not None:
            Re_snap, amp_snap = debug_snapshot
            if abs(amp - amp_snap) <= 1e-6 and abs(Re - Re_snap) <= 1e-6:
                save_solution(
                    solution_rom,
                    f"solution_rom_Re{Re_snap:g}_amp{amp_snap:g}_{branch_id}",
                    problem.mesh)




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
                    if res['converged'] and abs(res['C_L']) < cl_floor:
                        res = in_sweep_branch_jump_rom(
                            u_rom=solution.velocity_coeffs, p_rom=solution.pressure_coeffs,
                            Re=Re, amp=amp, operators_k=operators[k], M_uu_red_k=M_uu_red[k],
                            problem=problem, eps=2.0 * eps, sign=s, indicator=rom_ind,
                            offset=jump_offset, deim_ops=deim_ops_dict[k],
                            submesh=submesh_deims.get(k))
                    if res['converged']:
                        res['cluster'] = k
                        res['spawn_re'] = Re
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
            record['C_L_rom'] = float(forces.lift)

            # --- optional FOM comparison at this exact (Re, amp) on this branch ---
            if fom_compute:
                if prev_fom is None:
                    # first point on this leg: seed FOM from the ROM field,
                    # which fixes the branch the FOM converges onto.
                    guess = _assemble_mixed(
                        problem, solution_rom.velocity, solution_rom.pressure)
                else:
                    # continuation: seed from the previous FOM solution (cheaper,
                    # and stays on the true branch near the merge)
                    guess = prev_fom.solution
                fom, cl_fom, err_u, t_fom = _fom_compare(problem, solution_rom, guess, M_u)
                record['C_L_fom']       = cl_fom
                record['err_u']         = err_u
                record['fom_converged'] = bool(fom.converged)
                record['fom_iters']     = int(fom.num_iterations)
                record['t_fom']         = t_fom
                # per-point speed-up (solve-only, same convention on both sides)
                record['speedup'] = (t_fom / t_rom) if t_rom > 0 else float('nan')
                prev_fom = fom if fom.converged else None   # don't propagate a bad guess
                if verbose:
                    print(f"  [fom] C_L_rom={forces.lift:+.4f} C_L_fom={cl_fom:+.4f} "
                          f"err_u={err_u:.3e} fom_conv={fom.converged}")
                    print(f"  [time] t_rom={t_rom*1e3:.1f}ms t_fom={t_fom*1e3:.1f}ms "
                          f"speedup={record['speedup']:.1f}x")

            last_converged = {
                'u_coeffs': solution.velocity_coeffs,
                'p_coeffs': solution.pressure_coeffs,
                'k': k, 'amp': amp, 'Re': Re,
                'solution_rom': solution_rom,
            }
            converged_states.append(last_converged)
            if verbose:
                print(f"  [forces] C_L={forces.lift:.4f}")
        records.append(record)

        # --- C_L-collapse termination (asymmetric legs only) --------------
        # Below the amp-shifted bifurcation point Re_c(amp) the asymmetric
        # branch ceases to exist and merges into the symmetric one. Detect
        # that merge and stop: the symmetric trunk already covers Re < Re_c.
        if terminate_on_collapse:
            if solution.converged:
                cl_abs = abs(record['C_L'])
                cl_peak = max(cl_peak, cl_abs)
                if cl_peak > cl_floor and (
                        cl_abs < cl_floor or cl_abs < collapse_frac * cl_peak):
                    if verbose:
                        print(f"  [merge] {branch_id}: C_L collapsed at "
                              f"Re={Re:.2f} amp={amp:.2f} "
                              f"(|C_L|={cl_abs:.4f} < {collapse_frac:.2f}*"
                              f"{cl_peak:.4f}); Re_c(amp) reached, stopping leg")
                    records.pop()          # drop the merged (symmetric) point
                    break
            else:
                if cl_peak > cl_floor:
                    if verbose:
                        print(f"  [merge] {branch_id}: diverged at the merge "
                              f"Re={Re:.2f} amp={amp:.2f}; Re_c(amp) reached, "
                              f"stopping leg")
                    records.pop()          # drop the non-converged point
                    continue

        # --- history update ---
        # On a leg with no stop condition a single divergence would otherwise
        # poison every later point: keep the last *converged* state as the guess.
        if solution.converged or not keep_last_converged_guess:
            prev_k, prev_u_coeffs, prev_p_coeffs = (
                k, solution.velocity_coeffs, solution.pressure_coeffs)
            prev_amp, prev_solution_rom, prev_mu = amp, solution_rom, mu

    return records, spawns, last_converged, converged_states


# ---------------------------------------------------------------------------
# amp=0 three branches (trunk + two children)
# ---------------------------------------------------------------------------
def _three_branches(
    path, clustering, operators, deim_ops_dict, submesh_deims,
    problem, initial_solution, M_u, M_p, M_uu_red,
    *, eps=0.8, jump_offset=0, lock_children=False, fom_compute=False,
    debug_snapshot=None, verbose=True,
):
    all_records = []

    trunk_records, spawns, _, trunk_states = _continue(
        path, clustering, operators, deim_ops_dict, submesh_deims,
        problem, initial_solution, M_u, M_p, M_uu_red,
        seed=None, can_bifurcate=True, branch_id="sym",
        eps=eps, jump_offset=jump_offset, fom_compute=fom_compute,
        debug_snapshot=debug_snapshot, verbose=verbose)
    all_records += trunk_records

    anchors = []
    for sp in spawns:
        re_c = sp['spawn_re']
        child_path = [pt for pt in path if pt[0] > re_c]
        if not child_path:
            continue

        k_src = sp['cluster']
        amp_c = sp['amp']
        seed = {
            'u_coeffs': sp['sol'].velocity_coeffs,
            'p_coeffs': sp['sol'].pressure_coeffs,
            'k':        k_src,
            'amp':      amp_c,
            'solution_rom': reconstruct_solution(sp['sol'], operators[k_src], problem, amp=amp_c),
        }

        lock_k = None
        if lock_children:
            lock_k = select_nearest_cluster(clustering, seed['u_coeffs'], k_src)

        child_records, _, child_final, _ = _continue(
            child_path, clustering, operators, deim_ops_dict, submesh_deims,
            problem, initial_solution, M_u, M_p, M_uu_red,
            seed=seed, can_bifurcate=False, branch_id=f"asym{sp['sign']:+d}",
            eps=eps, jump_offset=jump_offset, lock_cluster=lock_k,
            fom_compute=fom_compute, verbose=verbose)
        all_records += child_records
        if child_final is not None:
            anchors.append((f"asym{sp['sign']:+d}", child_final))

    return all_records, anchors, trunk_states


# ---------------------------------------------------------------------------
# off-axis grid: amp-spine at one Re, then a Re sweep at every amp stop
# ---------------------------------------------------------------------------
def _pick_state(states, Re_target):
    """Converged state on a leg whose Re is closest to Re_target (None if empty)."""
    if not states:
        return None
    return min(states, key=lambda s: abs(s['Re'] - Re_target))


def _offaxis_grid(
    anchor, base_id, spine_Re, sweep_Re_list, neg, pos,
    clustering, operators, deim_ops_dict, submesh_deims,
    problem, initial_solution, M_u, M_p, M_uu_red,
    *, terminate_on_collapse, eps=0.8, jump_offset=0,
    lock_cluster=None, keep_last_converged_guess=False,
    fom_compute=False, verbose=True,
):
    """Walk amp 0 -> extreme at Re = spine_Re, then continue in Re along
    sweep_Re_list at every amp stop reached by the spine.

    Shared by the asymmetric children (terminate_on_collapse=True: each leg
    stops at the amp-shifted merge Re_c(amp)) and by the symmetric trunk
    (terminate_on_collapse=False: every leg runs the full Re list).
    """
    out = []
    if anchor is None:
        if verbose:
            print(f"  [bare] {base_id}: no anchor state, skipping off-axis grid")
        return out

    for side_name, amp_list in (("-", neg), ("+", pos)):
        if not amp_list:
            continue

        spine_path = [(spine_Re, a) for a in amp_list]
        if verbose:
            print(f"\n  [bare] {base_id} spine{side_name}: Re={spine_Re:.1f} "
                  f"amp 0 -> {amp_list[-1]:+.2f}")
        spine_records, _, _, spine_states = _continue(
            spine_path, clustering, operators, deim_ops_dict, submesh_deims,
            problem, initial_solution, M_u, M_p, M_uu_red,
            seed=anchor, can_bifurcate=False,
            branch_id=f"{base_id}@spine{side_name}",
            eps=eps, jump_offset=jump_offset, lock_cluster=lock_cluster,
            keep_last_converged_guess=keep_last_converged_guess,
            fom_compute=fom_compute, verbose=verbose)
        out += spine_records

        for st in spine_states:
            a = st['amp']
            leg_path = [(r, a) for r in sweep_Re_list]
            if not leg_path:
                continue
            if verbose:
                print(f"\n  [bare] {base_id} Re-sweep at amp={a:+.2f}: "
                      f"{sweep_Re_list[0]:.1f} -> {sweep_Re_list[-1]:.1f}")
            leg_records, _, _, _ = _continue(
                leg_path, clustering, operators, deim_ops_dict, submesh_deims,
                problem, initial_solution, M_u, M_p, M_uu_red,
                seed=st, can_bifurcate=False,
                branch_id=f"{base_id}@amp{a:+.2f}",
                eps=eps, jump_offset=jump_offset, lock_cluster=lock_cluster,
                terminate_on_collapse=terminate_on_collapse,
                keep_last_converged_guess=keep_last_converged_guess,
                fom_compute=fom_compute, verbose=verbose)
            out += leg_records

    return out


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------
def build_full_diagram_bare(
    Re_range, amp_values,
    clustering, operators, deim_ops_dict,
    problem, initial_solution, M_u, M_p, M_uu_red,
    use_deim=True,
    *,
    eps=0.8, jump_offset=0,
    lock_children=False,        # True -> pin each asym branch to one cluster (kink-free)
    fom_compute=False,          # True -> also solve FOM at each point, record error + C_L
    trace_symmetric=True,       # True -> also carry the symmetric trunk off-axis
    sym_start='high',            # 'low': spine at Re_min then sweep Re up (robust)
                                # 'high': spine at Re_max then sweep Re down
    sym_lock_cluster=None,      # int -> pin every symmetric off-axis leg to that cluster
    debug_snapshot=None,        # (Re, amp) -> dump the reconstructed field there
    verbose=True,
):
    """
    Phase 1: amp=0 -> sym + asym+1 + asym-1.
    Phase 2: for each asym anchor, walk amp 0->extreme at Re_max (spine),
             then sweep Re down at each amp stop; the Re-down legs terminate at
             the amp-shifted merge into symmetric (C_L collapse).
    Phase 3 (trace_symmetric=True): same off-axis grid, but seeded from the
             symmetric trunk and with NO termination condition -- every amp in
             amp_values gets the complete Re_range, converged or not.

    sym_start controls where the symmetric amp-spine sits:
      'low'  -- spine at Re_min, then each amp column sweeps Re upward. The
                low-Re solution is unique and stable, so the amp continuation
                and the subsequent upward sweep are both well-posed. Default.
      'high' -- spine at Re_max, then each column sweeps Re downward, mirroring
                the asymmetric phase. Riskier: above Re_c the symmetric state is
                unstable and amp!=0 unfolds the pitchfork, so the continuation
                can slide onto an asymmetric branch.

    Selection everywhere is bare nearest whitened centroid. No tau, no veto,
    no proj_tol, no seed classification.

    If fom_compute=True, every converged ROM point also gets a FOM solve and
    the record gains C_L_rom, C_L_fom, err_u, fom_converged, fom_iters.
    """
    Re_list = [float(r) for r in Re_range]
    Re_min, Re_max = Re_list[0], Re_list[-1]
    Re_down = Re_list[-2::-1]      # Re_max excluded: the spine already sits there
    Re_up   = Re_list[1:]          # Re_min excluded: idem for the low spine

    neg = sorted([float(a) for a in amp_values if a < 0], reverse=True)
    pos = sorted([float(a) for a in amp_values if a > 0])

    submesh_deims = {}
    if use_deim:
        for k in range(clustering.n_clusters):
            if deim_ops_dict.get(k) is not None:
                submesh_deims[k] = SubMeshDEIM(problem.velocity_space, deim_ops_dict[k])

    all_records = []

    # ---- Phase 1: amp = 0 ----
    path0 = [(r, 0.0) for r in Re_list]
    base_records, anchors, trunk_states = _three_branches(
        path0, clustering, operators, deim_ops_dict, submesh_deims,
        problem, initial_solution, M_u, M_p, M_uu_red,
        eps=eps, jump_offset=jump_offset, lock_children=lock_children,
        fom_compute=fom_compute, debug_snapshot=debug_snapshot, verbose=verbose)
    all_records += base_records

    # ---- Phase 2: off-axis grid, asymmetric children (terminates at Re_c) ----
    if anchors:
        for base_id, anchor in anchors:
            all_records += _offaxis_grid(
                anchor, base_id, Re_max, Re_down, neg, pos,
                clustering, operators, deim_ops_dict, submesh_deims,
                problem, initial_solution, M_u, M_p, M_uu_red,
                terminate_on_collapse=True,
                eps=eps, jump_offset=jump_offset,
                fom_compute=fom_compute, verbose=verbose)
    elif verbose:
        print("  [bare] no asymmetric anchors at amp=0; symmetric branch only")

    # ---- Phase 3: off-axis grid, symmetric trunk (no termination) ----
    if trace_symmetric and (neg or pos):
        if sym_start == 'low':
            spine_Re, sweep_list = Re_min, Re_up
        elif sym_start == 'high':
            spine_Re, sweep_list = Re_max, Re_down
        else:
            raise ValueError(f"sym_start must be 'low' or 'high', got {sym_start!r}")

        sym_anchor = _pick_state(trunk_states, spine_Re)
        if sym_anchor is None and verbose:
            print("  [bare] symmetric trunk has no converged state; "
                  "skipping symmetric off-axis grid")
        if verbose and sym_anchor is not None:
            print(f"\n  [bare] symmetric off-axis grid: spine at "
                  f"Re={sym_anchor['Re']:.1f}, sweep to "
                  f"Re={sweep_list[-1]:.1f}, no stop condition")
        all_records += _offaxis_grid(
            sym_anchor, "sym", spine_Re, sweep_list, neg, pos,
            clustering, operators, deim_ops_dict, submesh_deims,
            problem, initial_solution, M_u, M_p, M_uu_red,
            terminate_on_collapse=False,          # <-- full grid, always
            eps=eps, jump_offset=jump_offset,
            lock_cluster=sym_lock_cluster,
            keep_last_converged_guess=True,
            fom_compute=fom_compute, verbose=verbose)

    return all_records
