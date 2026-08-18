"""
Replay a stored point set with a different solver configuration.

Why
---
`build_full_diagram_bare` is a seeded continuation: spine at Re_max, then each
amplitude column marches in Re, then asymmetric branches spawn from converged
symmetric points. One failure high in that tree removes every point below it
without ever attempting them. That is why E4 (DEIM, tol 1e-8) stored 111 points
out of a ~4300-point grid: the spine diverged at Re=109 and eight whole
amplitude columns were never tried.

That answers "can DEIM be driven through parameter space?" (no). It does not
answer "where in the domain does DEIM actually solve?".

This module answers the second question. Every point is solved independently
from a stored initial guess, so no failure can suppress another point.

Seeding modes
-------------
  'stored'  reuse the exact reduced guess u_in the reference run handed to its
            solver at this point. Apples-to-apples: the replay solver sees
            precisely the same starting vector, so any difference in outcome is
            attributable to the operator alone. Requires the SAME reduced basis
            as the reference run.

  'fom'     project the stored FOM field into the cluster basis: the initial
            guess IS the exact solution, so there is no warm-start error at all.
            The most generous condition a ROM can be given, hence the stronger
            negative control. Only available where the reference actually
            retained a FOM field (check has_fom_field, not fom_converged).

Basis compatibility
-------------------
'stored' seeds are reduced coefficients in the reference run's cluster bases.
They are only meaningful if the replay loads the same cache (same K, same POD
tolerance, same clustering). E1 and E4 both resolve to
local_rom/K4_H1_tol1e-08/, so replaying E1's guesses under DEIM is valid.
Replaying them under K=1 is NOT -- use seed='fom' there, which reprojects from
the full field and is basis-agnostic.

Usage
-----
    from nsrom.io.state_store import StateWriter
    from nsrom.bifurcation.replay import replay_points, preflight

    preflight("states/E1_K4_tensor", operators)      # run this first

    writer = StateWriter("states/E13_replay_deim", mode='deim',
                         save_fom_fields=False,
                         meta={'replay_of': 'E1_K4_tensor', 'seed': 'stored'})
    records = replay_points(
        ref_state_dir="states/E1_K4_tensor",
        clustering=clustering, operators=operators,
        deim_ops_dict=deim_ops_dict, submesh_deims=submesh_deims,
        problem=problem, M_u=M_u, M_p=M_p, M_uu_red=M_uu_red,
        mode='deim', seed='stored',
        state_writer=writer)
    writer.close()
"""

import time
import numpy as np

from nsrom.rom import solve_rom, reconstruct_solution, project_to_rom_coefficients
from nsrom.navier_stokes import compute_forces
from nsrom.bifurcation.branch_jump import compute_rom_indicator
from nsrom.rom.local import select_cluster
from nsrom.io.state_store import StateStore


# ---------------------------------------------------------------------------
# preflight -- run this before committing hours to a replay
# ---------------------------------------------------------------------------
def preflight(ref_state_dir, operators=None):
    """Validate that a reference run can be replayed. Prints a report and
    returns True if seed='stored' is usable."""
    store = StateStore(ref_state_dir)
    meta = store.meta
    n = len(store)
    n_conv = sum(r['converged'] for r in store.records)
    n_field = sum(r.get('has_fom_field', False) for r in store.records)

    print(f"reference: {ref_state_dir}")
    print(f"  mode written:     {meta.get('mode', '?')}")
    print(f"  points:           {n}  ({n_conv} converged)")
    print(f"  FOM fields:       {n_field}")
    print(f"  fom_field_stride: {meta.get('fom_field_stride', '?')}")
    print(f"  fom_field_dtype:  {meta.get('fom_field_dtype', '?')}")

    ok = True
    if n == 0:
        print("  FAIL: no points.")
        return False

    step = max(1, n // 50)
    missing = sum(1 for i in range(0, n, step) if store[i].get('u_in') is None)
    if missing:
        print(f"  FAIL: u_in missing on {missing} sampled points.")
        ok = False
    else:
        print("  u_in present on all sampled points -- seed='stored' usable.")

    if n_field == 0:
        print("  NOTE: no FOM fields stored. seed='fom' unavailable and replay")
        print("        errors cannot be computed. Convergence map only.")
    elif n_field < n_conv:
        print(f"  NOTE: only {n_field}/{n_conv} converged points retained a FOM")
        print("        field, so err_u will be sparse. See the stride caveat.")

    if meta.get('fom_field_dtype') == 'float32':
        print("  NOTE: FOM fields are float32 (~1e-7 relative). Fine for the")
        print("        measured range (1e-5..1e-2); do not quote below 1e-6.")

    if operators is not None:
        seen = {}
        for pt in store:
            k = int(pt.get('cluster'))
            if k not in seen:
                seen[k] = len(pt.get('u_in'))
            if len(seen) >= len(operators):
                break
        for k, dim in sorted(seen.items()):
            expect = getattr(operators[k], 'n_velocity_modes', None)
            flag = ''
            if expect is not None and expect != dim:
                flag = f"  <-- MISMATCH, operators expect {expect}"
                ok = False
            print(f"  cluster {k}: stored u_in dim {dim}{flag}")
        if not ok:
            print("  FAIL: basis dimensions differ -- the replay is loading a")
            print("        different cache than the reference run used.")
            print("        Use seed='fom', or fix K / POD tolerance.")

    return ok


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------
def replay_points(
    ref_state_dir,
    clustering, operators, deim_ops_dict, submesh_deims,
    problem, M_u, M_p, M_uu_red,
    *,
    mode,
    seed='stored',                 # 'stored' | 'fom'
    cluster_source='stored',       # 'stored' | 'param'
    n_eigenvalues=6, target=0.0,
    point_filter=None,             # callable(StatePoint) -> bool
    state_writer=None,
    verbose=True,
):
    """Solve every reference point independently. No continuation, no spawns.

    Returns a list of record dicts using the same field names as the
    continuation driver, so downstream analysis needs no changes.

    cluster_source:
      'stored'  reuse the reference run's cluster assignment. Required when
                seed='stored', since the guess lives in that cluster's basis.
      'param'   re-run the cold-start selector select_cluster(Re, amp), which is
                what the online method does with no history. Honest, but only
                usable with seed='fom' because the reprojection makes the guess
                basis-independent.
    """
    if seed not in ('stored', 'fom'):
        raise ValueError(f"unknown seed mode {seed!r}")
    if seed == 'stored' and cluster_source != 'stored':
        raise ValueError(
            "seed='stored' requires cluster_source='stored': the stored guess "
            "is expressed in the reference run's cluster basis and is "
            "meaningless in another one. Use seed='fom' to re-select clusters.")

    store = StateStore(ref_state_dir)
    if verbose:
        print(f"[replay] {len(store)} reference points from {ref_state_dir}")

    records = []
    n_skipped = 0
    t_start = time.perf_counter()

    for pt in store:
        if point_filter is not None and not point_filter(pt):
            continue

        Re = float(pt.get('Re'))
        amp = float(pt.get('amp'))
        branch = pt.get('branch', 'replay')
        nu = 1.0 / Re

        # --- cluster ---
        if cluster_source == 'stored':
            k = int(pt.get('cluster'))
        else:
            k = int(select_cluster(clustering, Re, amp))

        # --- initial guess ---
        if seed == 'stored':
            u0, p0 = pt.get('u_in'), pt.get('p_in')
            if u0 is None or p0 is None:
                n_skipped += 1
                continue
            u0 = np.asarray(u0, dtype=float)
            p0 = np.asarray(p0, dtype=float)
        else:
            u_fom, p_fom = pt.get('u_fom'), pt.get('p_fom')
            if u_fom is None or p_fom is None:
                n_skipped += 1          # no field retained at this point
                continue
            u0, p0 = project_to_rom_coefficients(
                np.asarray(u_fom, dtype=float),
                np.asarray(p_fom, dtype=float),
                operators[k], amp=amp, M_u=M_u, M_p=M_p)

        # --- solve ---
        t0 = time.perf_counter()
        solution, w_rom, callback_data = solve_rom(
            operators=operators[k], nu=nu, amp=amp, problem=problem,
            mode=mode,
            deim_ops=deim_ops_dict[k] if mode == 'deim' else None,
            submesh_deim=submesh_deims.get(k),
            u_initial_guess=u0, p_initial_guess=p0,
            return_internals=True)
        t_rom = time.perf_counter() - t0

        record = {'branch': branch, 'Re': Re, 'amp': amp, 'cluster': k,
                  'converged': solution.converged,
                  'iterations': solution.iterations,
                  't_rom': t_rom, 'mu': float('nan'),
                  'seed': seed, 'replay_mode': mode}

        # carry reference-side scalars through so joins are trivial later
        for key in ('C_L_rom', 'C_L_fom', 'err_u', 't_rom', 't_fom',
                    'converged', 'iterations'):
            val = pt.get(key)
            if val is not None:
                record[f'ref_{key}'] = val

        if solution.converged:
            solution_rom = reconstruct_solution(
                solution, operators[k], problem, amp=amp)

            try:
                rom_ind = compute_rom_indicator(
                    w_rom, callback_data, operators[k], M_uu_red[k],
                    n_eigenvalues=n_eigenvalues, target=target)
                if rom_ind is not None:
                    record['mu'] = float(rom_ind['mu'])
            except Exception as e:
                if verbose:
                    print(f"  [indicator] failed at Re={Re:.2f} amp={amp:+.2f}: "
                          f"{type(e).__name__}: {e}")

            problem.reynolds.assign(Re)
            problem.amplitude.assign(amp)
            forces = compute_forces(solution_rom, problem)
            record['C_D'] = forces.drag
            record['C_L'] = forces.lift
            record['C_L_rom'] = float(forces.lift)

            # error against the reference run's stored FOM field -- free,
            # no new FOM solve, wherever the field was retained
            u_fom = pt.get('u_fom')
            if u_fom is not None:
                u_f = np.asarray(u_fom, dtype=float)
                u_r = solution_rom.velocity.dat.data_ro.flatten()
                d = u_f - u_r
                den = float(np.sqrt(u_f @ (M_u @ u_f)))
                record['err_u'] = (float(np.sqrt(d @ (M_u @ d))) / den
                                   if den > 0 else float('nan'))

        if verbose:
            status = "OK " if solution.converged else "DIV"
            print(f"[replay:{mode}] Re={Re:.2f} amp={amp:+.2f} k={k} {status} "
                  f"it={solution.iterations} mu={record['mu']:.2e}")

        if state_writer is not None:
            state_writer.write(
                Re=Re, amp=amp, branch=branch, cluster=k,
                converged=solution.converged,
                iterations=solution.iterations,
                u_in=u0, p_in=p0,
                u_out=solution.velocity_coeffs if solution.converged else None,
                p_out=solution.pressure_coeffs if solution.converged else None,
                t_rom=t_rom, mu=record['mu'],
                C_L_rom=record.get('C_L_rom'), C_D_rom=record.get('C_D'),
                err_u=record.get('err_u'))

        records.append(record)

    if verbose:
        n = len(records)
        n_ok = sum(r['converged'] for r in records)
        dt = time.perf_counter() - t_start
        print(f"\n[replay] mode={mode} seed={seed}  "
              f"{n_ok}/{n} converged ({100.0 * n_ok / max(n, 1):.1f}%), "
              f"{n_skipped} skipped, {dt:.0f}s")

    return records


def to_dataframe(records):
    import pandas as pd
    return pd.DataFrame(records)
