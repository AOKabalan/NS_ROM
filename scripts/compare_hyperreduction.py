"""
Test A -- controlled fixed-point comparison of hyper-reduction methods.

Question answered: given IDENTICAL inputs (same cluster, same initial guess,
same (Re, amp), same convergence tolerance), how accurate and how fast is each
hyper-reduction scheme?

Design
------
For each leg in `legs`:

  1. A *reference* continuation marches the leg with the tensor ROM, and a FOM
     natural continuation marches alongside it -- exactly the protocol in
     build_diagram_bare_with_sym._continue (first point of the leg seeded from the ROM,
     every subsequent point from the previous FOM solution).

  2. At each converged point the FOM velocity/pressure are projected onto
     cluster k's basis. That projection is the COMMON SEED.

  3. Every method in `methods` performs ONE solve from that common seed, with
     the cluster PINNED to the reference's choice. No method runs its own
     continuation, so branch-following never enters the comparison and err_u
     always compares like with like.

What this test does NOT measure: path-following robustness. That is Test B
(full diagram per method, fom_compute=False). Keep them separate.

Wiring required (two places, marked WIRE):
  * `methods` is a dict  name -> deim_ops_dict_or_None.  Pass None for the
    tensor-convection variant; pass a deim_ops_dict built at the desired
    budget for each DEIM variant.  Build those with whatever your offline
    routine is, e.g.
        deim_42  = build_deim_ops(..., m_J=42)
        deim_2r  = build_deim_ops(..., m_J=lambda r: 2*r)
        deim_3r  = build_deim_ops(..., m_J=lambda r: 3*r)
        methods  = {'tensor': None, 'deim_mJ42': deim_42,
                    'deim_2r': deim_2r, 'deim_3r': deim_3r}
  * `residual_history_of(callback_data)` must return the Newton residual norms
    for the solve.  The key name depends on your solve_rom internals.

Usage
-----
    from compare_hyperreduction import run_fixed_point_comparison, PROBE_LEGS

    rows = run_fixed_point_comparison(
        legs=PROBE_LEGS,
        methods=methods,
        clustering=clustering, operators=operators,
        reference_deim_ops=None,          # None -> tensor drives the reference path
        problem=problem, initial_solution=initial_solution,
        M_u=M_u, M_p=M_p,
        submesh_deims_by_method=submesh_deims_by_method,
    )
    import pandas as pd
    pd.DataFrame(rows).to_csv('hyperreduction_comparison.csv', index=False)
"""

import time

import numpy as np
from firedrake import Function

from nsrom.navier_stokes import compute_forces, solve_steady_navier_stokes
from nsrom.rom import solve_rom, reconstruct_solution, project_to_rom_coefficients
from nsrom.rom.local import _whitened_pod_distances

from build_diagram_bare_with_sym import _assemble_mixed, _norm_M, select_nearest_cluster


# ---------------------------------------------------------------------------
# Branch classification
# ---------------------------------------------------------------------------
# Baseline |C_L| on the symmetric branch is ~2e-4 (a constant offset from mesh
# asymmetry in the lift functional, flat in Re -- it does NOT grow through
# Re_c). Derailed states sit at ~2e-2. CL_SYM_TOL separates them with an order
# of magnitude of headroom either side.
CL_SYM_TOL = 5e-3


def branch_of(C_L, tol=CL_SYM_TOL):
    """'sym' | 'asym+' | 'asym-' from the lift coefficient."""
    if not np.isfinite(C_L):
        return 'failed'
    if abs(C_L) <= tol:
        return 'sym'
    return 'asym+' if C_L > 0 else 'asym-'


# ---------------------------------------------------------------------------
# Probe legs -- branch is an INPUT, and each branch marches in its safe
# direction.
# ---------------------------------------------------------------------------
# Symmetric legs: fixed Re, march amplitude outward from a = 0. The a = 0
#   trunk is reliable at every Re (90/90 in the tensor run), and sweeping a at
#   fixed Re crosses the critical curve in the merging direction, not the
#   splitting one.
# Asymmetric legs: fixed a, march Re DOWNWARD from Re_max, where |C_L| is
#   large and the three roots are well separated. Stops at the merge.
#
# Each leg: dict(branch=..., path=[(Re, amp), ...], seed=...)
#   seed='trunk'   -> take the a=0 symmetric anchor at the leg's first Re
#   seed='asym+'   -> take the asym+ anchor at Re_max
#   seed='asym-'   -> take the asym- anchor at Re_max

_AMP_LADDER_POS = [0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0]
_AMP_LADDER_NEG = [-a for a in _AMP_LADDER_POS]

# Re values for the symmetric amplitude sweeps: coarse coverage of P, plus
# rungs straddling the onsets seen in the tensor run (68.2 at a=0, ~78, ~83,
# ~97, ~105), plus two extrapolation rungs.
_SYM_RE = [40., 55., 68., 70., 78., 83., 90., 97., 100., 105., 109.]

# Amplitudes for the asymmetric down-sweeps, and the Re ladder they walk.
_ASYM_AMPS = [0.0, 0.25, 0.5, -0.25, -0.5, -1.0]
_ASYM_RE_DOWN = [109., 105., 100., 95., 90., 85., 80., 75., 70., 65., 60., 55.]

PROBE_LEGS = []
for _Re in _SYM_RE:
    for _ladder in (_AMP_LADDER_POS, _AMP_LADDER_NEG):
        PROBE_LEGS.append(dict(
            branch='sym',
            path=[(_Re, _a) for _a in _ladder],
            seed='trunk'))
for _a in _ASYM_AMPS:
    for _b in ('asym+', 'asym-'):
        PROBE_LEGS.append(dict(
            branch=_b,
            path=[(_r, _a) for _r in _ASYM_RE_DOWN],
            seed=_b))


def residual_history_of(callback_data):
    """WIRE: return the Newton residual norms recorded by solve_rom.

    Adjust the key to match your solve_rom internals. Returning None simply
    disables the residual-floor columns; everything else still works.
    """
    if callback_data is None:
        return None
    for key in ('residual_norms', 'residuals', 'fnorms', 'snes_history'):
        if isinstance(callback_data, dict) and key in callback_data:
            return list(np.asarray(callback_data[key], dtype=float).ravel())
    return None


# ---------------------------------------------------------------------------
# one method, one point, from a prescribed seed
# ---------------------------------------------------------------------------
def _solve_one(name, operators_k, deim_ops_k, submesh_deim_k,
               nu, amp, problem, u0, p0, k, u_fom, M_u):
    """Single ROM solve from the common seed. Returns a dict of measurements."""
    t0 = time.perf_counter()

    solution, w_rom, cb = solve_rom(
        operators=operators_k, nu=nu, amp=amp, problem=problem,
        deim_ops=deim_ops_k,
        u_initial_guess=u0, p_initial_guess=p0,
        submesh_deim=submesh_deim_k, return_internals=True)
    t_solve = time.perf_counter() - t0

    out = {
        'method': name,
        'cluster': k,
        'converged': bool(solution.converged),
        'iterations': int(solution.iterations),
        't_rom': t_solve,
        # separates "slower per step" from "needs more steps" -- report both
        't_per_iter': t_solve / max(solution.iterations, 1),
    }

    hist = residual_history_of(cb)
    if hist:
        out['res_initial'] = hist[0]
        out['res_final'] = hist[-1]
        # the DEIM stall signature: last-decade contraction ratio ~ 1
        out['res_floor_ratio'] = (hist[-1] / hist[-2]) if len(hist) > 1 else float('nan')
        out['res_history'] = ';'.join(f'{v:.6e}' for v in hist)

    if not solution.converged:
        out.update({'err_u': float('nan'), 'C_L_rom': float('nan')})
        return out, None

    sol_rom = reconstruct_solution(solution, operators_k, problem, amp=amp)
    u_rom = sol_rom.velocity.dat.data_ro.flatten()
    denom = _norm_M(u_fom, M_u)
    out['err_u'] = _norm_M(u_fom - u_rom, M_u) / denom if denom > 0 else float('nan')
    out['C_L_rom'] = float(compute_forces(sol_rom, problem).lift)
    return out, sol_rom


# ---------------------------------------------------------------------------
# anchors: known-branch states to seed the probe legs from
# ---------------------------------------------------------------------------
def build_branch_anchors(
    Re_list, clustering, operators, problem, initial_solution, M_u, M_p,
    *, reference_deim_ops=None, submesh_deims=None,
    branch_jump_fn=None, start_cluster=None, verbose=True,
):
    """Produce the states the probe legs are seeded from:

        anchors['sym'][Re]     symmetric trunk at a = 0, for every Re
        anchors['asym+'][Remax], anchors['asym-'][Remax]

    The a=0 trunk is marched upward in Re (unique and stable at low Re, and
    empirically derailment-free on this configuration). The two asymmetric
    anchors come from a branch jump at the top of the range, where the roots
    are furthest apart.

    `branch_jump_fn` must return (u_coeffs, p_coeffs) for a requested sign at
    the top Re; pass nsrom.bifurcation.branch_jump.in_sweep_branch_jump_rom
    wrapped to that signature. If None, only symmetric anchors are built.
    """
    submesh_deims = submesh_deims or {}
    anchors = {'sym': {}, 'asym+': {}, 'asym-': {}}
    prev_u = prev_p = prev_k = None

    prev_rom = None
    for Re in Re_list:
        nu = 1.0 / Re
        problem.reynolds.assign(Re)
        problem.amplitude.assign(0.0)
        if prev_u is None:
            # cluster 0 covers only Re>=62, amp<=-0.33 -- a poor basis for the
            # low-Re, a=0 start. Default to the widest-support cluster instead.
            k = (start_cluster if start_cluster is not None
                 else max(operators, key=lambda kk: operators[kk].Z_u.shape[1]))
            u_g, p_g = project_to_rom_coefficients(
                initial_solution.velocity.dat.data_ro.flatten(),
                initial_solution.pressure.dat.data_ro.flatten(),
                operators[k], amp=0.0, M_u=M_u, M_p=M_p)
        else:
            k = select_nearest_cluster(clustering, prev_u, prev_k)
            if k != prev_k:
                u_g, p_g = project_to_rom_coefficients(
                    prev_rom.velocity.dat.data_ro.flatten(),
                    prev_rom.pressure.dat.data_ro.flatten(),
                    operators[k], amp=0.0, M_u=M_u, M_p=M_p)
            else:
                u_g, p_g = prev_u, prev_p
        # else:
        #     k = select_nearest_cluster(clustering, prev_u, prev_k)
        #     u_g, p_g = prev_u, prev_p

        sol, _, _ = solve_rom(
            operators=operators[k], nu=nu, amp=0.0, problem=problem,
            deim_ops=(reference_deim_ops or {}).get(k),
            u_initial_guess=u_g, p_initial_guess=p_g,
            submesh_deim=submesh_deims.get(k), return_internals=True)
        if not sol.converged:
            if verbose:
                print(f"  [anchor] trunk diverged at Re={Re:.1f}; skipping")
            continue
        rom = reconstruct_solution(sol, operators[k], problem, amp=0.0)
        C_L = float(compute_forces(rom, problem).lift)
        if branch_of(C_L) != 'sym':
            # the trunk itself left the symmetric branch -- do not use it as an
            # anchor, and say so loudly: it invalidates the sym probe legs here
            if verbose:
                print(f"  [anchor] trunk NOT symmetric at Re={Re:.1f} "
                      f"(C_L={C_L:+.3e}); no sym anchor at this Re")
        else:
            anchors['sym'][Re] = {'u_coeffs': sol.velocity_coeffs,
                                  'p_coeffs': sol.pressure_coeffs,
                                  'k': k, 'amp': 0.0, 'Re': Re,
                                  'solution_rom': rom, 'C_L': C_L}
        prev_u, prev_p, prev_k = sol.velocity_coeffs, sol.pressure_coeffs, k
        prev_rom = rom
    if branch_jump_fn is not None and anchors['sym']:
        Re_top = max(anchors['sym'])
        for sign, name in ((+1, 'asym+'), (-1, 'asym-')):
            res = branch_jump_fn(anchors['sym'][Re_top], sign)
            if res is None:
                continue
            k = res['k']
            rom = reconstruct_solution(res['sol'], operators[k], problem, amp=0.0)
            problem.reynolds.assign(Re_top)
            problem.amplitude.assign(0.0)
            C_L = float(compute_forces(rom, problem).lift)
            got = branch_of(C_L)
            if got != name and verbose:
                print(f"  [anchor] requested {name}, jump gave {got} "
                      f"(C_L={C_L:+.3e}); storing under {got}")
            anchors[got][Re_top] = {'u_coeffs': res['sol'].velocity_coeffs,
                                    'p_coeffs': res['sol'].pressure_coeffs,
                                    'k': k, 'amp': 0.0, 'Re': Re_top,
                                    'solution_rom': rom, 'C_L': C_L}
    return anchors


def anchors_from_saved(anchors, saved, operators, clustering, problem,
                       M_u, M_p, Re_saved, verbose=True):
    """Seed the asymmetric anchors from ROM solutions already on disk, instead
    of a branch jump.

        saved     : {'asym+': rom_solution, 'asym-': rom_solution}
        Re_saved  : the Re those solutions were computed AT (e.g. 99.0), NOT
                    max(Re_list). _pick_anchor takes the nearest Re, so a leg
                    starting higher will jump from here on its first solve.

    The branch label is taken from C_L, not from the filename: if a file
    classifies the other way it is stored where it belongs and said out loud.
    """
    for name, rom in (saved or {}).items():
        if rom is None:
            continue
        u = rom.velocity.dat.data_ro.flatten()
        p = rom.pressure.dat.data_ro.flatten()

        # select in cluster 0's basis, then re-project into the chosen cluster
        a0, _ = project_to_rom_coefficients(
            u, p, operators[0], amp=0.0, M_u=M_u, M_p=M_p)
        k = select_nearest_cluster(clustering, a0, 0)
        u_c, p_c = project_to_rom_coefficients(
            u, p, operators[k], amp=0.0, M_u=M_u, M_p=M_p)

        problem.reynolds.assign(Re_saved)
        problem.amplitude.assign(0.0)
        C_L = float(compute_forces(rom, problem).lift)
        got = branch_of(C_L)
        if got != name and verbose:
            print(f"  [anchor] saved '{name}' classifies as '{got}' "
                  f"(C_L={C_L:+.3e}); storing under '{got}'")
        if got == 'sym' and verbose:
            print(f"  [anchor] WARNING saved '{name}' is symmetric "
                  f"(C_L={C_L:+.3e}) -- it will not seed an asym leg")
        anchors[got][Re_saved] = {'u_coeffs': u_c, 'p_coeffs': p_c, 'k': k,
                                  'amp': 0.0, 'Re': Re_saved,
                                  'solution_rom': rom, 'C_L': C_L}
    return anchors


def _project_for_selection(cfg, u_fom, p_fom, amp, M_u, M_p):
    """Reduced coefficients of the FOM state in config `cfg`'s own cluster-0
    basis, used only to drive that config's nearest-centroid selection.
    Returns (a_prev, k_prev) as select_nearest_cluster expects."""
    a0, _ = project_to_rom_coefficients(
        u_fom, p_fom, cfg['operators'][0], amp=amp, M_u=M_u, M_p=M_p)
    # k_prev MUST name the basis a0 actually lives in. Returning None reaches
    # _whitened_pod_distances as fs['G'][None] -> TypeError.
    return a0, 0

_SEED_ALIAS = {'trunk': 'sym'}

def _pick_anchor(anchors, kind, Re):
    pool = anchors.get(_SEED_ALIAS.get(kind, kind)) or {}
    if not pool:
        return None
    return pool[min(pool, key=lambda r: abs(r - Re))]

# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def run_fixed_point_comparison(
    legs, configs, anchors,
    clustering, operators,
    problem, initial_solution, M_u, M_p,
    *,
    reference_deim_ops=None,        # None -> tensor convection drives the path
    reference_submesh_deims=None,
    verbose=True,
):
    """Controlled comparison with the branch as an explicit input.

    `configs` is an ordered dict  name -> dict with keys

        clustering    : the clustering object for this configuration
        operators     : the per-cluster operator list
        deim_ops      : {k: deim_ops} or None for tensor convection
        submesh_deims : {k: SubMeshDEIM} or None
        pin_cluster   : bool. True when this config SHARES `clustering` with
                        the reference, so the reference's cluster choice can be
                        imposed (the correct control for a pure hyper-reduction
                        comparison). False when the config has its own
                        clustering -- e.g. the K=1 global ROM -- in which case
                        it runs its own selection, which is the honest thing to
                        compare.

    This lets ONE run cover both axes at once: hyper-reduction variants (same
    clustering, pin_cluster=True) and locality variants (own clustering,
    pin_cluster=False). Every configuration is evaluated from the SAME verified
    FOM solution at the SAME points, so the surviving point sets are identical
    by construction and the FOM is paid for once rather than once per config.

    At every point the reference FOM is classified by C_L and checked against
    the leg's requested branch. Points where the reference path derailed are
    NOT comparable: they are skipped and counted, never silently kept.

    Returns (rows, path_audit): the per-(point, method) records, and the list
    of points where the reference path left the requested branch.
    """
    # submesh_deims_by_method = submesh_deims_by_method or {}
    rows, path_audit = [], []

    for leg in legs:
        want = leg['branch']
        path = leg['path']
        if verbose:
            print(f"\n=== leg branch={want} "
                  f"({path[0][0]:.0f},{path[0][1]:+.2f}) -> "
                  f"({path[-1][0]:.0f},{path[-1][1]:+.2f})")

        anchor = _pick_anchor(anchors, leg['seed'], path[0][0])
        if anchor is None:
            if verbose:
                print(f"  no '{leg['seed']}' anchor available; skipping leg")
            continue

        prev_u, prev_p, prev_k = anchor['u_coeffs'], anchor['p_coeffs'], anchor['k']
        prev_sol_rom = anchor['solution_rom']
        prev_amp = anchor['amp']
        prev_fom = None

        for (Re, amp) in path:
            nu = 1.0 / Re
            problem.reynolds.assign(Re)
            problem.amplitude.assign(amp)

            # ---- 1. reference ROM step: defines the path and the cluster ----
            k_ref = select_nearest_cluster(clustering, prev_u, prev_k)
            if k_ref != prev_k:
                u_g, p_g = project_to_rom_coefficients(
                    prev_sol_rom.velocity.dat.data_ro.flatten(),
                    prev_sol_rom.pressure.dat.data_ro.flatten(),
                    operators[k_ref], amp=amp, M_u=M_u, M_p=M_p)
            else:
                u_g, p_g = prev_u, prev_p

            ref_sol, _, _ = solve_rom(
                operators=operators[k_ref], nu=nu, amp=amp, problem=problem,
                deim_ops=(reference_deim_ops or {}).get(k_ref),
                u_initial_guess=u_g, p_initial_guess=p_g,
                submesh_deim=(reference_submesh_deims or {}).get(k_ref),
                return_internals=True)
            if not ref_sol.converged:
                path_audit.append({'Re': Re, 'amp': amp, 'branch_requested': want,
                                   'reason': 'reference ROM diverged'})
                continue
            ref_rom = reconstruct_solution(ref_sol, operators[k_ref], problem, amp=amp)

            # ---- 2. FOM alongside; leg start seeded by ROM, then continuation ----
            guess = (_assemble_mixed(problem, ref_rom.velocity, ref_rom.pressure)
                     if prev_fom is None else prev_fom.solution)
            t0 = time.perf_counter()
            fom = solve_steady_navier_stokes(problem, initial_guess=guess)
            t_fom = time.perf_counter() - t0
            if not fom.converged:
                path_audit.append({'Re': Re, 'amp': amp, 'branch_requested': want,
                                   'reason': 'reference FOM diverged'})
                prev_u, prev_p, prev_k = (ref_sol.velocity_coeffs,
                                          ref_sol.pressure_coeffs, k_ref)
                prev_sol_rom, prev_amp = ref_rom, amp
                continue

            C_L_fom = float(compute_forces(fom, problem).lift)
            got = branch_of(C_L_fom)

            # ---- 3. VERIFY the branch. Do not trust the path. ----------------
            if got != want:
                path_audit.append({'Re': Re, 'amp': amp, 'branch_requested': want,
                                   'branch_actual': got, 'C_L_fom': C_L_fom,
                                   'reason': 'reference path left the branch'})
                if verbose:
                    print(f"  Re={Re:6.1f} a={amp:+.2f}  reference on '{got}', "
                          f"wanted '{want}' (C_L={C_L_fom:+.3e}) -- point not comparable")
                prev_fom = fom
                prev_u, prev_p, prev_k = (ref_sol.velocity_coeffs,
                                          ref_sol.pressure_coeffs, k_ref)
                prev_sol_rom, prev_amp = ref_rom, amp
                continue

            prev_fom = fom
            u_fom = fom.velocity.dat.data_ro.flatten()

            p_fom = fom.pressure.dat.data_ro.flatten()

            # ---- 4/5. every configuration, from the SAME verified FOM --------
            # The seed is the FOM projected onto whichever basis the config
            # uses, so no config is advantaged by its own solve history and the
            # global ROM is not judged on a basis it does not have.
            for name, cfg in configs.items():
                ops_c = cfg['operators']
                if cfg.get('pin_cluster', True):
                    k_c = k_ref              # same clustering -> impose it
                else:
                    k_c = select_nearest_cluster(
                        cfg['clustering'],
                        *_project_for_selection(cfg, u_fom, p_fom, amp, M_u, M_p))

                u0, p0 = project_to_rom_coefficients(
                    u_fom, p_fom, ops_c[k_c], amp=amp, M_u=M_u, M_p=M_p)

                rec, _ = _solve_one(
                    name, ops_c[k_c],
                    (cfg.get('deim_ops') or {}).get(k_c),
                    (cfg.get('submesh_deims') or {}).get(k_c),
                    nu, amp, problem, u0, p0, k_c, u_fom, M_u)
                got_m = branch_of(rec['C_L_rom']) if rec['converged'] else 'failed'
                rec.update({
                    'Re': Re, 'amp': amp,
                    'n_clusters': cfg['clustering'].n_clusters,
                    'branch_requested': want,
                    'branch_actual': got_m,
                    # given a seed VERIFIED on `want`, did the scheme stay there?
                    # this is the controlled basin-robustness measurement
                    'stayed_on_branch': (got_m == want),
                    'C_L_fom': C_L_fom,
                    't_fom': t_fom,
                    'speedup': (t_fom / rec['t_rom']) if rec['t_rom'] > 0 else float('nan'),
                })
                rows.append(rec)

                if verbose:
                    st = 'OK ' if rec['converged'] else 'DIV'
                    print(f"  Re={Re:6.1f} a={amp:+.2f} {name:<10s} {st} "
                          f"it={rec['iterations']:2d} "
                          f"err={rec.get('err_u', float('nan')):.3e} "
                          f"t={rec['t_rom']*1e3:6.1f}ms "
                          f"sp={rec['speedup']:5.1f}x"
                          + ("" if rec['stayed_on_branch'] else f"  [-> {got_m}]"))

            prev_u, prev_p, prev_k = (ref_sol.velocity_coeffs,
                                      ref_sol.pressure_coeffs, k_ref)
            prev_sol_rom, prev_amp = ref_rom, amp

    return rows, path_audit


# ---------------------------------------------------------------------------
# reduction to macro-ready numbers
# ---------------------------------------------------------------------------
def summarize(rows, Re_window=(40., 100.)):
    """Per-method statistics, split inside/outside the training window, with
    branch-mismatched and diverged rows excluded from the error columns."""
    import pandas as pd

    if not rows:
        raise RuntimeError(
            "summarize(): no rows. Every leg was skipped or every reference "
            "point was non-comparable. Check (a) leg seed names against the "
            "anchors dict keys, (b) whether asym anchors were actually built, "
            "(c) test_A_path_audit.csv.")

    df = pd.DataFrame(rows)
    lo, hi = Re_window
    df['in_P'] = (df.Re >= lo) & (df.Re <= hi)

    # never pool across branches, and never let an off-branch solve into
    # the error statistics -- it is not comparing the same solution
    ok = df[df.converged & df.stayed_on_branch]
    out = []
    for (name, br, in_P), g in ok.groupby(['method', 'branch_requested', 'in_P']):
        tot = df[(df.method == name) & (df.branch_requested == br)
                 & (df.in_P == in_P)]
        out.append({
            'method': name,
            'branch': br,
            'region': 'in_P' if in_P else 'extrapolation',
            'n_points': len(tot),
            'n_converged': int(tot.converged.sum()),
            'n_left_branch': int((tot.converged & ~tot.stayed_on_branch).sum()),
            'conv_rate': tot.converged.mean(),
            'on_branch_rate': (tot.converged & tot.stayed_on_branch).mean(),
            'err_mean': g.err_u.mean(),
            'err_median': g.err_u.median(),
            'err_max': g.err_u.max(),
            'iters_median': g.iterations.median(),
            't_rom_median': g.t_rom.median(),
            't_per_iter_median': g.t_per_iter.median(),
            'speedup_median': g.speedup.median(),
            'speedup_p10': g.speedup.quantile(0.10),
            'speedup_p90': g.speedup.quantile(0.90),
            'res_final_median': g.res_final.median() if 'res_final' in g else float('nan'),
        })
    return pd.DataFrame(out).sort_values(['region', 'branch', 'method'])


def to_macros(summary_df, prefix='HR'):
    """Emit \\newcommand lines for macros.tex, one block per method/region."""
    lines = []
    for _, r in summary_df.iterrows():
        _br = {'sym': 'Sym', 'asym+': 'AsymP', 'asym-': 'AsymM'}[r['branch']]
        tag = f"{prefix}{r['method'].replace('_', '').capitalize()}{_br}" \
              f"{'InP' if r['region'] == 'in_P' else 'Extrap'}"
        lines += [
            f"\\newcommand{{\\{tag}ErrMean}}{{{r['err_mean']:.2e}}}",
            f"\\newcommand{{\\{tag}ErrMax}}{{{r['err_max']:.2e}}}",
            f"\\newcommand{{\\{tag}ConvRate}}{{{100*r['conv_rate']:.1f}\\%}}",
            f"\\newcommand{{\\{tag}SpeedMed}}{{{r['speedup_median']:.1f}}}",
            f"\\newcommand{{\\{tag}OnBranch}}{{{100*r['on_branch_rate']:.1f}\\%}}",
            f"\\newcommand{{\\{tag}IterMed}}{{{r['iters_median']:.0f}}}",
        ]
    return '\n'.join(lines)
