"""
Validation suite for the oversampled-residual DEIM fix.

Run AFTER rebuilding all clusters with oversample_residual=True.

Five checks, in order of what they protect:
  1. No-regression: clusters that already worked (sym trunk, asym-1) still solve
     in the same iteration counts and diff_residual stays ~1e-4 at the solution.
  2. Conditioning: cond(U_F_P) dropped vs the square build (the mechanism, in one
     number). This is the direct evidence oversampling fixed the amplification.
  3. Full-leg convergence: the whole asym+1 leg (Re 92..99) AND the amp spines
     converge, not just the three points we instrumented.
  4. Speedup intact: t_rom per point didn't blow up from the wider residual gather.
  5. (optional, cheap) Out-of-span number at a converged state, as a baseline to
     compare against if divergence ever reappears after you change POD tol / Re.

WIRING: same objects as the diagnostic hooks. Provide a small setup helper that
returns the per-cluster structures, OR call the individual check functions from
inside main_local after setup. The per-point solve reuses solve_rom exactly as
build_diagram_bare does (basic line search).
"""

import time
import numpy as np

import sys
def _rs():
    import nsrom.rom as rom_pkg
    return rom_pkg, sys.modules[rom_pkg.solve_rom.__module__]
_ROM_PKG, _RS = _rs()


def _solve(operators, deim_ops_dict, submesh_deims, problem, k, Re, amp, u0, p0,
           params=None):
    nu = 1.0 / Re
    if params is None:
        params = {'snes_linesearch_type': 'basic'}
    t0 = time.perf_counter()
    sol, w, cd = _RS.solve_rom(
        operators=operators[k], nu=nu, amp=amp, problem=problem,
        deim_ops=deim_ops_dict[k], submesh_deim=submesh_deims.get(k),
        u_initial_guess=u0, p_initial_guess=p0,
        return_internals=True, verbose=False, solver_parameters=params,
    )
    dt = time.perf_counter() - t0
    return sol, w, cd, dt


# ------------------------------------------------------------------------------
# CHECK 1 + 3 : convergence over a set of points (regression + full-leg)
# Pass the (Re, amp, k) points you care about. For regression use the sym trunk
# and asym-1 clusters; for the full-leg use asym+1 across Re and amp spines.
# ------------------------------------------------------------------------------
def check_convergence(operators, deim_ops_dict, submesh_deims, problem,
                      points, guesses, label=""):
    """
    points  : list of (Re, amp, k)
    guesses : dict (Re, amp, k) -> (u0, p0)   the continuation guess per point
              (reuse the same guesses you'd feed in the real sweep so this is a
              faithful test, not a warm-start artifact).
    """
    print(f"\n{'='*68}\nCONVERGENCE  {label}\n{'='*68}")
    print(f"  {'Re':>5} {'amp':>6} {'k':>2}  {'conv':>5} {'it':>4} {'t_rom[s]':>9}")
    n_fail = 0
    for (Re, amp, k) in points:
        u0, p0 = guesses[(Re, amp, k)]
        sol, w, cd, dt = _solve(operators, deim_ops_dict, submesh_deims,
                                problem, k, Re, amp, u0, p0)
        ok = bool(sol.converged)
        n_fail += (not ok)
        flag = '' if ok else '   <-- FAIL'
        print(f"  {Re:5.1f} {amp:6.2f} {k:2d}  {str(ok):>5} "
              f"{sol.iterations:4d} {dt:9.4f}{flag}")
    print(f"  ---- {len(points)-n_fail}/{len(points)} converged ----")
    return n_fail == 0


# ------------------------------------------------------------------------------
# CHECK 2 : conditioning of the residual sample matrix, per cluster.
# Reads it straight off the built operators -- no solve. Compare oversampled vs
# what the square build would have given at the same m_F.
# ------------------------------------------------------------------------------
def check_conditioning(deim_ops_dict, basis_paths, m_F_square, clusters):
    """
    deim_ops_dict : the CURRENT (oversampled) operators, per cluster.
    basis_paths   : dict k -> basis .npz path (to recompute the square-build
                    points for a fair before/after cond comparison).
    m_F_square    : the m_F you used in the OLD square build (per cluster or int).
    """
    from nsrom.rom.deim import load_basis, deim_algorithm
    print(f"\n{'='*68}\nCONDITIONING  (residual sample matrix U_F_P)\n{'='*68}")
    print(f"  {'k':>2}  {'n_over':>6} {'cond_over':>11} "
          f"{'m_F_sq':>6} {'cond_square':>12}  {'ratio':>8}")
    for k in clusters:
        ops = deim_ops_dict[k]
        # oversampled: reconstruct U_F_P from stored indices_F + basis modes_F
        basis = load_basis(basis_paths[k])
        n_over = len(ops.indices_F)
        # infer m_F (number of residual modes) from V_T_B_F: (r, n_over) and the
        # stored metadata if present
        m_F = ops.metadata.get('m_F_modes',
              ops.metadata.get('n_deim_points_F', None))
        # fall back: modes used = rank of the basis slice actually referenced
        if m_F is None:
            m_F = basis['modes_F'].shape[1]
        U_F = basis['modes_F'][:, :m_F]
        cond_over = np.linalg.cond(U_F[ops.indices_F, :])

        mfs = m_F_square[k] if isinstance(m_F_square, dict) else m_F_square
        sq_pts = deim_algorithm(basis['modes_F'][:, :mfs])
        cond_sq = np.linalg.cond(basis['modes_F'][:, :mfs][sq_pts, :])

        ratio = cond_sq / cond_over if cond_over > 0 else np.inf
        print(f"  {k:2d}  {n_over:6d} {cond_over:11.2e} "
              f"{mfs:6d} {cond_sq:12.2e}  {ratio:8.1f}x")
    print("  (want cond_over << cond_square; ratio >> 1 confirms the mechanism)")


# ------------------------------------------------------------------------------
# CHECK 4 : speedup / cost sanity -- t_rom vs an exact-projection solve at the
# same point. Confirms the wider residual gather didn't erase the speedup.
# ------------------------------------------------------------------------------
def check_speedup(operators, deim_ops_dict, submesh_deims, problem,
                  point, guess):
    (Re, amp, k) = point
    u0, p0 = guess
    print(f"\n{'='*68}\nSPEEDUP  (Re={Re} amp={amp} k={k})\n{'='*68}")
    # DEIM (oversampled) solve
    _, _, _, t_deim = _solve(operators, deim_ops_dict, submesh_deims, problem,
                             k, Re, amp, u0, p0)
    # exact-projection solve (no deim_ops): pass deim_ops=None / submesh=None
    nu = 1.0 / Re
    t0 = time.perf_counter()
    sol_ex, _, _ = _RS.solve_rom(
        operators=operators[k], nu=nu, amp=amp, problem=problem,
        deim_ops=None, submesh_deim=None,
        u_initial_guess=u0, p_initial_guess=p0,
        return_internals=True, verbose=False,
        solver_parameters={'snes_linesearch_type': 'basic'},
    )
    t_exact = time.perf_counter() - t0
    print(f"  t_rom (oversampled DEIM): {t_deim:.4f} s")
    print(f"  t_rom (exact projection): {t_exact:.4f} s")
    print(f"  DEIM speedup vs exact-projection: {t_exact/t_deim:.2f}x")
    print("  (compare against your FOM time for the headline number)")


# ------------------------------------------------------------------------------
# CHECK 5 : out-of-span baseline of the convective residual at a converged state.
# Store this now; if divergence ever returns after changing POD tol / Re, re-run
# and compare -- a jump means the basis lost the direction (enrich snapshots).
# ------------------------------------------------------------------------------
def check_out_of_span(operators, deim_ops_dict, submesh_deims, problem,
                      point, guess, basis_path):
    from nsrom.rom.deim import load_basis
    (Re, amp, k) = point
    u0, p0 = guess
    sol, w_rom, cd, _ = _solve(operators, deim_ops_dict, submesh_deims,
                               problem, k, Re, amp, u0, p0)
    if not sol.converged:
        print("  [check5] point didn't converge; skip")
        return None
    # full convective residual at the converged state, via exact callback
    orig = cd['use_deim']
    try:
        cd['use_deim'] = False
        res_exact, _ = _RS.make_callbacks(cd)
        with w_rom.dat.vec_ro as v:
            res_exact(v)
    finally:
        cd['use_deim'] = orig
    # NOTE: conv_res_const holds the REDUCED residual; for the true out-of-span
    # number you want the FULL residual vector F(u) projected against U_F. If you
    # have the full assembled convective vector available in cd (e.g.
    # cd['conv_full']), use it here. Otherwise this reports the reduced-residual
    # magnitude as a proxy baseline to track over time.
    f = np.array(cd['conv_res_const'].dat.data_ro).ravel()
    print(f"\n{'='*68}\nOUT-OF-SPAN BASELINE (Re={Re} amp={amp} k={k})\n{'='*68}")
    print(f"  ||reduced convective residual|| at solution = {np.linalg.norm(f):.3e}")
    print("  store this; a large jump after changing POD tol/Re => enrich"
          " snapshots at Newton iterates, not oversample further.")
    return np.linalg.norm(f)


if __name__ == '__main__':
    print(__doc__)
    print("Import the check_* functions into main_local and call them after "
          "setup; each needs the per-cluster operators/deim_ops_dict/"
          "submesh_deims and the continuation guesses.")
