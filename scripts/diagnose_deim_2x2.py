"""
2x2 mixed-mode DEIM diagnostic  (wired to the real nsrom solver).

Ground truth: exact projection converges in ~3 iterations. DEIM diverges.
Since F and J share _reconstruct_velocity, sm.transfer_to_submesh and sm.u_sub,
everything common is exonerated. The fault is in one (or both) of the two
operations that differ between the exact and DEIM paths:
      sm.extract_residual(F_sub)      (DEIM residual)
      sm.extract_jacobian(J_data_sub) (MDEIM Jacobian)

This runs the reduced Newton solve with all four (F, J) source combinations at
a failing point, using the exact operators as a trusted reference:

    F      J        if this FAILS, it means
    ------ ------   ---------------------------------------------
    exact  exact    (baseline; must converge -- sanity check)
    exact  deim     MDEIM Jacobian (extract_jacobian) is the culprit
    deim   exact    DEIM residual (extract_residual) is the culprit
    deim   deim     (reproduces the observed divergence)

NOTE on line search: your driver uses snes_linesearch_type='basic' (full Newton
steps, no sufficient-decrease test). Under 'basic', F/J *mutual* inconsistency
cannot cause a line-search failure -- so divergence here means the Newton
DIRECTION is wrong, i.e. an individually-defective operator. Expect the culprit
to be a single mixed case (most likely exact-F/deim-J). We also run with 'bt' as
a cross-check, because 'bt' *is* sensitive to mutual inconsistency: if deim/deim
fails under 'bt' but both mixed cases pass, that's the mutual-consistency
signature that 'basic' would hide.

--------------------------------------------------------------------------------
HOW IT WORKS
The solver bakes callbacks in via make_callbacks() inside _setup_rom_solver.
We monkeypatch make_callbacks in the solver's own module namespace so it returns
a mismatched (res_cb, jac_cb) pair chosen by two module globals. Both closures
still close over the same callback_data (same submesh, same conv_* Constants), so
res writes conv_res_const via the F-mode path and jac writes conv_J_const via the
J-mode path -- exactly the mix we want. No edits to your source files.
--------------------------------------------------------------------------------

USAGE  (call from inside your own setup, where the per-cluster objects exist)

    from diagnose_deim_2x2 import run_2x2

    # e.g. right where _continue has computed u_coeffs, p_coeffs for the
    # failing point, or from main_local after operators/deim_ops_dict/
    # submesh_deims are built:
    run_2x2(
        operators=operators, deim_ops_dict=deim_ops_dict,
        submesh_deims=submesh_deims, problem=problem,
        k=1, Re=94.0, amp=0.0,
        u0=u_coeffs, p0=p_coeffs,       # the continuation guess (recommended)
    )

If u0/p0 are omitted the function first solves exact/exact from a zero guess and
reuses that converged state as the warm start for all four combos (isolates the
operator from the initial guess). Passing the real continuation guess is the
most faithful reproduction.
"""

import sys
import numpy as np


# ------------------------------------------------------------------------------
# Locate the solver module and its make_callbacks, path-agnostically.
# We resolve the module that actually defines solve_rom, then patch the
# make_callbacks name bound in THAT namespace (that's the one _setup_rom_solver
# calls).
# ------------------------------------------------------------------------------
def _resolve_solver_module():
    import nsrom.rom as rom_pkg
    rs = sys.modules[rom_pkg.solve_rom.__module__]
    if not hasattr(rs, 'make_callbacks'):
        raise RuntimeError(
            f"module {rs.__name__} has no 'make_callbacks' to patch; "
            "check that _setup_rom_solver calls the module-level make_callbacks."
        )
    return rom_pkg, rs


_ROM_PKG, _RS = _resolve_solver_module()
_REAL_MAKE_CALLBACKS = _RS.make_callbacks

# module globals selecting which source each callback comes from
_F_MODE = 'deim'   # 'deim' or 'exact'
_J_MODE = 'deim'


def _patched_make_callbacks(callback_data):
    """
    Build res from the F-mode path and jac from the J-mode path by toggling
    callback_data['use_deim'] around two real make_callbacks calls. The mode is
    captured inside make_callbacks at build time (closure over use_deim), so the
    returned closures are locked to the mode we chose. Both share callback_data.
    """
    orig = callback_data['use_deim']
    try:
        callback_data['use_deim'] = (_F_MODE == 'deim')
        res_cb, _ = _REAL_MAKE_CALLBACKS(callback_data)

        callback_data['use_deim'] = (_J_MODE == 'deim')
        _, jac_cb = _REAL_MAKE_CALLBACKS(callback_data)
    finally:
        callback_data['use_deim'] = orig
    return res_cb, jac_cb


# ------------------------------------------------------------------------------
# One solve at a chosen (F, J) source mix.
# ------------------------------------------------------------------------------
def _solve_once(operators, deim_ops_dict, submesh_deims, problem,
                k, nu, amp, u0, p0, fmode, jmode, linesearch, monitor):
    global _F_MODE, _J_MODE
    _F_MODE, _J_MODE = fmode, jmode

    params = {'snes_linesearch_type': linesearch}
    if monitor:
        params['snes_monitor'] = None
        params['snes_converged_reason'] = None

    _RS.make_callbacks = _patched_make_callbacks
    try:
        solution, w_rom, cd = _RS.solve_rom(
            operators=operators[k], nu=nu, amp=amp, problem=problem,
            deim_ops=deim_ops_dict[k], submesh_deim=submesh_deims.get(k),
            u_initial_guess=u0, p_initial_guess=p0,
            return_internals=True, verbose=False,
            solver_parameters=params,
        )
    finally:
        _RS.make_callbacks = _REAL_MAKE_CALLBACKS   # always restore

    return solution


# ------------------------------------------------------------------------------
# The 2x2 at one point.
# ------------------------------------------------------------------------------
def run_2x2(operators, deim_ops_dict, submesh_deims, problem,
            k, Re, amp, u0=None, p0=None,
            linesearch='basic', also_bt=True, monitor=False):
    """
    operators, deim_ops_dict : indexable by cluster k (list or dict)
    submesh_deims            : dict {k: SubMeshDEIM}, as built in
                               build_full_diagram_bare
    u0, p0                   : reduced-coord initial guess (recommended: the
                               continuation guess at this point). If None, an
                               exact/exact solve from zero bootstraps the guess.
    linesearch               : primary line search to test ('basic' matches your
                               driver).
    also_bt                  : if True, repeat under 'bt' to expose mutual
                               inconsistency that 'basic' hides.
    monitor                  : print PETSc snes residual history per combo.
    """
    nu = 1.0 / Re
    print(f"\n{'='*72}")
    print(f"2x2 DEIM diagnostic  |  Re={Re}  amp={amp}  cluster={k}  "
          f"linesearch={linesearch}")
    print('='*72)

    # --- bootstrap a warm start from exact/exact if none supplied ---
    if u0 is None or p0 is None:
        print("  [warm-start] no guess supplied; bootstrapping via exact/exact "
              "from zero")
        boot = _solve_once(operators, deim_ops_dict, submesh_deims, problem,
                           k, nu, amp, None, None, 'exact', 'exact',
                           linesearch, monitor=False)
        if not boot.converged:
            print("  [abort] exact/exact from zero did not converge; supply "
                  "u0/p0 (the continuation guess) for this point.")
            return None
        u0, p0 = boot.velocity_coeffs.copy(), boot.pressure_coeffs.copy()
        print(f"  [warm-start] ||u0||={np.linalg.norm(u0):.4e} "
              f"(exact/exact, it={boot.iterations})")

    combos = [('exact', 'exact'),
              ('exact', 'deim'),
              ('deim',  'exact'),
              ('deim',  'deim')]

    def _run_block(ls):
        print(f"\n  --- line search: {ls} ---")
        res = {}
        for (fm, jm) in combos:
            sol = _solve_once(operators, deim_ops_dict, submesh_deims, problem,
                              k, nu, amp, u0.copy(), p0.copy(), fm, jm, ls,
                              monitor)
            res[(fm, jm)] = (bool(sol.converged), int(sol.iterations))
            tag = 'OK ' if sol.converged else 'DIV'
            print(f"    F={fm:5s}  J={jm:5s}   {tag}  it={sol.iterations}")
        _interpret(res, ls)
        return res

    out = {linesearch: _run_block(linesearch)}
    if also_bt and linesearch != 'bt':
        out['bt'] = _run_block('bt')
    return out


def _interpret(r, ls):
    def ok(fm, jm):
        return r.get((fm, jm), (False, 0))[0]

    print("    ---- verdict ----")
    if not ok('exact', 'exact'):
        print("    baseline exact/exact FAILED from this guess -- not a clean")
        print("    test point (possible real fold, or bad guess). Stop.")
        return
    if ok('deim', 'deim'):
        print("    deim/deim CONVERGED here -- this point doesn't reproduce the")
        print(f"    bug under '{ls}'. Pick a harder point (e.g. deeper into the")
        print("    cascade) or try the other line search.")
        return

    ef_dj = ok('exact', 'deim')   # exact F, deim J
    df_ej = ok('deim',  'exact')  # deim F, exact J

    if not ef_dj and df_ej:
        print("    => DEIM JACOBIAN is the culprit (extract_jacobian).")
        print("       Next: check the CSR values-only extraction. The Jacobian")
        print("       callback passes only J_sub CSR *values* (no indptr/indices)")
        print("       to extract_jacobian, which assumes the submesh sparsity")
        print("       ordering is bit-stable across every online assembly. Diff")
        print("       the submesh J 'indices' array at a low-Re vs high-Re state.")
    elif ef_dj and not df_ej:
        print("    => DEIM RESIDUAL is the culprit (extract_residual).")
        print("       Unifying F/J points would have HIDDEN this. Inspect the")
        print("       residual index remap in SubMeshDEIM.")
    elif not ef_dj and not df_ej:
        print("    => BOTH DEIM operators individually break the solve.")
        print("       Two extraction bugs, or a shared submesh-transfer issue")
        print("       that only bites when BOTH sides read the submesh state.")
    else:
        # both mixed converged, only deim/deim failed
        print("    => Each DEIM operator is individually FINE; the failure is")
        print("       F/J MUTUAL inconsistency (different point sets =>")
        print("       J != d(DEIM-F)/da). Correct fix: build J as the derivative")
        print("       of the DEIM residual (same points & basis), NOT an")
        print("       independent MDEIM. (Expected to appear under 'bt', not")
        print("       'basic'.)")


# ------------------------------------------------------------------------------
# Residual diff: WHY is the DEIM residual wrong?
# The 2x2 fingered extract_residual. This localizes the fault inside it by
# evaluating the exact and DEIM convective residual callbacks at the SAME
# converged state and diffing them -- no solver in the comparison loop.
#
# We first run exact/exact to get a converged w_rom (the true reduced state),
# then call both residual callbacks on it. f_exact = Z_u^T F(u) is essentially
# the true reduced convective residual at that state, so the diff isolates the
# DEIM approximation error with nothing else moving.
# ------------------------------------------------------------------------------
def diff_residual(operators, deim_ops_dict, submesh_deims, problem,
                  k, Re, amp, u0=None, p0=None, n_worst=12):
    """
    Returns dict(f_exact, f_deim, rel) and prints a localization hint.

    Interpreting rel = ||f_deim - f_exact|| / ||f_exact||:
      rel ~ O(1) and ||f_deim|| << ||f_exact||
          -> DEIM residual not reconstructed: MISSING/incorrect DEIM lift.
             extract_residual should return the reduced, lifted residual
             Z_u^T U_F (P^T U_F)^-1 P^T F_sub, not the raw sampled entries.
      rel ~ O(1) and ||f_deim|| ~ ||f_exact|| and sorted(values) match
          -> ORDERING error: sample-row gather uses full-mesh vs submesh
             numbering, or the node map is applied in the wrong direction.
      rel ~ O(1), same magnitude, NOT a permutation
          -> wrong sample points P or wrong basis U_F in the lift.
      rel small but concentrated in a few rows that GROW as Re rises
          -> per-index remap error that only bites where the convective
             residual is large (matches the Re-escalation).
    """
    global _F_MODE, _J_MODE
    nu = 1.0 / Re
    print(f"\n{'='*72}")
    print(f"residual diff  |  Re={Re}  amp={amp}  cluster={k}")
    print('='*72)

    # --- converged reference state via exact/exact ---
    _F_MODE, _J_MODE = 'exact', 'exact'
    _RS.make_callbacks = _patched_make_callbacks
    try:
        sol, w_rom, cd = _RS.solve_rom(
            operators=operators[k], nu=nu, amp=amp, problem=problem,
            deim_ops=deim_ops_dict[k], submesh_deim=submesh_deims.get(k),
            u_initial_guess=u0, p_initial_guess=p0,
            return_internals=True, verbose=False,
            solver_parameters={'snes_linesearch_type': 'basic'},
        )
    finally:
        _RS.make_callbacks = _REAL_MAKE_CALLBACKS
    if not sol.converged:
        print("  [abort] exact/exact did not converge from this guess; "
              "pass u0/p0 (the continuation guess).")
        return None

    # --- build both residual callbacks on the SAME callback_data ---
    orig = cd['use_deim']
    try:
        cd['use_deim'] = False
        res_exact, _ = _REAL_MAKE_CALLBACKS(cd)
        cd['use_deim'] = True
        res_deim,  _ = _REAL_MAKE_CALLBACKS(cd)
    finally:
        cd['use_deim'] = orig

    # --- evaluate both at the converged state (read-only; no write-back) ---
    with w_rom.dat.vec_ro as v:
        res_exact(v)
    f_exact = np.array(cd['conv_res_const'].dat.data_ro).flatten().copy()
    with w_rom.dat.vec_ro as v:
        res_deim(v)
    f_deim = np.array(cd['conv_res_const'].dat.data_ro).flatten().copy()

    d = f_deim - f_exact
    n_exact = np.linalg.norm(f_exact)
    denom = n_exact if n_exact > 0 else 1.0
    rel = np.linalg.norm(d) / denom
    ratio = np.linalg.norm(f_deim) / denom

    print(f"  ||f_exact|| = {n_exact:.6e}   (~ true reduced convective residual)")
    print(f"  ||f_deim||  = {np.linalg.norm(f_deim):.6e}")
    print(f"  rel err     = {rel:.6e}")

    worst = np.argsort(-np.abs(d))[:n_worst]
    print(f"  worst {n_worst} rows   (row:   f_exact      f_deim       diff):")
    for i in worst:
        print(f"    {int(i):4d}:  {f_exact[i]:+.4e}  {f_deim[i]:+.4e}  {d[i]:+.4e}")

    # --- structural hint ---
    print("  ---- hint ----")
    if rel > 0.1 and ratio < 0.5:
        print("  ||f_deim|| << ||f_exact||: residual not being reconstructed.")
        print("  Suspect a MISSING/incorrect DEIM lift in extract_residual")
        print("  (returns raw sampled entries instead of the reduced lifted")
        print("  residual Z_u^T U_F (P^T U_F)^-1 P^T F_sub).")
    elif rel > 0.1:
        s_ex = np.sort(f_exact)
        s_dm = np.sort(f_deim)
        perm_rel = np.linalg.norm(s_dm - s_ex) / denom
        if perm_rel < 0.05:
            print("  f_deim is ~a PERMUTATION of f_exact (sorted values match).")
            print("  The sample-row gather uses the WRONG ordering: full-mesh vs")
            print("  submesh DOF numbering, or the node map applied backwards.")
        else:
            print("  Same magnitude but not a permutation: wrong sample points P")
            print("  or wrong basis U_F in the DEIM lift. Inspect both.")
    else:
        print("  Small global error, concentrated in the rows above. Re-run at")
        print("  higher Re (96, 98): if those rows grow, it's a per-index remap")
        print("  error that only bites where the convective residual is large.")

    return {'f_exact': f_exact, 'f_deim': f_deim, 'rel': rel}


# ------------------------------------------------------------------------------
# Optional standalone entry: only works if you expose a setup that returns the
# per-cluster objects. Easiest is to import run_2x2 from your main_local instead.
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    print(__doc__)
    print("Import run_2x2 into your setup script and call it there -- this file "
          "needs your per-cluster operators/deim_ops_dict/submesh_deims, which "
          "live in main_local's setup.")
