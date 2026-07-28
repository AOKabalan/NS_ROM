#nsrom/bifurcation/branch_jump.py

import numpy as np
from firedrake import (
    Function, assemble, inner, dx, PointEvaluator, TrialFunctions, TestFunctions, dot, Constant
)

from nsrom.navier_stokes import solve_steady_navier_stokes

from nsrom.navier_stokes import compute_forces
from nsrom.bifurcation.eigen_solver import solve_leftmost_real_eigenpairs
from nsrom.rom import (
    reconstruct_solution, make_callbacks, solve_rom_with_internals,solve_rom,
)

def compute_rom_indicator(w_rom, callback_data, operators_k, M_uu_red_k,
                          n_eigenvalues=6, target=0.0):
    """μ_leftmost_real and ν_red from the reduced (J_red, M̃_red) pencil."""
    _, jac_cb = make_callbacks(callback_data)
    with w_rom.dat.vec_ro as vec:
        jac_cb(vec)

    rom_mesh     = callback_data['rom_mesh']
    W_rom        = callback_data['W_rom']
    A_eff_const  = callback_data['A_eff_const']
    conv_J_const = callback_data['conv_J_const']
    B_const      = callback_data['B_const']

    du, dp = TrialFunctions(W_rom)
    v,  q  = TestFunctions(W_rom)

    J_form = (
        inner(dot(A_eff_const,  du), v) * dx
        + inner(dot(conv_J_const, du), v) * dx
        + inner(dot(B_const.T,    dp), v) * dx
        + inner(dot(B_const,      du), q) * dx
    )
    M_form = inner(dot(Constant(M_uu_red_k, rom_mesh), du), v) * dx

    J_assembled = assemble(J_form, mat_type='aij')
    M_assembled = assemble(M_form, mat_type='aij')

    eigvals, vrs, _, nconv = solve_leftmost_real_eigenpairs(
        J_assembled.M.handle, M_assembled.M.handle,
        n_eigenvalues=n_eigenvalues, target=target,
    )
    if nconv == 0:
        return None
    imag_tol  = 1e-6 * max(abs(e) for e in eigvals)
    real_mask = np.abs(eigvals.imag) < imag_tol
    real_idx  = np.where(real_mask)[0]
    if len(real_idx) == 0:
        return None

    leftmost = int(real_idx[np.argmin(eigvals[real_mask].real)])
    mu       = float(eigvals[leftmost].real)
    nu_red   = vrs[leftmost].array_r.copy()

    # M̃_red-normalize on velocity block
    N_u  = operators_k.Z_u.shape[1]
    nu_v = nu_red[:N_u]
    norm_sq = float(nu_v @ (M_uu_red_k @ nu_v))
    if norm_sq > 1e-14:
        nu_red /= np.sqrt(norm_sq)

    return {'mu': mu, 'nu_red': nu_red, 'eigvals': eigvals}

def vec_to_function(vec, space, name="eigenfunction"):
    """Convert PETSc Vec to Firedrake Function."""
    f = Function(space, name=name)
    with f.dat.vec_wo as f_vec:
        vec.copy(f_vec)
    return f


def velocity_l2_norm(u):
    return float(np.sqrt(assemble(inner(u, u) * dx)))


def normalize_velocity_mode(mode):
    """
    Normalize mixed eigenmode using only the velocity component.

    Pressure scaling is arbitrary, so do not use pressure in the norm.
    """
    mode_normed = Function(mode.function_space(), name="normalized_mode")
    mode_normed.assign(mode)

    u_mode, _ = mode_normed.subfunctions
    nrm = velocity_l2_norm(u_mode)

    if nrm < 1e-14:
        raise ValueError("Eigenmode velocity norm is too small.")

    with mode_normed.dat.vec as v:
        v.scale(1.0 / nrm)

    return mode_normed, nrm


def make_branch_jump_initial_guess(
    sol_c,
    mode,
    sign,
    amplitude,
    *,
    perturb_pressure=False,
):
    """
    Create initial guess:

        w_guess = w_c + sign * amplitude * mode

    By default only velocity is perturbed; pressure is copied from sol_c.
    """
    w_guess = Function(sol_c.solution.function_space(), name="branch_jump_guess")
    w_guess.assign(sol_c.solution)

    u_guess, p_guess = w_guess.subfunctions
    u_c, p_c = sol_c.solution.subfunctions
    u_mode, p_mode = mode.subfunctions

    u_guess.assign(u_c + sign * amplitude * u_mode)

    if perturb_pressure:
        p_guess.assign(p_c + sign * amplitude * p_mode)
    else:
        p_guess.assign(p_c)

    return w_guess

def probe_asymmetry_score(solution, x_probe=5.0, y_max=2.0, n_points=21):
    """
    Simple wake-based asymmetry diagnostic.

    For the symmetric pinball branch:
        u_x is symmetric in y
        u_y is antisymmetric in y

    For an asymmetric branch, these symmetries are broken.
    """
    w = solution.solution if hasattr(solution, "solution") else solution

    if hasattr(w, "subfunctions"):
        subs = w.subfunctions
        if len(subs) >= 2:
            u = subs[0]      # mixed (u, p)
        elif len(subs) == 1:
            u = subs[0]      # velocity-only with one subfunction
        else:
            u = w
    else:
        u = w

    mesh = u.function_space().mesh()


    ys = np.linspace(-y_max, y_max, n_points)
    points = np.array([(x_probe, y) for y in ys], dtype=float)

    evaluator = PointEvaluator(mesh, points)
    values = np.asarray(evaluator.evaluate(u))

    ux = values[:, 0]
    uy = values[:, 1]

    def antisym_score(arr):
        return -float(np.dot(arr, arr[::-1]) / (np.dot(arr, arr) + 1e-30))

    ux_score = antisym_score(ux)
    uy_score = antisym_score(uy)

    return {
        "ux_antisym_score": ux_score,
        "uy_antisym_score": uy_score,
    }


def branch_jump_pitchfork(
    problem,
    sol_c,
    eigvec_c,
    Re_c,
    *,
    eps_values=(0.5, 1.0, 2.0),
    amp_values=(1e-3, 3e-3, 1e-2, 3e-2, 1e-1),
    perturb_pressure=False,
    asymmetry_tol=1e-3,
):

    mode = vec_to_function(
        eigvec_c,
        problem.mixed_space,
        name="pitchfork_eigenmode",
    )

    mode, raw_norm = normalize_velocity_mode(mode)

    print("\n--- Branch jump setup ---")
    print(f"  Re_c:                     {Re_c:.8f}")
    print(f"  Raw eigenmode velocity L2: {raw_norm:.6e}")
    print("  Mode normalized to ||nu_u||_L2 = 1")

    results = []

    for eps in eps_values:
        Re_jump = float(Re_c + eps)

        problem.reynolds.assign(Re_jump)
        problem.amplitude.assign(float(sol_c.amplitude))

        print(f"\n--- Trying Re_jump = {Re_jump:.6f}  eps = {eps:.3e} ---")
        print("  Solving symmetric anchor at Re_jump...")

        try:
            sol_sym_jump = solve_steady_navier_stokes(problem, sol_c.solution)
        except Exception as e:
            print(f"  Symmetric anchor solve failed: {e}")
            continue

        for amp in amp_values:
            for sign in (+1, -1):
                print(f"  Trying sign={sign:+d}, amplitude={amp:.3e}")

                w_guess = make_branch_jump_initial_guess(
                    sol_sym_jump,
                    mode,
                    sign,
                    amp,
                    perturb_pressure=perturb_pressure,
                )

                try:
                    sol_new = solve_steady_navier_stokes(problem, w_guess)
                except Exception as e:
                    print(f"    Newton failed: {e}")
                    continue

                scores = probe_asymmetry_score(sol_new)
                forces = compute_forces(sol_new, problem)

                broken = (
                    scores["ux_antisym_score"] > asymmetry_tol
                    or scores["uy_antisym_score"] > asymmetry_tol
                )

                print(
                    "    converged | "
                    f"ux_score={scores['ux_antisym_score']:+.3f}, "
                    f"uy_score={scores['uy_antisym_score']:+.3f}, "
                    f"C_L={forces.lift:+.6e}, "
                    f"broken={broken}"
                )

                if broken:
                    results.append(
                        {
                            "solution": sol_new,
                            "Re": Re_jump,
                            "epsilon": eps,
                            "amplitude": amp,
                            "sign": sign,
                            "scores": scores,
                            "C_L": float(forces.lift),
                            "C_D": float(forces.drag),
                        }
                    )

        cls_this_eps = [r["C_L"] for r in results if r["epsilon"] == eps]

        has_negative_lift = any(cl < 0.0 for cl in cls_this_eps)
        has_positive_lift = any(cl > 0.0 for cl in cls_this_eps)

        if has_negative_lift and has_positive_lift:
            print("\n  Found both lift-sign pitchfork branches.")
            break

    return results



def continue_asymmetric_branch(
    problem,
    sol_start,
    Re_max,
    *,
    dRe_near=0.25,
    dRe_far=2.0,
    near_steps=5,
    cl_collapse_factor=0.5,
):
    """
    Warm-started Newton continuation along an asymmetric branch.

    Starts at sol_start (assumed on the asymmetric branch, with C_L ≠ 0),
    advances in Re using small steps near the bifurcation and larger steps
    further out. Aborts if C_L collapses or flips sign (branch-jump back
    to symmetric).

    Returns
    -------
    list[NavierStokesSolution]  : solutions sampled along the branch,
                                  including sol_start as element 0.
    list[dict]                  : per-step diagnostics
                                  ({Re, dRe, C_L, C_D, snes_iters}).
    """
    from nsrom.navier_stokes import compute_forces

    sols = [sol_start]
    diagnostics = []

    forces = compute_forces(sol_start, problem)
    cl_initial = forces.lift
    cl_sign = np.sign(cl_initial)
    cl_prev = cl_initial

    diagnostics.append({
        "Re": float(sol_start.reynolds),
        "dRe": 0.0,
        "C_L": float(cl_initial),
        "C_D": float(forces.drag),
        "step": 0,
    })
    print(
        f"  [step  0] Re={sol_start.reynolds:7.4f}  "
        f"C_L={cl_initial:+.4e}  (seed)"
    )

    step = 0
    Re = float(sol_start.reynolds)
    while Re < Re_max - 1e-12:
        step += 1
        dRe = dRe_near if step <= near_steps else dRe_far
        Re_new = min(Re + dRe, Re_max)

        problem.reynolds.assign(Re_new)
        problem.amplitude.assign(float(sol_start.amplitude))

        try:
            sol_new = solve_steady_navier_stokes(problem, sols[-1].solution)
        except Exception as e:
            print(f"  [step {step:2d}] Re={Re_new:7.4f}  Newton FAILED: {e}")
            # Try once with halved step from the same anchor
            dRe_retry = 0.5 * dRe
            Re_retry = Re + dRe_retry
            problem.reynolds.assign(Re_retry)
            try:
                sol_new = solve_steady_navier_stokes(problem, sols[-1].solution)
                Re_new = Re_retry
                print(f"             retry at dRe={dRe_retry:.3f}: OK")
            except Exception as e2:
                print(f"             retry FAILED: {e2}; aborting continuation.")
                break

        forces = compute_forces(sol_new, problem)
        cl_new = forces.lift

        # Branch-jump detection
        flipped = np.sign(cl_new) != cl_sign
        collapsed = abs(cl_new) < cl_collapse_factor * abs(cl_prev)
        if flipped or collapsed:
            reason = "sign flip" if flipped else "collapse"
            print(
                f"  [step {step:2d}] Re={Re_new:7.4f}  "
                f"C_L={cl_new:+.4e}  -- BRANCH JUMP DETECTED ({reason})"
            )
            print(f"             aborting; last good Re={Re:.4f}")
            break

        sols.append(sol_new)
        diagnostics.append({
            "Re": float(Re_new),
            "dRe": float(Re_new - Re),
            "C_L": float(cl_new),
            "C_D": float(forces.drag),
            "step": step,
        })
        print(
            f"  [step {step:2d}] Re={Re_new:7.4f}  dRe={Re_new - Re:.3f}  "
            f"C_L={cl_new:+.4e}  C_D={forces.drag:+.4e}"
        )

        Re = Re_new
        cl_prev = cl_new

    return sols, diagnostics

def branch_jump_rom(
    *,
    rom_res,
    Re,
    amp,
    operators_k,
    M_uu_red_k,
    problem,
    eps,
    sign,
    mode,
    w_rom = None,
    callback = None,
    offset = 1,
    n_eigenvalues=6,
    target=0.0,
    enforce_eigvec_sign=True,
):

    # 1. Leftmost-real eigenvector at the symmetric state
    if hasattr(rom_res, "w_rom"):


        w_rom = rom_res.w_rom
        callback = rom_res.callback_data      # no trailing comma
        # callback = rom_res.callback_data,

    indicator = compute_rom_indicator(
        w_rom, callback,
        operators_k, M_uu_red_k,
        n_eigenvalues=n_eigenvalues, target=target,
    )
    if indicator is None:
        raise RuntimeError("compute_rom_indicator returned None at kick state.")

    nu_red  = indicator["nu_red"].copy()
    mu_symm = indicator["mu"]
    N_u     = operators_k.Z_u.shape[1]

    # 2. Sign convention: orient ν_red so y-antisymmetry is positive
    if enforce_eigvec_sign:

        nu_v_lifted_dofs = operators_k.Z_u @ nu_red[:N_u]
        nu_v_lifted = Function(problem.velocity_space)
        nu_v_lifted.dat.data[:] = nu_v_lifted_dofs.reshape(
            nu_v_lifted.dat.data.shape
        )
        v_score = probe_asymmetry_score(nu_v_lifted)
        if v_score['uy_antisym_score']< 0:
            nu_red = -nu_red

    # 3. Extract α_symm, p_symm from converged symmetric w_rom
    alpha_func, p_func = w_rom.subfunctions
    alpha_symm = alpha_func.dat.data_ro.flatten().copy()
    p_symm     = p_func.dat.data_ro.flatten().copy()

    # 4. Kick the velocity coefficients
    alpha_kicked = alpha_symm + sign * eps * nu_red[:N_u]

    # 5. Re-solve at the same (Re, amp) in the same cluster — no cluster reselection
    sol_kicked, w_rom_kicked, cbdata_kicked = solve_rom_with_internals(
        operators=operators_k,
        nu=1.0 / (Re+offset),
        amp=amp,
        problem=problem,
        mode=mode,
        u_initial_guess=alpha_kicked,
        p_initial_guess=p_symm,
    )

    # 6. Diagnostic: reconstruct and probe asymmetry
    if sol_kicked.converged:

        u_kicked = reconstruct_solution(
            sol_kicked, operators_k, problem, amp=amp,
        )
        scores = probe_asymmetry_score(u_kicked)
        sx = scores["ux_antisym_score"]
        sy = scores["uy_antisym_score"]
        forces = compute_forces(u_kicked,problem)
        C_L = float(forces.lift)
        print(f'The lift coefficient: {forces.lift}')

    else:
        sx, sy = float("nan"), float("nan")
        C_L = float("nan")

    return {
        "sol":            sol_kicked,
        "w_rom":          w_rom_kicked,
        "callback_data": cbdata_kicked,
        "converged":      bool(sol_kicked.converged),
        "Re":             float(Re+offset),
        "amp":            float(amp),
        "eps":            float(eps),
        "sign":           int(sign),
        "mu_symm":        float(mu_symm),
        "nu_red":         nu_red,
        "C_L":            C_L,
        "asymmetry_x":    float(sx),
        "asymmetry_y":    float(sy),
    }


def compute_ift_slope(
    *,
    Re_0,
    amp_0,
    dRe,
    damp,
    clustering,
    operators,
    deim_ops_dict,
    problem,
    initial_solution,
    M_u,
    M_p,
    M_uu_red,
    mode,
    n_eigenvalues=6,
    target=0.0,
):
    """
    Centered-FD estimate of dRe_c/damp via IFT applied to μ(Re, amp):
        dRe_c/damp ≈ -(∂μ/∂amp) / (∂μ/∂Re)
    evaluated near a (Re_0, amp_0) close to the bifurcation curve.
    """
    from nsrom.bifurcation.detection import evaluate_rom_point
    common = dict(
        clustering=clustering, operators=operators,
        deim_ops_dict=deim_ops_dict, problem=problem,
        initial_solution=initial_solution,
        M_u=M_u, M_p=M_p, M_uu_red=M_uu_red,
        mode=mode,
        n_eigenvalues=n_eigenvalues, target=target,
    )

    print("\n" + "=" * 80)
    print(f"IFT SLOPE at (Re={Re_0}, amp={amp_0})  dRe={dRe}  damp={damp}")
    print("=" * 80)

    r_c  = evaluate_rom_point(Re=Re_0,        amp=amp_0,        **common)
    r_Rp = evaluate_rom_point(Re=Re_0 + dRe,  amp=amp_0,        **common)
    r_Rm = evaluate_rom_point(Re=Re_0 - dRe,  amp=amp_0,        **common)
    r_Ap = evaluate_rom_point(Re=Re_0,        amp=amp_0 + damp, **common)
    r_Am = evaluate_rom_point(Re=Re_0,        amp=amp_0 - damp, **common)

    for label, r in [("center", r_c), ("Re+", r_Rp), ("Re-", r_Rm),
                     ("amp+", r_Ap), ("amp-", r_Am)]:
        if not r["converged"]:
            raise RuntimeError(
                f"Stencil point '{label}' did not converge: {r['failed_reason']}"
            )

    dmu_dRe  = (r_Rp["mu"] - r_Rm["mu"]) / (2.0 * dRe)
    dmu_damp = (r_Ap["mu"] - r_Am["mu"]) / (2.0 * damp)

    if abs(dmu_dRe) < 1e-12:
        raise RuntimeError("∂μ/∂Re ≈ 0; can't invert IFT here.")

    slope = -dmu_damp / dmu_dRe

    print(f"  μ(center)        = {r_c['mu']:+.6e}")
    print(f"  μ(Re±dRe)        = {r_Rp['mu']:+.6e},  {r_Rm['mu']:+.6e}")
    print(f"  μ(amp±damp)      = {r_Ap['mu']:+.6e},  {r_Am['mu']:+.6e}")
    print(f"  ∂μ/∂Re           = {dmu_dRe:+.6e}")
    print(f"  ∂μ/∂amp          = {dmu_damp:+.6e}")
    print(f"  dRe_c/damp       = {slope:+.6e}")

    return {
        "Re_0": Re_0, "amp_0": amp_0, "dRe": dRe, "damp": damp,
        "mu_center": r_c["mu"],
        "dmu_dRe": dmu_dRe,
        "dmu_damp": dmu_damp,
        "slope": slope,
        "evals": {"center": r_c, "Re+": r_Rp, "Re-": r_Rm,
                  "amp+": r_Ap, "amp-": r_Am},
    }

def in_sweep_branch_jump_rom(
    u_rom,
    p_rom,
    Re,
    amp,
    operators_k,
    M_uu_red_k,
    problem,
    eps,
    sign,
    indicator,
    *,
    mode,
    offset = 1,
    enforce_eigvec_sign=True,
    deim_ops = None,
    submesh = None
):




    if indicator is None:
        raise RuntimeError("compute_rom_indicator returned None at kick state.")

    nu_red  = indicator["nu_red"].copy()
    mu_symm = indicator["mu"]
    N_u     = operators_k.Z_u.shape[1]

    # 2. Sign convention: orient ν_red so y-antisymmetry is positive
    if enforce_eigvec_sign:

        nu_v_lifted_dofs = operators_k.Z_u @ nu_red[:N_u]
        nu_v_lifted = Function(problem.velocity_space)
        nu_v_lifted.dat.data[:] = nu_v_lifted_dofs.reshape(
            nu_v_lifted.dat.data.shape
        )
        v_score = probe_asymmetry_score(nu_v_lifted)
        if v_score['uy_antisym_score']< 0:
            nu_red = -nu_red

    # 3. Extract α_symm, p_symm from converged symmetric w_rom
    alpha_symm = u_rom.copy()
    p_symm     = p_rom.copy()

    # 4. Kick the velocity coefficients
    alpha_kicked = alpha_symm + sign * eps * nu_red[:N_u]
    sol_kicked, w_rom_kicked, cbdata_kicked = solve_rom(
            operators=operators_k,
            nu=1.0 / (Re+offset),
            amp=amp,
            problem=problem,
            mode=mode,
            deim_ops=deim_ops,
            submesh_deim=submesh,
            u_initial_guess=alpha_kicked,
            p_initial_guess=p_symm,
            return_internals=True,
        )

    if not sol_kicked.converged and mode == 'deim':
        # Fall back to exact projection: DEIM is the usual culprit near a kick.
        print("Didn't converge with DEIM, retrying with exact projection")
        sol_kicked, w_rom_kicked, cbdata_kicked = solve_rom(
            operators=operators_k,
            nu=1.0 / (Re+offset),
            amp=amp,
            problem=problem,
            mode='exact',
            u_initial_guess=alpha_kicked,
            p_initial_guess=p_symm,
            return_internals=True,
        )
        if not sol_kicked.converged:
            print("Didn't converge with exact projection either, returning anyway")


    # 6. Diagnostic: reconstruct and probe asymmetry
    if sol_kicked.converged:

        u_kicked = reconstruct_solution(
            sol_kicked, operators_k, problem, amp=amp,
        )
        scores = probe_asymmetry_score(u_kicked)
        sx = scores["ux_antisym_score"]
        sy = scores["uy_antisym_score"]
        forces = compute_forces(u_kicked,problem)
        print(f'The lift coefficient: {forces.lift}')
        C_L = float(forces.lift)

    else:
        sx, sy = float("nan"), float("nan")
        C_L = float("nan")

    return {
        "sol":            sol_kicked,
        "w_rom":          w_rom_kicked,
        "callback_data": cbdata_kicked,
        "converged":      bool(sol_kicked.converged),
        "Re":             float(Re+offset),
        "amp":            float(amp),
        "eps":            float(eps),
        "sign":           int(sign),
        "mu_symm":        float(mu_symm),
        "nu_red":         nu_red,
        "C_L":            C_L,
        "asymmetry_x":    float(sx),
        "asymmetry_y":    float(sy),
    }


