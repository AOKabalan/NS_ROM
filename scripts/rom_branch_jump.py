"""
Script (2): ROM-level branch jump validation for the pitchfork.

Pure ROM coefficient-space design:

  - Anchor at Re_anchor = Re_c - δ_anchor (avoids Newton stalling at the
    singular Jacobian at Re_c).
  - Symmetric solve gives (α_sym, β_sym, k_anchor); indicator gives ν_red,
    M̃-normalized in cluster k_anchor's basis.
  - Each kick is a direct vector update:
        α_init = α_sym + sign · amp · ν_red[:N_u]
        β_init = β_sym
    and we call solve_rom_with_internals directly, fixing the cluster.
    No FOM round-trip; no cluster reselection.

If the asymmetric branch ends up needing a different basis at higher Re,
Newton iteration counts will rise and eventually diverge — an honest
diagnostic. Cluster switching is a follow-on, not pre-optimization.
"""

import numpy as np
from firedrake import (
    TrialFunctions, TestFunctions, inner, dot, dx, assemble, Constant,
)

from nsrom.rom import (
    assemble_inner_product_matrix, reconstruct_solution, make_callbacks,
    solve_rom_with_internals,
)
from nsrom.navier_stokes import (
    setup_navier_stokes_problem, load_solution, compute_forces,
)
from nsrom.snapshot_collection import load_snapshot_dofs
from nsrom.lifting_functions import load_lifting_functions
from nsrom.cluster_building import build_energy_snapshots, compute_or_load_clustering
from nsrom.local_rom_old import build_local_roms, solve_at_parameter_with_internals
from nsrom.layout import LocalROMLayout
from nsrom.config import LocalROMConfig
from nsrom.bifurcation.eigen_solver import solve_leftmost_real_eigenpairs
from nsrom.bifurcation.branch_jump import probe_asymmetry_score


# =============================================================================
# CONFIG
# =============================================================================

cfg = LocalROMConfig(
    n_clusters=5, inner_product_type="H1",
    pod_energy_tol=1e-8,
    n_velocity_max=50, n_pressure_max=50, n_supremizer_max=50,
    boundary_markers=(1, 3, 4, 5, 6),
    use_deim=False, compute_affine_convection=True,
    deim_energy_tol=1e-8, m_F_max=80, m_J_max=80,
    n_modes_F=None, n_modes_J=None,
    recompute_clustering=False, recompute_pod=False, recompute_deim=False,
)

SNAPSHOT_DIR   = "multi_param_multi_branch"
LIFTING_DIR    = "lifting"
MESH_FILE      = "mesh/mid_pinball.msh"
FOM_CHECKPOINT = "initial_states/velocity_checkpoint_symm"
REYNOLDS_INIT  = 100.0
AMPLITUDE_INIT = 0.0

AMP_PARAM      = 0.0
SWEEP_RESULTS  = "sweep_results.npz"
RE_C_FALLBACK  = 68.22

RE_ANCHOR_OFFSET = 0.5
JUMP_DELTAS      = (0.3, 0.5, 1.0, 2.0)
AMP_VALUES       = (1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
ASYMMETRY_TOL    = 1e-3
N_EIG            = 6


# =============================================================================
# HELPERS
# =============================================================================

def compute_rom_indicator(w_rom, callback_data, operators_k, M_uu_red_k,
                          n_eigenvalues=6, target=0.0):
    """μ_leftmost_real and ν_red from (J_red, M̃_red)."""
    _, jac_cb = make_callbacks(callback_data)
    with w_rom.dat.vec_ro as vec:
        jac_cb(vec)

    rom_mesh, W_rom = callback_data['rom_mesh'], callback_data['W_rom']
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

    N_u  = operators_k.Z_u.shape[1]
    nu_v = nu_red[:N_u]
    norm_sq = float(nu_v @ (M_uu_red_k @ nu_v))
    if norm_sq > 1e-14:
        nu_red /= np.sqrt(norm_sq)

    return {'mu': mu, 'nu_red': nu_red}


def detect_sign_change(re_values, mu_values):
    mu = np.asarray(mu_values, dtype=float)
    re = np.asarray(re_values, dtype=float)
    mask = ~np.isnan(mu)
    re, mu = re[mask], mu[mask]
    if len(mu) < 2:
        return None
    idx = np.where(np.diff(np.sign(mu)) != 0)[0]
    if len(idx) == 0:
        return None
    i = idx[0]
    t = -mu[i] / (mu[i+1] - mu[i])
    return float(re[i] + t * (re[i+1] - re[i]))


def load_re_c_rom():
    try:
        data = np.load(SWEEP_RESULTS)
        re_c = detect_sign_change(data['Re'], data['mu_rom'])
        if re_c is not None:
            print(f"  Re_c^ROM (from {SWEEP_RESULTS}): {re_c:.4f}")
            return re_c
    except FileNotFoundError:
        pass
    print(f"  Re_c^ROM (fallback):                  {RE_C_FALLBACK:.4f}")
    return RE_C_FALLBACK


def extract_coefficients(w_rom):
    """Extract converged α (velocity) and β (pressure) coefficient arrays."""
    u_sub, p_sub = w_rom.subfunctions
    with u_sub.dat.vec_ro as v:
        alpha = v.array.copy()
    with p_sub.dat.vec_ro as v:
        beta = v.array.copy()
    return alpha, beta


def solve_rom_in_cluster(
    Re, amp, k, alpha_init, beta_init,
    operators, deim_ops_dict, problem, use_deim,
):
    """
    Direct ROM solve in fixed cluster k with given initial coefficients.

    Bypasses cluster selection in solve_at_parameter_with_internals — used
    when the kick is intrinsically tied to cluster k's basis (ν_red is in
    that basis).
    """
    problem.reynolds.assign(Re)
    problem.amplitude.assign(amp)
    nu = 1.0 / Re

    deim_ops_k = deim_ops_dict[k]
    submesh_deim = None
    if use_deim and deim_ops_k is not None:
        from nsrom.local_rom_old import SubMeshDEIM
        submesh_deim = SubMeshDEIM(problem.velocity_space, deim_ops_k)

    return solve_rom_with_internals(
        operators=operators[k],
        nu=nu, amp=amp, problem=problem,
        deim_ops=deim_ops_k,
        u_initial_guess=alpha_init,
        p_initial_guess=beta_init,
        submesh_deim=submesh_deim,
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("ROM BRANCH JUMP — pitchfork validation (fixed-cluster, ROM-coeff space)")
    print("=" * 80)

    re_c_rom  = load_re_c_rom()
    re_anchor = re_c_rom - RE_ANCHOR_OFFSET
    print(f"  Re_anchor (symmetric solve): {re_anchor:.4f}  (Re_c - {RE_ANCHOR_OFFSET})")

    # ── Setup ────────────────────────────────────────────────────────────────
    layout = LocalROMLayout(
        base_dir="local_rom",
        n_clusters=cfg.n_clusters,
        inner_product_type=cfg.inner_product_type,
    )
    problem = setup_navier_stokes_problem(
        mesh_file=MESH_FILE,
        reynolds_init=REYNOLDS_INIT, amplitude_init=AMPLITUDE_INIT,
    )
    initial_solution = load_solution(FOM_CHECKPOINT, problem)

    snapshot_data = load_snapshot_dofs(SNAPSHOT_DIR)
    lifting       = load_lifting_functions(LIFTING_DIR, problem)
    velocity_snapshots = snapshot_data['velocity']
    pressure_snapshots = snapshot_data['pressure']
    parameters         = snapshot_data['parameters']

    S_homo, S_tilde, M_u, M_p, L, P = build_energy_snapshots(
        velocity_snapshots, parameters, lifting, problem,
        cfg.inner_product_type, homogenize=False,
    )
    clustering, _ = compute_or_load_clustering(
        S_tilde, S_homo, M_u, parameters, L, P,
        save_dir=layout.clustering_dir, n_clusters=cfg.n_clusters,
        method="pod_whitened", rank=10,
        recompute=cfg.recompute_clustering,
    )
    bases, operators, deim_ops_dict = build_local_roms(
        clustering=clustering,
        velocity_snapshots=velocity_snapshots,
        pressure_snapshots=pressure_snapshots,
        deim_snapshot_data=None, deim_snapshot_dir=SNAPSHOT_DIR,
        lifting=lifting, problem=problem, cfg=cfg, layout=layout,
    )

    M_L2 = assemble_inner_product_matrix(problem.velocity_space, 'L2')
    M_uu_red = {}
    for k in range(clustering.n_clusters):
        Z_u_k = operators[k].Z_u
        M_red = Z_u_k.T @ (M_L2 @ Z_u_k)
        M_uu_red[k] = 0.5 * (M_red + M_red.T)

    # ── Symmetric ROM solve at Re_anchor (full pipeline; cluster freely chosen) ─
    print(f"\n--- Symmetric ROM solve at Re_anchor = {re_anchor:.4f} ---")
    sym_res = solve_at_parameter_with_internals(
        Re=re_anchor, amp=AMP_PARAM,
        clustering=clustering, operators=operators,
        deim_ops_dict=deim_ops_dict, problem=problem,
        initial_solution=initial_solution,
        M_u=M_u, M_p=M_p, use_deim=cfg.use_deim,
    )
    k_anchor = sym_res.cluster
    alpha_sym, beta_sym = extract_coefficients(sym_res.w_rom)
    sym_full = reconstruct_solution(sym_res, operators[k_anchor], problem, amp=AMP_PARAM)
    sym_scores = probe_asymmetry_score(sym_full)
    print(f"  cluster {k_anchor}, {sym_res.iterations} Newton iterations")
    print(f"  asymmetry: ux={sym_scores['ux_antisym_score']:+.3e}, "
          f"uy={sym_scores['uy_antisym_score']:+.3e}")
    print(f"  ||α_sym||={np.linalg.norm(alpha_sym):.3e}, "
          f"||β_sym||={np.linalg.norm(beta_sym):.3e}")

    # ── Reduced eigenproblem → ν_red (in cluster k_anchor's basis) ──────────
    print(f"\n--- Reduced eigenproblem at Re_anchor ---")
    indicator = compute_rom_indicator(
        sym_res.w_rom, sym_res.callback_data,
        operators[k_anchor], M_uu_red[k_anchor],
        n_eigenvalues=N_EIG, target=0.0,
    )
    if indicator is None:
        print("  Indicator failed — aborting.")
        return
    nu_red = indicator['nu_red']
    N_u = operators[k_anchor].Z_u.shape[1]
    print(f"  μ_leftmost_real = {indicator['mu']:+.3e}  "
          f"(small positive expected, since Re_anchor < Re_c)")

    # Sanity: lifted L²-norm of ν_red[:N_u] should be ≈ 1
    u_kick_full = operators[k_anchor].Z_u @ nu_red[:N_u]
    kick_norm = float(np.sqrt(u_kick_full @ (M_L2 @ u_kick_full)))
    print(f"  ||Z_u @ ν_red[:N_u]||_L²  = {kick_norm:.6f}  (expect ≈ 1.0)")

    # ── Branch jump grid (fixed cluster k_anchor) ────────────────────────────
    print("\n" + "=" * 80)
    print(f"BRANCH JUMP GRID — fixed cluster k = {k_anchor}")
    print("=" * 80)

    found = []
    for delta in JUMP_DELTAS:
        Re_jump = re_c_rom + delta
        print(f"\n--- Re_jump = {Re_jump:.4f}  (Re_c + {delta:+.3f}) ---")

        for amp_kick in AMP_VALUES:
            for sign in (+1, -1):
                alpha_init = alpha_sym + sign * amp_kick * nu_red[:N_u]
                beta_init  = beta_sym.copy()

                try:
                    solution, w_rom, _ = solve_rom_in_cluster(
                        Re=Re_jump, amp=AMP_PARAM, k=k_anchor,
                        alpha_init=alpha_init, beta_init=beta_init,
                        operators=operators, deim_ops_dict=deim_ops_dict,
                        problem=problem, use_deim=cfg.use_deim,
                    )
                except Exception as e:
                    print(f"  sign={sign:+d}  amp={amp_kick:.0e}  Newton FAILED: "
                          f"{type(e).__name__}")
                    continue

                reconstructed = reconstruct_solution(
                    solution, operators[k_anchor], problem, amp=AMP_PARAM,
                )
                scores = probe_asymmetry_score(reconstructed)
                forces = compute_forces(reconstructed, problem)
                iters = getattr(solution, 'iterations', '?')

                broken = (scores['ux_symmetry_error'] > ASYMMETRY_TOL or
                          scores['uy_symmetry_error'] > ASYMMETRY_TOL)
                tag = " ← ASYMM" if broken else ""

                print(f"  sign={sign:+d}  amp={amp_kick:.0e}  "
                      f"it={iters}  "
                      f"ux={scores['ux_antisym_score']:+.3f}  "
                      f"uy={scores['uy_antisym_score']:+.3f}  "
                      f"C_L={forces.lift:+.3e}{tag}")

                if broken:
                    # Capture the converged α, β for the seed (used by script 3)
                    alpha_conv, beta_conv = extract_coefficients(w_rom)
                    found.append({
                        'Re': Re_jump, 'delta': delta, 'amp': amp_kick, 'sign': sign,
                        'alpha': alpha_conv, 'beta': beta_conv,
                        'C_L': float(forces.lift), 'C_D': float(forces.drag),
                        'scores': scores,
                    })

        signs_at_delta = {f['sign'] for f in found if f['delta'] == delta}
        if signs_at_delta == {+1, -1}:
            print(f"\n  ✓ Both ± branches reached at delta={delta}.")
            break

    # ── Validate ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("VALIDATION")
    print("=" * 80)

    if not found:
        print("  NO ASYMMETRIC BRANCH FOUND across the grid.")
        print("  → try larger delta or amp; or verify the indicator.")
        return

    delta_min = min(f['delta'] for f in found)
    candidates = [f for f in found if f['delta'] == delta_min]
    pos = next((f for f in candidates if f['sign'] == +1), None)
    neg = next((f for f in candidates if f['sign'] == -1), None)

    if pos is None or neg is None:
        signs_found = sorted({f['sign'] for f in found})
        print(f"  Only sign(s) {signs_found} reached an asymmetric branch.")
        return

    print(f"  Re_jump = {pos['Re']:.4f}  (Re_c + {delta_min:+.3f})\n")
    print(f"  branch +  amp={pos['amp']:.0e}  C_L = {pos['C_L']:+.4e}")
    print(f"  branch -  amp={neg['amp']:.0e}  C_L = {neg['C_L']:+.4e}")

    cl_max = max(abs(pos['C_L']), abs(neg['C_L']))
    cl_mirror_err = abs(pos['C_L'] + neg['C_L']) / cl_max
    print(f"\n  mirror error in C_L:   |C_L^+ + C_L^-| / max  = {cl_mirror_err:.2%}")

    err_p = max(pos['scores']['ux_symmetry_error'], pos['scores']['uy_symmetry_error'])
    err_n = max(neg['scores']['ux_symmetry_error'], neg['scores']['uy_symmetry_error'])
    asym_match_err = abs(err_p - err_n) / max(err_p, err_n)
    print(f"  match in |asymmetry|:  {asym_match_err:.2%}")

    if cl_mirror_err < 0.05 and asym_match_err < 0.05:
        print("\n  ✓ ROM pitchfork branches validated (mirror symmetric).")
    else:
        print("\n  ⚠ mirror symmetry weak — investigate.")

    # ── Save seeds for script (3) ────────────────────────────────────────────
    np.savez(
        "rom_branch_seeds.npz",
        re_c_rom=re_c_rom, re_anchor=re_anchor,
        Re_seed=pos['Re'], delta_seed=delta_min,
        cluster_anchor=k_anchor,
        # Converged asymmetric ROM coefficients on each branch — direct seeds
        alpha_pos=pos['alpha'], beta_pos=pos['beta'],
        alpha_neg=neg['alpha'], beta_neg=neg['beta'],
        amp_pos=pos['amp'],     amp_neg=neg['amp'],
        cl_pos=pos['C_L'],      cl_neg=neg['C_L'],
    )
    print("\n  Saved seeds → rom_branch_seeds.npz")


if __name__ == '__main__':
    main()
