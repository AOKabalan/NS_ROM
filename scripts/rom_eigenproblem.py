"""
Bifurcation sweep: track μ_leftmost_real(Re) at ROM and FOM levels across a
Reynolds range bracketing the pitchfork. Detects the sign change (ROM Re_c)
and verifies we stay on the symmetric branch throughout.

One-line-per-Re report; saves sweep_results.npz for plotting.
"""

import numpy as np
from firedrake import (
    TrialFunctions, TestFunctions, inner, dot, dx, assemble, Constant,
)

from nsrom.rom import (
    assemble_inner_product_matrix, reconstruct_solution, make_callbacks,
)
from nsrom.navier_stokes import (
    setup_navier_stokes_problem, load_solution, solve_steady_navier_stokes,
)
from nsrom.snapshot_collection import load_snapshot_dofs
from nsrom.lifting_functions import load_lifting_functions
from nsrom.cluster_building import build_energy_snapshots, compute_or_load_clustering
from nsrom.local_rom_old import build_local_roms, solve_at_parameter_with_internals
from nsrom.layout import LocalROMLayout
from nsrom.config import LocalROMConfig
from nsrom.bifurcation.eigen_solver import solve_leftmost_real_eigenpairs
from nsrom.bifurcation.branch_jump import probe_asymmetry_score
from nsrom.bifurcation.jacobian import build_state_jacobian_pencil


# =============================================================================
# CONFIGURATION
# =============================================================================

cfg = LocalROMConfig(
    n_clusters=5,
    inner_product_type="H1",
    pod_energy_tol=1e-8,
    n_velocity_max=50,
    n_pressure_max=50,
    n_supremizer_max=50,
    boundary_markers=(1, 3, 4, 5, 6),
    use_deim=False,
    compute_affine_convection=True,
    deim_energy_tol=1e-8,
    m_F_max=80,
    m_J_max=80,
    n_modes_F=None,
    n_modes_J=None,
    recompute_clustering=False,
    recompute_pod=False,
    recompute_deim=False,
)

SNAPSHOT_DIR   = "multi_param_multi_branch"
LIFTING_DIR    = "lifting"
MESH_FILE      = "mesh/mid_pinball.msh"
FOM_CHECKPOINT = "initial_states/velocity_checkpoint_symm"

REYNOLDS_INIT  = 100.0
AMPLITUDE_INIT = 0.0

# --- Sweep ---
AMP_SWEEP  = 0.0
RE_SWEEP   = np.linspace(65.0, 71.0, 13)   # brackets Re_c ≈ 68.2237
N_EIG      = 6
TARGET_EIG = 0.0
RESULTS_FILE = "sweep_results.npz"


# =============================================================================
# INDICATORS
# =============================================================================

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


def compute_fom_indicator(problem, fom_solution, n_eigenvalues=6, target=0.0):
    """μ_leftmost_real and ν from the FOM (J, M_L²) pencil."""
    J_fom, M_fom = build_state_jacobian_pencil(problem, fom_solution)

    eigvals, vrs, _, nconv = solve_leftmost_real_eigenpairs(
        J_fom, M_fom, n_eigenvalues=n_eigenvalues, target=target,
    )
    if nconv == 0:
        return None

    imag_tol  = 1e-6 * max(abs(e) for e in eigvals)
    real_mask = np.abs(eigvals.imag) < imag_tol
    real_idx  = np.where(real_mask)[0]
    if len(real_idx) == 0:
        return None

    leftmost = int(real_idx[np.argmin(eigvals[real_mask].real)])
    return {
        'mu':      float(eigvals[leftmost].real),
        'nu_full': vrs[leftmost].array_r.copy(),
        'eigvals': eigvals,
    }


def detect_sign_change(re_values, mu_values):
    """Linear interpolation of the first zero crossing of μ(Re)."""
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
    return float(re[i] + t * (re[i+1] - re[i])),i


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print(f"BIFURCATION SWEEP — K={cfg.n_clusters}, {cfg.inner_product_type}")
    print(f"  Re: {RE_SWEEP[0]:.2f} → {RE_SWEEP[-1]:.2f}  ({len(RE_SWEEP)} points)")
    print(f"  amp: {AMP_SWEEP:.2f}")
    print("=" * 80)

    # --- Setup ---
    layout = LocalROMLayout(
        base_dir="local_rom",
        n_clusters=cfg.n_clusters,
        inner_product_type=cfg.inner_product_type,
    )

    problem = setup_navier_stokes_problem(
        mesh_file=MESH_FILE,
        reynolds_init=REYNOLDS_INIT,
        amplitude_init=AMPLITUDE_INIT,
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
        save_dir=layout.clustering_dir,
        n_clusters=cfg.n_clusters,
        method="pod_whitened",
        rank=10,
        recompute=cfg.recompute_clustering,
    )

    bases, operators, deim_ops_dict = build_local_roms(
        clustering=clustering,
        velocity_snapshots=velocity_snapshots,
        pressure_snapshots=pressure_snapshots,
        deim_snapshot_data=None,
        deim_snapshot_dir=SNAPSHOT_DIR,
        lifting=lifting,
        problem=problem,
        cfg=cfg,
        layout=layout,
    )

    # Reduced L² mass per cluster (for ROM eigenproblem and ν normalization)
    M_L2 = assemble_inner_product_matrix(problem.velocity_space, 'L2')
    M_uu_red = {}
    for k in range(clustering.n_clusters):
        Z_u_k = operators[k].Z_u
        M_red = Z_u_k.T @ (M_L2 @ Z_u_k)
        M_uu_red[k] = 0.5 * (M_red + M_red.T)

    # =========================================================================
    # SWEEP
    # =========================================================================
    print("\n" + "=" * 80)
    print("SWEEP")
    print("=" * 80)
    cols = (f"  {'Re':>6} {'clus':>4} {'it':>3} "
            f"{'μ_ROM':>13} {'μ_FOM':>13} {'rel.err':>8} "
            f"{'ax':>10} {'ay':>10} {'align':>7}")
    print(cols)
    print("  " + "-" * (len(cols) - 2))

    results = []
    fom_state = initial_solution.solution   # warm-start chain for FOM

    for Re in RE_SWEEP:
        # --- ROM solve ---
        rom_res = solve_at_parameter_with_internals(
            Re=Re, amp=AMP_SWEEP,
            clustering=clustering, operators=operators,
            deim_ops_dict=deim_ops_dict, problem=problem,
            initial_solution=initial_solution,
            M_u=M_u, M_p=M_p, use_deim=cfg.use_deim,
        )
        k_active = rom_res.cluster

        # --- ROM indicator ---
        rom_ind = compute_rom_indicator(
            rom_res.w_rom, rom_res.callback_data,
            operators[k_active], M_uu_red[k_active],
            n_eigenvalues=N_EIG, target=TARGET_EIG,
        )
        mu_rom = rom_ind['mu'] if rom_ind else float('nan')

        # --- Branch check ---
        u_full = reconstruct_solution(rom_res, operators[k_active], problem, amp=AMP_SWEEP)
        asym   = probe_asymmetry_score(u_full)
        ax, ay = asym['ux_antisym_score'], asym['uy_antisym_score']

        # --- FOM solve + indicator (warm-started from previous Re) ---
        problem.reynolds.assign(Re)
        problem.amplitude.assign(AMP_SWEEP)
        fom_solution = solve_steady_navier_stokes(problem, fom_state)
        fom_state    = fom_solution.solution

        fom_ind = compute_fom_indicator(
            problem, fom_solution, n_eigenvalues=N_EIG, target=TARGET_EIG,
        )
        mu_fom = fom_ind['mu'] if fom_ind else float('nan')

        # --- Eigenvector alignment ---
        align = float('nan')
        if rom_ind is not None and fom_ind is not None:
            N_u        = operators[k_active].Z_u.shape[1]
            nu_lift    = operators[k_active].Z_u @ rom_ind['nu_red'][:N_u]
            n_vel      = problem.velocity_space.dim()
            nu_fom_vel = fom_ind['nu_full'][:n_vel]
            n1 = float(np.sqrt(nu_lift    @ (M_L2 @ nu_lift)))
            n2 = float(np.sqrt(nu_fom_vel @ (M_L2 @ nu_fom_vel)))
            if n1 > 0 and n2 > 0:
                align = abs((nu_lift / n1) @ (M_L2 @ (nu_fom_vel / n2)))

        rel_err = (abs(mu_rom - mu_fom) / abs(mu_fom)
                   if not (np.isnan(mu_rom) or np.isnan(mu_fom)) and abs(mu_fom) > 0
                   else float('nan'))

        print(f"  {Re:>6.2f} {k_active:>4d} {rom_res.iterations:>3d} "
              f"{mu_rom:>+13.4e} {mu_fom:>+13.4e} {rel_err:>7.2%} "
              f"{ax:>+10.2e} {ay:>+10.2e} {align:>7.4f}")

        results.append({
            'Re': float(Re), 'cluster': k_active, 'iters': rom_res.iterations,
            'mu_rom': mu_rom, 'mu_fom': mu_fom,
            'ax': ax, 'ay': ay, 'align': align,
        })

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    re_arr = np.array([r['Re']     for r in results])
    mu_rom = np.array([r['mu_rom'] for r in results])
    mu_fom = np.array([r['mu_fom'] for r in results])

    re_c_rom, i_rom = detect_sign_change(re_arr, mu_rom)
    re_c_fom, i_fom = detect_sign_change(re_arr, mu_fom)

    if re_c_rom is not None:
        print(f"  Re_c (ROM, linear interp): {re_c_rom:.4f}")
    else:
        print("  Re_c (ROM): no sign change in sweep range")
    if re_c_fom is not None:
        print(f"  Re_c (FOM, linear interp): {re_c_fom:.4f}")
    else:
        print("  Re_c (FOM): no sign change in sweep range")
    if re_c_rom is not None and re_c_fom is not None:
        print(f"  Δ Re_c (ROM − FOM):        {re_c_rom - re_c_fom:+.4f}")
        print(f"  reference (FOM Step 6):    Re_c ≈ 68.2237")

    asym_max = max(max(abs(r['ax']), abs(r['ay'])) for r in results)
    branch_msg = "on symmetric branch" if asym_max < 1e-2 else "WARNING: branch jumped"
    print(f"\n  max |asymmetry score|:     {asym_max:.2e}  ({branch_msg})")

    align_arr = np.array([r['align'] for r in results])
    align_clean = align_arr[~np.isnan(align_arr)]
    if len(align_clean) > 0:
        print(f"  min eigenvector alignment: {align_clean.min():.4f}")

    # Save for plotting
    np.savez(
        RESULTS_FILE,
        Re=re_arr,
        mu_rom=mu_rom,
        mu_fom=mu_fom,
        cluster=np.array([r['cluster'] for r in results]),
        ax=np.array([r['ax']    for r in results]),
        ay=np.array([r['ay']    for r in results]),
        align=align_arr,
    )
    print(f"\n  Saved: {RESULTS_FILE}")

    #

    # print("\n--- Branch jump from pitchfork ---")

    # jump_results = branch_jump_pitchfork(
    #     problem,
    #     sol_c,
    #     eigvec_c,
    #     Re_c,
    #     eps_values=(0.5,),
    #     amp_values=(3e-2, 1e-1),
    #     perturb_pressure=False,
    #     asymmetry_tol=1e-3,
    # )

    # print(f"\n  Successful jumps: {len(jump_results)}")

    # for j, r in enumerate(jump_results):
    #     print(
    #         f"  jump {j}: "
    #         f"sign={r['sign']:+d}, "
    #         f"Re={r['Re']:.6f}, "
    #         f"eps={r['epsilon']:.3e}, "
    #         f"amp={r['amplitude']:.3e}, "
    #         f"ux_score={r['scores']['ux_antisym_score']:+.3f}, "
    #         f"uy_score={r['scores']['uy_antisym_score']:+.3f}"
    #     )


if __name__ == '__main__':
    main()
