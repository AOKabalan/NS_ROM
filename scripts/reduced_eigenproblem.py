
import numpy as np
from nsrom.rom import assemble_inner_product_matrix, reconstruct_solution
from nsrom.navier_stokes import setup_navier_stokes_problem, load_solution
from nsrom.snapshot_collection import load_snapshot_dofs
from nsrom.lifting_functions import load_lifting_functions
from nsrom.cluster_building import build_energy_snapshots, compute_or_load_clustering
from nsrom.local_rom_old import (
    build_local_roms,
    solve_at_parameter_with_internals,
    compare_rom_vs_fom,
    precompute_fast_selection,
    precompute_change_of_basis,
)
from nsrom.layout import LocalROMLayout
from nsrom.config import LocalROMConfig, RunManifest, save_run

from sweep import build_snake_path, run_sweep, save_sweep_results
from plot import plot_lift_3d
from firedrake import *
from scipy.sparse import csr_matrix

from nsrom.rom import make_callbacks
from nsrom.bifurcation.eigen_solver import solve_leftmost_real_eigenpairs
from nsrom.bifurcation.branch_jump import probe_asymmetry_score

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

# --- Paths ---
SNAPSHOT_DIR      = "multi_param_multi_branch"
DEIM_SNAPSHOT_DIR = "multi_param_multi_branch"
LIFTING_DIR       = "lifting"
MESH_FILE         = "mesh/mid_pinball.msh"
FOM_CHECKPOINT    = "initial_states/velocity_checkpoint_symm"

REYNOLDS_INIT  = 100.0
AMPLITUDE_INIT = 0.0

# --- Test parameters ---
TEST_RE  = 93.0
TEST_AMP = -0.72

# --- Sweep ---
RE_VALUES  = np.arange(30.0, 90.0, 10.0)
AMP_VALUES = np.arange(-1.0, 1.0, 0.1)
FOM_EVERY  = 0       # 0 = skip FOM, N = compare every Nth point
RUN_SWEEP  = False


# =============================================================================
# MANIFEST and LAYOUT
# =============================================================================
from typing import Optional


def compute_bifurcation_indicator(
    w_rom, callback_data, operators_k, M_uu_red_k,
    n_eigenvalues=6, target=0.0,
):
    """
    Returns a dict with all the pieces needed for reporting.
    """
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
        inner(dot(A_eff_const, du), v) * dx
        + inner(dot(conv_J_const, du), v) * dx
        + inner(dot(B_const.T, dp), v) * dx
        + inner(dot(B_const, du), q) * dx
    )

    M_uu_const = Constant(M_uu_red_k, rom_mesh)
    M_form = inner(dot(M_uu_const, du), v) * dx

    J_assembled = assemble(J_form, mat_type='aij')
    M_assembled = assemble(M_form, mat_type='aij')

    eigvals, vrs, vis, nconv = solve_leftmost_real_eigenpairs(
        J_assembled.M.handle,
        M_assembled.M.handle,
        n_eigenvalues=n_eigenvalues,
        target=target,
    )

    if nconv == 0:
        return {
            'success': False, 'reason': 'no eigenvalues converged',
            'eigvals_all': eigvals,
        }

    # Filter to real eigenvalues
    imag_tol  = 1e-6 * max(abs(e) for e in eigvals)
    real_mask = np.abs(eigvals.imag) < imag_tol
    real_idx  = np.where(real_mask)[0]

    if len(real_idx) == 0:
        return {
            'success': False, 'reason': 'no real eigenvalues',
            'eigvals_all': eigvals, 'real_mask': real_mask,
        }

    # Leftmost real
    leftmost_in_real = int(np.argmin(eigvals[real_mask].real))
    i_leftmost       = int(real_idx[leftmost_in_real])

    mu_leftmost_real = float(eigvals[i_leftmost].real)
    mu_imag          = float(eigvals[i_leftmost].imag)
    nu_red           = vrs[i_leftmost].array_r.copy()

    # M̃_red-normalize on velocity block
    N_u     = operators_k.Z_u.shape[1]
    nu_v    = nu_red[:N_u]
    norm_sq = float(nu_v @ (M_uu_red_k @ nu_v))
    if norm_sq > 1e-14:
        nu_red /= np.sqrt(norm_sq)

    # Gap to next real
    real_sorted = np.sort(eigvals[real_mask].real)
    gap = float(real_sorted[1] - real_sorted[0]) if len(real_sorted) >= 2 else None

    return {
        'success'         : True,
        'mu_leftmost_real': mu_leftmost_real,
        'mu_imag'         : mu_imag,
        'gap_to_next_real': gap,
        'nu_red'          : nu_red,
        'eigvals_all'     : eigvals,
        'real_mask'       : real_mask,
        'i_leftmost'      : i_leftmost,
        'imag_tol'        : imag_tol,
    }
def print_indicator_report(
    Re, amp, k_active, iterations,
    score_x, score_y,
    indicator,
    M_uu_red_k, operators_k,
):
    """One-block report for a single indicator evaluation."""
    print()
    print("=" * 64)
    print(f"  INDICATOR AT (Re={Re:.2f}, amp={amp:.2f}) — cluster {k_active}")
    print("=" * 64)

    # ---- Convergence and branch ----
    branch_score = max(abs(score_x), abs(score_y))
    branch = "symmetric" if branch_score < 1e-2 else "asymmetric"
    print(f"  Solver:")
    print(f"    Newton iterations    : {iterations}")
    print(f"    branch               : {branch}")
    print(f"    antisymmetry scores  : x = {score_x:+.3e}, y = {score_y:+.3e}")

    if not indicator['success']:
        print(f"\n  Indicator: FAILED — {indicator['reason']}")
        print("=" * 64)
        return

    # ---- Headline indicator ----
    mu       = indicator['mu_leftmost_real']
    mu_imag  = indicator['mu_imag']
    gap      = indicator['gap_to_next_real']
    eigvals  = indicator['eigvals_all']
    real_msk = indicator['real_mask']
    i_left   = indicator['i_leftmost']

    sign_str = "stable (μ > 0)" if mu > 0 else "unstable (μ < 0)"
    print(f"\n  Pitchfork indicator:")
    print(f"    μ_leftmost_real      : {mu:+.6e}     ← {sign_str}")
    if gap is not None:
        print(f"    gap to next real     : {gap:+.3e}     "
              f"({'well isolated' if abs(gap) > 10*abs(mu) else 'tight — verify'})")
    print(f"    |Im(μ_leftmost)|     : {abs(mu_imag):.2e}     (numerical zero check)")

    # ---- Full spectrum, with markers ----
    imag_tol = indicator['imag_tol']
    print(f"\n  Full spectrum ({len(eigvals)} converged, ↑ Re(μ)):")
    for i, eig in enumerate(eigvals):
        is_real = abs(eig.imag) < imag_tol
        kind = "  R " if is_real else " ±I "
        marker = "  ← leftmost real" if i == i_left else ""
        print(f"    μ[{i}] = {eig.real:+.4e}  {kind}  "
              f"{eig.imag:+.3e}j{marker}")

    # ---- Eigenvector ----
    nu_red  = indicator['nu_red']
    N_u     = operators_k.Z_u.shape[1]
    nu_v    = nu_red[:N_u]
    norm_v  = float(np.sqrt(nu_v @ (M_uu_red_k @ nu_v)))
    print(f"\n  Eigenvector ν_red:")
    print(f"    shape                : {nu_red.shape}")
    print(f"    ||ν_red||_M̃_red      : {norm_v:.6f}     (should be 1.000000)")
    print("=" * 64)

manifest = RunManifest(
    snapshot_dir=SNAPSHOT_DIR,
    deim_snapshot_dir=DEIM_SNAPSHOT_DIR if cfg.use_deim else None,
    lifting_dir=LIFTING_DIR,
    mesh_file=MESH_FILE,
    fom_checkpoint=FOM_CHECKPOINT,
    reynolds_init=REYNOLDS_INIT,
    amplitude_init=AMPLITUDE_INIT,
    test_re=TEST_RE,
    test_amp=TEST_AMP,
    run_sweep=RUN_SWEEP,
    fom_every=FOM_EVERY,
    re_values=RE_VALUES.tolist(),
    amp_values=AMP_VALUES.tolist(),
)

layout = LocalROMLayout(
    base_dir="local_rom",
    n_clusters=cfg.n_clusters,
    inner_product_type=cfg.inner_product_type,
)


# =============================================================================
# MAIN
# =============================================================================

def main():
    save_run(cfg, manifest, layout.run_json())

    mode_str   = "DEIM" if cfg.use_deim else "Exact"
    affine_str = "Affine" if cfg.compute_affine_convection else "Full"

    print("=" * 80)
    print(f"LOCAL ROM — {mode_str} solver, {affine_str} convection, K={cfg.n_clusters}")
    print("=" * 80)
    print(f"  USE_DEIM:    {cfg.use_deim}")
    print(f"  AFFINE:      {cfg.compute_affine_convection}")
    print(f"  ROM modes:   tol={cfg.pod_energy_tol}, max={cfg.n_velocity_max}")
    print(f"  Inner prod:  {cfg.inner_product_type}")
    print(f"  Test params: Re={TEST_RE}, amp={TEST_AMP}")
    if cfg.use_deim:
        print(f"  DEIM:        tol={cfg.deim_energy_tol}, max_F={cfg.m_F_max}, max_J={cfg.m_J_max}")
    print()

    # --- Setup problem ---
    print("--- Setup problem ---")
    problem = setup_navier_stokes_problem(
        mesh_file=MESH_FILE,
        reynolds_init=REYNOLDS_INIT,
        amplitude_init=AMPLITUDE_INIT,
    )
    initial_solution = load_solution(FOM_CHECKPOINT, problem)

    # --- Load snapshots and lifting ---
    print("\n--- Load snapshots and lifting ---")
    snapshot_data = load_snapshot_dofs(SNAPSHOT_DIR)
    lifting       = load_lifting_functions(LIFTING_DIR, problem)

    velocity_snapshots = snapshot_data['velocity']
    pressure_snapshots = snapshot_data['pressure']
    parameters         = snapshot_data['parameters']
    lift_coefficients  = snapshot_data['lift_coefficients']

    print(f"  Snapshots:  {velocity_snapshots.shape[0]}")
    print(f"  Parameters: {len(parameters)} entries")

    # --- Energy-norm transform ---
    print("\n--- Energy-norm transform ---")

    S_homo, S_tilde, M_u, M_p, L, P = build_energy_snapshots(
        velocity_snapshots, parameters, lifting, problem, cfg.inner_product_type,
        homogenize = False
    )
    print(f"  S_tilde: {S_tilde.shape}")
    method = "pod_whitened" #"pod_whitened" or "cholesky"

    # --- Clustering ---
    print("\n--- Clustering ---")
    clustering, reason = compute_or_load_clustering(
        S_tilde, S_homo, M_u,
        parameters, L, P,
        save_dir=layout.clustering_dir,
        n_clusters=cfg.n_clusters,
        method=method,   # or "pod_whitened" or "cholesky" with rank=...
        rank=10,           # or e.g. 15
        recompute=cfg.recompute_clustering,
    )

    print(f"  {reason}")
    # clustering.summary()
    # plot_lift_3d(
    #     parameters,
    #     lift_coefficients,
    #     clustering.labels,
    #     title=f"Clusters (method={method}, K={clustering.n_clusters})",
    # )
    # --- Local ROMs ---
    print("\n--- Local ROMs ---")
    deim_snapshot_data = None
    if cfg.use_deim:
        deim_snapshot_data = load_snapshot_dofs(DEIM_SNAPSHOT_DIR)
        print(f"  DEIM snapshots: {deim_snapshot_data['velocity'].shape[0]}")

    bases, operators, deim_ops_dict = build_local_roms(
        clustering=clustering,
        velocity_snapshots=velocity_snapshots,
        pressure_snapshots=pressure_snapshots,
        deim_snapshot_data=deim_snapshot_data,
        deim_snapshot_dir=DEIM_SNAPSHOT_DIR,
        lifting=lifting,
        problem=problem,
        cfg=cfg,
        layout=layout,
    )

    for k in range(clustering.n_clusters):
        basis_k = bases[k]
        print(f"  Cluster {k}: {basis_k.n_velocity_modes} vel + "
              f"{basis_k.n_supremizer_modes} sup + "
              f"{basis_k.n_pressure_modes} pres")
        if cfg.use_deim:
            print(f"             DEIM m_F={deim_ops_dict[k].m_F}, m_J={deim_ops_dict[k].m_J}")

        operators_k = operators[k]
        N_u = basis_k.n_enriched_velocity_modes
        Z_u_k = operators_k.Z_u

        G = Z_u_k.T @ (M_u @ Z_u_k)
        err_frob    = np.linalg.norm(G - np.eye(N_u))
        err_offdiag = np.linalg.norm(G - np.diag(np.diag(G)))
        err_diag    = np.linalg.norm(np.diag(G) - 1.0)

        print(f"             Gram check (N_u={N_u}):")
        print(f"               ||G - I||_F      = {err_frob:.3e}")
        print(f"               ||G - diag(G)||_F = {err_offdiag:.3e}   (off-diagonal: orthogonality)")
        print(f"               ||diag(G) - 1||   = {err_diag:.3e}     (diagonal: normalization)")


    # ---- Offline: L² velocity mass + per-cluster reduced mass ----
    print("\n--- Precompute reduced L² mass per cluster ---")
    M_L2 = assemble_inner_product_matrix(problem.velocity_space, 'L2')
    print(f"  M_L2: {M_L2.shape}, nnz={M_L2.nnz}")

    M_uu_red = {}   # cluster index → (N_u, N_u) dense numpy array
    for k in range(clustering.n_clusters):
        Z_u_k = operators[k].Z_u
        N_u = Z_u_k.shape[1]

        M_red_k = Z_u_k.T @ (M_L2 @ Z_u_k)

        # Symmetrize (kill ~1e-15 asymmetry from accumulated roundoff)
        M_red_k = 0.5 * (M_red_k + M_red_k.T)

        # SPD diagnostics
        eigvals = np.linalg.eigvalsh(M_red_k)   # symmetric → eigvalsh
        lam_min = eigvals.min()
        lam_max = eigvals.max()
        cond = lam_max / lam_min if lam_min > 0 else np.inf

        M_uu_red[k] = M_red_k

        print(f"  Cluster {k}: M_uu_red {M_red_k.shape}")
        print(f"               λ_min = {lam_min:.3e}")
        print(f"               λ_max = {lam_max:.3e}")
        print(f"               cond  = {cond:.3e}")
        print(f"               SPD   = {'yes' if lam_min > 0 else 'NO'}")

    # # =========================================================================
    # # BIFURCATION INDICATOR — sanity check at one point
    # # =========================================================================
    # print("\n" + "=" * 80)
    # print("BIFURCATION INDICATOR — sanity check")
    # print("=" * 80)

    # from nsrom.local_rom import select_cluster_closest, solve_at_parameter
    # from nsrom.rom import project_to_rom_coefficients, solve_rom_with_internals

    TEST_RE_BIF  = 68.0
    TEST_AMP_BIF = 0.0
    # print(f"\n  Test point: Re = {TEST_RE_BIF}, amp = {TEST_AMP_BIF}")
    # print(f"  (just below known Re_c ≈ 68.22 for symmetric → asymmetric pitchfork)")
    # w_rom = result.w_rom
    # callback_data = result.callback_data

    # assert result.converged, "ROM did not converge — investigate before indicator"
    # k_active = result.cluster
    # # =========================================================================
    # # BIFURCATION INDICATOR — sanity check at one point
    # # =========================================================================
    # print("\n" + "=" * 80)
    # print("BIFURCATION INDICATOR — sanity check")
    # print("=" * 80)

    # Branch identification
    # u_full = reconstruct_solution(
    #     result, operators[result.cluster], problem, amp=TEST_AMP_BIF,
    # )

    # symm_results = probe_asymmetry_score(u_full)

    # Indicator
    result = solve_at_parameter_with_internals(
        Re=TEST_RE_BIF,
        amp=TEST_AMP_BIF,
        clustering=clustering,
        operators=operators,
        deim_ops_dict=deim_ops_dict,
        problem=problem,
        initial_solution=initial_solution,
        M_u=M_u,
        M_p=M_p,
        use_deim=cfg.use_deim,
    )


    u_full = reconstruct_solution(
        result, operators[result.cluster], problem, amp=TEST_AMP_BIF,
    )

    indicator = compute_bifurcation_indicator(
        result.w_rom, result.callback_data,
        operators[result.cluster], M_uu_red[result.cluster],
        n_eigenvalues=6, target=0.0,
    )

    # Single-block report
    # print_indicator_report(
    #     Re=TEST_RE_BIF, amp=TEST_AMP_BIF, k_active=result.cluster,
    #     iterations=result.iterations,
    #     score_x=symm_results['ux_antisym_score'], score_y= symm_results['uy_antisym_score'],
    #     indicator=indicator,
    #     M_uu_red_k=M_uu_red[result.cluster],
    #     operators_k=operators[result.cluster],
    # )
    # eig_track =[]
    # RE_SWEEP = np.linspace(50.8, 51.2, 11)
    # for Re_bif in RE_SWEEP:
    #     result = solve_at_parameter_with_internals(
    #         Re=Re_bif,
    #         amp=TEST_AMP_BIF,
    #         clustering=clustering,
    #         operators=operators,
    #         deim_ops_dict=deim_ops_dict,
    #         problem=problem,
    #         initial_solution=initial_solution,
    #         M_u=M_u,
    #         M_p=M_p,
    #         use_deim=cfg.use_deim,
    #     )


    #     u_full = reconstruct_solution(
    #         result, operators[result.cluster], problem, amp=TEST_AMP_BIF,
    #     )

    #     indicator = compute_bifurcation_indicator(
    #         result.w_rom, result.callback_data,
    #         operators[result.cluster], M_uu_red[result.cluster],
    #         n_eigenvalues=6, target=0.0,
    #     )
    #     eig_track.append(indicator['mu_leftmost_real'])
    # print(f"  Re={RE_SWEEP}   "
    #     f"μ={eig_track}  ")
    # ── FOM cross-check at Re=68 ──────────────────────────────────────────────────
    from nsrom.bifurcation.jacobian import build_state_jacobian_pencil
    from nsrom.navier_stokes import solve_steady_navier_stokes

    RE_CHECK, AMP_CHECK = 68.0, 0.0

    problem.reynolds.assign(RE_CHECK)
    problem.amplitude.assign(AMP_CHECK)

    fom_solution = solve_steady_navier_stokes(problem, initial_solution.solution)
    # assert fom_solution.converged, "FOM Newton failed at Re=68 symmetric"

    # Build FOM pencil (J_fom, M_fom), both L² by construction
    J_fom, M_fom = build_state_jacobian_pencil(problem, fom_solution)

    eigvals_fom, vrs_fom, vis_fom, nconv_fom = solve_leftmost_real_eigenpairs(
        J_fom, M_fom, n_eigenvalues=6, target=0.0,
    )
    assert nconv_fom > 0, "FOM eigensolver converged nothing"

    # Find leftmost REAL eigenvalue in FOM spectrum (same logic as ROM indicator)
    imag_tol_fom  = 1e-6 * max(abs(e) for e in eigvals_fom)
    real_mask_fom = np.abs(eigvals_fom.imag) < imag_tol_fom
    real_idx_fom  = np.where(real_mask_fom)[0]
    assert len(real_idx_fom) > 0, "No real eigenvalues in FOM spectrum"

    i_pitch_fom   = int(real_idx_fom[np.argmin(eigvals_fom[real_mask_fom].real)])
    mu_fom        = float(eigvals_fom[i_pitch_fom].real)

    # ── Check 1: μ_ROM ≈ μ_FOM ───────────────────────────────────────────────────
    mu_rom = indicator['mu_leftmost_real']   # from the earlier ROM indicator run
    rel_err = abs(mu_rom - mu_fom) / abs(mu_fom)
    print(f"\nCheck 1 — eigenvalue agreement:")
    print(f"  μ_ROM = {mu_rom:+.6e}")
    print(f"  μ_FOM = {mu_fom:+.6e}")
    print(f"  rel error = {rel_err:.2%}   (expect a few percent)")

    # ── Check 2: complex pair — real physics or ROM artifact? ────────────────────
    print(f"\nCheck 2 — full spectra (6 eigenvalues each):")
    print(f"  {'i':>2}  {'Re(μ_ROM)':>14}  {'Im(μ_ROM)':>12}  {'Re(μ_FOM)':>14}  {'Im(μ_FOM)':>12}")
    for i in range(min(len(eigvals_fom), len(indicator['eigvals_all']))):
        er = indicator['eigvals_all'][i]
        ef = eigvals_fom[i] if i < len(eigvals_fom) else complex(float('nan'))
        print(f"  {i:>2}  {er.real:+14.4e}  {er.imag:+12.4e}  {ef.real:+14.4e}  {ef.imag:+12.4e}")
    # If FOM also has a complex pair near the imaginary axis → real Hopf-precursor.
    # If FOM spectrum is purely real at Re=68 → the complex pair is a ROM artifact
    # (basis memory of higher-Re Hopf-active snapshots in neighbouring clusters).

    # ── Check 3: eigenvector alignment (ν_ROM lifted vs ν_FOM) ──────────────────
    k_active   = result.cluster
    N_u        = operators[k_active].Z_u.shape[1]
    nu_v_red   = indicator['nu_red'][:N_u]
    nu_lifted  = operators[k_active].Z_u @ nu_v_red          # FOM velocity DOFs

    nu_fom_full = vrs_fom[i_pitch_fom].array_r.copy()
    n_vel_dofs  = problem.velocity_space.dim()
    nu_fom_vel  = nu_fom_full[:n_vel_dofs]                   # velocity block only

    # Normalize both in M_L² before computing alignment
    M_L2_mat = assemble_inner_product_matrix(problem.velocity_space, 'L2')

    norm_lifted = np.sqrt(nu_lifted  @ (M_L2_mat @ nu_lifted))
    norm_fom    = np.sqrt(nu_fom_vel @ (M_L2_mat @ nu_fom_vel))
    nu_lifted_n = nu_lifted  / norm_lifted
    nu_fom_n    = nu_fom_vel / norm_fom

    alignment = abs(nu_lifted_n @ (M_L2_mat @ nu_fom_n))
    print(f"\nCheck 3 — eigenvector alignment (M_L²-inner product):")
    print(f"  ||ν_lifted||_L²  = {norm_lifted:.6f}")
    print(f"  ||ν_FOM||_L²     = {norm_fom:.6f}")
    print(f"  |<ν_lifted, ν_FOM>|_L²  = {alignment:.6f}   (expect ≈ 1.0)")
    # ── Actual eigenvector error ──────────────────────────────────────────────────
    # Both already normalized above (nu_lifted_n, nu_fom_n, both M_L²-unit).
    # Eigenvectors are defined up to sign, so take the sign that minimizes error.

    err_plus  = np.sqrt(abs((nu_lifted_n - nu_fom_n) @ (M_L2_mat @ (nu_lifted_n - nu_fom_n))))
    err_minus = np.sqrt(abs((nu_lifted_n + nu_fom_n) @ (M_L2_mat @ (nu_lifted_n + nu_fom_n))))

    err = min(err_plus, err_minus)
    sign_chosen = "same sign" if err_plus < err_minus else "opposite sign"

    print(f"  ||ν_lifted - ν_FOM||_L²  = {err:.6e}   ({sign_chosen})")
    print(f"  (theoretical lower bound from alignment: {np.sqrt(2*(1 - alignment)):.6e})")

if __name__ == '__main__':
    main()