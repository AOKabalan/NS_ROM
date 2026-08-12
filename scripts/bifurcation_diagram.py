"""
Bifurcation diagram: ROM vs FOM.

Reads:
    - sweep_results.npz
    - rom_branch_seeds.npz

Writes:
    - bifurcation_diagram.png
    - bifurcation_diagram.pdf
    - bifurcation_diagram.npz
"""

import numpy as np
import matplotlib.pyplot as plt

from nsrom.rom import reconstruct_solution, solve_rom_with_internals
from nsrom.navier_stokes import (
    setup_navier_stokes_problem,
    load_solution,
    solve_steady_navier_stokes,
    compute_forces,
)
from nsrom.snapshot_collection import load_snapshot_dofs
from nsrom.lifting_functions import load_lifting_functions
from nsrom.cluster_building import build_energy_snapshots, compute_or_load_clustering
from nsrom.local_rom import build_local_roms
from nsrom.layout import LocalROMLayout
from nsrom.config import LocalROMConfig

from nsrom.bifurcation.eigen_solver import solve_leftmost_real_eigenpairs
from nsrom.bifurcation.jacobian import build_state_jacobian_pencil
from nsrom.bifurcation.branch_jump import (
    branch_jump_pitchfork,
    continue_asymmetric_branch,
)


# =============================================================================
# CONFIG
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

SNAPSHOT_DIR = "multi_param_multi_branch"
LIFTING_DIR = "lifting"
MESH_FILE = "mesh/mid_pinball.msh"
FOM_CHECKPOINT = "initial_states/velocity_checkpoint_symm"

REYNOLDS_INIT = 100.0
AMPLITUDE_INIT = 0.0
AMP_PARAM = 0.0

SWEEP_RESULTS = "sweep_results.npz"
ROM_SEEDS = "rom_branch_seeds.npz"

RE_MAX = 90.0

DRE_NEAR = 0.5
DRE_FAR = 1.0
NEAR_STEPS = 4
CL_COLLAPSE = 0.5

# Use the same simple FOM branch-jump settings that worked before.
FOM_EPS_VALUES = (0.5,)
FOM_AMP_VALUES = (1e-1,)


# =============================================================================
# HELPERS
# =============================================================================

def detect_sign_change(re_values, mu_values):
    """Return linearly interpolated first zero crossing."""
    re = np.asarray(re_values, dtype=float)
    mu = np.asarray(mu_values, dtype=float)

    mask = ~np.isnan(mu)
    re = re[mask]
    mu = mu[mask]

    if len(mu) < 2:
        return None

    idx = np.where(np.diff(np.sign(mu)) != 0)[0]
    if len(idx) == 0:
        return None

    i = int(idx[0])
    t = -mu[i] / (mu[i + 1] - mu[i])
    return float(re[i] + t * (re[i + 1] - re[i]))


def extract_rom_coefficients(w_rom):
    """Extract velocity and pressure ROM coefficients."""
    u_sub, p_sub = w_rom.subfunctions

    with u_sub.dat.vec_ro as v:
        alpha = v.array.copy()

    with p_sub.dat.vec_ro as v:
        beta = v.array.copy()

    return alpha, beta


def solve_rom_fixed_cluster(
    Re,
    amp,
    cluster_id,
    alpha_init,
    beta_init,
    operators,
    deim_ops_dict,
    problem,
    use_deim,
):
    """Solve ROM in a fixed cluster with given initial coefficients."""
    problem.reynolds.assign(float(Re))
    problem.amplitude.assign(float(amp))

    nu = 1.0 / float(Re)

    deim_ops = deim_ops_dict[cluster_id]
    submesh_deim = None

    if use_deim and deim_ops is not None:
        from nsrom.local_rom import SubMeshDEIM
        submesh_deim = SubMeshDEIM(problem.velocity_space, deim_ops)

    return solve_rom_with_internals(
        operators=operators[cluster_id],
        nu=nu,
        amp=amp,
        problem=problem,
        deim_ops=deim_ops,
        u_initial_guess=alpha_init,
        p_initial_guess=beta_init,
        submesh_deim=submesh_deim,
    )


def fom_pitchfork_eigvec(problem, fom_solution, n_eig=6, target=0.0):
    """
    Return FOM pitchfork eigenvector.

    We select the real eigenvalue closest to zero, not the most negative one.
    """
    J, M = build_state_jacobian_pencil(problem, fom_solution)

    eigvals, vrs, _, nconv = solve_leftmost_real_eigenpairs(
        J,
        M,
        n_eigenvalues=n_eig,
        target=target,
    )

    if nconv == 0:
        raise RuntimeError("FOM eigensolver converged no eigenpairs.")

    scale = max(1.0, max(abs(e) for e in eigvals))
    imag_tol = 1e-8 * scale

    real_idx = np.where(np.abs(eigvals.imag) < imag_tol)[0]

    if len(real_idx) == 0:
        raise RuntimeError("No real FOM eigenvalue found near target.")

    i = int(real_idx[np.argmin(np.abs(eigvals.real[real_idx]))])

    return vrs[i].copy(), float(eigvals[i].real)


def continue_rom_branch(
    label,
    Re_seed,
    alpha_seed,
    beta_seed,
    cluster_id,
    operators,
    deim_ops_dict,
    problem,
    use_deim,
    amp,
):
    """Continue one ROM branch in fixed cluster."""
    print(f"\n--- ROM branch: {label} ---")

    records = []

    alpha_prev = alpha_seed.copy()
    beta_prev = beta_seed.copy()

    Re = float(Re_seed)

    try:
        solution, w_rom, _ = solve_rom_fixed_cluster(
            Re=Re,
            amp=amp,
            cluster_id=cluster_id,
            alpha_init=alpha_prev,
            beta_init=beta_prev,
            operators=operators,
            deim_ops_dict=deim_ops_dict,
            problem=problem,
            use_deim=use_deim,
        )
    except Exception as e:
        print(f"  Seed solve failed: {type(e).__name__}: {e}")
        return records

    reconstructed = reconstruct_solution(
        solution,
        operators[cluster_id],
        problem,
        amp=amp,
    )

    forces = compute_forces(reconstructed, problem)
    cl_prev = float(forces.lift)
    cl_sign = np.sign(cl_prev)
    iters = getattr(solution, "iterations", "?")

    records.append({
        "Re": Re,
        "C_L": cl_prev,
        "C_D": float(forces.drag),
        "iters": iters,
    })

    print(f"  step 0 | Re={Re:.4f} | C_L={cl_prev:+.6e} | it={iters}")

    alpha_prev, beta_prev = extract_rom_coefficients(w_rom)

    step = 0

    while Re < RE_MAX - 1e-12:
        step += 1

        dRe = DRE_NEAR if step <= NEAR_STEPS else DRE_FAR
        Re_new = min(Re + dRe, RE_MAX)

        try:
            solution, w_rom, _ = solve_rom_fixed_cluster(
                Re=Re_new,
                amp=amp,
                cluster_id=cluster_id,
                alpha_init=alpha_prev,
                beta_init=beta_prev,
                operators=operators,
                deim_ops_dict=deim_ops_dict,
                problem=problem,
                use_deim=use_deim,
            )
        except Exception as e:
            print(f"  step {step} | Re={Re_new:.4f} | Newton failed: {type(e).__name__}: {e}")
            break

        reconstructed = reconstruct_solution(
            solution,
            operators[cluster_id],
            problem,
            amp=amp,
        )

        forces = compute_forces(reconstructed, problem)
        cl_new = float(forces.lift)
        iters = getattr(solution, "iterations", "?")

        flipped = np.sign(cl_new) != cl_sign
        collapsed = abs(cl_new) < CL_COLLAPSE * abs(cl_prev)

        if flipped or collapsed:
            reason = "sign flip" if flipped else "collapse"
            print(
                f"  step {step} | Re={Re_new:.4f} | "
                f"C_L={cl_new:+.6e} | branch lost: {reason}"
            )
            break

        records.append({
            "Re": float(Re_new),
            "C_L": cl_new,
            "C_D": float(forces.drag),
            "iters": iters,
        })

        print(
            f"  step {step} | Re={Re_new:.4f} | "
            f"C_L={cl_new:+.6e} | it={iters}"
        )

        alpha_prev, beta_prev = extract_rom_coefficients(w_rom)
        cl_prev = cl_new
        Re = Re_new

    return records


def select_seed_by_lift(seeds, problem):
    """
    Select one negative-lift and one positive-lift seed.

    This is better than selecting by kick sign because eigenvector sign is arbitrary.
    """
    infos = []

    for seed in seeds:
        forces = compute_forces(seed["solution"], problem)

        infos.append({
            "seed": seed,
            "C_L": float(forces.lift),
            "C_D": float(forces.drag),
            "epsilon": float(seed["epsilon"]),
            "amplitude": float(seed["amplitude"]),
            "kick_sign": int(seed["sign"]),
            "Re": float(seed["Re"]),
        })

    if not infos:
        return None, None

    print("\nSuccessful FOM seeds:")
    for info in infos:
        print(
            f"  kick={info['kick_sign']:+d} | "
            f"Re={info['Re']:.4f} | "
            f"eps={info['epsilon']:.3e} | "
            f"amp={info['amplitude']:.3e} | "
            f"C_L={info['C_L']:+.6e}"
        )

    # Prefer closest seed to bifurcation.
    eps_min = min(info["epsilon"] for info in infos)
    near = [info for info in infos if info["epsilon"] == eps_min]

    neg = next((info for info in near if info["C_L"] < 0.0), None)
    pos = next((info for info in near if info["C_L"] > 0.0), None)

    # Fall back to all seeds if one sign is missing at eps_min.
    if neg is None:
        neg = next((info for info in infos if info["C_L"] < 0.0), None)

    if pos is None:
        pos = next((info for info in infos if info["C_L"] > 0.0), None)

    return neg, pos


def continue_fom_branch(label, seed_info, problem):
    """Continue one FOM branch from a selected seed."""
    if seed_info is None:
        print(f"\n--- FOM branch {label}: missing seed ---")
        return []

    seed = seed_info["seed"]

    print(
        f"\n--- FOM branch {label} ---\n"
        f"  seed Re={seed_info['Re']:.4f}, "
        f"C_L={seed_info['C_L']:+.6e}, "
        f"kick={seed_info['kick_sign']:+d}, "
        f"amp={seed_info['amplitude']:.3e}"
    )

    _, diagnostics = continue_asymmetric_branch(
        problem=problem,
        sol_start=seed["solution"],
        Re_max=RE_MAX,
        dRe_near=DRE_NEAR,
        dRe_far=DRE_FAR,
        near_steps=NEAR_STEPS,
        cl_collapse_factor=CL_COLLAPSE,
    )

    return diagnostics


def branch_arrays(records):
    if not records:
        return np.array([]), np.array([])

    return (
        np.array([r["Re"] for r in records]),
        np.array([r["C_L"] for r in records]),
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("BIFURCATION DIAGRAM — ROM vs FOM")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Load sweep and ROM seeds.
    # -------------------------------------------------------------------------

    sweep = np.load(SWEEP_RESULTS)

    re_sym = np.asarray(sweep["Re"])
    mu_rom = np.asarray(sweep["mu_rom"])
    mu_fom = np.asarray(sweep["mu_fom"])

    re_c_rom = detect_sign_change(re_sym, mu_rom)
    re_c_fom = detect_sign_change(re_sym, mu_fom)

    if re_c_rom is None:
        raise RuntimeError("Could not detect Re_c^ROM.")

    if re_c_fom is None:
        raise RuntimeError("Could not detect Re_c^FOM.")

    print(f"Re_c^ROM = {re_c_rom:.6f}")
    print(f"Re_c^FOM = {re_c_fom:.6f}")

    seed_data = np.load(ROM_SEEDS)

    Re_seed = float(seed_data["Re_seed"])
    cluster_id = int(seed_data["cluster_anchor"])

    alpha_pos = np.asarray(seed_data["alpha_pos"])
    beta_pos = np.asarray(seed_data["beta_pos"])

    alpha_neg = np.asarray(seed_data["alpha_neg"])
    beta_neg = np.asarray(seed_data["beta_neg"])

    print(f"ROM seed Re = {Re_seed:.6f}")
    print(f"ROM fixed cluster = {cluster_id}")

    # -------------------------------------------------------------------------
    # Build problem and ROM.
    # -------------------------------------------------------------------------

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
    lifting = load_lifting_functions(LIFTING_DIR, problem)

    velocity_snapshots = snapshot_data["velocity"]
    pressure_snapshots = snapshot_data["pressure"]
    parameters = snapshot_data["parameters"]

    S_homo, S_tilde, M_u, M_p, L, P = build_energy_snapshots(
        velocity_snapshots,
        parameters,
        lifting,
        problem,
        cfg.inner_product_type,
        homogenize=False,
    )

    clustering, _ = compute_or_load_clustering(
        S_tilde,
        S_homo,
        M_u,
        parameters,
        L,
        P,
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

    # -------------------------------------------------------------------------
    # Continue ROM branches.
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("ROM ASYMMETRIC BRANCHES")
    print("=" * 80)

    rom_branch_a = continue_rom_branch(
        label="ROM seed +",
        Re_seed=Re_seed,
        alpha_seed=alpha_pos,
        beta_seed=beta_pos,
        cluster_id=cluster_id,
        operators=operators,
        deim_ops_dict=deim_ops_dict,
        problem=problem,
        use_deim=cfg.use_deim,
        amp=AMP_PARAM,
    )

    rom_branch_b = continue_rom_branch(
        label="ROM seed -",
        Re_seed=Re_seed,
        alpha_seed=alpha_neg,
        beta_seed=beta_neg,
        cluster_id=cluster_id,
        operators=operators,
        deim_ops_dict=deim_ops_dict,
        problem=problem,
        use_deim=cfg.use_deim,
        amp=AMP_PARAM,
    )

    # -------------------------------------------------------------------------
    # FOM critical solution and branch jump.
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("FOM ASYMMETRIC BRANCHES")
    print("=" * 80)

    problem.reynolds.assign(float(re_c_fom))
    problem.amplitude.assign(float(AMP_PARAM))

    sol_c_fom = solve_steady_navier_stokes(
        problem,
        initial_solution.solution,
    )

    eigvec_c, mu_c = fom_pitchfork_eigvec(problem, sol_c_fom)

    print(
        f"FOM symmetric solution near Re_c:\n"
        f"  Re_c^FOM = {re_c_fom:.6f}\n"
        f"  mu_c     = {mu_c:+.6e}"
    )

    seeds_fom = branch_jump_pitchfork(
        problem=problem,
        sol_c=sol_c_fom,
        eigvec_c=eigvec_c,
        Re_c=re_c_fom,
        eps_values=FOM_EPS_VALUES,
        amp_values=FOM_AMP_VALUES,
    )

    neg_seed, pos_seed = select_seed_by_lift(seeds_fom, problem)

    if neg_seed is None or pos_seed is None:
        print("\nWARNING: did not find both FOM lift-sign branches.")
        print("The ROM diagram will still plot, but FOM comparison is incomplete.")

    fom_neg = continue_fom_branch("C_L < 0", neg_seed, problem)
    fom_pos = continue_fom_branch("C_L > 0", pos_seed, problem)

    # -------------------------------------------------------------------------
    # Prepare arrays.
    # -------------------------------------------------------------------------

    rom_a_Re, rom_a_CL = branch_arrays(rom_branch_a)
    rom_b_Re, rom_b_CL = branch_arrays(rom_branch_b)

    fom_neg_Re, fom_neg_CL = branch_arrays(fom_neg)
    fom_pos_Re, fom_pos_CL = branch_arrays(fom_pos)

    # -------------------------------------------------------------------------
    # Plot.
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("PLOTTING")
    print("=" * 80)

    fig, ax = plt.subplots(figsize=(8, 5.5))

    ax.axhline(
        0.0,
        color="0.7",
        lw=1.0,
        label="symmetric branch",
    )

    if len(rom_a_Re):
        label = "ROM, C_L > 0" if rom_a_CL[0] > 0 else "ROM, C_L < 0"
        ax.plot(rom_a_Re, rom_a_CL, "o-", ms=5, lw=1.5, label=label)

    if len(rom_b_Re):
        label = "ROM, C_L > 0" if rom_b_CL[0] > 0 else "ROM, C_L < 0"
        ax.plot(rom_b_Re, rom_b_CL, "s-", ms=5, lw=1.5, label=label)

    if len(fom_neg_Re):
        ax.plot(fom_neg_Re, fom_neg_CL, "--", lw=1.8, label="FOM, C_L < 0")

    if len(fom_pos_Re):
        ax.plot(fom_pos_Re, fom_pos_CL, "--", lw=1.8, label="FOM, C_L > 0")

    ax.axvline(
        re_c_rom,
        color="tab:purple",
        ls=":",
        lw=1.2,
        label=f"Re_c^ROM = {re_c_rom:.3f}",
    )

    ax.axvline(
        re_c_fom,
        color="black",
        ls=":",
        lw=1.2,
        label=f"Re_c^FOM = {re_c_fom:.3f}",
    )

    ax.set_xlabel("Re")
    ax.set_ylabel(r"$C_L$")
    ax.set_title(f"Pitchfork bifurcation: ROM cluster {cluster_id} vs FOM")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()

    fig.savefig("bifurcation_diagram.png", dpi=150)
    fig.savefig("bifurcation_diagram.pdf")

    print("Saved bifurcation_diagram.png")
    print("Saved bifurcation_diagram.pdf")

    # -------------------------------------------------------------------------
    # Save.
    # -------------------------------------------------------------------------

    np.savez(
        "bifurcation_diagram.npz",
        re_c_rom=re_c_rom,
        re_c_fom=re_c_fom,

        sym_Re=re_sym,
        sym_mu_rom=mu_rom,
        sym_mu_fom=mu_fom,

        rom_a_Re=rom_a_Re,
        rom_a_CL=rom_a_CL,
        rom_b_Re=rom_b_Re,
        rom_b_CL=rom_b_CL,

        fom_neg_Re=fom_neg_Re,
        fom_neg_CL=fom_neg_CL,
        fom_pos_Re=fom_pos_Re,
        fom_pos_CL=fom_pos_CL,
    )

    print("Saved bifurcation_diagram.npz")

    print("\nROM iteration counts:")
    print(f"  ROM branch A: {[r['iters'] for r in rom_branch_a]}")
    print(f"  ROM branch B: {[r['iters'] for r in rom_branch_b]}")


if __name__ == "__main__":
    main()
