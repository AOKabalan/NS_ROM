"""
Characterization cases for the nsrom baseline test suite.

Single source of truth: `generate_golden.py` and every `test_*.py` import the
case functions here, so captured goldens and asserted values never drift.

Design notes
------------
* Importing this module triggers the (slow) Firedrake import transitively via
  `nsrom.rom` / `nsrom.clustering.building`. That cost is paid ONCE per process.
* Tier A cases run on saved data only (snapshot DOFs + saved H1 mass matrix),
  no Firedrake *objects* are built — only the import is unavoidable.
* Tier B cases build real Firedrake objects (mesh, spaces, solves).
* Every case returns a plain dict of JSON-friendly captured values. Arrays are
  returned as lists; complex numbers as [re, im] pairs. Determinism-fragile
  quantities (raw POD modes, eigenvectors) are deliberately NOT captured.

Data on disk
------------
* multi_param_multi_branch/snapshot_dofs.npz : velocity/pressure DOFs, branch_ids
* mass/M_u_H1.npz                             : velocity H1 energy inner product
* local_rom/K4_H1/                            : prebuilt 4-cluster tensor ROM
* initial_states/base_velocity                : FOM warm-start checkpoint
* lifting/                                     : saved lifting functions
"""
import os

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import adjusted_rand_score

# --- nsrom imports (these pull Firedrake in transitively; paid once) ---------
from nsrom.rom.pod import (
    compute_pod_modes,
    truncate_modes,
    project_to_basis,
    check_orthonormality,
    compute_reconstruction_error,
)
from nsrom.deim import (
    deim_algorithm,
    build_deim_interpolation_matrix,
    check_interpolation_property,
)
from nsrom.clustering.building import (
    build_cholesky_transform,
    transform,
    inverse_transform,
    cluster_kmeans,
    cluster_kmeans_pod_whitened,
)

# --- fixed paths / parameters ------------------------------------------------
SNAPSHOT_DIR = "multi_param_multi_branch"
SNAPSHOT_NPZ = os.path.join(SNAPSHOT_DIR, "snapshot_dofs.npz")
M_U_PATH = "mass/M_u_H1.npz"
MESH_FILE = "mesh/mid_pinball.msh"
LIFTING_DIR = "lifting"
FOM_CHECKPOINT = "initial_states/base_velocity"
LOCAL_ROM_BASE = "local_rom"
BOUNDARY_MARKERS = (1, 3, 4, 5, 6)

PER_BRANCH = 13          # stratified snapshots per branch id
N_POD = 10               # POD modes for A1/A2
N_DEIM = 8               # DEIM indices for A4
N_CLUSTERS_A = 3         # clusters for the Tier-A kernel tests
WHITEN_RANK = 10         # POD rank for whitened clustering (A6)
ENERGY_TOL = 1.0 - 1e-8  # energy-threshold truncation (A3)

# B2 solve point (symmetric branch, low-ish Re for a short Newton)
B2_RE = 50.0
B2_AMP = 0.0


# =============================================================================
# Shared data loaders
# =============================================================================
def load_stratified_snapshots(per_branch=PER_BRANCH):
    """
    Velocity snapshot matrix S (n_dofs, n_sel), block-stratified across branch
    ids: `per_branch` EVENLY SPACED snapshots from each branch (so amp is
    spanned, not stuck at one extreme), concatenated in branch order so the
    selection is three contiguous blocks. Deterministic.

    Also returns one held-out branch-0 snapshot (NOT in the selection) for the
    non-tautological DEIM reconstruction test (A4).
    """
    d = np.load(SNAPSHOT_NPZ, allow_pickle=True)
    vel = d["velocity"]            # (n_snap, n_dofs)
    branch = d["branch_ids"]
    params = d["parameters"]

    idx_blocks = []
    per_branch_ranges = {}
    for bid in sorted(np.unique(branch)):
        bidx = np.nonzero(branch == bid)[0]                      # ascending
        pick = np.unique(np.linspace(0, len(bidx) - 1, per_branch).round().astype(int))
        sel = bidx[pick]
        idx_blocks.append(sel)
        per_branch_ranges[int(bid)] = {
            "n": int(len(sel)),
            "Re": [float(params[sel, 0].min()), float(params[sel, 0].max())],
            "amp": [float(params[sel, 1].min()), float(params[sel, 1].max())],
        }
    idx = np.concatenate(idx_blocks)                             # BLOCK order
    S = np.ascontiguousarray(vel[idx].T)                         # (n_dofs, n_sel)
    sel_branch = branch[idx]
    sel_params = params[idx]

    # Held-out branch-0 snapshot not among the selection (deterministic).
    chosen = set(idx.tolist())
    b0 = np.nonzero(branch == sorted(np.unique(branch))[0])[0]
    holdout_idx = int(next(i for i in b0 if i not in chosen))

    composition = {
        int(b): int((sel_branch == b).sum()) for b in sorted(np.unique(branch))
    }
    return {
        "S": S,
        "params": sel_params,
        "branch_ids": sel_branch,
        "idx": idx,
        "composition": composition,
        "per_branch_ranges": per_branch_ranges,
        "holdout": np.ascontiguousarray(vel[holdout_idx]),
        "holdout_idx": holdout_idx,
        "holdout_param": [float(params[holdout_idx, 0]), float(params[holdout_idx, 1])],
        "n_dofs": int(S.shape[0]),
        "n_sel": int(S.shape[1]),
    }


def load_Mu():
    """Velocity H1 energy inner product (scipy CSR), from saved file."""
    return sp.load_npz(M_U_PATH).tocsr()


def _identity_like(n):
    return sp.identity(n, format="csr")


# =============================================================================
# TIER A — numeric kernels (saved data only)
# =============================================================================
def case_A1_pod_euclidean(S):
    """POD with Euclidean inner product."""
    modes, eigs = compute_pod_modes(S, inner_product=None, n_modes=N_POD)
    I = _identity_like(S.shape[0])
    ortho_ok, ortho_err = check_orthonormality(modes, I)
    n_arr, err_curve = compute_reconstruction_error(S, modes, I)
    return {
        "eigenvalues": eigs[:N_POD].tolist(),
        "n_modes": int(modes.shape[1]),
        "orthonormality_max_err": float(ortho_err),
        "recon_error_at_full": float(err_curve[-1]),
        "eig_decay_ratio": float(eigs[0] / eigs[N_POD - 1]),
    }


def case_A2_pod_energy(S, M_u):
    """POD with H1 energy inner product."""
    modes, eigs = compute_pod_modes(S, inner_product=M_u, n_modes=N_POD)
    ortho_ok, ortho_err = check_orthonormality(modes, M_u)
    n_arr, err_curve = compute_reconstruction_error(S, modes, M_u)
    return {
        "eigenvalues": eigs[:N_POD].tolist(),
        "n_modes": int(modes.shape[1]),
        "orthonormality_max_err": float(ortho_err),
        "recon_error_at_full": float(err_curve[-1]),
        "eig_decay_ratio": float(eigs[0] / eigs[N_POD - 1]),
    }


def case_A3_pod_energy_truncation(S, M_u):
    """
    Energy-threshold truncation at several thresholds.

    At 1-1e-8 the truncation barely engages (retains most modes), so that alone
    has little discriminating power. Tighter thresholds (1-1e-2, 1-1e-4) make
    the retained-mode count actually vary, giving the golden real teeth.
    """
    modes_all, eigs_all = compute_pod_modes(S, inner_product=M_u)
    total = float(eigs_all.sum())
    out = {"n_modes_total": int(modes_all.shape[1])}
    for tol in (1.0 - 1e-2, 1.0 - 1e-4, ENERGY_TOL):   # 0.99, 0.9999, 1-1e-8
        modes_t, eigs_t = truncate_modes(modes_all, eigs_all, energy_threshold=tol)
        key = f"{tol:.10g}"
        out[f"n_retained@{key}"] = int(modes_t.shape[1])
        out[f"energy_frac@{key}"] = float(eigs_t.sum()) / total
    return out


def case_A4_deim(S, holdout):
    """
    DEIM index selection on an 8-mode Euclidean POD basis.

    `interpolation_residual` is exact-by-construction (test vectors live in the
    basis span) — kept only as a sanity check, NOT a discriminating golden. The
    discriminating quantities are:
      * cond_PtU   — conditioning of the DEIM point matrix P^T U
      * holdout_recon_rel_error — DEIM reconstruction error of a snapshot that
        is NOT in the basis (a held-out branch-0 column).
    """
    U_m, _ = compute_pod_modes(S, inner_product=None, n_modes=N_DEIM)
    indices = deim_algorithm(U_m)
    B = build_deim_interpolation_matrix(U_m, indices)
    interp_err = check_interpolation_property(B, indices)
    cond_PtU = float(np.linalg.cond(U_m[indices, :]))
    # Non-tautological: reconstruct a held-out snapshot via DEIM interpolation.
    approx = B @ holdout[indices]
    holdout_rel = float(np.linalg.norm(holdout - approx) / np.linalg.norm(holdout))
    return {
        "indices": [int(i) for i in indices],
        "n_indices": int(len(indices)),
        "interpolation_residual": float(interp_err),
        "cond_PtU": cond_PtU,
        "holdout_recon_rel_error": holdout_rel,
    }


def _clustering_capture(labels, inertia):
    sizes = sorted(int((labels == c).sum()) for c in np.unique(labels))
    return {
        "sorted_cluster_sizes": sizes,
        "inertia": float(inertia),
        "labels": [int(x) for x in labels],   # compared via ARI == 1.0
        "n_assigned": int(len(labels)),
    }


def case_A5_clustering_cholesky(S, M_u, branch_ids):
    """
    Energy-norm k-means via Cholesky whitening of the H1 inner product.

    The "clustering recovers the branches" claim is tested explicitly via
    adjusted_rand_score(labels, branch_ids) — permutation-invariant, not a
    visual pattern match. branch_ids is recorded alongside the labels.
    """
    L, P = build_cholesky_transform(M_u)
    S_tilde = transform(L, P, S)
    labels, centers, inertia = cluster_kmeans(S_tilde, N_CLUSTERS_A)
    cap = _clustering_capture(labels, inertia)
    cap["branch_ids"] = [int(b) for b in branch_ids]
    cap["adjusted_rand_vs_branch"] = float(adjusted_rand_score(branch_ids, labels))
    return cap


def case_A6_clustering_whitened(S, M_u):
    """POD-whitened k-means (Mahalanobis whitening in POD-coefficient space)."""
    labels, centers_dofs, inertia, modes, sigma, centers_pod = (
        cluster_kmeans_pod_whitened(S, N_CLUSTERS_A, WHITEN_RANK, M_u=M_u)
    )
    cap = _clustering_capture(labels, inertia)
    # sign-invariant summary of the centers (per-axis sign is arbitrary)
    cap["center_norms"] = [float(np.linalg.norm(c)) for c in centers_pod]
    return cap


def case_A7_transform_roundtrip(S, M_u):
    """Cholesky energy-transform round-trip residual (no verify_energy_equivalence)."""
    L, P = build_cholesky_transform(M_u)
    S_tilde = transform(L, P, S)
    S_back = inverse_transform(L, P, S_tilde)
    abs_res = float(np.max(np.abs(S_back - S)))
    rel_res = abs_res / float(np.max(np.abs(S)))
    # energy preservation: ||s||_M^2 vs ||s_tilde||_2^2 for one column
    col = S[:, 0]
    e_M = float(col @ (M_u @ col))
    e_tilde = float(S_tilde[:, 0] @ S_tilde[:, 0])
    return {
        "max_roundtrip_abs_residual": abs_res,
        "max_roundtrip_rel_residual": rel_res,
        "energy_ratio_tilde_over_M": e_tilde / e_M,
    }


# =============================================================================
# TIER B — Firedrake objects
# =============================================================================
def build_problem():
    from nsrom.navier_stokes import setup_navier_stokes_problem
    return setup_navier_stokes_problem(
        MESH_FILE, reynolds_init=100.0, amplitude_init=0.0
    )


def case_B1_lifting(problem):
    """
    Characterize the STORED lifting functions (load, don't recompute).

    compute_lifting_functions does not converge from a cold Re=100 start and
    cannot regenerate these artifacts — see docs/known-issues.md. We therefore
    pin the stored artifact via load_lifting_functions.
    """
    from firedrake import norm
    from nsrom.lifting_functions import load_lifting_functions
    lift = load_lifting_functions(LIFTING_DIR, problem)
    return {
        "source": "stored (lifting/)",
        "norm_u_base_L2": float(norm(lift.u_base)),
        "norm_u_control_L2": float(norm(lift.u_control)),
        "norm_u_base_H1": float(norm(lift.u_base, "H1")),
        "norm_u_control_H1": float(norm(lift.u_control, "H1")),
    }


def _build_rom_bundle_readonly(problem):
    """
    Load the prebuilt K4_H1 tensor ROM WITHOUT writing into local_rom/.

    Refuses any tensor rebuild (which would overwrite conv_tensor.npz). Returns
    (clustering, operators, deim_ops_dict, M_u, M_p, initial_solution).
    """
    from nsrom.snapshots.collection import load_snapshot_dofs
    from nsrom.lifting_functions import load_lifting_functions
    from nsrom.clustering.building import build_energy_snapshots, compute_or_load_clustering
    from nsrom.rom.local import (
        build_local_roms, precompute_fast_selection, precompute_change_of_basis,
    )
    from nsrom.layout import LocalROMLayout
    from nsrom.config import LocalROMConfig
    from nsrom.navier_stokes import load_solution
    from nsrom.tensor_convection import load_convection_tensor, TensorConvection

    cfg = LocalROMConfig(
        n_clusters=4, inner_product_type="H1", pod_energy_tol=1e-8,
        n_velocity_max=400, n_pressure_max=200, n_supremizer_max=200,
        boundary_markers=BOUNDARY_MARKERS, mode="tensor",
        compute_affine_convection=True,
        deim_energy_tol_F=1e-16, deim_energy_tol_J=1e-16,
        m_F_max=400, m_J_max=400, n_modes_F=None, n_modes_J=None,
        recompute_clustering=False, recompute_pod=False,
        recompute_deim=False, recompute_tensor=False,
    )
    layout = LocalROMLayout(base_dir=LOCAL_ROM_BASE, n_clusters=4, inner_product_type="H1")

    # Guard: every cluster must already be built, so build_local_roms only LOADS.
    for k in range(cfg.n_clusters):
        if not layout.pod_exists(k):
            raise RuntimeError(
                f"cluster {k} not prebuilt under {layout.root}; refusing to "
                "build (read-only characterization)."
            )

    snap = load_snapshot_dofs(SNAPSHOT_DIR)
    lifting = load_lifting_functions(LIFTING_DIR, problem)
    S_homo, S_tilde, M_u, M_p, L, P = build_energy_snapshots(
        snap["velocity"], snap["parameters"], lifting, problem,
        cfg.inner_product_type, homogenize=True,
    )
    clustering, _ = compute_or_load_clustering(
        S_tilde, S_homo, M_u, snap["parameters"], L, P,
        save_dir=layout.clustering_dir, n_clusters=cfg.n_clusters,
        method="pod_whitened", rank=WHITEN_RANK, recompute=False,
    )
    bases, operators, deim_ops_dict = build_local_roms(
        clustering=clustering, velocity_snapshots=snap["velocity"],
        pressure_snapshots=snap["pressure"], deim_snapshot_data=None,
        deim_snapshot_dir=SNAPSHOT_DIR, lifting=lifting, problem=problem,
        cfg=cfg, layout=layout, M_u=M_u,
    )
    # Attach tensors, read-only: load only, refuse rebuild.
    for k in range(clustering.n_clusters):
        Z_u_k = operators[k].Z_u
        tpath = os.path.join(layout.cluster_dir(k), "conv_tensor.npz")
        if not os.path.exists(tpath):
            raise RuntimeError(f"cluster {k}: conv_tensor.npz missing; refusing rebuild.")
        N = load_convection_tensor(tpath)
        if N.shape[0] != Z_u_k.shape[1]:
            raise RuntimeError(
                f"cluster {k}: cached tensor stale ({N.shape[0]} != {Z_u_k.shape[1]}); "
                "refusing rebuild to protect local_rom/."
            )
        operators[k].tensor_convection = TensorConvection(N)

    precompute_fast_selection(clustering, operators, M_u)
    precompute_change_of_basis(clustering, operators, M_u, M_p)

    initial_solution = load_solution(FOM_CHECKPOINT, problem)
    return clustering, operators, deim_ops_dict, M_u, M_p, initial_solution


def case_B2_rom_solve(problem):
    """
    Short tensor-mode ROM solve at (Re, amp) with snes_max_it=10.

    Driven via solve_rom directly (not solve_at_parameter) so we can set the
    Newton cap explicitly and distinguish genuine convergence from a cap-hit:
    with max_it=10, `converged=True` and `iterations<10` => real convergence.
    The PETSc SNES converged reason prints in the log (snes_converged_reason).

    Note: solve_rom / _extract_solution does NOT populate residual_norm (stays
    0.0) — recorded here as-is; see review notes.
    """
    from firedrake import norm
    from nsrom.rom.local import (
        select_the_cluster, project_initial_solution, reconstruct_solution,
    )
    from nsrom.rom.rom_solver import solve_rom, _DEFAULT_SOLVER_PARAMETERS

    (clustering, operators, deim_ops_dict,
     M_u, M_p, initial_solution) = _build_rom_bundle_readonly(problem)

    problem.reynolds.assign(B2_RE)
    problem.amplitude.assign(B2_AMP)
    nu = 1.0 / B2_RE

    k = select_the_cluster(clustering, B2_RE, B2_AMP, initial_solution, M_u,
                           verbose=False)
    u_c, p_c = project_initial_solution(initial_solution, operators[k],
                                        B2_AMP, M_u, M_p)

    # Production tolerances, but cap Newton at 10 to test convergence vs cap-hit.
    sp = dict(_DEFAULT_SOLVER_PARAMETERS)
    sp["snes_max_it"] = 10

    sol = solve_rom(
        operators=operators[k], nu=nu, amp=B2_AMP, problem=problem, mode="tensor",
        u_initial_guess=u_c, p_initial_guess=p_c, solver_parameters=sp,
        store_full_dofs=True, verbose=True,
    )
    rec = reconstruct_solution(sol, operators[k], problem, amp=B2_AMP)

    return {
        "Re": B2_RE, "amp": B2_AMP,
        "cluster_selected": int(k),
        "snes_max_it": 10,
        "converged": bool(sol.converged),
        "iterations": int(sol.iterations),
        "residual_norm_reported": float(sol.residual_norm),   # 0.0 (not populated)
        "recon_velocity_L2": float(norm(rec.velocity)),
    }


def case_B3_eigen(problem):
    """
    Values-only SLEPc characterization (NON-BLOCKING).

    The eigensolver sets no fixed initial vector, so eigenVALUES are
    reproducible to tolerance but eigenVECTORS are not — capture only the
    leftmost eigenvalues, and treat any drift as non-gating.

    Physics note (do NOT "fix" this): at Re=50 — above the Hopf bifurcation at
    Re≈18 — the base flow is linearly unstable, so an eigenvalue pair with
    POSITIVE real part (≈ +0.0439 ± 0.520i, an oscillatory/Hopf mode) is
    expected and correct, not a bug. The leftmost pair (≈ -0.0562 ± 0.509i) is
    stable. A future reader should not "correct" the positive-real-part pair.
    """
    from nsrom.navier_stokes import load_solution
    from nsrom.bifurcation.jacobian import build_state_jacobian_pencil
    from nsrom.bifurcation.eigen_solver import solve_leftmost_real_eigenpairs

    base = load_solution(FOM_CHECKPOINT, problem)
    J, M = build_state_jacobian_pencil(problem, base)
    eigvals, _vr, _vi, nconv = solve_leftmost_real_eigenpairs(
        J, M, n_eigenvalues=4, target=0.0
    )
    k = min(4, len(eigvals))
    return {
        "n_converged": int(nconv),
        "leftmost_eigenvalues": [[float(e.real), float(e.imag)] for e in eigvals[:k]],
        "leftmost_real": float(eigvals[0].real) if len(eigvals) else None,
    }


# =============================================================================
# Case registry
# =============================================================================
TIER_A = ["A1", "A2", "A3", "A4", "A5", "A6", "A7"]
TIER_B = ["B1", "B2", "B3"]          # B3 non-blocking
NONBLOCKING = {"B3"}
