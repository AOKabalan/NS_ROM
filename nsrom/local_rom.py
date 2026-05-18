"""
Local reduced-order model construction.

Per-cluster orchestration of POD bases, reduced operators, and DEIM
operators, plus online utilities (cluster selection, change-of-basis).

Depends on:
  * nsrom.cluster_building  — for ClusteringResult
  * nsrom.rom               — for POD/operator construction
  * nsrom.deim, nsrom.online_operators, nsrom.deim_helpers — for DEIM
"""
import os
import numpy as np

from nsrom.helper_functions import dofs_to_functions, modes_for_tolerance
from nsrom.rom import (
    ROMSolution,
    compute_pod_basis,
    save_pod_basis,
    load_pod_basis,
    build_reduced_operators,
    save_reduced_operators,
    load_reduced_operators,
    homogenize_snapshots,
    solve_rom,
    solve_rom_with_internals,
    reconstruct_solution,
    project_to_rom_coefficients,
    SubMeshDEIM,
)
from nsrom.deim import compute_and_save_basis
from nsrom.online_operators import DEIMOnlineOperators, build_deim_online_operators
from nsrom.deim_helpers import needs_basis_recompute, needs_ops_rebuild
import time
from dataclasses import dataclass



# =============================================================================
# Cluster selection
# =============================================================================
def select_cluster_by_box_then_centroid(clustering, Re, amp):

    query = np.array([Re, amp])

    centers = np.array([
        clustering.parameter_centroid(k)
        for k in range(clustering.n_clusters)
    ])

    # Global normalization scale
    scales = np.ptp(centers, axis=0)
    scales[scales == 0] = 1.0

    # Distances to all centroids
    raw_distances = np.linalg.norm((centers - query) / scales, axis=1)
    distances = np.full(clustering.n_clusters, np.inf)

    # Find clusters whose parameter box contains the query
    eligible = []

    for k in range(clustering.n_clusters):
        params_k = clustering.cluster_parameters(k)

        re_min, amp_min = params_k.min(axis=0)
        re_max, amp_max = params_k.max(axis=0)

        in_box = (
            re_min <= Re <= re_max
            and amp_min <= amp <= amp_max
        )

        if in_box:
            eligible.append(k)
            distances[k] = raw_distances[k]

    # Choose best cluster
    if eligible:
        best = int(np.argmin(distances))

        mode = "box + centroid"
    else:
        # fallback: no cluster contains query, so use true centroid distances
        distances = raw_distances.copy()
        best = int(np.argmin(distances))
        mode = "centroid fallback"
    print(f"  Query (Re={Re}, amp={amp})")
    print(f"  Selection mode: {mode}")

    for k, d in enumerate(distances):
        box_marker = " in-box" if k in eligible else ""
        best_marker = " ←" if k == best else ""
        print(f"  Cluster {k}: distance={d:.4f}{box_marker}{best_marker}")


    return best, distances

def select_cluster(clustering, Re, amp):
    """Parameter-space cluster selection with normalized distances."""
    query    = np.array([Re, amp])
    centers  = np.array([clustering.parameter_centroid(k)
                         for k in range(clustering.n_clusters)])
    scales = np.ptp(centers, axis=0)
    scales[scales == 0] = 1.0
    distances = np.linalg.norm((centers - query) / scales, axis=1)
    best = int(np.argmin(distances))

    print(f"  Query (Re={Re}, amp={amp})")
    for k, d in enumerate(distances):
        marker = " ←" if k == best else ""
        print(f"  Cluster {k}: distance={d:.4f}{marker}")
    return best,distances

def select_cluster_solution_space(clustering,initial_condition,M_u):
    K = clustering.n_clusters
    distances_sq = np.zeros(K)

    centroids = clustering.centroids_original
    print(f' The shape is {centroids.shape}')
    u_fun = initial_condition.velocity
    u_dof = u_fun.dat.data_ro.flatten()

    for j in range(K):
        diff = u_dof - centroids[:,j]
        distances_sq[j] = np.sqrt(diff.T @ M_u @ diff)

    best = int(np.argmin(distances_sq))
    for k, d in enumerate(distances_sq):
        marker = " ←" if k == best else ""
        print(f"  Cluster {k}: distance={d:.4f}{marker}")
    return best, distances_sq

def select_cluster_closest(d1,d2):
    score = np.sqrt((d1 - np.min(d1))**2 + (d2 - np.min(d2))**2)
    print(score)
    best_cluster = np.argmin(score)
    print(f'Closest commom cluster is: {best_cluster}')
    return best_cluster




def precompute_fast_selection(clustering, operators, M_u):
    """
    Precompute data for reduced-coordinate cluster selection.

    For each cluster j, stores:
        c_norm_sq[j] = ||c_j||²_M   (scalar)
    For each pair (i, j), stores:
        cross[i,j] = Z_i^T M c_j    (vector, length n_modes_i)

    Then: ||Z_i a - c_j||²_M = ||a||² - 2 a^T cross[i,j] + c_norm_sq[j]
    (uses M-orthonormality: Z_i^T M Z_i = I)
    """
    K = clustering.n_clusters
    centroids = clustering.centroids_original  # (n_dofs, K)

    c_norm_sq = np.zeros(K)
    for j in range(K):
        c_j = centroids[:, j]
        c_norm_sq[j] = float(c_j @ (M_u @ c_j))

    cross = {}
    for i in range(K):
        Z_i = operators[i].Z_u  # (n_dofs, n_modes_i)
        for j in range(K):
            c_j = centroids[:, j]
            cross[(i, j)] = Z_i.T @ (M_u @ c_j)

    clustering.fast_selection = {
        'c_norm_sq': c_norm_sq,
        'cross': cross,
    }

    print(f"  Fast selection precomputed: {K} clusters, {K*K} cross vectors")


def select_cluster_reduced(clustering, a_prev, k_prev):
    """
    Select cluster using reduced coordinates (no DOF expansion).

    Parameters
    ----------
    clustering : ClusteringResult with fast_selection populated
    a_prev     : (n_modes,) reduced velocity coefficients from previous solve
    k_prev     : int, cluster index of previous solve

    Returns
    -------
    best : int, selected cluster index
    """
    fs = clustering.fast_selection
    K = clustering.n_clusters

    a_norm_sq = float(a_prev @ a_prev)  # ||a||² (M-orthonormal basis)

    distances_sq = np.zeros(K)
    for j in range(K):
        distances_sq[j] = (
            a_norm_sq
            - 2.0 * float(a_prev @ fs['cross'][(k_prev, j)])
            + fs['c_norm_sq'][j]
        )

    best = int(np.argmin(distances_sq))

    print(f"  Reduced selection (from cluster {k_prev}):")
    for k in range(K):
        marker = " ←" if k == best else ""
        print(f"    Cluster {k}: d²_M = {distances_sq[k]:.4f}{marker}")
    return best


# =============================================================================
# Change of basis
# =============================================================================

def precompute_change_of_basis(clustering, operators, M_u, M_p):
    """
    Precompute change-of-basis matrices for cluster switching.

    CB_u[(i,j)]  = Z_j^T M_u Z_i             (velocity transfer)
    CB_p[(i,j)]  = Z_j^T M_p Z_i             (pressure transfer)
    d_u[(j,ell)] = Z_j^T M_u ψ_ell           (lifting correction per lifting function)

    Uses operators[0].lifting_dofs which are shared across clusters.
    """
    K = clustering.n_clusters
    lifting_dofs = operators[0].lifting_dofs  # (n_lifting, n_vel_dofs)
    n_lifting = operators[0].n_lifting

    CB_u = {}
    CB_p = {}
    d_u = {}

    for j in range(K):
        Z_u_j = operators[j].Z_u
        Z_p_j = operators[j].Z_p

        for ell in range(n_lifting):
            d_u[(j, ell)] = Z_u_j.T @ (M_u @ lifting_dofs[ell])

        for i in range(K):
            if i == j:
                continue
            CB_u[(i, j)] = Z_u_j.T @ (M_u @ operators[i].Z_u)
            CB_p[(i, j)] = Z_p_j.T @ (M_p @ operators[i].Z_p)

    clustering.change_of_basis = {
        'CB_u': CB_u,
        'CB_p': CB_p,
        'd_u': d_u,
        'n_lifting': n_lifting,
    }

    n_pairs = K * (K - 1)
    print(f"  Change-of-basis precomputed: {n_pairs} pairs, {n_lifting} lifting functions")


def apply_change_of_basis(clustering, i, j, alpha_i, beta_i, thetas_old, thetas_new):
    """
    Transfer reduced coefficients from cluster i to cluster j.

    α_j = C_ji α_i + Σ_ell (θ_ell^old - θ_ell^new) · d_u[(j, ell)]
    β_j = C_pji β_i
    """
    cb = clustering.change_of_basis
    alpha_j = cb['CB_u'][(i, j)] @ alpha_i
    for ell in range(cb['n_lifting']):
        delta = thetas_old[ell] - thetas_new[ell]
        if delta != 0.0:
            alpha_j += delta * cb['d_u'][(j, ell)]
    beta_j = cb['CB_p'][(i, j)] @ beta_i
    return alpha_j, beta_j


# =============================================================================
# Per-cluster builders
# =============================================================================

def build_cluster_pod(
    k,
    clustering,
    velocity_snapshots,
    pressure_snapshots,
    lifting,
    problem,
    cfg,
    layout,
    u_base,
    u_control,
):
    """
    Build (or load) the POD basis and reduced operators for cluster k.

    Returns
    -------
    basis_k     : POD basis object (truncated)
    operators_k : ReducedOperators object
    """
    if not cfg.recompute_pod and layout.pod_exists(k):
        basis_k = load_pod_basis(layout.pod_dir(k))
        operators_k = load_reduced_operators(layout.ops_dir(k))
        return basis_k, operators_k

    idx      = clustering.cluster_indices[k]
    params_k = clustering.cluster_parameters(k)

    S_u_k = velocity_snapshots[idx, :]
    S_p_k = pressure_snapshots[idx, :]

    basis_k = compute_pod_basis(
        velocity_snapshots=S_u_k,
        pressure_snapshots=S_p_k,
        problem=problem,
        inner_product_type=cfg.inner_product_type,
        compute_supremizer=True,
        parameters=params_k,
        u_base=u_base,
        u_control=u_control,
        amplitude_index=1,
        boundary_markers=list(cfg.boundary_markers),
    )

    n_vel  = modes_for_tolerance(basis_k.velocity_eigenvalues, cfg.pod_energy_tol, cfg.n_velocity_max)
    n_pres = modes_for_tolerance(basis_k.pressure_eigenvalues, cfg.pod_energy_tol, cfg.n_pressure_max)
    n_sup  = n_pres  # match pressure

    basis_k = basis_k.truncate(
        n_velocity=n_vel,
        n_pressure=n_pres,
        n_supremizer=n_sup,
    )

    operators_k = build_reduced_operators(
        problem,
        basis_k,
        lifting,
        inner_product_type=cfg.inner_product_type,
        compute_affine_convection=cfg.compute_affine_convection,
    )

    save_pod_basis(basis_k, layout.pod_dir(k))
    save_reduced_operators(operators_k, layout.ops_dir(k))

    return basis_k, operators_k


def build_cluster_deim(
    k,
    clustering,
    operators_k,
    deim_snapshot_data,
    deim_snapshot_dir,
    problem,
    cfg,
    layout,
    u_base,
    u_control,
):
    """
    Build (or load) the DEIM basis and online operators for cluster k.

    Returns
    -------
    deim_ops_k : DEIMOnlineOperators or None
        None if cfg.use_deim is False.
    """
    if not cfg.use_deim:
        return None

    Z_u_k         = operators_k.Z_u
    n_rom_modes_k = Z_u_k.shape[1]

    bp = layout.deim_basis(k)
    op = layout.deim_ops(k)
    idx = clustering.cluster_indices[k]

    # --- DEIM basis ---
    if cfg.recompute_deim or cfg.recompute_pod:
        recompute_basis = True
    else:
        recompute_basis, _ = needs_basis_recompute(
            bp, deim_snapshot_dir, cfg.n_modes_F, cfg.n_modes_J
        )

    if recompute_basis:
        deim_vel_k = deim_snapshot_data['velocity'][idx, :]
        deim_params_k = np.array([deim_snapshot_data['parameters'][i] for i in idx])
        if cfg.compute_affine_convection:
            deim_vel_k = homogenize_snapshots(
                deim_vel_k, deim_params_k, u_base, u_control,
            )
        deim_vel_functions_k = dofs_to_functions(deim_vel_k, problem.velocity_space)
        os.makedirs(os.path.dirname(bp), exist_ok=True)
        compute_and_save_basis(
            deim_vel_functions_k,
            problem.velocity_space,
            output_path=bp,
            n_modes_F=cfg.n_modes_F,
            n_modes_J=cfg.n_modes_J,
            snapshot_dir=deim_snapshot_dir,
        )
        del deim_vel_functions_k

    # --- Adaptive DEIM mode selection ---
    deim_data = np.load(bp)
    m_F_k = modes_for_tolerance(deim_data['eigenvalues_F'], cfg.deim_energy_tol, cfg.m_F_max)
    m_J_k = modes_for_tolerance(deim_data['eigenvalues_J'], cfg.deim_energy_tol, cfg.m_J_max)

    # --- DEIM online operators ---
    if cfg.recompute_deim or cfg.recompute_pod:
        rebuild_ops = True
    else:
        rebuild_ops, _ = needs_ops_rebuild(op, bp, m_F_k, m_J_k, n_rom_modes_k)

    if rebuild_ops:
        deim_ops_k = build_deim_online_operators(bp, Z_u_k, m_F_k, m_J_k)
        deim_ops_k.save(op, basis_path=bp)
    else:
        deim_ops_k = DEIMOnlineOperators.load(op)

    return deim_ops_k


def build_local_roms(
    clustering,
    velocity_snapshots,
    pressure_snapshots,
    deim_snapshot_data,
    deim_snapshot_dir,
    lifting,
    problem,
    cfg,
    layout,
):
    """
    Build (or load) all per-cluster local ROMs.

    Iterates over clusters and assembles POD bases, reduced operators,
    and (optionally) DEIM operators. Honors cache validity and the
    recompute flags in `cfg`.

    Returns
    -------
    bases         : dict[int, PODBasis]
    operators     : dict[int, ReducedOperators]
    deim_ops_dict : dict[int, DEIMOnlineOperators or None]
    """
    u_base    = lifting.u_base.dat.data_ro.flatten()
    u_control = lifting.u_control.dat.data_ro.flatten()

    bases         = {}
    operators     = {}
    deim_ops_dict = {}

    for k in range(clustering.n_clusters):
        basis_k, operators_k = build_cluster_pod(
            k, clustering, velocity_snapshots, pressure_snapshots,
            lifting, problem, cfg, layout, u_base, u_control,
        )

        deim_ops_k = build_cluster_deim(
            k, clustering, operators_k,
            deim_snapshot_data, deim_snapshot_dir,
            problem, cfg, layout, u_base, u_control,
        )

        bases[k]         = basis_k
        operators[k]     = operators_k
        deim_ops_dict[k] = deim_ops_k

    return bases, operators, deim_ops_dict

def _rom_solution_summary(self):
    """Print a short summary for a local ROM solve result."""
    cluster = getattr(self, 'cluster', self.metadata.get('cluster', None))
    t_rom = getattr(self, 't_rom', self.metadata.get('t_rom', None))

    if cluster is not None:
        print(f"  Cluster:    {cluster}")
    print(f"  Amplitude:  {self.amplitude}")
    print(f"  Converged:  {self.converged}")
    print(f"  Iterations: {self.iterations}")
    if t_rom is not None:
        print(f"  ROM time:   {t_rom:.3f}s")


def _as_local_rom_solution(
    solution: ROMSolution,
    *,
    cluster: int,
    reconstructed,
    t_rom: float,
    w_rom=None,
    callback_data=None,
) -> ROMSolution:
    """Attach local-ROM solve metadata to a ROMSolution instance.

    This keeps the public return type as ROMSolution while preserving the
    old ROMSolveResult-style attributes used by local-ROM workflows.
    Existing code that consumes plain ROMSolution fields continues to work
    because the core coefficient/convergence fields are unchanged.
    """
    solution.cluster = cluster
    solution.reconstructed = reconstructed
    solution.t_rom = t_rom
    solution.w_rom = w_rom
    solution.callback_data = callback_data

    # Backward-compatible aliases for code that previously consumed
    # ROMSolveResult.u_coeffs / ROMSolveResult.p_coeffs.
    solution.u_coeffs = solution.velocity_coeffs
    solution.p_coeffs = solution.pressure_coeffs

    # Backward-compatible alias for code that previously expected
    # ROMSolveResult.solution to hold the nested ROMSolution.
    solution.solution = solution

    solution.metadata.update({
        'cluster': cluster,
        't_rom': t_rom,
        'has_reconstruction': reconstructed is not None,
        'has_internals': w_rom is not None or callback_data is not None,
    })

    return solution


if not hasattr(ROMSolution, 'summary'):
    ROMSolution.summary = _rom_solution_summary

from nsrom.navier_stokes import ForceCoefficients
@dataclass
class ROMvsFOMComparison:
    """Errors and timings from comparing a ROM solve against a FOM solve."""
    u_err   : float
    p_err   : float
    t_rom   : float
    t_fom   : float
    speedup : float
    rom_forces: ForceCoefficients
    fom_forces: ForceCoefficients

    def summary(self, u_norm_label="energy"):
        print(f"  Velocity error ({u_norm_label}): {self.u_err*100:.4f}%")
        print(f"  Pressure error (L2):           {self.p_err*100:.4f}%")
        print(f"  ROM time:  {self.t_rom:.3f}s")
        print(f"  FOM time:  {self.t_fom:.3f}s")
        print(f"  Speedup:   {self.speedup:.1f}x")
        print(f"  FOM lift:  {self.fom_forces.lift:.6f}")
        print(f"  ROM lift:  {self.rom_forces.lift:.6f}")

def solve_at_parameter(
    Re,
    amp,
    clustering,
    operators,
    deim_ops_dict,
    problem,
    initial_solution,
    M_u,
    M_p,
    use_deim,
    cluster_forced = None,
):
    """
    Solve the local ROM at a single parameter point.

    Selects a cluster in parameter space, projects the initial guess onto
    that cluster's reduced basis, runs the ROM solver (with DEIM if
    enabled), and reconstructs the full-field solution.

    Returns
    -------
    ROMSolution
        The ROM solution augmented with local solve metadata:
        cluster, reconstructed, t_rom, w_rom, callback_data, u_coeffs,
        p_coeffs, and solution/self compatibility aliases.
    """
    problem.reynolds.assign(Re)
    problem.amplitude.assign(amp)
    nu = 1.0 / Re

    # k1, d1= select_cluster(clustering, Re, amp)
    k1, d1 = select_cluster_by_box_then_centroid(clustering, Re ,amp)
    k2, d2 = select_cluster_solution_space(clustering, initial_solution,M_u)
    if cluster_forced is not None:
        k = cluster_forced
        print(f' Forced to use cluster :{cluster_forced}')
    else:
        k =select_cluster_closest(d1,d2)
    velocity_dofs = initial_solution.velocity.dat.data_ro.flatten()
    pressure_dofs = initial_solution.pressure.dat.data_ro.flatten()
    u_coeffs, p_coeffs = project_to_rom_coefficients(
        velocity_dofs, pressure_dofs,
        operators[k],
        amp=amp,
        M_u=M_u,
        M_p=M_p,
    )

    deim_ops_k   = deim_ops_dict[k]
    submesh_deim = None
    if use_deim and deim_ops_k is not None:
        submesh_deim = SubMeshDEIM(problem.velocity_space, deim_ops_k)

    t0 = time.perf_counter()
    solution = solve_rom(
        operators=operators[k],
        nu=nu,
        amp=amp,
        problem=problem,
        deim_ops=deim_ops_k,
        u_initial_guess=u_coeffs,
        p_initial_guess=p_coeffs,
        submesh_deim=submesh_deim,
    )

    t_rom = time.perf_counter() - t0

    reconstructed = reconstruct_solution(solution, operators[k], problem, amp=amp)

    return _as_local_rom_solution(
        solution,
        cluster=k,
        reconstructed=reconstructed,
        t_rom=t_rom,
    )




def solve_at_parameter_with_internals(
    Re,
    amp,
    clustering,
    operators,
    deim_ops_dict,
    problem,
    initial_solution,
    M_u,
    M_p,
    use_deim,
):
    """
    Solve the local ROM at a single parameter point.

    Selects a cluster in parameter space, projects the initial guess onto
    that cluster's reduced basis, runs the ROM solver (with DEIM if
    enabled), and reconstructs the full-field solution.

    Returns
    -------
    ROMSolution
        The ROM solution augmented with local solve metadata and internals:
        cluster, reconstructed, t_rom, w_rom, callback_data, u_coeffs,
        p_coeffs, and solution/self compatibility aliases.
    """
    problem.reynolds.assign(Re)
    problem.amplitude.assign(amp)
    nu = 1.0 / Re

    k1, d1= select_cluster(clustering, Re, amp)
    k2, d2 = select_cluster_solution_space(clustering, initial_solution,M_u)
    k =select_cluster_closest(d1,d2)
    velocity_dofs = initial_solution.velocity.dat.data_ro.flatten()
    pressure_dofs = initial_solution.pressure.dat.data_ro.flatten()
    u_coeffs, p_coeffs = project_to_rom_coefficients(
        velocity_dofs, pressure_dofs,
        operators[k],
        amp=amp,
        M_u=M_u,
        M_p=M_p,
    )

    deim_ops_k   = deim_ops_dict[k]
    submesh_deim = None
    if use_deim and deim_ops_k is not None:
        submesh_deim = SubMeshDEIM(problem.velocity_space, deim_ops_k)

    t0 = time.perf_counter()

    solution, w_rom, callback_data = solve_rom_with_internals(
        operators=operators[k],
        nu=nu,
        amp=amp,
        problem=problem,
        deim_ops=deim_ops_k,
        u_initial_guess=u_coeffs,
        p_initial_guess=p_coeffs,
        submesh_deim=submesh_deim,
    )
    t_rom = time.perf_counter() - t0

    reconstructed = reconstruct_solution(solution, operators[k], problem, amp=amp)

    return _as_local_rom_solution(
        solution,
        cluster=k,
        reconstructed=reconstructed,
        t_rom=t_rom,
        w_rom=w_rom,
        callback_data=callback_data,
    )

def compare_rom_vs_fom(
    rom_result,
    problem,
    initial_solution,
    M_u,
    M_p,
):
    """
    Solve the FOM at the current problem state and compare against the
    reconstructed ROM solution.

    Assumes ``problem.reynolds`` and ``problem.amplitude`` are already
    set (typically by a preceding ``solve_at_parameter`` call). Returns
    relative errors in the velocity inner-product norm and the pressure
    L2 norm, plus FOM timing and speedup.

    Returns
    -------
    ROMvsFOMComparison
    """
    from nsrom.navier_stokes import solve_steady_navier_stokes, compute_forces, ForceCoefficients

    t0 = time.perf_counter()
    hf_solution = solve_steady_navier_stokes(problem, initial_solution.solution)
    t_fom = time.perf_counter() - t0

    def norm_M(v, M):
        return np.sqrt(float(v @ (M @ v)))

    u_fom      = hf_solution.velocity.dat.data_ro.flatten()
    u_rom_dofs = rom_result.reconstructed.velocity.dat.data_ro.flatten()
    p_fom      = hf_solution.pressure.dat.data_ro.flatten()
    p_rom_dofs = rom_result.reconstructed.pressure.dat.data_ro.flatten()

    u_err = norm_M(u_fom - u_rom_dofs, M_u) / norm_M(u_fom, M_u)
    p_err = norm_M(p_fom - p_rom_dofs, M_p) / norm_M(p_fom, M_p)

    fom_forces = compute_forces(hf_solution,problem)
    rom_forces = compute_forces(rom_result.reconstructed,problem)

    return ROMvsFOMComparison(
        u_err=u_err,
        p_err=p_err,
        t_rom=rom_result.t_rom,
        t_fom=t_fom,
        speedup=t_fom / rom_result.t_rom,
        rom_forces = rom_forces,
        fom_forces = fom_forces,
    )