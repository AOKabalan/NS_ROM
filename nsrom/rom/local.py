"""
Local reduced-order model construction.

Per-cluster orchestration of POD bases, reduced operators, and DEIM
operators, plus online utilities (cluster selection, change-of-basis).

This is the canonical ``nsrom.rom.local`` implementation. The legacy
``nsrom.local_rom`` module remains a compatibility shim.

Depends on:
  * nsrom.clustering.building — for ClusteringResult
  * nsrom.rom               — for POD/operator construction
  * nsrom.deim, nsrom.online_operators, nsrom.deim_helpers — for DEIM
"""
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from firedrake import Function  # for type hints only

from nsrom.helper_functions import dofs_to_functions, modes_for_tolerance
from nsrom.navier_stokes import NavierStokesSolution, ForceCoefficients
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
    reconstruct_solution,
    project_to_rom_coefficients,
    SubMeshDEIM,
)
from nsrom.deim import compute_and_save_basis
from nsrom.online_operators import DEIMOnlineOperators, build_deim_online_operators
from nsrom.deim_helpers import needs_basis_recompute, needs_ops_rebuild


# =============================================================================
# LocalROMResult — wraps a ROMSolution with local-solve metadata
# =============================================================================

@dataclass
class LocalROMResult:
    """
    Result of a local-ROM solve at a single parameter point.

    Wraps a ROMSolution together with local-solve metadata:
      - the cluster used,
      - the reconstructed full-order field,
      - the wall-clock solve time,
      - optionally the Firedrake internals (w_rom, callback_data) for
        post-solve work like the bifurcation indicator.

    Read-only properties mirror the underlying ROMSolution fields and
    provide the legacy ``u_coeffs``/``p_coeffs`` aliases used by older
    call sites.
    """
    solution: ROMSolution = field(repr=False)
    cluster: int
    reconstructed: NavierStokesSolution = field(repr=False)
    t_rom: float
    w_rom: Optional[Function] = field(default=None, repr=False)
    callback_data: Optional[dict] = field(default=None, repr=False)

    # ---- Pass-through accessors to the underlying ROMSolution ----
    @property
    def velocity_coeffs(self):
        return self.solution.velocity_coeffs

    @property
    def pressure_coeffs(self):
        return self.solution.pressure_coeffs

    @property
    def u_coeffs(self):
        """Legacy alias for velocity_coeffs."""
        return self.solution.velocity_coeffs

    @property
    def p_coeffs(self):
        """Legacy alias for pressure_coeffs."""
        return self.solution.pressure_coeffs

    @property
    def converged(self):
        return self.solution.converged

    @property
    def iterations(self):
        return self.solution.iterations

    @property
    def amplitude(self):
        return self.solution.amplitude

    @property
    def reynolds(self):
        return self.solution.reynolds

    @property
    def metadata(self):
        return self.solution.metadata

    def summary(self):
        """Print a short summary of the local-ROM solve."""
        print(f"  Cluster:    {self.cluster}")
        print(f"  Amplitude:  {self.amplitude}")
        print(f"  Converged:  {self.converged}")
        print(f"  Iterations: {self.iterations}")
        print(f"  ROM time:   {self.t_rom:.3f}s")


def _make_local_result(
    solution: ROMSolution,
    *,
    cluster: int,
    reconstructed: NavierStokesSolution,
    t_rom: float,
    w_rom: Optional[Function] = None,
    callback_data: Optional[dict] = None,
) -> LocalROMResult:
    """
    Bundle a ROMSolution with local-solve metadata into a LocalROMResult.

    Also writes cluster/t_rom into the underlying ROMSolution.metadata so
    that information survives a pure-ROMSolution save/load cycle.
    """
    solution.metadata.update({
        'cluster': cluster,
        't_rom': t_rom,
        'has_reconstruction': reconstructed is not None,
        'has_internals': w_rom is not None or callback_data is not None,
    })
    return LocalROMResult(
        solution=solution,
        cluster=cluster,
        reconstructed=reconstructed,
        t_rom=t_rom,
        w_rom=w_rom,
        callback_data=callback_data,
    )

def project_initial_solution(
    initial_solution,
    operators,
    amp,
    M_u,
    M_p,
):
    velocity_dofs = initial_solution.velocity.dat.data_ro.flatten()
    pressure_dofs = initial_solution.pressure.dat.data_ro.flatten()

    return project_to_rom_coefficients(
        velocity_dofs,
        pressure_dofs,
        operators,
        amp=amp,
        M_u=M_u,
        M_p=M_p,
    )
# =============================================================================
# Cluster selection
# =============================================================================
#
# Strategies available (the primary path used by `solve_at_parameter` is
# box+centroid in parameter space combined with the M-norm distance in
# solution space):
#
#   select_cluster_by_box_then_centroid(clustering, Re, amp)
#       Primary parameter-space selector. Prefers clusters whose parameter
#       bounding box contains the query; falls back to nearest centroid
#       (normalized) when no box matches. Returns (best, distances) where
#       non-eligible clusters have +inf entries unless we fell back.
#
#   select_cluster(clustering, Re, amp)
#       Simpler parameter-space alternative: nearest centroid by
#       normalized distance, no box test. Used for diagnostics.
#
#   select_cluster_solution_space(clustering, initial_solution, M_u)
#       Solution-space selector. M-norm distance from the initial
#       velocity field to each cluster's centroid in full-DOF space.
#
#   select_cluster_closest(d1, d2)
#       Combine two distance arrays into one ranking. Each input is
#       independently rescaled to [0, 1] so the combination is invariant
#       to the (very different) physical scales of parameter- vs.
#       solution-space distances; the cluster minimizing the Euclidean
#       norm in the rescaled plane is returned.
#
#   precompute_fast_selection(...) + select_cluster_reduced(...)
#       Fast online path for tight continuation loops. Precompute
#       Z_i^T M c_j once, then select in reduced coordinates without
#       expanding to full DOFs.


def select_cluster_by_box_then_centroid(clustering, Re, amp, verbose=True):
    """
    Parameter-space cluster selection with box check + centroid fallback.

    Returns (best, distances). When at least one cluster contains the
    query in its parameter box, `distances` has +inf for non-eligible
    clusters and the rescaled centroid distance for eligible ones.
    When no cluster contains the query, falls back to centroid distance
    over all clusters.
    """
    query = np.array([Re, amp])

    centers = np.array([
        clustering.parameter_centroid(k)
        for k in range(clustering.n_clusters)
    ])

    # Global normalization scale
    scales = np.ptp(centers, axis=0)
    scales[scales == 0] = 1.0

    raw_distances = np.linalg.norm((centers - query) / scales, axis=1)
    distances = np.full(clustering.n_clusters, np.inf)

    eligible = []
    for k in range(clustering.n_clusters):
        params_k = clustering.cluster_parameters(k)
        re_min, amp_min = params_k.min(axis=0)
        re_max, amp_max = params_k.max(axis=0)

        in_box = (re_min <= Re <= re_max) and (amp_min <= amp <= amp_max)

        if in_box:
            eligible.append(k)
            distances[k] = raw_distances[k]

    if eligible:
        best = int(np.argmin(distances))
        mode = "box + centroid"
    else:
        distances = raw_distances.copy()
        best = int(np.argmin(distances))
        mode = "centroid fallback"

    if verbose:
        print(f"  Query (Re={Re}, amp={amp})")
        print(f"  Selection mode: {mode}")
        for k, d in enumerate(distances):
            box_marker = " in-box" if k in eligible else ""
            best_marker = " ←" if k == best else ""
            print(f"  Cluster {k}: distance={d:.4f}{box_marker}{best_marker}")

    return best, distances


def select_cluster(clustering, Re, amp):
    """
    Simple parameter-space selector: nearest centroid by normalized
    distance, no box check.

    Used for diagnostics and as a quieter alternative to
    `select_cluster_by_box_then_centroid`.
    """
    query = np.array([Re, amp])
    centers = np.array([
        clustering.parameter_centroid(k)
        for k in range(clustering.n_clusters)
    ])
    scales = np.ptp(centers, axis=0)
    scales[scales == 0] = 1.0
    distances = np.linalg.norm((centers - query) / scales, axis=1)
    best = int(np.argmin(distances))
    return best


def select_cluster_solution_space(clustering, initial_solution, M_u):
    """
    Solution-space cluster selection.

    Computes the M-norm distance from the initial velocity field to each
    cluster's centroid in full-DOF space and picks the closest.
    """
    K = clustering.n_clusters
    centroids = clustering.centroids_original
    u_dof = initial_solution.velocity.dat.data_ro.flatten()

    distances = np.zeros(K)
    for j in range(K):
        diff = u_dof - centroids[:, j]
        distances[j] = np.sqrt(diff.T @ M_u @ diff)
    best = int(np.argmin(distances))

    # for k, d in enumerate(distances):
    #     best_marker = " ←" if k == best else ""
    #     print(f"  Cluster {k}: distance={d:.4f}{best_marker}")




    best = int(np.argmin(distances))
    return best, distances


def select_cluster_closest(d1, d2):
    """
    Combine two distance arrays and pick the minimum,
    ignoring clusters with +inf or NaN in either distance array.
    """
    d1 = np.asarray(d1, dtype=float)
    d2 = np.asarray(d2, dtype=float)

    # Only clusters with finite distances in both arrays are selectable
    finite = np.isfinite(d1) & np.isfinite(d2)

    if not np.any(finite):
        raise ValueError("No finite clusters available for selection.")

    def _normalize_finite(d, mask):
        out = np.full_like(d, np.inf, dtype=float)

        vals = d[mask]
        spread = vals.max() - vals.min()

        if spread < 1e-15:
            out[mask] = 0.0
        else:
            out[mask] = (vals - vals.min()) / spread

        return out

    # d1n = _normalize_finite(d1, finite)
    # d2n = _normalize_finite(d2, finite)

    score = np.full_like(d1, np.inf, dtype=float)
    score[finite] = np.sqrt(d1[finite] ** 2 + d2[finite] ** 2)
    # score[finite] = np.sqrt(d1n[finite] ** 2 + d2n[finite] ** 2)

    return int(np.argmin(score))

def select_the_cluster(clustering, Re, amp, initial_solution, M_u, cluster_forced = None, verbose = False):

    # ---- Cluster selection ----
    k1, d1 = select_cluster_by_box_then_centroid(
        clustering, Re, amp, verbose=verbose,
    )
    k2, d2 = select_cluster_solution_space(clustering, initial_solution, M_u)

    if verbose:
        print(f"  Closest in parameter space: {k1}")
        print(f"  Closest in solution space:  {k2}")

    if cluster_forced is not None:
        k = cluster_forced
        if verbose:
            print(f"  Forced to use cluster: {cluster_forced}")
    else:
        k = select_cluster_closest(d1, d2)
        if verbose:
            print(f"  Selected cluster:      {k}")
    return k

def precompute_fast_selection(clustering, operators, M_u, verbose=True):
    """
    Precompute data for cluster selection in whitened global-POD space —
    the SAME space the pod_whitened k-means clustering used.

    Requires (from method='pod_whitened'):
        clustering.pod_modes   : (n_dofs, rank)  global POD basis Phi, M_u-orthonormal
        clustering.pod_sigma   : (rank,)         per-mode sigma = sqrt(lambda)
        clustering.centers_pod : (K, rank)       centroids in RAW POD coords

    For each cluster i, stores the lift-to-global-POD operator
        G[i] = Phi^T M_u Z_i          (rank x n_modes_i)
    so that online the global POD coords of a state Z_i a are just G[i] @ a.
    The centroids are pre-whitened once: c_white = centers_pod / sigma.

    Online distance:  d_j = || (G[k_prev] @ a)/sigma  -  c_white_j ||.
    """
    for attr in ("pod_modes", "pod_sigma", "centers_pod"):
        if getattr(clustering, attr, None) is None:
            raise AttributeError(
                f"clustering.{attr} missing — whitened selection needs the "
                "pod_whitened artifacts. Recompute the clustering "
                "(recompute_clustering=True) or persist them in save/load."
            )

    Phi   = clustering.pod_modes          # (n_dofs, rank), M_u-orthonormal
    sigma = clustering.pod_sigma          # (rank,)
    K     = clustering.n_clusters

    G = {i: Phi.T @ (M_u @ operators[i].Z_u) for i in range(K)}   # (rank x n_modes_i)
    centers_white = clustering.centers_pod / sigma[None, :]       # (K, rank)

    clustering.fast_selection = {
        'G': G,
        'sigma': sigma,
        'centers_white': centers_white,
    }

    if verbose:
        print(f"  Fast selection precomputed (whitened POD): "
              f"{K} clusters, rank={Phi.shape[1]}")


def _whitened_pod_distances(clustering, a_prev, k_prev):
    """
    Distance from the state (given as local coeffs a_prev in cluster k_prev)
    to every centroid, measured in the whitened global-POD space used by
    k-means. Returns (K,) array of distances.
    """
    fs = clustering.fast_selection
    g  = (fs['G'][k_prev] @ a_prev) / fs['sigma']            # whitened state coords
    return np.linalg.norm(fs['centers_white'] - g[None, :], axis=1)


def select_cluster_reduced(clustering, a_prev, k_prev, verbose=True):
    """
    Select cluster using reduced coordinates (no DOF expansion).

    Parameters
    ----------
    clustering : ClusteringResult with fast_selection populated
    a_prev     : (n_modes,) reduced velocity coefficients from previous solve
    k_prev     : int, cluster index of previous solve
    verbose    : print per-cluster distances

    Returns
    -------
    best : int, selected cluster index
    """
    fs = clustering.fast_selection
    K = clustering.n_clusters

    distances = _whitened_pod_distances(clustering, a_prev, k_prev)  # (K,) whitened POD

    best = int(np.argmin(distances))

    if verbose:
        print(f"  Reduced selection (from cluster {k_prev}):")
        for k in range(K):
            marker = " ←" if k == best else ""
            print(f"    Cluster {k}: d_white = {distances[k]:.4f}{marker}")
    return best


# =============================================================================
# Change of basis
# =============================================================================

def precompute_change_of_basis(clustering, operators, M_u, M_p, verbose=True):
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

    if verbose:
        n_pairs = K * (K - 1)
        print(f"  Change-of-basis precomputed: {n_pairs} pairs, "
              f"{n_lifting} lifting functions")


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
    M_u,
):
    """
    Build (or load) the DEIM basis and online operators for cluster k.


    Returns
    -------
    deim_ops_k : DEIMOnlineOperators or None
        None unless cfg.mode == 'deim'.
    """
    if cfg.mode != 'deim':
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
        # if cfg.compute_affine_convection:
        #     deim_vel_k = homogenize_snapshots(
        #         deim_vel_k, deim_params_k, u_base, u_control,
        #     )
        if cfg.compute_affine_convection:
            deim_vel_k = homogenize_snapshots(deim_vel_k, deim_params_k, u_base, u_control)
            # NEW: evaluate F on the POD-truncated states the online solver visits,
            # not the raw fluctuations (kills the ~sqrt(pod_tol) tail-contamination floor)
            # A = Z_u_k.T @ (M_u @ deim_vel_k.T)      # (r, n_snaps)  M_u-orthonormal projection
            # deim_vel_k = (Z_u_k @ A).T
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
    m_F_k = modes_for_tolerance(deim_data['eigenvalues_F'], cfg.deim_energy_tol_F, cfg.m_F_max)
    m_J_k = modes_for_tolerance(deim_data['eigenvalues_J'], cfg.deim_energy_tol_J, cfg.m_J_max)

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
    M_u,
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
            problem, cfg, layout, u_base, u_control,M_u,
        )

        bases[k]         = basis_k
        operators[k]     = operators_k
        deim_ops_dict[k] = deim_ops_k

    return bases, operators, deim_ops_dict


# =============================================================================
# ROM vs FOM comparison
# =============================================================================

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


# =============================================================================
# Solve entry points
# =============================================================================
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
    mode,
    cluster_forced=None,
    return_internals: bool = False,
    verbose: bool = True,
) -> LocalROMResult:
    """
    Solve the local ROM at a single (Re, amp).

    Selects a cluster (parameter box + solution-space distance, unless
    forced), projects the initial guess onto that cluster's reduced
    basis, runs the ROM solver in the requested `mode`, and reconstructs
    the full-field solution.

    Parameters
    ----------
    mode : {'exact', 'deim', 'tensor'}
        Convection evaluation strategy, passed straight through to
        solve_rom. 'deim' requires deim_ops_dict[k] to be populated;
        'tensor' requires operators[k].tensor_convection.
    cluster_forced : int or None
        If provided, bypass cluster selection and use this cluster.
    return_internals : bool, default False
        If True, the returned LocalROMResult also carries the Firedrake
        internals (w_rom, callback_data) for post-solve operations such
        as the bifurcation indicator.
    verbose : bool, default True
        Pass False inside continuation/sweep loops to silence per-call
        cluster-selection and solver output.

    Returns
    -------
    LocalROMResult
    """
    problem.reynolds.assign(Re)
    problem.amplitude.assign(amp)
    nu = 1.0 / Re

    k = select_the_cluster(
        clustering, Re, amp, initial_solution, M_u,
        cluster_forced=cluster_forced, verbose=verbose,
    )

    # Project the FOM initial guess onto cluster k's basis.
    u_coeffs, p_coeffs = project_initial_solution(
        initial_solution,
        operators[k],
        amp,
        M_u,
        M_p,
    )

    # ---- DEIM handles (only in DEIM mode) ----
    deim_ops_k = deim_ops_dict.get(k) if mode == 'deim' else None
    submesh_deim = None
    if mode == 'deim':
        if deim_ops_k is None:
            raise ValueError(
                f"mode='deim' but deim_ops_dict[{k}] is None. "
                "Build the DEIM operators for this cluster first."
            )
        submesh_deim = SubMeshDEIM(problem.velocity_space, deim_ops_k)

    # ---- Solve ----
    t0 = time.perf_counter()
    rom_output = solve_rom(
        operators=operators[k],
        nu=nu,
        amp=amp,
        problem=problem,
        mode=mode,
        deim_ops=deim_ops_k,
        submesh_deim=submesh_deim,
        u_initial_guess=u_coeffs,
        p_initial_guess=p_coeffs,
        return_internals=return_internals,
        verbose=verbose,
    )
    t_rom = time.perf_counter() - t0

    if return_internals:
        solution, w_rom, callback_data = rom_output
    else:
        solution = rom_output
        w_rom = None
        callback_data = None

    # ---- Reconstruct full-order solution ----
    reconstructed = reconstruct_solution(solution, operators[k], problem, amp=amp)

    return _make_local_result(
        solution,
        cluster=k,
        reconstructed=reconstructed,
        t_rom=t_rom,
        w_rom=w_rom,
        callback_data=callback_data,
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
    mode,
    cluster_forced=None,
    verbose: bool = True,
) -> LocalROMResult:
    """
    Deprecated. Prefer ``solve_at_parameter(..., return_internals=True)``.

    Kept as a thin shim while call sites migrate.
    """
    return solve_at_parameter(
        Re, amp, clustering, operators, deim_ops_dict, problem,
        initial_solution, M_u, M_p, mode,
        cluster_forced=cluster_forced,
        return_internals=True,
        verbose=verbose,
    )



def compare_rom_vs_fom(
    rom_result: LocalROMResult,
    problem,
    initial_solution,
    M_u,
    M_p,
) -> ROMvsFOMComparison:
    """
    Solve the FOM at the current problem state and compare against the
    reconstructed ROM solution.

    Assumes ``problem.reynolds`` and ``problem.amplitude`` are already
    set (typically by a preceding ``solve_at_parameter`` call). Returns
    relative errors in the velocity inner-product norm and the pressure
    L2 norm, plus FOM timing and speedup.
    """
    from nsrom.navier_stokes import solve_steady_navier_stokes, compute_forces

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

    fom_forces = compute_forces(hf_solution, problem)
    rom_forces = compute_forces(rom_result.reconstructed, problem)

    return ROMvsFOMComparison(
        u_err=u_err,
        p_err=p_err,
        t_rom=rom_result.t_rom,
        t_fom=t_fom,
        speedup=t_fom / rom_result.t_rom,
        rom_forces=rom_forces,
        fom_forces=fom_forces,
    )




def _apply_cb(change_of_basis, src, tgt, vec):
    """
    Return (Z_tgt^T M Z_src) @ vec using the precomputed CB_u blocks.

    CB_u[(src, tgt)] = Z_tgt^T M Z_src.  On the diagonal Z^T M Z = I, so
    no block is stored and the vector passes through unchanged.
    """
    if src == tgt:
        return vec
    return change_of_basis['CB_u'][(src, tgt)] @ vec


def _nearest_centroid(clustering, candidates, Re, amp):
    """Nearest parameter centroid among a subset of cluster indices."""
    query = np.array([Re, amp])
    centers = np.array([
        clustering.parameter_centroid(k)
        for k in range(clustering.n_clusters)
    ])
    scales = np.ptp(centers, axis=0)
    scales[scales == 0] = 1.0
    d = np.linalg.norm((centers - query) / scales, axis=1)
    candidates = np.asarray(candidates)
    return int(candidates[int(np.argmin(d[candidates]))])


def select_cluster_predicted(
    clustering,
    a_cur,
    k_cur,
    a_old,
    k_old,
    *,
    Re=None,
    amp=None,
    tau=0.8,
    eps_tie=1e-2,
    verbose=False,
):
    """
    Best-approximation cluster selection on a secant-predicted state.

    Predicts the next homogeneous velocity fluctuation by a secant
    extrapolation of the last two solved reduced states,

        u_pred = 2 Z_{k_cur} a_cur - Z_{k_old} a_old,

    and selects the cluster whose POD space best represents it, i.e.
    minimizes the M-orthogonal projection loss

        loss_k = || u_pred - P_k u_pred ||_M^2 / || u_pred ||_M^2
               = 1 - || Z_k^T M u_pred ||^2 / || u_pred ||_M^2.

    Everything is evaluated in reduced coordinates via the precomputed
    change-of-basis blocks CB_u[(i,k)] = Z_k^T M Z_i (identity on the
    diagonal). No lifting enters: bases, centroids and reduced
    coordinates all live in the homogeneous space, and for uniform
    parameter stepping the lifting correction to the secant is zero.

    Decision rule (weight-free, lexicographic):
      1. rank clusters by projection loss;
      2. hysteresis: leave the incumbent k_cur only if the best
         cluster's loss is below tau * loss[k_cur];
      3. tie-break among near-best clusters (within eps_tie of the
         minimum loss) by preferring the incumbent, then the nearest
         parameter centroid.

    Note: if the last two states were in the same cluster, u_pred lies
    in that cluster's span, loss[k_cur] = 0, and the rule stays put. The
    cross-cluster question only opens up when a_cur and a_old were in
    different clusters (i.e. you straddled a boundary).

    Parameters
    ----------
    a_cur, k_cur : ndarray, int
        Most recent solved reduced velocity coefficients and cluster.
    a_old, k_old : ndarray, int
        The solved state one step earlier.
    Re, amp : float, optional
        Only used for the centroid tie-break.
    tau : float
        Hysteresis ratio in (0, 1]. Lower = stickier.
    eps_tie : float
        Loss-fraction tolerance for declaring a near-tie.

    Returns
    -------
    int
        Selected cluster index.
    """
    cb = clustering.change_of_basis
    if cb is None:
        raise ValueError(
            "select_cluster_predicted requires precompute_change_of_basis()."
        )

    K = clustering.n_clusters

    # --- norm of the predicted state (loss-ratio denominator) ---
    norm_cur_sq = float(a_cur @ a_cur)
    norm_old_sq = float(a_old @ a_old)
    # cross = a_cur^T (Z_{k_cur}^T M Z_{k_old}) a_old
    cross = float(a_cur @ _apply_cb(cb, k_old, k_cur, a_old))
    norm_pred_sq = 4.0 * norm_cur_sq - 4.0 * cross + norm_old_sq
    norm_pred_sq = max(norm_pred_sq, 1e-30)

    # --- projection loss per cluster ---
    loss = np.empty(K)
    for k in range(K):
        b_k = 2.0 * _apply_cb(cb, k_cur, k, a_cur) - _apply_cb(cb, k_old, k, a_old)
        capture_k = float(b_k @ b_k)
        loss[k] = (norm_pred_sq - capture_k) / norm_pred_sq

    loss = np.clip(loss, 0.0, 1.0)

    k_best = int(np.argmin(loss))
    for i in range(K):
        print(f' Loss Cluster {i} : {loss[i]} {'<- ' if i == k_best else ''}')

    # --- hysteresis + lexicographic tie-break ---
    if loss[k_best] < tau * loss[k_cur]:
        near = np.where(loss <= loss[k_best] + eps_tie)[0]
        if k_cur in near:
            chosen = int(k_cur)
        elif Re is not None and amp is not None:
            chosen = _nearest_centroid(clustering, near, Re, amp)
        else:
            chosen = int(k_best)
    else:
        chosen = int(k_cur)

    if verbose:
        print(f"  Predicted-state selection (incumbent k={k_cur}):")
        for k in range(K):
            mark = " ←" if k == chosen else (" *" if k == k_best else "")
            print(f"    Cluster {k}: loss={loss[k]:.4e}{mark}")
        if chosen == k_cur and k_best != k_cur:
            print(f"    (held incumbent k={k_cur} by hysteresis; "
                  f"best was k={k_best}, loss={loss[k_best]:.4e})")

    return chosen

def accept_switch_by_projection(
    clustering,
    a_cur,
    k_cur,
    a_old,
    k_old,
    k_candidate,
    *,
    rel_proj_tol=1e-3,
    verbose=False,
):
    cb = clustering.change_of_basis
    if cb is None:
        raise ValueError("Requires precompute_change_of_basis().")

    if k_candidate == k_cur:
        return k_cur

    a_cur = np.asarray(a_cur, dtype=float)
    a_old = np.asarray(a_old, dtype=float)

    # Norm of secant-predicted state:
    # u_pred = 2 Z_cur a_cur - Z_old a_old
    norm_cur_sq = float(a_cur @ a_cur)
    norm_old_sq = float(a_old @ a_old)
    cross = float(a_cur @ _apply_cb(cb, k_old, k_cur, a_old))

    norm_pred_sq = (
        4.0 * norm_cur_sq
        - 4.0 * cross
        + norm_old_sq
    )
    norm_pred_sq = max(norm_pred_sq, 1e-30)

    # Projection onto candidate cluster
    b = (
        2.0 * _apply_cb(cb, k_cur, k_candidate, a_cur)
        -     _apply_cb(cb, k_old, k_candidate, a_old)
    )

    capture = float(b @ b)
    rel_loss = max(0.0, min(1.0, (norm_pred_sq - capture) / norm_pred_sq))

    if verbose:
        print(
            f"  Switch check {k_cur} -> {k_candidate}: "
            f"rel_proj_loss={rel_loss:.3e}, tol={rel_proj_tol:.3e}"
        )

    if rel_loss < rel_proj_tol:
        return int(k_candidate)

    return int(k_cur)

# =============================================================================
# Diagnostics
# =============================================================================
# Relies on `np` and `_apply_cb`, both already defined in local_rom.py.
# Does NO solving: reduced distance and projection loss are cheap reduced
# quantities computed here; the eigenvalue `mu` (and optional convergence
# info) must be computed by the caller (the sweep force-solves each cluster)
# and passed in.

def diagnose_clusters(
    clustering,
    a_cur,
    k_cur,
    *,
    a_old=None,
    k_old=None,
    mu=None,
    converged=None,
    iterations=None,
    Re=None,
    amp=None,
    verbose=True,
):
    """
    Print, per cluster:
      1. reduced solution-space distance ||Z_{k_cur} a_cur - c_k||_M
         (needs precompute_fast_selection);
      2. projection loss onto cluster k's basis, current-state and
         (if a_old/k_old given) secant-predicted
         (needs precompute_change_of_basis);
      3. the reduced eigenvalue mu_k, if the caller passes it in.

    Parameters
    ----------
    a_cur, k_cur : current reduced velocity coeffs and their cluster.
    a_old, k_old : optional one-step-back state -> adds the predicted column.
    mu, converged, iterations : optional length-K arrays from the caller's
        per-cluster solves; displayed only, never computed here.
    Re, amp : optional, header only.

    Returns
    -------
    dict with 'reduced_dist', 'proj_loss_cur', 'proj_loss_pred' (length-K),
    echoing back any 'mu'/'converged'/'iterations' passed in.
    """
    K = clustering.n_clusters
    a_cur = np.asarray(a_cur, dtype=float)

    reduced_dist   = np.full(K, np.nan)
    proj_loss_cur  = np.full(K, np.nan)
    proj_loss_pred = np.full(K, np.nan)

    # 1. reduced solution-space distance to each centroid (whitened global-POD)
    fs = getattr(clustering, 'fast_selection', None)
    if fs is not None:
        reduced_dist = _whitened_pod_distances(clustering, a_cur, k_cur)
    elif verbose:
        print("  [diag] fast_selection not set -> skipping reduced distance.")

    # 2. projection loss onto each cluster basis
    cb = getattr(clustering, 'change_of_basis', None)
    if cb is not None:
        norm_cur_sq = max(float(a_cur @ a_cur), 1e-30)
        for k in range(K):
            b = _apply_cb(cb, k_cur, k, a_cur)
            proj_loss_cur[k] = max(0.0, min(1.0, 1.0 - float(b @ b) / norm_cur_sq))
        if a_old is not None and k_old is not None:
            a_old = np.asarray(a_old, dtype=float)
            norm_old_sq = float(a_old @ a_old)
            cross = float(a_cur @ _apply_cb(cb, k_old, k_cur, a_old))
            norm_pred_sq = max(4.0 * norm_cur_sq - 4.0 * cross + norm_old_sq, 1e-30)
            for k in range(K):
                b = 2.0 * _apply_cb(cb, k_cur, k, a_cur) - _apply_cb(cb, k_old, k, a_old)
                proj_loss_pred[k] = max(0.0, min(1.0, (norm_pred_sq - float(b @ b)) / norm_pred_sq))
    elif verbose:
        print("  [diag] change_of_basis not set -> skipping projection loss.")

    out = {
        'reduced_dist':   reduced_dist,
        'proj_loss_cur':  proj_loss_cur,
        'proj_loss_pred': proj_loss_pred,
        'mu':             None if mu is None else np.asarray(mu, dtype=float),
        'converged':      converged,
        'iterations':     iterations,
    }

    if verbose:
        have_pred = a_old is not None and k_old is not None
        have_mu   = mu is not None

        def _argmin_nan(x):
            return int(np.nanargmin(x)) if not np.all(np.isnan(x)) else -1

        best_dist = _argmin_nan(reduced_dist)
        best_cur  = _argmin_nan(proj_loss_cur)
        best_pred = _argmin_nan(proj_loss_pred) if have_pred else -1

        loc = f"  (Re={Re}, amp={amp})" if (Re is not None and amp is not None) else ""
        print(f"\nCluster diagnostics{loc}  incumbent k_cur={k_cur}")
        hdr = f"  {'k':>2} | {'red_dist':>10} | {'proj_loss':>11}"
        if have_pred:
            hdr += f" | {'proj_pred':>11}"
        if have_mu:
            hdr += f" | {'mu':>12} | {'conv':>4} | {'iters':>5}"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for k in range(K):
            row = (f"  {k:>2} | {reduced_dist[k]:>10.4f}"
                   f"{'*' if k == best_dist else ' '}"
                   f"| {proj_loss_cur[k]:>10.3e}"
                   f"{'*' if k == best_cur else ' '}")
            if have_pred:
                row += (f"| {proj_loss_pred[k]:>10.3e}"
                        f"{'*' if k == best_pred else ' '}")
            if have_mu:
                cval = converged[k] if converged is not None else None
                cstr = ('yes' if cval else 'no') if cval is not None else '-'
                istr = f"{iterations[k]}" if iterations is not None else '-'
                row += f"| {out['mu'][k]:>12.4e} | {cstr:>4} | {istr:>5}"
            row += "  <- incumbent" if k == k_cur else ""
            print(row)
        print("  (* = best in that column)\n")

    return out

def select_cluster_reduced_hysteresis(clustering, a_prev, k_prev, tau=0.8, verbose=False):
    K = clustering.n_clusters
    d = _whitened_pod_distances(clustering, a_prev, k_prev)   # whitened global-POD
    k_best = int(np.argmin(d))
    chosen = k_best if d[k_best] < tau * d[k_prev] else k_prev   # switch only if clearly closer
    if verbose:
        for k in range(K):
            mark = " <-" if k == chosen else (" *" if k == k_best else "")
            print(f"   cluster {k}: red_dist={d[k]:.4f}{mark}")


    return chosen



def select_cluster_reduced_projection(clustering, a_prev, k_prev, a_old=None, k_old=None,
                                      tau=0.8, proj_tol=0.01, verbose=False):
    K  = clustering.n_clusters
    cb = getattr(clustering, 'change_of_basis', None)

    # 1. proposer: whitened global-POD distance to each centroid
    d = _whitened_pod_distances(clustering, a_prev, k_prev)

    # 2. validator: projection loss of current & secant-predicted state onto each basis
    proj_loss_cur  = np.full(K, np.nan)
    proj_loss_pred = np.full(K, np.nan)
    if cb is not None:
        norm_cur_sq = max(float(a_prev @ a_prev), 1e-30)
        for k in range(K):
            b = _apply_cb(cb, k_prev, k, a_prev)
            proj_loss_cur[k] = max(0.0, min(1.0, 1.0 - float(b @ b) / norm_cur_sq))
        if a_old is not None and k_old is not None:
            a_old = np.asarray(a_old, dtype=float)
            a_pred = 2.0 * a_prev - _apply_cb(cb, k_old, k_prev, a_old)   # secant, in k_prev frame
            norm_pred_sq = max(float(a_pred @ a_pred), 1e-30)
            for k in range(K):
                b = _apply_cb(cb, k_prev, k, a_pred)
                proj_loss_pred[k] = max(0.0, min(1.0, 1.0 - float(b @ b) / norm_pred_sq))

    # 3. veto: a NON-incumbent basis that cannot represent the (predicted) state is ineligible.
    #    The incumbent is never vetoed -- its self-projection is 0 by construction, and it
    #    stays as the guaranteed fallback (hysteresis holds us there unless clearly beaten).
    veto = proj_loss_pred if (a_old is not None and k_old is not None) else proj_loss_cur
    d_eff = d.copy()
    if cb is not None:
        for k in range(K):
            if k != k_prev and np.isfinite(veto[k]) and veto[k] > proj_tol:
                d_eff[k] = np.inf

    # 4. propose nearest ELIGIBLE centroid, with hysteresis toward the incumbent
    k_best = int(np.argmin(d_eff))
    chosen = k_best if d_eff[k_best] < tau * d_eff[k_prev] else k_prev

    if verbose:
        for k in range(K):
            mark  = " <-" if k == chosen else (" *" if k == k_best else "")
            extra = " VETO" if np.isinf(d_eff[k]) else ""
            print(f"   cluster {k}: red_dist={d[k]:.4f} "
                  f"proj_cur={proj_loss_cur[k]:.4f} proj_pred={proj_loss_pred[k]:.4f}{mark}{extra}")
    return chosen


def classify_state_for_seed(clustering, a_state, k_source, proj_tol=0.01, verbose=False):
    """
    Classify a freshly-spawned (kicked) state -- given as reduced velocity coeffs
    a_state expressed in cluster k_source's basis -- onto the cluster whose basis
    best represents it.

    The source frame is EXCLUDED from candidacy: projecting a_state from k_source
    back onto k_source is the identity (loss 0), a tautology that carries no
    information (this is the round-trip degeneracy). Among the remaining clusters
    the projection loss is an honest subspace-overlap measure. We keep the
    clusters that represent the state within proj_tol and take the nearest in the
    whitened-POD metric -- the same proposer/validator rule as the sweep selector,
    minus the incumbent (there is no prior branch identity to preserve at a birth).
    """
    K  = clustering.n_clusters
    cb = getattr(clustering, 'change_of_basis', None)
    if cb is None:
        raise RuntimeError("change_of_basis not precomputed; needed for seed classification")

    a_state = np.asarray(a_state, dtype=float)
    norm_sq = max(float(a_state @ a_state), 1e-30)

    d = _whitened_pod_distances(clustering, a_state, k_source)        # proposer

    proj_loss = np.full(K, np.inf)
    for k in range(K):
        if k == k_source:
            continue                       # exclude source: self-projection is tautological (0)
        b = _apply_cb(cb, k_source, k, a_state)
        proj_loss[k] = max(0.0, min(1.0, 1.0 - float(b @ b) / norm_sq))

    candidates = [k for k in range(K) if k != k_source]
    eligible   = [k for k in candidates if proj_loss[k] <= proj_tol]
    if eligible:
        k_seed = min(eligible, key=lambda k: d[k])            # nearest well-representing cluster
    else:
        k_seed = min(candidates, key=lambda k: proj_loss[k])  # fallback: best representation

    if verbose:
        print(f"  [seed] kicked state framed in cluster {k_source} -> candidates:")
        for k in candidates:
            tag  = " <-" if k == k_seed else ""
            flag = " (veto)" if proj_loss[k] > proj_tol else ""
            print(f"         cluster {k}: dist={d[k]:.4f} proj={proj_loss[k]:.4f}{flag}{tag}")
    return k_seed
