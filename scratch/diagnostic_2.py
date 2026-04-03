"""
Multi-Parameter Bifurcation Mode Diagnostics
==============================================

For the fluidic pinball with TWO bifurcation parameters (Re, amplitude),
where:
  - Both Re and amplitude affect the bifurcation point
  - The lift coefficient can change sign within a branch as
    amplitude varies (so sign(C_L) does NOT label branches)
  - The bifurcation boundary is a CURVE in (Re, A) space, not a point

This script avoids assuming branch labels are known a priori. Instead,
it characterises each POD mode by:

  1. Smooth vs non-smooth behaviour across the parameter space
  2. Sensitivity to local bifurcation structure (via residuals)
  3. How much information each mode carries beyond smooth trends
  4. Whether whitening amplifies the non-smooth (bifurcation) content

The core idea:
  - Smooth parametric modes can be predicted from (Re, A) by a
    low-order polynomial. Their prediction residual is small.
  - Branch/bifurcation modes have sharp jumps, folds, or multi-valuedness
    that a smooth polynomial CANNOT capture. Their residual is large.
  - The residual-to-total-variance ratio tells us how "bifurcation-like"
    each mode is — WITHOUT needing to know the branch labels.

Diagnostics:
  1. Smooth polynomial fit residual per mode
  2. Correlation with force observables (C_L, |C_L|, C_D)
  3. Local variance analysis (nearby parameters, different coefficients?)
  4. Fisher discriminant ratio using Mahalanobis proxy clusters
  5. Distance contribution per mode before/after whitening
  6. Automatic mode classification
  7. Whitening amplification summary
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import rcParams

from nsrom.navier_stokes import (
    setup_navier_stokes_problem,
    NavierStokesSolution,
    compute_forces,
)
from nsrom.snapshot_collection import load_snapshot_dofs
from nsrom.lifting_functions import load_lifting_functions

from rom import (
    assemble_inner_product_matrix,
    homogenize_snapshots,
    compute_pod_modes,
    project_to_basis,
)
from nsrom.cluster_building import cluster_kmeans
from firedrake import Function

# =============================================================================
# CONFIG
# =============================================================================
# Professional styling
rcParams.update({
    'font.size': 10,
    'axes.linewidth': 1.2,
    'lines.linewidth': 1.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

SNAPSHOT_DIR = "two_param_snapshots_pod_dofs"
LIFTING_DIR    = "lifting"
MESH_FILE      = "mesh/mid_pinball.msh"
REYNOLDS_INIT  = 100.0
AMPLITUDE_INIT = 0.0

INNER_PRODUCT_TYPE = "H1"
POD_RANK = 10
HOMO = False

# Polynomial degree for smooth fit (2 = quadratic in Re, A)
POLY_DEGREE = 3

# Number of Mahalanobis clusters to use as proxy branch labels
N_PROXY_CLUSTERS = 2


# =============================================================================
# HELPERS
# =============================================================================

def section(title):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")


def compute_lift_drag(velocity_snapshots, pressure_snapshots,
                      parameters, problem):
    """Compute C_L and C_D for every snapshot."""
    W = problem.mixed_space
    n = len(parameters)
    lift = np.zeros(n)
    drag = np.zeros(n)

    for i in range(n):
        w = Function(W)
        u, p = w.subfunctions
        u.dat.data[:] = velocity_snapshots[i].reshape(u.dat.data.shape)
        p.dat.data[:] = pressure_snapshots[i].reshape(p.dat.data.shape)

        sol = NavierStokesSolution(
            solution=w, velocity=u, pressure=p,
            reynolds=parameters[i, 0], amplitude=parameters[i, 1],
        )
        forces = compute_forces(sol, problem)
        lift[i] = forces.lift
        drag[i] = forces.drag

    return lift, drag


def build_polynomial_features(parameters, degree):
    """
    Build polynomial features up to given degree in (Re, A).
    E.g. degree=3: [1, Re, A, Re^2, Re*A, A^2, Re^3, Re^2*A, Re*A^2, A^3]
    """
    Re = parameters[:, 0]
    A = parameters[:, 1]
    features = []
    for p in range(degree + 1):
        for q in range(p + 1):
            features.append((Re ** (p - q)) * (A ** q))
    return np.column_stack(features)


def fit_smooth_and_residual(a, parameters, degree):
    """
    Fit a polynomial of given degree to POD coefficient a(Re, A).
    Return: predicted (smooth part), residual (non-smooth part), R² score.
    """
    X = build_polynomial_features(parameters, degree)
    coeffs, _, _, _ = np.linalg.lstsq(X, a, rcond=None)
    a_smooth = X @ coeffs
    a_residual = a - a_smooth

    ss_res = np.sum(a_residual ** 2)
    ss_tot = np.sum((a - a.mean()) ** 2) + 1e-30
    r2 = 1.0 - ss_res / ss_tot

    return a_smooth, a_residual, r2


def local_variance(A, parameters, n_neighbors=8):
    """
    For each snapshot, compute variance of POD coefficients among
    its k nearest neighbours in parameter space.

    High local variance → nearby parameters give very different
    solutions → we are near a bifurcation boundary.
    """
    from sklearn.neighbors import NearestNeighbors

    params_norm = (parameters - parameters.mean(axis=0)) / \
                  (parameters.std(axis=0) + 1e-12)
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(params_norm)
    _, indices = nn.kneighbors(params_norm)

    n_modes, n_snap = A.shape
    loc_var = np.zeros((n_modes, n_snap))

    for i in range(n_snap):
        nbr = indices[i]
        for m in range(n_modes):
            loc_var[m, i] = np.var(A[m, nbr])

    return loc_var


def fisher_discriminant_ratio(a, labels):
    """Fisher J = between-class variance / within-class variance."""
    classes = np.unique(labels)
    grand_mean = a.mean()
    between, within = 0.0, 0.0
    for c in classes:
        mask = labels == c
        nc = np.sum(mask)
        if nc < 2:
            continue
        mu_c = a[mask].mean()
        between += nc * (mu_c - grand_mean) ** 2
        within += np.sum((a[mask] - mu_c) ** 2)
    return between / (within + 1e-30)


def sarle_bimodality(a):
    """Sarle's bimodality coefficient. BC > 5/9 suggests bimodality."""
    m = a.mean()
    m2 = np.mean((a - m) ** 2)
    m3 = np.mean((a - m) ** 3)
    m4 = np.mean((a - m) ** 4)
    skew = m3 / (m2 ** 1.5 + 1e-30)
    kurt = m4 / (m2 ** 2 + 1e-30) - 3
    return (skew ** 2 + 1) / (kurt + 3 + 1e-30)


# =============================================================================
# MAIN
# =============================================================================

def main():
    # --- Setup ---
    section("SETUP")

    problem = setup_navier_stokes_problem(
        mesh_file=MESH_FILE,
        reynolds_init=REYNOLDS_INIT,
        amplitude_init=AMPLITUDE_INIT,
    )

    snapshot_data = load_snapshot_dofs(SNAPSHOT_DIR)
    velocity_snapshots = snapshot_data['velocity']
    pressure_snapshots = snapshot_data['pressure']
    parameters = np.array(snapshot_data['parameters'])
    
    # === OPTIONAL: branch_ids (may not exist in all datasets) ===
    branch_ids = snapshot_data.get('branch_ids', None)
    if branch_ids is not None:
        branch_ids = np.array(branch_ids)
        unique_branches = np.unique(branch_ids)
        has_branch_labels = True
        print(f"  Snapshots: {velocity_snapshots.shape[0]}")
        print(f"  Branch IDs found: {unique_branches}")
        for b in unique_branches:
            mask = branch_ids == b
            print(f"    Branch {b}: {np.sum(mask)} snapshots, "
                  f"Re=[{parameters[mask,0].min():.0f},{parameters[mask,0].max():.0f}], "
                  f"Amp=[{parameters[mask,1].min():.2f},{parameters[mask,1].max():.2f}]")
    else:
        has_branch_labels = False
        unique_branches = None
        print(f"  Snapshots: {velocity_snapshots.shape[0]}")
        print("  No branch_ids found in snapshot data — using proxy clusters only")

    n_snap = velocity_snapshots.shape[0]

    lifting = load_lifting_functions(LIFTING_DIR, problem)
    u_base = lifting.u_base.dat.data_ro.flatten()
    u_control = lifting.u_control.dat.data_ro.flatten()

    if HOMO:
        homo = homogenize_snapshots(velocity_snapshots, parameters, u_base, u_control)
        S = homo.T
        print("  Using HOMOGENISED snapshots")
    else:
        S = velocity_snapshots.T
        print("  Using RAW snapshots")

    M_u = assemble_inner_product_matrix(problem.velocity_space, INNER_PRODUCT_TYPE)
    print(f"  n_dofs={S.shape[0]}, n_snap={n_snap}, inner product={INNER_PRODUCT_TYPE}")

    # --- POD ---
    section("POD COMPUTATION")

    modes, eigenvalues = compute_pod_modes(S, M_u, n_modes=POD_RANK)
    A = project_to_basis(S, modes, M_u)

    total_energy = np.sum(eigenvalues)
    cum_energy = np.cumsum(eigenvalues) / total_energy

    print(f"  Computed {POD_RANK} modes")
    for i in range(min(POD_RANK, 10)):
        print(f"    Mode {i+1:3d}: lambda={eigenvalues[i]:.4e}  "
              f"({eigenvalues[i]/total_energy*100:.2f}%, "
              f"cum {cum_energy[i]*100:.4f}%)")

    # === PLOT POD MODES (subplots) ===
    # Color map for distinct professional lines
    colors = plt.cm.viridis(np.linspace(0, 0.9, POD_RANK))

    # Create subfigures: 2 rows × 5 columns for 10 modes
    n_cols = 5
    n_rows = int(np.ceil(POD_RANK / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 7), sharex=True, sharey=True)
    axes = axes.ravel()  # Flatten to 1D for easy indexing

    # Plot each POD mode in its own subplot
    for idx in range(POD_RANK):
        ax = axes[idx]
        ax.plot(A[idx, :],
                color=colors[idx],
                linewidth=1.3,
                alpha=0.9)

        # Subplot title
        ax.set_title(f'Mode {idx+1}', fontsize=9, fontweight='semibold', pad=6)

        # Grid and ticks
        ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)
        ax.minorticks_on()
        ax.tick_params(axis='both', which='major', length=5, width=1, labelsize=8)
        ax.tick_params(axis='both', which='minor', length=2.5, width=0.5)

        # Only show y-label on leftmost column
        if idx % n_cols == 0:
            ax.set_ylabel('Amplitude', fontsize=9, fontweight='medium')
        else:
            ax.tick_params(labelleft=False)  # Hide y-tick labels for inner subplots

        # Only show x-label on bottom row
        if idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel('Spatial/Temporal Index', fontsize=9, fontweight='medium')
        else:
            ax.tick_params(labelbottom=False)  # Hide x-tick labels for top row

        # Reference line at y=0
        ax.axhline(y=0, color='k', linewidth=0.3, alpha=0.2)

    # Main title
    fig.suptitle('Proper Orthogonal Decomposition Modes',
                fontsize=14, fontweight='bold', y=1.02)

    # Hide any unused subplots (if POD_RANK < n_rows*n_cols)
    for idx in range(POD_RANK, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig('POD_modes_subplots.png', dpi=300, bbox_inches='tight')
    print("  Saved: POD_modes_subplots.png")
    plt.show()

    # --- Forces ---
    section("AERODYNAMIC FORCES")

    print("  Computing lift and drag...")
    lift, drag = compute_lift_drag(
        velocity_snapshots, pressure_snapshots, parameters, problem
    )
    abs_lift = np.abs(lift)

    print(f"  C_L  range: [{lift.min():.4f}, {lift.max():.4f}]")
    print(f"  |C_L| range: [{abs_lift.min():.4f}, {abs_lift.max():.4f}]")
    print(f"  C_D  range: [{drag.min():.4f}, {drag.max():.4f}]")

    # --- Proxy clusters (always available, even without branch_ids) ---
    section("BRANCH LABELS: Mahalanobis proxy clusters")

    # Use proxy clusters regardless of whether branch_ids exists
    if has_branch_labels:
        n_proxy = len(unique_branches)
        print(f"  Using {n_proxy} proxy clusters (matching {len(unique_branches)} true branches)")
    else:
        n_proxy = N_PROXY_CLUSTERS
        print(f"  Using {n_proxy} proxy clusters (default, no ground truth available)")

    sigma = np.sqrt(np.abs(eigenvalues[:POD_RANK, None]) + 1e-15)
    A_white = A / sigma
    proxy_labels, _, _ = cluster_kmeans(A_white, n_proxy)

    if has_branch_labels:
        print("\n  Ground truth branches (from snapshot data):")
        for b in unique_branches:
            mask = branch_ids == b
            print(f"    Branch {b}: {np.sum(mask)} snapshots, "
                  f"mean C_L={lift[mask].mean():+.4f}, "
                  f"mean |C_L|={abs_lift[mask].mean():.4f}")

    print(f"\n  Mahalanobis proxy clusters ({n_proxy} clusters):")
    for c in range(n_proxy):
        mask = proxy_labels == c
        if has_branch_labels:
            # Show which ground-truth branches each proxy cluster contains
            branch_counts = {b: np.sum((proxy_labels == c) & (branch_ids == b))
                            for b in unique_branches}
            bc_str = ", ".join(f"br{b}:{n}" for b, n in branch_counts.items() if n > 0)
            print(f"    Cluster {c}: {np.sum(mask)} snapshots ({bc_str})")
        else:
            print(f"    Cluster {c}: {np.sum(mask)} snapshots")

    # Agreement between proxy and ground truth (only if branch_ids exists)
    if has_branch_labels:
        from itertools import permutations

        best_agreement = 0
        for perm in permutations(range(n_proxy)):
            remapped = np.zeros_like(proxy_labels)
            for i, p in enumerate(perm):
                remapped[proxy_labels == i] = p
            agreement = np.sum(remapped == branch_ids) / n_snap
            best_agreement = max(best_agreement, agreement)

        print(f"\n  Best proxy-vs-truth agreement: {best_agreement*100:.1f}%")
        if best_agreement > 0.85:
            print("  >> Mahalanobis clustering recovers ground-truth branches well!")
        elif best_agreement > 0.65:
            print("  >> Partial agreement — some branch mixing at boundaries")
        else:
            print("  >> Low agreement — proxy clusters don't align with branches")
    else:
        best_agreement = None
        print("  (No ground truth available for comparison)")

    # =====================================================================
    # DIAGNOSTIC 1: Smooth polynomial fit residual
    # =====================================================================
    section("DIAG 1: Smooth polynomial fit residual per mode")

    print(f"  Fitting degree-{POLY_DEGREE} polynomial to each a_i(Re, A)")
    print(f"  High R^2 = mode varies smoothly with parameters")
    print(f"  Low  R^2 = non-smooth / multi-valued (bifurcation)")

    r2_scores = np.zeros(POD_RANK)
    residual_part = np.zeros_like(A)

    print(f"\n  {'Mode':>6}  {'R^2':>8}  {'% non-smooth':>14}  "
          f"{'lambda':>12}  {'Interpretation'}")

    for i in range(POD_RANK):
        a_smooth, a_resid, r2 = fit_smooth_and_residual(
            A[i, :], parameters, POLY_DEGREE
        )
        residual_part[i] = a_resid
        r2_scores[i] = r2

        pct = (1 - r2) * 100
        interp = ""
        if r2 > 0.95:
            interp = "smooth parametric"
        elif r2 > 0.8:
            interp = "mostly smooth"
        elif r2 > 0.5:
            interp = "mixed"
        else:
            interp = "** NON-SMOOTH / BIFURCATION **"

        print(f"  {i+1:>6}  {r2:>8.4f}  {pct:>13.1f}%  "
              f"{eigenvalues[i]:>12.4e}  {interp}")

    # =====================================================================
    # DIAGNOSTIC 2: Correlation with observables
    # =====================================================================
    section("DIAG 2: Correlation with force observables")

    print("  |C_L| = sign-independent measure of symmetry breaking")
    print("  Robust even when amplitude flips the lift sign within a branch")

    corr_lift = np.zeros(POD_RANK)
    corr_abslift = np.zeros(POD_RANK)
    corr_drag = np.zeros(POD_RANK)
    corr_re = np.zeros(POD_RANK)
    corr_amp = np.zeros(POD_RANK)

    print(f"\n  {'Mode':>6}  {'corr(C_L)':>10}  {'corr(|CL|)':>11}  "
          f"{'corr(C_D)':>10}  {'corr(Re)':>9}  {'corr(Amp)':>10}  "
          f"{'R^2':>6}")

    for i in range(POD_RANK):
        corr_lift[i] = np.corrcoef(A[i, :], lift)[0, 1]
        corr_abslift[i] = np.corrcoef(A[i, :], abs_lift)[0, 1]
        corr_drag[i] = np.corrcoef(A[i, :], drag)[0, 1]
        corr_re[i] = np.corrcoef(A[i, :], parameters[:, 0])[0, 1]
        corr_amp[i] = np.corrcoef(A[i, :], parameters[:, 1])[0, 1]

        print(f"  {i+1:>6}  {corr_lift[i]:>+10.4f}  {corr_abslift[i]:>+11.4f}  "
              f"{corr_drag[i]:>+10.4f}  {corr_re[i]:>+9.4f}  "
              f"{corr_amp[i]:>+10.4f}  {r2_scores[i]:>6.3f}")

    # =====================================================================
    # DIAGNOSTIC 3: Local variance
    # =====================================================================
    section("DIAG 3: Local variance in parameter space")

    print("  High local/global ratio = nearby parameters give very")
    print("  different coefficients = bifurcation boundary nearby")

    loc_var = local_variance(A, parameters, n_neighbors=8)
    mean_local_var = loc_var.mean(axis=1)
    global_var = np.var(A, axis=1)
    local_ratio = mean_local_var / (global_var + 1e-30)

    print(f"\n  {'Mode':>6}  {'Global var':>12}  {'Mean local':>12}  "
          f"{'Loc/Glob':>10}  {'Interpretation'}")

    for i in range(min(POD_RANK, 15)):
        interp = ""
        if local_ratio[i] > 0.5:
            interp = "** bifurcation **"
        elif local_ratio[i] > 0.2:
            interp = "* transition *"
        else:
            interp = "smooth"

        print(f"  {i+1:>6}  {global_var[i]:>12.4e}  {mean_local_var[i]:>12.4e}  "
              f"{local_ratio[i]:>10.4f}  {interp}")

    # =====================================================================
    # DIAGNOSTIC 4: Fisher ratio with proxy labels
    # =====================================================================
    section("DIAG 4: Fisher discriminant ratio")

    if has_branch_labels:
        print("  Using GROUND TRUTH branch_ids and Mahalanobis proxy labels")
        print("  Comparing which modes separate true branches vs proxy clusters")
    else:
        print("  Using Mahalanobis proxy labels only (no ground truth)")

    fisher_raw_true = np.zeros(POD_RANK)
    fisher_white_true = np.zeros(POD_RANK)
    fisher_raw_proxy = np.zeros(POD_RANK)
    fisher_white_proxy = np.zeros(POD_RANK)

    print(f"\n  {'Mode':>6}  {'J_raw':>7}  {'J_wht':>7}  {'Amp':>6}  "
          f"{'J_raw_px':>9}  {'J_wht_px':>9}  {'lambda':>12}  {'R^2':>6}")

    for i in range(POD_RANK):
        a_w = A[i, :] / (np.sqrt(abs(eigenvalues[i])) + 1e-15)

        if has_branch_labels:
            fisher_raw_true[i] = fisher_discriminant_ratio(A[i, :], branch_ids)
            fisher_white_true[i] = fisher_discriminant_ratio(a_w, branch_ids)

        fisher_raw_proxy[i] = fisher_discriminant_ratio(A[i, :], proxy_labels)
        fisher_white_proxy[i] = fisher_discriminant_ratio(a_w, proxy_labels)

        if has_branch_labels:
            amp_f = fisher_white_true[i] / (fisher_raw_true[i] + 1e-30)
            print(f"  {i+1:>6}  {fisher_raw_true[i]:>7.3f}  {fisher_white_true[i]:>7.3f}  "
                  f"{amp_f:>5.1f}x  {fisher_raw_proxy[i]:>9.3f}  "
                  f"{fisher_white_proxy[i]:>9.3f}  {eigenvalues[i]:>12.4e}  "
                  f"{r2_scores[i]:>6.3f}")
        else:
            amp_f = fisher_white_proxy[i] / (fisher_raw_proxy[i] + 1e-30)
            print(f"  {i+1:>6}  {'N/A':>7}  {'N/A':>7}  {'N/A':>6}  "
                  f"{fisher_raw_proxy[i]:>9.3f}  "
                  f"{fisher_white_proxy[i]:>9.3f}  {eigenvalues[i]:>12.4e}  "
                  f"{r2_scores[i]:>6.3f}")

    # Use proxy Fisher for all downstream analysis if no branch_ids
    if has_branch_labels:
        fisher_raw = fisher_raw_true
        fisher_white = fisher_white_true
    else:
        fisher_raw = fisher_raw_proxy
        fisher_white = fisher_white_proxy

    # =====================================================================
    # DIAGNOSTIC 5: Distance contribution
    # =====================================================================
    section("DIAG 5: Distance contribution per mode")

    raw_variance = np.var(A, axis=1)
    white_variance = np.var(A_white, axis=1)
    resid_variance = np.var(residual_part, axis=1)

    raw_frac = raw_variance / (raw_variance.sum() + 1e-30)
    white_frac = white_variance / (white_variance.sum() + 1e-30)
    resid_frac = resid_variance / (resid_variance.sum() + 1e-30)

    print(f"\n  {'Mode':>6}  {'Raw %':>8}  {'White %':>9}  "
          f"{'Resid %':>9}  {'R^2':>6}  {'Loc/Glob':>9}")

    for i in range(min(POD_RANK, 15)):
        print(f"  {i+1:>6}  {raw_frac[i]*100:>7.2f}%  "
              f"{white_frac[i]*100:>8.2f}%  {resid_frac[i]*100:>8.2f}%  "
              f"{r2_scores[i]:>6.3f}  {local_ratio[i]:>9.4f}")

    # =====================================================================
    # DIAGNOSTIC 6: Mode classification
    # =====================================================================
    section("DIAG 6: Mode classification")

    classifications = []

    print(f"\n  {'Mode':>6}  {'Classification':<25}  {'Evidence'}")

    for i in range(POD_RANK):
        bif_score = 0
        evidence = []

        if r2_scores[i] < 0.5:
            bif_score += 2
            evidence.append(f"R^2={r2_scores[i]:.2f}")
        elif r2_scores[i] < 0.8:
            bif_score += 1
            evidence.append(f"R^2={r2_scores[i]:.2f}")

        if local_ratio[i] > 0.5:
            bif_score += 2
            evidence.append(f"loc/glob={local_ratio[i]:.2f}")
        elif local_ratio[i] > 0.2:
            bif_score += 1
            evidence.append(f"loc/glob={local_ratio[i]:.2f}")

        if abs(corr_abslift[i]) > 0.5:
            bif_score += 1
            evidence.append(f"|corr(|CL|)|={abs(corr_abslift[i]):.2f}")

        amp_f = fisher_white[i] / (fisher_raw[i] + 1e-30)
        if amp_f > 5:
            bif_score += 1
            evidence.append(f"Fisher amp={amp_f:.0f}x")

        if bif_score >= 3:
            cls = "BIFURCATION MODE"
        elif bif_score >= 1:
            cls = "MIXED / TRANSITIONAL"
        else:
            cls = "smooth parametric"

        classifications.append(cls)
        print(f"  {i+1:>6}  {cls:<25}  {', '.join(evidence)}")

    bif_indices = [i for i, c in enumerate(classifications) if "BIFURCATION" in c]
    mixed_indices = [i for i, c in enumerate(classifications) if "MIXED" in c]
    smooth_indices = [i for i, c in enumerate(classifications) if "smooth" in c]

    print(f"\n  Summary: {len(bif_indices)} bifurcation, "
          f"{len(mixed_indices)} mixed, {len(smooth_indices)} smooth")

    # =====================================================================
    # DIAGNOSTIC 7: Whitening amplification summary
    # =====================================================================
    section("DIAG 7: Whitening amplification of bifurcation content")

    total_J_raw = np.sum(fisher_raw)
    total_J_white = np.sum(fisher_white)

    print(f"  Total Fisher J (raw):      {total_J_raw:.4f}")
    print(f"  Total Fisher J (whitened): {total_J_white:.4f}")
    print(f"  Overall amplification:     {total_J_white/(total_J_raw+1e-30):.1f}x")

    if bif_indices:
        J_bif_raw = sum(fisher_raw[i] for i in bif_indices)
        J_bif_white = sum(fisher_white[i] for i in bif_indices)
        print(f"\n  Bifurcation modes share of Fisher J:")
        print(f"    Raw:      {J_bif_raw/(total_J_raw+1e-30)*100:.1f}%")
        print(f"    Whitened: {J_bif_white/(total_J_white+1e-30)*100:.1f}%")

        bif_energy_pct = sum(eigenvalues[i] for i in bif_indices) / total_energy * 100
        bif_raw_pct = sum(raw_frac[i] for i in bif_indices) * 100
        bif_white_pct = sum(white_frac[i] for i in bif_indices) * 100

        print(f"\n  Bifurcation modes carry:")
        print(f"    {bif_energy_pct:.2f}% of total energy")
        print(f"    {bif_raw_pct:.2f}% of raw distance")
        print(f"    {bif_white_pct:.1f}% of whitened distance")
        print(f"    Amplification: {bif_white_pct/(bif_raw_pct+1e-30):.1f}x")

    # =====================================================================
    # PLOTS
    # =====================================================================
    section("GENERATING PLOTS")

    fig = plt.figure(figsize=(20, 22))
    gs = GridSpec(5, 3, figure=fig, hspace=0.4, wspace=0.35)

    mode_idx = np.arange(1, POD_RANK + 1)
    bif_color = '#D85A30'
    smooth_color = '#888780'
    mixed_color = '#185FA5'

    def mode_color(i):
        if "BIFURCATION" in classifications[i]: return bif_color
        if "MIXED" in classifications[i]: return mixed_color
        return smooth_color

    bar_colors = [mode_color(i) for i in range(POD_RANK)]

    # --- Row 0: Bifurcation diagram + branch overview ---
    # Only plot branch-colored figures if branch_ids exists

    ax = fig.add_subplot(gs[0, 0])
    if has_branch_labels:
        branch_cmap = {0: '#0F6E56', 1: '#D85A30', 2: '#185FA5',
                       3: '#534AB7', 4: '#993556'}
        for b in unique_branches:
            mask = branch_ids == b
            ax.scatter(parameters[mask, 0], lift[mask],
                       c=branch_cmap.get(b, '#888'), s=15, alpha=0.7,
                       label=f'branch {b}')
        ax.legend(fontsize=7)
    else:
        # Color by proxy clusters instead
        for c in range(n_proxy):
            mask = proxy_labels == c
            ax.scatter(parameters[mask, 0], lift[mask],
                       s=15, alpha=0.7, label=f'cluster {c}')
        ax.legend(fontsize=7)
    ax.set_xlabel('Re'); ax.set_ylabel('C_L')
    ax.set_title('Bifurcation diagram (C_L vs Re)')
    ax.axhline(0, color='gray', lw=0.5, ls='--')

    ax = fig.add_subplot(gs[0, 1])
    if has_branch_labels:
        branch_cmap = {0: '#0F6E56', 1: '#D85A30', 2: '#185FA5',
                       3: '#534AB7', 4: '#993556'}
        for b in unique_branches:
            mask = branch_ids == b
            ax.scatter(parameters[mask, 0], parameters[mask, 1],
                       c=branch_cmap.get(b, '#888'), s=15, alpha=0.7,
                       label=f'branch {b}')
        ax.legend(fontsize=7)
    else:
        for c in range(n_proxy):
            mask = proxy_labels == c
            ax.scatter(parameters[mask, 0], parameters[mask, 1],
                       s=15, alpha=0.7, label=f'cluster {c}')
        ax.legend(fontsize=7)
    ax.set_xlabel('Re'); ax.set_ylabel('Amplitude')
    ax.set_title('Parameter space by ' + ('branch' if has_branch_labels else 'cluster'))

    ax = fig.add_subplot(gs[0, 2])
    if has_branch_labels:
        branch_cmap = {0: '#0F6E56', 1: '#D85A30', 2: '#185FA5',
                       3: '#534AB7', 4: '#993556'}
        for b in unique_branches:
            mask = branch_ids == b
            ax.scatter(parameters[mask, 1], lift[mask],
                       c=branch_cmap.get(b, '#888'), s=15, alpha=0.7,
                       label=f'branch {b}')
        ax.legend(fontsize=7)
    else:
        for c in range(n_proxy):
            mask = proxy_labels == c
            ax.scatter(parameters[mask, 1], lift[mask],
                       s=15, alpha=0.7, label=f'cluster {c}')
        ax.legend(fontsize=7)
    ax.set_xlabel('Amplitude'); ax.set_ylabel('C_L')
    ax.set_title('C_L vs Amplitude by ' + ('branch' if has_branch_labels else 'cluster'))
    ax.axhline(0, color='gray', lw=0.5, ls='--')

    # --- Row 1: Mode diagnostics ---
    ax = fig.add_subplot(gs[1, 0])
    ax.bar(mode_idx, r2_scores, color=bar_colors, alpha=0.75)
    ax.axhline(0.8, color='gray', lw=0.5, ls='--')
    ax.axhline(0.5, color='gray', lw=0.5, ls=':')
    ax.set_xlabel('POD mode'); ax.set_ylabel('R^2')
    ax.set_title('Smooth fit quality'); ax.set_ylim(0, 1.05)

    ax = fig.add_subplot(gs[1, 1])
    ax.bar(mode_idx, local_ratio, color=bar_colors, alpha=0.75)
    ax.axhline(0.5, color='gray', lw=0.5, ls='--')
    ax.set_xlabel('POD mode'); ax.set_ylabel('Local / global var')
    ax.set_title('Local variance ratio')

    ax = fig.add_subplot(gs[1, 2])
    ax.bar(mode_idx, eigenvalues[:POD_RANK], color=bar_colors, alpha=0.75)
    ax.set_yscale('log')
    ax.set_xlabel('POD mode'); ax.set_ylabel('Eigenvalue')
    ax.set_title('Eigenvalue spectrum')
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=bif_color, alpha=0.75, label='bifurcation'),
        Patch(facecolor=mixed_color, alpha=0.75, label='mixed'),
        Patch(facecolor=smooth_color, alpha=0.75, label='smooth'),
    ], fontsize=7)

    # Row 2
    w = 0.35
    ax = fig.add_subplot(gs[2, 0])
    ax.bar(mode_idx - w/2, fisher_raw, w, label='raw', color=smooth_color, alpha=0.7)
    ax.bar(mode_idx + w/2, fisher_white, w, label='whitened', color=bif_color, alpha=0.7)
    ax.set_xlabel('POD mode'); ax.set_ylabel('Fisher J')
    ax.set_title('Branch/cluster separation'); ax.legend(fontsize=7)

    ax = fig.add_subplot(gs[2, 1])
    ax.bar(mode_idx - w/2, raw_frac*100, w, label='raw', color=smooth_color, alpha=0.7)
    ax.bar(mode_idx + w/2, white_frac*100, w, label='whitened', color=mixed_color, alpha=0.7)
    ax.set_xlabel('POD mode'); ax.set_ylabel('% distance')
    ax.set_title('Distance contribution'); ax.legend(fontsize=7)

    ax = fig.add_subplot(gs[2, 2])
    ax.plot(mode_idx, np.abs(corr_lift), 'o-', ms=4, label='|corr(C_L)|', color=bif_color)
    ax.plot(mode_idx, np.abs(corr_abslift), 's-', ms=4, label='|corr(|C_L|)|', color=mixed_color)
    ax.plot(mode_idx, np.abs(corr_re), '^-', ms=3, label='|corr(Re)|', color=smooth_color, alpha=0.5)
    ax.plot(mode_idx, np.abs(corr_amp), 'v-', ms=3, label='|corr(Amp)|', color='#0F6E56', alpha=0.5)
    ax.set_xlabel('POD mode'); ax.set_ylabel('|correlation|')
    ax.set_title('Observable correlations'); ax.legend(fontsize=7)

    # Row 3-4: coefficient maps a_i(Re, A) and residual maps
    def pick_first(cls_keyword):
        for i in range(POD_RANK):
            if cls_keyword in classifications[i]:
                return i
        return 0

    picks = []
    for key in ["smooth", "BIFURCATION", "MIXED"]:
        idx = pick_first(key)
        if idx not in [p[0] for p in picks]:
            picks.append((idx, key.strip().lower()))
    while len(picks) < 3:
        for i in range(POD_RANK):
            if i not in [p[0] for p in picks]:
                picks.append((i, classifications[i].strip().lower()))
                break

    Re = parameters[:, 0]
    Amp = parameters[:, 1]

    for col, (mi, label) in enumerate(picks[:3]):
        ax = fig.add_subplot(gs[3, col])
        sc = ax.scatter(Re, Amp, c=A[mi, :], cmap='RdBu_r', s=20, alpha=0.8)
        plt.colorbar(sc, ax=ax, label=f'a_{mi+1}', shrink=0.8)
        ax.set_xlabel('Re'); ax.set_ylabel('Amplitude')
        ax.set_title(f'Mode {mi+1} ({label})\n'
                     f'R^2={r2_scores[mi]:.2f}, lam={eigenvalues[mi]:.2e}')

        ax = fig.add_subplot(gs[4, col])
        sc = ax.scatter(Re, Amp, c=residual_part[mi, :], cmap='RdBu_r',
                        s=20, alpha=0.8)
        plt.colorbar(sc, ax=ax, label=f'residual', shrink=0.8)
        ax.set_xlabel('Re'); ax.set_ylabel('Amplitude')
        ax.set_title(f'Mode {mi+1} residual\n'
                     f'non-smooth fraction = {(1-r2_scores[mi])*100:.1f}%')

    fig.suptitle('Multi-parameter bifurcation mode diagnostics' +
                 ('' if has_branch_labels else ' (no ground-truth branches)'),
                 fontsize=14, fontweight='bold', y=1.01)

    plt.savefig('bifurcation_diagnostics_multiparam.png', dpi=150, bbox_inches='tight')
    print("  Saved: bifurcation_diagnostics_multiparam.png")
    plt.show()

    # =====================================================================
    # FINAL SUMMARY
    # =====================================================================
    section("FINAL SUMMARY")

    print("  Claim tested:")
    print("    'Dominant POD modes capture smooth parametric variation.")
    print("     Bifurcation modes have small eigenvalues and are")
    print("     amplified by Mahalanobis whitening.'")
    print()

    if bif_indices:
        bif_ranks = [i + 1 for i in bif_indices]
        smooth_ranks = [i + 1 for i in smooth_indices]

        print(f"  Bifurcation modes at ranks: {bif_ranks}")
        print(f"  Smooth modes at ranks:      {smooth_ranks[:8]}...")
        print()

        if smooth_ranks and np.mean(bif_ranks) > np.mean(smooth_ranks):
            print("  CONFIRMED: Bifurcation modes are lower-ranked than smooth modes.")
        else:
            print("  PARTIAL: Some bifurcation modes are among the dominant modes.")
            print("  (Expected when actuation directly drives large-scale changes.)")

        bif_energy_pct = sum(eigenvalues[i] for i in bif_indices) / total_energy * 100
        bif_raw_pct = sum(raw_frac[i] for i in bif_indices) * 100
        bif_white_pct = sum(white_frac[i] for i in bif_indices) * 100

        print(f"\n  Bifurcation modes:")
        print(f"    Energy:              {bif_energy_pct:.2f}%")
        print(f"    Raw distance:        {bif_raw_pct:.2f}%")
        print(f"    Whitened distance:   {bif_white_pct:.1f}%")
        print(f"    Amplification:       {bif_white_pct/(bif_raw_pct+1e-30):.1f}x")
        print(f"\n  This is why Mahalanobis whitening reveals the bifurcation structure.")

        if has_branch_labels:
            print(f"\n  Mahalanobis proxy vs ground-truth agreement: {best_agreement*100:.1f}%")
            if best_agreement > 0.85:
                print("  >> The whitening-based clustering successfully recovers")
                print("     the true branch structure from the data alone.")
        else:
            print("\n  (No ground-truth branches available for proxy comparison)")
    else:
        print("  No modes classified as bifurcation modes.")
        print("  Try reducing POLY_DEGREE or increasing POD_RANK.")


if __name__ == "__main__":
    main()
