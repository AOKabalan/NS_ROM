"""
Bifurcation-Aware Clustering Strategies
========================================

Strategies that explicitly target bifurcation structure rather than
relying on Mahalanobis whitening to find it incidentally.

The key insight: near a bifurcation point, the solution manifold
splits. The splitting direction is captured by specific POD modes
(typically the symmetry-breaking or Hopf modes). These strategies
identify and exploit those modes directly.

Strategies provided:
  1. symmetry_classifier  — hard split on the antisymmetric mode sign
  2. spectral_manifold    — spectral clustering on the solution manifold
  3. lift_based           — cluster using aerodynamic observables
  4. gap_weighted         — POD with spectral-gap-aware weighting
  5. local_pca            — detect manifold branching via local PCA
"""

import numpy as np
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.neighbors import kneighbors_graph


# =============================================================================
# STRATEGY 1: Symmetry-breaking mode classifier
# =============================================================================

def strategy_symmetry_classifier(snapshots, parameters, M_u,
                                  n_clusters=2, pod_rank=20,
                                  symmetry_mode_index=None):
    """
    Identify the symmetry-breaking POD mode and use its sign to
    classify snapshots into bifurcation branches.

    For the fluidic pinball:
      - Symmetric solutions (pre-pitchfork) have ~zero projection
        onto the antisymmetric mode.
      - Upper/lower branch solutions have positive/negative projections.

    The antisymmetric mode is detected automatically as the one with
    highest correlation to the sign of the lift coefficient. If lift
    data isn't available, it picks the mode with highest skewness
    (bimodal distribution → high |skew| or low skew but high kurtosis).

    After the hard branch split, sub-clustering within each branch
    captures the parametric variation along each branch.
    """
    from rom import compute_pod_modes, project_to_basis

    print(f"  [strategy] symmetry_classifier  (rank={pod_rank})")

    # --- Step 1: POD ---
    modes, eigenvalues = compute_pod_modes(snapshots, M_u, n_modes=pod_rank)
    A = project_to_basis(snapshots, modes, M_u)  # (pod_rank, n_snap)

    # --- Step 2: Find the symmetry-breaking mode ---
    if symmetry_mode_index is not None:
        sb_idx = symmetry_mode_index
        print(f"    Using user-specified symmetry mode: {sb_idx}")
    else:
        # Heuristic: the symmetry-breaking mode has the most bimodal
        # distribution of coefficients. Detect via kurtosis (bimodal → low
        # kurtosis, or negative excess kurtosis) or via the Hartigan dip test.
        #
        # Simpler approach: find the mode whose coefficient distribution
        # has the largest gap around zero (two clusters separated by zero).
        sb_idx = _find_symmetry_breaking_mode(A)
        print(f"    Auto-detected symmetry-breaking mode: {sb_idx}")

    a_sb = A[sb_idx, :]  # (n_snap,)
    print(f"    Mode {sb_idx}: mean={a_sb.mean():.4f}, "
          f"std={a_sb.std():.4f}, "
          f"min={a_sb.min():.4f}, max={a_sb.max():.4f}")

    # --- Step 3: Classify by sign ---
    # Three-way split: negative branch / near-zero (symmetric) / positive branch
    threshold = 0.1 * a_sb.std()  # points within 10% of std from zero
    labels = np.zeros(A.shape[1], dtype=int)

    if n_clusters == 2:
        # Simple binary: sign of the symmetry-breaking coefficient
        labels[a_sb >= 0] = 0
        labels[a_sb < 0] = 1
    elif n_clusters == 3:
        labels[a_sb > threshold] = 0    # upper branch
        labels[np.abs(a_sb) <= threshold] = 1  # symmetric / transition
        labels[a_sb < -threshold] = 2   # lower branch
    else:
        # For more clusters: first split by sign, then sub-cluster
        # within each branch using remaining POD coefficients
        pos_mask = a_sb >= 0
        neg_mask = a_sb < 0

        n_pos = max(1, n_clusters // 2)
        n_neg = n_clusters - n_pos

        labels = np.zeros(A.shape[1], dtype=int)

        if np.sum(pos_mask) > n_pos:
            A_pos = A[:, pos_mask]
            km = KMeans(n_clusters=n_pos, n_init=10, random_state=42)
            labels[pos_mask] = km.fit_predict(A_pos.T)

        if np.sum(neg_mask) > n_neg:
            A_neg = A[:, neg_mask]
            km = KMeans(n_clusters=n_neg, n_init=10, random_state=42)
            labels[neg_mask] = km.fit_predict(A_neg.T) + n_pos

    # --- Compute inertia ---
    unique = np.unique(labels)
    centers = np.column_stack([A[:, labels == c].mean(axis=1) for c in unique])
    inertia = sum(
        np.sum((A[:, labels == c] - centers[:, i:i+1]) ** 2)
        for i, c in enumerate(unique)
    )

    for c in unique:
        count = np.sum(labels == c)
        mean_sb = a_sb[labels == c].mean()
        print(f"    Cluster {c}: {count} snapshots, "
              f"mean symmetry coeff = {mean_sb:.4f}")

    return labels, centers.T, inertia


def _find_symmetry_breaking_mode(A, top_k=5):
    """
    Find the POD mode whose coefficient distribution is most bimodal.

    Heuristic: compute the "dip statistic" approximation for the top_k
    modes. The mode with the most bimodal distribution (lowest
    normalised density at zero relative to the tails) is likely the
    symmetry-breaking mode.
    """
    n_modes = min(top_k, A.shape[0])
    best_score = -np.inf
    best_idx = 1  # default: mode 1 (often the first antisymmetric one)

    for i in range(n_modes):
        a = A[i, :]
        a_centered = a - a.mean()

        # Score: ratio of variance in the two halves vs variance at center
        pos = a_centered[a_centered > 0]
        neg = a_centered[a_centered < 0]

        if len(pos) < 5 or len(neg) < 5:
            continue

        # Bimodality coefficient: uses skewness and kurtosis
        # BC = (skew^2 + 1) / kurtosis
        # Higher BC → more bimodal
        n = len(a)
        m2 = np.mean(a_centered ** 2)
        m3 = np.mean(a_centered ** 3)
        m4 = np.mean(a_centered ** 4)

        skew = m3 / (m2 ** 1.5 + 1e-30)
        kurt = m4 / (m2 ** 2 + 1e-30)

        # Sarle's bimodality coefficient
        bc = (skew ** 2 + 1) / (kurt + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)) + 1e-30)

        if bc > best_score:
            best_score = bc
            best_idx = i

    return best_idx


# =============================================================================
# STRATEGY 2: Spectral clustering on the solution manifold
# =============================================================================

def strategy_spectral_manifold(snapshots, parameters, M_u,
                                n_clusters=2, pod_rank=20,
                                n_neighbors=10, gamma=None):
    """
    Spectral clustering in POD coefficient space.

    Unlike k-means (which finds convex clusters), spectral clustering
    finds connected components of a similarity graph. This is ideal
    for bifurcation problems because:

      - Branches are connected manifolds that may be non-convex
      - Near the bifurcation point, branches are close but distinct
      - Spectral methods respect the manifold topology

    The similarity graph is built from kNN in whitened POD space,
    so it captures both the local manifold structure and the
    energy-norm geometry.
    """
    from rom import compute_pod_modes, project_to_basis

    print(f"  [strategy] spectral_manifold  "
          f"(rank={pod_rank}, knn={n_neighbors})")

    # --- POD + whitening ---
    modes, eigenvalues = compute_pod_modes(snapshots, M_u, n_modes=pod_rank)
    A = project_to_basis(snapshots, modes, M_u)

    # Whitening (same as Mahalanobis)
    sigma = np.sqrt(np.abs(eigenvalues[:pod_rank, None]) + 1e-15)
    A_white = A / sigma  # (pod_rank, n_snap)

    features = A_white.T  # (n_snap, pod_rank)

    # --- Spectral clustering ---
    if gamma is None:
        # Estimate gamma from median pairwise distance
        from scipy.spatial.distance import pdist
        dists = pdist(features)
        gamma = 1.0 / (2 * np.median(dists) ** 2 + 1e-30)

    print(f"    RBF gamma: {gamma:.4e}")

    model = SpectralClustering(
        n_clusters=n_clusters,
        affinity='nearest_neighbors',
        n_neighbors=n_neighbors,
        random_state=42,
        assign_labels='kmeans',
    )
    labels = model.fit_predict(features)

    # --- Inertia in whitened space ---
    unique = np.unique(labels)
    centers = np.column_stack([
        A_white[:, labels == c].mean(axis=1) for c in unique
    ])
    inertia = sum(
        np.sum((A_white[:, labels == c] - centers[:, i:i+1]) ** 2)
        for i, c in enumerate(unique)
    )

    return labels, centers.T, inertia


# =============================================================================
# STRATEGY 3: Observable-based clustering (lift/drag)
# =============================================================================

def strategy_lift_based(snapshots, parameters, M_u,
                        lift_coefficients, drag_coefficients=None,
                        n_clusters=2):
    """
    Cluster directly on aerodynamic observables.

    Bifurcation branches have distinct force characteristics:
      - Symmetric branch: C_L ≈ 0
      - Upper branch: C_L > 0
      - Lower branch: C_L < 0
      - Periodic branch: C_L oscillates (captured by amplitude)

    This is the most physically meaningful clustering because it
    directly reflects the bifurcation diagram.

    Args:
        lift_coefficients:  (n_snap,) lift values
        drag_coefficients:  (n_snap,) drag values (optional)
    """
    print(f"  [strategy] lift_based  (n_clusters={n_clusters})")

    n_snap = snapshots.shape[1]

    # Build feature vector from observables + parameters
    features = [parameters]  # (n_snap, 2)

    lift_norm = (lift_coefficients - lift_coefficients.mean()) / \
                (lift_coefficients.std() + 1e-12)
    features.append(lift_norm.reshape(-1, 1))

    if drag_coefficients is not None:
        drag_norm = (drag_coefficients - drag_coefficients.mean()) / \
                    (drag_coefficients.std() + 1e-12)
        features.append(drag_norm.reshape(-1, 1))

    # Normalise parameters too
    params_norm = (parameters - parameters.mean(axis=0)) / \
                  (parameters.std(axis=0) + 1e-12)
    features[0] = params_norm

    X = np.hstack(features)  # (n_snap, 3 or 4)
    print(f"    Feature matrix: {X.shape}")

    # k-means on observables
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = km.fit_predict(X)

    # Report
    for c in range(n_clusters):
        mask = labels == c
        print(f"    Cluster {c}: {np.sum(mask)} snapshots, "
              f"mean C_L = {lift_coefficients[mask].mean():.4f}")

    from rom import compute_pod_modes, project_to_basis
    modes, eigenvalues = compute_pod_modes(snapshots, M_u, n_modes=10)
    A = project_to_basis(snapshots, modes, M_u)
    unique = np.unique(labels)
    centers = np.column_stack([A[:, labels == c].mean(axis=1) for c in unique])
    inertia = sum(
        np.sum((A[:, labels == c] - centers[:, i:i+1]) ** 2)
        for i, c in enumerate(unique)
    )

    return labels, centers.T, inertia


# =============================================================================
# STRATEGY 4: Spectral-gap-weighted POD clustering
# =============================================================================

def strategy_gap_weighted(snapshots, parameters, M_u,
                           n_clusters=2, pod_rank=20):
    """
    Weight POD coefficients by the INVERSE of the spectral gap
    around each eigenvalue.

    Intuition: eigenvalues that are isolated (large gap to neighbours)
    capture distinct physical phenomena — often bifurcation-related
    modes. Eigenvalues in a smooth decay capture gradual parametric
    variation. By upweighting isolated eigenvalues, we emphasise
    the bifurcation-sensitive directions.

    Weight for mode i:
        w_i = 1 / (min(|λ_i - λ_{i-1}|, |λ_i - λ_{i+1}|) + eps)
        (normalised to unit mean)
    """
    from rom import compute_pod_modes, project_to_basis
    from .cluster_building import cluster_kmeans

    print(f"  [strategy] gap_weighted  (rank={pod_rank})")

    modes, eigenvalues = compute_pod_modes(snapshots, M_u, n_modes=pod_rank)
    A = project_to_basis(snapshots, modes, M_u)

    # Compute spectral gaps
    eigs = eigenvalues[:pod_rank]
    gaps = np.zeros(pod_rank)
    for i in range(pod_rank):
        left  = abs(eigs[i] - eigs[i-1]) if i > 0 else abs(eigs[i] - eigs[i+1])
        right = abs(eigs[i] - eigs[i+1]) if i < pod_rank - 1 else left
        gaps[i] = min(left, right)

    # Weight: inverse gap (isolated modes get upweighted)
    weights = 1.0 / (gaps + 1e-12)
    weights /= weights.mean()  # normalise to unit mean

    print(f"    Eigenvalues:  {eigs[:6]}")
    print(f"    Gaps:         {gaps[:6]}")
    print(f"    Weights:      {weights[:6]}")

    # Apply weights
    A_weighted = A * weights[:, None]

    return cluster_kmeans(A_weighted, n_clusters)


# =============================================================================
# STRATEGY 5: Local PCA manifold branching detection
# =============================================================================

def strategy_local_pca(snapshots, parameters, M_u,
                        n_clusters=2, pod_rank=20,
                        n_neighbors=15, local_dim=3):
    """
    Detect manifold branching via local PCA dimensionality.

    Near a bifurcation point, the local dimension of the solution
    manifold increases (one manifold splits into two). This strategy:

      1. For each snapshot, compute local PCA on its k nearest
         neighbours in whitened POD space.
      2. Measure the local intrinsic dimension via the eigenvalue
         spectrum of the local covariance.
      3. Points with high local dimension are near bifurcation points.
      4. Use this as a feature (along with POD coefficients) in
         spectral clustering to find branches.

    This is inspired by the manifold learning literature on
    detecting branching structures.
    """
    from rom import compute_pod_modes, project_to_basis
    from sklearn.neighbors import NearestNeighbors

    print(f"  [strategy] local_pca  "
          f"(rank={pod_rank}, knn={n_neighbors}, local_dim={local_dim})")

    modes, eigenvalues = compute_pod_modes(snapshots, M_u, n_modes=pod_rank)
    A = project_to_basis(snapshots, modes, M_u)

    # Whitening
    sigma = np.sqrt(np.abs(eigenvalues[:pod_rank, None]) + 1e-15)
    A_white = (A / sigma).T  # (n_snap, pod_rank)

    # --- Local PCA for each point ---
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(A_white)
    distances, indices = nn.kneighbors(A_white)

    n_snap = A_white.shape[0]
    local_dims = np.zeros(n_snap)
    local_residuals = np.zeros(n_snap)

    for i in range(n_snap):
        # Local neighbourhood
        neighbours = A_white[indices[i]]
        local_mean = neighbours.mean(axis=0)
        centered = neighbours - local_mean

        # Local PCA
        cov = centered.T @ centered / (n_neighbors - 1)
        local_eigs = np.linalg.eigvalsh(cov)[::-1]

        # Intrinsic dimension: ratio of explained variance
        cumvar = np.cumsum(local_eigs) / (np.sum(local_eigs) + 1e-30)
        local_dims[i] = np.searchsorted(cumvar, 0.95) + 1

        # Residual: variance not explained by top local_dim components
        if len(local_eigs) > local_dim:
            local_residuals[i] = np.sum(local_eigs[local_dim:]) / \
                                  (np.sum(local_eigs) + 1e-30)

    print(f"    Local dimension stats: "
          f"mean={local_dims.mean():.1f}, "
          f"min={local_dims.min():.0f}, max={local_dims.max():.0f}")
    print(f"    High-dim points (dim > {local_dim}): "
          f"{np.sum(local_dims > local_dim)}")

    # --- Spectral clustering with local dimension as extra feature ---
    # Augment whitened coefficients with local dimension info
    dim_feature = (local_dims - local_dims.mean()) / (local_dims.std() + 1e-12)
    res_feature = (local_residuals - local_residuals.mean()) / \
                  (local_residuals.std() + 1e-12)

    features = np.column_stack([A_white, dim_feature, res_feature])

    model = SpectralClustering(
        n_clusters=n_clusters,
        affinity='nearest_neighbors',
        n_neighbors=n_neighbors,
        random_state=42,
    )
    labels = model.fit_predict(features)

    # Inertia
    unique = np.unique(labels)
    centers = np.column_stack([
        A_white[labels == c].mean(axis=0) for c in unique
    ]).T  # careful with transpose
    inertia = sum(
        np.sum((A_white[labels == c] - centers[i:i+1]) ** 2)
        for i, c in enumerate(unique)
    )

    return labels, centers, inertia
