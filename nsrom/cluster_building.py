"""
k-means clustering for snapshot data, with two methods:

  * "cholesky"      — Cholesky-based energy transform turns the
                      M-norm k-means objective into a plain Euclidean
                      one. Equivalent to k-means in DOF space under
                      the M-inner-product.

  * "pod_whitened"  — k-means on Mahalanobis-whitened POD coefficients.
                      Decouples branch-separation from energy content;
                      bifurcation modes contribute to clustering distance
                      independently of their λ_i.

Both methods produce DOF-space centroids in
`ClusteringResult.centroids_original`, so downstream consumers (fast
cluster selection, change-of-basis) are agnostic to method.

Snapshot preprocessing in `build_energy_snapshots` always assembles the
mass matrices and runs the Cholesky transform (it's cheap, and `S_tilde`
serves as a stable cache key). Homogenization (lifting subtraction) is
controlled by a flag and defaults to True — required for any clustering
that should reflect flow physics rather than parameter-dependent forcing.

Requires: scikit-sparse  (sudo apt install libsuitesparse-dev && pip install scikit-sparse)
"""
import os
import json
import hashlib
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from sksparse.cholmod import cholesky
from sklearn.cluster import KMeans

from nsrom.rom import (
    assemble_inner_product_matrix,
    homogenize_snapshots,
    compute_pod_modes,
    project_to_basis,
)


# =============================================================================
# Result container
# =============================================================================

@dataclass
class ClusteringResult:
    """
    Holds everything produced by the clustering step.

    Attributes
    ----------
    n_clusters : int
    labels : (n_snapshots,) cluster index per snapshot
    cluster_indices : list of arrays — cluster_indices[k] = indices in cluster k
    centroids_original    : (n_dofs, n_clusters) DOF-space centroids
    centroids_transformed : (n_dofs, n_clusters) tilde-space centroids
    L : sparse lower-triangular — Cholesky factor of M
    P : (n_dofs,) int — fill-reducing permutation from CHOLMOD
    parameters : list of (Re, amp) tuples, length n_snapshots

    fast_selection, change_of_basis, sigma_min_history are populated by
    downstream modules (None by default).
    """
    n_clusters            : int
    labels                : np.ndarray
    cluster_indices       : list
    centroids_original    : np.ndarray
    centroids_transformed : np.ndarray
    L                     : object
    P                     : np.ndarray
    parameters            : list

    fast_selection    : Optional[dict] = field(default=None)
    change_of_basis   : Optional[dict] = field(default=None)
    sigma_min_history : Optional[dict] = field(default=None)

    # pod_whitened artifacts — needed by the whitened cluster selector.
    # Φ (M_u-orthonormal), per-mode σ, and centroids in raw POD coords.
    pod_modes   : Optional[np.ndarray] = field(default=None)
    pod_sigma   : Optional[np.ndarray] = field(default=None)
    centers_pod : Optional[np.ndarray] = field(default=None)

    def cluster_parameters(self, k):
        return np.array([self.parameters[i] for i in self.cluster_indices[k]])

    def cluster_snapshots(self, S, k):
        return S[:, self.cluster_indices[k]]

    def parameter_centroid(self, k):
        return self.cluster_parameters(k).mean(axis=0)

    def summary(self):
        print(f"ClusteringResult: {self.n_clusters} clusters, "
              f"{len(self.labels)} snapshots")
        for k in range(self.n_clusters):
            params_k = self.cluster_parameters(k)
            re_vals  = [p[0] for p in params_k]
            amp_vals = [p[1] for p in params_k]
            print(f"  Cluster {k}: {len(self.cluster_indices[k]):3d} snapshots | "
                  f"Re ∈ [{min(re_vals):.1f}, {max(re_vals):.1f}]  "
                  f"amp ∈ [{min(amp_vals):.2f}, {max(amp_vals):.2f}]")


# =============================================================================
# Cholesky energy transform
# =============================================================================

def build_cholesky_transform(M):
    """
    Factor M so that  L Lᵀ = P M Pᵀ  (scikit-sparse >= 0.5).

    cholesky() now returns a tuple (R, p) where R is the UPPER triangular
    factor (Rᵀ R = P M Pᵀ) and p is the permutation vector. The old
    Factor.L() returned the LOWER factor, which is Rᵀ — so we transpose to
    keep transform()/inverse_transform() working unchanged.
    """
    R, P = cholesky(sp.csc_matrix(M))
    L = sp.csc_matrix(R).T          # upper R -> lower L,  so Lᵀ = R
    return L, P

def transform(L, P, S):
    """s̃ = L^T P s"""
    return L.T @ S[P]


def inverse_transform(L, P, S_tilde):
    """s = P^T L^{-T} s̃"""
    S = spla.spsolve_triangular(L.T.tocsr(), S_tilde, lower=False)
    P_inv = np.argsort(P)
    return S[P_inv]


def verify_energy_equivalence(S, M, tol=1e-10):
    L, P = build_cholesky_transform(M)
    S_tilde = transform(L, P, S)

    i, j = np.random.choice(S.shape[1], 2, replace=False)
    diff       = S[:, i] - S[:, j]
    diff_tilde = S_tilde[:, i] - S_tilde[:, j]

    d_M = float(diff @ (M @ diff))
    d_E = float(diff_tilde @ diff_tilde)

    print(f"M-norm distance²:            {d_M:.10f}")
    print(f"Euclidean distance² (tilde): {d_E:.10f}")
    print(f"Absolute error:              {abs(d_M - d_E):.2e}")
    return abs(d_M - d_E) < tol


def apply_energy_transform(S, M):
    """Returns (L, P, S_tilde) where S_tilde = L^T P S."""
    if S.shape[0] < S.shape[1]:
        Snaps = S.T
    else:
        Snaps = S

    L, P = build_cholesky_transform(M)
    return L, P, transform(L, P, Snaps)


# =============================================================================
# Snapshot preprocessing
# =============================================================================

def build_energy_snapshots(
    velocity_snapshots, parameters, lifting, problem, inner_product_type,
    homogenize=True,
):
    """
    Preprocess velocity snapshots for clustering and ROM construction.

    Steps:
      1. Assemble M_u (velocity, energy norm) and M_p (pressure, L2).
      2. Optionally subtract the parameter-dependent lifting from
         velocity snapshots (homogeneous BCs).
      3. Apply the Cholesky-based energy transform: s̃ = L^T P s.

    Parameters
    ----------
    homogenize : bool, default True
        If False, snapshots are passed through to the transform without
        lifting subtraction. Diagnostic-only — flow-physics clustering
        requires True when the lifting is parameter-dependent.

    Returns
    -------
    S_homo  : (n_vel_dofs, n_snapshots)
        Homogenized snapshots if homogenize=True, else raw snapshots
        (transposed to column-major).
    S_tilde : (n_vel_dofs, n_snapshots)  Cholesky-whitened snapshots.
    M_u, M_p : sparse SPD mass matrices.
    L, P   : Cholesky factor and permutation of M_u.
    """
    M_u = assemble_inner_product_matrix(problem.velocity_space, inner_product_type)
    M_p = assemble_inner_product_matrix(problem.pressure_space, 'L2')

    if homogenize:
        u_base    = lifting.u_base.dat.data_ro.flatten()
        u_control = lifting.u_control.dat.data_ro.flatten()
        S_homo = homogenize_snapshots(velocity_snapshots, parameters, u_base, u_control)
    else:
        S_homo = velocity_snapshots

    L, P, S_tilde = apply_energy_transform(S_homo, M_u)

    # Ensure S_homo is in (n_dofs, n_snapshots) shape to match S_tilde.
    if S_homo.shape[0] < S_homo.shape[1]:
        S_homo = S_homo.T

    return S_homo, S_tilde, M_u, M_p, L, P


# =============================================================================
# k-means: shared helper
# =============================================================================

def cluster_kmeans(X, n_clusters, n_init=10, random_state=42):
    """
    k-means on a feature matrix X of shape (n_features, n_samples).

    sklearn expects (n_samples, n_features) so X is transposed inside.
    Returns (labels, cluster_centers, inertia) with centers having shape
    (n_clusters, n_features).
    """
    km = KMeans(
        n_clusters=n_clusters,
        init='k-means++',
        n_init=n_init,
        random_state=random_state,
    )
    km.fit(X.T)
    return km.labels_, km.cluster_centers_, km.inertia_


# =============================================================================
# k-means: Cholesky path
# =============================================================================

def build_clustering_result(S_tilde, labels, centers, L, P, parameters):
    """
    Assemble a ClusteringResult from k-means on tilde-space snapshots.

    `centers` is (n_clusters, n_dofs) in tilde space and is inverse-
    transformed to recover DOF-space centroids.
    """
    n_clusters = centers.shape[0]
    cluster_indices = [np.where(labels == k)[0] for k in range(n_clusters)]

    centroids_transformed = centers.T                              # (n_dofs, n_clusters)
    centroids_original = inverse_transform(L, P, centroids_transformed)

    return ClusteringResult(
        n_clusters=n_clusters,
        labels=labels,
        cluster_indices=cluster_indices,
        centroids_original=centroids_original,
        centroids_transformed=centroids_transformed,
        L=L, P=P,
        parameters=parameters,
    )


def build_clustering_result_from_dof_centroids(labels, centers_dofs, L, P, parameters):
    """
    Like build_clustering_result, but for clustering methods that
    produce centroids directly in DOF space (e.g. POD-whitened k-means).

    centers_dofs shape: (n_clusters, n_dofs). Tilde-space centroids
    are computed by forward transform for diagnostic parity.
    """
    n_clusters = centers_dofs.shape[0]
    cluster_indices = [np.where(labels == k)[0] for k in range(n_clusters)]

    centroids_original = centers_dofs.T                            # (n_dofs, n_clusters)
    centroids_transformed = transform(L, P, centroids_original)

    return ClusteringResult(
        n_clusters=n_clusters,
        labels=labels,
        cluster_indices=cluster_indices,
        centroids_original=centroids_original,
        centroids_transformed=centroids_transformed,
        L=L, P=P,
        parameters=parameters,
    )


# =============================================================================
# k-means: POD-whitened path
# =============================================================================

def get_pod_coefficients(matrix, rank=10, M_u=None):
    """
    Compute POD modes of `matrix` and project onto them.

    Returns
    -------
    A           : (rank, n_snapshots) projection coefficients
    eigenvalues : (n_snapshots,) all eigenvalues, descending
    modes       : (n_dofs, rank) POD modes
                  (M-orthonormal if M_u given, Euclidean otherwise)
    """
    modes, eigenvalues = compute_pod_modes(matrix, inner_product=M_u, n_modes=rank)
    A = project_to_basis(matrix, modes, M_u)
    return A, eigenvalues, modes


def mahalanobis_whitening(A, eigenvalues, rank, rel_floor=1e-12):
    """
    Whiten POD coefficients by their per-mode standard deviation:
        A_white[i, :] = A[i, :] / sqrt(λ_i)

    Each row of A_white has unit variance across snapshots, decoupling
    cluster-separation from per-mode energy content.

    Raises if any of the first `rank` eigenvalues falls below
    `rel_floor * max(eigenvalues)` — signals over-truncated rank or
    snapshot conditioning issues.
    """
    lam = eigenvalues[:rank]
    floor = float(eigenvalues.max()) * rel_floor

    if lam.min() < floor:
        raise ValueError(
            f"smallest eigenvalue in first {rank} modes is {lam.min():.3e}, "
            f"below relative floor {floor:.3e} "
            f"(largest eigenvalue: {eigenvalues.max():.3e}); "
            f"reduce rank or check snapshot conditioning"
        )

    sigma = np.sqrt(lam)[:, None]
    return A / sigma


def cluster_kmeans_pod_whitened(S, n_clusters, rank, M_u=None):
    """
    Cluster snapshots in Mahalanobis-whitened POD coefficient space.

    Parameters
    ----------
    S          : (n_dofs, n_snapshots) — typically homogenized snapshots
    n_clusters : int
    rank       : int — POD rank to use for whitening
    M_u        : sparse SPD or None — POD inner product
                 (None = Euclidean POD; pass M_u for energy-norm POD)

    Returns
    -------
    labels       : (n_snapshots,)
    centers_dofs : (n_clusters, n_dofs) — un-whitened, DOF-space centroids
    inertia      : float
    """
    A, eigenvalues, modes = get_pod_coefficients(S, rank=rank, M_u=M_u)
    A_white = mahalanobis_whitening(A, eigenvalues, rank)

    # assert np.allclose(A_white.var(axis=1), 1.0, atol=1e-6), \
    #     "POD whitening did not produce unit variance — POD pipeline inconsistency"

    labels, centers_white, inertia = cluster_kmeans(A_white, n_clusters)
    # centers_white: (n_clusters, rank)

    # Un-whiten: c_pod = c_white * sigma elementwise per mode.
    sigma = np.sqrt(eigenvalues[:rank])                            # (rank,)
    centers_pod = centers_white * sigma                            # (n_clusters, rank)

    # Lift to DOF space: c_dof = Φ c_pod.
    centers_dofs = centers_pod @ modes.T                           # (n_clusters, n_dofs)

    return labels, centers_dofs, inertia, modes, sigma, centers_pod   # <-- extended



# =============================================================================
# Persistence
# =============================================================================

def _cache_key(S_tilde, n_clusters, method, rank):
    """Hash that distinguishes runs by snapshots + clustering settings."""
    h = hashlib.md5()
    h.update(np.ascontiguousarray(S_tilde))
    h.update(str(n_clusters).encode())
    h.update(method.encode())
    h.update(str(rank).encode())
    return h.hexdigest()


def save_clustering_result(clustering, save_dir, S_tilde, n_clusters, method, rank):
    os.makedirs(save_dir, exist_ok=True)

    # Persist pod_whitened artifacts when present, so the whitened selector
    # works on cached (recompute=False) runs without recomputing the POD.
    extra = {}
    if getattr(clustering, "pod_modes", None) is not None:
        extra["pod_modes"]   = clustering.pod_modes
        extra["pod_sigma"]   = clustering.pod_sigma
        extra["centers_pod"] = clustering.centers_pod

    np.savez_compressed(
        os.path.join(save_dir, "clustering.npz"),
        labels=clustering.labels,
        centroids_original=clustering.centroids_original,
        centroids_transformed=clustering.centroids_transformed,
        P=clustering.P,
        **extra,
    )

    for k, idx in enumerate(clustering.cluster_indices):
        np.save(os.path.join(save_dir, f"indices_{k}.npy"), idx)

    sp.save_npz(os.path.join(save_dir, "L.npz"), clustering.L)
    np.save(os.path.join(save_dir, "parameters.npy"), np.array(clustering.parameters))

    meta = {
        "n_clusters": n_clusters,
        "method": method,
        "rank": rank,
        "cache_key": _cache_key(S_tilde, n_clusters, method, rank),
        "n_snapshots": len(clustering.labels),
    }
    with open(os.path.join(save_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Clustering saved to {save_dir}/  (method={method}, rank={rank})")


def load_clustering_result(save_dir):
    data = np.load(os.path.join(save_dir, "clustering.npz"))

    with open(os.path.join(save_dir, "meta.json")) as f:
        meta = json.load(f)

    n_clusters = meta["n_clusters"]
    cluster_indices = [
        np.load(os.path.join(save_dir, f"indices_{k}.npy"))
        for k in range(n_clusters)
    ]
    parameters = [tuple(p) for p in np.load(os.path.join(save_dir, "parameters.npy")).tolist()]
    L = sp.load_npz(os.path.join(save_dir, "L.npz"))

    result = ClusteringResult(
        n_clusters=n_clusters,
        labels=data["labels"],
        cluster_indices=cluster_indices,
        centroids_original=data["centroids_original"],
        centroids_transformed=data["centroids_transformed"],
        L=L,
        P=data["P"],
        parameters=parameters,
    )

    # Restore pod_whitened artifacts if they were saved.
    if "pod_modes" in data.files:
        result.pod_modes   = data["pod_modes"]
        result.pod_sigma   = data["pod_sigma"]
        result.centers_pod = data["centers_pod"]

    return result


def clustering_is_valid(save_dir, S_tilde, n_clusters, method, rank):
    meta_path = os.path.join(save_dir, "meta.json")
    if not os.path.exists(meta_path):
        return False, "No saved clustering found"

    with open(meta_path) as f:
        meta = json.load(f)

    if meta.get("n_clusters") != n_clusters:
        return False, f"n_clusters changed: {meta.get('n_clusters')} → {n_clusters}"
    if meta.get("method") != method:
        return False, f"method changed: {meta.get('method')} → {method}"
    if meta.get("rank") != rank:
        return False, f"rank changed: {meta.get('rank')} → {rank}"
    if meta.get("cache_key") != _cache_key(S_tilde, n_clusters, method, rank):
        return False, "snapshot or settings hash changed"

    return True, "Cache valid"


def compute_or_load_clustering(
    S_tilde, S_homo, M_u,
    parameters, L, P,
    save_dir, n_clusters,
    method="cholesky",
    rank=None,
    recompute=False,
):
    """
    Compute (or load) a clustering of the snapshots.

    Parameters
    ----------
    S_tilde     : (n_dofs, n_snapshots) Cholesky-whitened snapshots
                  (used for cache keying and the cholesky-method k-means)
    S_homo      : (n_dofs, n_snapshots) homogenized DOF-space snapshots
                  (used by the pod_whitened method for POD with M_u)
    M_u         : sparse SPD — velocity inner product (energy norm)
    parameters  : list of (Re, amp) tuples
    L, P        : Cholesky factor and permutation of M_u
    save_dir    : str
    n_clusters  : int
    method      : "cholesky" | "pod_whitened"
    rank        : int — required if method="pod_whitened"
    recompute   : bool

    Returns
    -------
    clustering : ClusteringResult
    reason     : str
    """
    valid, reason = clustering_is_valid(save_dir, S_tilde, n_clusters, method, rank)

    if recompute or not valid:
        if method == "cholesky":
            labels, centers, _ = cluster_kmeans(S_tilde, n_clusters)
            clustering = build_clustering_result(
                S_tilde, labels, centers, L, P, parameters
            )

        elif method == "pod_whitened":
            if rank is None:
                raise ValueError("method='pod_whitened' requires rank")

            labels, centers_dofs, _, modes, sigma, centers_pod = cluster_kmeans_pod_whitened(
                S_homo, n_clusters, rank, M_u=M_u
            )
            clustering = build_clustering_result_from_dof_centroids(labels, centers_dofs, L, P, parameters)
            clustering.pod_modes   = modes        # (n_dofs, rank), M_u-orthonormal
            clustering.pod_sigma   = sigma        # (rank,)
            clustering.centers_pod = centers_pod  # (K, rank)


        else:
            raise ValueError(f"unknown clustering method: {method!r}")

        save_clustering_result(clustering, save_dir, S_tilde, n_clusters, method, rank)
    else:
        clustering = load_clustering_result(save_dir)

    return clustering, reason
