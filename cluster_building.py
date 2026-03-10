"""
Cholesky-based transform for energy-norm k-means clustering.

Given an SPD mass matrix M, factors M = L Lᵀ and provides:
    forward:  s̃ = Lᵀ s       (transforms to Euclidean-equivalent space)
    inverse:  s  = L⁻ᵀ s̃     (maps back to original DOF space)

This allows standard k-means on {s̃} to minimise the M-norm objective:

    Σᵢ min_j ||sᵢ - cⱼ||²_M  =  Σᵢ min_j ||s̃ᵢ - c̃ⱼ||²

Requires: scikit-sparse  (sudo apt install libsuitesparse-dev && pip install scikit-sparse)
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from sksparse.cholmod import cholesky
from sklearn.cluster import KMeans
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import hashlib
import os
@dataclass
class ClusteringResult:
    """
    Holds everything produced by the clustering step.

    Attributes
    ----------
    n_clusters : int
    labels : (n_snapshots,)
        Cluster index for each snapshot.
    cluster_indices : list of arrays
        cluster_indices[k] = indices of snapshots in cluster k.
    centroids_original : (n_dofs, n_clusters)
        Cluster centroids in original DOF space.
    centroids_transformed : (n_dofs, n_clusters)
        Cluster centroids in transformed (L^T P) space.
    L : sparse lower-triangular
        Cholesky factor of M.
    P : (n_dofs,) int array
        Fill-reducing permutation from CHOLMOD.
    parameters : list of (Re, amp) tuples
        Full parameter list, length n_snapshots.

    # --- Group 2: fast online cluster selection ---
    fast_selection : dict or None
        Precomputed a, b, c_ctrl, P_i, Q_i for reduced-coordinate selection.

    # --- Group 3: change-of-basis matrices ---
    change_of_basis : dict or None
        CB[(i,j)] = Z_j^T M Z_i  for all cluster pairs (i,j).

    # --- Group 4: bifurcation detection ---
    sigma_min_history : dict or None
        sigma_min_history[k] = list of sigma_min values along the sweep
        for cluster k. A drop toward zero signals a bifurcation.
    """
    n_clusters        : int
    labels            : np.ndarray
    cluster_indices   : list
    centroids_original    : np.ndarray
    centroids_transformed : np.ndarray
    L                 : object           # scipy sparse
    P                 : np.ndarray
    parameters        : list

    # Future groups — None until built
    fast_selection    : Optional[dict] = field(default=None)
    change_of_basis   : Optional[dict] = field(default=None)
    sigma_min_history : Optional[dict] = field(default=None)

    def cluster_parameters(self, k):
        """Return (Re, amp) pairs for snapshots in cluster k."""
        return np.array([self.parameters[i] for i in self.cluster_indices[k]])

    def cluster_snapshots(self, S, k):
        """Extract snapshot columns belonging to cluster k from S (n_dofs, n_snapshots)."""
        return S[:, self.cluster_indices[k]]
    def parameter_centroid(self, k):
        """Mean (Re, amp) of snapshots in cluster k."""
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


def _hash_array(arr):
    """Fast hash of a numpy array for cache invalidation."""
    return hashlib.md5(np.ascontiguousarray(arr)).hexdigest()


def save_clustering_result(clustering, save_dir, S_tilde, n_clusters):
    """Save ClusteringResult to disk."""
    os.makedirs(save_dir, exist_ok=True)

    # Arrays
    np.savez_compressed(
        os.path.join(save_dir, "clustering.npz"),
        labels=clustering.labels,
        centroids_original=clustering.centroids_original,
        centroids_transformed=clustering.centroids_transformed,
        P=clustering.P,
    )

    # Cluster indices (variable length)
    for k, idx in enumerate(clustering.cluster_indices):
        np.save(os.path.join(save_dir, f"indices_{k}.npy"), idx)

    # Sparse L
    from scipy.sparse import save_npz
    save_npz(os.path.join(save_dir, "L.npz"), clustering.L)

    # Parameters
    np.save(os.path.join(save_dir, "parameters.npy"), np.array(clustering.parameters))

    # Metadata for cache invalidation
    import json
    meta = {
        "n_clusters": n_clusters,
        "S_tilde_hash": _hash_array(S_tilde),
        "n_snapshots": len(clustering.labels),
    }
    with open(os.path.join(save_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Clustering saved to {save_dir}/")


def load_clustering_result(save_dir):
    """Load ClusteringResult from disk."""
    from scipy.sparse import load_npz
    import json

    data = np.load(os.path.join(save_dir, "clustering.npz"))

    with open(os.path.join(save_dir, "meta.json")) as f:
        meta = json.load(f)

    n_clusters = meta["n_clusters"]

    cluster_indices = [
        np.load(os.path.join(save_dir, f"indices_{k}.npy"))
        for k in range(n_clusters)
    ]

    parameters = np.load(os.path.join(save_dir, "parameters.npy")).tolist()
    parameters = [tuple(p) for p in parameters]

    L = load_npz(os.path.join(save_dir, "L.npz"))

    return ClusteringResult(
        n_clusters=n_clusters,
        labels=data["labels"],
        cluster_indices=cluster_indices,
        centroids_original=data["centroids_original"],
        centroids_transformed=data["centroids_transformed"],
        L=L,
        P=data["P"],
        parameters=parameters,
    )


def clustering_is_valid(save_dir, S_tilde, n_clusters):
    """
    Check if saved clustering matches current S_tilde and n_clusters.
    Returns True if cache is valid, False if recomputation is needed.
    """
    import json

    meta_path = os.path.join(save_dir, "meta.json")
    if not os.path.exists(meta_path):
        return False, "No saved clustering found"

    with open(meta_path) as f:
        meta = json.load(f)

    if meta["n_clusters"] != n_clusters:
        return False, f"n_clusters changed: {meta['n_clusters']} → {n_clusters}"

    if meta["S_tilde_hash"] != _hash_array(S_tilde):
        return False, "S_tilde changed (snapshots or mass matrix changed)"

    return True, "Cache valid"

def build_clustering_result(S_tilde, labels, centers, L, P, parameters):
    """
    Assemble a ClusteringResult from the raw k-means outputs.

    Parameters
    ----------
    S_tilde    : (n_dofs, n_snapshots)  transformed snapshots
    labels     : (n_snapshots,)         cluster index per snapshot
    centers    : (n_clusters, n_dofs)   centroids in transformed space (sklearn output)
    L, P       : Cholesky factor and permutation
    parameters : list of (Re, amp)

    Returns
    -------
    ClusteringResult
    """

    n_clusters = centers.shape[0]

    # Per-cluster snapshot indices
    cluster_indices = [
        np.where(labels == k)[0] for k in range(n_clusters)
    ]

    # Centroids: sklearn gives (n_clusters, n_dofs), we store (n_dofs, n_clusters)
    centroids_transformed = centers.T

    # Map centroids back to original DOF space
    centroids_original = inverse_transform(L, P, centroids_transformed)

    return ClusteringResult(
        n_clusters=n_clusters,
        labels=labels,
        cluster_indices=cluster_indices,
        centroids_original=centroids_original,
        centroids_transformed=centroids_transformed,
        L=L,
        P=P,
        parameters=parameters,
    )

def build_cholesky_transform(M):
    factor = cholesky(sp.csc_matrix(M))
    L = factor.L()
    P = factor.P()          # permutation vector: P M P^T = L L^T
    return L, P

def transform(L, P, S):
    # s̃ = L^T P s
    return L.T @ S[P]       # apply permutation first, then L^T

def inverse_transform(L, P, S_tilde):
    # s = P^T L^{-T} s̃
    S = spla.spsolve_triangular(L.T.tocsr(), S_tilde, lower=False)
    P_inv = np.argsort(P)   # inverse permutation
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
    return abs(d_M - d_E)  < tol
def apply_energy_transform(S,M):
    if S.shape[0] < S.shape[1]:
        Snaps =S.T
    else:
        Snaps = S

    L,P = build_cholesky_transform(M)
    return L,P,transform(L,P,Snaps)

def cluster_kmeans(S_tilde, n_clusters, n_init=10, random_state=42):
    km = KMeans(
        n_clusters=n_clusters,
        init='k-means++',
        n_init=n_init,
        random_state=random_state,
    )
    km.fit(S_tilde.T)   # sklearn expects (n_samples, n_features)

    return km.labels_, km.cluster_centers_, km.inertia_
def select_cluster(clustering, Re, amp):
    query    = np.array([Re, amp])
    centers  = np.array([clustering.parameter_centroid(k)
                         for k in range(clustering.n_clusters)])
    distances = np.linalg.norm(centers - query, axis=1)
    best = int(np.argmin(distances))

    print(f"  Query (Re={Re}, amp={amp})")
    for k, d in enumerate(distances):
        marker = " ←" if k == best else ""
        print(f"  Cluster {k}: distance={d:.4f}{marker}")
    return best
