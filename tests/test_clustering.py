"""Clustering + energy-transform characterization (A5-A7). Pure numeric."""
import numpy as np
from sklearn.metrics import adjusted_rand_score

from tests import characterization as ch


def _assert_partition(cap, g):
    # Cluster sizes are permutation-invariant when sorted -> exact integer match.
    assert cap["sorted_cluster_sizes"] == g["sorted_cluster_sizes"]
    # Inertia is deterministic (seeded k-means, random_state=42); a tight
    # tolerance still flags a genuine repartition.
    assert np.isclose(cap["inertia"], g["inertia"], rtol=1e-6, atol=1e-6)
    # Labels are only defined up to a permutation of cluster IDs, so compare via
    # adjusted Rand index against the baseline labels: ARI == 1.0 iff the
    # partition is identical up to relabeling.
    assert adjusted_rand_score(g["labels"], cap["labels"]) >= 1.0 - 1e-9


def test_A5_clustering_cholesky(S, M_u, snapshots, golden):
    g = golden("A5")
    cap = ch.case_A5_clustering_cholesky(S, M_u, snapshots["branch_ids"])
    _assert_partition(cap, g)
    # Records that the partition does NOT track branch identity (ARI ~ 0.014).
    # See docs/known-issues.md #4.
    assert np.isclose(
        cap["adjusted_rand_vs_branch"], g["adjusted_rand_vs_branch"], atol=1e-6
    )


def test_A6_clustering_whitened(S, M_u, golden):
    g = golden("A6")
    cap = ch.case_A6_clustering_whitened(S, M_u)
    _assert_partition(cap, g)


def test_A7_energy_transform_roundtrip(S, M_u):
    cap = ch.case_A7_transform_roundtrip(S, M_u)
    # Cholesky round-trip is exact up to machine precision.
    assert cap["max_roundtrip_abs_residual"] < 1e-10
    assert np.isclose(cap["energy_ratio_tilde_over_M"], 1.0, rtol=1e-9)
