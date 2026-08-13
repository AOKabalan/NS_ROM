"""POD characterization (A1-A3). Pure numeric; no @slow / @firedrake objects."""
import numpy as np

from tests import characterization as ch


def test_A1_pod_euclidean(S, golden):
    g = golden("A1")
    cap = ch.case_A1_pod_euclidean(S)
    # Eigenvalues come from scipy's symmetric eig (deterministic). Only BLAS
    # threading perturbs the low bits, so rtol=1e-6 is orders of magnitude below
    # any physically meaningful change while still catching a real regression.
    np.testing.assert_allclose(cap["eigenvalues"], g["eigenvalues"], rtol=1e-6)
    assert cap["n_modes"] == g["n_modes"]
    # Reconstruction error is a derived scalar (~1e-2); slightly looser rtol.
    assert np.isclose(cap["recon_error_at_full"], g["recon_error_at_full"], rtol=1e-5)
    # Orthonormality residual is ~1e-13 (near machine zero) -> absolute bound.
    assert cap["orthonormality_max_err"] < 1e-9


def test_A2_pod_energy(S, M_u, golden):
    g = golden("A2")
    cap = ch.case_A2_pod_energy(S, M_u)
    np.testing.assert_allclose(cap["eigenvalues"], g["eigenvalues"], rtol=1e-6)
    assert cap["n_modes"] == g["n_modes"]
    assert np.isclose(cap["recon_error_at_full"], g["recon_error_at_full"], rtol=1e-5)
    assert cap["orthonormality_max_err"] < 1e-9


def test_A3_energy_truncation(S, M_u, golden):
    g = golden("A3")
    cap = ch.case_A3_pod_energy_truncation(S, M_u)
    assert cap["n_modes_total"] == g["n_modes_total"]
    for key, val in g.items():
        if key.startswith("n_retained@"):
            # Retained-mode counts are integers -> exact match.
            assert cap[key] == val, key
        elif key.startswith("energy_frac@"):
            # Cumulative energy fraction is essentially exact (deterministic sum).
            assert np.isclose(cap[key], val, rtol=1e-9), key
