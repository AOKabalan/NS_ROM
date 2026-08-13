"""Short ROM-solve characterization (B2). Firedrake objects; slow."""
import numpy as np
import pytest

from tests import characterization as ch


@pytest.mark.firedrake
@pytest.mark.slow
def test_B2_short_rom_solve(problem, golden):
    g = golden("B2")
    cap = ch.case_B2_rom_solve(problem)
    # Genuine convergence, not a cap-hit: converged AND finished strictly under
    # the Newton cap (snes_max_it=10; the solve takes ~3 iterations).
    assert cap["converged"] is True
    assert cap["iterations"] < cap["snes_max_it"]
    # Cluster selection is deterministic for this (Re, amp).
    assert cap["cluster_selected"] == g["cluster_selected"]
    # Reconstructed velocity norm is basis-sign-invariant and physical.
    assert np.isclose(cap["recon_velocity_L2"], g["recon_velocity_L2"], rtol=1e-5)
