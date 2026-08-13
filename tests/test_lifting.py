"""Lifting characterization (B1). Firedrake objects; slow."""
import numpy as np
import pytest

from tests import characterization as ch


@pytest.mark.firedrake
@pytest.mark.slow
def test_B1_stored_lifting_norms(problem, golden):
    g = golden("B1")
    cap = ch.case_B1_lifting(problem)
    # Norms of the STORED lifting functions (deterministic artifact on disk).
    # NB: compute_lifting_functions cannot regenerate these from a cold start;
    # see docs/known-issues.md #2.
    for key in ("norm_u_base_L2", "norm_u_control_L2",
                "norm_u_base_H1", "norm_u_control_H1"):
        assert np.isclose(cap[key], g[key], rtol=1e-6), key
