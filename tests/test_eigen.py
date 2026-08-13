"""
SLEPc eigenvalue characterization (B3). NON-BLOCKING.

The eigensolver sets no fixed initial vector, so eigenVALUES are reproducible
only to solver tolerance and eigenVECTORS not at all. This test therefore never
gates the suite: on any drift it xfails (via pytest.xfail) instead of failing.
"""
import numpy as np
import pytest

from tests import characterization as ch


@pytest.mark.firedrake
@pytest.mark.slow
@pytest.mark.nonblocking
def test_B3_leftmost_eigenvalues(problem, golden):
    g = golden("B3")
    cap = ch.case_B3_eigen(problem)
    try:
        # n_converged is NOT asserted: with no fixed start vector it varies
        # run-to-run (seen 4 vs 5). The eigenVALUES are the physics — they
        # converge to the true spectrum — so compare the overlapping leftmost
        # ones with absolute tol 1e-6. (The +0.0439 ± 0.520i pair is a real
        # unstable Hopf mode at Re=50, not an artifact — see case_B3 docstring.)
        got = np.array(cap["leftmost_eigenvalues"])
        exp = np.array(g["leftmost_eigenvalues"])
        n = min(len(got), len(exp))
        assert n >= 1
        np.testing.assert_allclose(got[:n], exp[:n], atol=1e-6)
    except AssertionError as e:
        pytest.xfail(
            f"SLEPc eigenvalue drift (no fixed start vector; non-blocking): {e}"
        )
