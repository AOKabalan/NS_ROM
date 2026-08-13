"""DEIM characterization (A4). Pure numeric."""
import numpy as np

from tests import characterization as ch


def test_A4_deim_indices(S, snapshots, golden):
    g = golden("A4")
    cap = ch.case_A4_deim(S, snapshots["holdout"])
    # DEIM greedy selection is deterministic (argmax over |residual|), and the
    # indices are integers -> exact match. This is the primary golden.
    assert cap["indices"] == g["indices"]
    # Conditioning of the point matrix P^T U (deterministic linear algebra).
    assert np.isclose(cap["cond_PtU"], g["cond_PtU"], rtol=1e-6)
    # Held-out reconstruction error: the discriminating quantity (~2.6%).
    assert np.isclose(
        cap["holdout_recon_rel_error"], g["holdout_recon_rel_error"], rtol=1e-5
    )
