"""
Offline: build one exact convection tensor per cluster and verify it.

Run ONCE. For each cluster it assembles N = Z^T J(phi_j) Z (r full-mesh
Jacobian assemblies), saves it, and verifies against a fresh exact assembly.
The verify MUST pass (~1e-12) before you trust the online tensor path.
"""

import numpy as np
from nsrom.tensor_convection import (
    build_convection_tensor, save_convection_tensor, TensorConvection,
    verify_against_exact,
)

# ---- adapt these to your project ----
K = 4
ROOT = "local_rom/K4_H1"

def load_cluster_Zu(k):
    """Return the SAME Z_u array the solver puts in callback_data['Z_u']
    for cluster k (velocity basis INCLUDING supremizer columns)."""
    raise NotImplementedError("wire to your loader")

def get_velocity_space():
    """Return the full-mesh VectorFunctionSpace (P2 velocity)."""
    raise NotImplementedError("wire to your problem setup")
# -------------------------------------

def main():
    Vv = get_velocity_space()
    for k in range(K):
        print(f"\n=== cluster {k} ===")
        Z_u = load_cluster_Zu(k)
        N = build_convection_tensor(Z_u, Vv)
        save_convection_tensor(N, f"{ROOT}/cluster_{k}/conv_tensor.npz")

        tc = TensorConvection(N)
        r = Z_u.shape[1]
        # Verify on a random state AND (ideally) a real solution's alpha.
        verify_against_exact(tc, Z_u, Vv, np.random.randn(r))
        # e.g. also: verify_against_exact(tc, Z_u, Vv, load_solution_alpha(k))

if __name__ == "__main__":
    main()
