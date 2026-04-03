# setup_investigation.py
import numpy as np
from nsrom.navier_stokes import setup_navier_stokes_problem
from nsrom.snapshot_collection import load_snapshot_dofs
from nsrom.lifting_functions import load_lifting_functions
from nsrom.rom import assemble_inner_product_matrix, homogenize_snapshots, compute_pod_modes, project_to_basis

SNAPSHOT_DIR   = "multi_param_multi_branch"
LIFTING_DIR    = "lifting"
MESH_FILE      = "mesh/mid_pinball.msh"
REYNOLDS_INIT  = 100.0
AMPLITUDE_INIT = 0.0
INNER_PRODUCT  = "H1"
POD_RANK       = 10

def load_everything():
    problem = setup_navier_stokes_problem(MESH_FILE, REYNOLDS_INIT, AMPLITUDE_INIT)
    data = load_snapshot_dofs(SNAPSHOT_DIR)
    lifting = load_lifting_functions(LIFTING_DIR, problem)

    S = data['velocity'].T
    parameters = np.array(data['parameters'])
    branch_ids = np.array(data['branch_ids'])
    M_u = assemble_inner_product_matrix(problem.velocity_space, INNER_PRODUCT)

    S_centered = S - S.mean(axis=1, keepdims=True)

    modes, eigenvalues = compute_pod_modes(S, M_u, n_modes=POD_RANK)
    modes_c, eigenvalues_c = compute_pod_modes(S_centered, M_u, n_modes=POD_RANK)

    A = project_to_basis(S, modes, M_u)
    A_c = project_to_basis(S_centered, modes_c, M_u)

    return dict(
        problem=problem, S=S, S_centered=S_centered,
        parameters=parameters, branch_ids=branch_ids, M_u=M_u,
        modes=modes, eigenvalues=eigenvalues, A=A,
        modes_c=modes_c, eigenvalues_c=eigenvalues_c, A_c=A_c,
        velocity=data['velocity'], pressure=data['pressure'],
    )