"""Isolate the DEIM error at ONE state into three orthogonal components.

The exact-mode branch of rom_callbacks.py defines the ground-truth online
convention: F(u') with u' = Z_u @ alpha (quadratic_only), assembled on the
full mesh. Against that we measure, at a chosen exact-ROM solution:

  (A) FULL-VECTOR DEIM error      red_exact  vs  deim_ops.reduced_convective(F_full)
        -> span + sampling error of the offline operators, NO submesh involved
  (B) SUBMESH vs FULL sample values   F_sub[sub_idx]  vs  F_full[idx]
        -> pure submesh assembly error, NO approximation theory involved
  (C) TRAINING-SPAN floor         || (I - U U^T) F_full || / ||F_full||
        -> can the training basis represent this F at all (full available rank)

Verdict table:
  B large                 -> submesh assembly bug (hyper_reduction.py / Submesh)
  B tiny, A large, C large -> offline/online convention mismatch: the training
                             snapshots were NOT F(fluctuation) with the online
                             theta convention. Go diff the snapshot generator.
  B tiny, A large, C tiny  -> operator build bug (indices/ordering) -- unlikely
                             after the full-rank run, but the table will say so.
  everything tiny at the failing amp -> the quadratic term is innocent; the
                             inconsistency is in the affine part (thetas) and
                             we check that separately below (theta check).

Run it at one WORKING amp (0.0) and one FAILING amp (-0.5): the comparison
of the two rows is the diagnosis.

Wiring: needs the exact-ROM states from diagnose_spine's Phase A1
(exact_states list) plus Z_u for the cluster (same array the solver puts in
callback_data['Z_u']) and, for the theta check, the online thetas + lifting
DOFs at the given amp (same objects callback_data carries in full mode).
"""

import numpy as np
import scipy.sparse as sp
from firedrake import (
    Function, TestFunction, TrialFunction, inner, grad, dx, assemble,
)
from nsrom.rom import load_pod_basis
def _rel(a, b):
    return np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30)


def isolate_deim_error(
    u_coeffs,            # reduced velocity coeffs of the exact-ROM solution
    amp, Z_u,            # amp of the state; (n_dofs, N_u) enriched basis
    deim_ops, submesh_deim, problem, layout,
    thetas=None, lifting_dofs=None, full_velocity=None,
    label="",
):
    V_full = problem.velocity_space
    u_work = Function(V_full)
    v_test = TestFunction(V_full)
    u_trial = TrialFunction(V_full)

    N_u = Z_u.shape[1]
    alpha = np.asarray(u_coeffs)[:N_u]

    # ---- the fluctuation EXACTLY as the quadratic_only online path builds it
    u_np = Z_u @ alpha
    u_work.dat.data[:] = u_np.reshape(u_work.dat.data.shape)

    print(f"\n===== isolate_deim_error {label} (amp={amp:+.3f}) =====")

    # ---- optional theta check: does Z_u@alpha equal full_vel - sum(theta*lift)?
    # Catches a mismatch between _reconstruct_velocity's convention and the
    # convention that produced the stored solution / snapshots.
    if thetas is not None and lifting_dofs is not None and full_velocity is not None:
        flu2 = (np.asarray(full_velocity)
                - np.einsum('i,ij->j', np.asarray(thetas),
                            np.asarray(lifting_dofs)))
        e_theta = _rel(u_np, flu2)
        print(f"  [theta] ||Z_u a - (u_full - sum theta_k lift_k)|| rel = "
              f"{e_theta:.2e}  "
              f"({'OK' if e_theta < 1e-8 else 'CONVENTION MISMATCH'})")

    # =====================================================================
    # Ground truth: exact-mode assembly (verbatim the callback's else-branch)
    # =====================================================================
    F_full = assemble(
        inner(grad(u_work) * u_work, v_test) * dx
    ).dat.data_ro.flatten()
    red_exact = Z_u.T @ F_full

    # ---- (A) full-vector DEIM: span + sampling, no submesh ----------------
    red_deim_full = deim_ops.reduced_convective(F_full)
    eA = _rel(red_deim_full, red_exact)

    # ---- (B) submesh path verbatim, compared at the raw sample values -----
    sm = submesh_deim
    sm.transfer_to_submesh(u_work)
    F_sub = assemble(inner(grad(sm.u_sub) * sm.u_sub, sm.v_test) * dx)
    samp_full = F_full[deim_ops.indices_F]
    samp_sub = F_sub.dat.data_ro.flatten()[sm.sub_indices_F]
    eB = _rel(samp_sub, samp_full)
    red_deim_sub = sm.extract_residual(F_sub)
    eB_red = _rel(red_deim_sub, red_deim_full)

    # ---- (C) training-span floor at FULL available rank -------------------
    # basis = load_pod_basis(deim_ops.metadata.get('basis_path', ''))
    basis = np.load( layout.deim_basis(1))


    U = basis['modes_F']
    keep = np.linalg.norm(U, axis=0) > 1e-10
    Uq, _ = np.linalg.qr(U[:, keep])
    eC = _rel(F_full, Uq @ (Uq.T @ F_full))  # == ||(I-UU^T)F||/||F||

    print(f"  F: (A) full-vector DEIM vs exact reduced : {eA:.2e}")
    print(f"     (B) submesh vs full at sample DOFs    : {eB:.2e}"
          f"   (reduced-space: {eB_red:.2e})")
    print(f"     (C) training-span floor, full rank    : {eC:.2e}")

    # =====================================================================
    # Jacobian analogues
    # =====================================================================
    J_ass = assemble(
        (inner(grad(u_trial) * u_work, v_test)
         + inner(grad(u_work) * u_trial, v_test)) * dx,
        mat_type='aij')
    indptr, cols, J_data = J_ass.M.handle.getValuesCSR()

    # pattern check against the training-time sparsity stored in the basis
    same_pattern = (len(indptr) == len(basis['indptr'])
                    and np.array_equal(indptr, basis['indptr'])
                    and np.array_equal(cols, basis['col_indices']))
    print(f"  J: sparsity pattern online == training  : "
          f"{'YES' if same_pattern else 'NO  <-- indices invalid, stop here'}")

    if same_pattern:
        J_sp = sp.csr_matrix((J_data, cols, indptr),
                             shape=(len(indptr) - 1, len(indptr) - 1))
        redJ_exact = Z_u.T @ (J_sp @ Z_u)
        redJ_deim_full = deim_ops.reduced_jacobian(J_data)
        eAJ = _rel(redJ_deim_full, redJ_exact)

        J_sub = assemble(
            (inner(grad(sm.u_trial) * sm.u_sub, sm.v_test)
             + inner(grad(sm.u_sub) * sm.u_trial, sm.v_test)) * dx,
            mat_type='aij')
        _, _, J_data_sub = J_sub.M.handle.getValuesCSR()
        sampJ_full = J_data[deim_ops.indices_J]
        sampJ_sub = J_data_sub[sm.sub_indices_J]
        eBJ = _rel(sampJ_sub, sampJ_full)

        U_J = basis['modes_J']
        keepJ = np.linalg.norm(U_J, axis=0) > 1e-10
        UqJ, _ = np.linalg.qr(U_J[:, keepJ])
        eCJ = _rel(J_data, UqJ @ (UqJ.T @ J_data))

        print(f"     (A) full-vector MDEIM vs exact       : {eAJ:.2e}")
        print(f"     (B) submesh vs full at sample entries: {eBJ:.2e}")
        print(f"     (C) training-span floor, full rank   : {eCJ:.2e}")

    return dict(eA=eA, eB=eB, eB_red=eB_red, eC=eC)


if __name__ == "__main__":
    # After running diagnose_spine Phase A1 (exact_states), for cluster k=1:
    #
    # amp0   = exact_states[0]   # (amp, k, full, u_coeffs, p_coeffs) at  0.0
    # amp05  = exact_states[5]   # at -0.5
    # Z_u    = <same array solve_rom puts in callback_data['Z_u'] for k=1>
    # sm     = submesh_deims[1]
    # ops    = deim_ops_dict[1]
    #
    # for st, lbl in [(amp0, "WORKING"), (amp05, "FAILING")]:
    #     amp, k, full, uc, pc = st
    #     isolate_deim_error(uc, amp, Z_u, ops, sm, problem,layout,
    #                        full_velocity=full.velocity.dat.data_ro.flatten(),
    #                        label=lbl)
    pass
