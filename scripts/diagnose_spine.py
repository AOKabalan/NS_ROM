"""Diagnose WHERE the DEIM approximation breaks along the amp-spine.

Built on conv_test.py's bare continuation. Three phases, in order of
increasing wiring effort and decreasing decisiveness-per-minute:

  Phase A  (no new wiring needed)
      Walk the spine with EXACT reduced assembly (use_deim=False),
      storing the converged full-space solution at every amp. Then, at
      each amp, project that exact solution into the cluster basis and
      run the DEIM solve seeded AT THE ANSWER.
        -> converges in ~1-3 its, tiny drift : DEIM equations are fine at
           the solution; failure is path/basin (off-manifold iterates).
        -> diverges / drifts from the answer : the DEIM-reconstructed
           residual is wrong AT the solution despite training coverage.

  Phase B  (needs two assembly hooks, marked TODO below)
      At each exact converged state, assemble the full convective vector
      F(u; amp) and Jacobian data J(u; amp) with THE SAME routine that
      generated the DEIM training snapshots, and decompose the error:
        e_proj  = ||(I - U U^T) F|| / ||F||         (basis floor)
        e_deim  = ||F - U (P^T U)^+ P^T F|| / ||F|| (floor x amplification)
        e_red   = error in the actual reduced quantity the solver uses
      If e_proj is already large: coverage/truncation. If e_proj is tiny
      but e_deim is large: point-selection/amplification. If both tiny at
      converged states: the problem lives on the Newton path, not at the
      solutions (consistent with Phase A converging).

  Phase C  (same hooks)
      Evaluate the same metrics at the MISMATCH states the continuation
      actually feeds Newton: previous-amp solution + current-amp lifting.
      This is the first iterate of every continuation step.

Sanity anchor: Phase B is also evaluated at one TRAINING snapshot, where
e_proj must be ~ POD tolerance (1e-11-ish). If it is not, the assembly
hooks do not match the training pipeline -- which would itself be the bug.
"""

import time
import numpy as np
import scipy.sparse as sp

from nsrom.navier_stokes import compute_forces
from nsrom.rom import (
    solve_rom,
    reconstruct_solution,
    project_to_rom_coefficients,
    SubMeshDEIM,
)
from nsrom.rom.local import select_cluster, _whitened_pod_distances

# load_basis lives next to the DEIM code; adjust import if the path differs
# try:
#     from nsrom.rom.deim import load_basis
# except ImportError:  # fallback if the module layout differs
#     from nsrom.rom import load_basis  # noqa


# ===========================================================================
# TODO: wire these two hooks to the SAME assembly used to generate the DEIM
# training snapshots (the routine that produced modes_F / modes_J inputs).
# Signature contract:
#   u_full : (n_dofs,) full velocity DOF vector (lifted total field or
#            fluctuation -- WHICHEVER the training snapshots used; be exact)
#   amp    : control amplitude (sets the lifting / BCs exactly as training)
#   nu     : viscosity 1/Re
# Returns:
#   assemble_F_full -> (n_dofs,) convective residual vector, same convention
#                      (sign, lifting handling, nu scaling) as training
#   assemble_J_full -> (nnz,) CSR data of the convective Jacobian on the
#                      SAME sparsity pattern (indptr/col_indices) stored in
#                      the DEIM basis file
# ===========================================================================
def assemble_F_full(u_full, amp, nu, problem):
    raise NotImplementedError(
        "wire to the training-snapshot assembly, e.g. "
        "nsrom.rom.deim.assemble_convective_vector(...)")


def assemble_J_full(u_full, amp, nu, problem):
    raise NotImplementedError(
        "wire to the training-snapshot assembly, e.g. "
        "nsrom.rom.deim.assemble_convective_jacobian_data(...)")


# ---------------------------------------------------------------------------
# metric helpers
# ---------------------------------------------------------------------------
def _orthonormal(U, tol=1e-8):
    G = U.T @ U
    return np.linalg.norm(G - np.eye(U.shape[1])) < tol * np.sqrt(U.shape[1])


def vector_metrics(F, U, idx, reduced_exact=None, reduced_deim=None):
    """Error decomposition for one function snapshot F against basis U
    sampled at idx. Returns dict of relative errors."""
    nF = np.linalg.norm(F)
    if not _orthonormal(U):
        # projection needs an orthonormal basis; orthonormalize a copy
        U, _ = np.linalg.qr(U)
    # basis floor
    e_proj = np.linalg.norm(F - U @ (U.T @ F)) / nF
    # DEIM / gappy-LSQ reconstruction (pinv covers square and tall alike)
    Up = U[idx, :]
    F_rec = U @ (np.linalg.pinv(Up) @ F[idx])
    e_deim = np.linalg.norm(F - F_rec) / nF
    out = {"e_proj": e_proj, "e_deim": e_deim,
           "amplif": e_deim / max(e_proj, 1e-16)}
    if reduced_exact is not None and reduced_deim is not None:
        out["e_red"] = (np.linalg.norm(reduced_deim - reduced_exact)
                        / max(np.linalg.norm(reduced_exact), 1e-16))
    return out


def jacobian_metrics(J_data, basis, ops, V):
    """Same decomposition for the Jacobian data vector, plus the error in
    the reduced matrix the Newton solver actually uses."""
    U_J = basis["modes_J"][:, :ops.J_r_modes.shape[0]]
    base = vector_metrics(J_data, U_J, ops.indices_J)
    # reduced-matrix error: MDEIM J_r vs exact V^T J V
    J_full = sp.csr_matrix((J_data, basis["col_indices"], basis["indptr"]),
                           shape=(basis["n_dofs"], basis["n_dofs"]))
    Jr_exact = V.T @ (J_full @ V)
    Jr_mdeim = ops.reduced_jacobian(J_data)
    base["e_red"] = (np.linalg.norm(Jr_mdeim - Jr_exact)
                     / np.linalg.norm(Jr_exact))
    return base


# ---------------------------------------------------------------------------
# Phase A driver: exact walk + start-at-answer DEIM test
# ---------------------------------------------------------------------------
def diagnose_spine(
    Re, amp_values,
    clustering, operators, deim_ops_dict,
    problem, initial_solution, M_u, M_p,
    *,
    start_cluster=1,
    V_dict=None,            # {k: (n_dofs, r) velocity+supremizer basis}; needed
                            # for Phase B/C reduced-Jacobian comparison
    run_phase_bc=False,     # flip on once the assembly hooks are wired
    verbose=True,
):
    nu = 1.0 / Re
    problem.reynolds.assign(Re)
    amps = sorted((float(a) for a in amp_values), reverse=True)

    submesh_deims = {}
    for k in range(clustering.n_clusters):
        if deim_ops_dict.get(k) is not None:
            submesh_deims[k] = SubMeshDEIM(problem.velocity_space,
                                           deim_ops_dict[k])

    # ---- seed exactly as conv_test.py does -------------------------------
    init_vel = initial_solution.velocity.dat.data_ro.flatten()
    init_pres = initial_solution.pressure.dat.data_ro.flatten()
    prev_u, prev_p = project_to_rom_coefficients(
        init_vel, init_pres, operators[start_cluster], amp=0.0,
        M_u=M_u, M_p=M_p)
    prev_k, prev_amp = start_cluster, 0.0
    prev_full = initial_solution

    exact_states = []          # (amp, k, full-space solution, u_coeffs, p_coeffs)

    print("=" * 72)
    print("PHASE A1: exact-assembly spine walk (reference states)")
    print("=" * 72)
    for amp in amps:
        k = int(np.argmin(_whitened_pod_distances(clustering, prev_u, prev_k)))
        if k == prev_k:
            u_g, p_g = prev_u, prev_p
        else:
            pv = prev_full.velocity.dat.data_ro.flatten()
            pp = prev_full.pressure.dat.data_ro.flatten()
            u_g, p_g = project_to_rom_coefficients(
                pv, pp, operators[k], amp=prev_amp, M_u=M_u, M_p=M_p)

        sol, w, cb = solve_rom(
            operators=operators[k], nu=nu, amp=amp, problem=problem,
            deim_ops=None,                       # EXACT assembly
            u_initial_guess=u_g, p_initial_guess=p_g,
            submesh_deim=None, return_internals=True)
        full = reconstruct_solution(sol, operators[k], problem, amp=amp)
        ok = bool(sol.converged)
        print(f"  [exact] amp={amp:+.3f} k={k} "
              f"{'OK ' if ok else 'DIV'}")
        if not ok:
            print("    !! exact walk failed here; diagnostics stop at this amp")
            break
        exact_states.append((amp, k, full,
                             sol.velocity_coeffs, sol.pressure_coeffs))
        for amp, k, full, uc, pc in exact_states:
            a = np.asarray(uc)[:41]
            print(f"amp={amp:+.2f}  sup fraction = {np.linalg.norm(a[27:])/np.linalg.norm(a):.3e}")
        prev_k, prev_amp, prev_full = k, amp, full
        prev_u, prev_p = sol.velocity_coeffs, sol.pressure_coeffs

    # ---- Phase A2: DEIM solve seeded AT the exact answer -----------------
    print()
    print("=" * 72)
    print("PHASE A2: DEIM solve seeded at the exact answer, same amp")
    print("  (fast convergence + small drift => DEIM eqs fine AT solution;")
    print("   divergence/drift from the answer => DEIM eqs wrong there)")
    print("=" * 72)
    rows = []
    for amp, k, full, uc, pc in exact_states:
        pv = full.velocity.dat.data_ro.flatten()
        pp = full.pressure.dat.data_ro.flatten()
        u_g, p_g = project_to_rom_coefficients(
            pv, pp, operators[k], amp=amp, M_u=M_u, M_p=M_p)
        t0 = time.perf_counter()
        sol, w, cb = solve_rom(
            operators=operators[k], nu=nu, amp=amp, problem=problem,
            deim_ops=deim_ops_dict[k],
            u_initial_guess=u_g, p_initial_guess=p_g,
            submesh_deim=submesh_deims.get(k), return_internals=True)
        dt = time.perf_counter() - t0
        drift = (np.linalg.norm(sol.velocity_coeffs - u_g)
                 / max(np.linalg.norm(u_g), 1e-16))
        its = getattr(sol, "iterations", None)
        if its is None:
            its = cb.get("iterations", "?") if isinstance(cb, dict) else "?"
        cl = np.nan
        if sol.converged:
            problem.amplitude.assign(amp)
            rec = reconstruct_solution(sol, operators[k], problem, amp=amp)
            cl = float(compute_forces(rec, problem).lift)
        rows.append((amp, k, bool(sol.converged), its, drift, cl))
        print(f"  amp={amp:+.3f} k={k} "
              f"{'OK ' if sol.converged else 'DIV'} it={its} "
              f"drift={drift:.2e} C_L={cl:+.4f} ({dt:.2f}s)")

    # ---- Phases B & C ----------------------------------------------------
    if run_phase_bc:
        _phase_bc(exact_states, deim_ops_dict, V_dict, problem, nu)

    return exact_states, rows


def _phase_bc(exact_states, deim_ops_dict, V_dict, problem, nu):
    print()
    print("=" * 72)
    print("PHASE B: F/J error decomposition AT exact converged states")
    print("  e_proj = basis floor | e_deim = with sampling | amplif = ratio")
    print("=" * 72)
    basis_cache = {}
    hdr = (f"  {'amp':>7} {'F:e_proj':>9} {'F:e_deim':>9} {'F:amp':>7} "
           f"{'J:e_proj':>9} {'J:e_deim':>9} {'J:e_red':>8}")
    print(hdr)
    for amp, k, full, uc, pc in exact_states:
        ops = deim_ops_dict[k]
        bp = ops.metadata.get("basis_path", "")
        if bp not in basis_cache:
            basis_cache[bp] = load_basis(bp)
        basis = basis_cache[bp]
        u_full = full.velocity.dat.data_ro.flatten()

        F = assemble_F_full(u_full, amp, nu, problem)
        Jd = assemble_J_full(u_full, amp, nu, problem)

        n_modes_F = ops.metadata.get("n_modes_F", ops.V_T_B_F.shape[0])
        U_F = basis["modes_F"][:, :n_modes_F]
        V = V_dict[k] if V_dict else None
        red_exact = V.T @ F if V is not None else None
        red_deim = ops.reduced_convective(F) if V is not None else None
        mF = vector_metrics(F, U_F, ops.indices_F, red_exact, red_deim)
        mJ = (jacobian_metrics(Jd, basis, ops, V)
              if V is not None else
              vector_metrics(Jd, basis["modes_J"][:, :ops.J_r_modes.shape[0]],
                             ops.indices_J))
        print(f"  {amp:+7.3f} {mF['e_proj']:9.2e} {mF['e_deim']:9.2e} "
              f"{mF['amplif']:7.1f} {mJ['e_proj']:9.2e} {mJ['e_deim']:9.2e} "
              f"{mJ.get('e_red', float('nan')):8.2e}")

    print()
    print("=" * 72)
    print("PHASE C: same metrics at MISMATCH states")
    print("  (state from previous amp, lifting/BCs at current amp --")
    print("   i.e. the first Newton iterate of every continuation step)")
    print("=" * 72)
    print(hdr)
    for i in range(1, len(exact_states)):
        amp_prev, k_prev, full_prev, _, _ = exact_states[i - 1]
        amp_cur, k_cur, _, _, _ = exact_states[i]
        ops = deim_ops_dict[k_cur]
        basis = basis_cache[ops.metadata.get("basis_path", "")]
        u_full = full_prev.velocity.dat.data_ro.flatten()

        F = assemble_F_full(u_full, amp_cur, nu, problem)      # mismatch!
        Jd = assemble_J_full(u_full, amp_cur, nu, problem)

        n_modes_F = ops.metadata.get("n_modes_F", ops.V_T_B_F.shape[0])
        U_F = basis["modes_F"][:, :n_modes_F]
        V = V_dict[k_cur] if V_dict else None
        red_exact = V.T @ F if V is not None else None
        red_deim = ops.reduced_convective(F) if V is not None else None
        mF = vector_metrics(F, U_F, ops.indices_F, red_exact, red_deim)
        mJ = (jacobian_metrics(Jd, basis, ops, V)
              if V is not None else
              vector_metrics(Jd, basis["modes_J"][:, :ops.J_r_modes.shape[0]],
                             ops.indices_J))
        print(f"  {amp_cur:+7.3f} {mF['e_proj']:9.2e} {mF['e_deim']:9.2e} "
              f"{mF['amplif']:7.1f} {mJ['e_proj']:9.2e} {mJ['e_deim']:9.2e} "
              f"{mJ.get('e_red', float('nan')):8.2e}")


def sanity_check_training_snapshot(u_train_full, amp_train, nu, problem,
                                   deim_ops, V=None):
    """Run the Phase-B metrics at ONE training snapshot. e_proj must come
    out ~ POD tolerance (1e-10-ish for tol_F=1e-11). If it does not, the
    assembly hooks do not reproduce the training pipeline -- fix that
    before trusting any other number this script prints."""
    basis = load_basis(deim_ops.metadata.get("basis_path", ""))
    F = assemble_F_full(u_train_full, amp_train, nu, problem)
    n_modes_F = deim_ops.metadata.get("n_modes_F",
                                      deim_ops.V_T_B_F.shape[0])
    m = vector_metrics(F, basis["modes_F"][:, :n_modes_F],
                       deim_ops.indices_F)
    print(f"[sanity] training snapshot: F e_proj={m['e_proj']:.2e} "
          f"e_deim={m['e_deim']:.2e}  "
          f"({'OK' if m['e_proj'] < 1e-6 else 'MISMATCH -- fix hooks first!'})")
    return m


if __name__ == "__main__":
    # Mirror your conv_test.py setup, then:
    #
    # exact_states, rows = diagnose_spine(
    #     Re=99.0,
    #     amp_values=np.linspace(0.0, -1.0, 11),
    #     clustering=clustering, operators=operators,
    #     deim_ops_dict=deim_ops_dict,
    #     problem=problem, initial_solution=initial_solution,
    #     M_u=M_u, M_p=M_p,
    #     start_cluster=1,
    #     V_dict=None,          # supply {k: V} to get e_red columns
    #     run_phase_bc=False,   # True once assemble_F_full/J_full are wired
    # )
    pass
