"""
cluster_diagnostics.py
======================

Symmetry / consistency diagnostics for the two asymmetric clusters of the
local ROM (by default cluster 1 = branch +, cluster 3 = branch -).

Motivation
----------
On-axis the pinball's two asymmetric branches are exact mirror images under
the y->-y reflection R, so their local ROMs should behave identically. They
don't: one cluster's continuation blows up at high Re while its mirror is
fine, and the blow-up gets WORSE as POD tolerance is tightened (more modes).
That monotonicity rules out "under-resolved" and points at an inconsistency
that low-energy modes amplify -- most likely the DEIM residual / MDEIM
Jacobian mismatch (F uses m_F points, J uses a FIXED m_J points while r grows).

These tests separate the two independent effects:
  * per-cluster consistency budget  (m_J vs r, U_J_P conditioning)   -> why it blows up at all
  * cluster-1-vs-3 asymmetry        (spectra, reflected principal angles) -> why + and - differ

Usage (recommended: import and call from your main after build_local_roms)
--------------------------------------------------------------------------
    from cluster_diagnostics import run_all_diagnostics
    run_all_diagnostics(
        clustering=clustering, bases=bases, operators=operators,
        deim_ops_dict=deim_ops_dict, M_u=M_u, M_p=M_p, problem=problem,
        velocity_snapshots=velocity_snapshots,
        pressure_snapshots=pressure_snapshots,
        parameters=parameters, branch_ids=branch_ids,
        k_a=1, k_b=3,
    )

Everything except test 6 (Jacobian consistency, which needs your online
residual/Jacobian callables) runs on objects already built in your pipeline.
"""

import numpy as np
from nsrom.rom import make_callbacks
from firedrake import Function

# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _rule(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def _get(obj, *names, default=None):
    """First present attribute among names, else default."""
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return default


def _param_array(parameters):
    """Coerce the parameters container to an (N, 2) [Re, amp] float array."""
    arr = np.asarray([list(p) for p in parameters], dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"parameters -> unexpected shape {arr.shape}")
    return arr[:, :2]


def _dominant_branch(branch_ids, idx):
    """Modal branch id over a set of snapshot indices."""
    b = np.asarray(branch_ids)[idx]
    vals, counts = np.unique(b, return_counts=True)
    return int(vals[np.argmax(counts)])


def m_orthonormalize(Z, M, rank_tol=1e-12):
    """
    Loewdin (symmetric) M-orthonormalization of the columns of Z.

    Returns Q with Q^T M Q = I (numerically), dropping directions whose
    Gram eigenvalue is below rank_tol * max. Robust to the supremizer
    enrichment making Z^T M Z != I.
    """
    G = Z.T @ (M @ Z)
    G = 0.5 * (G + G.T)
    w, V = np.linalg.eigh(G)
    keep = w > rank_tol * w.max()
    w = w[keep]
    V = V[:, keep]
    Q = Z @ (V / np.sqrt(w)[None, :])
    return Q, int(keep.sum())


def principal_angles_M(Z1, Z2, M):
    """
    Principal angles (degrees, ascending) between span(Z1) and span(Z2)
    in the M inner product. cos(theta_i) = singular values of Q1^T M Q2
    with Q1, Q2 M-orthonormal.
    """
    Q1, r1 = m_orthonormalize(Z1, M)
    Q2, r2 = m_orthonormalize(Z2, M)
    C = Q1.T @ (M @ Q2)
    s = np.linalg.svd(C, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    return np.degrees(np.arccos(s)), r1, r2


# ---------------------------------------------------------------------------
# reflection operator R (y -> -y), signed permutation on velocity DOFs
# ---------------------------------------------------------------------------
def try_build_reflection(problem, axis="y", center=0.0, tol=None, verbose=True):
    """
    Build the DOF-space reflection R for the velocity space, as a sparse
    signed permutation. (R u) is the reflection of u about {axis = center}:
    the mirrored component flips sign, the others are copied to the mirror
    node.

    Coordinates are obtained by interpolating SpatialCoordinate into the
    SAME velocity space, so the ordering matches the snapshot flatten order
    (node-major, component-minor) exactly, independent of Firedrake version.

    Returns R (scipy.sparse) or None if it can't be built / validated.
    """
    try:
        from firedrake import Function, SpatialCoordinate
        import scipy.sparse as sp
        from scipy.spatial import cKDTree
    except Exception as e:  # pragma: no cover - environment dependent
        if verbose:
            print(f"  [R] cannot build reflection ({e}); reflection tests skipped.")
        return None

    W = problem.velocity_space
    mesh = W.mesh()
    gdim = mesh.geometric_dimension
    if gdim != 2:
        if verbose:
            print(f"  [R] geometric dim {gdim} != 2; reflection tests skipped.")
        return None

    Xf = Function(W).interpolate(SpatialCoordinate(mesh))
    coords = np.asarray(Xf.dat.data_ro).reshape(-1, gdim)   # (n_nodes, 2)
    n_nodes = coords.shape[0]

    flip = {"y": 1, "x": 0}[axis]
    refl = coords.copy()
    refl[:, flip] = 2.0 * center - coords[:, flip]

    scale = float(np.linalg.norm(coords.max(0) - coords.min(0)))
    if tol is None:
        tol = 1e-6 * scale

    tree = cKDTree(coords)
    dist, j = tree.query(refl, k=1)
    matched = dist < tol
    match_rate = matched.mean()

    if verbose:
        print(f"  [R] nodes={n_nodes}  matched={match_rate*100:.2f}%  "
              f"max matched residual={dist[matched].max() if matched.any() else np.nan:.2e}  "
              f"(tol={tol:.1e}, axis={axis}, center={center})")
    if match_rate < 0.999:
        if verbose:
            print("  [R] mesh does not look symmetric about the chosen axis/center; "
                  "try a different `center` or `axis`. Skipping reflection tests.")
        return None

    rows, cols, vals = [], [], []
    for i in range(n_nodes):
        if not matched[i]:
            continue
        ji = int(j[i])
        for c in range(gdim):
            rows.append(ji * gdim + c)
            cols.append(i * gdim + c)
            vals.append(-1.0 if c == flip else 1.0)
    N = n_nodes * gdim
    R = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))

    # involution check
    x = np.random.default_rng(0).standard_normal(N)
    err_inv = np.linalg.norm(R @ (R @ x) - x) / np.linalg.norm(x)
    if verbose:
        print(f"  [R] involution ||RRx-x||/||x|| = {err_inv:.2e}")
    if err_inv > 1e-10:
        if verbose:
            print("  [R] R is not a clean involution; skipping reflection tests.")
        return None
    return R


def validate_reflection_against_M(R, M, verbose=True):
    """Check R^T M R = M (R is an M-isometry) via random probes."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((M.shape[0], 4))
    lhs = R.T @ (M @ (R @ X))
    rhs = M @ X
    rel = np.linalg.norm(lhs - rhs) / np.linalg.norm(rhs)
    if verbose:
        print(f"  [R] ||R^T M R - M|| / ||M|| (probed) = {rel:.2e}  "
              f"({'OK' if rel < 1e-8 else 'NOT invariant -- angles below may be biased'})")
    return rel


# ---------------------------------------------------------------------------
# Test 1 -- structural & spectral parity
# ---------------------------------------------------------------------------
def test_structural_spectral_parity(bases, operators, deim_ops_dict, k_a, k_b):
    _rule(f"TEST 1  structural & spectral parity   cluster {k_a} vs {k_b}")
    ba, bb = bases[k_a], bases[k_b]

    def row(label, va, vb):
        print(f"  {label:<22} {str(va):>14} {str(vb):>14}")

    print(f"  {'quantity':<22} {'cluster '+str(k_a):>14} {'cluster '+str(k_b):>14}")
    row("n_velocity_modes", ba.n_velocity_modes, bb.n_velocity_modes)
    row("n_supremizer_modes", ba.n_supremizer_modes, bb.n_supremizer_modes)
    row("n_pressure_modes", ba.n_pressure_modes, bb.n_pressure_modes)
    row("r = dim(Z_u)", operators[k_a].Z_u.shape[1], operators[k_b].Z_u.shape[1])

    # eigenvalue spectra: if the two snapshot sets were true mirrors these
    # are identical; the deviation quantifies the data asymmetry.
    for name, ea, eb in (
        ("velocity", np.asarray(ba.velocity_eigenvalues, float),
                      np.asarray(bb.velocity_eigenvalues, float)),
        ("pressure", np.asarray(ba.pressure_eigenvalues, float),
                      np.asarray(bb.pressure_eigenvalues, float)),
    ):
        m = min(len(ea), len(eb))
        ea, eb = ea[:m], eb[:m]
        denom = np.maximum(np.abs(ea), 1e-300)
        rel = np.abs(ea - eb) / denom
        print(f"\n  {name} POD eigenvalues (first {m} shared):")
        print(f"    max |lambda_a - lambda_b| / |lambda_a| = {rel.max():.3e}")
        print(f"    mean rel deviation                     = {rel.mean():.3e}")
        show = min(8, m)
        print("    idx      lambda_a        lambda_b       rel.dev")
        for i in range(show):
            print(f"    {i:>3}  {ea[i]:>14.6e}  {eb[i]:>14.6e}  {rel[i]:>10.3e}")

    # best-effort reduced-operator norm parity (attribute names guessed)
    print("\n  reduced-operator norms (best-effort; skipped if attr absent):")
    for names in (("A_N", "A_r", "A_reduced"),
                  ("B_N", "B_r", "B_reduced"),
                  ("f_N", "f_r"),
                  ("g_N", "g_r")):
        va = _get(operators[k_a], *names)
        vb = _get(operators[k_b], *names)
        if va is not None and vb is not None:
            na, nb = np.linalg.norm(va), np.linalg.norm(vb)
            print(f"    {names[0]:<6}  ||.||_a={na:.6e}  ||.||_b={nb:.6e}  "
                  f"rel.diff={abs(na-nb)/max(na,1e-300):.3e}")


# ---------------------------------------------------------------------------
# Test 2 -- DEIM consistency budget (all clusters)
# ---------------------------------------------------------------------------
def test_deim_budget(operators, deim_ops_dict):
    _rule("TEST 2  DEIM consistency budget   (m_J vs r is the key ratio)")
    print(f"  {'k':>2} {'r':>5} {'m_F':>5} {'m_J':>5} {'m_F/r':>7} {'m_J/r':>7} "
          f"{'cond(U_J_P)':>12} {'dup_idx_J':>10}")
    for k in sorted(deim_ops_dict):
        do = deim_ops_dict[k]
        if do is None:
            print(f"  {k:>2}  (no DEIM)")
            continue
        r = operators[k].Z_u.shape[1]
        mF, mJ = int(do.m_F), int(do.m_J)
        UJinv = _get(do, "U_J_P_inv")
        condUJ = np.linalg.cond(UJinv) if UJinv is not None else np.nan
        idxJ = _get(do, "indices_J")
        dup = (len(idxJ) - len(np.unique(idxJ))) if idxJ is not None else -1
        print(f"  {k:>2} {r:>5} {mF:>5} {mJ:>5} {mF/r:>7.2f} {mJ/r:>7.2f} "
              f"{condUJ:>12.2e} {dup:>10}")
    print("\n  Reading: if m_J/r drops well below ~1 (or m_J is pinned by m_J_max")
    print("  while r grows with tighter POD tol), the MDEIM Jacobian can no longer")
    print("  represent the r x r reduced Jacobian -> Newton line search stalls.")
    print("  This is the mechanism consistent with 'more modes made it worse'.")


# ---------------------------------------------------------------------------
# Test 3 -- basis conditioning
# ---------------------------------------------------------------------------
def test_basis_conditioning(operators, M_u, ks):
    _rule("TEST 3  basis M-conditioning   ||Z^T M Z - I|| and numerical rank")
    print(f"  {'k':>2} {'r':>5} {'||G-I||':>12} {'cond(G)':>12} {'num_rank':>9}")
    for k in ks:
        Z = operators[k].Z_u
        G = Z.T @ (M_u @ Z)
        G = 0.5 * (G + G.T)
        I = np.eye(G.shape[0])
        w = np.linalg.eigvalsh(G)
        w = w[w > 0]
        cond = w.max() / w.min() if w.size else np.inf
        _, nr = m_orthonormalize(Z, M_u)
        print(f"  {k:>2} {Z.shape[1]:>5} {np.linalg.norm(G-I):>12.3e} "
              f"{cond:>12.3e} {nr:>9d}")
    print("\n  A large ||G-I|| or rank < r means the enriched basis is not")
    print("  M-orthonormal / is near-rank-deficient; that alone can make the")
    print("  reduced Newton system ill-conditioned and is a per-cluster asymmetry.")


# ---------------------------------------------------------------------------
# Test 4 -- reflected principal angles (are 1 and 3 genuine mirrors?)
# ---------------------------------------------------------------------------
def test_reflection_basis_angles(operators, M_u, R, k_a, k_b, thresh_deg=1.0):
    _rule(f"TEST 4  reflected principal angles   span(R Z_{k_a}) vs span(Z_{k_b})")
    if R is None:
        print("  [skip] reflection R not available.")
        return
    Z_a = operators[k_a].Z_u
    Z_b = operators[k_b].Z_u
    RZ_a = R @ Z_a
    angles, r1, r2 = principal_angles_M(RZ_a, Z_b, M_u)
    n_big = int((angles > thresh_deg).sum())
    print(f"  effective dims: R Z_{k_a} -> {r1},  Z_{k_b} -> {r2}")
    print(f"  # principal angles > {thresh_deg} deg : {n_big} / {len(angles)}")
    print(f"  max angle = {angles.max():.3f} deg   mean = {angles.mean():.3f} deg")
    show = min(10, len(angles))
    print(f"  largest {show} angles (deg): " +
          ", ".join(f"{a:.2f}" for a in np.sort(angles)[::-1][:show]))
    print("\n  ~0 everywhere  => clusters are genuine mirrors; you can BUILD cluster")
    print(f"  {k_b} as R * cluster {k_a} (exactly symmetric diagram, half the offline cost).")
    print("  Large tail angles => the two snapshot sets are not mirror-symmetric;")
    print("  the +/- discrepancy is baked into the data, not just the solver.")


# ---------------------------------------------------------------------------
# Test 5 -- snapshot symmetry audit (discovers the true amp mapping)
# ---------------------------------------------------------------------------
def test_snapshot_symmetry(velocity_snapshots, parameters, branch_ids, R,
                           k_a_branch, k_b_branch, re_tol=0.5):
    _rule(f"TEST 5  snapshot symmetry audit   branch {k_a_branch} reflected vs branch {k_b_branch}")
    if R is None:
        print("  [skip] reflection R not available.")
        return
    P = _param_array(parameters)
    b = np.asarray(branch_ids)
    ia = np.where(b == k_a_branch)[0]
    ib = np.where(b == k_b_branch)[0]
    if ia.size == 0 or ib.size == 0:
        print(f"  [skip] empty branch set (a:{ia.size}, b:{ib.size}).")
        return

    Sa = np.asarray(velocity_snapshots)[ia, :]           # (na, ndof)
    Sb = np.asarray(velocity_snapshots)[ib, :]            # (nb, ndof)
    RSa = (R @ Sa.T).T                                    # reflect branch-a fields

    # sanity: a symmetric field should be ~unchanged under R (use branch 0 if present)
    i0 = np.where(b == 0)[0]
    if i0.size:
        s0 = np.asarray(velocity_snapshots)[i0[0], :]
        rel0 = np.linalg.norm(R @ s0 - s0) / max(np.linalg.norm(s0), 1e-300)
        print(f"  symmetric-field check: ||R s0 - s0|| / ||s0|| = {rel0:.2e} "
              f"({'OK' if rel0 < 5e-2 else 'symmetric branch is NOT R-invariant -- check axis/center'})")

    rel_errs, amp_from, amp_to, re_at = [], [], [], []
    for t, i in enumerate(ia):
        Re_i, a_i = P[i]
        cand = np.where(np.abs(P[ib, 0] - Re_i) < re_tol)[0]
        if cand.size == 0:
            continue
        diffs = Sb[cand] - RSa[t][None, :]
        d = np.linalg.norm(diffs, axis=1)
        jbest = cand[int(np.argmin(d))]
        rel = d.min() / max(np.linalg.norm(RSa[t]), 1e-300)
        rel_errs.append(rel)
        amp_from.append(a_i)
        amp_to.append(P[ib[jbest], 1])
        re_at.append(Re_i)

    if not rel_errs:
        print("  [skip] no Re-matched mirror candidates found.")
        return
    rel_errs = np.asarray(rel_errs)
    amp_from = np.asarray(amp_from)
    amp_to = np.asarray(amp_to)
    print(f"  matched {len(rel_errs)} branch-{k_a_branch} snapshots to nearest "
          f"branch-{k_b_branch} mirror")
    print(f"  reflection match error  min/median/max = "
          f"{rel_errs.min():.2e} / {np.median(rel_errs):.2e} / {rel_errs.max():.2e}")

    # discovered amp mapping: fit amp_to ~ slope * amp_from
    mask = np.abs(amp_from) > 1e-9
    if mask.any():
        slope = np.median(amp_to[mask] / amp_from[mask])
        print(f"  discovered amp mapping  amp_mirror ~ ({slope:+.2f}) * amp  "
              f"(expect -1 if reflection flips control sign, +1 if it preserves it)")
    print("\n  Small match error => data IS symmetric; asymmetry is solver-side")
    print("  (basis/DEIM), fixable by building cluster 3 = R*cluster 1.")
    print("  Large match error => regenerate mirror snapshots or symmetrize the")
    print("  dataset before expecting a symmetric diagram.")


# ---------------------------------------------------------------------------
# Test 6 -- MDEIM Jacobian consistency (scaffold; needs your online callables)
# ---------------------------------------------------------------------------
def jacobian_consistency(residual_fn, jac_fn, a0, eps=1e-6, label=""):
    """
    Compare the reduced Jacobian jac_fn(a0) against a finite-difference
    Jacobian of residual_fn at a0. A large relative gap is the DEIM-residual
    vs MDEIM-Jacobian inconsistency that kills the line search.

    residual_fn(a) -> (r,)   the reduced residual the SOLVER actually uses
    jac_fn(a)      -> (r,r)  the reduced Jacobian the SOLVER actually uses
    a0             -> (r,)   a converged reduced state on the branch of interest

    HOW TO OBTAIN residual_fn / jac_fn:
      These live inside your online path (nsrom.rom.solve_rom /
      nsrom.online_operators / SubMeshDEIM). Wrap the same callbacks the SNES
      uses so that residual_fn(a) evaluates F_DEIM(a) at fixed (nu, amp,
      cluster) and jac_fn(a) evaluates the MDEIM J. If your solver exposes a
      PETSc SNES, SNESComputeFunction / SNESComputeJacobian give exactly these.
    """
    a0 = np.asarray(a0, float)
    r = a0.size
    J = np.asarray(jac_fn(a0), float)
    Jfd = np.empty((r, r))
    f0 = np.asarray(residual_fn(a0), float)
    for j in range(r):
        da = np.zeros(r)
        da[j] = eps * max(1.0, abs(a0[j]))
        Jfd[:, j] = (np.asarray(residual_fn(a0 + da), float) - f0) / da[j]
    rel = np.linalg.norm(J - Jfd) / max(np.linalg.norm(Jfd), 1e-300)
    print(f"  [{label}] ||J_mdeim - J_fd|| / ||J_fd|| = {rel:.3e}  (r={r})")
    return rel


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def run_all_diagnostics(*, clustering, bases, operators, deim_ops_dict,
                        M_u, M_p, problem,
                        velocity_snapshots, pressure_snapshots,
                        parameters, branch_ids,
                        k_a=1, k_b=3,
                        reflection_axis="y", reflection_center=0.0):
    """Run tests 1-5 (6 is opt-in). Safe to call after build_local_roms()."""
    _rule(f"CLUSTER DIAGNOSTICS   asymmetric pair (k_a={k_a}, k_b={k_b})")

    # map clusters -> their dominant branch id for the snapshot audit
    br_a = _dominant_branch(branch_ids, clustering.cluster_indices[k_a])
    br_b = _dominant_branch(branch_ids, clustering.cluster_indices[k_b])
    print(f"  cluster {k_a} dominant branch = {br_a}")
    print(f"  cluster {k_b} dominant branch = {br_b}")

    test_structural_spectral_parity(bases, operators, deim_ops_dict, k_a, k_b)
    test_deim_budget(operators, deim_ops_dict)
    test_basis_conditioning(operators, M_u, ks=sorted(operators))

    _rule("build reflection operator R")
    R = try_build_reflection(problem, axis=reflection_axis,
                             center=reflection_center, verbose=True)
    if R is not None:
        validate_reflection_against_M(R, M_u, verbose=True)

    test_reflection_basis_angles(operators, M_u, R, k_a, k_b)
    test_snapshot_symmetry(velocity_snapshots, parameters, branch_ids, R,
                           k_a_branch=br_a, k_b_branch=br_b)

    _rule("done  (test 6 Jacobian-consistency is opt-in: see jacobian_consistency)")
    return R

def test_fj_consistency(callback_data, a0, eps=1e-6):
    N_u = callback_data['N_u']
    N_p = callback_data['N_p']
    r = N_u + N_p
    res_cb, jac_cb = make_callbacks(callback_data)
    W_rom = callback_data['W_rom']

    w_func = Function(W_rom)

    def eval_res(a):
        # write reduced state into the PETSc Vec and evaluate inside the block
        with w_func.dat.vec as v:      # rw: syncs in and out
            v.array[:] = a
            res_cb(v)
        return np.array(callback_data['conv_res_const'].dat.data_ro).ravel()

    def eval_jac(a):
        with w_func.dat.vec as v:
            v.array[:] = a
            jac_cb(v)
        return np.array(callback_data['conv_J_const'].dat.data_ro)

    # --- residual and MDEIM Jacobian at a0 ---
    f0 = eval_res(a0)
    J_mdeim = eval_jac(a0)

    # --- FD Jacobian of the DEIM residual ---

    m = f0.size
    J_fd = np.zeros((m, r))
    for j in range(r):
        a_pert = a0.copy()
        da = eps * max(1.0, abs(a0[j]))
        a_pert[j] += da
        f_pert = eval_res(a_pert)
        J_fd[:, j] = (f_pert - f0) / da

    # convective Jacobian is velocity-only: compare the (N_u, N_u) block
    J_fd_vel = J_fd[:, :N_u]
    rel = np.linalg.norm(J_mdeim - J_fd_vel) / max(np.linalg.norm(J_fd_vel), 1e-300)

    # sanity: pressure columns of the FD residual Jacobian should be ~0
    p_leak = np.linalg.norm(J_fd[:, N_u:])

    return rel, J_mdeim, J_fd_vel, p_leak

if __name__ == "__main__":
    print(__doc__)
    print("This module is meant to be imported from your main after "
          "build_local_roms(); see the usage block above.")
