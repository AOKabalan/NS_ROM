#!/usr/bin/env python3
"""
dump_mass.py -- persist the M_u / M_p used by every error norm in the paper, and
identify which inner product they actually are.

Why this exists
---------------
`err_u` in every run is `sqrt(d^T M_u d) / sqrt(u_fom^T M_u u_fom)`, with M_u
coming from build_energy_snapshots(..., cfg.inner_product_type). Post-hoc error
scripts must use the SAME matrices or they report a different quantity under the
same symbol. Rebuilding the matrices independently risks exactly that, so dump
the objects the run held.

Both matrices depend only on the mesh, the FE spaces and the inner-product type
-- not on K, not on the POD tolerance, not on the mode. One dump serves the
whole experiment matrix, provided every run used the same inner_product_type
(check the emitted meta against each run's cache directory name).

Install
-------
Put this next to main_local.py, then add four lines after the
build_energy_snapshots call (around line 412):

    S_homo, S_tilde, M_u, M_p, L, P = build_energy_snapshots(...)
    print(f"  S_tilde: {S_tilde.shape}")

    # --- persist the norms used by every reported error -----------------
    if _env_bool('NSROM_DUMP_MASS', False):
        from dump_mass import dump_mass_matrices
        dump_mass_matrices({'M_u': M_u, 'M_p': M_p},
                           outdir=_env_str('NSROM_MASS_DIR', 'mass'),
                           inner_product=cfg.inner_product_type,
                           problem=problem)

Run it without paying for a sweep:

    NSROM_DUMP_MASS=1 NSROM_RUN_SWEEP=0 NSROM_K=4 NSROM_POD_TOL=1e-8 \
        python scripts/main_local.py

Writes mass/M_u_<ip>.npz, mass/M_p_<ip>.npz and mass/mass_meta.json, with the
inner-product tag in the filename so two conventions can never be confused.
"""

from __future__ import annotations

import json
import os

import numpy as np

__all__ = ['to_csr', 'dump_mass_matrices', 'load_mass']


# ---------------------------------------------------------------------------
# conversion
# ---------------------------------------------------------------------------

def to_csr(M):
    """
    Coerce whatever build_energy_snapshots returned into scipy CSR.

    Handles, in order: scipy sparse, Firedrake Matrix (.M.handle), bare PETSc
    Mat (.getValuesCSR), dense numpy. Raises with the actual type on anything
    else rather than guessing.
    """
    import scipy.sparse as sp

    if sp.issparse(M):
        return M.tocsr(), 'scipy.sparse'

    # Firedrake assembled Matrix -> PETSc Mat
    handle = None
    if hasattr(M, 'M') and hasattr(M.M, 'handle'):
        handle = M.M.handle
    elif hasattr(M, 'petscmat'):
        handle = M.petscmat
    elif hasattr(M, 'getValuesCSR'):
        handle = M

    if handle is not None:
        ai, aj, av = handle.getValuesCSR()
        shape = tuple(handle.getSize())
        return sp.csr_matrix((av, aj, ai), shape=shape), 'petsc'

    arr = np.asarray(M)
    if arr.ndim == 2:
        return sp.csr_matrix(arr), 'dense'

    raise TypeError(f"cannot convert {type(M)!r} to a matrix")


# ---------------------------------------------------------------------------
# identification
# ---------------------------------------------------------------------------

def _identify(name, A, problem=None):
    """
    Report enough to tell an L2 mass matrix from an H1/energy matrix without
    trusting a config string.

    Two obvious-looking tests DO NOT work and are recorded for information only:

      * row sums. For M + K the stiffness annihilates constants, so the row
        sums of an H1 matrix equal those of the mass matrix exactly. Likewise
        1^T M 1 = |Omega| for both.
      * negative off-diagonals. A P2 mass matrix has negative vertex-vertex
        entries of its own, so their presence proves nothing on a Taylor-Hood
        velocity space.

    What does discriminate is SCALING: diag(M) = O(h^d) whereas
    diag(M + K) = O(h^(d-2)). So diag_max divided by the mean row sum is O(1)
    for a mass matrix and O(h^-2) -- thousands on this mesh -- for anything
    carrying a gradient term. That test needs no reference matrix. The direct
    comparison against a freshly assembled L2 mass matrix, when a `problem` is
    available, is definitive.
    """
    import scipy.sparse as sp

    n = A.shape[0]
    rec = {'shape': list(A.shape), 'nnz': int(A.nnz), 'dtype': str(A.dtype)}

    asym = abs(A - A.T)
    rec['symmetric'] = bool(asym.nnz == 0 or asym.max() < 1e-10 * abs(A).max())

    d = A.diagonal()
    rec['diag_min'] = float(d.min())
    rec['diag_max'] = float(d.max())
    rec['diag_all_positive'] = bool((d > 0).all())

    offdiag = A - sp.diags(d)
    rec['n_negative_offdiag'] = int((offdiag.data < 0).sum()) if offdiag.nnz else 0
    rec['frac_negative_offdiag'] = (
        round(rec['n_negative_offdiag'] / offdiag.nnz, 4) if offdiag.nnz else 0.0)
    rec['_note_negative_offdiag'] = 'not discriminating on P2 spaces'

    one = np.ones(n)
    rs = A @ one
    total = float(rs.sum())
    rec['row_sum_total'] = total
    rec['ones_quadratic_form'] = float(one @ (A @ one))
    rec['_note_row_sums'] = 'identical for M and M+K; informational only'

    # the decisive reference-free test. Uses the MEDIAN diagonal, not the max:
    # on a graded mesh (5.5x here) the largest element's diagonal sits well
    # above the mean row sum and inflates the ratio on a matrix that is exactly
    # a mass matrix. The median is insensitive to that.
    mean_row_sum = total / n if n else 0.0
    d_med = float(np.median(d))
    rec['diag_median'] = d_med
    if mean_row_sum > 0:
        ratio = d_med / mean_row_sum
        rec['diag_to_mean_rowsum'] = float(ratio)
        if ratio > 50.0:
            rec['verdict'] = (f"gradient-containing (H1/energy): median diagonal "
                              f"is {ratio:.3g}x the mean row sum, i.e. O(h^-2) "
                              f"scaling, not O(1)")
        elif ratio < 10.0:
            rec['verdict'] = (f"consistent with a mass matrix: median diagonal "
                              f"is {ratio:.3g}x the mean row sum, i.e. O(1)")
        else:
            rec['verdict'] = (f"ambiguous: median diag/mean-row-sum = "
                              f"{ratio:.3g}; deferring to the L2 comparison")
    else:
        rec['diag_to_mean_rowsum'] = None
        rec['verdict'] = 'inconclusive: non-positive row-sum total'

    # optional cross-check against a freshly assembled L2 mass matrix. This is
    # conclusive, so it OVERRIDES the scaling verdict when available.
    if problem is not None:
        try:
            rec['l2_comparison'] = _compare_to_l2(name, A, problem)
        except Exception as exc:  # noqa: BLE001
            rec['l2_comparison'] = f"skipped: {type(exc).__name__}: {exc}"
        lc = rec['l2_comparison']
        if isinstance(lc, dict) and 'is_l2' in lc:
            rec['verdict'] = (
                "L2 mass matrix (verified against a freshly assembled one)"
                if lc['is_l2'] else
                "NOT the L2 mass matrix (verified); "
                + rec['verdict'].split(':')[0])
    return rec


def _compare_to_l2(name, A, problem):
    """Assemble the plain L2 mass matrix on the matching space and compare."""
    from firedrake import TestFunction, TrialFunction, assemble, dx, inner

    space = None
    for attr in (('velocity_space',) if name == 'M_u' else ('pressure_space',)):
        space = getattr(problem, attr, None)
        if space is not None:
            break
    if space is None:
        return 'no matching space attribute on problem'

    v, w = TestFunction(space), TrialFunction(space)
    M_l2, _ = to_csr(assemble(inner(v, w) * dx))
    if M_l2.shape != A.shape:
        return {'note': 'shape mismatch -- different space',
                'l2_shape': list(M_l2.shape), 'given_shape': list(A.shape)}
    diff = abs(A - M_l2)
    denom = abs(M_l2).max()
    return {'l2_shape': list(M_l2.shape),
            'max_abs_diff': float(diff.max()) if diff.nnz else 0.0,
            'relative_diff': float(diff.max() / denom) if denom else None,
            'is_l2_mass': bool(denom and diff.max() < 1e-10 * denom)}


# ---------------------------------------------------------------------------

def dump_mass_matrices(mats, outdir='mass', inner_product='unknown',
                       problem=None, verbose=True):
    """Persist each matrix as CSR .npz and write an identification report."""
    from scipy.sparse import save_npz

    os.makedirs(outdir, exist_ok=True)
    tag = str(inner_product).replace('/', '-').replace(' ', '')
    meta = {'inner_product_type': str(inner_product), 'matrices': {}}

    for name, M in mats.items():
        if M is None:
            meta['matrices'][name] = {'error': 'None'}
            continue
        A, source = to_csr(M)
        path = os.path.join(outdir, f"{name}_{tag}.npz")
        save_npz(path, A)
        rec = _identify(name, A, problem=problem)
        rec['source_type'] = source
        rec['path'] = path
        rec['size_mb'] = round(os.path.getsize(path) / 1e6, 4)
        meta['matrices'][name] = rec

    meta_path = os.path.join(outdir, 'mass_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, default=str)

    if verbose:
        print(f"\n  [mass] inner_product_type = {inner_product!r}")
        for name, rec in meta['matrices'].items():
            if 'error' in rec:
                print(f"  [mass] {name}: {rec['error']}")
                continue
            print(f"  [mass] {name} -> {rec['path']}  "
                  f"{rec['shape']} nnz={rec['nnz']} ({rec['size_mb']:.1f} MB, "
                  f"from {rec['source_type']})")
            print(f"           symmetric={rec['symmetric']}  "
                  f"diag/mean-row-sum={rec.get('diag_to_mean_rowsum')}")
            print(f"           {rec['verdict']}")
            lc = rec.get('l2_comparison')
            if isinstance(lc, dict) and 'is_l2_mass' in lc:
                print(f"           vs assembled L2 mass: is_l2={lc['is_l2_mass']} "
                      f"rel_diff={lc['relative_diff']}")
            elif lc:
                print(f"           L2 check: {lc}")
        print(f"  [mass] report -> {meta_path}")
        print("  [mass] NOTE: if M_u is gradient-containing, every reported "
              "err_u is an\n         energy-norm error, not an L2 error -- "
              "label it accordingly.\n")

    return meta


def load_mass(path):
    """Counterpart loader for the figure/error scripts."""
    from scipy.sparse import load_npz
    return load_npz(path).tocsr()


if __name__ == '__main__':
    print(__doc__)
