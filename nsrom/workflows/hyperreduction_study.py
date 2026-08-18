"""
Runner for the hyper-reduction study (§6.3) and the locality ablation (§6.6).

Call it from main_local.py once the snapshots, clustering, lifting, problem and
mass matrices are already loaded, i.e. right after build_local_roms:

    from nsrom.workflows.hyperreduction_study import run_study, report_deim_ceiling

    report_deim_ceiling(operators, layout, cfg)      # STEP 0, run this first
    run_study(clustering, operators, problem, initial_solution,
              M_u, M_p, M_uu_red, layout, cfg,
              Re_list=RE_VALUES.tolist(), amp_values=AMP_VALUES.tolist())

Why it builds DEIM operators directly instead of going through
build_cluster_deim: that path keys its cache on (m_F, m_J, r) only, so flipping
an oversampling flag does NOT invalidate it and you would silently compare a
variant against itself. Here every variant is built explicitly, held in memory,
and never written to a shared path -- no cache, no collision.
"""

import json
import time

import numpy as np

from nsrom.helper_functions import modes_for_tolerance
from nsrom.rom import SubMeshDEIM

from nsrom.workflows.hyperreduction_comparison import (
    PROBE_LEGS, build_branch_anchors, run_fixed_point_comparison,
    summarize, to_macros,
)
import gc, hashlib
from pathlib import Path

from nsrom.online_operators import build_deim_online_operators, DEIMOnlineOperators

# ---------------------------------------------------------------------------
# The variant ladder. `tol_F`/`tol_J` feed modes_for_tolerance; the flags go
# straight to build_deim_online_operators.
# ---------------------------------------------------------------------------
_OVERSAMPLED = dict(oversample_residual=True, include_jacobian_cols=True,
                    oversample_jacobian=True, jacobian_filter_cols=False)
_SQUARE = dict(oversample_residual=False, include_jacobian_cols=False,
               oversample_jacobian=False, jacobian_filter_cols=False)

DEIM_VARIANTS = [
    # the setting a practitioner would actually pick: DEIM resolved to the same
    # energy as the state itself (cfg.pod_energy_tol). This is the cell that
    # answers "you compared against an unrealistically expensive DEIM".
    dict(name='deim_matched', tol_F=1e-8,  tol_J=1e-8,  flags=_OVERSAMPLED),
    dict(name='deim_tight',   tol_F=1e-12, tol_J=1e-12, flags=_OVERSAMPLED),
    # everything the basis contains -- the ceiling
    dict(name='deim_max',     tol_F=1e-16, tol_J=1e-16, flags=_OVERSAMPLED),
    # textbook square DEIM/MDEIM at the ceiling: isolates how much the
    # oversampling machinery is buying, i.e. whether the residual floor is a
    # sampling artefact or F/J inconsistency
    dict(name='deim_square',  tol_F=1e-16, tol_J=1e-16, flags=_SQUARE),
]


# ---------------------------------------------------------------------------
# STEP 0 -- is the tolerance ladder even meaningful?
# ---------------------------------------------------------------------------
def report_deim_ceiling(operators, layout, cfg):
    """Print, per cluster, how many DEIM modes the saved basis actually holds
    and how many each tolerance in the ladder would select.

    If the selected count equals the available count at every tolerance, the
    basis (i.e. the DEIM snapshot count) is the binding constraint, not the
    tolerance -- and §6.3's argument becomes 'DEIM had every mode available and
    still stalled' rather than 'we gave DEIM a larger budget'.
    """
    print("\n=== STEP 0: DEIM basis ceiling ===")
    rows = []
    for k in sorted(operators):
        d = np.load(layout.deim_basis(k))
        avail_F = int(d['eigenvalues_F'].shape[0])
        avail_J = int(d['eigenvalues_J'].shape[0])
        r = operators[k].Z_u.shape[1]
        sel = {}
        for v in DEIM_VARIANTS:
            sel[v['name']] = (
                int(modes_for_tolerance(d['eigenvalues_F'], v['tol_F'], cfg.m_F_max)),
                int(modes_for_tolerance(d['eigenvalues_J'], v['tol_J'], cfg.m_J_max)),
            )
        print(f"  cluster {k}: r={r}  available F={avail_F} J={avail_J}")
        for name, (mF, mJ) in sel.items():
            cap = " <-- AT CEILING" if (mF >= avail_F or mJ >= avail_J) else ""
            print(f"      {name:<14s} m_F={mF:4d} m_J={mJ:4d}  m_J/r={mJ/r:5.2f}{cap}")
        rows.append(dict(cluster=k, r=r, avail_F=avail_F, avail_J=avail_J, **sel))
    return rows


# ---------------------------------------------------------------------------
# STEP 2 -- build one operator set per variant
# ---------------------------------------------------------------------------
CACHE_ROOT = Path('hyperstudy_ops')


def _cache_key(v, k, m_F, m_J, Z_u, basis_path):
    """Everything that changes the built operator goes in the key."""
    st = Path(basis_path).stat()
    z = hashlib.md5(np.ascontiguousarray(Z_u).tobytes()).hexdigest()[:8]
    payload = json.dumps({
        'cluster': int(k), 'm_F': int(m_F), 'm_J': int(m_J),
        'r': int(Z_u.shape[1]), 'pod': z,
        'basis_size': st.st_size, 'basis_mtime': int(st.st_mtime),
        **v['flags'],
    }, sort_keys=True)
    return hashlib.md5(payload.encode()).hexdigest()[:12]


def build_deim_variants(operators, problem, layout, cfg, variants=DEIM_VARIANTS,
                        verbose=True, cache_root=CACHE_ROOT, force=False):
    cache_root.mkdir(parents=True, exist_ok=True)
    out = {}
    for v in variants:
        if verbose:
            print(f"\n--- {v['name']} (tol_F={v['tol_F']:g}, tol_J={v['tol_J']:g}, "
                  f"oversample={v['flags']['oversample_residual']})")
        ops, sms, meta = {}, {}, {}
        for k in sorted(operators):
            bp = layout.deim_basis(k)
            with np.load(bp) as d:
                m_F = int(modes_for_tolerance(d['eigenvalues_F'], v['tol_F'], cfg.m_F_max))
                m_J = int(modes_for_tolerance(d['eigenvalues_J'], v['tol_J'], cfg.m_J_max))
            Z_u = operators[k].Z_u
            path = cache_root / f"{v['name']}_c{k}_{_cache_key(v, k, m_F, m_J, Z_u, bp)}.npz"

            if path.exists() and not force:
                ops[k] = DEIMOnlineOperators.load(path)
                if verbose:
                    print(f"  c{k}: CACHED  m_F={ops[k].m_F} m_J={ops[k].m_J}")
            else:
                if verbose:
                    print(f"  c{k}: BUILD   requested m_F={m_F} m_J={m_J}")
                ops[k] = build_deim_online_operators(bp, Z_u, m_F, m_J, **v['flags'])
                ops[k].save(path, basis_path=str(bp))
            sms[k] = SubMeshDEIM(problem.velocity_space, ops[k])
            meta[k] = dict(ops[k].metadata)
            gc.collect()
        out[v['name']] = {'deim_ops': ops, 'submesh_deims': sms, 'realized': meta}
    return out
# def build_deim_variants(operators, problem, layout, cfg, variants=DEIM_VARIANTS,
#                         verbose=True):
#     """Returns {variant_name: {'deim_ops': {k: ops}, 'submesh_deims': {k: sm},
#                                'realized': {k: metadata}}}."""
#     out = {}
#     for v in variants:
#         if verbose:
#             print(f"\n--- building {v['name']} "
#                   f"(tol_F={v['tol_F']:g}, tol_J={v['tol_J']:g}, "
#                   f"oversample={v['flags']['oversample_residual']})")
#         ops, sms, meta = {}, {}, {}
#         for k in sorted(operators):
#             bp = layout.deim_basis(k)
#             d = np.load(bp)
#             m_F = modes_for_tolerance(d['eigenvalues_F'], v['tol_F'], cfg.m_F_max)
#             m_J = modes_for_tolerance(d['eigenvalues_J'], v['tol_J'], cfg.m_J_max)
#             ops[k] = build_deim_online_operators(
#                 bp, operators[k].Z_u, m_F, m_J, **v['flags'])
#             sms[k] = SubMeshDEIM(problem.velocity_space, ops[k])
#             meta[k] = dict(ops[k].metadata)
#         out[v['name']] = {'deim_ops': ops, 'submesh_deims': sms, 'realized': meta}
#     return out


# ---------------------------------------------------------------------------
# STEP 3 -- Test C: Newton residual histories at four probe points
# ---------------------------------------------------------------------------
TEST_C_POINTS = [
    (60.0,  0.0,  'far from any branch point'),
    (68.0,  0.0,  'at Re_c(0)'),
    (109.0, -1.0, 'negative-amp spine, top of range'),
    (109.0,  0.5, 'extrapolation'),
]


def run_test_C(configs, anchors, clustering, operators, problem,
               initial_solution, M_u, M_p, out_csv='test_C_residuals.csv'):
    """One solve per (point, config) from a common seed, keeping the residual
    history. Tiny, and it validates the plumbing before Tests A and B."""
    import pandas as pd

    legs = [dict(branch='sym', path=[(Re, amp)], seed='trunk')
            for (Re, amp, _) in TEST_C_POINTS]
    rows, audit = run_fixed_point_comparison(
        legs, configs, anchors, clustering, operators,
        problem, initial_solution, M_u, M_p, verbose=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\n[test C] wrote {out_csv}  ({len(rows)} rows, "
          f"{len(audit)} points not comparable)")
    return rows


# ---------------------------------------------------------------------------
# STEP 4/5 -- the whole study
# ---------------------------------------------------------------------------
def run_study(
    clustering, operators, problem, initial_solution,
    M_u, M_p, M_uu_red, layout, cfg,
    Re_list, amp_values,
    *,
    global_clustering=None, global_operators=None,   # K=1, for the locality axis
    branch_jump_fn=None,
    run_diagram=True,
    verbose=True,
):
    t_start = time.perf_counter()
# ---- anchors FIRST: tensor only, no DEIM needed ----------------------
    print("\n=== building branch anchors (K4 + tensor) ===")
    anchors = build_branch_anchors(
        Re_list, clustering, operators, problem, initial_solution, M_u, M_p,
        branch_jump_fn=branch_jump_fn, verbose=verbose)
    print(f"  sym anchors at {len(anchors['sym'])} Re values; ...")

    # ---- STEP 2: DEIM variants (cached) ----------------------------------
    variants = build_deim_variants(operators, problem, layout, cfg, verbose=verbose)

    with open('deim_realized_modes.json', 'w') as f:
        json.dump({n: {str(k): m for k, m in v['realized'].items()}
                   for n, v in variants.items()}, f, indent=2)
    print("\n[step 2] realized mode/sample counts -> deim_realized_modes.json")

    # ---- assemble the config table ---------------------------------------
    # pin_cluster=True wherever the clustering is SHARED with the reference, so
    # only the hyper-reduction differs. False for a different clustering (K=1),
    # which must run its own selection to be judged honestly.
    configs = {'K4_tensor': dict(clustering=clustering, operators=operators,
                                 deim_ops=None, submesh_deims=None,
                                 pin_cluster=True)}
    for name, v in variants.items():
        configs[f'K4_{name}'] = dict(
            clustering=clustering, operators=operators,
            deim_ops=v['deim_ops'], submesh_deims=v['submesh_deims'],
            pin_cluster=True)
    if global_operators is not None:
        configs['K1_tensor'] = dict(
            clustering=global_clustering, operators=global_operators,
            deim_ops=None, submesh_deims=None, pin_cluster=False)

    # # ---- anchors: built once, from the most reliable configuration -------
    # print("\n=== building branch anchors (K4 + tensor) ===")
    # anchors = build_branch_anchors(
    #     Re_list, clustering, operators, problem, initial_solution, M_u, M_p,
    #     branch_jump_fn=branch_jump_fn, verbose=verbose)
    # print(f"  sym anchors at {len(anchors['sym'])} Re values; "
    #       f"asym+ {len(anchors['asym+'])}, asym- {len(anchors['asym-'])}")

    # ---- STEP 3: Test C ---------------------------------------------------
    print("\n=== TEST C: residual histories ===")
    run_test_C(configs, anchors, clustering, operators, problem,
               initial_solution, M_u, M_p)

    # ---- STEP 4: Test A ---------------------------------------------------
    print("\n=== TEST A: fixed-point comparison ===")
    import pandas as pd
    rows, audit = run_fixed_point_comparison(
        PROBE_LEGS, configs, anchors, clustering, operators,
        problem, initial_solution, M_u, M_p, verbose=verbose)
    pd.DataFrame(rows).to_csv('test_A_fixedpoint.csv', index=False)
    pd.DataFrame(audit).to_csv('test_A_path_audit.csv', index=False)

    summary = summarize(rows)
    summary.to_csv('test_A_summary.csv', index=False)
    with open('test_A_macros.tex', 'w') as f:
        f.write(to_macros(summary))
    print(summary.to_string(index=False))
    print(f"\n[test A] {len(rows)} rows, {len(audit)} points dropped as "
          f"non-comparable -- CHECK test_A_path_audit.csv before trusting anything")

    # ---- STEP 5: Test B ---------------------------------------------------
    if run_diagram:
        from nsrom.bifurcation.diagram import build_full_diagram_bare
        print("\n=== TEST B: full diagram per configuration ===")
        for name, c in configs.items():
            if c['clustering'] is not clustering:
                continue                      # K=1 diagram is a separate study
            for sym_start in ('high', 'low'):
                tag = f"{name}_{sym_start}"
                print(f"\n--- diagram {tag}")
                t0 = time.perf_counter()
                recs = build_full_diagram_bare(
                    Re_range=np.array(Re_list), amp_values=amp_values,
                    clustering=clustering, operators=operators,
                    deim_ops_dict=(c['deim_ops']
                                   or {k: None for k in operators}),
                    problem=problem, initial_solution=initial_solution,
                    M_u=M_u, M_p=M_p, M_uu_red=M_uu_red,
                    mode=('deim' if c['deim_ops'] is not None else 'tensor'),
                    fom_compute=False,        # failure map only; errors come from Test A
                    trace_symmetric=True,
                    sym_start=sym_start,
                    verbose=False)
                wall = time.perf_counter() - t0
                df = pd.DataFrame(recs)
                df['config'] = name
                df['sym_start'] = sym_start
                df['diagram_wall_time'] = wall
                df.to_csv(f'test_B_diagram_{tag}.csv', index=False)
                n_ok = int(df.converged.sum())
                print(f"    {n_ok}/{len(df)} converged, {wall/60:.1f} min wall")

    print(f"\n=== study complete in {(time.perf_counter()-t_start)/60:.1f} min ===")
    return configs, anchors
