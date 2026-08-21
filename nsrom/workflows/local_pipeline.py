"""
Local ROM pipeline.

Configuration is held in `cfg` (LocalROMConfig); the convection-evaluation
strategy is chosen once there via `cfg.mode` ('exact' | 'deim' | 'tensor')
and threaded down to every solver call.

Paths, test parameters, and sweep ranges remain module-level constants.

File structure:
    local_rom/K{cfg.n_clusters}_{cfg.inner_product_type}/
        clustering/
        cluster_0/
            pod_basis/
            operators/
            deim_basis.npz      (if cfg.mode == 'deim')
            deim_ops.npz        (if cfg.mode == 'deim')
            conv_tensor.npz     (if cfg.mode == 'tensor')
        cluster_1/
            ...
"""

import os

import numpy as np

from nsrom.navier_stokes import setup_navier_stokes_problem, load_solution
from nsrom.snapshots.collection import load_snapshot_dofs
from nsrom.lifting_functions import load_lifting_functions
from nsrom.clustering.building import build_energy_snapshots, compute_or_load_clustering
from nsrom.rom.local import (
    build_local_roms,
    solve_at_parameter,
    compare_rom_vs_fom,
    precompute_fast_selection,
    precompute_change_of_basis,
)
from nsrom.layout import LocalROMLayout
from nsrom.config import LocalROMConfig, RunManifest, save_run
from nsrom.bifurcation.detection import build_reduced_velocity_l2_mass_matrices
from nsrom.tensor_convection import (
    build_convection_tensor,
    load_convection_tensor,
    save_convection_tensor,
    TensorConvection,
    verify_against_exact,
)
from nsrom.rom import SubMeshDEIM   # at module top, next to the other imports

from nsrom.bifurcation.sweep import build_fixed_amp_path, save_sweep_results
from nsrom.bifurcation.diagram import build_full_diagram_bare
from nsrom.plotting.speedup import speedup_summary
from nsrom.plotting.plots import plot_lift_3d


# =============================================================================
# CONFIGURATION
# =============================================================================

# cfg = LocalROMConfig(
#     n_clusters=4,
#     inner_product_type="H1",   # "semi" only when no Cholesky is needed (POD before clustering)
#     pod_energy_tol=1e-8,
#     n_velocity_max=400,
#     n_pressure_max=200,
#     n_supremizer_max=200,
#     boundary_markers=(1, 3, 4, 5, 6),
#     mode='tensor',             # 'exact' | 'deim' | 'tensor'
#     compute_affine_convection=True,
#     deim_energy_tol_F=1e-16,
#     deim_energy_tol_J=1e-16,
#     m_F_max=400,
#     m_J_max=400,
#     n_modes_F=None,
#     n_modes_J=None,
#     recompute_clustering=False,
#     recompute_pod=False,
#     recompute_deim=False,
#     recompute_tensor=False,
# )

# # --- Paths ---
# SNAPSHOT_DIR      = "multi_param_multi_branch"
# DEIM_SNAPSHOT_DIR = "multi_param_multi_branch"
# LIFTING_DIR       = "lifting"
# MESH_FILE         = "mesh/mid_pinball.msh"
# FOM_CHECKPOINT    = "initial_states/base_velocity"

# REYNOLDS_INIT  = 100.0
# AMPLITUDE_INIT = 0.0

# # --- Test parameters (single-point path, RUN_SWEEP=False) ---
# TEST_RE  = 80.0
# TEST_AMP = 0.1

# # --- Sweep ---
# RE_VALUES  = np.arange(50.0, 100.0, 1.0)
# AMP_VALUES = np.arange(-1.0, 1.0, 0.1)
# AMP_FIXED  = 0.0
# FOM_EVERY  = 0       # 0 = skip FOM, N = compare every Nth point
# RUN_SWEEP  = True

# # --- Diagram (build_full_diagram_bare) ---
# DIAGRAM_RE_RANGE = np.arange(80.0, 90.0, 1.0)
# DIAGRAM_AMP_VALUES = [-0.3, 0.3]
# # DIAGRAM_RE_RANGE  = np.arange(20.0, 110.0, 1.0)
# # DIAGRAM_AMP_VALUES = [
# #     -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,
# #     0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
# # ]   # amp=0 is handled by phase 1
# DIAGRAM_FOM_COMPUTE = True

# # --- Diagnostics (off by default: they cost a full pass over the snapshots) ---
# RUN_CLUSTER_DIAGNOSTICS = False
# CLUSTER_METHOD = "pod_whitened"    # "pod_whitened" or "cholesky"
# CLUSTER_RANK   = 10


# # =============================================================================
# # MANIFEST and LAYOUT
# # =============================================================================

# manifest = RunManifest(
#     snapshot_dir=SNAPSHOT_DIR,
#     deim_snapshot_dir=DEIM_SNAPSHOT_DIR if cfg.mode == 'deim' else None,
#     lifting_dir=LIFTING_DIR,
#     mesh_file=MESH_FILE,
#     fom_checkpoint=FOM_CHECKPOINT,
#     reynolds_init=REYNOLDS_INIT,
#     amplitude_init=AMPLITUDE_INIT,
#     test_re=TEST_RE,
#     test_amp=TEST_AMP,
#     run_sweep=RUN_SWEEP,
#     fom_every=FOM_EVERY,
#     re_values=RE_VALUES.tolist(),
#     amp_values=AMP_VALUES.tolist(),
# )

# layout = LocalROMLayout(
#     base_dir="local_rom",
#     n_clusters=cfg.n_clusters,
#     inner_product_type=cfg.inner_product_type,
# )

# =============================================================================
# ENV HELPERS
# =============================================================================

def _env_str(name, default):
    return os.environ.get(name, default)


def _env_int(name, default):
    return int(os.environ.get(name, default))


def _env_float(name, default):
    return float(os.environ.get(name, default))


def _env_bool(name, default):
    return os.environ.get(name, str(default)).strip().lower() in ('1', 'true', 'yes', 'on')


def _env_floats(name, default):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return [float(x) for x in raw.replace(' ', '').split(',') if x]


# =============================================================================
# CONFIGURATION
# =============================================================================

RUN_TAG = _env_str('NSROM_RUN_TAG', 'manual')

cfg = LocalROMConfig(
    n_clusters=_env_int('NSROM_K', 4),
    inner_product_type="H1",
    pod_energy_tol=_env_float('NSROM_POD_TOL', 1e-8),
    # Caps are safety rails only: the tolerance must be what selects N.
    # If a realised mode count ever equals one of these, the run is invalid.
    n_velocity_max=_env_int('NSROM_NVEL_MAX', 400),
    n_pressure_max=_env_int('NSROM_NPRES_MAX', 200),
    n_supremizer_max=_env_int('NSROM_NSUP_MAX', 200),
    boundary_markers=(1, 3, 4, 5, 6),
    mode=_env_str('NSROM_MODE', 'tensor'),
    compute_affine_convection=True,
    deim_energy_tol_F=_env_float('NSROM_DEIM_TOL', 1e-16),
    deim_energy_tol_J=_env_float('NSROM_DEIM_TOL', 1e-16),
    m_F_max=_env_int('NSROM_MF_MAX', 400),
    m_J_max=_env_int('NSROM_MJ_MAX', 400),
    n_modes_F=None,
    n_modes_J=None,
    recompute_clustering=_env_bool('NSROM_RECOMPUTE_CLUSTERING', False),
    recompute_pod=_env_bool('NSROM_RECOMPUTE_POD', False),
    recompute_deim=_env_bool('NSROM_RECOMPUTE_DEIM', False),
    recompute_tensor=_env_bool('NSROM_RECOMPUTE_TENSOR', False),
)

# --- Paths ---
SNAPSHOT_DIR      = "multi_param_multi_branch"
DEIM_SNAPSHOT_DIR = "multi_param_multi_branch"
LIFTING_DIR       = "lifting"
MESH_FILE         = "mesh/mid_pinball.msh"
FOM_CHECKPOINT    = "initial_states/base_velocity"

REYNOLDS_INIT  = 100.0
AMPLITUDE_INIT = 0.0

# --- Test parameters (single-point path, RUN_SWEEP=False) ---
TEST_RE  = 80.0
TEST_AMP = 0.1

# --- Sweep bookkeeping (manifest only) ---
RE_VALUES  = np.arange(50.0, 100.0, 1.0)
AMP_VALUES = np.arange(-1.0, 1.0, 0.1)
FOM_EVERY  = 0
RUN_SWEEP  = _env_bool('NSROM_RUN_SWEEP', True)

# --- Diagram ---
DIAGRAM_RE_RANGE = np.arange(
    _env_float('NSROM_RE_MIN', 20.0),
    _env_float('NSROM_RE_MAX', 110.0),
    _env_float('NSROM_RE_STEP', 1.0),
)
DIAGRAM_AMP_VALUES = _env_floats('NSROM_AMPS', [
    -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,
    0.0,
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
])
DIAGRAM_FOM_COMPUTE = _env_bool('NSROM_FOM_COMPUTE', True)
# Symmetric amp-spine strategy for build_full_diagram_bare. Canonical published
# runs used 'high' and pin it explicitly via NSROM_SYM_START in run.sh; the
# fallback here matches diagram.py's library default so a bare invocation is
# unchanged. Threaded explicitly into the diagram call and recorded in
# run_meta.json, so published configuration is never inherited from a mutable
# Python default.
DIAGRAM_SYM_START = _env_str('NSROM_SYM_START', 'low')

# --- State store ---
STATE_DIR         = _env_str('NSROM_STATE_DIR', f"states/{RUN_TAG}")
SAVE_FOM_FIELDS   = _env_bool('NSROM_SAVE_FOM_FIELDS', True)
FOM_FIELD_STRIDE  = _env_int('NSROM_FOM_FIELD_STRIDE', 1)

# --- Diagnostics ---
RUN_CLUSTER_DIAGNOSTICS = _env_bool('NSROM_CLUSTER_DIAG', False)
CLUSTER_METHOD = "pod_whitened"
CLUSTER_RANK   = 10


# =============================================================================
# MANIFEST and LAYOUT
# =============================================================================

manifest = RunManifest(
    snapshot_dir=SNAPSHOT_DIR,
    deim_snapshot_dir=DEIM_SNAPSHOT_DIR if cfg.mode == 'deim' else None,
    lifting_dir=LIFTING_DIR,
    mesh_file=MESH_FILE,
    fom_checkpoint=FOM_CHECKPOINT,
    reynolds_init=REYNOLDS_INIT,
    amplitude_init=AMPLITUDE_INIT,
    test_re=TEST_RE,
    test_amp=TEST_AMP,
    run_sweep=RUN_SWEEP,
    fom_every=FOM_EVERY,
    re_values=RE_VALUES.tolist(),
    amp_values=AMP_VALUES.tolist(),
)

# The POD tolerance is part of the cache key: without it, a tolerance sweep
# silently reloads bases built at a different tolerance and every point of
# the "N convergence" study comes out identical.
layout = LocalROMLayout(
    base_dir="local_rom",
    n_clusters=cfg.n_clusters,
    inner_product_type=f"{cfg.inner_product_type}_tol{cfg.pod_energy_tol:g}",
)

#
# Finally, make the sweep CSV per-run so nothing overwrites:
#

# =============================================================================
# HELPERS
# =============================================================================

def print_banner():
#
    print(f"  Run tag:     {RUN_TAG}")
    print(f"  POD tol:     {cfg.pod_energy_tol:g}")
    print(f"  Layout:      {layout.base_dir} K={cfg.n_clusters} "
          f"{cfg.inner_product_type}_tol{cfg.pod_energy_tol:g}")
    print(f"  State dir:   {STATE_DIR}")
    print(f"  Re range:    {DIAGRAM_RE_RANGE[0]:g} -> {DIAGRAM_RE_RANGE[-1]:g} "
          f"({len(DIAGRAM_RE_RANGE)} pts)")
    print(f"  Amps:        {len(DIAGRAM_AMP_VALUES)} values")
    print(f"  FOM compute: {DIAGRAM_FOM_COMPUTE}")
#
    affine_str = "Affine" if cfg.compute_affine_convection else "Full"
    print("=" * 80)
    print(f"LOCAL ROM — mode={cfg.mode}, {affine_str} convection, K={cfg.n_clusters}")
    print("=" * 80)
    print(f"  Mode:        {cfg.mode}")
    print(f"  Affine:      {cfg.compute_affine_convection}")
    print(f"  ROM modes:   tol={cfg.pod_energy_tol}, max={cfg.n_velocity_max}")
    print(f"  Inner prod:  {cfg.inner_product_type}")
    print(f"  Test params: Re={TEST_RE}, amp={TEST_AMP}")
    if cfg.mode == 'deim':
        print(f"  DEIM:        tol_F={cfg.deim_energy_tol_F}, "
              f"tol_J={cfg.deim_energy_tol_J}, "
              f"max_F={cfg.m_F_max}, max_J={cfg.m_J_max}")
    print()


def attach_convection_tensors(operators, clustering, problem, verbose=True):
    """
    Build (or load) the exact convection tensor for every cluster and attach
    it to that cluster's ReducedOperators.

    A cached tensor is discarded when its size no longer matches the current
    POD basis, which is the usual way a stale tensor shows up after a
    recompute_pod run.
    """
    for k in range(clustering.n_clusters):
        Z_u_k = operators[k].Z_u              # (n_dofs, N_u), includes supremizers
        tpath = os.path.join(layout.cluster_dir(k), "conv_tensor.npz")

        N = None
        if os.path.exists(tpath) and not cfg.recompute_tensor:
            N = load_convection_tensor(tpath)
            if N.shape[0] != Z_u_k.shape[1]:
                if verbose:
                    print(f"  cluster {k}: cached tensor is stale "
                          f"({N.shape[0]} != {Z_u_k.shape[1]}), rebuilding")
                N = None
        if N is None:
            N = build_convection_tensor(Z_u_k, problem.velocity_space)
            save_convection_tensor(N, tpath)

        tc = TensorConvection(N)
        verify_against_exact(                 # must report ~1e-12
            tc, Z_u_k, problem.velocity_space,
            np.random.randn(Z_u_k.shape[1]),
        )
        operators[k].tensor_convection = tc

    if verbose:
        for k in range(clustering.n_clusters):
            attached = getattr(operators[k], 'tensor_convection', None) is not None
            print(f"  cluster {k}: tensor attached = {attached}")


def cluster_agreement_diagnostic(clustering, S_homo, M_u):
    """
    Compare the whitened k-means labels against a plain energy-norm
    nearest-centroid assignment. Prints the disagreement rate and the
    row-normalized confusion matrix.
    """
    C      = clustering.centroids_original    # (n_dofs, K)
    labels = clustering.labels
    K, n   = clustering.n_clusters, S_homo.shape[1]

    energy_labels = np.empty(n, dtype=int)
    for s in range(n):
        Diff = C - S_homo[:, [s]]
        d2   = np.einsum('ij,ij->j', Diff, M_u @ Diff)
        energy_labels[s] = int(np.argmin(d2))

    mism = int(np.sum(energy_labels != labels))
    print(f"energy-norm vs whitened k-means: "
          f"{mism}/{n} snapshots disagree ({100 * mism / n:.1f}%)")

    conf = np.array([
        [np.sum((labels == a) & (energy_labels == b)) for b in range(K)]
        for a in range(K)
    ])
    print(conf)
    print((conf / conf.sum(axis=1, keepdims=True)).round(2))


# =============================================================================
# MAIN
# =============================================================================

def main():
    save_run(cfg, manifest, layout.run_json())
    print_banner()

    # --- Setup problem ---
    print("--- Setup problem ---")
    problem = setup_navier_stokes_problem(
        mesh_file=MESH_FILE,
        reynolds_init=REYNOLDS_INIT,
        amplitude_init=AMPLITUDE_INIT,
    )
    initial_solution = load_solution(FOM_CHECKPOINT, problem)

    # --- Load snapshots and lifting ---
    print("\n--- Load snapshots and lifting ---")
    snapshot_data = load_snapshot_dofs(SNAPSHOT_DIR)
    lifting       = load_lifting_functions(LIFTING_DIR, problem)

    velocity_snapshots = snapshot_data['velocity']
    pressure_snapshots = snapshot_data['pressure']
    parameters         = snapshot_data['parameters']
    lift_coefficients  = snapshot_data['lift_coefficients']

    print(f"  Snapshots:  {velocity_snapshots.shape[0]}")
    print(f"  Parameters: {len(parameters)} entries")

    # --- Energy-norm transform ---
    print("\n--- Energy-norm transform ---")
    S_homo, S_tilde, M_u, M_p, L, P = build_energy_snapshots(
        velocity_snapshots, parameters, lifting, problem,
        cfg.inner_product_type, homogenize=True,
    )
    print(f"  S_tilde: {S_tilde.shape}")
    # --- persist the norms every reported error is measured in -------------
    if _env_bool('NSROM_DUMP_MASS', False):
        from nsrom.io.mass import dump_mass_matrices
        dump_mass_matrices({'M_u': M_u, 'M_p': M_p},
                           outdir=_env_str('NSROM_MASS_DIR', 'mass'),
                           inner_product=cfg.inner_product_type,
                           # mirrors build_energy_snapshots: velocity in the
                           # configured inner product, pressure always L2
                           norms={'M_u': cfg.inner_product_type, 'M_p': 'L2'},
                           problem=problem)
    # --- Clustering ---
    print("\n--- Clustering ---")
    clustering, reason = compute_or_load_clustering(
        S_tilde, S_homo, M_u,
        parameters, L, P,
        save_dir=layout.clustering_dir,
        n_clusters=cfg.n_clusters,
        method=CLUSTER_METHOD,
        rank=CLUSTER_RANK,
        recompute=cfg.recompute_clustering,
    )
    print(f"  {reason}")
    clustering.summary()

    # plot_lift_3d(
    #     parameters,
    #     lift_coefficients,
    #     clustering.labels,
    #     title=f"Clusters (method={CLUSTER_METHOD}, K={clustering.n_clusters})",
    # )

    # --- Local ROMs ---
    print("\n--- Local ROMs ---")
    deim_snapshot_data = None
    if cfg.mode == 'deim':
        deim_snapshot_data = load_snapshot_dofs(DEIM_SNAPSHOT_DIR)
        print(f"  DEIM snapshots: {deim_snapshot_data['velocity'].shape[0]}")

    bases, operators, deim_ops_dict = build_local_roms(
        clustering=clustering,
        velocity_snapshots=velocity_snapshots,
        pressure_snapshots=pressure_snapshots,
        deim_snapshot_data=deim_snapshot_data,
        deim_snapshot_dir=DEIM_SNAPSHOT_DIR,
        lifting=lifting,
        problem=problem,
        cfg=cfg,
        layout=layout,
        M_u=M_u,
    )

    if cfg.mode == 'deim':
        from nsrom.workflows.hyperreduction_study import report_deim_ceiling
        report_deim_ceiling(operators, layout, cfg)

    if cfg.mode == 'tensor':
        print("\n--- Exact convection tensors ---")
        attach_convection_tensors(operators, clustering, problem)

    # --- Reduced velocity mass matrices (for the bifurcation indicator) ---
    M_uu_red = build_reduced_velocity_l2_mass_matrices(
        problem=problem,
        clustering=clustering,
        operators=operators,
    )

    if RUN_CLUSTER_DIAGNOSTICS:
        print("\n--- Cluster agreement diagnostic ---")
        cluster_agreement_diagnostic(clustering, S_homo, M_u)

    _replay_ref = os.environ.get("NSROM_REPLAY_REF")
    # --- Online: sweep or single point ---
    if _replay_ref:
        run_replay(clustering, operators, deim_ops_dict, problem,
                    M_u, M_p, M_uu_red,_replay_ref)
    elif RUN_SWEEP:

        run_diagram(clustering, operators, deim_ops_dict, problem,
                    initial_solution, M_u, M_p, M_uu_red)
    else:
        run_single_point(clustering, operators, deim_ops_dict, problem,
                         initial_solution, M_u, M_p)
    # s = StateStore("states/K4_tensor"); s.summary()
    # print(s[0])
def run_replay(clustering, operators, deim_ops_dict, problem,
                M_u, M_p, M_uu_red,_replay_ref):
    print("\n--- Precompute fast cluster selection ---")
    precompute_fast_selection(clustering, operators, M_u)

    print("\n--- Precompute change-of-basis ---")
    precompute_change_of_basis(clustering, operators, M_u, M_p)


    import sys
    from nsrom.io.state_store import StateWriter
    from nsrom.bifurcation.replay import replay_points, preflight

    _seed = os.environ.get("NSROM_REPLAY_SEED", "stored")
    _stride = int(os.environ.get("NSROM_REPLAY_STRIDE", "1"))

    print(f"\n{'=' * 62}")
    print(f"REPLAY  ref={_replay_ref}  mode={cfg.mode}  seed={_seed}"
            + (f"  stride={_stride}" if _stride > 1 else ""))
    print(f"{'=' * 62}")

    if not preflight(_replay_ref, operators):
        print("\nPreflight failed -- refusing to run. See messages above.")
        sys.exit(2)
    #men hon

    # la hon
    submesh_deims = {}
    if cfg.mode == 'deim':
        for k in range(clustering.n_clusters):
            if deim_ops_dict.get(k) is not None:
                submesh_deims[k] = SubMeshDEIM(problem.velocity_space, deim_ops_dict[k])

    # stride via a counter, not pt['idx']: replay_points calls the filter once
    # per point in store order, so this is exact and schema-independent
    _filt = None
    if _stride > 1:
        import itertools
        _counter = itertools.count()
        _filt = lambda pt: next(_counter) % _stride == 0

    if os.path.abspath(STATE_DIR) == os.path.abspath(_replay_ref):
        print(f"\nRefusing to write the replay into its own reference "
              f"({STATE_DIR}). Set NSROM_STATE_DIR.")
        sys.exit(2)

    _writer = StateWriter(
        STATE_DIR, mode=cfg.mode,
        save_fom_fields=False,          # fields live in the reference run
        meta={'replay_of': _replay_ref,
              'replay_seed': _seed,
              'replay_stride': _stride,
              'K': cfg.n_clusters,
              'pod_tol': cfg.pod_energy_tol,
              'deim_tol': cfg.deim_energy_tol_F if cfg.mode == 'deim' else None})

    _records = replay_points(
        ref_state_dir=_replay_ref,
        clustering=clustering, operators=operators,
        deim_ops_dict=deim_ops_dict, submesh_deims=submesh_deims,
        problem=problem, M_u=M_u, M_p=M_p, M_uu_red=M_uu_red,
        mode=cfg.mode,
        seed=_seed,
        cluster_source='stored' if _seed == 'stored' else 'param',
        point_filter=_filt,
        state_writer=_writer,
        verbose=True)

    _writer.close()
    save_sweep_results(_records, os.path.join(STATE_DIR, "sweep_results.csv"))
    print(f"\nReplay saved to {STATE_DIR}/sweep_results.csv")
    sys.exit(0)

def run_diagram(clustering, operators, deim_ops_dict, problem,
                initial_solution, M_u, M_p, M_uu_red):
    """Full bifurcation diagram over (Re, amp), with optional FOM comparison."""
    print("\n--- Precompute fast cluster selection ---")
    precompute_fast_selection(clustering, operators, M_u)

    print("\n--- Precompute change-of-basis ---")
    precompute_change_of_basis(clustering, operators, M_u, M_p)

    print("\n--- Bifurcation diagram ---")
    print(f"  Re:  {DIAGRAM_RE_RANGE[0]:.1f} -> {DIAGRAM_RE_RANGE[-1]:.1f} "
          f"({len(DIAGRAM_RE_RANGE)} points)")
    print(f"  amp: {len(DIAGRAM_AMP_VALUES)} off-axis values")
    print(f"  sym_start: {DIAGRAM_SYM_START}")
    print(f"  FOM comparison: {'on' if DIAGRAM_FOM_COMPUTE else 'off'}")

    results = build_full_diagram_bare(
        Re_range=DIAGRAM_RE_RANGE,
        amp_values=DIAGRAM_AMP_VALUES,
        clustering=clustering,
        operators=operators,
        deim_ops_dict=deim_ops_dict,
        problem=problem,
        initial_solution=initial_solution,
        M_u=M_u, M_p=M_p, M_uu_red=M_uu_red,
        mode=cfg.mode,
        fom_compute=DIAGRAM_FOM_COMPUTE,
        state_dir=STATE_DIR,
        sym_start=DIAGRAM_SYM_START,
        save_fom_fields=SAVE_FOM_FIELDS,
        fom_field_stride=FOM_FIELD_STRIDE,
    )


    speedup_summary(results)                    # prints the table
    save_sweep_results(results, f"{STATE_DIR}/sweep_results.csv")
    return results


def run_single_point(clustering, operators, deim_ops_dict, problem,
                     initial_solution, M_u, M_p):
    """Solve at (TEST_RE, TEST_AMP) and compare against a FOM solve."""
    print("\n" + "#" * 80)
    print(f"--- Online solve (Re={TEST_RE}, amp={TEST_AMP}) ---")

    rom_result = solve_at_parameter(
        Re=TEST_RE,
        amp=TEST_AMP,
        clustering=clustering,
        operators=operators,
        deim_ops_dict=deim_ops_dict,
        problem=problem,
        initial_solution=initial_solution,
        M_u=M_u,
        M_p=M_p,
        mode=cfg.mode,
    )
    rom_result.summary()

    print("#" * 80)
    print("\n--- FOM comparison ---")
    comp = compare_rom_vs_fom(rom_result, problem, initial_solution, M_u, M_p)

    k = rom_result.cluster
    affine_str = "Affine" if cfg.compute_affine_convection else "Full"

    print(f"\n{'=' * 80}")
    print(f"LOCAL ROM — mode={cfg.mode}, {affine_str} convection, K={cfg.n_clusters}")
    print(f"{'=' * 80}")
    print(f"  Re={TEST_RE}, amp={TEST_AMP}")
    print(f"  Cluster selected:  {k}")
    print(f"  ROM modes:         {operators[k].n_velocity_modes} vel "
          f"× {operators[k].n_pressure_modes} pres")
    print(f"  Affine convection: {operators[k].has_affine_convection}")
    if cfg.mode == 'deim':
        deim_ops_k = deim_ops_dict[k]
        print(f"  DEIM:              m_F={deim_ops_k.m_F}, m_J={deim_ops_k.m_J}")
    print(f"  Converged:         {rom_result.converged} "
          f"({rom_result.iterations} iterations)")
    comp.summary(u_norm_label=cfg.inner_product_type)
    print(f"{'=' * 80}")

    return rom_result, comp
