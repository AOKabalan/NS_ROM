# CLAUDE.md

Guidance for working in this repository.

## What this project is

`nsrom` is a **reduced-order modeling (ROM) framework for the fluidic pinball** —
steady, incompressible Navier–Stokes flow past three rotating cylinders,
parametrized by Reynolds number `Re` and cylinder rotation amplitude `A`. The
scientific goal is to study the **bifurcation of steady solutions** in the
`(Re, A)` parameter space: the flow exhibits a Hopf bifurcation near `Re ≈ 18`
and a steady **pitchfork** near `Re ≈ 68.22`, and the code tracks branches and
locates the critical curve across the two-parameter plane.

The method is **POD–Galerkin projection with DEIM/MDEIM hyper-reduction**:

- Snapshots collected over a sampling of `(Re, A)` across all branches.
- POD with **supremizer enrichment** for inf–sup stability; pressure basis
  separate from the velocity+supremizer basis.
- **Local ROMs** via energy-norm k-means++ clustering (optional Cholesky or POD
  whitening); each cluster carries its own POD basis and DEIM operators.
- **Newton solves on the reduced residual** with cluster-aware change-of-basis
  warm starts and snake-ordered continuation over the sweep grid.
- **Bifurcation detection** via a reduced generalized eigenproblem
  `J_red ν = μ M̃_red ν`, tracking the sign of the leftmost real eigenvalue;
  branch switching by eigenvector-kick warm starts, cross-validated against a
  full-order SLEPc Krylov–Schur pipeline.

The convection term can be evaluated three ways, selected by a single `mode`
field on `LocalROMConfig` (`nsrom/config.py`) and threaded through every solver
call — `VALID_MODES = ('exact', 'deim', 'tensor')`:
- `exact` — re-assemble the nonlinear term on the full mesh (reference).
- `deim` — DEIM/MDEIM hyper-reduction with submesh assembly (affine online cost).
- `tensor` — precomputed quadratic convection tensor (requires
  `compute_affine_convection=True`).

Built on **Firedrake** (which provides PETSc and SLEPc).

## Layout

- **`nsrom/`** — the installed package (see `pyproject.toml`, `pip install -e .`).
  - `nsrom/rom/` — POD, DEIM/MDEIM hyper-reduction, reduced operators, Galerkin
    Newton solver, and local-ROM implementation (`rom_solver.py` owns the
    authoritative `VALID_MODES`).
  - `nsrom/bifurcation/` — detection, branch jumping, sweep, diagram
    construction, and replay.
  - `nsrom/clustering/` — clustering and whitening utilities.
  - `nsrom/snapshots/` — snapshot generation, collection, and storage helpers.
  - `nsrom/io/` — state-store and mass-matrix persistence.
  - `nsrom/plotting/` — plot helpers and speedup reporting. Figure *style*
    is not here; it moved to `render/style.py` with the renderers.
  - `nsrom/workflows/` — the local pipeline and hyper-reduction workflows.
  - top level: Navier–Stokes problem setup, lifting functions, tensor
    convection, and config/layout/cache helpers.
- **`scripts/`** — command-line and diagnostic entry points. `scripts/main_local.py`
  is the supported shell-facing local-ROM command and delegates to
  `nsrom.workflows.local_pipeline`. Standalone tools: `solve_FOM.py`,
  `fom_bifurcation_diagram_fom.py`, `audit_section6.py`, cluster/spine
  diagnostics, point-error analysis, and snapshot plotting. The compatibility
  shims that used to live here are gone — import from `nsrom.*` directly.
- **`render/`** — every manuscript figure and table is produced here, and
  nowhere else. `style.py` holds the style knobs, `common.py` the shared data
  access and LaTeX emission, one `fig_*.py` / `tab_*.py` per artifact, and
  `out/` the generated PDFs, CSVs and `.tex`.
- **`run.sh`** — the experiment matrix: every run the manuscript depends on,
  each tagged with the figures and tables it feeds. `./run.sh --list`.
- **`Makefile`** — the dependency graph tying the two together.

## Regenerating results

The Makefile knows which run feeds which artifact, so only the affected things
rebuild. `make help` prints this; the short version:

| changed | command |
|---|---|
| a colour, a cutoff, which amplitudes a table shows | `make render/out/tab_k_sensitivity.tex` |
| a style knob in `render/style.py` | `make figures` |
| a numerical parameter for one run | `./run.sh <TAG>` then `make paper` |
| the training snapshots | `make clean-derived && make runs && make paper` |

Two rules the graph enforces, both deliberate:

- **Experiments are never make targets.** A renderer takes seconds and may
  rebuild on a timestamp; a sweep takes hours and must not. `states/` appears
  only as a prerequisite. `make paper` will never start a sweep.
- **A renderer rule exists only when its runs do.** `render/out/` holds
  committed `.csv` and `.tex` — the asset that lets the tables rebuild without
  a sweep — and those are also renderer outputs. On a checkout without
  `states/`, make leaves them alone rather than regenerating from nothing.
  `make list` shows which runs are present.

Adding a figure or table: write `render/fig_x.py`, add its rule to the
Makefile, add it to `FIGS` or `TABS`, and add whatever run it needs to
`run.sh`. Nothing else needs to know about it.

## Environment

- Firedrake lives in a venv at **`~/venv-firedrake`**. **Activate it before
  running anything that imports Firedrake:**
  ```bash
  source ~/venv-firedrake/bin/activate
  ```
- **⚠️ Interpreter version is currently ambiguous.** `~/venv-firedrake/pyvenv.cfg`
  declares Python **3.14.4**, but `__pycache__` directories contain **both**
  `cpython-312` and `cpython-314` bytecode. This means the tree has been run
  under two different interpreters. Confirm which interpreter is actually in use
  (`python --version` after activating) before trusting any environment-sensitive
  behavior, and flag this to Ali rather than assuming.

- **⚠️ `import firedrake` HANGS in singleton mode — launch everything under
  `mpiexec -n 1`.** On this stack (OpenMPI **5.0.10**, Python 3.14.4) a plain
  `python -c "import firedrake"` hangs forever inside `MPI_Init`
  (`petsc4py.init` → PETSc init → `MPI_Init`; `from mpi4py import MPI` hangs the
  same way). It is **not** a stale cache/lock or hostname problem — a fresh cache
  and loopback-only TCP both still hang. Running under the MPI launcher avoids
  the broken singleton-init path:
  ```bash
  mpiexec -n 1 python your_script.py
  mpiexec -n 1 pytest            # e.g. the characterization suite in tests/
  ```
  This is still **one rank with no numerical effect** — purely a launch
  workaround. Anyone reproducing this work will hit the hang; always use
  `mpiexec -n 1`. See `env/environment.md` and `docs/known-issues.md`.

## IRREPLACEABLE COMPUTED RESULTS — never delete or modify

These directories and files hold expensive computed results (solution
checkpoints, snapshots, DEIM operators, sweep outputs, logs, paper figures).
**They are not reproducible on a whim** — some represent long runs. Treat them
as strictly read-only:

- `states/`, `states_snapshot_prepru/`, `states_retired/`
- `snapshots/`, `snapshots_sparse/`
- `logs/`
- `paper_data/`
- `render/out/`
- **all `*.h5`, `*.npz`, and `*.csv` files at the repository root**

(These are gitignored, so git will not protect you — a wrong `rm` or overwrite
is permanent.)

## Rules for this refactor

1. **Never delete or overwrite files in the results directories listed above.**
   Read them, don't touch them. If a task seems to require regenerating one, stop
   and ask.
2. **Never run a full parameter sweep, diagram build, or anything expected to
   take more than ~2 minutes without asking Ali first.** The sweep/diagram
   drivers (`main_local.py` with `RUN_SWEEP=True`, `./run.sh`, `make runs`,
   `make run-<TAG>`) fall in this category. Short smoke tests are fine —
   `make tables` on a checkout without `states/` is one.
3. **Make small, focused commits — one logical change per commit.**
4. **Do not change numerical defaults as a side effect of refactoring** —
   tolerances (`pod_energy_tol`, `deim_energy_tol_*`), mode counts
   (`n_velocity_max`, `m_F_max`, …), solver settings, etc. If you believe a
   default is wrong, **tell Ali — do not silently fix it.**
