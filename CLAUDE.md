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
    Newton solver (`rom_solver.py` owns the authoritative `VALID_MODES`).
  - `nsrom/bifurcation/` — eigenproblem detection, Jacobian, branch jumping,
    sweep study.
  - top level: snapshot collection, clustering, Navier–Stokes problem setup,
    lifting functions, tensor convection, config/layout/cache helpers.
- **`scripts/`** — CLI drivers / entry points. `main_local.py` is the primary
  local-ROM pipeline; also `solve_FOM.py`, `sweep.py`, `build_diagram_bare*.py`,
  `replay_diagram.py`, `state_store.py`, plotting scripts. Config lives in
  module-level constants at the top of each script.
- **`section_6_figures/`** and **`paper_figures/`** — generate manuscript
  figures and LaTeX macro/table files. `section_6_figures/out/` holds the
  generated PDFs, CSVs, and `.tex` output.

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
- `section_6_figures/out/`
- **all `*.h5`, `*.npz`, and `*.csv` files at the repository root**

(These are gitignored, so git will not protect you — a wrong `rm` or overwrite
is permanent.)

## Rules for this refactor

1. **Never delete or overwrite files in the results directories listed above.**
   Read them, don't touch them. If a task seems to require regenerating one, stop
   and ask.
2. **Never run a full parameter sweep, diagram build, or anything expected to
   take more than ~2 minutes without asking Ali first.** The sweep/diagram
   drivers (`sweep.py`, `build_diagram_bare*.py`, `main_local.py` with
   `RUN_SWEEP=True`, `run_all.sh`, `run_gaps.sh`, `run_replay.sh`) fall in this
   category. Short smoke tests are fine.
3. **Make small, focused commits — one logical change per commit.**
4. **Do not change numerical defaults as a side effect of refactoring** —
   tolerances (`pod_energy_tol`, `deim_energy_tol_*`), mode counts
   (`n_velocity_max`, `m_F_max`, …), solver settings, etc. If you believe a
   default is wrong, **tell Ali — do not silently fix it.**
