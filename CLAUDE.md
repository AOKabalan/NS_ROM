# CLAUDE.md

Persistent guidance for a Claude Code session working in this repository. This
file is for the coding agent — user-facing setup and workflow live in the docs
linked at the bottom. Keep it short.

## What this project is

`nsrom` is a reduced-order modeling (ROM) framework for the **fluidic pinball**:
steady, incompressible Navier–Stokes flow past three rotating cylinders,
parametrized by Reynolds number `Re` and cylinder rotation amplitude `A`. The
scientific goal is the bifurcation structure of steady solutions in `(Re, A)` —
branch tracking and localization of the critical curve. The steady pitchfork
sits near `Re ≈ 68.22`.

The methodology: snapshots over `(Re, A)` across solution branches; POD with
supremizer enrichment for inf–sup stability; separate pressure and
velocity/supremizer reduced spaces; local ROMs from energy-norm clustering;
Galerkin reduced Newton solves with cluster-aware warm starts; and reduced
bifurcation detection.

The nonlinear convection term is evaluated per the `mode` field in
`LocalROMConfig`:

- `exact`  — reassemble the nonlinear term on the full-order mesh;
- `deim`   — DEIM/MDEIM hyper-reduction;
- `tensor` — precomputed quadratic convection tensor.

`tensor` is the canonical manuscript mode; `deim` is one option, not the
defining strategy. Confirm the mode from the current experiment configuration
before making claims about paper methodology.

Built on Firedrake, PETSc, SLEPc, MPI, NumPy/SciPy.

## Scientific priority

This is research software: scientific correctness and reproducibility outrank
cosmetic cleanup. Refactoring must preserve numerical behavior unless the task
explicitly asks for a scientific change.

**Never silently change** equations/weak forms, boundary conditions, POD
definitions, inner products (velocity `H1`, pressure `L2`), lifting/
homogenization, clustering metrics, basis dimensions, tolerances, solver
settings, continuation logic, bifurcation criteria, tensor index conventions,
or numerical defaults.

- If a numerical default looks wrong, **report it — do not change it.**
- If a structural change moves numerical outputs, **stop and investigate why.**

The full safety policy is `.claude/rules/scientific-safety.md`.

## Architecture

```text
nsrom/     installable package — all reusable numerical code
scripts/   shell-facing entry points and diagnostics
render/    manuscript figure/table generation (style.py, common.py, fig_*, tab_*, out/)
run.sh     the explicit experiment matrix (expensive runs)
Makefile   cheap dependency graph for regenerating manuscript artifacts
```

- **Reusable numerical functionality belongs in `nsrom/`**, not duplicated in
  scripts. Subpackages: `rom/` (POD/DEIM, operators, solvers, local ROM),
  `bifurcation/` (detection, sweep, diagram, replay), `clustering/`,
  `snapshots/`, `io/`, `plotting/`, `workflows/`. Legacy top-level modules
  (`local_rom`, `cluster_building`, `snapshot_collection`, `scripts.sweep`,
  `scripts.generate_snapshots`) were migrated into these subpackages — do not
  reintroduce the old import paths.
- **`scripts/`** should parse args, call package functionality, save/report —
  no reusable algorithms, no `sys.path` hacks where a normal import works.
- **`render/`** owns manuscript artifacts. The LaTeX manuscript is authoritative
  about which figures/tables are actually used; distinguish manuscript artifacts
  from exploratory/diagnostic plots.
- **`run.sh`** defines the expensive experiments, one tagged entry each
  (`./run.sh --list`).
- **`Makefile`** rebuilds cheap artifacts on a timestamp. **Expensive
  experiments are never automatic Make targets** — `make paper` renders from
  existing `states/` and must never launch a sweep. A renderer takes seconds; a
  sweep takes hours.

## Expensive and irreplaceable data

Treat these as **read-only** unless a task explicitly requires regeneration, and
**never delete or overwrite them as part of cleanup**:

```text
states/  states_snapshot_prepru/  states_retired/
snapshots/  snapshots_sparse/  multi_param_multi_branch/
local_rom/  mass/  logs/  paper_data/  render/out/
```

Also be conservative with any `*.h5`, `*.npz`, or scientific `*.csv`, especially
when not tracked by Git. Rebuilding `local_rom/` or `mass/` re-derives the norms
every reported error is quoted in. If unsure whether a dataset is replaceable,
**stop and report** rather than regenerate. `render/out/*.{csv,tex}` are tracked
reproducibility assets — regenerate them only through the renderers, never by
hand.

## Environment (machine-specific caveat)

The project uses Firedrake (with PETSc/SLEPc/MPI); a generic
`pip install -r requirements.txt` will not reproduce it. On Ali's current
development machine the venv is activated with:

```bash
source ~/venv-firedrake/bin/activate
```

That path is a local detail, not a portable requirement. On this machine a plain
`import firedrake` currently **hangs** in singleton `MPI_Init`; the workaround is
to launch under one MPI rank:

```bash
mpiexec -n 1 python your_script.py
mpiexec -n 1 pytest
```

This is a **machine-specific workaround, not a universal project requirement.**
Full details — versions, PETSc build, the hang analysis — are in
`env/environment.md` and `docs/known-issues.md`.

## Working discipline

```text
inspect → minimal change → test → inspect diff → continue
```

- Before repository-wide work, inspect `git status`, `git diff`,
  `git diff --cached`. There may be legitimate unfinished work in the tree —
  never discard or overwrite changes you do not understand.
- Prefer cheap verification first (syntax/static → imports → light unit tests →
  loading existing outputs → regenerating cheap artifacts) over expensive
  regeneration. Never claim a command passed unless it was actually run; if it
  cannot be run, say why.
- Before deleting/archiving a file, check its references (imports, `run.sh`,
  `Makefile`, manuscript, Git history). When uncertain, classify it as an
  obsolete candidate rather than deleting it.
- **Do not commit unless explicitly requested.**

## Where to look

Do not duplicate these here — link and defer:

- **`README.md`** — project overview, install, user quick start.
- **`docs/workflow.md`** — dependency graph, the full recomputation matrix (what
  to rerun when style / grid / POD tolerance / `K` / `sym_start` / snapshots
  change), and rebuilding ROM caches from existing snapshots.
- **`docs/data.md`** — included-vs-generated data layout, `data_manifest.json`,
  the external-data bundles and verifier.
- **`docs/known-issues.md`** — known scientific/cache issues (e.g. the invalid
  `*_tol1e-12` bases; global-baseline notes).
- **`env/environment.md`** — environment reconstruction and the MPI workaround.
- **`.claude/rules/scientific-safety.md`** — the enforced data/scientific safety
  rules.
