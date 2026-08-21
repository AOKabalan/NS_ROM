# NS_ROM

**Reduced-order modeling for the fluidic pinball, built on Firedrake.**

`nsrom` is a Python library for constructing local reduced-order models (ROMs) of the steady, incompressible Navier–Stokes equations on bifurcating parametric problems. The reference benchmark is the **fluidic pinball** — flow past three rotating cylinders parametrized by Reynolds number `Re` and rotation amplitude `A` — which exhibits a Hopf bifurcation near `Re ≈ 18` and a steady pitchfork at `Re ≈ 68.22`.

The code implements the full offline–online workflow: snapshot collection, POD with supremizer enrichment, (M)DEIM hyper-reduction, energy-norm clustering for local bases, Galerkin Newton solves with cluster-aware warm starts, and a generalized-eigenproblem-based bifurcation detector with branch-switching.

## Features

- **POD–Galerkin projection** for the parametric steady Navier–Stokes equations, with supremizer enrichment for inf–sup stability.
- **DEIM / MDEIM hyper-reduction** with submesh-based assembly for fully affine online cost on the nonlinear convection term and its Jacobian.
- **Local ROMs via energy-norm k-means++ clustering**, with optional Cholesky or POD whitening; each cluster carries its own POD basis and DEIM operators.
- **Newton solver with change-of-basis warm starts** between clusters, plus snake-ordered continuation for sweeps over `(Re, A)`.
- **Bifurcation detection** via a reduced generalized eigenproblem `J_red ν = μ M̃_red ν`, tracking the sign of the leftmost real eigenvalue along a branch.
- **Branch switching** through eigenvector-kick warm starts at detected pitchfork points, cross-validated against a full-order SLEPc Krylov–Schur pipeline.

## Repository layout

```
NS_ROM/
├── nsrom/                    # installable package
│   ├── rom/                  # POD/DEIM, operators, solvers, local ROMs
│   ├── bifurcation/          # detection, sweep, diagrams, replay
│   ├── clustering/           # clustering and whitening utilities
│   ├── snapshots/            # snapshot generation, collection, storage helpers
│   ├── io/                   # state store and mass-matrix persistence
│   ├── plotting/             # plotting and speedup reporting
│   └── workflows/            # local pipeline and hyper-reduction studies
├── render/                   # every paper figure and table, plus their style
├── scripts/                  # CLI and diagnostic entry points
├── mesh/                     # pinball meshes
├── states/                   # solution checkpoints
├── run.sh                    # the experiment matrix
├── Makefile                  # the dependency graph
└── pyproject.toml
```

`scripts/main_local.py` is the supported shell-facing local-ROM entry point; it
delegates to `nsrom.workflows.local_pipeline`. Standalone tools such as
`scripts/fom_bifurcation_diagram_fom.py`, `scripts/cluster_diagnostics.py`, and
the point-error and plotting diagnostics sit intentionally outside the package
workflows.

## Installation

`nsrom` depends on [Firedrake](https://www.firedrakeproject.org/), which itself provides PETSc and SLEPc. Install Firedrake first, then activate its virtual environment and install this package in editable mode:

```bash
source /path/to/firedrake/bin/activate
git clone https://github.com/AOKabalan/NS_ROM.git
cd NS_ROM
pip install -e .
```

Verify the install:

```bash
python -c "import nsrom; print(nsrom.__version__)"
```

## Quick start

Run a full-order solve on a single `(Re, A)` point:

```bash
python scripts/solve_FOM.py
```

Build a local ROM from snapshots and run an online evaluation:

```bash
python scripts/main_local.py
```

The same local-pipeline entry point runs configured parameter diagrams and FOM
comparisons. Its module-level configuration lives in
`nsrom/workflows/local_pipeline.py`; standalone commands retain their own
module-level configuration under `scripts/`.

## Reproducing the results

Every experiment the manuscript depends on is in `run.sh`, one entry per run,
each tagged with the figures and tables it feeds:

```bash
./run.sh --list          # the matrix, and what each run is for
./run.sh E1_K4_tensor    # one run (hours)
./run.sh all             # everything not already present
```

`make` knows which run feeds which artifact, so only the affected things
rebuild:

```bash
make paper     # every figure, table and macro file the manuscript uses
make figures   # the figures only
make tables    # the tables only
make list      # what is built, and which runs are present
make sync      # copy render/out into the paper repo
```

Experiments are deliberately not `make` targets: a renderer takes seconds and
may rebuild on a timestamp, a sweep takes hours and must not. `make paper` will
never start one — use `make run-<TAG>` or `./run.sh`.

From a fresh checkout with no `states/`, `make tables` still rebuilds
`tab_critcurve.tex` and `macros_critcurve.tex` from the tracked
`render/out/critical_curve.csv`. Everything else needs the run artefacts, which
live outside Git; a renderer rule is defined only when its runs are present, so
make leaves committed output alone rather than regenerating it from nothing.

For **what to recompute when a scientific parameter changes** — style, grid,
POD tolerance, `K`, `sym_start`, or the training snapshots — and for rebuilding
the ROM caches from existing snapshots without a full sweep, see
[`docs/workflow.md`](docs/workflow.md).

Tests run through one MPI rank, as does anything importing Firedrake:

```bash
make test-fast
make test
```

## External data

Large snapshots, ROM caches, and computed experiment states are distributed
separately from the Git repository. They are inventoried in `data_manifest.json`.

Download both archives from the links documented in
[`docs/data.md`](docs/data.md):

1. `NS_ROM_external_data_full_636bbb2.tar.zst`
2. `NS_ROM_external_data_supplement_0664a90.tar.zst`

Verify the accompanying SHA-256 checksums, then extract the archives into the
repository root **in that order**:

```bash
tar --extract --zstd \
  --file=/path/to/NS_ROM_external_data_full_636bbb2.tar.zst \
  --directory=/path/to/NS_ROM

tar --extract --zstd \
  --file=/path/to/NS_ROM_external_data_supplement_0664a90.tar.zst \
  --directory=/path/to/NS_ROM
```

Then verify the installation:

```bash
make list        # all 18 canonical run tags present
make test-fast   # 31 passed, 3 deselected
make paper       # regenerate manuscript artifacts from existing results
```

Without the external data, data-dependent characterization tests are skipped
(see [`docs/data.md`](docs/data.md)). With both archives installed, all 18
canonical run tags are available.

## Method in brief

The offline stage assembles a snapshot matrix over a sampling of `(Re, A)` covering all relevant solution branches. Snapshots are partitioned by energy-norm k-means++ clustering — with optional whitening to amplify low-energy but branch-discriminative POD modes that would otherwise be drowned out by high-energy smooth components. Each cluster yields its own velocity-plus-supremizer POD basis, a pressure basis, and DEIM/MDEIM bases for the non-affine convection and Jacobian terms.

Online, the solver evaluates the current reduced state against cluster centroids, switches basis if needed through a precomputed change-of-basis matrix, and runs Newton iterations on the reduced residual. Along a continuation path, snake-ordering of the `(Re, A)` grid keeps neighbouring points close in solution space so warm starts remain effective across the sweep.

To locate bifurcations, the reduced Jacobian and a velocity-only mass matrix `M̃_red` define a generalized eigenproblem whose leftmost real eigenvalue changes sign at a pitchfork; the corresponding eigenvector is then used as a kick direction to seed a new branch sub-sweep.

## Citation

If you use this code in academic work, please cite the associated manuscript (in preparation, SISSA mathLab) and reference this repository.

## Author

Ali O. Kabalan, Research Fellow, [SISSA mathLab](https://mathlab.sissa.it/) (Trieste, Italy), under Prof. Gianluigi Rozza. Developed within the iNEST project (Spoke 9).
