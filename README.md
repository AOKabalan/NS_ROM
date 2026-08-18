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
├── scripts/                  # wrappers plus standalone CLI/diagnostic tools
├── mesh/                     # pinball meshes
├── states/                   # solution checkpoints
└── pyproject.toml
```

`scripts/main_local.py` is the supported shell-facing local-ROM entry point; it
delegates to `nsrom.workflows.local_pipeline`. Not every script is a wrapper:
standalone tools such as `scripts/fom_bifurcation_diagram_fom.py`,
`scripts/cluster_diagnostics.py`, and the point-error and plotting diagnostics
remain intentionally outside the package workflows.

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

## Reproducibility commands

Run Firedrake-backed tests through one MPI rank:

```bash
make test-fast
make test
```

Regenerate the proven repository-contained renderer subset:

```bash
make figures
make figures-section6
make figures-paper
```

`make figures` is intentionally partial. The Section 6 target reads the tracked
`section_6_figures/out/critical_curve.csv` and rewrites the derived
`tab_critcurve.tex` and `macros_critcurve.tex` in that directory. The paper
target reads `mesh/mid_pinball.msh` and writes `figures/pinball_geometry.pdf`
and `figures/pinball_geometry.png`.

Other Section 6 and paper figures require local `states/`, `local_rom/`,
`paper_data/`, `mass/`, POD, or sweep artefacts that are not available from a
fresh checkout. Heavy FOM/ROM solves, basis construction, data generation, and
full parameter sweeps are excluded from these rendering targets. Commands that
import Firedrake, including both test targets, must run through one MPI rank.

## Method in brief

The offline stage assembles a snapshot matrix over a sampling of `(Re, A)` covering all relevant solution branches. Snapshots are partitioned by energy-norm k-means++ clustering — with optional whitening to amplify low-energy but branch-discriminative POD modes that would otherwise be drowned out by high-energy smooth components. Each cluster yields its own velocity-plus-supremizer POD basis, a pressure basis, and DEIM/MDEIM bases for the non-affine convection and Jacobian terms.

Online, the solver evaluates the current reduced state against cluster centroids, switches basis if needed through a precomputed change-of-basis matrix, and runs Newton iterations on the reduced residual. Along a continuation path, snake-ordering of the `(Re, A)` grid keeps neighbouring points close in solution space so warm starts remain effective across the sweep.

To locate bifurcations, the reduced Jacobian and a velocity-only mass matrix `M̃_red` define a generalized eigenproblem whose leftmost real eigenvalue changes sign at a pitchfork; the corresponding eigenvector is then used as a kick direction to seed a new branch sub-sweep.

## Citation

If you use this code in academic work, please cite the associated manuscript (in preparation, SISSA mathLab) and reference this repository.

## Author

Ali O. Kabalan, Research Fellow, [SISSA mathLab](https://mathlab.sissa.it/) (Trieste, Italy), under Prof. Gianluigi Rozza. Developed within the iNEST project (Spoke 9).
