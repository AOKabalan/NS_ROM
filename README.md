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
├── nsrom/        # the library (installable package)
│   ├── rom/     # POD, DEIM/MDEIM, reduced operators, Galerkin solvers
│   └── ...      # snapshot collection, clustering, Navier–Stokes problem, helpers
├── scripts/     # runnable entry points (solve_fom, main_local, sweep, plotting)
├── scratch/     # exploratory diagnostics and one-off experiments
├── mesh/        # pinball meshes
├── states/      # solution checkpoints
└── pyproject.toml
```

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
python scripts/solve_fom.py
```

Build a local ROM from snapshots and run an online evaluation:

```bash
python scripts/main_local.py
```

Sweep the `(Re, A)` parameter grid with the ROM and compare against the FOM at selected checkpoints:

```bash
python scripts/sweep.py
python scripts/plot_sweep.py
```

Configuration (mesh path, parameter grid, basis sizes, DEIM/MDEIM toggles, clustering strategy) is set at the top of each script.

## Method in brief

The offline stage assembles a snapshot matrix over a sampling of `(Re, A)` covering all relevant solution branches. Snapshots are partitioned by energy-norm k-means++ clustering — with optional whitening to amplify low-energy but branch-discriminative POD modes that would otherwise be drowned out by high-energy smooth components. Each cluster yields its own velocity-plus-supremizer POD basis, a pressure basis, and DEIM/MDEIM bases for the non-affine convection and Jacobian terms.

Online, the solver evaluates the current reduced state against cluster centroids, switches basis if needed through a precomputed change-of-basis matrix, and runs Newton iterations on the reduced residual. Along a continuation path, snake-ordering of the `(Re, A)` grid keeps neighbouring points close in solution space so warm starts remain effective across the sweep.

To locate bifurcations, the reduced Jacobian and a velocity-only mass matrix `M̃_red` define a generalized eigenproblem whose leftmost real eigenvalue changes sign at a pitchfork; the corresponding eigenvector is then used as a kick direction to seed a new branch sub-sweep.

## Citation

If you use this code in academic work, please cite the associated manuscript (in preparation, SISSA mathLab) and reference this repository.

## Author

Ali O. Kabalan, Research Fellow, [SISSA mathLab](https://mathlab.sissa.it/) (Trieste, Italy), under Prof. Gianluigi Rozza. Developed within the iNEST project (Spoke 9).
