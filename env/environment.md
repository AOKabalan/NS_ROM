# Environment / reproducibility record

This directory records the exact software environment used to run `nsrom`, so
the toolchain can be reconstructed on another machine. It was captured on
**2026-08-12** from the working machine.

The three machine-readable companion files are:

| File | What it holds | How it was produced |
|------|---------------|---------------------|
| `pip-freeze.txt` | Every installed Python package + exact version | `pip freeze` (venv active) |
| `firedrake-status.txt` | Firedrake / PETSc / SLEPc / MPI versions **and the external PETSc git commit + configure options** — the reconstruction-critical part | reconstructed by hand (see note below) |
| `system-info.txt` | Interpreter, venv config, OS, CPU, MPI | `python`, `lscpu`, `/etc/os-release`, `ompi_info` |

## ⚠️ Running: `import firedrake` hangs — use `mpiexec -n 1`

On this stack, a plain `python -c "import firedrake"` **hangs indefinitely**. The
hang is inside `MPI_Init`: the import stack is `petsc4py/__init__.py:init` →
PETSc initialization → `MPI_Init`, and `from mpi4py import MPI` hangs identically.
It is **not** a stale compile-cache lock or a hostname-resolution problem — a
fresh `XDG_CACHE_HOME`, loopback-only TCP (`OMPI_MCA_btl_tcp_if_include=lo`), and
PMIx-isolation env vars all still hang. It is an **OpenMPI 5.0.10 singleton
`MPI_Init` hang**.

**Workaround — launch under the MPI runner:**

```bash
mpiexec -n 1 python your_script.py
mpiexec -n 1 pytest            # e.g. tests/  (characterization suite)
```

Under `mpiexec -n 1`, `MPI_Init` completes and Firedrake imports in ~4 s. This is
**one rank with no numerical effect** — purely a launch workaround, not a change
to the computation. Any reproduction of this environment must launch Firedrake
processes this way.

## The short version (what a rebuild needs)

- **Python 3.14.4** (system `/usr/bin/python3.14`), in a venv at `~/venv-firedrake`.
- **Firedrake 2026.4.1**, installed from **PyPI** (not a git checkout).
- **PETSc 3.25.0**, built from **source** at `/home/ali/petsc`
  (`PETSC_ARCH=arch-firedrake-default`), git commit
  **`319083dccc8af9b143f30afbea30a2cdaff26b66`** (tag `v3.25.0`).
- **SLEPc 3.25.0**, built by PETSc via `--download-slepc`.
- **Open MPI 5.0.10** (system install), compilers GCC 15.2.0.
- OS: **Ubuntu 26.04 LTS**, kernel 7.0.0-22-generic, x86_64.
- Hardware capture: Intel Core i7-10750H, 6 cores / 12 threads (single socket).
  Note `-march=native` in the PETSc flags below — the build is tuned to this
  CPU microarchitecture (Comet Lake).

## Why `pip freeze` is not enough

`firedrake-status` (the usual one-shot provenance command) is **not installed**
in this venv, and — more importantly — **PETSc is not a pip package here**. It is
an external source build living at `/home/ali/petsc`, and `petsc4py` was compiled
against it. `pip freeze` records `petsc4py==3.25.0` but says nothing about which
PETSc commit it was built against or with which configure options. Those are the
pieces that actually pin the numerical behavior, so they are captured explicitly
in `firedrake-status.txt`:

- the PETSc **git commit** (`319083d…`), and
- the full PETSc **configure line** (SLEPc, MUMPS, hypre, SuiteSparse, SuperLU_dist,
  ScaLAPACK, (P)netCDF, HDF5, PTScotch/METIS, FFTW; `--with-debugging=0`,
  `-O3 -march=native`).

To reconstruct: create a Python 3.14 venv, check out PETSc at that commit, run
`./configure` with those options and `make`, then `pip install` the packages
pinned in `pip-freeze.txt` (which include `petsc4py==3.25.0`, `slepc4py==3.25.0`,
`mpi4py==4.1.2`, `firedrake==2026.4.1`, `fenics-ufl==2025.3.0`,
`firedrake-fiat==2026.4.0`, `petsctools==2026.0`) against that PETSc, with an
Open MPI 5.0.10 toolchain.

`firedrake-status.txt` was assembled from: package versions via
`importlib.metadata`; `petsc4py.get_config()` for `PETSC_DIR`/`PETSC_ARCH`;
`git -C /home/ali/petsc` for the commit; and
`$PETSC_DIR/$PETSC_ARCH/lib/petsc/conf/petscvariables` for `CONFIGURE_OPTIONS`.
Importing `firedrake` (or the PETSc runtime) directly was avoided because it is
very slow to initialize; none of the reconstruction-critical facts require it.

## Interpreter ambiguity — resolved

The tree carried mixed bytecode — **30 `cpython-312`** and **55 `cpython-314`**
`.pyc` files — which raised the question of which interpreter actually runs the
code. It is now settled: **Python 3.14.4**.

Evidence:

1. **The active venv is 3.14.4, unambiguously.** `~/venv-firedrake/pyvenv.cfg`
   declares `version = 3.14.4` with `executable = /usr/bin/python3.14`, and
   `python --version` after activation reports `Python 3.14.4`
   (built Jun 18 2026, GCC 15.2.0).
2. **Nothing pins any other version.** Script shebangs are version-agnostic
   (`#!/usr/bin/env python` / `python3`, plus `bash`) — none name `python3.12`.
   There is no CI config, no `tox.ini`, no `.python-version`, and the `run_*.sh`
   drivers pin no interpreter; they all defer to whatever the active venv is.
3. **`python3.12` is no longer installed** — only `/usr/bin/python3.14` exists on
   the system.
4. **The bytecode ages don't overlap.** The `cpython-312` `.pyc` are dated
   **2026-05-08 → 06-04**; the `cpython-314` ones **2026-06-22 → 07-31**, running
   right up to the current work. Both sets sit in the same package
   `__pycache__/` dirs (`nsrom/`, `nsrom/rom/`, `nsrom/bifurcation/`, `scripts/`).

**Best guess at how both were generated:** the project ran under Python 3.12
through early June 2026, then the machine was upgraded (Ubuntu 26.04 ships Python
3.14) and the venv was rebuilt on 3.14.4 around 2026-06-22. Every run since has
regenerated `cpython-314` bytecode, while the older `cpython-312` files were
simply never cleaned out — `__pycache__/` is gitignored, so nothing pruned them.
They are **stale residue from the previous interpreter, not evidence of a second
live interpreter.** The authoritative interpreter is Python 3.14.4; the 3.12
`.pyc` can be ignored (or deleted with `find . -name '*.cpython-312.pyc' -delete`,
though that is not required).
