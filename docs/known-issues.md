# Known issues

Reproducibility gaps and environment traps discovered while building the
characterization test suite. These are recorded to be **fixed later**, not now.

## 1. `import firedrake` hangs in singleton mode (environment blocker)

On the current stack (OpenMPI **5.0.10**, Python 3.14.4), a plain
`python -c "import firedrake"` hangs forever inside `MPI_Init`
(`petsc4py.init` → PETSc init → `MPI_Init`; `from mpi4py import MPI` hangs the
same way). Not a stale cache/lock or hostname issue — fresh cache, loopback-only
TCP, and PMIx-isolation env vars all still hang.

**Workaround:** launch everything under `mpiexec -n 1` (one rank, no numerical
effect):

```bash
mpiexec -n 1 python your_script.py
mpiexec -n 1 pytest
```

Also documented in `CLAUDE.md` and `env/environment.md`.

## 2. `compute_lifting_functions` cannot regenerate the stored lifting

`nsrom.lifting_functions.compute_lifting_functions(problem, reynolds_ref=100.0)`
does **not** converge from a cold Re=100 start:

- `u_base` solve **diverges** — `DIVERGED_LINE_SEARCH` after 28 SNES iterations —
  and returns a non-physical field with `‖u_base‖_L2 ≈ 13.86`.
- The **stored** lifting in `lifting/` (loaded via `load_lifting_functions`) has
  `‖u_base‖_L2 ≈ 19.37` — a different function.
- `u_control` is fine (converges in ~7 iterations; matches the stored
  `‖u_control‖ ≈ 1.3607`).

So the stored lifting functions **cannot currently be regenerated** by the
public entry point from a cold start — a reproducibility gap. The stored
artifacts were presumably produced with a continuation / warm start that the
current code path does not apply.

**Impact on tests:** the B1 characterization case therefore pins the **stored**
artifact (via `load_lifting_functions`), not a fresh recompute.

**To fix later (not now):** give the `u_base` lifting solve a proper initial
guess / Reynolds continuation so `compute_lifting_functions` reproduces the
stored functions, then B1 can characterize the recompute path.
