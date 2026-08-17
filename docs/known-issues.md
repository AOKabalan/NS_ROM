# Known issues

Reproducibility gaps and environment traps discovered while building the
characterization test suite. These are recorded to be **fixed later**, not now.

## TODO — Phase 1 staged trash not yet emptied

Phase 1 cleanup moved ~**4.7 GB** of untracked, superseded/backup files to
**`/home/ali/nsrom_phase1_trash/`** (manifest in `MOVED.log` there) instead of
deleting them. They are still recoverable — move back if needed. **Reclaim the
space with `rm -rf /home/ali/nsrom_phase1_trash` only after confirming nothing
is needed.** Notably it holds `states_snapshot_prepru/` (the pre-pruning
snapshot; its unique pre-prune records are also preserved in
`states/**/*.prebak_*`). Do not empty during Phase 2.

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

## 3. `ROMSolution.residual_norm` is never populated — silently 0.0 (latent bug)

`nsrom/rom/rom_solver.py::_extract_solution` constructs the `ROMSolution` with
`iterations` and `converged` but **never sets `residual_norm`**, so it keeps its
default of `0.0` (`nsrom/rom/data_structures.py:381`). The true final Newton
residual (e.g. `1.94e-8` for the B2 characterization solve) exists only in the
SNES stdout, never on the returned object.

Every consumer therefore reads a silent `0.0`. Affected call sites:

- `nsrom/bifurcation/detection.py:227, 250, 261` — the bifurcation detector
  records `rom_solution.residual_norm` in its per-Re result dict; always `0.0`
  for solved points (the `:213` path uses an explicit `None` instead).
- `nsrom/rom/data_structures.py:607` — serializes `solution.residual_norm` into
  saved solution metadata; always `0.0`.
- `nsrom/rom/data_structures.py:666` — batch save writes a `residual_norms`
  array that is all `0.0`; `:710` reads it back.
- Downstream readers of those saved arrays (e.g.
  `scripts/compare_hyperreduction.py:140`, which looks for `residual_norms` /
  `fnorms` keys) consequently see `0.0`.

**To fix later (not now):** populate `residual_norm` in `_extract_solution`
(e.g. from the SNES final function norm) so detection records and saved
metadata carry the real value. Until then, treat any `residual_norm` in
detection output or saved sweeps as meaningless.

## 4. Energy-norm k-means clustering does NOT track branch identity

Characterization finding (case A5). On a 39-snapshot, amp-spanning,
block-stratified subsample (13 each from branches 0/1/2), energy-norm k-means
(Cholesky whitening of the H1 inner product, `n_clusters=3`) gives
`adjusted_rand_score(labels, branch_ids) ≈ 0.014` — i.e. **no agreement with
branch identity**. The clusters partition by energy / Reynolds regime, not by
branch.

This is expected and correct for local-ROM partitioning (clusters group
solutions that share a reduced basis well, which is an energy-similarity
criterion, not a topological branch label). But it means the statement
"**clusters correspond to branches**" is **false** and must not be assumed in
write-ups or figure captions. An earlier impression that clustering recovered
the branches was an artifact of an interleaved subsample ordering, not a real
correspondence.

## Note — global ROM = local pipeline with `n_clusters=1`

The global ROM baseline is obtained by running the **local** pipeline
(`scripts/main_local.py`, `LocalROMConfig(n_clusters=1, ...)`), not a separate
global builder. The former standalone scripts `scripts/main_global_rom.py` and
`scripts/build_diagram_bare_global_rom.py` were **retired** (moved to
`/home/ali/nsrom_phase1_trash/scripts/`) as redundant.

Why they're equivalent: at `n_clusters=1` the single cluster contains **all**
snapshots (`cluster_indices[0] = np.where(labels==0)[0]` = every index), and the
local basis path (`build_cluster_pod`) calls the **same** `compute_pod_basis`
(same snapshots, `inner_product_type='H1'`, `compute_supremizer=True`, same
`u_base`/`u_control` centering, same `boundary_markers`), the **same** truncation
rule (`modes_for_tolerance(evals, pod_energy_tol, n_*_max)` with `n_sup = n_pres`),
the **same** `build_reduced_operators`, and the **same** `solve_rom`. The K=1
cluster-selection / change-of-basis machinery is a no-op.

**Caveat for write-ups — the one real difference:** the truncation *cap* differed.
The retired global script hard-coded `N_VELOCITY_MODES = N_PRESSURE_MODES = 50`,
whereas the local pipeline uses `n_velocity_max=400`, `n_pressure_max=200`
(defaults, env-overridable). So the POD **modes are identical**, but if the energy
tolerance retains more than 50 modes at K=1 (likely — the K=4 run already reached
53 velocity modes in one cluster), the two would keep **different mode counts**.
The current baseline therefore uses the energy-tolerance truncation (capped at
400/200), consistent with every other local ROM — state this rather than the
old 50-mode cap when describing the global baseline.

## `*_tol1e-12` POD bases — invalid caches, reproduction, and orthonormality

### Invalid historical caches

- `local_rom/K1_H1_tol1e-12/` and `local_rom/K4_H1_tol1e-12/` contain **invalid
  27-velocity-mode bases** and **must not be cited as genuine `tol=1e-12` results.**
- The stored K1 `velocity_energy_fraction` is **`0.9999627062891424`**.
- A fresh production-matrix calculation gives cumulative energy **`0.9999627062891426`**
  at mode 27 — agreement **within 1 ULP, not bit-for-bit**. (An earlier exact match
  used the *saved* mass matrix; the freshly *assembled* H1 matrix differs by 1 ULP.)
- This corresponds to an **effective discarded-energy tolerance of ≈ `3.73e-5` (`~4e-5`)**,
  not `1e-12`.
- The **precise historical build-time mistake remains unknown.** These are **stale /
  mislabeled artifacts with unresolved build provenance**; skipped recomputation or
  cache carry-over are unverified possibilities and are **not** asserted as proven.
- **Ruled-out explanations (evidence from the tested real K1 spectrum, not a universal
  guarantee for every possible spectrum):** the boundary and negative-tail explanations
  were ruled out. The `1e-12` target and `cumulative.max()` differed by approximately
  **4499 ULPs**; an out-of-range `searchsorted` result drives the selected count
  **upward toward the configured cap, not down to 27**. NumPy's `np.linalg.eigvalsh` and
  the production pipeline's SciPy `scipy.linalg.eigh` produced the same effective
  spectrum, with no negative-tail collapse capable of explaining the stored basis.
- K4: the existing `K4_H1_tol1e-12` cache is invalid based on its stored dimensions
  (27 velocity modes in every cluster) and energy behavior, but **an isolated K4
  rebuild has NOT been performed** — this note does not claim K4 was rebuilt.

### Successful current-pipeline reproduction (K1)

An isolated K1 rebuild on **2026-08-17**, using the current unmodified production POD
path and a fresh output root outside the repository, produced:

- velocity: **182** modes
- pressure: **106** modes
- supremizer: **106** modes
- velocity retained energy: ≈ **`0.9999999999990026`**
- pressure retained energy: ≈ **`0.9999999999990763`**
- enriched velocity / operator dimension: **`288 = 182 + 106`**
- consistent reduced-operator shapes (`A_N (288,288)`, `B_N (106,288)`, `Z_u` 288, `Z_p` 106)
- an exact-mode `Re=80`, `amp=0.1` smoke solve **converging in four iterations**

This demonstrates:

- `tol=1e-12` is **accepted and reproducible** by the current K1 pipeline;
- the current `modes_for_tolerance` selects **182** K1 velocity modes, **not 27**;
- the historical K1 cache is **invalid**, rather than evidence of a current truncation bug.

**Latent (defensive) hardening of `modes_for_tolerance`** (`nsrom/helper_functions.py:150`)
remains worth doing later but **did not cause the historical 27-mode artifact**:
negative-tail handling (clip negatives before the cumulative sum), explicit
bounds/clamping of the selected count, and a warning when a requested tolerance is
unattainable (`tol < 1 − cumulative.max()`). Deferred; no code changed here.

**Cache status / required action.** The isolated K1 rebuild remains **validation output
outside the repository** and has **not** replaced `local_rom/K1_H1_tol1e-12/`. Both
historical `tol1e-12` production caches remain **quarantined from scientific use.** K1
must be **explicitly replaced from a reviewed rebuild**, and K4 must be **independently
rebuilt and validated**, before either cache is used as a genuine `tol=1e-12` result.

### Tight-tolerance raw-POD orthonormality degrades with retained low-energy modes

Read-only comparison using the current production-assembled H1 velocity and L2 pressure
matrices. Raw POD **(M)-orthonormality degrades progressively** as increasingly
low-energy modes are retained. Velocity H1 evidence (`max |off-diagonal of ΦᵀMΦ|`):

| Basis                | Velocity modes | max off-diagonal of `ΦᵀMΦ` | `1e-7` check |
| -------------------- | -------------: | -------------------------: | ------------ |
| `tol=1e-4`           |             23 |                 `2.20e-13` | pass         |
| `tol=1e-6`           |             49 |                 `2.90e-11` | pass         |
| untolled `K1_H1`     |             70 |                 `2.61e-10` | pass         |
| `tol=1e-8`           |             84 |                  `1.96e-9` | pass         |
| `tol=1e-10`          |            128 |                  `2.18e-7` | fail         |
| isolated `tol=1e-12` |            182 |                  `2.36e-5` | fail         |

- Pressure shows the **same progression** and fails the `1e-7` check at `1e-10` and
  `1e-12` (max off-diagonal `3.4e-7` and `2.58e-5`).
- The worst errors occur **between the highest-index, lowest-energy modes**.
- At `1e-12`, the offending modes' eigenvalues are ≈ **`2.5e-13` of the leading
  eigenvalue**.
- **Gram condition numbers remain ≈ 1**, so this is a **gradual loss of strict
  orthogonality**, not an ill-conditioned or singular basis.
- The isolated rebuild therefore **passed** truncation, energy, metadata,
  operator-dimension, and smoke-solve checks, but **did NOT pass** the existing `1e-7`
  raw-POD orthonormality criterion (velocity `2.36e-5`, pressure `2.58e-5`). The
  **same progressive trend is already visible in the pre-existing `tol=1e-10` cache,
  which also fails the current `1e-7` raw-POD criterion** — not rebuild-specific.
- Raw `[velocity, supremizer]` stacks are **not** expected to be orthonormal (measured
  off-diagonal ≈ `0.10–0.14` across every basis). `build_reduced_operators` enriches the
  velocity basis via `enrich_velocity_basis` (`nsrom/rom/operators.py`), which
  Gram-Schmidt-orthonormalizes the supremizer modes against the velocity modes in the
  `M_u` (H1) inner product — **reconfirmed directly from the current code**.

**Interpretation (not a formally proven root cause):** normalization of increasingly
tiny method-of-snapshots eigenmodes (`φᵢ = S vᵢ / ‖S vᵢ‖_M`) amplifies floating-point
error, so cross-`M` inner products between near-null-space modes drift from zero.

**Conclusion:** `tol=1e-12` is **usable as a sensitivity experiment**, but a tighter
retained-energy tolerance does **not automatically imply a more accurate ROM** once
modes near the numerical null space are included. Downstream error / stability
comparisons are still required before treating `tol=1e-12` as the preferred publication
configuration.

## 5. SLEPc eigenvalue results depend on solve history within the process

`nsrom/bifurcation/eigen_solver.py::solve_leftmost_real_eigenpairs` sets **no**
`setInitialSpace` on the EPS, so SLEPc uses a start vector drawn from PETSc's
global random state. That state is **advanced by any earlier solve in the same
process** — e.g. a preceding SNES solve — so the eigenvalue result depends on
what ran before it:

- **Standalone, B3 is byte-reproducible** (repeated fresh processes give
  identical eigenvalues).
- Run **after** the B2 ROM solve in the same process, B3 returns a **different**
  set (observed `n_converged` 4 vs 5, and leftmost real part −0.0562 vs −0.0653)
  because B2's Newton solve advanced the PETSc random state.
- The B3 golden is therefore currently frozen as a **first-solve** capture
  (matching how `pytest` invokes it — before any solve). **This will break if
  test ordering changes** so that a solve precedes B3; the test is non-blocking
  (xfail on drift) precisely to absorb that.

**Production implication (not just a test artifact):** in a real bifurcation
sweep, every eigenvalue computation runs after many prior solves in the same
process, so **detected eigenvalues — and thus bifurcation detection — are not
reproducible across runs** whenever solve history differs. The values are close
in character (leftmost real part, the Hopf pair) but not bit-stable.

**Recommended fix (later, not now):** set a deterministic initial vector on the
EPS (`eps.setInitialSpace(v0)` with a fixed `v0`, or seed PETSc's random via
`PetscRandom` before the solve) so results are independent of solve history.
This would make bifurcation detection reproducible across runs.
