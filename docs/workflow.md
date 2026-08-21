# Workflow and recomputation

This note answers one question: **if I change X, what must I recompute?**
It complements `README.md` (quick start) and `make help` (the three kinds of
change). It does not replace them.

## The dependency graph

```text
snapshots  (multi_param_multi_branch/)          # training data, external
   │
   ▼
energy transform  →  M_u (H1), M_p (L2)         # mass matrices, in-memory
   │
   ▼
clustering        (cached, key: K)              # local_rom/.../clustering
   │
   ▼
POD bases + reduced operators + tensor/DEIM     # cached in
   │        (cache key: K, inner product, POD tol)   local_rom/K<K>_<ip>_tol<tol>
   ▼
ROM sweep  (build_full_diagram_bare)  →  states/<TAG>/
   │
   ▼
render/*.py  →  render/out/*.{csv,tex,pdf}  →  manuscript (make sync)
```

Two facts make the graph cheap to walk:

- **The `local_rom/` cache is keyed by `K`, the velocity inner product, and the
  POD tolerance** (`nsrom/layout.py`). Changing any of them selects a *different*
  cache directory, so the affected bases/operators rebuild automatically while
  everything else is reused.
- **Experiments are never `make` targets.** A renderer is cheap and may rebuild
  on a timestamp; a sweep takes hours and is launched only by `./run.sh <TAG>`
  (or `make run-<TAG>`). `make paper` never starts a sweep.

## Recomputation matrix

Levels of work, cheapest to most expensive:

1. **render** — re-run a renderer, seconds.
2. **analysis** — re-run the sweep over existing ROM caches (no basis rebuild).
3. **operators** — rebuild POD / tensor / DEIM from existing snapshots.
4. **clustering** — rebuild the cluster partition, then operators.
5. **snapshots/FOM** — regenerate the training data itself.

| # | Change | Recompute | Do NOT recompute | How |
|---|--------|-----------|------------------|-----|
| A | Plot style / font / line width (`render/style.py`) | render (figures) | tables need not rebuild unless shared | `make figures` |
| B | Table formatting (a table renderer) | render (that table) | figures, states, ROM | `make tables` |
| C | Table-only threshold / display cutoff (in a renderer) | render (that artifact) | states, ROM, snapshots | `make <artifact>` / `make tables` |
| D | Validation grid — the swept `(Re, A)` points (`NSROM_AMPS`, `NSROM_RE_*`) | the sweep → `states/<TAG>`, then artifacts | snapshots, clustering, POD, operators (caches reused) | `./run.sh <TAG>` → `make paper` |
| E | ROM solver setting (Newton tol / solver params) | the sweep → `states/<TAG>`, then artifacts | snapshots, clustering, POD, operators | `./run.sh <TAG>` → `make paper` |
| F | POD tolerance (`NSROM_POD_TOL`) | POD + operators + tensor/DEIM (new cache key), then sweep + artifacts | snapshots, clustering | `./run.sh <TAG>` with `NSROM_RECOMPUTE_POD=1 NSROM_RECOMPUTE_TENSOR=1` (DEIM: `NSROM_RECOMPUTE_DEIM=1`) → `make paper` |
| G | Number of clusters `K` | clustering + POD + operators (new cache), then sweep + artifacts | snapshots | `./run.sh <TAG>` (new `NSROM_K`, recompute flags as F) → `make paper` |
| H | Training snapshot selection | **everything downstream**: clustering, POD, operators, all states, all artifacts | mesh / FOM problem (unless snapshots are themselves re-solved) | `make clean-derived && make runs && make paper` |
| I | FOM discretization / mesh / governing problem | snapshots (re-solve FOM) **then everything in H** | nothing downstream is safe | regenerate snapshots, then as H |
| J | `sym_start` | the sweep → `states/<TAG>`, then artifacts | snapshots, clustering, POD, operators (caches reused) | `./run.sh <TAG>` (value is pinned per run via `NSROM_SYM_START`) → `make paper` |

D, E, F, G, H, I, J all launch at least one **expensive** experiment. Only
A, B, C are cheap renderer-only work.

## Rebuild ROM data from existing snapshots

The snapshots are trusted; you want fresh clustering / POD / operators. Running
the pipeline with the matching `NSROM_RECOMPUTE_*` flags rebuilds exactly those
caches from `multi_param_multi_branch/` and reuses the rest. The canonical
paper caches are produced this way as a byproduct of the experiments — e.g.
`E7_*` rebuild POD + tensor, `E4`/`E9`/`E15` rebuild DEIM (see `./run.sh --list`
and the `NSROM_RECOMPUTE_*` flags in `run.sh`).

To rebuild the caches **without** the hours-long diagram sweep, the pipeline
exposes `NSROM_RUN_SWEEP=0`, which builds clustering + POD + operators and then
does a single-point FOM comparison instead of the full sweep:

```bash
# rebuild the K=4, tol=1e-8 tensor ROM caches from snapshots, no sweep
NSROM_K=4 NSROM_MODE=tensor NSROM_POD_TOL=1e-8 \
  NSROM_RECOMPUTE_CLUSTERING=1 NSROM_RECOMPUTE_POD=1 NSROM_RECOMPUTE_TENSOR=1 \
  NSROM_RUN_SWEEP=0 \
  mpiexec -n 1 python scripts/main_local.py
```

(Firedrake is imported, so use the `mpiexec -n 1` launcher — see
[`../env/environment.md`](../env/environment.md).)

## Full regeneration after changing training snapshots

```bash
make clean-derived          # removes local_rom/ caches and rendered PDFs ONLY
make runs                   # EXPENSIVE — reruns every experiment in run.sh
make paper                  # rebuild every manuscript artifact
```

`make clean-derived` deletes the `local_rom/` cache and the rendered PDFs. It
**never** touches `states/`, `snapshots/`, `logs/`, or `paper_data/` (the
irreplaceable data — see `CLAUDE.md`). Reconstructing `local_rom/` is not cheap:
it is rebuilt only by the POD/operator stage of the experiments, so a bare
`make clean-derived` commits you to `make runs` before the paper can be rebuilt.
Run it deliberately.
