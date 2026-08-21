---
name: repo-audit
description: Audit NS_ROM architecture, unfinished refactoring, Git state, reproducibility gaps, and scientific workflow before repository-wide changes.
disable-model-invocation: true
context: fork
agent: Explore
---

# NS_ROM repository audit

Perform a read-only audit.

DO NOT edit, move, delete, generate, or commit files.

Inspect at minimum:

- Git status
- staged and unstaged diffs
- recent commits
- repository tree
- `nsrom/`
- `scripts/`
- figure/table generation code
- tests
- README/documentation
- package/environment configuration
- `.gitignore`

Search for legacy imports and incomplete migrations involving:

- `nsrom.local_rom`
- `cluster_building`
- `snapshot_collection`
- `scripts.sweep`
- `generate_snapshots`

Understand the actual workflow from:

FOM / snapshots
→ preprocessing
→ clustering
→ POD
→ reduced operators
→ ROM
→ validation
→ paper artifacts

Identify:

1. current architecture,
2. completed refactoring,
3. incomplete or inconsistent refactoring,
4. compatibility shims still in use,
5. duplicated implementations,
6. fragile imports or path assumptions,
7. duplicated/hardcoded configuration,
8. undocumented workflow dependencies,
9. reproducibility weaknesses,
10. obsolete-file candidates,
11. scientific risks associated with further cleanup.

The repository already appears to contain a reproducibility architecture based
on `run.sh`, `Makefile`, and `render/`.

Audit whether this architecture is complete and correct.

Prefer finishing or fixing it over replacing it with a new workflow system.
For every important finding, reference concrete files.

Return:

## Architecture

## Refactoring already completed

## Incomplete / suspicious refactoring

## Reproducibility gaps

## Paper-generation situation

## Obsolete or questionable files

Do not delete them.

## Scientific risks

## Minimal ordered finishing plan

Do not implement the plan.
