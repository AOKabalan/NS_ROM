---
name: reproduce
description: Test the NS_ROM reproducibility workflow locally using the cheapest meaningful path and existing scientific outputs.
disable-model-invocation: true
---

# Local reproducibility verification

Act as a researcher who just received this repository.

Use only the repository and its documentation.

Verify progressively.

## Level 1

- syntax
- imports
- package installation/import behavior
- CLI/help commands where applicable

## Level 2

Exercise lightweight components such as:

- loading existing snapshots,
- loading clustering data,
- loading POD bases,
- loading reduced operators,
- configuration parsing.

## Level 3

Regenerate representative paper figures/tables from existing computed results.

Do not recompute expensive FOM data merely to produce figures.

## Level 4

If practical, execute a deliberately small numerical smoke test exercising an
important FOM/ROM path.

Do not run the full training campaign.

## Regression

Where reference outputs exist, compare important numerical quantities such as:

- cluster sizes,
- POD dimensions,
- operator dimensions,
- ROM errors,
- bifurcation quantities,
- paper table values.

Use numerical tolerances where appropriate.

## Report

For every test report:

command
→ outcome
→ failure
→ fix, if applicable

Then give exact commands for:

1. smoke testing,
2. regenerating paper artifacts from existing data,
3. rebuilding ROM data from existing snapshots,
4. full downstream regeneration after changing snapshots.

Never claim a test succeeded unless it was actually executed.
