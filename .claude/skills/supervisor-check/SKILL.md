---
name: supervisor-check
description: Perform a final supervisor-readiness review of NS_ROM after refactoring and reproducibility work is complete.
disable-model-invocation: true
context: fork
agent: Explore
---

# Supervisor readiness review

Pretend you are the supervisor receiving NS_ROM for the first time.

Do not use undocumented knowledge from prior development.

Evaluate whether the repository itself explains:

1. project purpose,
2. software/environment requirements,
3. repository structure,
4. included versus generated data,
5. primary scientific workflow,
6. scientific configuration,
7. expensive versus cheap stages,
8. how to regenerate paper figures/tables,
9. how to rebuild ROMs from snapshots,
10. how to rerun downstream work after changing training snapshots,
11. how to run tests.

Inspect the documented commands and verify them where reasonably cheap.

Identify any place where a new researcher would have to guess.

Return one verdict:

READY
READY WITH MINOR CAVEATS
NOT READY

Then list the specific reasons and remaining actions.
