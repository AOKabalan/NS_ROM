# Scientific and data safety

Scientific correctness takes priority over code cleanup.

Never silently change numerical methods or defaults.

Treat these as read-only unless explicitly instructed otherwise:

- states/
- states_snapshot_prepru/
- states_retired/
- snapshots/
- snapshots_sparse/
- logs/
- paper_data/
- render/out/

Be conservative with HDF5, NPZ and scientific CSV data.

Do not run full sweeps, diagram builds, training campaigns, `make runs`, or
`./run.sh <TAG>` without explicit permission.

Do not delete an apparently unused scientific file until its references,
provenance and role have been investigated.

Do not commit changes unless explicitly requested.
