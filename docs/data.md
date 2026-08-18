# External scientific data

Large scientific inputs and computed state stores are intentionally kept out of
Git. They include Firedrake checkpoints, production snapshot arrays, selected
ROM/POD products, and stored online states that are expensive or impossible to
reconstruct exactly from source alone.

`data_manifest.json` is the machine-readable inventory. Its entry schema is:

- `path`: repository-relative file path or glob.
- `path_type`: `file`, `directory`, or `glob`.
- `required_by`: scripts or workflows that consume the entry.
- `reason`: why the entry belongs in the bundle.
- `tracked` / `external`: exactly one is true. Tracked entries document Git
  inputs but must not be copied into the external archive.
- `exists_locally`, `file_count`, `size_bytes`: the audited local inventory.

## Bundles

The `figures` bundle is the smallest reviewed set that supports the active
scripts in both `section_6_figures/` and `paper_figures/` from existing results,
without launching FOM/ROM solves. It includes scalar state indices, the selected
stored point fields used by the paper pressure-error figure, reviewed error
tables, POD/tensor products read directly by renderers, and production
snapshot/lifting inputs used by the active POD/clustering figure scripts.
Its audited external inventory is 3,417 files and 2,574,060,445 bytes, plus
three small inputs already tracked by Git.

The `full` bundle inherits `figures` and adds the source-like checkpoints needed
by the supported local-ROM, snapshot-generation, lifting, and preserved FOM
workflows. Rebuildable ROM caches and generated online runs are not added merely
because they are large or expensive; direct consumers are recorded explicitly
in the manifest. It adds 5 files and 25,201,456 bytes, for an inherited external
total of 3,422 files and 2,599,261,901 bytes.

## Verification

From the repository root:

```bash
python scripts/verify_external_data.py --bundle figures
python scripts/verify_external_data.py --bundle full
```

To verify an extracted checkout elsewhere:

```bash
python scripts/verify_external_data.py --bundle figures --root /path/to/NS_ROM
```

The future archive must be extracted at the repository root (`.`), preserving
the relative paths in the manifest. The verifier is read-only, skips entries
already tracked by Git, rejects absolute or escaping paths, and reports missing,
empty, or inventory-mismatched external requirements.

Ali will review and upload the archive to OneDrive manually. A direct download
link, archive filename, archive size, and checksum will be documented only after
that upload; none is invented here.

The machine-local Phase 1 trash directory under `/home/ali/` is recovery
material. It must remain untouched until the refactor is complete and must never
be included in either bundle or the future archive.

Logs, debug investigations, retired archives, bytecode, tool caches, generated
PDF/PNG/SVG/TEX outputs, credentials, machine configuration, and unreferenced
results are excluded. See the manifest's `excluded` records for large directory
sizes and the scientific reason for each exclusion.
