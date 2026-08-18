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

## Published archive

The reviewed `full` bundle from manifest/source commit `636bbb2` is published
anonymously through OneDrive/SharePoint:

- Archive share link:
  <https://sissa-my.sharepoint.com/:u:/g/personal/akabalan_sissa_it/IQD0pSOikm31RrCEIbpSqAsNAfRVn2rytm58XSpdNqhTK1k?e=qlMYap>
- Checksum share link:
  <https://sissa-my.sharepoint.com/:u:/g/personal/akabalan_sissa_it/IQC59clv_MMsTK-41AR6F5eIAeiPL_GnfDTCr-Vyr26HPdk?e=UCrIcv>

Archive details:

- Filename: `NS_ROM_external_data_full_636bbb2.tar.zst`
- Format: GNU tar with Zstandard compression
- Compressed size: 2,560,264,740 bytes
- Uncompressed source data: 2,599,261,901 bytes
- Members: 3,422
- SHA-256:
  `f68d0f97472ca7fee286185edce64b91450f364264aaabeaccf679e33edffdaa`
- Checksum filename: `NS_ROM_external_data_full_636bbb2.tar.zst.sha256`

The following stable `download=1` forms were tested without authentication,
cookies, or an existing SharePoint session. Download both files into the same
directory:

```bash
curl --disable --no-netrc --cookie '' --location --fail \
  --output NS_ROM_external_data_full_636bbb2.tar.zst \
  'https://sissa-my.sharepoint.com/:u:/g/personal/akabalan_sissa_it/IQD0pSOikm31RrCEIbpSqAsNAfRVn2rytm58XSpdNqhTK1k?e=qlMYap&download=1'

curl --disable --no-netrc --cookie '' --location --fail \
  --output NS_ROM_external_data_full_636bbb2.tar.zst.sha256 \
  'https://sissa-my.sharepoint.com/:u:/g/personal/akabalan_sissa_it/IQC59clv_MMsTK-41AR6F5eIAeiPL_GnfDTCr-Vyr26HPdk?e=UCrIcv&download=1'
```

Verify the downloaded archive before extraction:

```bash
sha256sum -c NS_ROM_external_data_full_636bbb2.tar.zst.sha256
```

**Extract only into a fresh NS_ROM checkout, or into a checkout that contains no
existing external data. Extraction can overwrite files at matching paths.** The
archive paths are relative to the repository root, so extract with GNU tar as
follows:

```bash
tar --extract --zstd \
  --file=/path/to/NS_ROM_external_data_full_636bbb2.tar.zst \
  --directory=/path/to/fresh/NS_ROM
```

Then verify the extracted data from the checkout root:

```bash
python scripts/verify_external_data.py --bundle full
```

The verifier is read-only, skips entries already tracked by Git, rejects
absolute or escaping manifest paths, and reports missing, empty, or
inventory-mismatched external requirements.

The machine-local Phase 1 trash directory under `/home/ali/` is recovery
material. It must remain untouched until the refactor is complete and must never
be included in either bundle or the published archive.

Logs, debug investigations, retired archives, bytecode, tool caches, generated
PDF/PNG/SVG/TEX outputs, credentials, machine configuration, and unreferenced
results are excluded. See the manifest's `excluded` records for large directory
sizes and the scientific reason for each exclusion.
