#!/usr/bin/env python3
"""Verify external scientific data described by data_manifest.json."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath


class ManifestError(ValueError):
    """Raised when the manifest is unsafe or structurally invalid."""


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def safe_relative_path(raw: object) -> str:
    if not isinstance(raw, str) or not raw:
        raise ManifestError("entry path must be a non-empty string")
    posix = PurePosixPath(raw)
    if posix.is_absolute() or PureWindowsPath(raw).is_absolute():
        raise ManifestError(f"absolute manifest path rejected: {raw!r}")
    if ".." in posix.parts:
        raise ManifestError(f"escaping manifest path rejected: {raw!r}")
    return raw


def load_manifest(root: Path) -> dict:
    path = root / "data_manifest.json"
    try:
        with path.open(encoding="utf-8") as stream:
            manifest = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot load {path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ManifestError("manifest root must be an object")
    if manifest.get("archive_extraction_root") != ".":
        raise ManifestError("archive_extraction_root must be '.'")
    if not isinstance(manifest.get("bundles"), dict):
        raise ManifestError("manifest has no bundles object")
    return manifest


def collect_entries(manifest: dict, bundle_name: str) -> list[dict]:
    bundles = manifest["bundles"]
    ordered: list[dict] = []
    seen_bundles: set[str] = set()
    active: set[str] = set()

    def visit(name: str) -> None:
        if name in active:
            raise ManifestError(f"bundle inheritance cycle at {name!r}")
        if name in seen_bundles:
            return
        bundle = bundles.get(name)
        if not isinstance(bundle, dict):
            raise ManifestError(f"unknown bundle: {name!r}")
        active.add(name)
        for parent in bundle.get("inherits", []):
            if not isinstance(parent, str):
                raise ManifestError(f"invalid parent bundle in {name!r}")
            visit(parent)
        entries = bundle.get("entries", [])
        if not isinstance(entries, list):
            raise ManifestError(f"entries for {name!r} must be a list")
        ordered.extend(entries)
        active.remove(name)
        seen_bundles.add(name)

    visit(bundle_name)
    seen_paths: set[str] = set()
    for entry in ordered:
        if not isinstance(entry, dict):
            raise ManifestError("bundle entries must be objects")
        path = safe_relative_path(entry.get("path"))
        if path in seen_paths:
            raise ManifestError(f"duplicate inherited entry path: {path!r}")
        seen_paths.add(path)
        if entry.get("path_type") not in {"file", "directory", "glob"}:
            raise ManifestError(f"invalid path_type for {path!r}")
        tracked = entry.get("tracked")
        external = entry.get("external")
        if not isinstance(tracked, bool) or not isinstance(external, bool):
            raise ManifestError(f"tracked/external must be booleans: {path!r}")
        if tracked == external:
            raise ManifestError(
                f"entry must be exactly one of tracked/external: {path!r}"
            )
    return ordered


def ensure_within_root(root: Path, path: Path, label: str) -> None:
    resolved = path.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ManifestError(f"resolved path escapes root: {label!r}") from exc


def matching_files(root: Path, entry: dict) -> tuple[str, list[Path]]:
    raw = safe_relative_path(entry["path"])
    kind = entry["path_type"]
    if kind == "glob":
        matches = [Path(item) for item in glob.glob(str(root / raw))]
        for match in matches:
            ensure_within_root(root, match, raw)
        files = sorted(path for path in matches if path.is_file())
        return ("missing" if not matches else "present"), files

    target = root / raw
    ensure_within_root(root, target, raw)
    if not target.exists():
        return "missing", []
    if kind == "file":
        if not target.is_file():
            return "missing", []
        return "present", [target]
    if not target.is_dir():
        return "missing", []
    files = sorted(path for path in target.rglob("*") if path.is_file())
    for path in files:
        ensure_within_root(root, path, raw)
    return "present", files


def verify(root: Path, manifest: dict, bundle_name: str) -> int:
    entries = collect_entries(manifest, bundle_name)
    counters = {
        "present": 0,
        "missing": 0,
        "empty": 0,
        "mismatch": 0,
        "tracked": 0,
    }
    print(f"bundle={bundle_name} root={root}")
    for entry in entries:
        raw = entry["path"]
        if entry.get("tracked"):
            counters["tracked"] += 1
            print(f"SKIP tracked  {raw}")
            continue
        if not entry.get("required", True):
            continue
        status, files = matching_files(root, entry)
        if status == "missing":
            counters["missing"] += 1
            print(f"MISSING       {raw}")
            continue
        sizes = [path.stat().st_size for path in files]
        count = len(files)
        size = sum(sizes)
        empty_files = sizes.count(0)
        if count == 0 or empty_files:
            counters["empty"] += 1
            print(
                f"EMPTY         {raw} files={count} bytes={size} "
                f"empty_files={empty_files}"
            )
            continue
        expected_count = entry.get("file_count")
        expected_size = entry.get("size_bytes")
        if ((expected_count is not None and count != expected_count)
                or (expected_size is not None and size != expected_size)):
            counters["mismatch"] += 1
            print(
                f"MISMATCH      {raw} files={count}/{expected_count} "
                f"bytes={size}/{expected_size}"
            )
            continue
        counters["present"] += 1
        print(f"PRESENT       {raw} files={count} bytes={size}")

    print(
        "summary "
        + " ".join(f"{key}={value}" for key, value in counters.items())
    )
    return 0 if not any(counters[key] for key in ("missing", "empty", "mismatch")) else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, help="bundle name to verify")
    parser.add_argument(
        "--root",
        type=Path,
        default=repository_root(),
        help="repository root containing data_manifest.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    try:
        manifest = load_manifest(root)
        return verify(root, manifest, args.bundle)
    except ManifestError as exc:
        print(f"manifest error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
