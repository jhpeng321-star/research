#!/usr/bin/env python3
"""Fetch registered datasets into a verified, disposable local cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = Path(__file__).with_name("data_registry.json")
DEFAULT_CACHE = ROOT / "data_cache"
DEFAULT_BUDGET = 20 * 1024**3
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
MD5_RE = re.compile(r"^[a-fA-F0-9]{32}$")


class DataError(Exception):
    """A validation or data operation failed safely."""


def _load_registry(path: Path) -> dict[str, dict[str, Any]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataError(f"cannot read registry {path}: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise DataError("registry must be an object with schema_version 1")
    datasets = raw.get("datasets")
    if not isinstance(datasets, dict):
        raise DataError("registry datasets must be an object")
    for name, entry in datasets.items():
        if not isinstance(name, str) or not NAME_RE.fullmatch(name):
            raise DataError(f"invalid dataset name: {name!r}")
        if not isinstance(entry, dict):
            raise DataError(f"{name}: entry must be an object")
        drive_path = entry.get("drive_path")
        size_bytes = entry.get("size_bytes")
        local_path = entry.get("local_path")
        manifest = entry.get("md5_manifest")
        if not isinstance(drive_path, str) or not re.fullmatch(
            r"gdrive:research-archive/\S+", drive_path
        ):
            raise DataError(f"{name}: drive_path must be under gdrive:research-archive/")
        if not isinstance(size_bytes, int) or isinstance(size_bytes, bool) or size_bytes < 0:
            raise DataError(f"{name}: size_bytes must be a non-negative integer")
        if not isinstance(local_path, str) or not local_path.startswith("data_cache/"):
            raise DataError(f"{name}: local_path must be below data_cache/")
        local_parts = PurePosixPath(local_path).parts
        if (
            "\\" in local_path
            or any(not re.fullmatch(r"[A-Za-z0-9._-]+", part) for part in local_parts)
            or any(part in (".", "..") for part in local_parts)
        ):
            raise DataError(f"{name}: invalid local cache path")
        if not isinstance(manifest, dict) or not manifest:
            raise DataError(f"{name}: md5_manifest must be a non-empty object")
        for rel, digest in manifest.items():
            relpath = PurePosixPath(rel) if isinstance(rel, str) else None
            if (
                relpath is None
                or relpath.is_absolute()
                or not relpath.parts
                or any(part in ("", ".", "..") for part in relpath.parts)
                or "\\" in rel
                or not isinstance(digest, str)
                or not MD5_RE.fullmatch(digest)
            ):
                raise DataError(f"{name}: invalid manifest path or MD5: {rel!r}")
        if not manifest:
            raise DataError(f"{name}: empty MD5 manifest")
        entry["md5_manifest"] = {str(k): v.lower() for k, v in manifest.items()}
    return datasets


def _cache_target(cache_root: Path, local_path: str) -> Path:
    relative = PurePosixPath(local_path)
    if (
        relative.is_absolute()
        or relative.parts[0] != "data_cache"
        or len(relative.parts) < 2
        or any(part in ("", ".", "..") for part in relative.parts)
        or "\\" in local_path
    ):
        raise DataError(f"unsafe local cache path: {local_path!r}")
    target = cache_root.joinpath(*relative.parts[1:])
    if not target.resolve(strict=False).is_relative_to(cache_root.resolve(strict=False)):
        raise DataError(f"cache path escapes data_cache: {local_path!r}")
    return target


def _ensure_cache(cache_root: Path) -> None:
    if cache_root.exists() and cache_root.is_symlink():
        raise DataError(f"refusing symlink cache root: {cache_root}")
    cache_root.mkdir(parents=True, exist_ok=True)


def _inspect_tree(root: Path, manifest: dict[str, str]) -> tuple[bool, str, int]:
    if not root.is_dir() or root.is_symlink():
        return False, "missing", 0
    actual: set[str] = set()
    total = 0
    for path in root.rglob("*"):
        if path.is_symlink():
            return False, f"symlink refused: {path.relative_to(root)}", total
        if path.is_file():
            rel = path.relative_to(root).as_posix()
            actual.add(rel)
            total += path.stat().st_size
    expected = set(manifest)
    if actual != expected:
        extra = sorted(actual - expected)
        missing = sorted(expected - actual)
        return False, f"member mismatch (missing={missing}, extra={extra})", total
    for rel, expected_md5 in manifest.items():
        digest = hashlib.md5(usedforsecurity=False)
        with (root / Path(*PurePosixPath(rel).parts)).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected_md5:
            return False, f"MD5 mismatch: {rel}", total
    return True, f"verified {len(actual)} files", total


def _entry_target(cache_root: Path, entry: dict[str, Any]) -> Path:
    if cache_root.exists() and cache_root.is_symlink():
        raise DataError(f"refusing symlink cache root: {cache_root}")
    target = _cache_target(cache_root, entry["local_path"])
    # Do not permit an existing symlink anywhere in the target's parent chain.
    current = target
    while current != cache_root and cache_root in current.parents:
        if current.is_symlink():
            raise DataError(f"refusing symlink in cache path: {current}")
        current = current.parent
    return target


def _used_bytes(cache_root: Path, exclude: Path | None = None) -> int:
    used = 0
    for path in cache_root.iterdir():
        if exclude is not None and path == exclude:
            continue
        if path.is_symlink():
            raise DataError(f"refusing symlink in cache: {path}")
        if path.is_file():
            used += path.stat().st_size
        elif path.is_dir():
            for child in path.rglob("*"):
                if child.is_symlink():
                    raise DataError(f"refusing symlink in cache: {child}")
                if child.is_file():
                    used += child.stat().st_size
    return used


def _fetch(name: str, entry: dict[str, Any], cache_root: Path, budget: int, rclone: str) -> str:
    _ensure_cache(cache_root)
    target = _entry_target(cache_root, entry)
    valid, detail, _ = _inspect_tree(target, entry["md5_manifest"])
    if valid:
        return f"{name}: already present ({detail})"
    if target.exists():
        raise DataError(f"{name}: destination exists but is not verified ({detail}); preserving it")
    used = _used_bytes(cache_root)
    if used + entry["size_bytes"] > budget:
        raise DataError(
            f"{name}: cache budget exceeded ({used} + {entry['size_bytes']} > {budget} bytes)"
        )
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".fetch-", dir=parent))
    try:
        result = subprocess.run(
            [rclone, "copy", entry["drive_path"], str(staging)],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
            raise DataError(f"rclone copy failed for {name}: {detail}")
        valid, detail, total = _inspect_tree(staging, entry["md5_manifest"])
        if not valid:
            raise DataError(f"{name}: downloaded data failed verification: {detail}")
        if total != entry["size_bytes"]:
            raise DataError(
                f"{name}: byte total mismatch ({total} != {entry['size_bytes']})"
            )
        # Same-parent rename is atomic on the cache volume; destination must not exist.
        if target.exists():
            raise DataError(f"{name}: destination appeared during fetch; preserving both")
        staging.rename(target)
        return f"{name}: fetched and verified ({detail}, {total} bytes)"
    finally:
        # The only recursively removed path is this freshly created direct child of data_cache.
        if staging.parent == parent and cache_root in staging.parents:
            if staging.is_symlink():
                staging.unlink()
            elif staging.exists():
                shutil.rmtree(staging)


def _format_status(name: str, entry: dict[str, Any], cache_root: Path) -> str:
    target = _entry_target(cache_root, entry)
    valid, detail, total = _inspect_tree(target, entry["md5_manifest"])
    expected_size = entry["size_bytes"]
    if valid and total != expected_size:
        valid, detail = False, f"size mismatch ({total} != {expected_size})"
    return f"{name}: {'verified' if valid else 'not verified'} — {detail}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--rclone", default=os.environ.get("RCLONE", "rclone"))
    parser.add_argument("--budget-bytes", type=int, default=DEFAULT_BUDGET)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    for command in ("status", "verify"):
        cmd = sub.add_parser(command)
        cmd.add_argument("name", nargs="?")
    get = sub.add_parser("get")
    get.add_argument("name")
    args = parser.parse_args(argv)
    try:
        datasets = _load_registry(args.registry)
        if args.budget_bytes < 0:
            raise DataError("budget-bytes must be non-negative")
        if args.command == "list":
            for name in sorted(datasets):
                entry = datasets[name]
                print(f"{name}\t{entry['size_bytes']}\t{entry['drive_path']}\t{entry['local_path']}")
            return 0
        if args.command == "get":
            if args.name not in datasets:
                raise DataError(f"unknown dataset: {args.name}")
            print(_fetch(args.name, datasets[args.name], args.cache_dir, args.budget_bytes, args.rclone))
            return 0
        names = sorted(datasets) if args.name is None else [args.name]
        unknown = [name for name in names if name not in datasets]
        if unknown:
            raise DataError(f"unknown dataset: {unknown[0]}")
        if args.command == "status":
            for name in names:
                print(_format_status(name, datasets[name], args.cache_dir))
            return 0
        failed = False
        for name in names:
            entry = datasets[name]
            target = _entry_target(args.cache_dir, entry)
            valid, detail, total = _inspect_tree(target, entry["md5_manifest"])
            valid = valid and total == entry["size_bytes"]
            print(f"{name}: {'OK' if valid else 'FAIL'} — {detail}")
            failed |= not valid
        return 1 if failed else 0
    except (DataError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
