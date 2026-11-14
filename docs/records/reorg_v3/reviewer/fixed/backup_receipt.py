"""Metadata-only preflight and durable backup receipts; no deletion commands.

Hashes are rclone-generated text metadata. Arrays/runs are never loaded here.
This helper is part of the unexecuted offload copy, not a review-time test.
"""
import datetime
import json
import os
from pathlib import Path
import re
import stat
import sys


def append(path, event, **fields):
    record = dict(event=event, ts=datetime.datetime.now(datetime.timezone.utc).isoformat(), **fields)
    with Path(path).open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, ensure_ascii=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def check_entry(path):
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
        raise ValueError(f"link/reparse point refused: {path}")
    if re.search(r"sealed|holdout|chunk-?5|j4b|inc1|(^|[\\/])cal([\\/]|$)", str(path), re.I):
        raise ValueError("protected path refused")
    if re.search(r"rclone\.conf|credentials?|id_rsa|id_ed25519|\.pem$|\.key$|\.env$", path.name, re.I):
        raise ValueError("credential-like filename refused")


def preflight(source):
    root = Path(source)
    if not root.is_absolute() or not root.is_dir():
        raise ValueError("requires an existing absolute source directory")
    for parent in reversed([root, *root.parents]):
        check_entry(parent)
    pending = [root]
    while pending:
        with os.scandir(pending.pop()) as entries:
            for entry in entries:
                path = Path(entry.path)
                check_entry(path)
                if entry.is_dir(follow_symlinks=False):
                    pending.append(path)


def hashes(path):
    result = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            match = re.fullmatch(r"([0-9a-fA-F]{32})  (.+)\n?", line)
            if not match:
                raise ValueError("missing/invalid MD5 in checksum listing")
            digest, name = match.groups()
            if name in result:
                raise ValueError("duplicate checksum member")
            result[name] = digest.lower()
    if not result:
        raise ValueError("empty checksum listing cannot authorize backup verification")
    return result


def main(args):
    mode = args[0]
    if mode == "preflight":
        preflight(args[1])
    elif mode == "init":
        anchor, destination = map(Path, args[1:])
        if destination.parent.parent != anchor or not anchor.is_dir():
            raise ValueError("unexpected receipt directory")
        check_entry(anchor)
        bootstrap = anchor / (destination.name + ".bootstrap.jsonl")
        for directory in (destination.parent, destination):
            if directory.exists():
                check_entry(directory)
            else:
                append(bootstrap, "before_directory_create", path=str(directory))
                directory.mkdir()
                append(bootstrap, "after_directory_create", path=str(directory))
    elif mode == "record":
        append(args[1], args[2], local=args[3], remote=args[4], deletion_authorized=False)
    elif mode == "compare":
        before, remote, after = map(hashes, args[1:])
        if before != after or any(remote.get(name) != digest for name, digest in before.items()):
            raise ValueError("source changed or remote checksum member missing/mismatched")
        print(f"one-way MD5 comparison passed: {len(before)} source members")
    else:
        raise ValueError("unknown mode")


if __name__ == "__main__":
    main(sys.argv[1:])
