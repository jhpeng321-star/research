#!/usr/bin/env python
"""Unpack a C4 bundle (made by pack_bundle.py) into a repo root on the box.

    python unpack_bundle.py --bundle /workspace/c4/bundle --root /workspace/c4/rd

1. verifies every tar against SHA256SUMS (skip with --no-verify-tars);
2. extracts members (repo-relative names; absolute / '..' names are refused);
3. for every pack group, recreates the full-size sparse .npy arrays from C4_RANGES.json and
   fills the selected rows, checking the sha256 of every stored row block;
4. removes the temporary __ranges__ parts (keep with --keep-parts).

numpy-free on purpose except for a final shape/dtype sanity check (skipped if numpy is absent).
Never overwrites an existing different file silently: an existing target must be byte-identical
(sha256) or the run stops.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
from pathlib import Path

CHUNK = 1 << 24
RANGES_NAME = "C4_RANGES.json"
PART_DIR = "__ranges__"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def read_sums(bundle: Path) -> dict[str, str]:
    sums = {}
    for line in (bundle / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, _, name = line.partition("  ")
            sums[name.strip()] = digest.strip()
    return sums


def safe_members(tf: tarfile.TarFile):
    for m in tf:
        name = m.name.replace("\\", "/")
        if name.startswith("/") or ".." in name.split("/") or (len(name) > 1 and name[1] == ":"):
            raise SystemExit(f"refusing unsafe tar member {m.name!r}")
        if not (m.isfile() or m.isdir()):
            raise SystemExit(f"refusing non-regular tar member {m.name!r}")
        yield m


def extract(tar_path: Path, root: Path) -> int:
    n = 0
    with tarfile.open(tar_path, "r") as tf:
        for m in safe_members(tf):
            target = root / m.name
            if m.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            src = tf.extractfile(m)
            if target.exists() and target.is_file() and not (PART_DIR in target.parts):
                if target.stat().st_size == m.size:
                    h = hashlib.sha256()
                    while True:
                        b = src.read(CHUNK)
                        if not b:
                            break
                        h.update(b)
                    if h.hexdigest() == sha256_file(target):
                        n += 1
                        continue  # identical, nothing to do
                raise SystemExit(f"{target} exists and differs; refuse to overwrite")
            with open(target, "wb") as out:
                shutil.copyfileobj(src, out, CHUNK)
            n += 1
    return n


def rebuild_pack(ranges_json: Path, root: Path, keep_parts: bool) -> dict:
    doc = json.loads(ranges_json.read_text(encoding="utf-8"))
    pack_dir = ranges_json.parent
    summary = {"pack": doc["pack"], "arrays": 0, "bytes_filled": 0}
    for name, spec in sorted(doc["arrays"].items()):
        target = pack_dir / name
        part = root / spec["part"]
        header = bytes.fromhex(spec["header_hex"])
        if target.exists():
            raise SystemExit(f"{target} already exists; refusing to rebuild over it")
        h = hashlib.sha256()
        with open(part, "rb") as src, open(target, "wb") as out:
            out.write(header)
            out.truncate(spec["total_bytes"])          # sparse hole where no rows are stored
            rb = spec["row_bytes"]
            for a, b in spec["ranges"]:
                out.seek(spec["offset"] + a * rb)
                left = (b - a) * rb
                while left:
                    buf = src.read(min(CHUNK, left))
                    if not buf:
                        raise SystemExit(f"{part}: short part file")
                    h.update(buf)
                    out.write(buf)
                    left -= len(buf)
            summary["bytes_filled"] += spec["part_bytes"]
        if h.hexdigest() != spec["part_sha256"]:
            raise SystemExit(f"{part}: row-block sha256 mismatch")
        if target.stat().st_size != spec["total_bytes"]:
            raise SystemExit(f"{target}: size {target.stat().st_size} != {spec['total_bytes']}")
        try:
            import numpy as np
            arr = np.load(target, mmap_mode="r", allow_pickle=False)
            if list(arr.shape) != spec["shape"] or str(arr.dtype) != spec["dtype"]:
                raise SystemExit(f"{target}: shape/dtype differ from the source array")
            del arr
        except ImportError:
            pass
        summary["arrays"] += 1
        if not keep_parts:
            part.unlink()
    if not keep_parts:
        parts = pack_dir / PART_DIR
        if parts.is_dir() and not any(parts.iterdir()):
            parts.rmdir()
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True, help="repo root to extract into")
    ap.add_argument("--only", default=None, help="comma list of tar names (without .tar)")
    ap.add_argument("--no-verify-tars", action="store_true")
    ap.add_argument("--keep-parts", action="store_true")
    args = ap.parse_args(argv)

    sums = read_sums(args.bundle)
    manifest = json.loads((args.bundle / "BUNDLE_MANIFEST.json").read_text(encoding="utf-8"))
    if sums.get("BUNDLE_MANIFEST.json") != sha256_file(args.bundle / "BUNDLE_MANIFEST.json"):
        raise SystemExit("BUNDLE_MANIFEST.json does not match SHA256SUMS")
    tars = sorted(p for p in args.bundle.glob("*.tar"))
    if args.only:
        wanted = set(args.only.split(","))
        tars = [t for t in tars if t.stem in wanted]
    args.root.mkdir(parents=True, exist_ok=True)
    for t in tars:
        if not args.no_verify_tars:
            got = sha256_file(t)
            if got != sums.get(t.name) or got != manifest["tars"][t.name]["sha256"]:
                raise SystemExit(f"{t.name}: sha256 mismatch (transfer corrupted?) got {got}")
        n = extract(t, args.root)
        print(f"extracted {t.name}: {n} members", flush=True)
    for rj in sorted(args.root.glob(f"**/{RANGES_NAME}")):
        s = rebuild_pack(rj, args.root, args.keep_parts)
        print(f"rebuilt {s['pack']}: {s['arrays']} arrays, {s['bytes_filled'] / 2**30:.2f} GiB filled", flush=True)
    # member-level verification of everything that is not a row block
    bad = 0
    for t in tars:
        for row in manifest["tars"][t.name]["members"]:
            if row.get("kind") == "ranges":
                continue
            p = args.root / row["name"]
            if not p.is_file() or p.stat().st_size != row["size"]:
                print(f"MISSING/SIZE {row['name']}", file=sys.stderr)
                bad += 1
            elif row["size"] < 64 * 2**20 and sha256_file(p) != row["sha256"]:
                print(f"SHA MISMATCH {row['name']}", file=sys.stderr)
                bad += 1
    if bad:
        raise SystemExit(f"{bad} member verification failure(s)")
    print("unpack OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
