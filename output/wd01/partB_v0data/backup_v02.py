"""Backup a V0 dataset root (MANIFEST.json + worlds/) to the backup
area, write a SHA256SUMS manifest over every copied file, and re-verify
the copy against it (paste-able stdout).  NEVER deletes: if the target
exists and is non-empty, abort.

Usage:  python backup_v02.py [--src DATASET_ROOT]
                             [--dst BACKUPS_ROOT/wd01/partB_v0data_v02]
(default src: this dir = the v0.2 root; default BACKUPS_ROOT:
E:/research-dense-backups)
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import wdpaths  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_DST = str(Path(wdpaths.BACKUPS_ROOT) / "wd01" / "partB_v0data_v02")


def sha256_file(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(HERE),
                    help="dataset root to back up (default: this dir)")
    ap.add_argument("--dst", default=DEFAULT_DST)
    args = ap.parse_args(argv)
    src_root, dst_root = Path(args.src), Path(args.dst)
    if dst_root.exists() and any(dst_root.iterdir()):
        print(f"ABORT: {dst_root} exists and is non-empty; nothing deleted")
        return 1

    files = ["MANIFEST.json"] + \
        [p.relative_to(src_root).as_posix()
         for p in sorted((src_root / "worlds").rglob("*")) if p.is_file()]
    t0 = time.time()
    n = 0
    sums = []
    for rel in files:
        src = src_root / rel
        dst = dst_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        sha_src, sha_dst = sha256_file(src), sha256_file(dst)
        if sha_src != sha_dst:
            print(f"COPY MISMATCH: {rel}")
            return 1
        sums.append(f"{sha_dst}  {rel}")
        n += 1
        if n % 200 == 0:
            print(f"  copied {n}/{len(files)}", flush=True)
    (dst_root / "SHA256SUMS").write_text("\n".join(sums) + "\n",
                                         encoding="utf-8")
    print(f"copied {n} files + SHA256SUMS ({len(sums)} entries) to "
          f"{dst_root} in {time.time() - t0:.1f}s")

    # independent verification pass over the backup
    n_ok = n_bad = 0
    for line in (dst_root / "SHA256SUMS").read_text(encoding="utf-8") \
            .splitlines():
        sha, rel = line.split("  ", 1)
        p = dst_root / rel
        if p.is_file() and sha256_file(p) == sha:
            n_ok += 1
        else:
            n_bad += 1
            print(f"VERIFY MISMATCH: {rel}")
    print(f"backup verification: {n_ok} OK, {n_bad} mismatches "
          f"(expect 0)")
    print("BACKUP_VERIFY: " + ("PASS" if n_bad == 0 and n_ok == n else
                               "FAIL"))
    return 0 if n_bad == 0 and n_ok == n else 1


if __name__ == "__main__":
    sys.exit(main())
