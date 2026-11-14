"""WD-01-PB-V04A root close-out: finalize + COPY-only backup + verify.

  finalize  copy the arm-verification evidence into <root>/results/,
            write README_DATA_NEVER_DELETE.md, regenerate SHA256SUMS over
            MANIFEST.json + worlds/** + results/** and verify the root
            against it;
  backup    COPY (shutil.copy2; never a hard link) every SHA256SUMS entry
            + SHA256SUMS + README to <backups>/wd01/partB_v0data_v04a,
            asserting per file that source and copy are distinct inodes /
            nlink == 1 on the copy, then re-verify the backup against its
            SHA256SUMS.  NEVER deletes; aborts if the target is non-empty.

Usage:
  python finalize_backup_v04a.py finalize --root R --arm-verification J
  python finalize_backup_v04a.py backup   --root R --dst D
  python finalize_backup_v04a.py verify   --root R
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import wdpaths  # noqa: E402

DEFAULT_ROOT = str(Path(wdpaths.RESEARCH_DENSE_ROOT) / "output"
                   / "wd01_partB_v0data_v04a")
DEFAULT_DST = str(Path(wdpaths.BACKUPS_ROOT) / "wd01"
                  / "partB_v0data_v04a")


def sha256_file(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def listing(root: Path) -> list[str]:
    rels = ["MANIFEST.json", "README_DATA_NEVER_DELETE.md"]
    for sub in ("worlds", "results"):
        d = root / sub
        if d.is_dir():
            rels += [p.relative_to(root).as_posix()
                     for p in sorted(d.rglob("*")) if p.is_file()]
    return [r for r in rels if (root / r).is_file()]


def write_sums(root: Path) -> int:
    rels = listing(root)
    lines = [f"{sha256_file(root / r)}  {r}" for r in rels]
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n",
                                     encoding="utf-8")
    return len(lines)


def verify(root: Path, label: str = "root") -> tuple[int, int, int]:
    """(entries, mismatches, missing) of root vs its SHA256SUMS."""
    n = bad = miss = 0
    for line in (root / "SHA256SUMS").read_text(encoding="utf-8") \
            .splitlines():
        sha, rel = line.split("  ", 1)
        p = root / rel
        n += 1
        if not p.is_file():
            miss += 1
            print(f"MISSING: {rel}")
        elif sha256_file(p) != sha:
            bad += 1
            print(f"MISMATCH: {rel}")
    print(f"{label} SHA256SUMS check: {n} entries, {bad} mismatches, "
          f"{miss} missing")
    return n, bad, miss


def cmd_finalize(args) -> int:
    root = Path(args.root)
    res = root / "results"
    res.mkdir(exist_ok=True)
    if args.arm_verification:
        shutil.copy2(args.arm_verification, res / "arm_verification.json")
    man = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    sha = sha256_file(root / "MANIFEST.json")
    readme = [
        "# DATA - NEVER DELETE: WD-01 Part B v0.4a value dataset",
        "",
        "Registration: output/wd01/REGISTRATION_PARTB_V04A.md",
        "(LF sha256 38a6fa406ca9d598fdaae986263f4fa81ee15dfa8c632414cc5f28dbe5bdee00).",
        "",
        "- Built by COPY/derivation from the raw runs (no hard links).",
        "- Sources: chunk-4 base / native_base / native / pi / fixed1-2 /",
        "  switching1-3; chunk-2/3 base + J3 true native/pi; auxiliary",
        "  (training only, `aux=true`) per-CAV switching arms of chunks",
        "  1-3 + fix27 and the chunk-1 bases.",
        "- HOLDOUT worlds were never enumerated or opened.",
        "- Train: python -m partB_v0 train --value-model gbm --gbm-variant auto",
        "  --features deployable_strict|deployable_detector --data <root>",
        "- Do not delete, move or rewrite anything here.",
        "",
        f"MANIFEST.json sha256 {sha}",
        f"sources: {len(man['worlds'])}, refused: "
        f"{man['v04a']['n_refused']}",
        "Backup: E:/research-dense-backups/wd01/partB_v0data_v04a",
        ""]
    (root / "README_DATA_NEVER_DELETE.md").write_text("\n".join(readme),
                                                      encoding="utf-8")
    n = write_sums(root)
    print(f"finalize: SHA256SUMS regenerated ({n} entries)")
    n2, bad, miss = verify(root)
    print("ROOT_VERIFY: " + ("PASS" if bad == 0 and miss == 0
                             and n2 == n else "FAIL"))
    return 0 if bad == 0 and miss == 0 else 1


def cmd_backup(args) -> int:
    root, dst = Path(args.root), Path(args.dst)
    if dst.exists() and any(dst.iterdir()):
        print(f"ABORT: {dst} exists and is non-empty; nothing deleted")
        return 1
    n0, bad0, miss0 = verify(root, "source")
    if bad0 or miss0:
        print("ABORT: source root does not verify")
        return 1
    rels = [line.split("  ", 1)[1] for line in
            (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines()]
    rels.append("SHA256SUMS")
    t0 = time.time()
    n_hl = 0
    for i, rel in enumerate(rels, 1):
        src, out = root / rel, dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out)
        s_src, s_out = os.stat(src), os.stat(out)
        if s_out.st_nlink != 1 or (s_src.st_ino and
                                   s_src.st_ino == s_out.st_ino
                                   and s_src.st_dev == s_out.st_dev):
            n_hl += 1
            print(f"HARDLINK DETECTED: {rel}")
        if i % 300 == 0:
            print(f"  copied {i}/{len(rels)}", flush=True)
    print(f"copied {len(rels)} files to {dst} in {time.time() - t0:.1f}s; "
          f"hardlinks detected: {n_hl}")
    n, bad, miss = verify(dst, "BACKUP")
    ok = (n_hl == 0 and bad == 0 and miss == 0 and n == n0)
    print("BACKUP_VERIFY: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def cmd_verify(args) -> int:
    n, bad, miss = verify(Path(args.root))
    print("ROOT_VERIFY: " + ("PASS" if bad == 0 and miss == 0 else "FAIL"))
    return 0 if bad == 0 and miss == 0 else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("finalize")
    f.add_argument("--root", default=DEFAULT_ROOT)
    f.add_argument("--arm-verification", default=None)
    b = sub.add_parser("backup")
    b.add_argument("--root", default=DEFAULT_ROOT)
    b.add_argument("--dst", default=DEFAULT_DST)
    v = sub.add_parser("verify")
    v.add_argument("--root", default=DEFAULT_ROOT)
    args = ap.parse_args(argv)
    return {"finalize": cmd_finalize, "backup": cmd_backup,
            "verify": cmd_verify}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
