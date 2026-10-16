#!/usr/bin/env python
"""Pull logs / summaries / checkpoints of a C4 run back from a box, verifying sha256 (run on the laptop).

    python scripts/c4/fetch_results.py --host gpu-host --run wm_main_20261013T0900Z
    python scripts/c4/fetch_results.py --host gpu-host --run wm_main_... --include-large "out/wm_s_best.pt"
    python scripts/c4/fetch_results.py --host cpuA --run eval_... --max-small-mb 2 --list-only

* Remote run directory = $C4_HOME/runs/<run> (C4_HOME default /workspace/c4; --remote-home to change).
* SMALL files (<= --max-small-mb, default 5; no *.pt/*.pth/*.npy/*.bin/*.tar) go to
  runs/active/WM-S-C4-20261013/fetched/<host>/<run>/ (inside the repo's evidence area).
* LARGE files are fetched only when named with --include-large (glob, repeatable); they go to
  <repo>/tmp/c4_fetch/<host>/<run>/ (git-ignored, never under runs/active) and are sha256-verified.
* Every file is hashed on the box (sha256sum) and again locally; a mismatch aborts and leaves the file
  renamed *.BAD.  Names with a restricted token (holdout, heldout, cal, sealed, chunk5, j4b, inc1) are skipped.
* One attempt per call: no retry loops.  The script never starts, stops or powers anything on the box;
  it only runs ls/sha256sum/tar.  --local-root DIR substitutes a local directory for the remote run dir
  (used for testing the logic without any machine).
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import io
import re
import shlex
import subprocess
import sys
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EVIDENCE = REPO / "runs" / "active" / "WM-S-C4-20261013"
LARGE_SUFFIXES = (".pt", ".pth", ".npy", ".bin", ".tar", ".gz", ".parquet")
RESTRICTED = re.compile(
    r"(?:^|[^a-z0-9])(cal(?:ibration)?|cal_split\w*|holdout|heldout|sealed|chunk[-_]?5|j4b|inc1)"
    r"(?:$|[^a-z0-9])", re.IGNORECASE)
MAX_TAR_BYTES = 800 * 2**20


class Remote:
    def __init__(self, host: str | None, run_dir: str, local_root: Path | None):
        self.host, self.run_dir, self.local = host, run_dir, local_root

    def _ssh(self, command: str, stdin: bytes | None = None) -> bytes:
        cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=30", self.host, command]
        p = subprocess.run(cmd, input=stdin, capture_output=True)
        if p.returncode != 0:
            raise SystemExit(f"ssh failed (rc {p.returncode}): {p.stderr.decode(errors='replace')[:400]}")
        return p.stdout

    def list_files(self) -> list[tuple[str, int]]:
        if self.local is not None:
            return sorted((f.relative_to(self.local).as_posix(), f.stat().st_size)
                          for f in self.local.rglob("*") if f.is_file())
        out = self._ssh(f"cd {shlex.quote(self.run_dir)} && find . -type f -printf '%s %P\\n'").decode()
        rows = []
        for line in out.splitlines():
            size, _, name = line.partition(" ")
            if name:
                rows.append((name, int(size)))
        return sorted(rows)

    def sha256(self, names: list[str]) -> dict[str, str]:
        if self.local is not None:
            return {n: hashlib.sha256((self.local / n).read_bytes()).hexdigest() for n in names}
        listing = "\n".join(names).encode() + b"\n"
        out = self._ssh(f"cd {shlex.quote(self.run_dir)} && xargs -d '\\n' sha256sum --", stdin=listing).decode()
        sums = {}
        for line in out.splitlines():
            digest, _, name = line.partition("  ")
            sums[name.strip()] = digest.strip()
        return sums

    def fetch(self, names: list[str], dest: Path) -> None:
        if self.local is not None:
            for n in names:
                (dest / n).parent.mkdir(parents=True, exist_ok=True)
                (dest / n).write_bytes((self.local / n).read_bytes())
            return
        listing = "\n".join(names).encode() + b"\n"
        data = self._ssh(f"tar -C {shlex.quote(self.run_dir)} -cf - -T -", stdin=listing)
        with tarfile.open(fileobj=io.BytesIO(data), mode="r") as tf:
            for m in tf:
                if m.name.startswith("/") or ".." in m.name.split("/"):
                    raise SystemExit(f"unsafe member {m.name}")
                tf.extract(m, dest)


def sha256_local(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", help="ssh alias (e.g. gpu-host or a CPU host)")
    ap.add_argument("--run", required=True, help="run id under <remote-home>/runs")
    ap.add_argument("--remote-home", default="/workspace/c4")
    ap.add_argument("--max-small-mb", type=float, default=5.0)
    ap.add_argument("--include-large", action="append", default=[], metavar="GLOB",
                    help="glob relative to the run dir of large files to fetch (e.g. 'out/wm_s_best.pt')")
    ap.add_argument("--exclude", action="append", default=["*.prev_*", "*/__pycache__/*"], metavar="GLOB")
    ap.add_argument("--dest", type=Path, default=None, help="override the small-file destination")
    ap.add_argument("--list-only", action="store_true")
    ap.add_argument("--local-root", type=Path, default=None, help="test mode: treat this directory as the run dir")
    args = ap.parse_args(argv)
    if args.local_root is None and not args.host:
        raise SystemExit("--host is required (or --local-root for a dry test)")
    host_tag = args.host or "local"
    remote = Remote(args.host, f"{args.remote_home.rstrip('/')}/runs/{args.run}", args.local_root)
    rows = remote.list_files()

    small, large, skipped = [], [], []
    for name, size in rows:
        if RESTRICTED.search(name) or any(fnmatch.fnmatch(name, g) for g in args.exclude):
            skipped.append((name, "filtered"))
            continue
        is_large = size > args.max_small_mb * 2**20 or name.endswith(LARGE_SUFFIXES)
        if not is_large:
            small.append((name, size))
        elif any(fnmatch.fnmatch(name, g) for g in args.include_large):
            large.append((name, size))
        else:
            skipped.append((name, f"large ({size / 2**20:.1f} MB), not requested"))
    small_total = sum(s for _, s in small)
    print(f"{args.run}: {len(rows)} files; small to fetch {len(small)} ({small_total / 2**20:.1f} MB); "
          f"large requested {len(large)} ({sum(s for _, s in large) / 2**20:.1f} MB); skipped {len(skipped)}")
    for name, why in skipped[:25]:
        print(f"   skip {name}  [{why}]")
    if args.list_only:
        for name, size in small + large:
            print(f"   {size:>12d}  {name}")
        return 0
    if small_total > MAX_TAR_BYTES:
        raise SystemExit("small-file set exceeds 800 MB; lower --max-small-mb")

    jobs = [(small, args.dest or EVIDENCE / "fetched" / host_tag / args.run),
            (large, REPO / "tmp" / "c4_fetch" / host_tag / args.run)]
    bad = 0
    for files, dest in jobs:
        if not files:
            continue
        names = [n for n, _ in files]
        sums = remote.sha256(names)
        missing = [n for n in names if n not in sums]
        if missing:
            raise SystemExit(f"no remote sha256 for {missing[:3]}")
        dest.mkdir(parents=True, exist_ok=True)
        for n in names:
            if (dest / n).exists():
                raise SystemExit(f"{dest / n} already exists; fetch into a fresh --dest (never overwritten)")
        remote.fetch(names, dest)
        for n in names:
            got = sha256_local(dest / n)
            if got != sums[n]:
                (dest / n).rename(dest / (n + ".BAD"))
                print(f"SHA MISMATCH {n}: remote {sums[n]} local {got}", file=sys.stderr)
                bad += 1
        (dest / "FETCH_SHA256SUMS").write_text(
            "".join(f"{sums[n]}  {n}\n" for n in names), encoding="utf-8", newline="\n")
        print(f"fetched {len(names)} files -> {dest} ({'with MISMATCHES' if bad else 'all sha256 verified'})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
