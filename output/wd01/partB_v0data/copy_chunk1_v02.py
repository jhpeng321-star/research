"""One-shot v0.2 migration step: copy the existing chunk-1 V0 dataset from
the wd01n worktree into THIS dataset root, hash-verified against the old
MANIFEST before and after the copy.

Read-only on the source (never modified, never deleted).  HOLDOUT worlds
are never built and have no files; the copied file set is asserted to
contain no HOLDOUT world's data.  ZERO SIMULATION.

Usage:  python copy_chunk1_v02.py [--src RD_WT_ROOT/wd01n/output/wd01/partB_v0data]
(default RD_WT_ROOT: E:/rd-wt)
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import wdpaths  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_SRC = str(Path(wdpaths.RD_WT_ROOT) / "wd01n" / "output" / "wd01" /
                  "partB_v0data")


def sha256_file(path, block=1 << 20):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC)
    args = ap.parse_args(argv)
    src, dst = Path(args.src), HERE

    man = json.loads((src / "MANIFEST.json").read_text(encoding="utf-8"))
    holdouts = {s["world"]
                for s in man["chunks"]["chunk1"].get("skipped_holdout", [])}
    expected = sorted({rel for e in man["worlds"].values()
                       for rel in e["files"]})
    assert all(rel.startswith("worlds/chunk1/") for rel in expected), \
        "old MANIFEST lists non-chunk1 files"

    # HOLDOUT guard: no listed file may belong to a HOLDOUT world
    leaked = [rel for rel in expected
              for w in holdouts if f"/{w}__" in rel]
    assert not leaked, f"HOLDOUT files in manifest: {leaked[:3]}"
    print(f"source MANIFEST: {len(man['worlds'])} sources, "
          f"{len(expected)} data files, {len(holdouts)} HOLDOUT worlds "
          f"listed (no files)", flush=True)

    n_ok = 0
    for rel in expected:
        s, d = src / rel, dst / rel
        assert s.is_file(), f"missing in source: {rel}"
        sha_man = next(e["files"][rel] for e in man["worlds"].values()
                       if rel in e["files"])
        sha_before = sha256_file(s)
        assert sha_before == sha_man, f"source hash mismatch: {rel}"
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s, d)
        sha_after = sha256_file(d)
        assert sha_after == sha_man, f"copy hash mismatch: {rel}"
        n_ok += 1
        if n_ok % 100 == 0:
            print(f"  copied+verified {n_ok}/{len(expected)}", flush=True)

    # resume markers (done.json): not manifest-listed; validated against
    # their embedded per-file hashes, then copied for builder resumability
    n_mark = 0
    for key, e in man["worlds"].items():
        marker = src / f"worlds/chunk1/{key.split('/')[1]}_done.json"
        assert marker.is_file(), f"missing done marker: {marker.name}"
        d = json.loads(marker.read_text(encoding="utf-8"))
        assert d.get("files") == e["files"], \
            f"done marker hashes disagree with MANIFEST: {key}"
        shutil.copy2(marker, dst / "worlds" / "chunk1" / marker.name)
        n_mark += 1

    copied = sorted(p.name for p in (dst / "worlds" / "chunk1").iterdir())
    want = sorted([Path(rel).name for rel in expected] +
                  [f"{k.split('/')[1]}_done.json" for k in man["worlds"]])
    assert copied == want, "destination dir does not match the file set"
    shutil.copy2(src / "MANIFEST.json", dst / "MANIFEST.json")
    print(f"chunk-1 copy: {n_ok}/{len(expected)} files sha256-verified "
          f"before AND after copy; {n_mark} done markers validated and "
          f"copied; dir set == manifest set + markers", flush=True)
    print(f"MANIFEST.json copied: src sha {sha256_file(src / 'MANIFEST.json')}"
          f" dst sha {sha256_file(dst / 'MANIFEST.json')}", flush=True)
    print("holdout_accessed=false", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
