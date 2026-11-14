"""Verify the consolidated V0 dataset: re-hash EVERY file listed in
MANIFEST.json and print PASS/FAIL counts.  Also checks the manifest
structure (per-chunk plan path + plan sha256 + runs root, world keys,
skipped lists).  Exit code 1 on any FAIL.

Usage:  python verify_v0data.py [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha256_file(path, block=1 << 20):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE))
    args = ap.parse_args(argv)
    out = Path(args.out)
    t0 = time.time()
    man = json.loads((out / "MANIFEST.json").read_text(encoding="utf-8"))

    n_pass = n_fail = 0
    failures = []
    for key, entry in sorted(man["worlds"].items()):
        for rel, sha in sorted(entry["files"].items()):
            p = out / rel
            if not p.is_file():
                n_fail += 1
                failures.append(f"{rel}: MISSING")
                continue
            if sha256_file(p) != sha:
                n_fail += 1
                failures.append(f"{rel}: HASH MISMATCH")
            else:
                n_pass += 1

    # structural checks (every chunk present in the MANIFEST)
    struct = []
    for chunk_id in sorted(man.get("chunks", {})):
        sec = man["chunks"][chunk_id]
        for field in ("plan", "plan_sha256", "runs_root"):
            if not sec.get(field):
                struct.append(f"chunks.{chunk_id}.{field}: missing")
        bad = [k for k in sec.get("world_keys", [])
               if k not in man["worlds"]]
        if bad:
            struct.append(f"chunks.{chunk_id}.world_keys not in worlds: "
                          f"{bad[:3]}")
        stale = [k for k in man["worlds"]
                 if k.startswith(f"{chunk_id}/")
                 and k not in set(sec.get("world_keys", []))]
        if stale:
            struct.append(f"chunks.{chunk_id}: worlds not in world_keys: "
                          f"{stale[:3]}")

    print(f"MANIFEST sources: {len(man['worlds'])} "
          f"(chunks {sorted(man.get('chunks', {}))})")
    print(f"file hash verification: PASS={n_pass} FAIL={n_fail} "
          f"({time.time() - t0:.1f}s)")
    for f in failures[:20]:
        print(f"  FAIL {f}")
    for s in struct:
        print(f"  STRUCT {s}")
    ok = n_fail == 0 and not struct
    print("VERIFY_V0DATA: " + ("PASS (0 FAIL)" if ok else "FAIL"))
    (out / "results").mkdir(exist_ok=True)
    (out / "results" / "verify_v0data.json").write_text(json.dumps(
        dict(schema="wd01_partB_v0data_verify/1",
             generated=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
             n_sources=len(man["worlds"]), n_files_pass=n_pass,
             n_files_fail=n_fail, structural_issues=struct,
             failures=failures, ok=ok), indent=1), encoding="utf-8")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
