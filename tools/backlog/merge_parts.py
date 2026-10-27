"""S-BACKLOG step 2b: merge per-host re-export parts into ONE dataset mirroring the original.

Each part holds converted episodes for the worlds whose tapes live on that host.  The merged
dataset has exactly the original manifest's episodes (same order, same ``dir`` layout, same
split_role/source), each carrying the new ``queue_arrays`` + ``backlog_reexport`` receipt.
Every file that also exists in the ORIGINAL dataset (hash table from ``--orig-sums``) must be
byte-identical (the converter is deterministic); only the new clock arrays are added.
Fail closed on any missing/extra episode or any changed pre-existing array.  Originals are
never touched; ``--out`` must not exist.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--orig-manifest", required=True)
    ap.add_argument("--orig-sums", required=True, help="json {relpath: sha256} of the original dataset dir")
    ap.add_argument("--parts", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        print(f"REFUSED: {out} exists", file=sys.stderr)
        return 2
    orig = json.loads(Path(a.orig_manifest).read_text(encoding="utf-8"))
    osums = json.loads(Path(a.orig_sums).read_text(encoding="utf-8"))
    parts = {}
    for p in a.parts:
        m = json.loads((Path(p) / "C4_EPISODES_MANIFEST.json").read_text(encoding="utf-8"))
        for e in m["episodes"]:
            if e["episode"] in parts:
                print(f"REFUSED: duplicate episode {e['episode']}", file=sys.stderr)
                return 3
            parts[e["episode"]] = (Path(p), e)
    want = [e["episode"] for e in orig["episodes"]]
    if set(parts) != set(want):
        print(f"REFUSED: episode set differs: missing={sorted(set(want) - set(parts))} "
              f"extra={sorted(set(parts) - set(want))}", file=sys.stderr)
        return 4
    entries, same, added, changed = [], 0, 0, []
    for oe in orig["episodes"]:
        root, ne = parts[oe["episode"]]
        src = root / ne["dir"]
        dst = out / oe["dir"]
        shutil.copytree(src, dst)
        for f in sorted(dst.rglob("*")):
            if not f.is_file():
                continue
            rel = f.relative_to(out).as_posix()
            if rel in osums:
                if sha256(f) == osums[rel]:
                    same += 1
                else:
                    changed.append(rel)
            else:
                added += 1
        entry = {**oe, "queue_arrays": ne["queue_arrays"], "backlog_reexport": ne["backlog_reexport"]}
        entries.append(entry)
    if changed:
        print(f"REFUSED: pre-existing arrays differ from the original: {changed[:5]} ({len(changed)})",
              file=sys.stderr)
        shutil.rmtree(out)
        return 5
    manifest = {**{k: v for k, v in orig.items() if k != "episodes"}, "episodes": entries,
                "backlog_reexport": {"task": "S-BACKLOG", "n_episodes": len(entries),
                                     "files_identical_to_original": same, "files_added": added,
                                     "still_waiting_total": sum(e["queue_arrays"]["still_waiting_at_end"]
                                                                for e in entries),
                                     "j_receipt_checks": sum(1 for _ in entries)}}
    (out / "C4_EPISODES_MANIFEST.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    sums = {p.relative_to(out).as_posix(): sha256(p) for p in sorted(out.rglob("*"))
            if p.is_file() and p.name != "SHA256SUMS.json"}
    (out / "SHA256SUMS.json").write_text(json.dumps(sums, indent=1, sort_keys=True), encoding="utf-8")
    print(json.dumps({"out": str(out), "episodes": len(entries), "identical": same, "added": added}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
