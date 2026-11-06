"""Convert one completed world on its collector; export only observable artifacts.

No SUMO is started here. No scenario, assignment XML, or truth file is copied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify(root):
    files = json.loads((root / "HASHES.json").read_text())
    if not files:
        raise ValueError("empty world artifact")
    for rel, expected in files.items():
        path = root / rel
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("artifact escaped root")
        if digest(path) != expected:
            raise ValueError(f"world artifact hash mismatch: {rel}")
    return files


def build(work, split_seed, val_frac):
    from wms.data.c4_episodes import convert
    from wms.control.real_credit14 import gather
    out = work / "artifact"
    out.mkdir(exist_ok=False)
    pairs = out / "pairs"
    convert([work / "collect/branches"], pairs, val_frac=val_frac, split_seed=split_seed)
    manifest_path = pairs / "C4_EPISODES_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    for index, entry in enumerate(manifest["episodes"]):
        # Retain the original npz digest in the ordinary conversion receipt, and
        # make summary-dependent acceptance/real-return export independent of spot storage.
        target = out / "summaries" / str(index)
        target.mkdir(parents=True)
        shutil.copy2(Path(entry["source"]) / "summary.json", target / "summary.json")
        entry["source"] = "@artifact/summaries/" + str(index)
    manifest_path.write_text(json.dumps(manifest, indent=1))
    gather(work / "collect", out / "SUMOCREDIT_POINTS.jsonl", out / "GATE_CALIBRATION.jsonl")
    hashes = {p.relative_to(out).as_posix(): digest(p) for p in sorted(out.rglob("*")) if p.is_file()}
    (out / "HASHES.json").write_text(json.dumps(hashes, sort_keys=True))
    verify(out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--val-frac", type=float, default=0.)
    args = p.parse_args()
    build(args.work, args.split_seed, args.val_frac)
