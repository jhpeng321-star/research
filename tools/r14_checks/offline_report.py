"""S-R14 offline evidence report (CPU, no training): hashes every rollout file and stores the
deterministic ana_r14 analyses that the R14 gate recomputes (round14_gate.check_offline).

  python offline_report.py --root DIAG_DIR --commit SHA --out DIAG_DIR/R14_OFFLINE_REPORT.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ana_r14  # noqa: E402
import d2_merge_timing  # noqa: E402,F401  (provenance: same commit)
import sanity_roll as SR  # noqa: E402

ARMS = ["13E", "13EL"]
ANALYSES = {"panel120": dict(dir="panel120", horizon_s=120.0, tags=["ref"]),
            "pairs300": dict(dir="pairs300", horizon_s=300.0, tags=["ref", "act"])}


def sha(path):
    h = hashlib.sha256()
    with open(SR.guard(path), "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--commit", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = Path(SR.guard(a.root)).resolve()
    files, analyses = {}, {}
    for key, spec in ANALYSES.items():
        d = root / spec["dir"]
        for f in sorted(d.glob("p*_*.npz")):
            files[f.relative_to(root).as_posix()] = sha(f)
        for f in sorted(d.glob("run_meta_*.json")):
            files[f.relative_to(root).as_posix()] = sha(f)
        result = ana_r14.analyze(str(d), spec["horizon_s"], ARMS, spec["tags"])
        analyses[key] = {**spec, "result": json.loads(json.dumps(result, allow_nan=False))}
    doc = {"status": "complete", "study": "S-R14 offline (inference only, gpuR CPU, nice 19)",
           "checkpoint_sha256": SR.CKPT_SHA, "arms": ARMS, "code_commit": a.commit,
           "harness": "tools/r14_checks/roll_r14.py (panel: R13E TRAIN panel; pairs: LONGDIAG sel48 5 pairs)",
           "files": files, "analyses": analyses, "written": time.strftime("%F %T")}
    Path(a.out).write_text(json.dumps(doc, indent=1, allow_nan=False))
    print(f"files {len(files)}; analyses {list(analyses)}")


if __name__ == "__main__":
    main()
