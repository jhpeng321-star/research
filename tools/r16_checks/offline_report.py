"""Write R16_OFFLINE_REPORT.json next to the panel directories (file hashes + deterministic analyses).

  python offline_report.py DIAG_DIR CODE_COMMIT panel120:120 [panel300:300]
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import r16_panel_ana as A  # noqa: E402
import numpy as np  # noqa: E402

CKPT = "952ec059de5a9c0285d2327bcea57c836fd5d736c9975592ffbb932a506e84c8"
ARMS = ["R14", "R16"]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    base, commit, specs = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
    files, analyses = {}, {}
    for spec in specs:
        name, horizon = spec.split(":")
        d = base / name
        for f in sorted(d.glob("p*_ref_*.npz")):
            files[f"{name}/{f.name}"] = sha(f)
        d2 = A.ana_r14.clean(A.d2_analysis(str(d), ARMS, np.random.default_rng(16)))
        r14m = A.ana_r14.analyze(str(d), float(horizon), ARMS, ["ref"], "R14")
        r14m.pop("per_run", None)
        analyses[name] = {"dir": name, "horizon_s": float(horizon), "d2": d2, "r14_metrics": r14m}
    doc = {"status": "complete", "study": "S-RAMPSPEED R16 offline (inference only, gpuR CPU, nice 19)",
           "checkpoint_sha256": CKPT, "physics": "R14 (13E + lane_change_secure_gap + emergency_onset=driver_b)",
           "arms": ARMS, "code_commit": commit, "init": "src/wms/model/ramp_speed_adapt_init.json (TRAIN fit)",
           "harness": "tools/r16_checks/roll_r16.py (R13E TRAIN-role panel, CRN seed 130013 + state index)",
           "files": files, "analyses": analyses, "written": time.strftime("%F %T")}
    out = base / "R16_OFFLINE_REPORT.json"
    out.write_text(json.dumps(A.ana_r14.clean(doc), indent=1, allow_nan=False))
    print(out, len(files))


if __name__ == "__main__":
    main()
