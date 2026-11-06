"""Read-only, explicit-path readiness inventory; executed over SSH on each box."""
import hashlib
import json
import os
import pathlib
import shutil
import sys


def summary_qual(path, label):
    p = pathlib.Path(path).expanduser()
    if not p.is_file():
        print(json.dumps({"path": str(p), "exists": False}))
        return
    d = json.loads(p.read_text())
    print(json.dumps({"path": str(p), "keys": list(d)}))
    d = d.get(label, d)
    print(json.dumps({"label": label, "keys": list(d), "core": d.get("core"),
                      "support": d.get("support"), "table": d.get("table"),
                      "safety": d.get("safety"), "fid": d.get("fid")}, default=str))


if sys.argv[1] == "cpu":
    summary_qual("~/actqual/runs/r13e/QUAL_r13e.json", "r13e")
    summary_qual("~/actqual/runs/r14b/QUAL_r14b.json", "r14b")
    p = pathlib.Path("~/autopilot_r10/runs/r14b").expanduser()
    print("autopilot_r14b", [x.name for x in p.iterdir()] if p.exists() else "absent")
    print("python", os.path.realpath("/home/ubuntu/c4/venv/bin/python"))
elif sys.argv[1] == "fid":
    p = pathlib.Path("~/autopilot_r10/runs/r13e/score/r13e/fid.json").expanduser()
    d = json.loads(p.read_text())
    print(json.dumps(d, default=str))
else:
    v5 = pathlib.Path("/root/autodl-tmp/c4/v5")
    for rel in ("policy14_stage", "runs/c4_wm_v10_round13e_b2", "data/lowpen_20261025"):
        p = v5 / rel
        print(rel, [x.name for x in p.iterdir()] if p.exists() else "absent")
    p = v5 / "runs/c4_wm_v10_round13e_b2/wm_s.pt"
    print("b2_sha", hashlib.sha256(p.read_bytes()).hexdigest())
    print("disk", shutil.disk_usage(v5))
