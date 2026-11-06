"""Use the qualification pool's released six slots for the already-registered repairs."""
from __future__ import annotations

import concurrent.futures as F
import json
import os
import sys
import time
from pathlib import Path

import drive as D


def main():
    os.nice(19)
    assert json.loads((D.ROOT / "QUAL_TASKS.json").read_text())["complete"]
    D.ENV["NATIVELANE_WORKER_CAP"] = "3"
    request = json.loads((D.ROOT / "REPAIR_ON60.json").read_text())
    prior = Path.home() / "mappodiag"
    units = []
    for p in sorted((prior / "wm_c").glob("w*_t*.json")):
        d = json.loads(p.read_text())
        if int(d["world"][1:7]) in (967406, 967407, 967408):
            units.append(d["world"] + "@" + str(int(d["t0"])))
    assert len(units) == 3
    common = ["--repo", D.REPO, "--wm", D.WM, "--wm-config", "configs/c4_wm_v10round13e_stage_b.toml",
              "--mappo", prior.parent / "coopmeas/mappo_v5_final.pt", "--samples", "4", "--workers", "3"]
    tasks = [("repair_on60_early", [sys.executable, "tools/nativelane/launch.py", "mappo", prior / "wm_mappo.py",
              "--source", prior / "src_bsrc", "--units", request["units"], "--horizon", "60", "--pad-max", "64",
              "--out", D.ROOT / "on/wm_coop", *common], True),
             ("repair_off300_early", [sys.executable, "tools/nativelane/launch.py", "mappo", prior / "wm_mappo.py",
              "--source", prior / "src_cbase", "--units", ",".join(units), "--horizon", "300", "--pad-max", "300",
              "--out", D.ROOT / "off/clock_early", *common], False)]
    with F.ThreadPoolExecutor(2) as pool:
        rc = list(pool.map(lambda t: D.run(*t), tasks))
    assert rc == [0, 0], rc
    for folder, n, seconds in (("on/wm_coop", 33, 60), ("off/clock_early", 3, 300)):
        docs = [json.loads(p.read_text()) for p in (D.ROOT / folder).glob("w*_t*.json")]
        assert len(docs) == n and all("error" not in d and d.get("actual_horizon_s") == seconds for d in docs)
    D.write(D.ROOT / "EARLY_REPAIRS_DONE.json", {"time": time.time(), "workers_per_pool": 3,
                                               "off300_units": units, "on60_units": 33})


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        D.write(D.ROOT / "EARLY_REPAIRS_FAILED.json", {"error": repr(error), "time": time.time()})
        raise
