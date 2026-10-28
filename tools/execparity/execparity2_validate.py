"""S-EXECPARITY2 offline identity validation: shared law vs the recorded executor.

Replays the deployment executor's own per-tick ledger rows of the 00:20
always-commanded SUMO runs (977001-977003) through the SHARED exact law
(``wms.control.sumo_idm``) using only deployment-observable inputs:

* base law identity: ``sumo_idm.base_accel_scalar`` on the logged
  (speed_before, getLeader gap, leader speed) plus the tape's lane/position
  allowed speed must reproduce the logged ``base_law_mps2`` - the function the
  executor applied, verified on the executor's own records;
* executed-speed identity: SUMO's realised next speed must equal
  ``max(0, min(v + issued*dt, vSafe, maxNextSpeed))`` with the shared
  ``safe_speed_cap_scalar`` ceiling - the speedMode-27 composition.

No SUMO execution, no new data: reads the fetched job outputs only.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(Path(__file__).resolve().parent)]

from replay import sha, summary
from wms.control import sumo_idm


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs", type=Path, required=True,
                    help="dir of fetched s*_alwaysx job outputs")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    jobs = sorted(p for p in args.jobs.iterdir()
                  if p.is_dir() and (p / "execution_log.jsonl.gz").is_file()
                  and p.name != "s977000_always0")
    assert jobs, "no fetched always-commanded job outputs"
    worlds = {}
    base_err, cap_err, base_exact, cap_exact, n_rows = [], [], 0, 0, 0
    for job in jobs:
        with np.load(job / "deploy/episode/episode.npz", allow_pickle=False) as z:
            tape = dict(z)
        times = tape["time_s"]
        vids = tape["vehicle_ids"]
        rows = []
        with gzip.open(job / "execution_log.jsonl.gz", "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if float(r["t_s"]) >= 300. - 1e-9:
                    rows.append(r)
        w_base, w_cap, w_base_exact, w_cap_exact, w_n = [], [], 0, 0, 0
        for r in rows:
            t = float(r["t_s"])
            i = int(np.searchsorted(times, t))
            if i + 1 >= len(times) or abs(times[i] - t) > 1e-6:
                continue
            slot_ids = vids[tape["vid_index"][i]]
            hits = np.flatnonzero(slot_ids == r["vehicle_id"])
            if len(hits) != 1:
                continue
            k = int(hits[0])
            if not (tape["present"][i, k] and tape["present"][i + 1, k]
                    and tape["vid_index"][i + 1, k] == tape["vid_index"][i, k]):
                continue
            v = float(r["speed_before_mps"])
            if abs(v - float(tape["v_mps"][i, k])) > 1e-6:
                continue
            lane = int(tape["lane"][i, k])
            s_m = float(tape["s_m"][i, k])
            prev_lane = (int(tape["lane"][i - 1, k])
                         if i > 0 and tape["vid_index"][i - 1, k] == tape["vid_index"][i, k]
                         else None)
            allowed = sumo_idm.allowed_speed_scalar(lane, s_m, prev_lane=prev_lane)
            leader = r["leader"]
            # WM bumper gap convention: getLeader gap + minGap
            gap = None if leader is None else float(leader["gap_m"]) + 2.0
            base = sumo_idm.base_accel_scalar(
                v, 0. if leader is None else float(leader["speed_mps"]),
                0. if gap is None else gap, allowed,
                has_leader=leader is not None)
            w_base.append(base - float(r["base_law_mps2"]))
            if abs(w_base[-1]) < 1e-9:
                w_base_exact += 1
            cap = sumo_idm.safe_speed_cap_scalar(
                v, 0. if leader is None else float(leader["speed_mps"]),
                0. if gap is None else gap, allowed,
                has_leader=leader is not None, lane=lane, s_m=s_m)
            v_next_pred = max(0., min(v + float(r["issued_mps2"]) * .5,
                                      v + cap * .5))
            w_cap.append((v_next_pred - float(tape["v_mps"][i + 1, k])) / .5)
            if abs(v_next_pred - float(tape["v_mps"][i + 1, k])) < 1e-6:
                w_cap_exact += 1
            w_n += 1
        base_err.extend(w_base)
        cap_err.extend(w_cap)
        base_exact += w_base_exact
        cap_exact += w_cap_exact
        n_rows += w_n
        worlds[job.name] = {
            "n_rows": w_n,
            "base_exact_fraction": w_base_exact / w_n,
            "base_minus_sumo_base": summary(np.array(w_base)),
            "executed_speed_exact_fraction": w_cap_exact / w_n,
            "executed_speed_accel_error": summary(np.array(w_cap)),
        }
        print(job.name, w_n, flush=True)
    result = {
        "schema": "execparity2_base_validation/1",
        "shared_module": "wms.control.sumo_idm",
        "law": "SUMO 1.27.1 MSCFModel_IDM::_v on cav_typical (a=1.2, b=2.0, T=1.2, "
               "s0=2.0, delta=4, speedFactor=1.0; iterations=2 at dt=0.5; "
               "NUMERICAL_EPS=1e-10); legacy double-minGap base gap convention; "
               "allowed speed = lane limit (mainline 31.29 / ramp 22.22 / "
               "merge connector 26.75 via pre-junction lane)",
        "worlds": worlds,
        "jobs": {job.name: {"execution_log_sha256": sha(job / "execution_log.jsonl.gz"),
                            "episode_sha256": sha(job / "deploy/episode/episode.npz")}
                 for job in jobs},
        "pooled": {
            "n_rows": n_rows,
            "base_exact_fraction_1e-9": base_exact / n_rows,
            "base_minus_sumo_base": summary(np.array(base_err)),
            "executed_speed_exact_fraction_1e-6": cap_exact / n_rows,
            "executed_speed_accel_error": summary(np.array(cap_err)),
        },
        "measured_real_sumo": True,
        "new_sumo_sim_s": 0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                        encoding="utf-8")
    print(json.dumps(result["pooled"], indent=2))


if __name__ == "__main__":
    main()
