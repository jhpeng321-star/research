"""S-EXECPARITY2 remaining-error decomposition: base law vs cap vs state.

Splits the 32,617 replayed rows of the 00:20 recordings by whether the WM
frame's leader (``native_cav.leader_gap_dv``: nearest same-lane vehicle ahead
in the corridor frame) is the SAME leader the deployment executor used
(recorded ``getLeader`` gap/speed, compared as bumper gap). On matched rows the
shared exact law's base error is the pure function-identity error and the
mirror error is the cap/state residual; mismatched rows isolate the WM frame's
leader-identification error (mid-merge vehicles whose corridor projection
places them ~0-2 m ahead of a mainline ego that SUMO does not yet count as its
leader). No SUMO execution.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(Path(__file__).resolve().parent)]

from replay import states_from_episode, summary
from wms.control.alwayscmd import executed, inferred_base, inferred_cap
from wms.control.native_cav import head_native_accel, leader_gap_dv

GAP_TOL_M = 0.05
SPEED_TOL_MPS = 0.05


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    jobs = sorted(p for p in args.jobs.iterdir()
                  if p.is_dir() and (p / "execution_log.jsonl.gz").is_file()
                  and p.name != "s977000_always0")
    assert jobs, "no fetched always-commanded job outputs"
    groups = {"matched": {}, "mismatched": {}}
    worlds = {}
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
        state_at = states_from_episode(tape)
        acc = {"matched": {k: [] for k in ("imagined", "mirror", "base", "head")},
               "mismatched": {k: [] for k in ("imagined", "mirror", "base", "head")}}
        phantom_gap = 0
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
            stable = (tape["present"][i, k] and tape["present"][i + 1, k]
                      and tape["is_cav"][i, k] and tape["lane"][i, k] >= 0
                      and tape["vid_index"][i + 1, k] == tape["vid_index"][i, k])
            if not stable:
                continue
            v0 = float(tape["v_mps"][i, k])
            if abs(v0 - float(r["speed_before_mps"])) > 1e-6:
                continue
            realised = (float(tape["v_mps"][i + 1, k]) - v0) / 0.5
            st = state_at(i)
            keep = torch.nonzero(st["present"][0]).flatten()
            local = int(np.flatnonzero(keep.numpy() == k)[0])
            st = {key: value.index_select(1, keep) for key, value in st.items()}
            residual = torch.zeros_like(st["v_mps"])
            residual[0, local] = float(r["residual_mps2"])
            with torch.no_grad():
                head = float(head_native_accel(st)[0, local])
                base = float(inferred_base(st, None)[0, local])
                cap = float(inferred_cap(st)[0, local])
                imag = float(executed(st, residual)[0, local])
                gap_t, dv_t, has_t = leader_gap_dv(st)
            mirror = max(-v0 / .5, min(float(r["issued_mps2"]), cap))
            rec = r["leader"]
            frame_has = bool(has_t[0, local])
            frame_gap = float(gap_t[0, local])
            frame_vl = float(st["v_mps"][0, local] - dv_t[0, local])
            if rec is None:
                matched = not frame_has
            else:
                matched = (frame_has
                           and abs(float(rec["gap_m"]) + 2.0 - frame_gap) <= GAP_TOL_M
                           and abs(float(rec["speed_mps"]) - frame_vl) <= SPEED_TOL_MPS)
            group = "matched" if matched else "mismatched"
            if not matched and rec is not None and frame_has and frame_gap < float(rec["gap_m"]) + 2.0:
                phantom_gap += 1
            acc[group]["imagined"].append(imag - realised)
            acc[group]["mirror"].append(mirror - realised)
            acc[group]["base"].append(base - float(r["base_law_mps2"]))
            acc[group]["head"].append(head - realised)
        worlds[job.name] = {g: {k: summary(np.array(v)) if v else {"n": 0}
                                for k, v in keys.items()} for g, keys in acc.items()}
        for g, keys in acc.items():
            for k, v in keys.items():
                groups[g].setdefault(k, []).extend(v)
        print(job.name, {g: len(v["imagined"]) for g, v in acc.items()}, flush=True)
    n_m = len(groups["matched"]["imagined"])
    n_x = len(groups["mismatched"]["imagined"])
    result = {
        "schema": "execparity2_decomposition/1",
        "classification": ("frame leader == recorded executor leader (bumper gap within "
                           f"{GAP_TOL_M} m, leader speed within {SPEED_TOL_MPS} m/s)"),
        "leader_matched": {"n_rows": n_m, "fraction": n_m / (n_m + n_x),
                           **{"imagined_minus_realised": summary(np.array(groups["matched"]["imagined"])),
                              "mirror_from_issued_minus_realised": summary(np.array(groups["matched"]["mirror"])),
                              "base_minus_sumo_base": summary(np.array(groups["matched"]["base"])),
                              "head_minus_realised": summary(np.array(groups["matched"]["head"]))}},
        "leader_mismatched": {"n_rows": n_x, "fraction": n_x / (n_m + n_x),
                              "phantom_frame_leader_closer": phantom_gap,
                              **{"imagined_minus_realised": summary(np.array(groups["mismatched"]["imagined"])),
                                 "mirror_from_issued_minus_realised": summary(np.array(groups["mismatched"]["mirror"])),
                                 "base_minus_sumo_base": summary(np.array(groups["mismatched"]["base"])),
                                 "head_minus_realised": summary(np.array(groups["mismatched"]["head"]))}},
        "worlds": worlds,
        "measured_real_sumo": True,
        "new_sumo_sim_s": 0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                        encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("leader_matched", "leader_mismatched")},
                     indent=2))


if __name__ == "__main__":
    main()
