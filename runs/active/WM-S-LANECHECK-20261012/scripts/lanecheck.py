"""One-step recorded-state MOBIL comparison and saved rollout lane attribution.

Offline-only. Reads the declared development wall_same / seen_controls states,
the evaluation-only oracle driver map, and the previously saved phys_true rollout.
No model training or simulator calls.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
from wms.eval.normalize import initial_frame, iter_windows, recorded_for  # noqa: E402
from wms.physics.config import PhysicsConfig  # noqa: E402
from wms.physics.idm import IDMParams  # noqa: E402
from wms.physics.mobil import mobil_merge_logits  # noqa: E402

RUN = Path(__file__).resolve().parents[1]
TRUTH_PATH = ROOT / "runs/active/WM-S-R4-DIAG-ORACLE-20261011/out/truth.json"
ROLLOUT_PATH = ROOT / "runs/active/WM-S-R4-DIAG-ORACLE-20261011/out/variant_phys_true.pkl"
SETS = {"wall_same": ROOT / "data_cache/wallc1_states_h60/wall_same.json",
        "seen_controls": ROOT / "data_cache/wallc1_states_h60/seen_controls.json"}
DT = 0.5
JAM_SPEED = 5.0
NEAR_M = 25.0
TYPE_IDM = {"aggressive_car": (31.29 * 1.08, .95, 1.5, 3.0, 4.5),
            "normal_car": (31.29, 1.10, 2.5, 2.6, 4.5),
            "cautious_car": (31.29 * .92, 1.50, 3.5, 2.0, 3.0)}


def state_arrays(record: dict, arm: str):
    """Return aligned [tick, vehicle] state. Internal lane -1 is forward-filled."""
    t0 = float(record["t0_s"])
    frames = [initial_frame(record), *recorded_for(record, arm)]
    by_tick = {}
    ids = set()
    for fr in frames:
        tick = int(round((float(fr["time_s"]) - t0) / DT))
        if tick < 0 or tick > 240:
            continue
        by_tick[tick] = fr
        ids.update(str(v["id"]) for v in fr.get("vehicles", []))
    ids = sorted(ids)
    ix = {vid: j for j, vid in enumerate(ids)}
    shape = (241, len(ids))
    s = np.zeros(shape, np.float32); v = np.zeros(shape, np.float32)
    lane = np.full(shape, -1, np.int16); present = np.zeros(shape, bool)
    cav = np.zeros(shape, bool)
    for tick, fr in by_tick.items():
        for row in fr.get("vehicles", []):
            if not row.get("present", True):
                continue
            j = ix[str(row["id"])]
            s[tick,j] = float(row["s"]); v[tick,j] = float(row["v"])
            lane[tick,j] = int(row.get("lane", -1)); present[tick,j] = True
            cav[tick,j] = bool(row.get("is_cav", False))
    f_lane = lane.copy()
    for j in range(len(ids)):
        last = -1
        for tick in range(241):
            if f_lane[tick,j] >= 0:
                last = int(f_lane[tick,j])
            elif last >= 0:
                f_lane[tick,j] = last
    return ids, s, v, lane, f_lane, present, cav, by_tick


def driver_parameters(ids, world, truth, device="cpu"):
    vals = [TYPE_IDM.get(truth.get(world, {}).get("vehicles", {}).get(i, "normal_car"),
                         TYPE_IDM["normal_car"]) for i in ids]
    x = torch.tensor(vals, dtype=torch.float32, device=device)
    out = {k: x[:, n].unsqueeze(0) for n, k in enumerate(
        ("v0", "time_headway", "min_gap", "accel", "decel"))}
    out.update({"politeness": torch.full((1, len(ids)), .2),
                "mobil_threshold": torch.full((1, len(ids)), .1),
                "merge_gap": torch.full((1, len(ids)), 6.0),
                "merge_urgency": torch.ones((1, len(ids)))})
    return out


def make_kin(s, v, lane):
    t, n = lane.shape
    kin = torch.zeros((t, n, 7), dtype=torch.float32)
    kin[..., 0] = torch.as_tensor(s)
    for li in range(3):
        kin[..., 1 + li] = torch.as_tensor(lane == li, dtype=torch.float32)
    kin[..., 4] = torch.as_tensor(v)
    kin[..., 6] = 5.0
    return kin


def event_category(old, new, position):
    if old == 2 and new == 1:
        return "ramp_to_mainline"
    if old == 1 and new == 0:
        return "mainline_left_overtake"
    if old == 0 and new == 1:
        return "mainline_right_return"
    return "other_lane_change"


def run_set(name, path, truth, physics):
    aggregate = {mode: Counter() for mode in ("default", "true_type")}
    categories = {mode: defaultdict(Counter) for mode in aggregate}
    events = []
    step_stats = {mode: Counter() for mode in aggregate}
    for record in iter_windows(path):
        arm = "wall" if "wall" in record.get("recorded", {}) else next(iter(record["recorded"]))
        ids, s, v, raw_lane, lane, present, cav, by_tick = state_arrays(record, arm)
        wall_ids = {str(value) for value in record.get("wall_members", [])}
        n = len(ids)
        p = torch.as_tensor(present[:, :] & (lane >= 0) & ~cav)
        kin = make_kin(s, v, lane)
        all_changes = {mode: [] for mode in aggregate}
        for mode in aggregate:
            params = (driver_parameters(ids, str(record.get("world")), truth)
                      if mode == "true_type" else None)
            if params is not None:
                params = {key: value.expand(kin.shape[0] - 1, -1)
                          for key, value in params.items()}
            logits = mobil_merge_logits(kin[:-1], p[:-1], physics, params)
            want = logits.argmax(-1).numpy()
            curr = lane[:-1]
            want_event = (want != curr) & p[:-1].numpy()
            step_stats[mode]["hdv_valid_decisions"] += int(p[:-1].sum())
            step_stats[mode]["wanted_vehicle_steps"] += int(want_event.sum())
            # Collapse consecutive wanted ticks for a vehicle/direction into one intent episode.
            for j, vid in enumerate(ids):
                mask = want_event[:, j]
                ticks = np.flatnonzero(mask)
                if ticks.size:
                    groups = np.split(ticks, np.where(np.diff(ticks) > 1)[0] + 1)
                    for group in groups:
                        if len(group):
                            tick = int(group[0]); old = int(curr[tick, j]); new = int(want[tick, j])
                            all_changes[mode].append({"id": vid, "tick": tick, "time_s": tick * DT,
                                "from": old, "to": new, "category": event_category(old,new,s[tick,j]),
                                "position_m": float(s[tick,j]), "speed_mps": float(v[tick,j]),
                                "near_cav_25m": bool(np.any(present[tick] & cav[tick] &
                                    (np.abs(s[tick] - s[tick,j]) <= NEAR_M))),
                                "near_wall_member_25m": bool(any(wid in ids and
                                    present[tick, ids.index(wid)] and
                                    abs(float(s[tick, ids.index(wid)] - s[tick,j])) <= NEAR_M
                                    for wid in wall_ids)),
                                "jam": bool(v[tick,j] < JAM_SPEED)})
        actual = []
        for tick in range(1, 241):
            for j, vid in enumerate(ids):
                # TRUE label: forward-fill junction -1 first, then compare valid lanes.
                a, b = int(lane[tick-1,j]), int(lane[tick,j])
                if not (present[tick-1,j] and present[tick,j] and not cav[tick,j]
                        and a >= 0 and b >= 0 and a != b):
                    continue
                ev = {"id": vid, "tick": tick-1, "time_s": (tick-1)*DT, "from": a, "to": b,
                      "category": event_category(a,b,s[tick,j]), "position_m": float(s[tick,j]),
                      "speed_mps": float(v[tick,j]), "jam": bool(v[tick,j] < JAM_SPEED),
                      "timing_by_model": {},
                      "near_cav_25m": bool(np.any(present[tick] & cav[tick] &
                          (np.abs(s[tick] - s[tick,j]) <= NEAR_M))),
                      "near_wall_member_25m": bool(any(wid in ids and
                          present[tick, ids.index(wid)] and
                          abs(float(s[tick, ids.index(wid)] - s[tick,j])) <= NEAR_M
                          for wid in wall_ids))}
                actual.append(ev)
                if name == "wall_same":
                    events.append({"set":name,"world":record.get("world"),
                                   "state_key":record.get("key"),**ev})
        for mode, wanted in all_changes.items():
            by_tick = {(e["id"],e["tick"],e["to"]): e for e in wanted}
            used = set(); matches = {}
            for i, ev in enumerate(actual):
                k = (ev["id"], ev["tick"], ev["to"])
                if k in by_tick and k not in used:
                    used.add(k); matches[i] = by_tick[k]
            c = aggregate[mode]
            c["recorded_events"] += len(actual); c["wanted_episodes"] += len(wanted)
            c["matched_same_transition_direction"] += len(matches)
            for i, ev in enumerate(actual):
                same = matches.get(i)
                cat = categories[mode][ev["category"]]
                cat["recorded"] += 1
                candidates = [e for e in wanted if e["id"] == ev["id"] and e["to"] == ev["to"]]
                if candidates:
                    near = min(candidates, key=lambda e: abs(e["tick"] - ev["tick"]))
                    off = (near["tick"] - ev["tick"]) * DT
                    ev.setdefault("timing_by_model", {})[mode] = {
                        "nearest_wanted_time_s": near["time_s"],
                        "offset_s_mobil_minus_recorded": off,
                        "classification": "early" if off < 0 else "late" if off > 0 else "on_time"}
                    cat["timing_matched"] += 1
                    cat["offset_sum_s"] += off
                    cat["offset_abs_sum_s"] += abs(off)
                    bucket = "early" if off < 0 else "late" if off > 0 else "on_time"
                    cat[bucket] += 1
                else:
                    ev.setdefault("timing_by_model", {})[mode] = {
                        "nearest_wanted_time_s": None, "offset_s_mobil_minus_recorded": None,
                        "classification": "never"}
                    cat["never"] += 1
                if same:
                    cat["same_transition_match"] += 1
                if ev["near_cav_25m"]:
                    cat["near_cav_recorded"] += 1
                if ev["near_wall_member_25m"]:
                    cat["near_wall_recorded"] += 1
                if ev["jam"]:
                    cat["jam_recorded"] += 1
                if ev["category"] == "ramp_to_mainline":
                    cat["merge_position_sum_m"] += ev["position_m"]
            for e in wanted:
                wc = categories[mode][e["category"]]
                wc["wanted_episodes"] += 1
                wc["wanted_position_sum_m"] += e["position_m"]
                wc["near_cav_wanted"] += int(e["near_cav_25m"])
                wc["near_wall_wanted"] += int(e["near_wall_member_25m"])
                wc["jam_wanted"] += int(e["jam"])
                if (e["id"],e["tick"],e["to"]) not in used:
                    categories[mode][e["category"]]["unmatched_wanted_episodes"] += 1
    summary = {}
    for mode, c in aggregate.items():
        tp = c["matched_same_transition_direction"]
        summary[mode] = {"events":dict(c),
            "valid_hdv_decisions":int(step_stats[mode]["hdv_valid_decisions"]),
            "wanted_vehicle_steps":int(step_stats[mode]["wanted_vehicle_steps"]),
            "wanted_step_rate":step_stats[mode]["wanted_vehicle_steps"] /
                step_stats[mode]["hdv_valid_decisions"] if step_stats[mode]["hdv_valid_decisions"] else None,
            "unmatched_wanted_episodes":c["wanted_episodes"]-c["matched_same_transition_direction"],
            "precision_same_transition": tp / c["wanted_episodes"] if c["wanted_episodes"] else None,
            "recall_same_transition": tp / c["recorded_events"] if c["recorded_events"] else None,
            "categories":{cat:dict(vals) for cat, vals in categories[mode].items()},
            "one_step_definition":"a wanted event is an argmax lane change from the recorded current lane, collapsed over adjacent wanted ticks per vehicle/direction; exact same 0.5s transition and direction is TP"}
    return summary, events


def rollout_attribution(truth):
    rows = pickle.load(ROLLOUT_PATH.open("rb"))
    by_h = {}
    for h in (60, 120):
        tick = int(h / DT)
        total_error = 0.0
        bad = Counter(); sums = Counter(); counts = Counter()
        for row in rows:
            if not row or row.get("error"):
                continue
            pa, ra = row["pred"], row["rec"]
            rindex = {v:i for i,v in enumerate(row["rids"])}
            world = row["world"]
            for pi, vid in enumerate(row["pids"]):
                ri = rindex.get(vid)
                if ri is None or not pa["present"][tick,pi] or not ra["present"][tick,ri] or ra["is_cav"][tick,ri]:
                    continue
                err = abs(float(pa["s"][tick,pi] - ra["s"][tick,ri])); total_error += err
                if pa["lane"][tick,pi] == ra["lane"][tick,ri]:
                    continue
                # The latest lane divergence on either path is a descriptive category attribution.
                cat = "unclassified"
                for k in range(tick, 0, -1):
                    rp0, rp1 = int(ra["lane"][k-1,ri]), int(ra["lane"][k,ri])
                    pp0, pp1 = int(pa["lane"][k-1,pi]), int(pa["lane"][k,pi])
                    if rp0 != rp1 or pp0 != pp1:
                        choices = []
                        if rp0 >= 0 and rp1 >= 0 and rp0 != rp1: choices.append((rp0,rp1))
                        if pp0 >= 0 and pp1 >= 0 and pp0 != pp1: choices.append((pp0,pp1))
                        for old,new in choices:
                            cat = event_category(old,new,float(ra["s"][k,ri])); break
                        if choices: break
                if cat == "unclassified":
                    cat = "ramp_to_mainline" if vid.startswith("ramp.") else "mainline_other"
                bad[cat] += 1; sums[cat] += err; counts[cat] += 1
        by_h[str(h)] = {"wrong_lane_vehicles": int(sum(bad.values())), "matched_hdv_total_abs_ds_m":total_error,
            "categories":{k:{"vehicles":int(bad[k]),"mean_abs_ds_m":sums[k]/bad[k],
                "abs_ds_sum_m":sums[k],"share_of_wrong_lane_abs_ds":sums[k]/sum(sums.values()) if sum(sums.values()) else 0.0,
                "share_of_all_matched_hdv_abs_ds":sums[k]/total_error if total_error else 0.0}
                for k in bad}}
    return by_h


def main():
    torch.set_num_threads(1)
    truth = json.loads(TRUTH_PATH.read_text(encoding="utf-8"))
    physics = PhysicsConfig(sumo_fidelity=True)
    all_results = {}; all_events = []
    for name,path in SETS.items():
        summary, events = run_set(name,path,truth,physics)
        all_results[name] = summary; all_events.extend(events)
    data = {"schema":"wm_s_lanecheck/1", "scope":{"wall_same_states":67,"additional_set":"seen_controls",
        "dt_s":DT,"jam_threshold_mps":"<5","near_cav_m":NEAR_M,
        "true_type_idm_source":"evaluation-only out/truth.json; v0 uses mainline speed limit x speedFactor; lane params remain PhysicsConfig defaults",
        "rollout":"reused variant_phys_true.pkl; no simulation/training/GPU"},
        "sumo_lane_change_setup":{
            "evidence_files":["runs/campaign_stage1_chunk4/{world}__base/run/scenario/scenario.sumocfg",
                             "runs/campaign_stage1_chunk4/{world}__base/run/scenario/requests.rou.xml",
                             "traffic_shepherd/simulation.py::start_sumo",
                             "traffic_shepherd/scenarios.py::build_scenario"],
            "simulation_step_s":0.5,"action_step_length":"not overridden (SUMO default)",
            "lane_change_model":"LC2013 (no laneChangeModel override found)",
            "sublane_model":"not enabled; no lateral-resolution option in cfg or launcher",
            "lane_change_duration":"not configured; no gradual sublane motion configured",
            "vtype_explicit_lane_parameters":{
                vtype:{k:attrs[k] for k in ("lcCooperative","lcAssertive","impatience") if k in attrs}
                for vtype,attrs in next(iter(truth.values()))["vtypes"].items()},
            "lane_parameters_not_set":["lcStrategic","lcSpeedGain","lcKeepRight","lcLookaheadLeft"],
            "unset_parameters_use_sumo_defaults":True},
        "one_step":all_results,"recorded_change_events":all_events,
        "rollout_wrong_lane_attribution":rollout_attribution(truth)}
    out = RUN / "results.json"
    out.write_text(json.dumps(data, separators=(",",":"), allow_nan=False), encoding="utf-8")
    print(json.dumps({"output":str(out),"bytes":out.stat().st_size,
        "one_step":{s:{m:{"events":v["events"],"precision":v["precision_same_transition"],"recall":v["recall_same_transition"]}
            for m,v in r.items()} for s,r in all_results.items()},
        "rollout":data["rollout_wrong_lane_attribution"]}, indent=2))


if __name__ == "__main__":
    main()
