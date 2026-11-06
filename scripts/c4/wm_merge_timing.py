"""R5TOOLS: censor-aware WM vs SUMO merge-timing evaluation (CPU, no SUMO, no training).

For one or more WM checkpoints, roll TRAIN-val starts (default: the 72 ``wall_same`` h60
starts used by ``tools/v8wm3_check.py`` plus 24 ``v10delaycheck`` starts) and compare the
ramp HDVs (t0 lane 2, non-CAV) against the recorded SUMO run of the *same* cars:

* merge-time survival/CDF (Kaplan-Meier) for WM and SUMO, median/p90 waiting time;
* early/late timing error for cars matched by identity;
* merge position (last ramp-lane position before the merge);
* residual non-merge (stuck) fraction;
* gap-availability strata from the t0 lane-1 gap (tight < 30 m, medium < 60 m, open);
* merge-zone occupancy, mean speed and ramp spillback vs SUMO.

``--k`` repeats every start with fixed seeds (``--seed + state_index * 1000 + sample``) and
reports per-state merge probabilities.  Outputs JSON + Markdown.  Paths carrying restricted
tokens and evaluation seeds 964000-964199 are refused.

    PYTHONPATH="src;.;tools" python scripts/c4/wm_merge_timing.py \\
        --ckpt "v5m=path/to/wm_s_best_ap.pt" --k 3 --horizon 120 --workers 3 \\
        --out check/merge_timing.json --md check/merge_timing.md
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

DT_S = 0.5
MERGE_END_M = 75.44
RAMP_LANE = 2
FORBIDDEN = ("sealed", "holdout", "heldout", "cal_split", "chunk5", "j4b", "inc1")
EVAL_SEED_MIN, EVAL_SEED_MAX = 964000, 964199
DEFAULT_WALL = "E:/agent-work/c4-v6merge/_ref/states_h60/wall_same.json"
DEFAULT_DELAY = ("E:/research-dense/runs/active/C4-V7FT-20261017/v10delaycheck/inputs/"
                 "delay_states.json")
DEFAULT_DELAY_LIMIT = 24
GAP_TIGHT_M = 30.0
GAP_MEDIUM_M = 60.0
GAP_BUCKETS = ("tight", "medium", "open", "unknown")

_WORKER: dict = {}


# --------------------------------------------------------------------------- pure helpers
def guard(path) -> str:
    """Refuse restricted path tokens (the same screen as the other C4 tools)."""
    text = str(path)
    if any(token in text.replace("\\", "/").lower() for token in FORBIDDEN):
        raise SystemExit(f"restricted path token refused: {text}")
    return text


def refused_seed(*values) -> bool:
    """True when any value is an evaluation seed (964000-964199)."""
    for value in values:
        try:
            seed = int(value)
        except (TypeError, ValueError):
            continue
        if EVAL_SEED_MIN <= seed <= EVAL_SEED_MAX:
            return True
    return False


def merge_time(lane, mask, k: int) -> float:
    """First time (s, t0 = 0) the car is present in lane 1; ``inf`` if never."""
    rows = mask[:, k] & (lane[:, k] == 1)
    idx = [i for i, hit in enumerate(rows) if bool(hit)]
    return float(idx[0] * DT_S) if idx else math.inf


def censor_time(mask, k: int) -> float:
    """Last present time (s) of the car; ``0`` when it was never present."""
    idx = [i for i, hit in enumerate(mask[:, k]) if bool(hit)]
    return float(idx[-1] * DT_S) if idx else 0.0


def merge_position(pos, lane, mask, k: int, end: float = MERGE_END_M) -> float:
    """Position at the last ramp-lane row before the first lane-1 row (NaN if no merge)."""
    merged = [i for i, hit in enumerate(mask[:, k] & (lane[:, k] == 1)) if bool(hit)]
    if not merged:
        return math.nan
    first = merged[0]
    ramp = [i for i in range(first) if bool(mask[i, k] and lane[i, k] == RAMP_LANE)]
    row = ramp[-1] if ramp else first
    return float(pos[row, k])


def present_end(mask, k: int) -> float:
    """Observation end of one car (its last present row)."""
    return censor_time(mask, k)


def km_survival(durations, events) -> list[dict]:
    """Kaplan-Meier curve of right-censored waiting times.

    ``durations`` are seconds, ``events[i]`` True for an observed merge.  Censored
    observations stay at risk at their censoring time.  Returns one row per event time with
    ``t``, ``survival``, ``cdf``, ``at_risk`` and ``events``.
    """
    pairs = sorted((float(d), bool(e)) for d, e in zip(durations, events)
                   if math.isfinite(float(d)))
    if not pairs:
        return []
    times = sorted({d for d, e in pairs if e})
    survival = 1.0
    curve = []
    for t in times:
        at_risk = sum(1 for d, _ in pairs if d >= t)
        deaths = sum(1 for d, e in pairs if e and d == t)
        if at_risk:
            survival *= 1.0 - deaths / at_risk
        curve.append({"t": t, "survival": survival, "cdf": 1.0 - survival,
                      "at_risk": at_risk, "events": deaths})
    return curve


def km_quantile(curve: list[dict], q: float):
    """Time at which the KM survival drops to ``1 - q`` (``None`` when unreached)."""
    target = 1.0 - float(q)
    for row in curve:
        if row["survival"] <= target:
            return row["t"]
    return None


def km_summary(durations, events) -> dict:
    """Median/p90 plus counts and the survival curve of one censored sample."""
    durations = [float(d) for d in durations]
    events = [bool(e) for e in events]
    curve = km_survival(durations, events)
    return {
        "n": len(durations),
        "events": sum(events),
        "censored": len(durations) - sum(events),
        "median_s": km_quantile(curve, 0.5),
        "p90_s": km_quantile(curve, 0.9),
        "curve": curve,
    }


def gap_bucket(gap_m) -> str:
    """Gap-availability stratum of a t0 lane-1 gap (metres)."""
    if gap_m is None or not math.isfinite(float(gap_m)):
        return "unknown"
    gap = float(gap_m)
    if gap < GAP_TIGHT_M:
        return "tight"
    if gap < GAP_MEDIUM_M:
        return "medium"
    return "open"


def t0_lane1_gap(kin, present, k: int) -> float:
    """Distance to the nearest t0 lane-1 neighbour (leader or follower), ``inf`` if none."""
    lanes = kin[:, 1:4]
    lane1 = [j for j in range(len(lanes))
             if bool(present[j]) and int(lanes[j].argmax()) == 1]
    if not lane1:
        return math.inf
    s0 = float(kin[k, 0])
    ahead = [float(kin[j, 0]) - s0 for j in lane1 if float(kin[j, 0]) >= s0]
    behind = [s0 - float(kin[j, 0]) for j in lane1 if float(kin[j, 0]) < s0]
    gaps = [g for g in (min(ahead) if ahead else None, min(behind) if behind else None)
            if g is not None]
    return min(gaps) if gaps else math.inf


def zone_stats(pos, vel, mask, lane, end: float = MERGE_END_M) -> dict:
    """Time-mean merge-zone occupancy/speed and ramp spillback of one rollout."""
    rows = pos.shape[0]
    occ, spill, spill_steps, speeds = [], [], [], []
    for i in range(rows):
        live = mask[i]
        inside = live & (pos[i] >= 0.0) & (pos[i] < end) & (lane[i] >= 0)
        occ.append(float(inside.sum()))
        if inside.any():
            speeds.extend(float(v) for v in vel[i][inside])
        behind = live & (lane[i] == RAMP_LANE) & (pos[i] < 0.0)
        count = float(behind.sum())
        spill.append(count)
        spill_steps.append(1.0 if count > 0 else 0.0)
    return {
        "occupancy": float(sum(occ) / rows) if rows else 0.0,
        "speed_mps": float(sum(speeds) / len(speeds)) if speeds else None,
        "spillback_veh": float(sum(spill) / rows) if rows else 0.0,
        "spillback_step_fraction": float(sum(spill_steps) / rows) if rows else 0.0,
    }


def _median(values):
    vals = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    if not vals:
        return None
    mid = len(vals) // 2
    return vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2.0


def _h(value) -> str:
    return "inf" if value is None or not math.isfinite(value) else f"{float(value):.1f}"


def timing_block(wm_t, sumo_t) -> dict:
    """Matched-car timing errors (WM minus SUMO)."""
    return timing_pairs(zip(wm_t, sumo_t))


def timing_pairs(pairs) -> dict:
    """Timing errors from identity-matched ``(wm_t, sumo_t)`` pairs."""
    errs = [float(a) - float(b) for a, b in pairs
            if math.isfinite(float(a)) and math.isfinite(float(b))]
    return {
        "matched": len(errs),
        "median_signed_s": _median(errs),
        "median_abs_s": _median([abs(e) for e in errs]),
        "early_gt_2s": sum(1 for e in errs if e < -2.0),
        "late_gt_2s": sum(1 for e in errs if e > 2.0),
    }


def probability_summary(values) -> dict:
    """Mean/median/min/max of per-state Bernoulli probabilities."""
    vals = [float(v) for v in values if math.isfinite(float(v))]
    if not vals:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {"n": len(vals), "mean": sum(vals) / len(vals), "median": _median(vals),
            "min": min(vals), "max": max(vals)}


# --------------------------------------------------------------------------- loading
def _record_arm(rec: dict) -> str:
    recorded = rec.get("recorded") or {}
    if isinstance(recorded, dict) and "wall" in recorded:
        return "wall"
    arm = rec.get("arm") or rec.get("source_arm")
    if arm:
        return str(arm)
    return "native"


def state_key(rec: dict, index: int) -> str:
    return str(rec.get("key") or rec.get("world") or f"state_{index}")


def load_sources(args) -> list[tuple[str, int, dict, str]]:
    """-> ``[(source, index, record, arm)]`` with restricted seeds/paths refused."""
    from wms.eval.normalize import iter_windows

    rows: list[tuple[str, int, dict, str]] = []
    if args.states:
        path = Path(guard(args.states))
        if path.is_dir():
            path = path / "delay_states.json"
        path = guard(path)
        for i, rec in enumerate(iter_windows(path)):
            if refused_seed(rec.get("seed"), rec.get("sumo_seed")):
                raise SystemExit(f"refusing evaluation seed in {path} state {i}")
            rows.append(("states", i, rec, args.arm or _record_arm(rec)))
        if args.limit:
            rows = rows[: args.limit]
        return rows
    index = 0
    if not args.no_wall and Path(DEFAULT_WALL).is_file():
        for i, rec in enumerate(iter_windows(guard(DEFAULT_WALL))):
            if i % max(1, args.every) != 0:
                continue
            if refused_seed(rec.get("seed"), rec.get("sumo_seed")):
                raise SystemExit(f"refusing evaluation seed in wall state {i}")
            rows.append(("wall", index, rec, _record_arm(rec)))
            index += 1
    if not args.no_delay and Path(DEFAULT_DELAY).is_file():
        for i, rec in enumerate(iter_windows(guard(DEFAULT_DELAY))):
            if i >= args.delay_limit:
                break
            if refused_seed(rec.get("seed"), rec.get("sumo_seed")):
                raise SystemExit(f"refusing evaluation seed in delay state {i}")
            rows.append(("delay", index, rec, _record_arm(rec)))
            index += 1
    if not rows:
        raise SystemExit("no states found; pass --states or enable the default sources")
    return rows


def _horizon_for(rec: dict, arm: str, requested: float) -> float:
    recorded = (rec.get("recorded") or {}).get(arm) or []
    if isinstance(recorded, list) and recorded:
        span = float(recorded[-1].get("time_s", 0.0)) - float(rec.get("t0_s", 0.0))
        if span > DT_S:
            return min(float(requested), span)
    return float(requested)


# --------------------------------------------------------------------------- rollout
def parse_ckpt(spec: str):
    name, rest = spec.split("=", 1)
    mode = "AS_IS"
    if "@" in rest and rest.rsplit("@", 1)[1].upper() in ("ON", "OFF", "WAIT", "YIELD", "AS_IS"):
        rest, mode = rest.rsplit("@", 1)
        mode = mode.upper()
    return name, guard(rest), mode


def init(specs, entries: str, device: str) -> None:
    sys.path.insert(0, "tools")
    import torch

    import wm_eval_common as wc

    torch.set_num_threads(1)
    _WORKER["p"] = {}
    for name, path, mode in specs:
        p, cap = wc.load_predictor(wc.guard(path), mode, device=device, hdv_exits="sumo",
                                   entries=entries)
        _WORKER["p"][name] = (p, cap)


def rollout_arrays(p, cap, rec: dict, arm: str, horizon_s: float, seed: int):
    """Roll one start -> WM/SUMO ``[T+1, N]`` arrays of the t0 roster (or ``None``)."""
    from wms.eval.rollout_inputs import recorded_teacher, t0_vehicles

    n_steps = max(1, int(round(float(horizon_s) / DT_S)))
    t0, ids, kin, is_cav = t0_vehicles(rec)
    teacher, present_seq, _ = recorded_teacher(rec, t0, ids, kin, n_steps, arm,
                                               p.backend.tools)
    if teacher is None:
        return None
    if getattr(p, "flags", None) is not None:
        p.flags = dataclasses.replace(p.flags, lane_seed=int(seed))
    p.predict_states(rec, {**rec.get("cav_controllers", {}), "arm": arm}, horizon_s, seed)
    raw = cap["last"]
    n = len(ids)
    tr = teacher[0].numpy()
    rp = present_seq[0].numpy().astype(bool)
    pos = np.concatenate([tr[:1, :, 0], raw["positions"][0, :, :n].numpy()], 0)
    lan = np.concatenate([tr[:1, :, 1], raw["lanes"][0, :, :n].numpy()], 0)
    msk = np.concatenate([rp[:1], raw["masks"][0, :, :n].numpy().astype(bool)], 0)
    if "speeds" in raw:
        vel = np.concatenate([tr[:1, :, 2], raw["speeds"][0, :, :n].numpy()], 0)
    else:
        vel = np.vstack([tr[:1, :, 2], np.diff(pos, axis=0) / DT_S])
    return {"ids": list(ids), "kin": kin, "present": rp[0], "is_cav": is_cav,
            "wm_pos": pos, "wm_lan": lan, "wm_msk": msk, "wm_vel": vel,
            "rec_pos": tr[:, :, 0], "rec_lan": tr[:, :, 1], "rec_msk": rp,
            "rec_vel": tr[:, :, 2]}


def ramp_indices(arrays) -> list[int]:
    return [k for k in range(len(arrays["ids"]))
            if not bool(arrays["is_cav"][k]) and int(arrays["kin"][k, 1:4].argmax()) == 2]


def sample_record(arrays, ramp: list[int], key: str, sample: int, seed: int) -> dict:
    """Per-sample per-car timing rows (WM and recorded SUMO)."""
    cars = []
    for k in ramp:
        wm_t = merge_time(arrays["wm_lan"], arrays["wm_msk"], k)
        sumo_t = merge_time(arrays["rec_lan"], arrays["rec_msk"], k)
        wm_event = math.isfinite(wm_t)
        sumo_event = math.isfinite(sumo_t)
        cars.append({
            "id": arrays["ids"][k],
            "gap_m": t0_lane1_gap(arrays["kin"], arrays["present"], k),
            "wm_t": wm_t if wm_event else None,
            "sumo_t": sumo_t if sumo_event else None,
            "wm_duration_s": wm_t if wm_event else present_end(arrays["wm_msk"], k),
            "sumo_duration_s": (sumo_t if sumo_event
                                else present_end(arrays["rec_msk"], k)),
            "wm_event": wm_event, "sumo_event": sumo_event,
            "wm_pos": merge_position(arrays["wm_pos"], arrays["wm_lan"],
                                     arrays["wm_msk"], k),
            "sumo_pos": merge_position(arrays["rec_pos"], arrays["rec_lan"],
                                       arrays["rec_msk"], k),
        })
    return {"key": key, "sample": sample, "seed": seed, "cars": cars,
            "zone_wm": zone_stats(arrays["wm_pos"], arrays["wm_vel"], arrays["wm_msk"],
                                  arrays["wm_lan"]),
            "zone_sumo": zone_stats(arrays["rec_pos"], arrays["rec_vel"], arrays["rec_msk"],
                                    arrays["rec_lan"])}


def work(payload) -> dict:
    """Roll one start for every variant and sample (process-pool entry point)."""
    key, source, rec, arm, requested_horizon, k, base_seed, state_no = payload
    horizon_s = _horizon_for(rec, arm, requested_horizon)
    out = {"key": key, "source": source, "arm": arm, "horizon_s": horizon_s, "variants": {}}
    for name, (p, cap) in _WORKER["p"].items():
        rows = []
        for sample in range(int(k)):
            seed = int(base_seed) + int(state_no) * 1000 + sample
            arrays = rollout_arrays(p, cap, rec, arm, horizon_s, seed)
            if arrays is None:
                rows.append({"key": key, "sample": sample, "seed": seed, "error": "no arm"})
                continue
            rows.append(sample_record(arrays, ramp_indices(arrays), key, sample, seed))
        out["variants"][name] = rows
    return out


# --------------------------------------------------------------------------- aggregation
def aggregate_variant(rows: list[dict], k: int, horizon_s: float) -> dict:
    """Summarise per-state samples of one variant (KM, timing, gaps, zones, probabilities)."""
    scored = [r for r in rows if r.get("cars")]
    unscorable = [r["key"] for r in rows if not r.get("cars")]
    wm_dur, wm_ev, sumo_dur, sumo_ev = [], [], [], []
    matched, wm_pos, sumo_pos, gaps = [], [], [], []
    zone_wm, zone_sumo = [], []
    by_gap = {b: {"wm_dur": [], "wm_ev": [], "sumo_dur": [], "sumo_ev": [],
                  "pair": []} for b in GAP_BUCKETS}
    seen_sumo = set()
    states_acc: dict[str, dict] = {}
    for row in scored:
        car_probs = []
        zone_wm.append(row["zone_wm"])
        zone_sumo.append(row["zone_sumo"])
        for car in row["cars"]:
            bucket = gap_bucket(car["gap_m"])
            block = by_gap[bucket]
            gaps.append(car["gap_m"])
            wm_dur.append(car["wm_duration_s"])
            wm_ev.append(car["wm_event"])
            block["wm_dur"].append(car["wm_duration_s"])
            block["wm_ev"].append(car["wm_event"])
            if car["wm_event"]:
                wm_pos.append(car["wm_pos"])
            if car["sumo_event"]:
                sumo_pos.append(car["sumo_pos"])
            if car["wm_event"] and car["sumo_event"]:
                pair = (car["wm_t"], car["sumo_t"])
                matched.append(pair)
                block["pair"].append(pair)
            dedupe = (row["key"], car["id"])
            if dedupe not in seen_sumo:
                seen_sumo.add(dedupe)
                sumo_dur.append(car["sumo_duration_s"])
                sumo_ev.append(car["sumo_event"])
                block["sumo_dur"].append(car["sumo_duration_s"])
                block["sumo_ev"].append(car["sumo_event"])
            car_probs.append(1.0 if car["wm_event"] else 0.0)
        acc = states_acc.setdefault(row["key"], {"samples": 0, "cars": {}, "all": []})
        acc["samples"] += 1
        acc["all"].append(1.0 if all(car_probs) else 0.0)
        for car in row["cars"]:
            acc["cars"].setdefault(car["id"], []).append(1.0 if car["wm_event"] else 0.0)
    per_state = []
    for key, acc in sorted(states_acc.items()):
        car_prob = {cid: sum(values) / len(values) for cid, values in acc["cars"].items()}
        merged = sum(car_prob.values()) / len(car_prob) if car_prob else None
        per_state.append({
            "key": key, "samples": acc["samples"], "ramp_cars": len(car_prob),
            "merge_probability": merged,
            "all_merged_probability": sum(acc["all"]) / len(acc["all"]),
            "car_merge_probability": car_prob,
        })
    state_probs = [p["merge_probability"] for p in per_state]
    all_merged_probs = [p["all_merged_probability"] for p in per_state]
    stuck_probs = [None if p["merge_probability"] is None
                   else 1.0 - p["merge_probability"] for p in per_state]
    return {
        "states": len({row["key"] for row in rows}),
        "scored_states": len({row["key"] for row in scored}),
        "unscorable_states": sorted(set(unscorable)),
        "samples_per_state": int(k),
        "horizon_s": float(horizon_s),
        "wm": {
            "merge_fraction": (sum(wm_ev) / len(wm_ev)) if wm_ev else None,
            "stuck_fraction": (1.0 - sum(wm_ev) / len(wm_ev)) if wm_ev else None,
            "km": km_summary(wm_dur, wm_ev),
            "merge_position_median_m": _median(wm_pos),
        },
        "sumo": {
            "merge_fraction": (sum(sumo_ev) / len(sumo_ev)) if sumo_ev else None,
            "stuck_fraction": (1.0 - sum(sumo_ev) / len(sumo_ev)) if sumo_ev else None,
            "km": km_summary(sumo_dur, sumo_ev),
            "merge_position_median_m": _median(sumo_pos),
        },
        "timing": timing_pairs(matched),
        "merge_position_delta_median_m": (
            None if _median(wm_pos) is None or _median(sumo_pos) is None
            else _median(wm_pos) - _median(sumo_pos)),
        "gap_m_median": _median(gaps),
        "zone": {
            "wm": {key: _median([z[key] for z in zone_wm]) for key in
                   ("occupancy", "speed_mps", "spillback_veh", "spillback_step_fraction")},
            "sumo": {key: _median([z[key] for z in zone_sumo]) for key in
                     ("occupancy", "speed_mps", "spillback_veh", "spillback_step_fraction")},
        },
        "per_state_probability": probability_summary(state_probs),
        "per_state_all_merged_probability": probability_summary(all_merged_probs),
        "per_state_stuck_probability": probability_summary(stuck_probs),
        "by_gap": {
            bucket: {
                "cars": len(block["wm_dur"]),
                "wm": {"merge_fraction": (sum(block["wm_ev"]) / len(block["wm_ev"]))
                       if block["wm_ev"] else None,
                       "km": km_summary(block["wm_dur"], block["wm_ev"])},
                "sumo": {"merge_fraction": (sum(block["sumo_ev"]) / len(block["sumo_ev"]))
                         if block["sumo_ev"] else None,
                         "km": km_summary(block["sumo_dur"], block["sumo_ev"])},
                "timing": timing_pairs(block["pair"]),
            } for bucket, block in by_gap.items()
        },
        "per_state": per_state,
    }


# --------------------------------------------------------------------------- output
def _variant_line(name: str, block: dict) -> str:
    wm, sumo, tm = block["wm"], block["sumo"], block["timing"]
    zone = block["zone"]
    return (f"| {name} | {block['scored_states']}/{block['states']} | "
            f"{wm['km']['n']} | {_pct(wm['merge_fraction'])} | {_pct(sumo['merge_fraction'])} | "
            f"{_h(wm['km']['median_s'])} | {_h(sumo['km']['median_s'])} | "
            f"{_h(wm['km']['p90_s'])} | {_h(sumo['km']['p90_s'])} | "
            f"{tm['matched']} | {_h(tm['median_signed_s'])} | {_h(tm['median_abs_s'])} | "
            f"{tm['early_gt_2s']}/{tm['late_gt_2s']} | {_pct(wm['stuck_fraction'])} | "
            f"{_pct(sumo['stuck_fraction'])} | {_h(wm['merge_position_median_m'])} / "
            f"{_h(sumo['merge_position_median_m'])} | "
            f"{_h(zone['wm']['occupancy'])}/{_h(zone['sumo']['occupancy'])} | "
            f"{_h(zone['wm']['speed_mps'])}/{_h(zone['sumo']['speed_mps'])} | "
            f"{_h(zone['wm']['spillback_step_fraction'])}/"
            f"{_h(zone['sumo']['spillback_step_fraction'])} |")


def _pct(value) -> str:
    return "n/a" if value is None else f"{100.0 * float(value):.0f}%"


def markdown(doc: dict) -> str:
    lines = [
        "# WM merge timing vs recorded SUMO",
        "",
        f"Horizon {doc['horizon_s']:.1f} s, k={doc['k']}, states={doc['n_states']} "
        f"({doc['sources']}). Censored (unmerged) cars stay in the Kaplan-Meier risk set; "
        "waiting time is measured from t0.",
        "",
        "| variant | scored | ramp car-samples | WM merges | SUMO merges | WM med s | "
        "SUMO med s | WM p90 s | SUMO p90 s | matched | med signed s | med |err| s | "
        "early>2/late>2 | WM stuck | SUMO stuck | merge pos WM/SUMO m | "
        "occupancy WM/SUMO | speed m/s WM/SUMO | spillback frac WM/SUMO |",
        "|" + "---|" * 19,
    ]
    for name, block in doc["variants"].items():
        lines.append(_variant_line(name, block))
    for name, block in doc["variants"].items():
        lines += ["", f"## {name}: gap-availability strata", "",
                  "| gap | cars | WM merged | SUMO merged | WM med s | SUMO med s | "
                  "matched | med |err| s |", "|" + "---|" * 8]
        for bucket in GAP_BUCKETS:
            part = block["by_gap"][bucket]
            if not part["cars"]:
                continue
            lines.append(
                f"| {bucket} | {part['cars']} | {_pct(part['wm']['merge_fraction'])} | "
                f"{_pct(part['sumo']['merge_fraction'])} | "
                f"{_h(part['wm']['km']['median_s'])} | {_h(part['sumo']['km']['median_s'])} | "
                f"{part['timing']['matched']} | {_h(part['timing']['median_abs_s'])} |")
        prob = block["per_state_probability"]
        lines += ["", f"Per-state merge probability (mean/median/min/max over states): "
                  f"{_pct(prob['mean'])} / {_pct(prob['median'])} / {_pct(prob['min'])} / "
                  f"{_pct(prob['max'])}; all-merged probability mean "
                  f"{_pct(block['per_state_all_merged_probability']['mean'])}; "
                  f"stuck probability mean {_pct(block['per_state_stuck_probability']['mean'])}."]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- CLI
def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", action="append", required=True,
                    help="NAME=path[@MODE] (repeatable); MODE is ON/OFF/WAIT/YIELD/AS_IS")
    ap.add_argument("--states", help="states/windows JSON (default: wall_same + delay set)")
    ap.add_argument("--arm", help="recorded arm override (default: record arm/wall/native)")
    ap.add_argument("--no-wall", action="store_true", help="skip the 72 wall_same starts")
    ap.add_argument("--no-delay", action="store_true", help="skip the delay starts")
    ap.add_argument("--delay-limit", type=int, default=DEFAULT_DELAY_LIMIT,
                    help="number of delay states from the default file (default 24)")
    ap.add_argument("--every", type=int, default=2, help="wall-state stride (default 2 = 72)")
    ap.add_argument("--limit", type=int, default=None, help="cap for --states records")
    ap.add_argument("--horizon", type=float, default=120.0)
    ap.add_argument("--k", type=int, default=1, help="samples per start with fixed seeds")
    ap.add_argument("--seed", type=int, default=0, help="base seed (per-state seeds derive)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--entries", choices=("off", "queue"), default="off")
    ap.add_argument("--out", required=True)
    ap.add_argument("--md")
    return ap


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.k < 1:
        raise SystemExit("--k must be >= 1")
    if args.workers < 1 or args.workers > 3:
        raise SystemExit("--workers must be in [1, 3] on the laptop")
    specs = [parse_ckpt(spec) for spec in args.ckpt]
    rows = load_sources(args)
    if any(refused_seed(args.seed + i * 1000 + sample)
           for i in range(len(rows)) for sample in range(args.k)):
        raise SystemExit("--seed would produce an evaluation seed (964000-964199)")
    tasks = [(f"{source}:{state_key(rec, i)}", source, rec, arm, float(args.horizon),
              int(args.k), int(args.seed), i)
             for i, (source, _n, rec, arm) in enumerate(rows)]
    t0 = time.time()
    results = []
    if args.workers <= 1:
        init(specs, args.entries, args.device)
        for task in tasks:
            results.append(work(task))
    else:
        from multiprocessing import Pool

        with Pool(args.workers, initializer=init,
                  initargs=(specs, args.entries, args.device)) as pool:
            for j, res in enumerate(pool.imap_unordered(work, tasks, chunksize=1)):
                results.append(res)
                if (j + 1) % 8 == 0:
                    print(f"{j + 1}/{len(tasks)} {time.time() - t0:.0f}s", flush=True)
    results.sort(key=lambda r: r["key"])
    variants = {}
    for name, _path, _mode in specs:
        rows_v = [row for res in results
                  for row in res["variants"].get(name, [])]
        variants[name] = aggregate_variant(
            rows_v, args.k, max(res["horizon_s"] for res in results))
    doc = {
        "schema": "c4_wm_merge_timing/1",
        "generated_unix": time.time(),
        "elapsed_s": time.time() - t0,
        "k": int(args.k), "seed": int(args.seed), "horizon_s": float(args.horizon),
        "n_states": len(results),
        "sources": ", ".join(f"{src}={sum(1 for r in results if r['source'] == src)}"
                             for src in sorted({r["source"] for r in results})),
        "entries": args.entries,
        "variants": {name: {"checkpoint": path, "mode": mode, **block}
                     for (name, path, mode), block in
                     ((spec, variants[spec[0]]) for spec in specs)},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(doc, indent=2, allow_nan=False), encoding="utf-8")
    if args.md:
        Path(args.md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.md).write_text(markdown(doc), encoding="utf-8")
    print(markdown(doc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
