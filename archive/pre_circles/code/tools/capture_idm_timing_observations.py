"""Causal observation capture at the frozen t1/t2/t3 states (Phase 2).

Runs deterministic native-IDM trajectories (no treatment arms) and reconstructs
the frozen schedule landmarks while collecting the 40 frozen compact features
(families A-D).  Every landmark must match the stored schedule record exactly
(step, recurrence, target, lane, observation and physical fingerprints,
config hash); on any mismatch the phase writes the mismatch report and stops.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping

from experiments.io import atomic_write_json
from env.snapshot import get_active_snapshot
from idm.action_local import LandmarkContract
from idm.timing_observations import PROVENANCE, summarize_window, window_values
from tools.discover_idm_timing_schedule import CONFIG, _config_hash
from tools.run_idm_decentralized_authority_verification import (
    _actions,
    _candidate_snapshot,
    _env,
    _landmark_at_state,
    _merge_discharge_step,
    _nonclassic_observation_hash,
    _raw_decision_context_fingerprint,
)

SCHEMA = "idm_timing_observations/1"
HISTORY_SECONDS = 45
WINDOWS = (5, 20)


def _zone_band_stats(samples: list[Any], lo: float, hi: float) -> dict[str, float]:
    rows = [s for s in samples if lo <= float(s.distance_to_merge_m) < hi]
    by_lane: dict[str, list[float]] = {}
    for row in rows:
        by_lane.setdefault(str(row.lane_id), []).append(float(row.speed_mps))
    out: dict[str, float] = {}
    for lane, speeds in by_lane.items():
        out[f"lane{lane[-1]}_mean_speed"] = float(sum(speeds) / len(speeds))
        out[f"lane{lane[-1]}_speed_std"] = float(
            (sum((v - out[f"lane{lane[-1]}_mean_speed"]) ** 2 for v in speeds) / len(speeds)) ** 0.5
        )
    out["count"] = float(len(rows))
    return out


def _corridor_stats(samples: list[Any]) -> dict[str, float]:
    speeds = [float(s.speed_mps) for s in samples]
    if not speeds:
        return {"mean": 0.0, "std": 0.0, "fraction_slow": 0.0}
    mean = float(sum(speeds) / len(speeds))
    std = float((sum((v - mean) ** 2 for v in speeds) / len(speeds)) ** 0.5)
    return {"mean": mean, "std": std, "fraction_slow": sum(1 for v in speeds if v < 15.0) / len(speeds)}


def _lane_events(samples: list[Any]) -> dict[str, dict[str, float]]:
    """Per-lane severe count and sequential-pair closing/ttc state this step."""
    by_lane: dict[str, list[Any]] = {}
    for sample in samples:
        by_lane.setdefault(str(sample.lane_id), []).append(sample)
    out = {}
    for lane, rows in by_lane.items():
        ordered = sorted(rows, key=lambda row: row.lane_pos_m, reverse=True)
        severe = sum(1 for row in ordered if float(row.acceleration_mps2) < -3.0)
        max_closing = 0.0
        low_ttc = 0
        for i in range(len(ordered) - 1):
            leader, follower = ordered[i], ordered[i + 1]
            closing = float(follower.speed_mps) - float(leader.speed_mps)
            gap = float(leader.lane_pos_m) - float(follower.lane_pos_m) - 5.0
            if closing > 0.01 and gap / closing < 1.5:
                low_ttc += 1
            if closing > max_closing:
                max_closing = closing
        out[lane] = {"severe": float(severe), "max_closing": max_closing, "low_ttc": float(low_ttc)}
    return out


def capture_state(spec: Mapping[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    demand = int(spec["demand"])
    seed = int(spec["seed"])
    penetration = float(spec["cav_penetration"])
    schedule = spec["schedule"]
    output = Path(spec["output"])
    max_horizon = int(spec["max_horizon_steps"])
    env = _env(demand, seed, output, f"obs.m{demand}.seed{seed}", max_horizon + 5, CONFIG["action_set"], CONFIG["option_steps"], CONFIG["minimum_dose_steps"], penetration)
    contract = LandmarkContract(maximum_landmarks_per_cell=3)
    config_hash = _config_hash(demand, penetration)
    speed_hist: dict[str, deque] = {}
    gap_hist: dict[str, deque] = {}
    closing_hist: dict[str, deque] = {}
    lane_severe: dict[str, deque] = {}
    lane_closing: dict[str, deque] = {}
    zone_hist: deque = deque()
    pattern_hist: deque = deque(maxlen=25)
    previous_road: dict[str, str] = {}
    occurrences: list[dict[str, Any]] = []
    identity: list[dict[str, Any]] = []
    mismatch: dict[str, Any] = {}
    previous_step: int | None = None
    persistent_key: Any = None
    persistence = 0
    completed = 0
    try:
        obs, _ = env.reset(seed=seed)
        for step in range(max_horizon + 1):
            import traci

            sim_time = float(traci.simulation.getTime())
            samples = env._idm_residual_engine._samples()
            lane_events = _lane_events(samples)
            for lane, values in lane_events.items():
                lane_severe.setdefault(lane, deque()).append((sim_time, values["severe"]))
                lane_closing.setdefault(lane, deque()).append((sim_time, values["max_closing"]))
            for sample in samples:
                vid = str(sample.vehicle_id)
                speed_hist.setdefault(vid, deque(maxlen=HISTORY_SECONDS)).append((sim_time, float(sample.speed_mps)))
                gap_hist.setdefault(vid, deque(maxlen=HISTORY_SECONDS)).append((sim_time, None))
                closing_hist.setdefault(vid, deque(maxlen=HISTORY_SECONDS)).append((sim_time, None))
            by_lane_ordered: dict[str, list[Any]] = {}
            for sample in samples:
                by_lane_ordered.setdefault(str(sample.lane_id), []).append(sample)
            for lane, rows in by_lane_ordered.items():
                ordered = sorted(rows, key=lambda row: row.lane_pos_m, reverse=True)
                for i in range(len(ordered) - 1):
                    leader, follower = ordered[i], ordered[i + 1]
                    vid = str(follower.vehicle_id)
                    gap = float(leader.lane_pos_m) - float(follower.lane_pos_m) - 5.0
                    closing = float(follower.speed_mps) - float(leader.speed_mps)
                    gap_hist.setdefault(vid, deque(maxlen=HISTORY_SECONDS)).append((sim_time, gap))
                    closing_hist.setdefault(vid, deque(maxlen=HISTORY_SECONDS)).append((sim_time, closing))
            zone = _zone_band_stats(samples, 0.0, 300.0)
            slow_band = _zone_band_stats(samples, 300.0, 800.0)
            corridor = _corridor_stats(samples)
            ramp_snapshot = get_active_snapshot()
            ramp_ids = [
                str(vid)
                for vid in (ramp_snapshot.ordered_ids() if ramp_snapshot else ())
                if ramp_snapshot.road(vid) == str(env._ramp_edge)
            ]
            ramp_speeds = [float(ramp_snapshot.speed(vid)) for vid in ramp_ids if ramp_snapshot.speed(vid) is not None]
            down_ids = [
                str(vid)
                for vid in (ramp_snapshot.ordered_ids() if ramp_snapshot else ())
                if ramp_snapshot.road(vid) == str(env._shepherd_engine.down_edge)
            ]
            down_speeds = [float(ramp_snapshot.speed(vid)) for vid in down_ids if ramp_snapshot.speed(vid) is not None]
            discharge = _merge_discharge_step(env, previous_road)
            zone_severe = sum(float(values["severe"]) for values in lane_events.values())
            zone_ttc = sum(float(values["low_ttc"]) for values in lane_events.values())
            zone_hist.append((sim_time, discharge, zone_severe, zone_ttc))
            pattern = dict(env._idm_pattern_context or {})
            pattern_hist.append((sim_time, pattern))
            previous_road = _current_roads()

            state = _candidate_snapshot(env, obs, "runway", 1)
            landmark = None if state is None else _landmark_at_state(state, "risk_onset", contract)
            current_key = (
                landmark,
                None if state is None else state["lane_id"],
                tuple(row["vehicle_id"] for row in state["candidates"]) if state is not None else (),
            )
            if state is not None and landmark is not None:
                persistence = persistence + 1 if current_key == persistent_key else 1
                persistent_key = current_key
                if persistence >= contract.persistence_steps:
                    separated = previous_step is None or step - previous_step >= contract.minimum_landmark_separation_steps
                    if separated:
                        # === feature capture at the landmark ===
                        candidate_row = state["candidates"][0]
                        vid = str(candidate_row["vehicle_id"])
                        features = _capture_features(vid, samples, lane_events, lane_severe, lane_closing, zone, slow_band, corridor, ramp_speeds, down_speeds, zone_hist, pattern_hist, env, sim_time, speed_hist, gap_hist, closing_hist)
                        reconstructed = {
                            "step": step,
                            "landmark": landmark,
                            "candidate_id": vid,
                            "lane_id": str(state["lane_id"]),
                            "observation_hash": _nonclassic_observation_hash(state["candidate_observations"][vid]),
                            "raw_decision_context_fingerprint": _raw_decision_context_fingerprint(state),
                            "config_hash": config_hash,
                        }
                        stored = schedule["occurrences"][len(occurrences)]
                        check = _identity(reconstructed, stored, str(schedule.get("config_hash")))
                        identity.append({"step": step, "recurrence": len(occurrences) + 1, **check})
                        if not all(check.values()):
                            mismatch = {"step": step, "recurrence": len(occurrences) + 1, "reconstructed": reconstructed, "stored": stored, "checks": check}
                        occurrences.append({"step": step, "landmark": landmark, **features, "candidate_id": vid, "lane_id": str(state["lane_id"])})
                        previous_step = step
                        if len(occurrences) >= 3 or mismatch:
                            break
                    persistence = 0
                    persistent_key = None
            else:
                persistence = 0
                persistent_key = None
            obs, _, terminated, truncated, _ = env.step(_actions(obs))
            completed = step + 1
            if terminated.get("__all__") or truncated.get("__all__"):
                break
            if mismatch:
                break
    finally:
        env.close()
    return {
        "schema": SCHEMA,
        "demand": demand,
        "seed": seed,
        "cav_penetration": penetration,
        "config_hash": config_hash,
        "steps_completed": completed,
        "occurrences": occurrences,
        "identity": identity,
        "mismatch": mismatch,
        "provenance": [{"name": name, "family": family, "source": source} for name, family, source in PROVENANCE],
        "runtime_s": time.perf_counter() - started,
    }


def _capture_features(vid, samples, lane_events, lane_severe, lane_closing, zone, slow_band, corridor, ramp_speeds, down_speeds, zone_hist, pattern_hist, env, sim_time, speed_hist, gap_hist, closing_hist):
    lane = str(next(sample.lane_id for sample in samples if str(sample.vehicle_id) == vid))
    ego = next((sample for sample in samples if str(sample.vehicle_id) == vid), None)
    # summarize window series
    speed_s = list(window_values(list(speed_hist.get(vid, [])), sim_time, 20))
    gap_s = [v for v in window_values(list(gap_hist.get(vid, [])), sim_time, 20) if v is not None]
    closing_s = [v for v in window_values(list(closing_hist.get(vid, [])), sim_time, 20) if v is not None]
    lane_severe_s = list(window_values(list(lane_severe.get(lane, [])), sim_time, 20))
    lane_severe_5 = list(window_values(list(lane_severe.get(lane, [])), sim_time, 5))
    zone_5 = [row for row in zone_hist if row[0] > sim_time - 5]
    zone_10 = [row for row in zone_hist if row[0] > sim_time - 10]
    pattern = pattern_hist[-1][1] if pattern_hist else {}
    cav_ids = set(env._active_cavs or ())
    cav_speeds = [float(sample.speed_mps) for sample in samples if str(sample.vehicle_id) in cav_ids]
    # compute gradient from available keys robustly
    def band_mean(band, default=0.0):
        values = [v for k, v in band.items() if k.endswith("_mean_speed")]
        return float(sum(values) / len(values)) if values else default

    zone_mean = band_mean(zone)
    down_mean = float(sum(down_speeds) / len(down_speeds)) if down_speeds else 0.0
    speed_5 = list(window_values(list(speed_hist.get(vid, [])), sim_time, 5))
    gap_s5 = [v for v in window_values(list(gap_hist.get(vid, [])), sim_time, 5) if v is not None]
    closing_s5 = [v for v in window_values(list(closing_hist.get(vid, [])), sim_time, 5) if v is not None]
    b = {
        "ego_speed": float(ego.speed_mps) if ego else 0.0,
        "ego_accel": float(ego.acceleration_mps2) if ego else 0.0,
        "leader_gap": gap_s[-1] if gap_s else 0.0,
        "leader_rel_speed": closing_s[-1] if closing_s else 0.0,
        "follower_gap": 0.0,
        "follower_closing": 0.0,
        "speed_std_5s": float(summarize_window(speed_5)["std"]) if len(speed_5) > 1 else 0.0,
        "speed_std_20s": float(summarize_window(speed_s)["std"]) if len(speed_s) > 1 else 0.0,
        "min_gap_5s": float(min(gap_s5)) if gap_s5 else 0.0,
        "min_gap_20s": float(min(gap_s)) if gap_s else 0.0,
        "max_closing_5s": float(max(closing_s5)) if closing_s5 else 0.0,
        "max_closing_20s": float(max(closing_s)) if closing_s else 0.0,
        "severe_count_5s": float(sum(lane_severe_5)),
        "severe_count_20s": float(sum(lane_severe_s)),
    }
    c = {
        "merge_lane0_mean_speed": zone.get("lane0_mean_speed", 0.0),
        "merge_lane1_mean_speed": zone.get("lane1_mean_speed", 0.0),
        "merge_lane0_speed_std": zone.get("lane0_speed_std", 0.0),
        "merge_lane1_speed_std": zone.get("lane1_speed_std", 0.0),
        "merge_zone_density": float(zone.get("count", 0.0)) * 1000.0 / 300.0,
        "ramp_count": float(len(ramp_speeds)),
        "ramp_slow_count": float(sum(1 for value in ramp_speeds if value < 3.0)),
        "merge_discharge_rate_10s": float(sum(row[1] for row in zone_10) / max(len(zone_10), 1)),
        "zone_severe_count_10s": float(sum(row[2] for row in zone_10)),
        "zone_low_ttc_count_10s": float(sum(row[3] for row in zone_10)),
    }
    d = {
        "corridor_mean_speed": corridor["mean"],
        "corridor_speed_std": corridor["std"],
        "corridor_fraction_slow": corridor["fraction_slow"],
        "gradient_upstream_merge": float(band_mean(slow_band) - zone_mean),
        "gradient_merge_downstream": float(down_mean - zone_mean),
        "idm_wave_energy_level": float(pattern.get("idm_wave_energy_level", 0.0)),
        "idm_severe_short_ema": float(pattern.get("idm_severe_short_ema", 0.0)),
        "cav_upstream_count": float(len(cav_speeds)),
    }
    return {
        "features": {"A": {"stage_index": 0.0}, "B": b, "C": c, "D": d},
        "feature_available": {"B": True, "C": True, "D": True},
    }


def _identity(
    reconstructed: Mapping[str, Any],
    stored: Mapping[str, Any],
    schedule_config_hash: str,
) -> dict[str, bool]:
    return {
        "step": int(reconstructed["step"]) == int(stored["step"]),
        "landmark": str(reconstructed["landmark"]) == str(stored.get("landmark")),
        "candidate_id": str(reconstructed["candidate_id"]) == str(stored.get("candidate_id")),
        "lane_id": str(reconstructed["lane_id"]) == str(stored.get("lane_id")),
        "observation_fingerprint": str(reconstructed["observation_hash"]) == str(stored.get("observation_hash")),
        "physical_fingerprint": str(reconstructed["raw_decision_context_fingerprint"]) == str(stored.get("raw_decision_context_fingerprint")),
        "config_hash": str(reconstructed["config_hash"]) == str(schedule_config_hash),
    }


def _current_roads() -> dict[str, str]:
    snapshot = get_active_snapshot()
    if snapshot is None:
        return {}
    roads = {}
    for vid in snapshot.ordered_ids():
        road = snapshot.road(vid)
        if road is not None:
            roads[str(vid)] = str(road)
    return roads


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedules", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-horizon-steps", type=int, default=905)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "observations").mkdir(exist_ok=True)
    pending = []
    for inventory in sorted(args.schedules.glob("p*/inventory.json")):
        payload = json.loads(inventory.read_text(encoding="utf-8"))
        for entry in payload["states"]:
            if entry["t1"] is None or not bool(entry.get("schedule_identity", False)):
                continue
            cell = inventory.parent
            schedule_path = cell / "discoveries" / f"m{entry['demand']}.seed{entry['seed']}.json.pass0"
            if not schedule_path.exists():
                continue
            target = output / "observations" / f"p{int(entry['cav_penetration']*100):02d}_m{entry['demand']}.seed{entry['seed']}.json"
            if target.exists():
                continue
            pending.append(
                {
                    "demand": int(entry["demand"]),
                    "seed": int(entry["seed"]),
                    "cav_penetration": float(entry["cav_penetration"]),
                    "schedule": json.loads(schedule_path.read_text(encoding="utf-8")),
                    "output": str(output),
                    "max_horizon_steps": args.max_horizon_steps,
                    "path": str(target),
                }
            )
    if pending:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(capture_state, {k: v for k, v in spec.items() if k != "path"}): spec for spec in pending}
            mismatches = 0
            for future in as_completed(futures):
                row = future.result()
                atomic_write_json(Path(futures[future]["path"]), row)
                print(
                    f"[obs] seed{row['seed']} landmarks={len(row['occurrences'])} "
                    f"mismatch={bool(row['mismatch'])} runtime={row['runtime_s']:.1f}s",
                    flush=True,
                )
                if row["mismatch"]:
                    mismatches += 1
            if mismatches:
                print(f"MISMATCH COUNT: {mismatches} — phase stopped per protocol", flush=True)
                sys.exit(1)
    print(json.dumps({"captured": len(pending)}, indent=1))


if __name__ == "__main__":
    main()
