"""Exposed, no-training compatibility check of a pinned CIRCLES export.

Not reproduction of the publication: 1 s control / 0.5 s SUMO, same-lane
100 m sensing, and explicit native fallback. Publication provenance is pending.
"""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import hashlib
import json
import msvcrt
from pathlib import Path
import random
import shutil
import time
import xml.etree.ElementTree as ET

import numpy as np
import onnxruntime as ort
import traci

from experiments.idm import deterministic_sumo  # noqa: F401
from experiments.idm.train_native_damping import atomic_json, exclusive_run
from env.sumo_utils import start_sumo, close_sumo
from env.damping_v2_rollout import RequestLedger, observation_map
from env.native_damping_sumo import SumoLocalSensor
from idm.damping_scenario import build_scenario, request_schedule, EPISODE_SECONDS

ROOT = Path("runs/probe/published_idm_fit_check_2026-09-06").resolve()
CASES = ((911001, 2000), (911003, 2200))
RATES = (0.05, 0.10, 0.20, 0.30)
MODEL_HASH = "762fde0763c693c0e88f41b98b22fc0a11135a312b78c4c567e6211a3035ccee"


class LaunchLock:
    """Lock an existing byte, never append while another process owns it."""

    def __enter__(self):
        self.handle = open(ROOT / "launch.lock", "r+b")
        msvcrt.locking(self.handle.fileno(), msvcrt.LK_LOCK, 1)
        return self

    def __exit__(self, *_):
        try:
            self.handle.seek(0)
            msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
        finally:
            self.handle.close()


# Isolated correction of the existing append-before-lock Windows launcher.
deterministic_sumo._InterprocessLaunchLock = LaunchLock


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def designation(seed, vehicle_id, rate):
    return random.Random(f"damping_cav_v1:{seed}:{vehicle_id}").random() < rate


def load_model():
    path = ROOT / "candidate.onnx"
    if digest(path) != MODEL_HASH:
        raise ValueError("candidate identity changed")
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def inference(model, rows):
    # Upstream runtime evaluates one vehicle per call. Preserve that shape;
    # the exported output has two symbolic batch axes.
    return np.asarray(
        [
            float(
                model.run(None, {model.get_inputs()[0].name: np.asarray([row], np.float32)})[
                    0
                ].item()
            )
            for row in rows
        ]
    )


def target(frame, raw, mode, min_gap):
    if not np.isfinite(raw):
        return None
    if frame.front is None or min(frame.front_coverage_m, frame.rear_coverage_m) <= 0:
        return None
    if mode == "workspace":
        lo = max(-1.0, frame.acceleration_mps2 - 0.5, -frame.speed_mps)
        hi = min(0.5, frame.acceleration_mps2 + 0.5, frame.allowed_speed_mps - frame.speed_mps)
        hi = min(hi, frame.front.relative_speed_mps + max(0, frame.front.gap_m - min_gap) / 3)
        if frame.rear is not None:
            lo = max(lo, frame.rear.relative_speed_mps - frame.rear.gap_m / 4)
    elif mode == "direct":
        # Separate diagnostic only: approximate paper's acceleration envelope,
        # without workspace jerk/front/rear projection. SUMO speedMode31 stays on.
        lo, hi = max(-3.0, -frame.speed_mps), min(1.5, frame.allowed_speed_mps - frame.speed_mps)
    else:
        raise ValueError(mode)
    return None if lo > hi else float(np.clip(raw, lo, hi))


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    if not (ROOT / "launch.lock").exists():
        (ROOT / "launch.lock").write_bytes(b"0")
    model = load_model()
    samples = [[5, 5, 20], [30, 10, 50], [5, 20, 30], [20, 20, 35]]
    outputs = inference(model, samples)
    if not np.isfinite(outputs).all():
        raise ValueError("nonfinite model output")
    jobs, scenes = [], []
    for seed, loading in CASES:
        scene, requests, identity = build_scenario(
            ROOT / f"scenario_{seed}", seed, loading, role="development"
        )
        if any(r.is_cav != designation(seed, r.vehicle_id, 0.2) for r in requests):
            raise ValueError("p20 differs from original assignment")
        types = ET.parse(scene.parent / "requests.rou.xml").getroot().findall("vType")
        if not types or any(x.get("carFollowModel") != "IDM" for x in types):
            raise ValueError("non-IDM traffic")
        scenes.append(identity)
        jobs.append((seed, loading, 0.0, "native"))
        jobs.extend((seed, loading, p, "workspace") for p in RATES)
        jobs.append((seed, loading, 0.2, "direct"))
    manifest = {
        "schema": "published_idm_fit/1",
        "model": json.loads((ROOT / "model_audit.json").read_text()),
        "runner_sha256": digest(__file__),
        "scenes": scenes,
        "jobs": jobs,
        "max_rollouts": len(jobs),
        "max_simulated_seconds": len(jobs) * EPISODE_SECONDS,
        "sensor": "existing same-lane nearest-neighbor 100m; missing leader returns native",
        "input": ["ego_speed_mps", "leader_speed_mps", "bumper_gap_m"],
        "normalization": "raw SI inputs; division by [40,40,100] embedded in ONNX graph",
        "output": "absolute acceleration confirmed by ONNX graph, not an IDM residual",
        "invalid_output": "release override and return to native IDM; singleton inference only",
        "timing": "1s command,0.5s SUMO; differs from upstream ROS deployment",
        "workspace": "existing -1/+0.5 acceleration,0.5 jerk,front/rear bounds,speedMode31",
        "direct": "p20 diagnostic -3/+1.5 acceleration, speedMode31; NOT workspace-bound compliant",
        "selection": "first RL-labelled candidate listed in earlier audit; no outcome selection",
        "acceptance": False,
        "training": False,
        "source_seeds": "already exposed development",
        "provenance_limit": "RL-labelled export; exact published/field checkpoint identity unverified",
        "inference_examples": {str(x): float(y) for x, y in zip(samples, outputs)},
    }
    manifest = json.loads(json.dumps(manifest))
    path = ROOT / "fit_manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("frozen check changed")
    atomic_json(path, manifest)
    return jobs


def run_job(job):
    seed, loading, rate, mode = job
    key = f"s{seed}_p{round(rate * 100):02}_{mode}"
    out = ROOT / "results" / f"{key}.json"
    out.parent.mkdir(exist_ok=True)
    if out.exists():
        return json.loads(out.read_text())
    started_path = out.with_suffix(".started.json")
    if started_path.exists():
        raise ValueError(f"unfinished allocation must be reconciled: {key}")
    atomic_json(started_path, {"job": job, "time": time.time()})
    scene = ROOT / f"scenario_{seed}" / "scenario.sumocfg"
    original = request_schedule(seed, loading, role="development")
    registration = json.loads((ROOT / "fit_manifest.json").read_text())
    identity = next(s for s in registration["scenes"] if s["seed"] == seed)
    if digest(__file__) != registration["runner_sha256"]:
        raise ValueError("runner changed after registration")
    requests = [replace(r, is_cav=designation(seed, r.vehicle_id, rate)) for r in original]
    # The XML is independent of CAV designation; no vehicle type is changed.
    cav_ids = {r.vehicle_id for r in requests if r.is_cav}
    model = None if mode == "native" else load_model()
    sumo = shutil.which("sumo")
    if not sumo:
        raise RuntimeError("SUMO not found")
    start = time.monotonic()
    start_sumo(
        sumo,
        str(scene),
        scene.parent,
        seed=seed,
        extra_args=[
            "--step-length",
            "0.5",
            "--end",
            "2401",
            "--no-step-log",
            "true",
            "--duration-log.disable",
            "true",
        ],
    )
    ledger = RequestLedger(requests)
    sensor = SumoLocalSensor(traci, lane_end_distance_to_merge=observation_map(traci))
    counts = Counter()
    owned, collided = set(), set()
    gap_cache = {}
    rows, input_rows, action_rows = [], [], []
    try:
        for step in range(EPISODE_SECONDS):
            now = float(traci.simulation.getTime())
            if abs(now - step) > 1e-6:
                raise ValueError("clock mismatch")
            live_cavs = cav_ids.intersection(traci.vehicle.getIDList())
            owned.intersection_update(live_cavs)
            frames = {}
            if model is not None:
                for cav in sorted(live_cavs):
                    frame = sensor.capture(cav, now_s=now)
                    counts["cav_decisions"] += 1
                    if (
                        frame.front is not None
                        and min(frame.front_coverage_m, frame.rear_coverage_m) > 0
                    ):
                        frames[cav] = frame
                    else:
                        counts["missing_or_uncovered"] += 1
                        if cav in owned:
                            traci.vehicle.setSpeed(cav, -1)
                            owned.remove(cav)
                            counts["release"] += 1
                inputs = [
                    [f.speed_mps, f.speed_mps + f.front.relative_speed_mps, f.front.gap_m]
                    for f in frames.values()
                ]
                raw_outputs = inference(model, inputs)
                for (cav, frame), x, raw in zip(frames.items(), inputs, raw_outputs):
                    if not np.isfinite(raw):
                        counts["invalid_output"] += 1
                    if cav not in gap_cache:
                        gap_cache[cav] = traci.vehicle.getMinGap(cav)
                    a = target(frame, raw, mode, gap_cache[cav])
                    input_rows.append([step, *x])
                    action_rows.append([raw, np.nan if a is None else a])
                    if a is None:
                        counts["infeasible"] += 1
                        if cav in owned:
                            traci.vehicle.setSpeed(cav, -1)
                            owned.remove(cav)
                            counts["release"] += 1
                    else:
                        if traci.vehicle.getSpeedMode(cav) != 31:
                            raise ValueError("native safety disabled")
                        traci.vehicle.slowDown(cav, max(0, frame.speed_mps + a), 1.0)
                        owned.add(cav)
                        counts["commands"] += 1
                        counts["output_modified"] += int(abs(a - raw) > 1e-6)
            braking = effort = ttc = collisions = teleports = 0.0
            prior_j, prior_w = ledger.system_vehicle_seconds, ledger.source_vehicle_seconds
            for _ in range(2):
                traci.simulationStep()
                live = traci.vehicle.getIDList()
                speeds = {v: traci.vehicle.getSpeed(v) for v in live}
                accelerations = np.asarray([traci.vehicle.getAcceleration(v) for v in live])
                braking += 0.5 * float(np.sum(accelerations < -3))
                effort += 0.5 * float(accelerations @ accelerations)
                for v in live:
                    leader = traci.vehicle.getLeader(v, 200.0)
                    if not leader or leader[0] not in speeds:
                        continue
                    closing = speeds[v] - speeds[leader[0]]
                    if closing > 1e-6:
                        if v not in gap_cache:
                            gap_cache[v] = traci.vehicle.getMinGap(v)
                        ttc += 0.5 * ((max(0, leader[1] + gap_cache[v]) / closing) < 3)
                ledger.advance(
                    float(traci.simulation.getTime()),
                    list(traci.simulation.getDepartedIDList()),
                    list(traci.simulation.getArrivedIDList()),
                )
                collision_now = set(traci.simulation.getCollidingVehiclesIDList())
                collisions += len(collision_now - collided)
                collided = collision_now
                teleports += traci.simulation.getStartingTeleportNumber()
            rows.append(
                [
                    braking,
                    effort,
                    ledger.system_vehicle_seconds - prior_j,
                    ledger.source_vehicle_seconds - prior_w,
                    ttc,
                    collisions,
                    teleports,
                ]
            )
            if step % 600 == 599:
                print(
                    json.dumps({"job": key, "step": step + 1, "arrived": len(ledger.arrived)}),
                    flush=True,
                )
        totals = np.asarray(rows).sum(axis=0)
        record = {
            "job": job,
            "key": key,
            "scenario": identity,
            "requested": len(requests),
            "designated_cavs": len(cav_ids),
            "realized_penetration": len(cav_ids) / len(requests),
            "arrived": len(ledger.arrived),
            "inserted": len(ledger.inserted),
            "unfinished": len(requests) - len(ledger.arrived),
            "braking_vehicle_seconds": float(totals[0]),
            "acceleration_effort": float(totals[1]),
            "system_seconds_per_request": float(totals[2] / len(requests)),
            "source_seconds_per_request": float(totals[3] / len(requests)),
            "ttc_below3_vehicle_seconds": float(totals[4]),
            "collision_involvements": int(totals[5]),
            "teleports": int(totals[6]),
            "counts": dict(counts),
            "elapsed_seconds": time.monotonic() - start,
            "accepted": False,
        }
        np.savez_compressed(
            out.with_suffix(".npz"),
            intervals=rows,
            observations=np.asarray(input_rows),
            actions=np.asarray(action_rows),
        )
        record["arrays_sha256"] = digest(out.with_suffix(".npz"))
        atomic_json(out, record)
        print(json.dumps({"finished": key, "seconds": record["elapsed_seconds"]}), flush=True)
        return record
    finally:
        close_sumo()


def summarize(records):
    native = {r["job"][0]: r for r in records if r["job"][3] == "native"}
    rows = []
    for r in records:
        n = native[r["job"][0]]
        rows.append(
            {
                "key": r["key"],
                "job": r["job"],
                "braking_change_pct": 100
                * (r["braking_vehicle_seconds"] / n["braking_vehicle_seconds"] - 1),
                "effort_change_pct": 100
                * (r["acceleration_effort"] / n["acceleration_effort"] - 1),
                "system_time_delta": r["system_seconds_per_request"]
                - n["system_seconds_per_request"],
                "source_wait_delta": r["source_seconds_per_request"]
                - n["source_seconds_per_request"],
                "ttc_exposure_delta": r["ttc_below3_vehicle_seconds"]
                - n["ttc_below3_vehicle_seconds"],
                "collision_delta": r["collision_involvements"] - n["collision_involvements"],
                "unfinished_delta": r["unfinished"] - n["unfinished"],
                "realized_penetration": r["realized_penetration"],
                "counts": r["counts"],
            }
        )
    atomic_json(
        ROOT / "summary.json",
        {
            "status": "complete",
            "comparisons": rows,
            "accepted": False,
            "rollouts": len(records),
            "simulation_seconds": len(records) * 2400,
        },
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", action="store_true")
    args = p.parse_args()
    with exclusive_run(ROOT / "execution"):
        jobs = prepare()
        if args.run:
            with ProcessPoolExecutor(max_workers=3) as pool:
                records = list(pool.map(run_job, jobs))
            summarize(records)
        else:
            print(json.dumps({"prepared": len(jobs), "simulations": 0}))
