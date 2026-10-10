"""Short exposed-seed mechanical check, NOT damping efficacy or training.

Run native, true-zero, and scripted correction from identical starts. This
explicitly capped prefix experiment cannot be used as a full-profile outcome.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import traci

from experiments.idm import deterministic_sumo  # noqa: F401
from env.native_damping import DampingContract, LocalHistory
from env.native_damping_sumo import SumoDampingActuator, SumoLocalSensor
from env.sumo_utils import close_sumo, start_sumo
from experiments import REPO_ROOT
from experiments.io import atomic_write_json


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _cav(vehicle_id: str, seed: int) -> bool:
    value = int(
        hashlib.sha256(f"damping-mechanics:{seed}:{vehicle_id}".encode()).hexdigest()[:16], 16
    )
    return value / 2**64 < 0.2


def _state():
    return [
        [
            v,
            traci.vehicle.getLaneID(v),
            traci.vehicle.getLanePosition(v),
            traci.vehicle.getSpeed(v),
            traci.vehicle.getAcceleration(v),
            traci.vehicle.getTau(v),
        ]
        for v in sorted(traci.vehicle.getIDList())
    ]


def run_arm(args, arm: str, cfg: DampingContract) -> dict:
    started = time.monotonic()
    source = REPO_ROOT / "scenarios" / "idm_transition"
    scenario = f"osm.dense.idm_m{args.demand}r400.sumocfg"
    start_sumo(
        args.sumo,
        scenario,
        source,
        seed=args.seed,
        extra_args=[
            "--step-length",
            "0.5",
            "--no-step-log",
            "true",
            "--duration-log.disable",
            "true",
            "--end",
            str(args.steps + 1),
        ],
    )
    sensor = SumoLocalSensor(traci, cfg)
    actuator, history = SumoDampingActuator(traci, cfg), LocalHistory(cfg)
    counts, reasons = Counter(), Counter()
    hashes, prefix = [], []
    collisions = teleports = 0
    retired = set()
    try:
        with (args.output / f"{arm}_commands.jsonl").open("w", encoding="utf-8") as stream:
            for step in range(args.steps):
                now = float(traci.simulation.getTime())
                live = set(traci.vehicle.getIDList())
                cavs = {v for v in live if _cav(v, args.seed)}
                for cav in retired - cavs:
                    history.drop(cav)
                retired = cavs
                frames = {}
                for cav in sorted(cavs):
                    observed = sensor.capture(
                        cav,
                        now_s=now,
                        previous_command_delta=actuator.previous_command_delta.get(cav, 0),
                    )
                    history.add(observed, now_s=now)
                    vector = history.vector(cav, now_s=now)
                    counts["actor_observations"] += 1
                    assert vector.shape == (cfg.observation_dim,)
                    frames[cav] = observed
                if arm != "native":
                    action = -0.2 if arm == "scripted" and 100 <= step < 200 else 0.0
                    commands = actuator.apply(
                        frames,
                        {v: action for v in frames},
                        now_s=now,
                        live_cav_ids=cavs,
                    )
                    for command in commands:
                        counts[command["operation"]] += 1
                        reasons.update(command.get("reasons", ()))
                for _ in range(2):
                    traci.simulationStep()
                    collisions += traci.simulation.getCollidingVehiclesNumber()
                    teleports += traci.simulation.getStartingTeleportNumber()
                after_cavs = {v for v in traci.vehicle.getIDList() if _cav(v, args.seed)}
                executed = actuator.observe_execution(
                    now_s=float(traci.simulation.getTime()),
                    live_cav_ids=after_cavs,
                )
                for row in executed:
                    stream.write(json.dumps({"step": step, **row}, allow_nan=False) + "\n")
                state_hash = _digest(_state())
                hashes.append(state_hash)
                if step < 100:
                    prefix.append(state_hash)
        return {
            "arm": arm,
            "sumo_version": list(traci.getVersion()),
            "global_control_steps": args.steps,
            "simulation_end_s": float(traci.simulation.getTime()),
            "counts": dict(counts),
            "reasons": dict(reasons),
            "trajectory_digest": _digest(hashes),
            "prefix_digest": _digest(prefix),
            "colliding_vehicle_ticks": collisions,
            "starting_teleports": teleports,
            "outstanding_command_owners": sorted(actuator.owned),
            "elapsed_seconds": time.monotonic() - started,
        }
    finally:
        close_sumo()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=range(4107, 4121), default=4107)
    parser.add_argument("--demand", type=int, choices=(2000, 2200, 2600, 2700), default=2000)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--sumo", default=r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo.exe")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 250 <= args.steps <= 600:
        parser.error("mechanical prefix must contain release and remain within 250–600 seconds")
    args.output = args.output.resolve()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("use a fresh output directory; evidence is never overwritten")
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = DampingContract()
    files = [
        Path(__file__),
        REPO_ROOT / "env" / "native_damping.py",
        REPO_ROOT / "env" / "native_damping_sumo.py",
        REPO_ROOT / "scenarios" / "idm_transition" / f"osm.dense.idm_m{args.demand}r400.sumocfg",
        REPO_ROOT / "scenarios" / "idm_transition" / f"idm_m{args.demand}r400.rou.xml",
    ]
    net_value = ET.parse(files[-2]).find("./input/net-file").get("value")
    network = (files[-2].parent / net_value).resolve()
    manifest = {
        "schema": "native_damping_mechanics/1",
        "scope": "development_mechanics_only",
        "seed": args.seed,
        "demand_vph": args.demand,
        "steps": args.steps,
        "controller": cfg.manifest(),
        "scripted_interval": [100, 200],
        "scripted_residual_mps2": -0.2,
        "cav_assignment": "sha256_p20_existing_IDM_parameters_retained_in_every_arm",
        "full_horizon_efficacy": False,
        "network_path": str(network),
        "network_sha256": hashlib.sha256(network.read_bytes()).hexdigest(),
        "source_hashes": {
            str(p.relative_to(REPO_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
        },
        "parameters": asdict(cfg),
    }
    atomic_write_json(args.output / "manifest.json", manifest)
    results = {}
    for arm in ("native", "zero", "scripted"):
        result = run_arm(args, arm, cfg)
        results[arm] = result
        atomic_write_json(args.output / f"{arm}.json", result)
        print(json.dumps(result), flush=True)
    checks = {
        "zero_native_exact": results["native"]["trajectory_digest"]
        == results["zero"]["trajectory_digest"],
        "identical_pretreatment_prefix": len({r["prefix_digest"] for r in results.values()}) == 1,
        "nonzero_command_authority": results["scripted"]["counts"].get("slow_down", 0) > 0,
        "script_changes_trajectory": results["scripted"]["trajectory_digest"]
        != results["native"]["trajectory_digest"],
        "no_outstanding_override": not results["scripted"]["outstanding_command_owners"],
        "zero_colliding_vehicle_ticks": all(
            r["colliding_vehicle_ticks"] == 0 for r in results.values()
        ),
        "zero_teleports": all(r["starting_teleports"] == 0 for r in results.values()),
    }
    atomic_write_json(
        args.output / "summary.json",
        {"manifest": manifest, "arms": results, "checks": checks, "passed": all(checks.values())},
    )
    print(json.dumps({"checks": checks, "passed": all(checks.values())}), flush=True)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
