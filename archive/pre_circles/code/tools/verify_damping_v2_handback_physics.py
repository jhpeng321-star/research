"""One bounded, paired physical handback check; no learning or corridor runs.

Branch before a timed command has finished. This distinguishes actual release
from waiting until the command expires and does not use getSpeedWithoutTraCI
as an oracle. The existing v2 controller source is imported without alteration.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import traci

from env.damping_v2 import FilteredHistory, SelectableActuator
from env.native_damping_sumo import SumoLocalSensor
from experiments.idm.check_damping_v2 import build_scene


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "runs/probe/native_selectable_damping_v2_handback_physics"
SOURCES = [
    "tools/verify_damping_v2_handback_physics.py",
    "env/damping_v2.py", "env/native_damping.py", "env/native_damping_sumo.py",
    "experiments/idm/check_damping_v2.py",
]


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def snapshot(connection):
    return {
        "time": connection.simulation.getTime(),
        "vehicles": {
            vid: [connection.vehicle.getLanePosition(vid), connection.vehicle.getSpeed(vid),
                  connection.vehicle.getAcceleration(vid), connection.vehicle.getTau(vid)]
            for vid in ["leader", "cav", "follower"]
        },
    }


def branch(scene: Path, name: str):
    # The allocation is reserved before SUMO is started. Failures are not retried.
    save(DESTINATION / f"{name}.started.json", {"reserved_global_seconds": 20, "seed": 8101})
    traci.start([
        "C:/Program Files (x86)/Eclipse/Sumo/bin/sumo.exe", "-c", str(scene),
        "--seed", "8101", "--step-length", "0.5", "--no-step-log", "true",
        "--duration-log.disable", "true", "--error-log", str(DESTINATION / f"{name}.sumo.log"),
    ], label=name)
    connection = traci.getConnection(name)
    sensor = SumoLocalSensor(connection)
    actuator = SelectableActuator(connection)
    history = FilteredHistory()
    trace, commands = [], []

    def action(mode):
        now = connection.simulation.getTime()
        frame = sensor.capture("cav", now_s=now)
        history.add(frame, now_s=now)
        observation = history.vector("cav", now_s=now, actuator=actuator)
        rows = actuator.apply(
            {"cav": frame}, {"cav": (mode, -0.3 if mode else 0.29)},
            now_s=now, live_cav_ids={"cav"}, observations={"cav": observation},
        )
        commands.extend({"time": now, **row} for row in rows)
        return rows[0]

    try:
        connection.simulationStep()
        connection.vehicle.setSpeed("leader", 15)
        while connection.simulation.getTime() < 3:
            connection.simulationStep()
        prefix = snapshot(connection)
        first_command = action(1)
        assert first_command["operation"] == "slow_down"
        connection.simulationStep()  # Only 0.5 s of the 1 s command has elapsed.
        branch_state = snapshot(connection)
        if name == "release":
            assert action(0)["operation"] == "release_to_native"
            assert "cav" not in actuator.owned
        connection.simulationStep()
        first_after_branch = snapshot(connection)
        trace.append(first_after_branch)
        if name == "release":
            blocked = action(1)
            assert blocked["operation"] == "native_no_command"
            assert blocked["reasons"] == ("native_cooldown",)
        while connection.simulation.getTime() < 8.5:
            connection.simulationStep()
            trace.append(snapshot(connection))
        if name == "release":
            assert action(1)["operation"] == "slow_down"
            assert "cav" in actuator.owned
        connection.simulationStep()
        trace.append(snapshot(connection))
        if name == "release":
            assert action(0)["operation"] == "release_to_native"
            assert "cav" not in actuator.owned
        while connection.simulation.getTime() < 10:
            connection.simulationStep()
            trace.append(snapshot(connection))
        result = {
            "prefix": prefix, "first_command": first_command, "branch_state": branch_state,
            "first_after_branch": first_after_branch, "commands": commands, "trace": trace,
            "completed_global_seconds": connection.simulation.getTime(),
            "final_owned": sorted(actuator.owned),
        }
        assert all(row["vehicles"]["cav"][3] == 1.5 for row in trace)
        save(DESTINATION / f"{name}.json", result)
        return result
    finally:
        connection.close()


def main():
    DESTINATION.mkdir(parents=True, exist_ok=False)
    hashes = {source: hashlib.sha256((ROOT / source).read_bytes()).hexdigest() for source in SOURCES}
    for source in SOURCES:
        output = DESTINATION / "source_snapshot" / source
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / source, output)
    save(DESTINATION / "manifest.json", {
        "schema": "v2_unexpired_command_handback_mechanics_v1",
        "source_hashes": hashes, "seed": 8101, "development_only": True,
        "max_new_global_seconds": 40, "allocations": 2, "per_allocation_ceiling": 20,
        "planned_duration_per_allocation": 10, "training_seconds": 0,
        "corridor_seconds": 0, "automatic_retry": False,
        "comparison": "identical state after half of a 1 s active command; release versus retain",
        "criteria": {
            "prefix_and_branch_state": "exact identity",
            "first_halfstep_after_release": "speed exceeds held command by >0.01 m/s",
            "held_halfstep": "acceleration matches retained target within 1e-8 m/s2",
            "switching": "cooldown rejects re-engagement, admits it at expiry, then releases again",
            "tau": "unchanged at 1.5 s",
        },
        "limits": "mechanical cancellation only; not smoothness, safety or damping efficacy",
    })
    scene = build_scene(DESTINATION / "scene")
    try:
        held = branch(scene, "held")
        released = branch(scene, "release")
        assert held["prefix"] == released["prefix"]
        assert held["first_command"] == released["first_command"]
        assert held["branch_state"] == released["branch_state"]
        rh = held["first_after_branch"]["vehicles"]["cav"]
        rr = released["first_after_branch"]["vehicles"]["cav"]
        speed_difference = rr[1] - rh[1]
        assert speed_difference > 0.01
        assert np.isclose(rh[2], held["first_command"]["target_acceleration"], rtol=0, atol=1e-8)
        assert not released["final_owned"]
        assert hashes == {source: hashlib.sha256((ROOT / source).read_bytes()).hexdigest() for source in SOURCES}
        summary = {
            "pass": True, "same_prefix_and_branch_state_exact": True,
            "branch_time": released["branch_state"]["time"],
            "first_after_branch_time": released["first_after_branch"]["time"],
            "held_speed_mps": rh[1], "released_speed_mps": rr[1],
            "released_minus_held_speed_mps": speed_difference,
            "held_acceleration_mps2": rh[2], "released_acceleration_mps2": rr[2],
            "release_commands": released["commands"], "tau_unchanged": True,
            "completed_global_seconds": held["completed_global_seconds"] + released["completed_global_seconds"],
            "charged_global_seconds": 40, "training_seconds": 0,
            "getSpeedWithoutTraCI_used_as_evidence": False,
            "interpretation": "Native action cancels an unexpired owned command; no efficacy/smoothness claim.",
        }
        save(DESTINATION / "summary.json", summary)
        print(json.dumps(summary, indent=2))
    except Exception as error:
        save(DESTINATION / "failure.json", {"type": type(error).__name__, "error": str(error)})
        raise


if __name__ == "__main__":
    main()
