"""Bounded 2 x 10-second ownership check on an isolated IDM lane.

This is actuator verification, not a traffic-benefit experiment or wall gate.
No retries and no shared network modifications. Every output directory is new.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

from .config import load_config
from .damping import CirclesController
from .ownership import CommandOwnership
from .sensing import SumoLocalSensor
from .simulation import start_sumo, close_sumo
from .verification import hash_file


class LoggedVehicle:
    def __init__(self, api, commands):
        self.api, self.commands = api, commands

    def __getattr__(self, name):
        return getattr(self.api, name)

    def slowDown(self, vehicle, target, duration):
        self.commands.append(["slowDown", vehicle, target, duration])
        self.api.slowDown(vehicle, target, duration)

    def setSpeed(self, vehicle, speed):
        self.commands.append(["setSpeed", vehicle, speed])
        self.api.setSpeed(vehicle, speed)


def run_check(output, sumo_binary):
    import traci
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = load_config()
    manifest = {"schema": "circles_ownership_mechanics/1", "purpose":
                "Actual incomplete-command release, wall/damping switching and HDV isolation",
                "runs": 2, "seconds_per_run": 10, "maximum_simulated_seconds": 20,
                "automatic_retries": 0, "scientifically_accepted": False,
                "seed": 1, "seed_scope": "isolated synthetic three-vehicle lane, not corridor evaluation",
                "model_sha256": config.model_sha256,
                "adapter_sha256": hash_file(Path(__file__).resolve().parents[1] / "baselines/circles/adapter.json")}
    (output / "budget.json").write_text(json.dumps(manifest, indent=2))
    (output / "nodes.xml").write_text('<nodes><node id="a" x="0" y="0"/><node id="b" x="2000" y="0"/></nodes>')
    (output / "edges.xml").write_text('<edges><edge id="road" from="a" to="b" numLanes="1" speed="30"/></edges>')
    converter = Path(sumo_binary).parent / ("netconvert.exe" if Path(sumo_binary).suffix == ".exe" else "netconvert")
    result = subprocess.run([str(converter), "--node-files", str(output / "nodes.xml"),
                             "--edge-files", str(output / "edges.xml"), "--output-file",
                             str(output / "network.net.xml")], capture_output=True, text=True)
    (output / "netconvert.log").write_text(result.stdout + result.stderr)
    result.check_returncode()
    (output / "routes.rou.xml").write_text('''<routes>
<vType id="idm" carFollowModel="IDM" accel="2.6" decel="4.5" tau="1.1" minGap="2.5" sigma="0" speedFactor="1" speedDev="0" length="5"/>
<route id="route" edges="road"/>
<vehicle id="leader_hdv" type="idm" route="route" depart="0" departPos="150" departSpeed="15"/>
<vehicle id="cav" type="idm" route="route" depart="0" departPos="100" departSpeed="15"/>
<vehicle id="rear_hdv" type="idm" route="route" depart="0" departPos="50" departSpeed="15"/>
</routes>''')
    scene = output / "scenario.sumocfg"
    scene.write_text('<configuration><input><net-file value="network.net.xml"/><route-files value="routes.rou.xml"/></input><processing><time-to-teleport value="-1"/><collision.action value="warn"/></processing></configuration>')
    outcomes = {}
    for branch in ("held", "released"):
        commands, records, physical = [], [], []
        api = SimpleNamespace(vehicle=LoggedVehicle(traci.vehicle, commands), lane=traci.lane)
        controller = CirclesController(config)
        owner = CommandOwnership(api, ["cav"], controller)
        sensor = SumoLocalSensor(api, config)
        collisions = teleports = 0
        start_sumo(sumo_binary, scene, seed=1, config=config)
        try:
            for _ in range(20):
                traci.simulationStep()
                now = traci.simulation.getTime()
                live = traci.vehicle.getIDList()
                collisions += traci.simulation.getCollidingVehiclesNumber()
                teleports += traci.simulation.getStartingTeleportNumber()
                physical.append({"time_s": now, "speed_mps": traci.vehicle.getSpeed("cav"),
                                 "acceleration_mps2": traci.vehicle.getAcceleration("cav"),
                                 "hdv_types": [traci.vehicle.getTypeID(v) for v in ("leader_hdv", "rear_hdv")],
                                 "speed_modes": [traci.vehicle.getSpeedMode(v) for v in live]})
                if now == 3.5 and branch == "released":
                    # Terminate an unfinished 3→4s slowDown, not merely a command
                    # that has already expired. The held branch is causal control.
                    owner.release_all(live_ids=live)
                if now.is_integer() and now < 10:
                    frame = sensor.capture("cav", now_s=now)
                    frames = {} if now == 6 else {"cav": frame}
                    wall = {"cav": -.8} if now in (2, 3, 8) else {}
                    native = ["cav"] if now == 4 else []
                    record = owner.step(frames, now_s=now, live_ids=live,
                                        wall_commands=wall, native_ids=native)[0]
                    records.append(record.as_dict())
            owner.release_all(live_ids=traci.vehicle.getIDList())
        finally:
            close_sumo()
        data = dict(commands=commands, decisions=records, measured_responses=physical,
                    collisions=collisions, teleports=teleports)
        (output / f"{branch}.json").write_text(json.dumps(data, indent=2, allow_nan=False))
        outcomes[branch] = data
    held, released = outcomes["held"], outcomes["released"]
    prefix_equal = held["measured_responses"][:7] == released["measured_responses"][:7]
    held4, release4 = held["measured_responses"][7], released["measured_responses"][7]
    difference = release4["speed_mps"] - held4["speed_mps"]
    expected = ["damping", "wall", "wall", "native", "damping", "native", "damping", "wall", "damping"]
    checks = {
        "matched_prefix_through_3_5s": prefix_equal,
        "incomplete_command_release_changes_next_physics": difference > 1e-6,
        "ownership_sequence_and_reengagement": all([d["owner"] for d in x["decisions"]] == expected for x in outcomes.values()),
        "missing_frame_fallback_releases_override": all(x["decisions"][5]["released_to_native"] and
                x["decisions"][5]["fallback_reason"] == "missing_observation" for x in outcomes.values()),
        "only_cav_writes": all(c[1] == "cav" for x in outcomes.values() for c in x["commands"]),
        "hdvs_remain_idm_and_safety31": all(r["hdv_types"] == ["idm", "idm"] and set(r["speed_modes"]) == {31}
                                                for x in outcomes.values() for r in x["measured_responses"]),
        "zero_collisions_teleports": all(x["collisions"] == x["teleports"] == 0 for x in outcomes.values())}
    summary = dict(manifest, checks=checks, passed=all(checks.values()),
                   actual_simulated_seconds=20, release_speed_difference_mps=difference,
                   held_acceleration_at_4s_mps2=held4["acceleration_mps2"],
                   released_acceleration_at_4s_mps2=release4["acceleration_mps2"],
                   decision_owners=dict(Counter(d["owner"] for x in outcomes.values() for d in x["decisions"])))
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sumo-binary", default="C:/Program Files (x86)/Eclipse/Sumo/bin/sumo.exe")
    parser.add_argument("--candidate", choices=("pi-external-u", "pi-regions"))
    parser.add_argument("--registration")
    args = parser.parse_args(argv)
    if args.candidate is not None:
        if args.registration is None:
            parser.error("--registration is required with --candidate")
        if args.candidate == "pi-regions":
            from output.controller_verification_v10.launch_contract import run_registered
        elif args.candidate == "pi-external-u":
            from output.controller_verification_v8.pi_mechanics import run_registered
        result = run_registered(args.registration, args.output, args.sumo_binary)
        print(json.dumps(result, indent=2))
        return 0 if result.get("status") == "complete" else 1
    if args.registration is not None:
        parser.error("--registration requires --candidate")
    result = run_check(args.output, args.sumo_binary)
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
