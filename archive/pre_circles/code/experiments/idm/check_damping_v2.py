"""One preregistered, bounded short-scene learning and mechanics check.

No acceptance cases, wall training, parameter search, or automatic campaign.
The 120 s platoon objective is a mechanism surrogate, not corridor acceptance.
"""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch
import traci

from env.damping_v2 import SelectableActuator, FilteredHistory, SelectableContract
from env.native_damping_sumo import SumoLocalSensor, SumoDampingActuator
from idm.damping_v2_actor import SelectableActor, FixedNative, FixedExpert
from idm.damping_v2_reward import RewardContract, prepare_batch
from idm.damping_ppo import TeamCritic, PPOContract, update_ppo
from experiments.idm.train_native_damping import atomic_json, exclusive_run, save_rollout

ROOT = Path(__file__).resolve().parents[2]
PROFILES = ("steady", "pulse", "catchup", "sharp_brake")
SOURCES = (
    "env/damping_v2.py",
    "env/damping_v2_rollout.py",
    "idm/damping_v2_actor.py",
    "idm/damping_v2_reward.py",
    "idm/damping_ppo.py",
    "idm/damping_credit.py",
    "env/native_damping.py",
    "env/native_damping_sumo.py",
    "env/damping_rollout.py",
    "experiments/idm/check_damping_v2.py",
    "experiments/idm/train_native_damping.py",
    "docs/NATIVE_SELECTABLE_DAMPING_V2_PROTOCOL.md",
)


def build_scene(root):
    root.mkdir(parents=True, exist_ok=True)
    (root / "nodes.xml").write_text(
        '<nodes><node id="a" x="0" y="0"/><node id="b" x="6000" y="0"/></nodes>'
    )
    (root / "edges.xml").write_text(
        '<edges><edge id="road" from="a" to="b" numLanes="1" speed="30"/></edges>'
    )
    subprocess.run(
        [
            "C:/Program Files (x86)/Eclipse/Sumo/bin/netconvert.exe",
            "--node-files",
            str(root / "nodes.xml"),
            "--edge-files",
            str(root / "edges.xml"),
            "--output-file",
            str(root / "network.xml"),
        ],
        check=True,
        capture_output=True,
    )
    (root / "routes.xml").write_text("""<routes>
    <vType id="idm" carFollowModel="IDM" accel="2.6" decel="4.5" emergencyDecel="9" tau="1.5" minGap="2.5" length="5" speedDev="0"/>
    <route id="r" edges="road"/>
    <vehicle id="leader" type="idm" route="r" depart="0" departPos="300" departSpeed="15"/>
    <vehicle id="cav" type="idm" route="r" depart="0" departPos="268" departSpeed="15"/>
    <vehicle id="follower" type="idm" route="r" depart="0" departPos="236" departSpeed="15"/>
    </routes>""")
    (root / "scene.sumocfg").write_text(
        """<configuration><input><net-file value="network.xml"/><route-files value="routes.xml"/></input><processing><time-to-teleport value="-1"/><collision.action value="warn"/></processing></configuration>"""
    )
    return root / "scene.sumocfg"


def leader_speed(profile, t):
    if profile == "steady":
        return 15.0
    if profile == "pulse":
        return float(np.interp(t, [0, 25, 33, 47, 70, 78, 92, 120], [15, 15, 8, 15, 15, 8, 15, 15]))
    if profile == "catchup":
        return float(np.interp(t, [0, 25, 40, 120], [15, 15, 20, 20]))
    if profile == "sharp_brake":
        return float(np.interp(t, [0, 25, 29, 35, 61, 120], [15, 15, 2, 2, 15, 15]))
    raise ValueError(profile)


def run_scene(
    scene,
    profile,
    actor=None,
    critic=None,
    *,
    stochastic=False,
    seed=8101,
    switch_test=False,
    random_seed=81,
):
    torch.manual_seed(random_seed)
    traci.start(
        [
            "C:/Program Files (x86)/Eclipse/Sumo/bin/sumo.exe",
            "-c",
            str(scene),
            "--seed",
            str(seed),
            "--step-length",
            "0.5",
            "--no-step-log",
            "true",
            "--duration-log.disable",
            "true",
        ]
    )
    sensor = SumoLocalSensor(traci)
    actuator = SelectableActuator(traci)
    analytical = SumoDampingActuator(traci)
    history = FilteredHistory()
    observations, actions, logps, states, values, intervals, duty = [], [], [], [], [], [], []
    trace, ttcs, commands = [], [], []
    old_collisions = set()
    try:
        traci.simulationStep()
        initial_tau = traci.vehicle.getTau("cav")
        for step in range(120):
            now = traci.simulation.getTime()
            # Scripted leader is an exogenous test fixture, never an actor input.
            traci.vehicle.slowDown("leader", leader_speed(profile, step + 1), 1.0)
            frame = sensor.capture(
                "cav",
                now_s=now,
                previous_command_delta=actuator.previous_command_delta.get("cav", 0),
            )
            history.add(frame, now_s=now)
            x = history.vector("cav", now_s=now, actuator=actuator)
            state = np.zeros(22, dtype=np.float32)
            state[:5] = [
                step / 120,
                traci.vehicle.getSpeed("leader") / 30,
                frame.speed_mps / 30,
                traci.vehicle.getSpeed("follower") / 30,
                frame.front.gap_m / 100 if frame.front else 0,
            ]
            states.append(state)
            with torch.no_grad():
                values.append(0 if critic is None else float(critic(torch.tensor(state))))
            row = []
            fraction = 0
            if actor == "analytical":
                delta = (
                    0
                    if frame.front is None
                    else float(np.clip(frame.front.relative_speed_mps / 5, -0.5, 0.3))
                )
                row = analytical.apply(
                    {"cav": frame}, {"cav": delta}, now_s=now, live_cav_ids={"cav"}
                )
            elif actor is not None:
                with torch.no_grad():
                    tensor = torch.tensor(x[None])
                    if switch_test:
                        # Choose native immediately after active, then challenge cooldown.
                        mode = int(step in (10, 11, 13, 17, 18, 24))
                        action = torch.tensor([[mode, -0.3 if mode else 0]], dtype=torch.float32)
                        lp = torch.zeros(1)
                    elif stochastic:
                        action, lp = actor.sample(tensor)
                    else:
                        action = actor(tensor)
                        lp = actor.log_prob(tensor, action)
                a = action.numpy()[0]
                row = actuator.apply(
                    {"cav": frame},
                    {"cav": a},
                    now_s=now,
                    live_cav_ids={"cav"},
                    observations={"cav": x},
                )
                observations.append(x)
                actions.append(a)
                logps.append(float(lp[0]))
                fraction = float(a[0])
            duty.append(fraction)
            commands.extend(dict(step=step, **c) for c in row)
            braking = effort = ttc = 0.0
            collisions = teleports = 0
            for _ in range(2):
                traci.simulationStep()
                data = []
                for cav in ("leader", "cav", "follower"):
                    v = traci.vehicle.getSpeed(cav)
                    a = traci.vehicle.getAcceleration(cav)
                    data.append([traci.vehicle.getLanePosition(cav), v, a])
                    if cav != "leader":
                        braking += 0.5 * (a < -3)
                        effort += 0.5 * a * a
                        lead = traci.vehicle.getLeader(cav, 200)
                        if lead:
                            closing = v - traci.vehicle.getSpeed(lead[0])
                            gap = lead[1] + traci.vehicle.getMinGap(cav)
                            ttc += 0.5 * (closing > 0 and gap / max(closing, 1e-9) < 3)
                c = set(traci.simulation.getCollidingVehiclesIDList())
                collisions += len(c - old_collisions)
                old_collisions = c
                teleports += traci.simulation.getStartingTeleportNumber()
                trace.append(
                    dict(
                        step=step,
                        time=traci.simulation.getTime(),
                        vehicles=data,
                        native_speed=traci.vehicle.getSpeedWithoutTraCI("cav"),
                        tau=traci.vehicle.getTau("cav"),
                    )
                )
            intervals.append(
                dict(
                    braking_vehicle_seconds=float(braking),
                    acceleration_energy=float(effort),
                    requested_not_arrived_vehicle_seconds=2.0,
                    added_collision_events=collisions,
                    added_teleport_events=teleports,
                )
            )
            ttcs.append(float(ttc))
        totals = {k: sum(r[k] for r in intervals) for k in intervals[0]}
        identity = hashlib.sha256(
            (scene.parent / "routes.xml").read_bytes()
            + (scene.parent / "network.xml").read_bytes()
            + profile.encode()
        ).hexdigest()
        summary = dict(
            scenario_identity=identity,
            seed=seed,
            population=2,
            global_control_steps=120,
            team_terminal=True,
            system_seconds_per_requested_vehicle=120,
            source_wait_vehicle_seconds=0,
            ttc_below_3_vehicle_seconds=sum(ttcs),
            arrived=0,
            unfinished_requested=2,
            totals=totals,
            profile=profile,
            initial_tau=initial_tau,
            final_positions=trace[-1]["vehicles"],
            mean_requested_active_fraction=float(np.mean(duty)),
            mean_executed_active_fraction=sum(c["operation"] == "slow_down" for c in commands)
            / 120,
            objective="finite_short_platoon_surrogate_not_full_corridor_episode",
        )
        return dict(
            actor_observations=np.asarray(observations, dtype=np.float32).reshape(-1, 89),
            actions=np.asarray(actions, dtype=np.float32).reshape(-1, 2),
            old_logp=np.asarray(logps, dtype=np.float32),
            actor_times=np.arange(len(actions), dtype=np.int64),
            actor_ids=["cav"] * len(actions),
            critic_observations=np.asarray(states, dtype=np.float32),
            old_values=np.asarray(values + [0], dtype=np.float32),
            intervals=intervals,
            intervention_fractions=duty,
            source_intervals=[0] * 120,
            ttc_intervals=ttcs,
            trace=trace,
            commands=commands,
            summary=summary,
        )
    finally:
        traci.close()


def physical_comparison(rollout, native):
    a, b = rollout["summary"], native["summary"]
    result = {k: a["totals"][k] - b["totals"][k] for k in a["totals"]}
    result.update(
        ttc_delta=a["ttc_below_3_vehicle_seconds"] - b["ttc_below_3_vehicle_seconds"],
        position_delta_m=[a["final_positions"][i][0] - b["final_positions"][i][0] for i in (1, 2)],
        requested_active=a["mean_requested_active_fraction"],
        executed_active=a["mean_executed_active_fraction"],
    )
    result["effort_improvement_pct"] = 100 * (
        1 - a["totals"]["acceleration_energy"] / max(b["totals"]["acceleration_energy"], 1e-9)
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    with exclusive_run(root):
        if (root / "manifest.json").exists():
            raise RuntimeError("one-shot budget: no automatic repeat or overwrite")
        torch.set_num_threads(1)
        torch.manual_seed(81)
        manifest = dict(
            schema="native_selectable_repair_check/2.0",
            controller=SelectableContract().manifest(),
            reward=RewardContract().manifest(),
            profiles=PROFILES,
            optimizer_seed=81,
            short_rollout_cap=30,
            short_global_step_cap=3600,
            training_updates=12,
            corridor_rollout_cap=8,
            corridor_global_step_cap=19200,
            acceptance_sealed=True,
            wall_training_paused=True,
            source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SOURCES},
        )
        atomic_json(root / "manifest.json", manifest)
        for name in SOURCES:
            target = root / "source_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / name).read_bytes())
        scene = build_scene(root / "scene")
        allocated = 0

        def run(name, profile, actor=None, critic=None, **kwargs):
            nonlocal allocated
            allocated += 1
            if allocated > 30:
                raise RuntimeError("short rollout budget exhausted")
            atomic_json(
                root / f"{name}.started.json",
                dict(allocated_rollouts=allocated, allocated_steps=allocated * 120),
            )
            result = run_scene(scene, profile, actor, critic, **kwargs)
            save_rollout(root / f"{name}.json", result)
            return result

        native = {p: run(f"native_{p}", p) for p in PROFILES}
        null = run("explicit_native_identity", "pulse", FixedNative())
        identity = np.array([r["vehicles"] for r in null["trace"]])
        expected = np.array([r["vehicles"] for r in native["pulse"]["trace"]])
        if not np.array_equal(identity, expected) or any(
            c["operation"] != "native_no_command" for c in null["commands"]
        ):
            raise RuntimeError("native branch does not reproduce untouched IDM")
        switches = run("native_release_switching", "steady", FixedNative(), switch_test=True)
        commands = {c["step"]: c for c in switches["commands"]}
        assert commands[10]["operation"] == "slow_down"
        assert commands[12]["operation"] == "release_to_native"
        assert commands[13]["reasons"] == ("native_cooldown",)
        assert commands[17]["operation"] == "slow_down"
        assert commands[19]["operation"] == "release_to_native"
        for row in switches["trace"]:
            assert row["tau"] == switches["summary"]["initial_tau"]
            if commands[row["step"]]["operation"] in ("native_no_command", "release_to_native"):
                assert abs(row["vehicles"][1][1] - row["native_speed"]) < 1e-8
        atomic_json(
            root / "mechanics.json",
            dict(
                native_trajectory_exact=True,
                release_native_speed_exact=True,
                reengagement=True,
                cooldown=True,
                tau_unchanged=True,
            ),
        )
        expert = {p: run(f"expert_{p}", p, FixedExpert()) for p in PROFILES}
        analytical = {p: run(f"analytical_{p}", p, "analytical") for p in PROFILES}
        # Scale the short task once from its four native trajectories. No tuned rewards.
        reward = replace(
            RewardContract(),
            braking_scale_vehicle_seconds=max(
                1,
                np.mean(
                    [r["summary"]["totals"]["braking_vehicle_seconds"] for r in native.values()]
                ),
            ),
            effort_scale_m2_per_s3=max(
                1, np.mean([r["summary"]["totals"]["acceleration_energy"] for r in native.values()])
            ),
        )
        atomic_json(root / "short_reward_scales.json", reward.manifest())
        actor, critic = SelectableActor(), TeamCritic(22)
        actor_opt = torch.optim.Adam(actor.parameters(), lr=1e-4)
        critic_opt = torch.optim.Adam(critic.parameters(), lr=1e-4)
        rng = np.random.default_rng(81)
        training = []
        for iteration in range(12):
            profile = PROFILES[iteration % 4]
            rollout = run(
                f"train_{iteration:02}",
                profile,
                actor,
                critic,
                stochastic=True,
                seed=8101,
                random_seed=8100 + iteration,
            )
            batch = prepare_batch(rollout, native[profile], reward)
            learner = update_ppo(
                actor,
                critic,
                actor_opt,
                critic_opt,
                batch,
                rng=rng,
                contract=PPOContract(time_minibatch=32),
            )
            record = dict(
                iteration=iteration + 1,
                profile=profile,
                reward=batch["reward_audit"],
                learner=learner,
            )
            training.append(record)
            print(json.dumps(record), flush=True)
        torch.save(
            dict(
                schema="native_selectable_damping/2.0",
                actor=actor.state_dict(),
                critic=critic.state_dict(),
                controller=SelectableContract().manifest(),
                accepted=False,
                training=training,
            ),
            root / "checkpoint_012.pt",
        )
        learned = {p: run(f"learned_{p}", p, actor) for p in PROFILES}
        comparisons = {
            arm: {p: physical_comparison(data[p], native[p]) for p in PROFILES}
            for arm, data in (("expert", expert), ("analytical", analytical), ("learned", learned))
        }
        # Progress loss prevents reduced effort from simply parking. Sharp braking
        # remains a safety stress, never an incentive to suppress necessary braking.
        checks = dict(
            steady_abstention=learned["steady"]["summary"]["mean_executed_active_fraction"] <= 0.05,
            pulse_effort=comparisons["learned"]["pulse"]["effort_improvement_pct"] >= 5,
            no_added_hard_events=all(
                v["added_collision_events"] <= 0 and v["added_teleport_events"] <= 0
                for v in comparisons["learned"].values()
            ),
            no_ttc_increase=all(v["ttc_delta"] <= 0 for v in comparisons["learned"].values()),
            progress=all(
                min(v["position_delta_m"]) >= -30 for v in comparisons["learned"].values()
            ),
        )
        summary = dict(
            status="bounded_check_complete",
            mechanics_pass=True,
            training_updates=12,
            allocated_short_rollouts=allocated,
            allocated_short_steps=allocated * 120,
            comparisons=comparisons,
            checks=checks,
            mechanism_gate_pass=all(checks.values()),
            accepted_checkpoint=False,
            corridor_training_authorized=False,
            training=training,
            decision="bounded_corridor_diagnostic_only_no_full_campaign",
        )
        atomic_json(root / "summary.json", summary)
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
