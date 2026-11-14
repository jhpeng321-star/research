"""Four fixed SUMO mechanism probes, no training or corridor campaign."""

from pathlib import Path
import hashlib
import json

import numpy as np
import torch

from env.damping_v21 import SelectableActuator, SelectableContract, expert_from_observation
from experiments.idm import check_damping_v2 as fixture
from experiments.idm.train_native_damping import (
    atomic_json,
    exclusive_run,
    save_rollout,
    load_rollout,
)
from idm.damping_v2_actor import FixedExpert

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "runs/probe/native_selectable_damping_v2_check"
OUTPUT = ROOT / "runs/probe/native_selectable_damping_v21_mechanism"
SOURCES = (
    "env/damping_v21.py",
    "env/damping_v2.py",
    "env/native_damping.py",
    "env/native_damping_sumo.py",
    "idm/damping_v2_actor.py",
    "experiments/idm/check_damping_v2.py",
    "experiments/idm/check_damping_recovery_v21.py",
    "experiments/idm/train_native_damping.py",
    "docs/NATIVE_DAMPING_RECOVERY_DERIVATION_2026-09-06.md",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    with exclusive_run(OUTPUT):
        if (OUTPUT / "manifest.json").exists():
            raise RuntimeError("one-shot mechanism allocation exists; do not repeat")
        torch.set_num_threads(1)
        torch.manual_seed(81)
        previous = json.loads((BASE / "manifest.json").read_text())
        for name, expected in previous["source_sha256"].items():
            assert sha(ROOT / name) == expected
        native = {p: load_rollout(BASE / f"native_{p}.json") for p in fixture.PROFILES}
        manifest = dict(
            schema="damping_recovery_mechanism/2.1",
            controller=SelectableContract().manifest(),
            cases=fixture.PROFILES,
            role="already_exposed_mechanism_only",
            new_global_step_cap=480,
            new_rollout_cap=4,
            training_updates=0,
            corridor_runs=0,
            acceptance_sealed=True,
            wall_training_paused=True,
            source_sha256={p: sha(ROOT / p) for p in SOURCES},
            scene_inputs={
                str(BASE / "scene" / p): sha(BASE / "scene" / p)
                for p in ("scene.sumocfg", "routes.xml", "network.xml")
            },
            native_inputs={
                str(BASE / f"native_{p}{suffix}"): sha(BASE / f"native_{p}{suffix}")
                for p in fixture.PROFILES
                for suffix in (".json", ".npz")
            },
            fixture_adapter="reuse frozen run_scene; replace its actuator factory only in this process",
        )
        atomic_json(OUTPUT / "manifest.json", manifest)
        for name in SOURCES:
            path = OUTPUT / "source_snapshot" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((ROOT / name).read_bytes())
        # Explicit fixture dependency substitution; no on-disk v2 source mutation.
        # The local observations, mode head, trajectories and metrics stay the same.
        fixture.SelectableActuator = SelectableActuator
        scene = BASE / "scene/scene.sumocfg"
        outcomes = {}
        for index, profile in enumerate(fixture.PROFILES):
            atomic_json(
                OUTPUT / f"{profile}.started.json",
                dict(allocated_rollouts=index + 1, allocated_global_seconds=(index + 1) * 120),
            )
            result = fixture.run_scene(scene, profile, FixedExpert())
            save_rollout(OUTPUT / f"{profile}.json", result)
            for field in (
                "scenario_identity",
                "seed",
                "population",
                "global_control_steps",
                "team_terminal",
            ):
                assert result["summary"][field] == native[profile]["summary"][field]
            for i, row in enumerate(result["commands"]):
                expected = float(expert_from_observation(result["actor_observations"][i])[0])
                assert row["expert_acceleration"] == expected
            outcomes[profile] = dict(
                paired=fixture.physical_comparison(result, native[profile]),
                absolute=result["summary"],
            )
        sharp = load_rollout(OUTPUT / "sharp_brake.json")
        recovery_loss = [
            c["step"]
            for c in sharp["commands"]
            if 35 <= c["step"] <= 90
            and c["operation"] == "release_to_native"
            and sharp["actor_observations"][c["step"], 77] <= 0.5
        ]
        steady = load_rollout(OUTPUT / "steady.json")
        exact = np.array_equal(
            np.array([r["vehicles"] for r in steady["trace"]]),
            np.array([r["vehicles"] for r in native["steady"]["trace"]]),
        )
        checks = dict(
            steady_exact_native=exact,
            pulse_effort=outcomes["pulse"]["paired"]["effort_improvement_pct"] >= 5,
            no_added_severe_braking=all(
                r["paired"]["braking_vehicle_seconds"] <= 0 for r in outcomes.values()
            ),
            no_added_ttc_or_hard_events=all(
                r["paired"]["ttc_delta"] <= 0
                and r["paired"]["added_collision_events"] <= 0
                and r["paired"]["added_teleport_events"] <= 0
                for r in outcomes.values()
            ),
            endpoint_progress=all(
                min(r["paired"]["position_delta_m"]) >= -30 for r in outcomes.values()
            ),
            no_recovery_sensor_loss_release=not recovery_loss,
            sharp_effort=outcomes["sharp_brake"]["paired"]["acceleration_energy"] <= 0,
        )
        summary = dict(
            checks=checks,
            mechanism_pass=all(checks.values()),
            outcomes=outcomes,
            recovery_sensor_loss_release_steps=recovery_loss,
            new_global_seconds=480,
            training_updates=0,
            accepted_checkpoint=False,
            decision="mechanism_only_no_learning_or_corridor_authorized_by_this_result",
        )
        atomic_json(OUTPUT / "summary.json", summary)
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
