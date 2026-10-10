"""Six new corridor diagnostics; reuse two verified analytical controls.

Completes the fixed v2 repair check. Never trains or opens acceptance data.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from env.damping_v2_rollout import collect_episode
from experiments.idm.check_damping_v2 import SOURCES
from experiments.idm.train_native_damping import (
    atomic_json,
    exclusive_run,
    save_rollout,
    load_rollout,
)
from idm.damping_scenario import build_scenario, request_schedule
from idm.damping_selection import case_metrics
from idm.damping_v2_actor import FixedExpert, SelectableActor
from idm.damping_v2_reward import reward_series, prepare_batch

ROOT = Path(__file__).resolve().parents[2]
CASES = ((911001, 2000), (911003, 2200))
ARMS = ("native", "expert", "learned")
CAMPAIGN = ROOT / "runs/rllib/native_damping_campaign_v1"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sources(check_root, registration):
    for name, expected in registration["source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"frozen source changed: {name}")
    if sha(check_root / "checkpoint_012.pt") != registration["checkpoint_sha256"]:
        raise ValueError("checkpoint changed")


def worker(check_root, seed, arm):
    output = check_root / "corridor"
    registration = read(output / "manifest.json")
    verify_sources(check_root, registration)
    if seed not in [c[0] for c in CASES] or arm not in ARMS:
        raise ValueError("outside diagnostic allocation")
    path = output / f"{arm}_{seed}.json"
    started = path.with_suffix(".started.json")
    if path.exists() or started.exists():
        raise RuntimeError("no repeated or overwritten allocations")
    loading = dict(CASES)[seed]
    scene = output / f"scene_{seed}" / "scenario.sumocfg"
    requests = request_schedule(seed, loading, role="development")
    torch.set_num_threads(1)
    actor = None
    if arm == "expert":
        actor = FixedExpert()
    elif arm == "learned":
        actor = SelectableActor()
        actor.load_state_dict(
            torch.load(check_root / "checkpoint_012.pt", weights_only=True, map_location="cpu")[
                "actor"
            ]
        )
    atomic_json(started, dict(pid=os.getpid(), seed=seed, arm=arm, allocated_global_steps=2400))
    rollout = collect_episode(
        scene,
        requests,
        seed=seed,
        sumo_binary="C:/Program Files (x86)/Eclipse/Sumo/bin/sumo.exe",
        actor=actor,
    )
    save_rollout(path, rollout)
    verify_sources(check_root, registration)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-root", type=Path, required=True)
    parser.add_argument("--worker-seed", type=int)
    parser.add_argument("--worker-arm", choices=ARMS)
    args = parser.parse_args()
    check_root = args.check_root.resolve()
    if args.worker_seed is not None:
        worker(check_root, args.worker_seed, args.worker_arm)
        return
    output = check_root / "corridor"
    with exclusive_run(output):
        if (output / "manifest.json").exists():
            raise RuntimeError("one-shot allocation already exists; inspect instead of repeating")
        previous = read(check_root / "manifest.json")
        result = read(check_root / "summary.json")
        if not result["mechanics_pass"] or result["training_updates"] != 12:
            raise ValueError("required bounded check incomplete")
        for name, expected in previous["source_sha256"].items():
            if sha(ROOT / name) != expected:
                raise ValueError(f"short-check source changed: {name}")
        old = read(CAMPAIGN / "manifest.json")
        for name, expected in old["source_sha256"].items():
            if sha(ROOT / name) != expected:
                raise ValueError(f"cached reference source changed: {name}")
        cache = {}
        for seed, loading in CASES:
            build_scenario(output / f"scene_{seed}", seed, loading, role="development")
            for folder in ("references", "analytical"):
                path = CAMPAIGN / folder / f"development_{seed}.json"
                data = load_rollout(path)
                if (
                    data["summary"]["seed"] != seed
                    or not data["summary"]["team_terminal"]
                    or len(data["intervals"]) != 2400
                ):
                    raise ValueError("invalid cached full reference")
                cache[str(path)] = dict(
                    json_sha256=sha(path), arrays_sha256=sha(path.with_suffix(".npz"))
                )
        names = (
            *SOURCES,
            "experiments/idm/evaluate_damping_v2_check.py",
            "idm/damping_scenario.py",
            "idm/damping_selection.py",
            "experiments/idm/deterministic_sumo.py",
            "env/sumo_utils.py",
            "config/constants.py",
            "idm/contract.py",
            "policies/velocity_matching_damping.py",
        )
        registration = dict(
            schema="native_selectable_corridor_diagnostic/2.0",
            cases=CASES,
            arms=ARMS,
            source_sha256={n: sha(ROOT / n) for n in names},
            checkpoint_sha256=sha(check_root / "checkpoint_012.pt"),
            new_rollouts=6,
            new_global_seconds=14400,
            reused_analytical_rollouts=2,
            cache=cache,
            maximum_workers=2,
            short_gate_pass=result["mechanism_gate_pass"],
            purpose="complete_bounded_physical_diagnosis_not_training_or_acceptance",
            acceptance_sealed=True,
            wall_training_paused=True,
        )
        atomic_json(output / "manifest.json", registration)
        for name in names:
            path = output / "source_snapshot" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((ROOT / name).read_bytes())

        def launch(job):
            seed, arm = job
            with (output / f"{arm}_{seed}.log").open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "experiments.idm.evaluate_damping_v2_check",
                        "--check-root",
                        str(check_root),
                        "--worker-seed",
                        str(seed),
                        "--worker-arm",
                        arm,
                    ],
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
            return dict(seed=seed, arm=arm, exit_code=completed.returncode)

        with ThreadPoolExecutor(max_workers=2) as pool:
            executions = list(pool.map(launch, [(seed, arm) for arm in ARMS for seed, _ in CASES]))
        atomic_json(output / "executions.json", executions)
        if any(row["exit_code"] for row in executions):
            raise RuntimeError("diagnostic failure preserved; inspect logs, do not repeat")
        comparisons = {}
        audits = {}
        for seed, _ in CASES:
            native = load_rollout(output / f"native_{seed}.json")
            prior = load_rollout(CAMPAIGN / "references" / f"development_{seed}.json")
            # New native instrumentation must leave the underlying simulation exact.
            native_delta = case_metrics(native["summary"], prior["summary"])
            assert all(native_delta[k] == 0 for k in native_delta if k != "native_ttc")
            assert native["intervals"] == prior["intervals"]
            rows = {}
            for arm in ("analytical", "expert", "learned"):
                actual = (
                    load_rollout(CAMPAIGN / "analytical" / f"development_{seed}.json")
                    if arm == "analytical"
                    else load_rollout(output / f"{arm}_{seed}.json")
                )
                metrics = case_metrics(actual["summary"], native["summary"])
                rows[arm] = dict(metrics=metrics, absolute=actual["summary"])
                if arm != "analytical":
                    rewards, audit = reward_series(actual, native)
                    batch = prepare_batch(actual, native)
                    # All actor decisions include the final terminal penalty.
                    assert np.isfinite(batch["actor_advantages"]).all()
                    audits[f"{arm}_{seed}"] = dict(
                        audit,
                        first_return=float(rewards.sum()),
                        last_actor_time=int(max(actual["actor_times"])),
                        actor_advantage_min=float(batch["actor_advantages"].min()),
                        actor_advantage_max=float(batch["actor_advantages"].max()),
                    )
            comparisons[str(seed)] = dict(
                native=native["summary"], arms=rows, native_repeat_exact=True
            )
        learned_metrics = [comparisons[str(s)]["arms"]["learned"]["metrics"] for s, _ in CASES]
        constraints = all(
            m["system_time"] <= 2
            and m["source_wait"] <= 2
            and m["arrived"] >= 0
            and m["unfinished"] <= 0
            and m["uninserted"] <= 0
            and m["ttc"] <= max(5, 0.05 * m["native_ttc"])
            and m["collisions"] <= 0
            and m["teleports"] <= 0
            for m in learned_metrics
        )
        benefit = np.mean([m["composite"] for m in learned_metrics]) >= 0.05 and all(
            np.mean([m[k] for m in learned_metrics]) >= 0 for k in ("braking", "energy")
        )
        atomic_json(
            output / "summary.json",
            dict(
                comparisons=comparisons,
                reward_audits=audits,
                diagnostic_constraints_pass=bool(constraints),
                diagnostic_mean_benefit_pass=bool(benefit),
                short_mechanism_pass=result["mechanism_gate_pass"],
                accepted_checkpoint=False,
                new_global_seconds=14400,
                total_repair_new_global_seconds=18000,
                decision="no_expensive_campaign_without_complete_mechanism_and_physical_evidence",
            ),
        )
        print(
            json.dumps(dict(complete=True, constraints=bool(constraints), benefit=bool(benefit))),
            flush=True,
        )


if __name__ == "__main__":
    main()
