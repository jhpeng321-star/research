"""Bounded learning-signal check on four already exposed development cases.

This is a separate PyTorch PPO implementation, not an RLlib success claim.
Complete 2400-second team episodes are the only admitted training samples.
"""

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import torch

from env.damping_rollout import CRITIC_DIM, collect_episode
from env.native_damping import DampingContract
from idm.damping_actor import SharedDampingActor
from idm.damping_credit import TeamCostScales
from idm.damping_ppo import PPOContract, TeamCritic, prepare_batch, update_ppo
from idm.damping_scenario import EPISODE_SECONDS, build_scenario
from policies.sensor_local_damping import CHECKPOINT_SCHEMA


CASES = ((4107, 2000), (4109, 2000), (4107, 2200), (4109, 2200))
ROOT = Path(__file__).resolve().parents[2]
SOURCES = (
    "env/native_damping.py",
    "env/native_damping_sumo.py",
    "env/damping_rollout.py",
    "env/sumo_utils.py",
    "idm/damping_actor.py",
    "idm/damping_credit.py",
    "idm/damping_ppo.py",
    "idm/damping_scenario.py",
    "idm/contract.py",
    "config/constants.py",
    "policies/sensor_local_damping.py",
    "experiments/idm/deterministic_sumo.py",
    "experiments/idm/train_native_damping.py",
)
ARRAY_KEYS = (
    "actor_observations",
    "actions",
    "old_logp",
    "actor_times",
    "critic_observations",
    "old_values",
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path, payload):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


@contextmanager
def exclusive_run(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / "writer.lock").open("a+b") as stream:
        stream.seek(0, 2)
        if not stream.tell():
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def save_rollout(path, rollout):
    """JSON is the commit marker; an incomplete collection is never reused."""
    temp = path.with_suffix(".npz.tmp")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, **{key: rollout[key] for key in ARRAY_KEYS})
    temp.replace(path.with_suffix(".npz"))
    atomic_json(path, {key: value for key, value in rollout.items() if key not in ARRAY_KEYS})


def load_rollout(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    with np.load(path.with_suffix(".npz"), allow_pickle=False) as arrays:
        payload.update({key: arrays[key] for key in ARRAY_KEYS})
    return payload


def run_identity(sumo_binary):
    return {
        "schema": "native_damping_learning_check/2",
        "cases": [list(case) for case in CASES],
        "controller": json.loads(json.dumps(DampingContract().manifest())),
        "ppo": PPOContract().manifest(),
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES
        },
        "sumo_binary_sha256": hashlib.sha256(Path(sumo_binary).read_bytes()).hexdigest(),
        "torch_version": str(torch.__version__),
        "numpy_version": np.__version__,
        "policy_global_step_cap": 19200,
        "reference_global_step_cap": 9600,
        "claim": "development_learning_signal_only_not_accepted_damping",
    }


def collect_cached(path, scene, requests, **kwargs):
    if path.exists():
        return load_rollout(path)
    started = path.with_suffix(".started.json")
    if started.exists():
        raise RuntimeError(
            f"Interrupted rollout at {path}; budget must be reconciled before another simulation"
        )
    atomic_json(
        started,
        {
            "pid": os.getpid(),
            "seed": kwargs["seed"],
            "allocated_global_steps": kwargs.get("diagnostic_steps") or EPISODE_SECONDS,
        },
    )
    rollout = collect_episode(scene, requests, **kwargs)
    save_rollout(path, rollout)
    return rollout


def import_native_references(source, root, identity, episodes):
    """Reuse only physical baselines from the specifically stopped v1 check."""
    previous = json.loads((source / "manifest.json").read_text())
    stopped = json.loads((source / "summary.json").read_text())
    if stopped["status"] != "stopped_service_penalty_erased_by_episode_centering":
        raise ValueError("reference reuse requires the documented stopped v1 check")
    if (episodes - 1) * EPISODE_SECONDS > stopped["remaining_original_19200_step_budget"]:
        raise ValueError("repair would exceed the original allocated training budget")
    for name, value in identity["source_sha256"].items():
        if name not in {"idm/damping_ppo.py", "experiments/idm/train_native_damping.py"}:
            if previous["source_sha256"][name] != value:
                raise ValueError(f"physical reference implementation changed: {name}")
    for key in (
        "scenarios",
        "controller",
        "sumo_binary_sha256",
        "numpy_version",
        "torch_version",
        "optimizer_seed",
    ):
        if previous[key] != identity[key]:
            raise ValueError(f"physical reference identity mismatch: {key}")
    reference = json.loads((source / "reward_reference.json").read_text())
    for index in range(len(CASES)):
        row = load_rollout(source / f"native_{index}.json")
        if (
            digest({"summary": row["summary"], "intervals": row["intervals"]})
            != reference["native_digests"][index]
        ):
            raise ValueError("native reference evidence changed")
        for suffix in ("json", "npz"):
            name = f"native_{index}.{suffix}"
            content = (source / name).read_bytes()
            target = root / name
            if target.exists() and target.read_bytes() != content:
                raise ValueError("refusing to overwrite different native data")
            if not target.exists():
                target.write_bytes(content)
    # This sample preceded every failed v1 update. Verify its generating
    # actor/critic before the corrected learner can consume it.
    for source_name, target_name in (
        ("policy_000.json", "policy_000.json"),
        ("policy_000.npz", "policy_000.npz"),
        ("trainer_000.pt", "source_initial_policy.pt"),
    ):
        content = (source / source_name).read_bytes()
        target = root / target_name
        if target.exists() and target.read_bytes() != content:
            raise ValueError("initial on-policy evidence mismatch")
        if not target.exists():
            target.write_bytes(content)
    return dict(
        source_manifest_sha256=digest(previous),
        previous_allocated_policy_global_steps=stopped["allocated_policy_global_steps"],
        new_reference_global_steps=0,
        reused_reference_global_steps=len(CASES) * EPISODE_SECONDS,
        reused_initial_policy_global_steps=EPISODE_SECONDS,
        source_initial_policy_sha256=hashlib.sha256(
            (source / "trainer_000.pt").read_bytes()
        ).hexdigest(),
    )


def save_checkpoint(
    path, actor, critic, actor_opt, critic_opt, rng, *, identity, iteration, history
):
    payload = dict(
        schema="native_damping_trainer/1",
        identity=identity,
        iteration=iteration,
        global_steps=iteration * EPISODE_SECONDS,
        actor=actor.state_dict(),
        critic=critic.state_dict(),
        actor_optimizer=actor_opt.state_dict(),
        critic_optimizer=critic_opt.state_dict(),
        numpy_rng=rng.bit_generator.state,
        torch_rng=torch.get_rng_state(),
        history=history,
        accepted_checkpoint=False,
    )
    temp = path.with_suffix(".pt.tmp")
    torch.save(payload, temp)
    temp.replace(path)


def train(root, scenes, identity, sumo_binary, episodes, optimizer_seed):
    references = []
    for index, ((seed, _), (scene, requests, _)) in enumerate(zip(CASES, scenes)):
        references.append(
            collect_cached(
                root / f"native_{index}.json", scene, requests, seed=seed, sumo_binary=sumo_binary
            )
        )
    brakes = max(
        1.0, float(np.mean([r["summary"]["totals"]["braking_vehicle_seconds"] for r in references]))
    )
    energy = max(
        1.0, float(np.mean([r["summary"]["totals"]["acceleration_energy"] for r in references]))
    )
    reward_reference = dict(
        braking_vehicle_seconds=brakes,
        acceleration_energy=energy,
        native_digests=[
            digest({"summary": r["summary"], "intervals": r["intervals"]}) for r in references
        ],
    )
    atomic_json(root / "reward_reference.json", reward_reference)
    identity = {**identity, "reward_reference": reward_reference}
    torch.set_num_threads(1)
    torch.manual_seed(optimizer_seed)
    actor, critic = SharedDampingActor(), TeamCritic(CRITIC_DIM)
    if "reference_reuse" in identity:
        initial = torch.load(
            root / "source_initial_policy.pt", map_location="cpu", weights_only=True
        )
        for name, model in (("actor", actor), ("critic", critic)):
            if model.state_dict().keys() != initial[name].keys() or any(
                not torch.equal(value, initial[name][key])
                for key, value in model.state_dict().items()
            ):
                raise ValueError("reused initial rollout was generated by different weights")
        raw = load_rollout(root / "policy_000.json")
        with torch.no_grad():
            logp = actor.log_prob(
                torch.from_numpy(raw["actor_observations"]), torch.from_numpy(raw["actions"])
            )
            if not np.allclose(logp.numpy(), raw["old_logp"], atol=1e-4, rtol=1e-5):
                raise ValueError("reused initial rollout is not on-policy")
    cfg = PPOContract()
    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.learning_rate)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.learning_rate)
    rng = np.random.default_rng(optimizer_seed)
    history, begin = [], 0
    checkpoints = sorted(root.glob("trainer_*.pt"))
    if checkpoints:
        state = torch.load(checkpoints[-1], map_location="cpu", weights_only=True)
        if state["identity"] != identity:
            raise ValueError("resume source/scenario/reward identity mismatch")
        actor.load_state_dict(state["actor"])
        critic.load_state_dict(state["critic"])
        actor_opt.load_state_dict(state["actor_optimizer"])
        critic_opt.load_state_dict(state["critic_optimizer"])
        rng.bit_generator.state = state["numpy_rng"]
        torch.set_rng_state(state["torch_rng"])
        history, begin = state["history"], state["iteration"]
    else:
        save_checkpoint(
            root / "trainer_000.pt",
            actor,
            critic,
            actor_opt,
            critic_opt,
            rng,
            identity=identity,
            iteration=0,
            history=[],
        )
    for iteration in range(begin, episodes):
        index = iteration % len(CASES)
        seed, _ = CASES[index]
        scene, requests, _ = scenes[index]
        rollout = collect_cached(
            root / f"policy_{iteration:03}.json",
            scene,
            requests,
            seed=seed,
            sumo_binary=sumo_binary,
            actor=actor,
            critic=critic,
            stochastic=True,
            random_seed=optimizer_seed * 1000 + iteration,
        )
        scales = TeamCostScales(brakes, energy, len(requests))
        batch = prepare_batch(rollout, references[index], scales, cfg)
        probe = torch.from_numpy(rollout["actor_observations"][:256])
        with torch.no_grad():
            before = actor(probe).clone()
        metrics = update_ppo(actor, critic, actor_opt, critic_opt, batch, rng=rng, contract=cfg)
        with torch.no_grad():
            after = actor(probe).clone()
        record = dict(
            iteration=iteration + 1,
            global_steps=(iteration + 1) * EPISODE_SECONDS,
            case=list(CASES[index]),
            reward_audit=batch["reward_audit"],
            learner=metrics,
            deterministic_probe_change_rms=float((after - before).square().mean().sqrt()),
            exploratory_action_mean=float(rollout["actions"].mean()),
            exploratory_action_std=float(rollout["actions"].std()),
            summary=rollout["summary"],
        )
        history.append(record)
        path = root / f"trainer_{iteration + 1:03}.pt"
        save_checkpoint(
            path,
            actor,
            critic,
            actor_opt,
            critic_opt,
            rng,
            identity=identity,
            iteration=iteration + 1,
            history=history,
        )
        restored = SharedDampingActor()
        restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["actor"])
        with torch.no_grad():
            if not torch.equal(restored(probe), after):
                raise RuntimeError("checkpoint inference roundtrip failed")
        atomic_json(
            root / "progress.json",
            dict(status="learning_check", history=history, accepted_checkpoint=False),
        )
        print(
            json.dumps(
                {
                    "iteration": iteration + 1,
                    "global_steps": record["global_steps"],
                    "reward": record["reward_audit"],
                    "learner": metrics,
                }
            ),
            flush=True,
        )
    mapping = references[0]["summary"]["observation_map"]
    inference = dict(
        schema=CHECKPOINT_SCHEMA,
        controller=DampingContract().manifest(),
        hidden=actor.mean_head.in_features,
        observation_map=mapping,
        accepted_checkpoint=False,
        actor_state=actor.state_dict(),
    )
    inference_path = root / f"inference_{episodes:03}.pt"
    if not inference_path.exists():
        with inference_path.open("xb") as stream:
            torch.save(inference, stream)
    atomic_json(
        root / "summary.json",
        dict(
            status="bounded_learning_check_complete",
            accepted_checkpoint=False,
            policy_global_steps=episodes * EPISODE_SECONDS,
            reused_initial_policy_global_steps=identity.get("reference_reuse", {}).get(
                "reused_initial_policy_global_steps", 0
            ),
            new_policy_global_steps=episodes * EPISODE_SECONDS
            - identity.get("reference_reuse", {}).get("reused_initial_policy_global_steps", 0),
            reference_global_steps=len(CASES) * EPISODE_SECONDS,
            new_reference_global_steps=0
            if "reference_reuse" in identity
            else len(CASES) * EPISODE_SECONDS,
            total_new_simulated_seconds=episodes * EPISODE_SECONDS
            - identity.get("reference_reuse", {}).get("reused_initial_policy_global_steps", 0)
            + (0 if "reference_reuse" in identity else len(CASES) * EPISODE_SECONDS),
            previous_allocated_policy_global_steps=identity.get("reference_reuse", {}).get(
                "previous_allocated_policy_global_steps", 0
            ),
            optimizer_seed=optimizer_seed,
            history=history,
            limitation="No prospective evaluation or deterministic efficacy comparison in this check",
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["diagnostic", "train"], required=True)
    parser.add_argument("--episodes", type=int, choices=range(1, 9), default=8)
    parser.add_argument("--optimizer-seed", type=int, default=71)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reference-output", type=Path)
    args = parser.parse_args()
    sumo_binary = shutil.which("sumo")
    if not sumo_binary:
        raise RuntimeError("SUMO binary is required on PATH")
    root = args.output.resolve()
    if root.exists() and any(root.iterdir()) and not args.resume:
        raise FileExistsError("nonempty run directory; use --resume only for the same frozen run")
    with exclusive_run(root):
        scenes = [build_scenario(root / f"scenario_{i}", *case) for i, case in enumerate(CASES)]
        identity = {
            **run_identity(sumo_binary),
            "scenarios": [row[2] for row in scenes],
            "mode": args.mode,
            "episodes": args.episodes,
            "optimizer_seed": args.optimizer_seed,
        }
        if args.reference_output:
            if args.mode != "train":
                raise ValueError("reference reuse is training-only")
            identity["reference_reuse"] = import_native_references(
                args.reference_output.resolve(), root, identity, args.episodes
            )
        manifest_path = root / "manifest.json"
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != identity:
            raise ValueError("run identity changed; refusing reuse")
        atomic_json(manifest_path, identity)
        if args.mode == "train":
            train(root, scenes, identity, sumo_binary, args.episodes, args.optimizer_seed)
        else:
            torch.manual_seed(args.optimizer_seed)
            actor = SharedDampingActor()
            scene, requests, _ = scenes[0]
            rows = {}
            for arm in ("native", "zero", "stochastic"):
                rows[arm] = collect_cached(
                    root / f"{arm}.json",
                    scene,
                    requests,
                    seed=CASES[0][0],
                    sumo_binary=sumo_binary,
                    actor=None if arm == "native" else actor,
                    stochastic=arm == "stochastic",
                    random_seed=args.optimizer_seed,
                    diagnostic_steps=300,
                )
            checks = dict(
                zero_interval_cost_identity=rows["native"]["intervals"]
                == rows["zero"]["intervals"],
                zero_no_ramps=rows["zero"]["summary"]["commands"].get("slow_down", 0) == 0,
                stochastic_issues_ramps=rows["stochastic"]["summary"]["commands"].get(
                    "slow_down", 0
                )
                > 0,
                all_nonterminal=all(not row["summary"]["team_terminal"] for row in rows.values()),
            )
            atomic_json(
                root / "summary.json",
                dict(
                    checks=checks,
                    passed=all(checks.values()),
                    summaries={key: row["summary"] for key, row in rows.items()},
                    accepted_checkpoint=False,
                ),
            )
            if not all(checks.values()):
                raise RuntimeError("collector integration diagnostic failed")


if __name__ == "__main__":
    main()
