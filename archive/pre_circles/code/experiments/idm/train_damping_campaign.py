"""Run one prespecified stage of the three-lineage standalone damping campaign."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import torch

from env.damping_rollout import CRITIC_DIM
from env.native_damping import DampingContract
from experiments.idm.train_native_damping import (
    SOURCES,
    atomic_json,
    collect_cached,
    digest,
    exclusive_run,
    save_checkpoint,
)
from idm.damping_actor import SharedDampingActor
from idm.damping_campaign import (
    CHECKPOINT_EPISODES,
    DEVELOPMENT_CASES,
    OPTIMIZER_SEEDS,
    TRAINING_CASES,
    campaign_manifest,
)
from idm.damping_credit import TeamCostScales
from idm.damping_ppo import PPOContract, TeamCritic, prepare_batch, update_ppo
from idm.damping_scenario import build_scenario
from idm.damping_selection import (
    SelectionContract,
    assess_candidate,
    case_metrics,
    select_candidate,
)
from policies.sensor_local_damping import CHECKPOINT_SCHEMA
from policies.velocity_matching_damping import VelocityMatchingDamping


ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN_SOURCES = (
    *SOURCES,
    "idm/damping_campaign.py",
    "idm/damping_selection.py",
    "policies/velocity_matching_damping.py",
    "experiments/idm/train_damping_campaign.py",
)
UNCHANGED_LEARNER_SOURCES = (
    "env/native_damping.py",
    "env/native_damping_sumo.py",
    "env/damping_rollout.py",
    "idm/damping_actor.py",
    "idm/damping_credit.py",
    "idm/damping_ppo.py",
    "policies/sensor_local_damping.py",
)


def frozen_identity(source, sumo, scenarios):
    previous = json.loads((source / "manifest.json").read_text())
    completed = json.loads((source / "summary.json").read_text())
    if (
        completed["status"] != "bounded_learning_check_complete"
        or completed["policy_global_steps"] != 9600
        or completed["new_policy_global_steps"] != 7200
    ):
        raise ValueError("campaign initialization requires the documented repaired four-update run")
    hashes = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in CAMPAIGN_SOURCES
    }
    for name in UNCHANGED_LEARNER_SOURCES:
        if hashes[name] != previous["source_sha256"][name]:
            raise ValueError(f"verified learning/control implementation changed: {name}")
    cfg = json.loads(json.dumps(DampingContract().manifest()))
    if previous["controller"] != cfg or previous["ppo"] != PPOContract().manifest():
        raise ValueError("verified controller/PPO contract changed")
    sumo_hash = hashlib.sha256(Path(sumo).read_bytes()).hexdigest()
    if sumo_hash != previous["sumo_binary_sha256"]:
        raise ValueError("SUMO version changed")
    if any(
        row["network_sha256"] != previous["scenarios"][0]["network_sha256"]
        for row in scenarios.values()
    ):
        raise ValueError("verified IDM network changed")
    reference = json.loads((source / "reward_reference.json").read_text())
    return dict(
        **campaign_manifest(),
        controller=cfg,
        ppo=PPOContract().manifest(),
        selection=SelectionContract().manifest(),
        analytical=VelocityMatchingDamping().manifest(),
        source_sha256=hashes,
        scenario_manifests=scenarios,
        sumo_binary_sha256=sumo_hash,
        torch_version=str(torch.__version__),
        numpy_version=np.__version__,
        initial_checkpoint_sha256=hashlib.sha256(
            (source / "trainer_004.pt").read_bytes()
        ).hexdigest(),
        reward_scales={k: reference[k] for k in ("braking_vehicle_seconds", "acceleration_energy")},
        original_reward_reference_sha256=digest(reference),
    )


def initialize(root, source, sumo):
    scenes = {}
    for role, cases in (("training", TRAINING_CASES), ("development", DEVELOPMENT_CASES)):
        for seed, loading in cases:
            scenes[(role, seed)] = build_scenario(
                root / "scenarios" / f"{role}_{seed}", seed, loading, role=role
            )
    scenario_manifests = {f"{role}_{seed}": row[2] for (role, seed), row in scenes.items()}
    identity = frozen_identity(source, sumo, scenario_manifests)
    path = root / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError("campaign source/scenario/budget identity changed")
    atomic_json(path, identity)
    for name, expected in identity["source_sha256"].items():
        content = (ROOT / name).read_bytes()
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError("source changed during initialization")
        target = root / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != content:
            raise ValueError("source snapshot mismatch")
        if not target.exists():
            target.write_bytes(content)
    return identity, scenes


def rollout_at(root, scenes, *, role, seed, sumo, path, **kwargs):
    scene, requests, expected = scenes[(role, seed)]
    path.parent.mkdir(parents=True, exist_ok=True)
    row = collect_cached(path, scene, requests, seed=seed, sumo_binary=sumo, **kwargs)
    if (
        not row["summary"]["team_terminal"]
        or row["summary"]["population"] != expected["population"]
    ):
        raise ValueError("incomplete or mismatched campaign rollout")
    return row


def native_reference(root, scenes, role, seed, sumo):
    return rollout_at(
        root,
        scenes,
        role=role,
        seed=seed,
        sumo=sumo,
        path=root / "references" / f"{role}_{seed}.json",
    )


def models_from_state(state, seed):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    actor, critic = SharedDampingActor(), TeamCritic(CRITIC_DIM)
    cfg = PPOContract()
    aopt = torch.optim.Adam(actor.parameters(), lr=cfg.learning_rate)
    copt = torch.optim.Adam(critic.parameters(), lr=cfg.learning_rate)
    rng = np.random.default_rng(seed)
    if state is not None:
        actor.load_state_dict(state["actor"], strict=True)
        critic.load_state_dict(state["critic"], strict=True)
        aopt.load_state_dict(state["actor_optimizer"])
        copt.load_state_dict(state["critic_optimizer"])
        rng.bit_generator.state = state["numpy_rng"]
        torch.set_rng_state(state["torch_rng"])
    return actor, critic, aopt, copt, rng


def evaluate_checkpoint(root, scenes, identity, lineage, episode, actor, sumo):
    directory = root / "development" / f"optimizer_{lineage}_episode_{episode:03}"
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = root / f"lineage_{lineage}" / f"trainer_{episode:03}.pt"
    evaluation_identity = dict(
        campaign=digest(identity),
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        optimizer_seed=lineage,
        optimized_global_steps=episode * 2400,
    )
    header = directory / "manifest.json"
    if header.exists() and json.loads(header.read_text()) != evaluation_identity:
        raise ValueError("development checkpoint identity mismatch")
    atomic_json(header, evaluation_identity)
    rows = []
    for seed, loading in DEVELOPMENT_CASES:
        native = native_reference(root, scenes, "development", seed, sumo)
        analytic = rollout_at(
            root,
            scenes,
            role="development",
            seed=seed,
            sumo=sumo,
            path=root / "analytical" / f"development_{seed}.json",
            actor=VelocityMatchingDamping(),
        )
        policy = rollout_at(
            root,
            scenes,
            role="development",
            seed=seed,
            sumo=sumo,
            path=directory / f"policy_{seed}.json",
            actor=actor,
        )
        rows.append(
            dict(
                case=[seed, loading],
                policy=policy["summary"],
                native=native["summary"],
                analytical=analytic["summary"],
                analytical_relative_to_native=case_metrics(analytic["summary"], native["summary"]),
            )
        )
    assessment = assess_candidate(rows)
    record = dict(
        **evaluation_identity, checkpoint_path=str(checkpoint), cases=rows, assessment=assessment
    )
    atomic_json(directory / "summary.json", record)
    candidates = [
        json.loads(path.read_text())
        for path in sorted((root / "development").glob("*/summary.json"))
    ]
    chosen = select_candidate(candidates)
    atomic_json(
        root / "development_selection.json",
        dict(
            candidates=[
                {
                    key: row[key]
                    for key in (
                        "optimizer_seed",
                        "optimized_global_steps",
                        "checkpoint_path",
                        "assessment",
                    )
                }
                for row in candidates
            ],
            selected=None
            if chosen is None
            else {
                key: chosen[key]
                for key in ("optimizer_seed", "optimized_global_steps", "checkpoint_path")
            },
            accepted_checkpoint=False,
            acceptance_evaluation_authorized=False,
        ),
    )
    print(
        json.dumps(
            {
                "development_complete": episode,
                "optimizer_seed": lineage,
                "eligible": assessment["eligible"],
                "failures": assessment["failures"],
            }
        ),
        flush=True,
    )


def run_lineage(root, source, scenes, identity, sumo, seed, through):
    directory = root / f"lineage_{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    run_id = {**identity, "optimizer_seed": seed}
    checkpoints = sorted(directory.glob("trainer_*.pt"))
    if checkpoints:
        state = torch.load(checkpoints[-1], map_location="cpu", weights_only=True)
        if state["identity"] != run_id:
            raise ValueError("lineage checkpoint identity mismatch")
    elif seed == 71:
        state = torch.load(source / "trainer_004.pt", map_location="cpu", weights_only=True)
        if state["iteration"] != 4 or state["identity"]["ppo"] != PPOContract().manifest():
            raise ValueError("invalid repaired initialization checkpoint")
    else:
        state = None
    actor, critic, aopt, copt, rng = models_from_state(state, seed)
    begin, history = (0, []) if state is None else (state["iteration"], state["history"])
    if begin > through:
        raise ValueError("cannot move a lineage backward")
    if not checkpoints:
        save_checkpoint(
            directory / f"trainer_{begin:03}.pt",
            actor,
            critic,
            aopt,
            copt,
            rng,
            identity=run_id,
            iteration=begin,
            history=history,
        )
    # Finish an interrupted checkpoint evaluation before any later optimization.
    resume_rng = torch.get_rng_state()
    for point in CHECKPOINT_EPISODES:
        if (
            point <= begin
            and not (
                root / "development" / f"optimizer_{seed}_episode_{point:03}" / "summary.json"
            ).exists()
        ):
            saved = torch.load(
                directory / f"trainer_{point:03}.pt", map_location="cpu", weights_only=True
            )
            frozen = SharedDampingActor()
            frozen.load_state_dict(saved["actor"])
            evaluate_checkpoint(root, scenes, identity, seed, point, frozen, sumo)
    torch.set_rng_state(resume_rng)
    for iteration in range(begin, through):
        index = (iteration - (4 if seed == 71 else 0)) % len(TRAINING_CASES)
        scenario_seed, loading = TRAINING_CASES[index]
        native = native_reference(root, scenes, "training", scenario_seed, sumo)
        row = rollout_at(
            root,
            scenes,
            role="training",
            seed=scenario_seed,
            sumo=sumo,
            path=directory / f"policy_{iteration:03}.json",
            actor=actor,
            critic=critic,
            stochastic=True,
            random_seed=seed * 1000 + iteration,
        )
        with torch.no_grad():
            logp = actor.log_prob(
                torch.from_numpy(row["actor_observations"]), torch.from_numpy(row["actions"])
            )
            if not np.allclose(logp.numpy(), row["old_logp"], atol=1e-4, rtol=1e-5):
                raise ValueError("stored campaign rollout is not on-policy")
        scales = TeamCostScales(
            **identity["reward_scales"], requested_population=row["summary"]["population"]
        )
        batch = prepare_batch(row, native, scales)
        metrics = update_ppo(actor, critic, aopt, copt, batch, rng=rng)
        if metrics["actor_parameter_change_l2"] <= 0:
            raise RuntimeError("learner failed to update the actor")
        history.append(
            dict(
                iteration=iteration + 1,
                case=[scenario_seed, loading],
                global_steps=(iteration + 1) * 2400,
                learner=metrics,
                reward_audit=batch["reward_audit"],
                summary=row["summary"],
            )
        )
        path = directory / f"trainer_{iteration + 1:03}.pt"
        save_checkpoint(
            path,
            actor,
            critic,
            aopt,
            copt,
            rng,
            identity=run_id,
            iteration=iteration + 1,
            history=history,
        )
        restored = SharedDampingActor()
        restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["actor"])
        probe = torch.from_numpy(row["actor_observations"][:256])
        with torch.no_grad():
            if not torch.equal(actor(probe), restored(probe)):
                raise RuntimeError("campaign inference roundtrip failed")
        atomic_json(
            directory / "progress.json",
            dict(
                status="training",
                optimized_episodes=iteration + 1,
                optimized_global_steps=(iteration + 1) * 2400,
                latest=history[-1],
                accepted_checkpoint=False,
            ),
        )
        print(
            json.dumps(
                {
                    "optimizer_seed": seed,
                    "episode": iteration + 1,
                    "case": [scenario_seed, loading],
                    "reward": batch["reward_audit"],
                    "learner": metrics,
                }
            ),
            flush=True,
        )
        if iteration + 1 in CHECKPOINT_EPISODES:
            # Preserve learner RNG across read-only development collection.
            torch_rng = torch.get_rng_state()
            evaluate_checkpoint(root, scenes, identity, seed, iteration + 1, actor, sumo)
            torch.set_rng_state(torch_rng)
    mapping = scenes[("training", TRAINING_CASES[0][0])][2]
    # Inference map comes from an observed training episode, never the privileged critic.
    observed_map = history[-1]["summary"]["observation_map"]
    inference = directory / f"inference_{through:03}.pt"
    if not inference.exists():
        with inference.open("xb") as stream:
            torch.save(
                dict(
                    schema=CHECKPOINT_SCHEMA,
                    controller=DampingContract().manifest(),
                    hidden=actor.mean_head.in_features,
                    observation_map=observed_map,
                    actor_state=actor.state_dict(),
                    accepted_checkpoint=False,
                    scenario_family=mapping["schema"],
                ),
                stream,
            )
    atomic_json(
        directory / "stage_summary.json",
        dict(
            status="stage_complete",
            optimizer_seed=seed,
            optimized_episodes=through,
            optimized_global_steps=through * 2400,
            accepted_checkpoint=False,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initialization", type=Path, required=True)
    parser.add_argument("--optimizer-seed", type=int, choices=OPTIMIZER_SEEDS, required=True)
    parser.add_argument("--through-episode", type=int, choices=CHECKPOINT_EPISODES, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    root, source = args.output.resolve(), args.initialization.resolve()
    sumo = shutil.which("sumo")
    if not sumo:
        raise RuntimeError("SUMO must be on PATH")
    if root.exists() and any(root.iterdir()) and not args.resume:
        raise FileExistsError("campaign already exists; exact-identity resume is required")
    with exclusive_run(root):
        identity, scenes = initialize(root, source, sumo)
        if args.prepare_only:
            print(
                json.dumps(
                    {
                        "status": "prepared",
                        "scenario_count": len(scenes),
                        "budget": identity["budget"],
                        "new_simulated_seconds": 0,
                    }
                ),
                flush=True,
            )
            return
        atomic_json(
            root / "active_stage.json",
            dict(
                optimizer_seed=args.optimizer_seed,
                through_episode=args.through_episode,
                status="running",
                pid=os.getpid(),
            ),
        )
        try:
            run_lineage(
                root, source, scenes, identity, sumo, args.optimizer_seed, args.through_episode
            )
        except Exception as exc:
            atomic_json(
                root / "active_stage.json",
                dict(
                    optimizer_seed=args.optimizer_seed,
                    through_episode=args.through_episode,
                    status="failed",
                    pid=os.getpid(),
                    error=str(exc),
                ),
            )
            raise
        atomic_json(
            root / "active_stage.json",
            dict(
                optimizer_seed=args.optimizer_seed,
                through_episode=args.through_episode,
                status="complete",
            ),
        )


if __name__ == "__main__":
    main()
