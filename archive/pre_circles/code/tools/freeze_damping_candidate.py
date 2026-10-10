"""Freeze a prospective-test candidate after all registered development work.

Does not launch SUMO, generate acceptance cases, or mark a checkpoint accepted.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from env.native_damping import DampingContract
from experiments.idm.train_native_damping import atomic_json, digest, exclusive_run, load_rollout
from idm.damping_acceptance import AcceptanceContract
from idm.damping_actor import SharedDampingActor
from idm.damping_campaign import (
    ACCEPTANCE_CASES,
    CHECKPOINT_EPISODES,
    DEVELOPMENT_CASES,
    OPTIMIZER_SEEDS,
)
from idm.damping_scenario import request_schedule
from idm.damping_selection import SelectionContract, assess_candidate, select_candidate
from policies.sensor_local_damping import CHECKPOINT_SCHEMA

ROOT = Path(__file__).resolve().parents[1]
REGISTRATION = ROOT / "runs/probe/native_damping_acceptance_preregistration_v1/manifest.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_complete_development(root):
    root = Path(root)
    for seed in OPTIMIZER_SEEDS:
        path = root / f"lineage_{seed}" / "stage_summary.json"
        if not path.exists():
            raise RuntimeError("candidate export requires all three completed full lineages")
        state = json.loads(path.read_text())
        if state.get("status") != "stage_complete" or state.get("optimized_episodes") != 100:
            raise RuntimeError("candidate export requires all three completed full lineages")
    if json.loads((root / "active_stage.json").read_text()).get("status") != "complete":
        raise RuntimeError("campaign must be terminal before candidate export")
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["selection"] != SelectionContract().manifest():
        raise ValueError("frozen selection criteria changed")
    for name, expected in manifest["source_sha256"].items():
        if sha(ROOT / name) != expected or sha(root / "source_snapshot" / name) != expected:
            raise ValueError(f"campaign source verification failed: {name}")
    scenario_ids = {}
    for seed, loading in DEVELOPMENT_CASES:
        folder = root / "scenarios" / f"development_{seed}"
        requests = request_schedule(seed, loading, role="development")
        requested = json.dumps([asdict(row) for row in requests], sort_keys=True).encode()
        scenario_ids[seed] = hashlib.sha256(
            requested
            + (folder / "scenario.sumocfg").read_bytes()
            + (folder / "requests.rou.xml").read_bytes()
        ).hexdigest()
    records, evidence = [], {}
    for seed in OPTIMIZER_SEEDS:
        for episode in CHECKPOINT_EPISODES:
            directory = root / "development" / f"optimizer_{seed}_episode_{episode:03}"
            path = directory / "summary.json"
            if not path.exists():
                raise RuntimeError("all nine prespecified development evaluations are required")
            record = json.loads(path.read_text())
            checkpoint = root / f"lineage_{seed}" / f"trainer_{episode:03}.pt"
            if (
                record["campaign"] != digest(manifest)
                or record["optimizer_seed"] != seed
                or record["optimized_global_steps"] != episode * 2400
                or record["checkpoint_sha256"] != sha(checkpoint)
            ):
                raise ValueError("development evaluation/checkpoint identity mismatch")
            if len(record["cases"]) != len(DEVELOPMENT_CASES):
                raise ValueError("incomplete development record")
            for case in record["cases"]:
                case_seed, loading = case["case"]
                if (case_seed, loading) not in DEVELOPMENT_CASES:
                    raise ValueError("unregistered development case")
                paths = dict(
                    policy=directory / f"policy_{case_seed}.json",
                    native=root / "references" / f"development_{case_seed}.json",
                    analytical=root / "analytical" / f"development_{case_seed}.json",
                )
                for arm, data_path in paths.items():
                    raw = json.loads(data_path.read_text())
                    summary = raw["summary"]
                    if (
                        summary != case[arm]
                        or summary["seed"] != case_seed
                        or summary["scenario_identity"] != scenario_ids[case_seed]
                    ):
                        raise ValueError("stored case does not match its trajectory/scenario")
                    if len(raw["intervals"]) != 2400 or any(
                        not np.isclose(sum(row[key] for row in raw["intervals"]), value)
                        for key, value in summary["totals"].items()
                    ):
                        raise ValueError("physical trajectory cost accounting mismatch")
                    evidence[str(data_path.relative_to(root))] = sha(data_path)
            assessment = assess_candidate(record["cases"])
            if assessment != record["assessment"]:
                raise ValueError("stored eligibility disagrees with recomputed selection")
            evidence[str(path.relative_to(root))] = sha(path)
            records.append({**record, "checkpoint_path": str(checkpoint)})
    return manifest, records, evidence


def verify_deterministic_actions(actor, rollout):
    times = rollout["actor_times"]
    if not len(times) or np.any(np.diff(times) < 0):
        raise ValueError("expected ordered nonempty CAV action records")
    if len(times) != len(rollout["actions"]) or len(times) != len(rollout["actor_observations"]):
        raise ValueError("CAV action record alignment mismatch")
    # Reproduce original per-tick batch shapes to avoid a different floating
    # point reduction near the exact-zero native-passthrough threshold.
    boundaries = np.r_[0, np.flatnonzero(np.diff(times)) + 1, len(times)]
    actor.eval()
    with torch.no_grad():
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            predicted = actor(torch.from_numpy(rollout["actor_observations"][start:end]))
            if not torch.equal(predicted, torch.from_numpy(rollout["actions"][start:end])):
                raise ValueError("checkpoint does not reproduce recorded deterministic actions")


def freeze(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if not root.is_dir():
        raise ValueError("campaign directory does not exist")
    if output.exists():
        raise FileExistsError("candidate bundles are immutable")
    with exclusive_run(root):
        manifest, candidates, evidence = read_complete_development(root)
        registration = json.loads(REGISTRATION.read_text())
        if (
            registration["campaign_manifest_sha256"] != sha(root / "manifest.json")
            or registration["acceptance_contract"] != AcceptanceContract().manifest()
            or registration["acceptance_cases"] != [list(case) for case in ACCEPTANCE_CASES]
        ):
            raise ValueError("prospective acceptance registration changed")
        for name, expected in registration["source_sha256"].items():
            if (
                sha(ROOT / name) != expected
                or sha(REGISTRATION.parent / "source" / name) != expected
            ):
                raise ValueError(f"prospective acceptance source changed: {name}")
        selected = select_candidate(candidates)
        if selected is None:
            raise RuntimeError("no development-eligible candidate; acceptance remains sealed")
        checkpoint = Path(selected["checkpoint_path"])
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if payload["identity"] != {**manifest, "optimizer_seed": selected["optimizer_seed"]}:
            raise ValueError("selected trainer identity mismatch")
        if payload["iteration"] * 2400 != selected["optimized_global_steps"]:
            raise ValueError("selected trainer iteration mismatch")
        torch.set_num_threads(1)
        actor = SharedDampingActor()
        actor.load_state_dict(payload["actor"], strict=True)
        directory = (
            root
            / "development"
            / f"optimizer_{selected['optimizer_seed']}_episode_{payload['iteration']:03}"
        )
        maps = []
        for seed, _ in DEVELOPMENT_CASES:
            path = directory / f"policy_{seed}.json"
            rollout = load_rollout(path)
            verify_deterministic_actions(actor, rollout)
            maps.append(rollout["summary"]["observation_map"])
            evidence[str(path.with_suffix(".npz").relative_to(root))] = sha(
                path.with_suffix(".npz")
            )
        if any(mapping != maps[0] for mapping in maps):
            raise ValueError("development inference maps differ")
        output.mkdir(parents=True, exist_ok=False)
        weights = output / "candidate_weights.pt"
        (output / "acceptance_registration.json").write_bytes(REGISTRATION.read_bytes())
        torch.save(
            dict(
                schema=CHECKPOINT_SCHEMA,
                controller=DampingContract().manifest(),
                hidden=actor.mean_head.in_features,
                observation_map=maps[0],
                accepted_checkpoint=False,
                actor_state=actor.state_dict(),
            ),
            weights,
        )
        for name in manifest["source_sha256"]:
            target = output / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((root / "source_snapshot" / name).read_bytes())
        acceptance_sources = {}
        for name in registration["source_sha256"]:
            target = output / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / name).read_bytes())
            acceptance_sources[name] = sha(target)
        result = dict(
            schema="native_damping_prospective_candidate/1",
            campaign_manifest=manifest,
            optimizer_seed=selected["optimizer_seed"],
            optimized_global_steps=selected["optimized_global_steps"],
            selected_trainer_sha256=sha(checkpoint),
            candidate_weights_sha256=sha(weights),
            development_evidence_sha256=evidence,
            selected_assessment=selected["assessment"],
            acceptance_contract=AcceptanceContract().manifest(),
            acceptance_analysis_source_sha256=sha(ROOT / "idm/damping_acceptance.py"),
            acceptance_bundle_source_sha256=acceptance_sources,
            acceptance_registration_sha256=sha(REGISTRATION),
            acceptance_cases=[list(case) for case in ACCEPTANCE_CASES],
            accepted_checkpoint=False,
            wall_generation_authorized=False,
            status="candidate_frozen_for_prospective_testing_only",
        )
        # Manifest is the commit marker and is written after every artifact.
        atomic_json(output / "manifest.json", result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = freeze(args.campaign, args.output)
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("schema", "optimizer_seed", "optimized_global_steps", "status")
            }
        )
    )


if __name__ == "__main__":
    main()
