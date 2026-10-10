"""Two exposed-case magnitude ablations; never training or acceptance data."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil

import torch
from torch import nn

from experiments.idm.train_native_damping import atomic_json, collect_cached, digest, exclusive_run
from idm.damping_actor import SharedDampingActor
from idm.damping_scenario import request_schedule
from idm.damping_selection import case_metrics

ROOT = Path(__file__).resolve().parents[2]
CASES = ((911001, 2000), (911003, 2200))
EPSILON = 1e-6
CHECKPOINT_SHA = "9aef1dae1c2fc0b627476d1287693f86cbebb3568fddee6f01098df9d3f48b60"


class MagnitudeAblationActor(nn.Module):
    """Retain learned sign/zero on current local inputs; remove most magnitude."""

    def __init__(self, base):
        super().__init__()
        self.base = base
        self.contract = base.contract
        if EPSILON <= self.contract.zero_tolerance:
            raise ValueError("ablation must remain above native-zero tolerance")

    def forward(self, observations):
        return self.base(observations).clamp(-EPSILON, EPSILON)

    def log_prob(self, observations, actions):
        if actions.shape != self(observations).shape:
            raise ValueError("invalid diagnostic action shape")
        # Deterministic storage placeholder, not a stochastic likelihood.
        # These trajectories must never enter PPO or checkpoint selection.
        return torch.zeros(actions.shape[:-1], dtype=actions.dtype, device=actions.device)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(campaign, output, sumo):
    manifest = read(campaign / "manifest.json")
    for name, expected in manifest["source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"frozen source changed: {name}")
    if not sumo or sha(Path(sumo)) != manifest["sumo_binary_sha256"]:
        raise ValueError("SUMO binary mismatch")
    checkpoint = campaign / "lineage_71/trainer_020.pt"
    if sha(checkpoint) != CHECKPOINT_SHA:
        raise ValueError("diagnostic checkpoint mismatch")
    development = campaign / "development/optimizer_71_episode_020"
    evaluation = read(development / "summary.json")
    if evaluation["campaign"] != digest(manifest) or evaluation["checkpoint_sha256"] != CHECKPOINT_SHA:
        raise ValueError("completed development identity mismatch")
    inputs, evidence = [], {}
    for seed, loading in CASES:
        requests = request_schedule(seed, loading, role="development")
        folder = campaign / "scenarios" / f"development_{seed}"
        scene = folder / "scenario.sumocfg"
        identity = hashlib.sha256(
            json.dumps([asdict(r) for r in requests], sort_keys=True).encode()
            + scene.read_bytes() + (folder / "requests.rou.xml").read_bytes()
        ).hexdigest()
        native_path = campaign / "references" / f"development_{seed}.json"
        policy_path = development / f"policy_{seed}.json"
        native, policy = read(native_path)["summary"], read(policy_path)["summary"]
        for summary in (native, policy):
            if (summary["seed"] != seed or summary["scenario_identity"] != identity
                    or summary["population"] != len(requests) or not summary["team_terminal"]
                    or summary["global_control_steps"] != 2400):
                raise ValueError("cached comparison identity or completion mismatch")
        evidence[str(native_path)] = sha(native_path)
        evidence[str(policy_path)] = sha(policy_path)
        inputs.append((seed, loading, scene, requests, native, policy))
    registration = dict(
        schema="native_damping_magnitude_ablation/1", campaign=digest(manifest),
        checkpoint_sha256=CHECKPOINT_SHA, cases=[list(c) for c in CASES],
        epsilon_mps2=EPSILON, policy="clip(frozen_deterministic_actor(current_local_history), -epsilon, epsilon)",
        selected_after_outcomes=True, role="exposed_mechanism_diagnosis_only",
        new_full_rollouts=2, allocated_simulator_seconds=4800,
        conservative_development_ceiling_accounting="98 registered/previous + 2 diagnostic = 100",
        likelihoods="zero storage placeholders; never PPO data",
        accepted_checkpoint=False, alters_training_or_selection=False,
        runner_sha256=sha(Path(__file__)), source_sha256=manifest["source_sha256"],
        comparison_sha256=evidence,
    )
    path = output / "manifest.json"
    if path.exists() and read(path) != registration:
        raise ValueError("diagnostic registration changed")
    atomic_json(path, registration)
    snapshot = output / "source_snapshot.py"
    if snapshot.exists() and sha(snapshot) != registration["runner_sha256"]:
        raise ValueError("diagnostic source snapshot changed")
    if not snapshot.exists():
        snapshot.write_bytes(Path(__file__).read_bytes())
    return checkpoint, inputs, registration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    campaign, output = args.campaign.resolve(), args.output.resolve()
    with exclusive_run(output):
        sumo = shutil.which("sumo")
        checkpoint, inputs, registration = prepare(campaign, output, sumo)
        if not args.run:
            print(json.dumps(registration), flush=True)
            return
        torch.set_num_threads(1)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        base = SharedDampingActor()
        base.load_state_dict(payload["actor"], strict=True)
        actor = MagnitudeAblationActor(base).eval()
        rows = []
        for seed, loading, scene, requests, native, policy in inputs:
            rollout = collect_cached(output / f"ablation_{seed}.json", scene, requests,
                                     seed=seed, sumo_binary=sumo, actor=actor)
            summary = rollout["summary"]
            rows.append(dict(case=[seed, loading], ablation=summary,
                             ablation_relative_to_native=case_metrics(summary, native),
                             ablation_relative_to_learned=case_metrics(summary, policy)))
        atomic_json(output / "summary.json", dict(registration=registration, cases=rows,
                    status="complete", accepted_checkpoint=False,
                    interpretation="two outcome-selected causal policy comparisons, not a generalization or attribution-percentage estimate"))
        print(json.dumps(dict(status="diagnosis_complete", cases=list(CASES))), flush=True)


if __name__ == "__main__":
    main()
