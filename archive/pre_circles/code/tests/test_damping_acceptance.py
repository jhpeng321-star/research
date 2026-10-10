"""Synthetic metric fixtures only; no reserved SUMO requests or outcomes."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from idm.damping_acceptance import AcceptanceContract, assess_acceptance, stratified_interval
from idm.damping_campaign import ACCEPTANCE_CASES


def synthetic_rows():
    rows = []
    for seed, loading in ACCEPTANCE_CASES:
        native = dict(
            scenario_identity=f"synthetic-{seed}",
            seed=seed,
            population=100,
            global_control_steps=2400,
            team_terminal=True,
            arrived=100,
            uninserted_due=0,
            unfinished_requested=0,
            system_seconds_per_requested_vehicle=50.0,
            source_wait_vehicle_seconds=0.0,
            ttc_below_3_vehicle_seconds=100.0,
            totals=dict(
                braking_vehicle_seconds=100.0,
                acceleration_energy=100.0,
                added_collision_events=0,
                added_teleport_events=0,
            ),
        )
        policy, analytical = deepcopy(native), deepcopy(native)
        for key in ("braking_vehicle_seconds", "acceleration_energy"):
            policy["totals"][key] = 90.0
            analytical["totals"][key] = 95.0
        rows.append(dict(case=[seed, loading], policy=policy, analytical=analytical, native=native))
    return rows


def test_constant_paired_improvement_interval_and_stratum_weights():
    cfg = replace(AcceptanceContract(), bootstrap_draws=1000)
    report = stratified_interval([0.1] * 4 + [0.2] * 8, [2000] * 4 + [2200] * 8, cfg)
    assert report["mean"] == pytest.approx(0.15)
    assert report["lower"] == pytest.approx(0.15)
    assert report["upper"] == pytest.approx(0.15)
    assert report == stratified_interval([0.1] * 4 + [0.2] * 8, [2000] * 4 + [2200] * 8, cfg)


def test_all_cases_and_harmful_tail_required_even_with_positive_mean():
    rows = synthetic_rows()
    result = assess_acceptance(rows)
    assert result["passes_statistical_and_physical_rules"]
    assert not result["accepted_checkpoint"]
    rows[-1]["policy"]["totals"]["added_collision_events"] = 1
    result = assess_acceptance(rows)
    assert not result["passes_statistical_and_physical_rules"]
    assert any(row["constraint"] == "hard_events" for row in result["failures"])
    with pytest.raises(ValueError, match="32 reserved"):
        assess_acceptance(rows[:-1])


def test_native_improvement_does_not_hide_inferiority_to_analytical():
    rows = synthetic_rows()
    for row in rows:
        for key in ("braking_vehicle_seconds", "acceleration_energy"):
            row["analytical"]["totals"][key] = 70.0
    result = assess_acceptance(rows)
    assert any(
        row["constraint"] == "analytical_comparator_noninferiority" for row in result["failures"]
    )


def test_uncertain_positive_mean_and_wrong_seed_labels_are_rejected():
    rows = synthetic_rows()
    for index, row in enumerate(rows[:16]):
        # Positive aggregate improvement with a deliberately wide seed spread.
        for key in ("braking_vehicle_seconds", "acceleration_energy"):
            row["policy"]["totals"][key] = 109.0 if index % 8 else 1.0
    result = assess_acceptance(rows)
    assert any(row["constraint"] == "lower_improvement_uncertainty" for row in result["failures"])
    rows[0]["policy"]["seed"] += 1
    with pytest.raises(ValueError, match="label"):
        assess_acceptance(rows)
    with pytest.raises(ValueError, match="invalid paired"):
        stratified_interval([np.nan, 1.0], [2000, 2000])


def test_incomplete_campaign_cannot_export_candidate(tmp_path):
    import json
    from tools.freeze_damping_candidate import read_complete_development

    for seed in (71, 72, 73):
        directory = tmp_path / f"lineage_{seed}"
        directory.mkdir()
        (directory / "stage_summary.json").write_text(
            json.dumps(dict(status="stage_complete", optimized_episodes=20))
        )
    with pytest.raises(RuntimeError, match="three completed"):
        read_complete_development(tmp_path)


def test_frozen_actor_must_reproduce_recorded_actions_per_tick():
    import torch
    from idm.damping_actor import SharedDampingActor
    from tools.freeze_damping_candidate import verify_deterministic_actions

    torch.set_num_threads(1)
    actor = SharedDampingActor()
    obs = torch.randn(5, actor.contract.observation_dim)
    with torch.no_grad():
        actions = torch.cat([actor(obs[:2]), actor(obs[2:])]).numpy()
    row = dict(
        actor_times=np.array([1, 1, 2, 2, 2]), actor_observations=obs.numpy(), actions=actions
    )
    verify_deterministic_actions(actor, row)
    row["actions"][1, 0] += 0.01
    with pytest.raises(ValueError, match="recorded deterministic"):
        verify_deterministic_actions(actor, row)


def test_complete_development_export_audit_checks_raw_trajectory_identity(tmp_path, monkeypatch):
    import hashlib
    import json
    from tools import freeze_damping_candidate as bundle
    from idm.damping_campaign import CHECKPOINT_EPISODES, DEVELOPMENT_CASES, OPTIMIZER_SEEDS
    from idm.damping_selection import SelectionContract, assess_candidate

    # Synthetic filesystem fixture: no scenario generation or SUMO execution.
    monkeypatch.setattr(bundle, "request_schedule", lambda *args, **kwargs: [])

    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    manifest = dict(selection=SelectionContract().manifest(), source_sha256={})
    write(tmp_path / "manifest.json", manifest)
    write(tmp_path / "active_stage.json", dict(status="complete"))
    rows = []
    for (seed, loading), synthetic in zip(DEVELOPMENT_CASES, synthetic_rows()):
        scene = tmp_path / "scenarios" / f"development_{seed}"
        scene.mkdir(parents=True)
        for name in ("scenario.sumocfg", "requests.rou.xml"):
            (scene / name).write_bytes(b"synthetic")
        row = deepcopy(synthetic)
        row["case"] = [seed, loading]
        for arm in ("policy", "native", "analytical"):
            row[arm]["seed"] = seed
            row[arm]["scenario_identity"] = hashlib.sha256(b"[]syntheticsynthetic").hexdigest()
        rows.append(row)

    def trajectory(summary):
        return dict(
            summary=summary,
            intervals=[{key: value / 2400 for key, value in summary["totals"].items()}] * 2400,
        )

    for row in rows:
        seed = row["case"][0]
        write(tmp_path / "references" / f"development_{seed}.json", trajectory(row["native"]))
        write(tmp_path / "analytical" / f"development_{seed}.json", trajectory(row["analytical"]))
    for optimizer in OPTIMIZER_SEEDS:
        lineage = tmp_path / f"lineage_{optimizer}"
        write(lineage / "stage_summary.json", dict(status="stage_complete", optimized_episodes=100))
        for episode in CHECKPOINT_EPISODES:
            checkpoint = lineage / f"trainer_{episode:03}.pt"
            checkpoint.write_bytes(b"synthetic checkpoint identity only")
            directory = tmp_path / "development" / f"optimizer_{optimizer}_episode_{episode:03}"
            write(
                directory / "summary.json",
                dict(
                    campaign=bundle.digest(manifest),
                    optimizer_seed=optimizer,
                    optimized_global_steps=episode * 2400,
                    checkpoint_sha256=bundle.sha(checkpoint),
                    cases=rows,
                    assessment=assess_candidate(rows),
                ),
            )
            for row in rows:
                write(directory / f"policy_{row['case'][0]}.json", trajectory(row["policy"]))
    _, records, evidence = bundle.read_complete_development(tmp_path)
    assert len(records) == 9
    assert len(evidence) == 97  # 72 policies, 16 shared arms, nine summaries.
    path = tmp_path / "references" / f"development_{DEVELOPMENT_CASES[0][0]}.json"
    corrupted = json.loads(path.read_text())
    corrupted["summary"]["seed"] += 1
    write(path, corrupted)
    with pytest.raises(ValueError, match="trajectory/scenario"):
        bundle.read_complete_development(tmp_path)
