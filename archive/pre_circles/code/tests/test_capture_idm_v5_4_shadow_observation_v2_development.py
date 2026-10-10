from __future__ import annotations

import json
from pathlib import Path

import pytest

from counterfactual.idm_shadow_observation_v2_contract import (
    SEED_SCHEMA,
    SOURCE_SHA256,
)
from counterfactual.twin_replay import ReplayTrace, StepSnapshot
from tools import capture_idm_v5_4_shadow_observation_v2_development as capture


def _manifest() -> dict:
    inventory = {
        "paired_label_seed_files": {
            f"labels/seeds/seed{seed}.json": f"label-{seed}"
            for seed in capture.DEVELOPMENT_SEEDS
        },
        "gate_f_feasibility_seed_files": {
            f"feasibility/seeds/seed{seed}.json": f"feasibility-{seed}"
            for seed in capture.DEVELOPMENT_SEEDS
        },
    }
    return capture.build_capture_manifest(SOURCE_SHA256, inventory)


def _valid_seed(seed: int = 4401) -> dict:
    manifest = _manifest()
    cases = []
    for index, decision in enumerate(capture.FROZEN_DECISIONS):
        admitted = index < 2
        cases.append(
            {
                "schema": capture.CASE_SCHEMA,
                "case_id": capture._case_id(seed, decision),
                "seed": seed,
                "decision_step": decision,
                "gate_f_admitted": admitted,
                "capture_phase": capture.CAPTURE_PHASE,
                "predecision_record": {
                    "schema": "idm_v5_4_shadow_observation_v2_predecision_live/1",
                    "capture_phase": capture.CAPTURE_PHASE,
                    "decision_step": decision,
                },
                "baseline_observation": {"model_features": {"traffic_speed": 1.0}},
                "shadow_observation": (
                    {"model_features": {"shadow_speed_delta": 0.1}}
                    if admitted
                    else None
                ),
                "integrity": {
                    "prefix_identity_passed": True,
                    "shadow_generated": admitted,
                },
            }
        )
    return {
        "schema": SEED_SCHEMA,
        "scientific_role": capture.SCIENTIFIC_ROLE,
        "manifest_sha256": manifest["manifest_sha256"],
        "contract_sha256": manifest["contract_sha256"],
        "source_sha256": manifest["source_sha256"],
        "seed": seed,
        "decision_steps": list(capture.FROZEN_DECISIONS),
        "cases": cases,
        "admitted_shadow_count": 2,
        "control_trace": {
            "decision_step": 700,
            "horizon_steps": 10,
            "trace_digest": "digest",
            "terminated_early": False,
        },
        "labels_joined_during_capture": False,
        "paired_outcome_rows_opened": False,
        "new_realized_outcomes": False,
        "wall_clock_seconds": 1.0,
    }


def _snapshot(step: int) -> StepSnapshot:
    return StepSnapshot(
        relative_step=step,
        sim_time=float(step),
        action=0,
        supervisor_observation=(0.0,),
        vehicles=(),
        active_cavs=(),
        collision_vehicle_ids=(),
        starting_teleport_vehicle_ids=(),
        digest=f"d{step}",
        vehicle_count=0,
        mean_speed=0.0,
        active_time_loss_sum=0.0,
    )


def _trace(*, decision_step: int = 700) -> ReplayTrace:
    terminal = decision_step + 10
    return ReplayTrace(
        seed=4401,
        decision_step=decision_step,
        horizon_steps=10,
        prefix_action=0,
        post_action=0,
        treatment_steps=None,
        reset_info={},
        snapshots=tuple(_snapshot(step) for step in range(terminal + 1)),
        terminated_early=False,
        controller_audits=tuple(
            {"relative_step": step, "command": 0} for step in range(terminal + 1)
        ),
    )


def test_cli_rejects_more_than_four_workers_and_exposes_no_seed_partition() -> None:
    with pytest.raises(SystemExit):
        capture._parse_args(["--workers", "5"])
    args = capture._parse_args(["--workers", "4"])
    assert args.workers == 4
    assert not hasattr(args, "seeds")
    assert capture.DEVELOPMENT_SEEDS == tuple(range(4401, 4521))
    with pytest.raises(capture.CaptureIntegrityError):
        capture.load_feasibility_rows(4601)
    with pytest.raises(capture.CaptureIntegrityError):
        capture.load_feasibility_rows(5001)


def test_manifest_is_deterministic_and_binds_frozen_sources_and_arms() -> None:
    first = _manifest()
    second = _manifest()
    assert first == second
    assert first["source_sha256"] == dict(sorted(SOURCE_SHA256.items()))
    assert len(first["source_seed_sha256"]["paired_label_seed_files"]) == 120
    assert len(first["source_seed_sha256"]["gate_f_feasibility_seed_files"]) == 120
    assert first["capture"]["all_decisions"] == 480
    assert first["capture"]["treatment_twins_gate_f_only"] == 199
    assert first["capture"]["labels_joined_during_capture"] is False
    assert first["base_control_arm"]
    assert first["wall_profile"]


def test_exact_development_seed_inventories_bind_all_bytes_without_discovery() -> None:
    inventory = capture.hash_frozen_seed_inventories()
    assert set(inventory) == {
        "paired_label_seed_files",
        "gate_f_feasibility_seed_files",
    }
    assert len(inventory["paired_label_seed_files"]) == 120
    assert len(inventory["gate_f_feasibility_seed_files"]) == 120
    assert all(any(path.endswith("/seed4401.json") for path in rows) for rows in (
        inventory["paired_label_seed_files"],
        inventory["gate_f_feasibility_seed_files"],
    ))
    assert all(any(path.endswith("/seed4520.json") for path in rows) for rows in (
        inventory["paired_label_seed_files"],
        inventory["gate_f_feasibility_seed_files"],
    ))


def test_immutable_manifest_is_reused_and_changed_manifest_fails(tmp_path: Path) -> None:
    manifest = _manifest()
    capture.ensure_immutable_manifest(tmp_path, manifest)
    capture.ensure_immutable_manifest(tmp_path, manifest)
    changed = json.loads(json.dumps(manifest))
    changed["scenario"]["cav_penetration"] = 0.2
    with pytest.raises(capture.CaptureIntegrityError, match="differs"):
        capture.ensure_immutable_manifest(tmp_path, changed)


def test_validated_seed_resume_skips_only_exact_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest()
    seed_root = tmp_path / "seeds"
    seed_root.mkdir()
    payload = _valid_seed()
    capture.atomic_write_json(seed_root / "seed4401.json", payload)
    assert capture.load_valid_seed(seed_root / "seed4401.json", 4401, manifest) == payload

    def forbidden_replay(**_kwargs):
        raise AssertionError("resume must not launch SUMO")

    monkeypatch.setattr(capture, "run_control_capture", forbidden_replay)
    result = capture.run_seed(
        {
            "seed": 4401,
            "output": str(tmp_path),
            "sumo_binary": "sumo",
            "manifest": manifest,
        }
    )
    assert result["status"] == "skipped"


def test_malformed_or_mismatched_resume_artifact_fails_closed(tmp_path: Path) -> None:
    manifest = _manifest()
    path = tmp_path / "seed4401.json"
    path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(capture.CaptureIntegrityError, match="unreadable"):
        capture.load_valid_seed(path, 4401, manifest)

    payload = _valid_seed()
    payload["contract_sha256"] = "wrong"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(capture.CaptureIntegrityError, match="contract hash"):
        capture.load_valid_seed(path, 4401, manifest)

    payload = _valid_seed()
    payload["decision_steps"][-1] = 701
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(capture.CaptureIntegrityError, match="decision grid"):
        capture.load_valid_seed(path, 4401, manifest)


def test_progress_is_atomic_and_records_resume_counts(tmp_path: Path) -> None:
    manifest = _manifest()
    capture.write_progress(
        tmp_path,
        manifest,
        valid_complete=20,
        newly_completed=10,
        pending=100,
    )
    payload = json.loads((tmp_path / "progress.json").read_text(encoding="utf-8"))
    assert payload["valid_complete_seed_count"] == 20
    assert payload["newly_completed_seed_count"] == 10
    assert payload["pending_seed_count"] == 100
    assert list(tmp_path.glob("*.tmp")) == []


def test_shared_control_trace_view_has_exact_decision_and_ten_second_horizon() -> None:
    trace = _trace()
    view = capture.control_trace_view(trace, 620)
    assert view.decision_step == 620
    assert view.horizon_steps == 10
    assert view.snapshots[-1].relative_step == 630
    assert view.controller_audits[-1]["relative_step"] == 630


def test_treatment_twin_freezes_common_episode_horizon(monkeypatch) -> None:
    observed = {}

    def fake_run_replay(**kwargs):
        observed.update(kwargs)
        return _trace(decision_step=620)

    monkeypatch.setattr(capture, "run_replay", fake_run_replay)
    capture.run_treatment_twin(seed=4401, decision_step=620, sumo_binary="sumo")
    assert callable(observed["environment_factory"])

    import env.core

    monkeypatch.setattr(
        env.core,
        "TrafficShepherdMultiAgentEnv",
        lambda config: dict(config),
    )
    environment = observed["environment_factory"]({"max_episode_steps": 635})
    assert environment["max_episode_steps"] == 715


def test_seed_capture_passes_enriched_record_and_replays_only_gate_f_admissions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "seeds").mkdir()
    manifest = _manifest()
    admitted_steps = {620, 680}
    feasibility = {
        step: {"decision_step": step, "admitted": step in admitted_steps}
        for step in capture.FROZEN_DECISIONS
    }
    live = {
        step: {
            "schema": "idm_v5_4_shadow_observation_v2_predecision_live/1",
            "capture_phase": capture.CAPTURE_PHASE,
            "decision_step": step,
            "vehicles": [],
        }
        for step in capture.FROZEN_DECISIONS
    }
    control = _trace()
    helper_live_steps: list[int] = []
    treatment_steps: list[int] = []

    def fake_baseline(
        _control,
        *,
        decision_step,
        feasibility_row,
        live_predecision_record,
    ):
        assert feasibility_row["decision_step"] == decision_step
        assert live_predecision_record is not live[decision_step]
        assert live_predecision_record["decision_step"] == decision_step
        helper_live_steps.append(decision_step)
        return {"model_features": {"baseline_value": 1.0}}

    def fake_shadow(_control, treatment, *, decision_step, feasibility_row):
        assert treatment.decision_step == decision_step
        assert feasibility_row["admitted"] is True
        return {"model_features": {"shadow_value": 1.0}}

    def fake_treatment(*, seed, decision_step, sumo_binary):
        assert seed == 4401 and sumo_binary == "sumo"
        treatment_steps.append(decision_step)
        return _trace(decision_step=decision_step)

    monkeypatch.setattr(capture, "load_feasibility_rows", lambda _seed: feasibility)
    monkeypatch.setattr(
        capture,
        "run_control_capture",
        lambda **_kwargs: (control, live),
    )
    monkeypatch.setattr(capture, "run_treatment_twin", fake_treatment)
    monkeypatch.setattr(
        capture,
        "_load_shadow_helpers",
        lambda: (fake_baseline, fake_shadow, lambda _features: None),
    )
    result = capture.run_seed(
        {
            "seed": 4401,
            "output": str(tmp_path),
            "sumo_binary": "sumo",
            "manifest": manifest,
        }
    )
    assert result["status"] == "completed"
    assert helper_live_steps == list(capture.FROZEN_DECISIONS)
    assert treatment_steps == sorted(admitted_steps)
    payload = capture.load_valid_seed(
        tmp_path / "seeds" / "seed4401.json", 4401, manifest
    )
    assert payload is not None
    assert payload["admitted_shadow_count"] == 2


class _FakeVehicle:
    def getIDList(self):
        return ("cav.2", "cav.1", "hdv.1")

    def getRoadID(self, vehicle_id):
        return "main"

    def getLaneID(self, vehicle_id):
        return "main_0" if vehicle_id != "cav.2" else "main_1"

    def getLaneIndex(self, vehicle_id):
        return 1 if vehicle_id == "cav.2" else 0

    def getLanePosition(self, vehicle_id):
        return {"cav.1": 100.0, "hdv.1": 80.0, "cav.2": 102.0}[vehicle_id]

    def getSpeed(self, vehicle_id):
        return {"cav.1": 20.0, "hdv.1": 22.0, "cav.2": 19.0}[vehicle_id]

    def getAcceleration(self, _vehicle_id):
        return -0.1

    def getLength(self, _vehicle_id):
        return 5.0

    def getTypeID(self, vehicle_id):
        return "cav" if vehicle_id.startswith("cav") else "hdv"

    def getRouteIndex(self, _vehicle_id):
        return 1

    def getLeader(self, vehicle_id, _distance):
        return ("cav.1", 15.0) if vehicle_id == "hdv.1" else None

    def getFollower(self, vehicle_id, _distance):
        return ("hdv.1", 15.0) if vehicle_id == "cav.1" else ("", -1.0)

    def getTau(self, _vehicle_id): return 1.0
    def getMinGap(self, _vehicle_id): return 2.5
    def getAccel(self, _vehicle_id): return 2.0
    def getDecel(self, _vehicle_id): return 4.5
    def getEmergencyDecel(self, _vehicle_id): return 9.0
    def getApparentDecel(self, _vehicle_id): return 4.5
    def getSpeedFactor(self, _vehicle_id): return 1.0
    def getAllowedSpeed(self, _vehicle_id): return 30.0
    def getMaxSpeed(self, _vehicle_id): return 33.0
    def getActionStepLength(self, _vehicle_id): return 1.0
    def getSpeedMode(self, _vehicle_id): return 31
    def getLaneChangeMode(self, _vehicle_id): return 1621


class _FakeTraci:
    vehicle = _FakeVehicle()


class _FakePolicy:
    wall_params = {"d_lo": 200.0, "d_hi": 1000.0}
    compact_release_boundary_m = 50.0
    last_audit = {
        "command": 0,
        "wall_active": False,
        "base_control_ids": ("cav.1", "cav.2"),
        "base_actions": {"cav.1": [0.0], "cav.2": [0.1]},
        "base_diagnostics": {"commands": [{"vehicle_id": "cav.1"}]},
    }

    def _live_ids(self):
        return {"cav.1", "cav.2", "hdv.1"}

    def _lane_id(self, vehicle_id):
        return _FakeTraci.vehicle.getLaneID(vehicle_id)

    def _distance_in_band(self, vehicle_id):
        return {"cav.1": 500.0, "cav.2": 510.0}.get(vehicle_id)


def test_live_predecision_capture_contains_required_physics_and_no_model_tensor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        capture,
        "distance_to_merge",
        lambda vehicle_id: {"cav.1": 500.0, "cav.2": 510.0, "hdv.1": 520.0}[vehicle_id],
    )
    feasibility = {
        "formation": {
            "controlled_cav_ids": ["cav.1"],
            "required_lane_ids": ["main_0"],
        },
        "lane_coverage": {"coverage_fraction": 1.0},
        "projected_dose": {"projected_duration_fraction": 1.0},
        "activation_safety": {"safe": True},
    }
    record = capture.capture_live_predecision_record(
        _FakeTraci,
        _FakePolicy(),
        decision_step=620,
        obs={"cav.1": (), "cav.2": ()},
        action_dict={"cav.1": [0.0], "cav.2": [0.1]},
        feasibility_row=feasibility,
        inflow_tracker={"mainline": [1, 2]},
    )
    assert record["capture_phase"] == capture.CAPTURE_PHASE
    assert len(record["vehicles"]) == 3
    assert record["vehicles"][0]["parameters"]["tau_s"]["available"] is True
    cav1 = next(row for row in record["vehicles"] if row["internal_vehicle_id"] == "cav.1")
    assert cav1["follower"]["net_gap_m"] == 15.0
    assert record["damping"]["projection_diagnostics"]
    reserves = record["proposed_wall"]["ranked_same_lane_reserves"]["main_0"]
    assert reserves == []  # cav.2 is in the other lane
    assert record["internal_identifiers_are_linkage_metadata_only"] is True
    assert "model_features" not in record
