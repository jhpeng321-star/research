from types import SimpleNamespace

from tools.audit_idm_v5_5_t880_control_identity import (
    SUPERVISOR_PROGRESS_INDEX,
    _validate_source_artifacts,
    compare_control_traces,
    normalized_supervisor_digest,
    physical_state_digest,
)


def _snapshot(*, progress: float, speed: float = 10.0, other_obs: float = 0.0):
    observation = [0.0] * 16
    observation[3] = other_obs
    observation[SUPERVISOR_PROGRESS_INDEX] = progress
    return SimpleNamespace(
        relative_step=1,
        sim_time=5.0,
        action=0,
        supervisor_observation=tuple(observation),
        vehicles=(("veh", "edge", "lane", 10.0, speed, 0.0, 0.0, 0, "type", 1.0),),
        active_cavs=("veh",),
        collision_vehicle_ids=(),
        starting_teleport_vehicle_ids=(),
    )


def _trace(snapshot):
    audit = {
        "relative_step": 1,
        "command": 0,
        "command_name": "DAMPING_ONLY",
        "base_action_ids": (),
        "base_actions": {},
        "base_direct_control_ids": ("veh",),
        "base_control_ids": ("veh",),
        "base_diagnostics": {"commands": ()},
        "applied_env_actions": {},
        "wall_active": False,
        "wall_override_ids": (),
        "out_of_band_damping_ids": ("veh",),
        "transition_reason": "damping_only",
    }
    return SimpleNamespace(
        snapshots=(snapshot,),
        controller_audits=(audit,),
        trace_digest=f"raw-{snapshot.supervisor_observation[SUPERVISOR_PROGRESS_INDEX]}",
        reset_info={"episode_time": 4.0, "paired_seed": 4401},
    )


def test_normalized_identity_excludes_only_episode_progress() -> None:
    left = _trace(_snapshot(progress=0.000922))
    right = _trace(_snapshot(progress=0.000925))
    result = compare_control_traces(left, right)
    assert result["passed"] is True
    assert physical_state_digest(left.snapshots) == physical_state_digest(right.snapshots)
    assert normalized_supervisor_digest(left.snapshots) == normalized_supervisor_digest(
        right.snapshots
    )


def test_normalized_identity_rejects_vehicle_state_difference() -> None:
    left = _trace(_snapshot(progress=0.000922, speed=10.0))
    right = _trace(_snapshot(progress=0.000925, speed=9.9))
    result = compare_control_traces(left, right)
    assert result["passed"] is False
    assert result["checks"]["microscopic_physical_state_exact"] is False


def test_normalized_identity_rejects_another_supervisor_difference() -> None:
    left = _trace(_snapshot(progress=0.000922, other_obs=0.1))
    right = _trace(_snapshot(progress=0.000925, other_obs=0.2))
    result = compare_control_traces(left, right)
    assert result["passed"] is False
    assert result["checks"]["normalized_supervisor_observation_exact"] is False


def test_normalized_identity_rejects_dynamic_control_difference() -> None:
    left = _trace(_snapshot(progress=0.000922))
    right = _trace(_snapshot(progress=0.000925))
    right.controller_audits[0]["base_diagnostics"] = {"commands": ({"x": 1},)}
    result = compare_control_traces(left, right)
    assert result["passed"] is False
    assert result["checks"]["dynamic_damping_control_exact"] is False


def test_normalized_identity_rejects_reset_difference() -> None:
    left = _trace(_snapshot(progress=0.000922))
    right = _trace(_snapshot(progress=0.000925))
    right.reset_info["paired_seed"] = 9999
    result = compare_control_traces(left, right)
    assert result["passed"] is False
    assert result["checks"]["reset_info_exact"] is False


def test_source_validation_rejects_wrong_scenario() -> None:
    from experiments.idm.idm_v5_5_t880_contract import contract_manifest
    from policies.idm_analytical import frozen_idm_analytical_manifest

    base = frozen_idm_analytical_manifest()
    density = {
        "schema": "idm_v5_5_density_mapper_branch/1",
        "seed": 4401,
        "repeat": False,
        "demand_mainline_vph": 2700,
        "demand_ramp_vph": 400,
        "cav_penetration": 0.3,
        "base_control_arm": base,
    }
    pilot = {
        "schema": "idm_v5_5_t880_wall_pilot_case/1",
        "seed": 4401,
        "repeat": False,
        "demand_mainline_vph": 2700,
        "demand_ramp_vph": 400,
        "cav_penetration": 0.3,
        "contract_fingerprint_sha256": contract_manifest()[
            "contract_fingerprint_sha256"
        ],
        "base_control_arm": base,
        "arms": {"0": {}, "1": {}},
    }
    assert all(_validate_source_artifacts(density, pilot, seed=4401).values())
    pilot["demand_mainline_vph"] = 2400
    assert _validate_source_artifacts(density, pilot, seed=4401)[
        "pilot_scenario"
    ] is False
