from counterfactual.twin_replay import (
    FrozenPulseSchedule,
    ReplayTrace,
    StepSnapshot,
    compare_traces,
    run_replay,
)
from experiments.counterfactual.verify_counterfactual_escalation import _audit_contract


def _step(*, digest: str, speed: float = 10.0) -> StepSnapshot:
    return StepSnapshot(
        relative_step=1,
        sim_time=1.0,
        action=0,
        supervisor_observation=(0.5, 0.0),
        vehicles=(("veh.0", "edge", "edge_0", 10.0, speed, 0.0, 0.0, 0, "car"),),
        active_cavs=("veh.0",),
        collision_vehicle_ids=(),
        starting_teleport_vehicle_ids=(),
        digest=digest,
        vehicle_count=1,
        mean_speed=speed,
        active_time_loss_sum=0.0,
    )


def _trace(step: StepSnapshot) -> ReplayTrace:
    return ReplayTrace(
        seed=101,
        decision_step=0,
        horizon_steps=1,
        prefix_action=0,
        post_action=0,
        treatment_steps=None,
        reset_info={"active_cavs": ["veh.0"], "warmup_steps": 1},
        snapshots=(step,),
        terminated_early=False,
    )


def test_compare_traces_passes_for_identical_replays():
    comparison = compare_traces(_trace(_step(digest="same")), _trace(_step(digest="same")))

    assert comparison["passed"] is True
    assert comparison["first_mismatch"] is None


def test_compare_traces_reports_first_vehicle_divergence():
    comparison = compare_traces(
        _trace(_step(digest="left", speed=10.0)),
        _trace(_step(digest="right", speed=9.5)),
    )

    assert comparison["passed"] is False
    assert comparison["first_mismatch"]["relative_step"] == 1
    assert comparison["first_mismatch"]["changed_vehicle_ids_sample"] == ["veh.0"]
    assert comparison["first_mismatch"]["vehicle_numeric_max_abs_difference"] == 0.5


def test_frozen_pulse_schedule_is_explicit_and_post_decision():
    schedule = FrozenPulseSchedule(decision_step=2, treatment_steps=2)

    assert [schedule(step) for step in range(0, 7)] == [0, 0, 0, 1, 1, 0, 0]


class _FakeBasePolicy:
    def reset(self, env, seed=None):
        self.seed = seed

    def act(self, obs):
        return {vehicle_id: 2 for vehicle_id in obs}

    def close(self):
        pass


class _FakeOverlay:
    def __init__(self, *, base_policy, **_kwargs):
        self.base_policy = base_policy
        self.command = 0
        self.last_audit = {}

    def reset(self, env, seed=None):
        self.base_policy.reset(env, seed=seed)

    def set_command(self, command):
        self.command = int(command)

    def act(self, obs):
        actions = self.base_policy.act(obs)
        self.last_audit = {
            "command": self.command,
            "command_name": "DAMPING_PLUS_WALL" if self.command else "DAMPING_ONLY",
            "base_actions": dict(actions),
            "wall_active": bool(self.command),
            "wall_override_ids": ["cav.0"] if self.command else [],
        }
        return actions

    def close(self):
        self.base_policy.close()


class _FakeEnv:
    instances = []

    def __init__(self, config):
        self.config = dict(config)
        self.steps = []
        self.current_step = 0
        self.__class__.instances.append(self)

    def reset(self, seed=None):
        self.seed = seed
        return {"cav.0": (0.0,)}, {"paired_seed": seed, "active_cavs": ["cav.0"]}

    def step(self, actions):
        self.current_step += 1
        self.steps.append(dict(actions))
        return (
            {"cav.0": (float(self.current_step),)},
            {"cav.0": 0.0},
            {"__all__": False},
            {"__all__": False},
            {},
        )

    def close(self):
        self.closed = True


def _fake_capture(
    _traci,
    *,
    relative_step,
    action,
    supervisor_observation,
    active_cav_ids=(),
    digits=6,
):
    del digits
    speed = float(relative_step) + 10.0
    return StepSnapshot(
        relative_step=relative_step,
        sim_time=float(relative_step),
        action=int(action),
        supervisor_observation=tuple(supervisor_observation),
        vehicles=(("cav.0", "edge", "edge_0", float(relative_step), speed, 0.0, 0.0, 0, "car"),),
        active_cavs=tuple(active_cav_ids),
        collision_vehicle_ids=(),
        starting_teleport_vehicle_ids=(),
        digest=f"step-{relative_step}-command-{int(action)}",
        vehicle_count=1,
        mean_speed=speed,
        active_time_loss_sum=0.0,
    )


def test_policy_replay_uses_base_factory_and_explicit_schedule(monkeypatch, tmp_path):
    import env.wall_supervisor
    import policies.counterfactual_escalation
    import counterfactual.twin_replay as replay_module

    _FakeEnv.instances.clear()
    monkeypatch.setattr(
        policies.counterfactual_escalation,
        "CounterfactualEscalationPolicy",
        _FakeOverlay,
    )
    monkeypatch.setattr(
        env.wall_supervisor,
        "corridor_state",
        lambda *_args, **_kwargs: (0.5, 0.0),
    )
    monkeypatch.setattr(replay_module, "capture_step", _fake_capture)

    trace = run_replay(
        project_root=tmp_path,
        sumo_binary="sumo",
        sumo_config="scenario.sumocfg",
        seed=101,
        decision_step=2,
        horizon_steps=4,
        treatment_steps=2,
        base_policy_factory=_FakeBasePolicy,
        intervention_schedule=FrozenPulseSchedule(2, 2),
        environment_factory=_FakeEnv,
    )

    assert [snapshot.action for snapshot in trace.snapshots] == [0, 0, 0, 1, 1, 0, 0]
    assert [audit["command"] for audit in trace.controller_audits] == [0, 0, 0, 1, 1, 0, 0]
    assert trace.controller_metadata["runner"] == "frozen_damping_with_explicit_overlay"
    assert len(_FakeEnv.instances) == 1
    assert _FakeEnv.instances[0].config["shepherd_modes"] == ["coast", "recover", "guard"]
    assert _FakeEnv.instances[0].steps == [{"cav.0": 2}] * 6


def test_policy_replay_requires_factory_and_schedule_together(tmp_path):
    try:
        run_replay(
            project_root=tmp_path,
            sumo_binary="sumo",
            sumo_config="scenario.sumocfg",
            seed=101,
            decision_step=2,
            horizon_steps=4,
            base_policy_factory=_FakeBasePolicy,
        )
    except ValueError as exc:
        assert "must be provided together" in str(exc)
    else:
        raise AssertionError("policy replay must require an intervention schedule")


def test_overlay_contract_reports_realized_dose_coverage_and_handoffs():
    base = {"cav.0": 2, "cav.1": 1}
    control_audits = tuple(
        {
            "relative_step": step,
            "command": 0,
            "base_actions": base,
            "applied_env_actions": base,
            "wall_active": False,
            "wall_override_ids": (),
            "out_of_band_damping_ids": tuple(base),
        }
        for step in range(1, 5)
    )
    treatment_audits = (
        {
            "relative_step": 1,
            "command": 1,
            "base_actions": base,
            "applied_env_actions": {"cav.1": 1},
            "wall_active": True,
            "wall_override_ids": ("cav.0",),
            "out_of_band_damping_ids": ("cav.1",),
            "transition_reason": "wall_started",
            "lane_coverage": {
                "required_lane_count": 1,
                "covered_lane_count": 1,
            },
            "member_changes": ({"event": "latched", "member_id": "cav.0"},),
            "control_deltas": {
                "cav.0": {
                    "base_action_suppressed": 2,
                    "pre_override_speed_mps": 20.0,
                    "member_projected_speed_target_mps": 18.75,
                    "member_applied_speed_command_mps": 19.5,
                    "wall_a_max_dec_mps2": 1.0,
                    "treatment_phase": "ASSEMBLING",
                }
            },
        },
        {
            "relative_step": 2,
            "command": 1,
            "base_actions": base,
            "applied_env_actions": {"cav.0": 2},
            "wall_active": True,
            "wall_override_ids": ("cav.1",),
            "out_of_band_damping_ids": ("cav.0",),
            "transition_reason": "wall_holding",
            "lane_coverage": {
                "required_lane_count": 1,
                "covered_lane_count": 1,
            },
            "member_changes": (
                {
                    "event": "handoff",
                    "previous_member_id": "cav.0",
                    "member_id": "cav.1",
                },
            ),
            "control_deltas": {
                "cav.1": {
                    "base_action_suppressed": 1,
                    "pre_override_speed_mps": 19.0,
                    "member_projected_speed_target_mps": 18.75,
                    "member_applied_speed_command_mps": 18.75,
                    "wall_a_max_dec_mps2": 1.0,
                    "treatment_phase": "HOLDING",
                }
            },
        },
        {
            "relative_step": 3,
            "command": 1,
            "base_actions": base,
            "applied_env_actions": {"cav.0": 2},
            "wall_active": True,
            "wall_override_ids": ("cav.1",),
            "out_of_band_damping_ids": ("cav.0",),
            "transition_reason": "wall_holding",
            "lane_coverage": {
                "required_lane_count": 1,
                "covered_lane_count": 1,
            },
            "member_changes": (),
            "control_deltas": {
                "cav.1": {
                    "base_action_suppressed": 1,
                    "pre_override_speed_mps": 18.75,
                    "member_projected_speed_target_mps": 18.75,
                    "member_applied_speed_command_mps": 18.75,
                    "wall_a_max_dec_mps2": 1.0,
                    "treatment_phase": "HOLDING",
                }
            },
        },
        {
            "relative_step": 4,
            "command": 0,
            "base_actions": base,
            "applied_env_actions": base,
            "wall_active": False,
            "wall_override_ids": (),
            "out_of_band_damping_ids": tuple(base),
            "transition_reason": "pulse_released_to_damping",
        },
    )
    control = ReplayTrace(
        **{
            **_trace(_step(digest="control")).__dict__,
            "controller_audits": control_audits,
        }
    )
    treatment = ReplayTrace(
        **{
            **_trace(_step(digest="treatment")).__dict__,
            "controller_audits": treatment_audits,
        }
    )

    result = _audit_contract(
        control,
        treatment,
        decision_step=0,
        treatment_steps=3,
        step_length_s=0.5,
    )

    assert result["passed"] is True
    assert result["treatment_dose"]["activation_feasible"] is True
    assert result["treatment_dose"]["requested_duration_s"] == 1.5
    assert result["treatment_dose"]["realized_override_duration_steps"] == 3
    assert result["treatment_dose"]["controlled_cav_seconds"] == 1.5
    assert result["treatment_dose"]["lane_coverage_fraction"] == 1.0
    assert result["treatment_dose"]["dose_at_least_80_percent"] is True
    assert (
        result["treatment_dose"]["dose_at_least_80_percent_scope"]
        == "duration_and_lane_coverage_only"
    )
    assert result["treatment_dose"]["duration_and_lane_dose_qualified"] is True
    assert result["treatment_dose"]["compact_composite_dose_qualified"] is True
    assert result["treatment_dose"]["planned_release_step"] == 4
    assert result["treatment_dose"]["completion_reason"] == "planned_duration_complete"
    assert result["treatment_dose"]["member_change_counts"] == {
        "latched": 1,
        "handoff": 1,
    }
    assert result["treatment_dose"]["configured_wall_deceleration_bounds_mps2"] == [1.0]
    assert result["treatment_dose"]["member_applied_command_sample_count"] == 3
    assert result["treatment_dose"]["max_commanded_deceleration_mps2"] == 1.0
    holding_delivery = result["treatment_dose"]["holding_speed_delivery"]
    assert holding_delivery["sample_count"] == 2
    assert holding_delivery["mean_actual_speed_mps"] == 18.875
    assert holding_delivery["mean_projected_target_mps"] == 18.75
    assert holding_delivery["mean_absolute_target_error_mps"] == 0.125
    assert holding_delivery["fraction_within_1_mps"] == 1.0
    terminal_delivery = result["treatment_dose"]["terminal_tail_speed_delivery"]
    assert terminal_delivery["holding_step_count"] == 2
    assert terminal_delivery["fraction_within_1_mps"] == 1.0
