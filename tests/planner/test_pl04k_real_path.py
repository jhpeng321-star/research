"""PL-04k: the REAL-path gaps that blocked the PL-EXP1 oracle experiment.

Gap 1 -- ``RealMasterHarness.step_physics`` rows carry the campaign
``physics_steps.jsonl`` fields (road_id, lane_id, lane_position_m, ...,
colliding_ids, arrived_ids, time_s) next to the runner's own keys.
Gap 2 -- the real master and every lookahead branch bind a REAL registered
PI session through ONE shared construction (``build_mixed_session``), so
pi-external-u plans execute; without a PI session the named
``PiSessionMissingError`` refuses. Fakes only: no SUMO.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.planner import test_pl04i_e2e_real_run as e2e_mod
from traffic_shepherd.planner import closed_loop as cl
from traffic_shepherd.planner.closed_loop import (
    ClosedLoopConfig,
    ClosedLoopRunner,
    build_real_components,
    initial_plan_for_roster,
    resolve_base_run,
)

FIXTURE = Path(__file__).parent / "fixtures" / "pl04k_campaign_physics_row_keys.json"
RUNNER_EXTRA_ROW_KEYS = {"kind", "t_s", "dt_s", "cav_speed_mode"}
RUNNER_EXTRA_VEHICLE_KEYS = {"s_m", "v_mps", "lane_index", "accel_mps2"}
PRE_MERGE_LANE = e2e_mod.PRE_MERGE_LANE


def _run(tmp_path, monkeypatch, *, initial_plan=None, run_id="pl04k", near_merge=True):
    """One REAL-runner run over the tiny world on the pre-merge lane.
    ``near_merge`` puts the CAVs within the RSU's merge coverage (the
    planner decides); otherwise they sit mid-lane with a leader in sensor
    range for the whole run (the PI adapter needs a covered leader)."""
    base = (e2e_mod.make_base_run_pl04j if near_merge else e2e_mod.make_base_run_pl04i)(
        tmp_path
    )
    resolved = resolve_base_run(base)
    monkeypatch.setenv("SUMO_BINARY", sys.executable)
    monkeypatch.delenv("SUMO_HOME", raising=False)
    traci = e2e_mod._TinyTraci(lane=PRE_MERGE_LANE)
    monkeypatch.setitem(sys.modules, "traci", traci)
    components = build_real_components(e2e_mod._real_args(base))
    if initial_plan:
        components.initial_plan = initial_plan_for_roster(components.roster, initial_plan)
    config = ClosedLoopConfig(
        out_root=tmp_path,
        run_id=run_id,
        world="w_pl04k",
        seed=e2e_mod.SEED,
        horizon_s=10.0,
        warmup_s=0.0,
        decision_period_s=10.0,
        predictor="perfect_sim",
        lookahead_horizon_s=5.0,
        lookahead_commit_s=None,
        k_samples=1,
        max_sim_s=500.0,
        base_run=resolved.base_run,
        base_t0_s=resolved.t0_s,
        base_state_sha256=resolved.state_sha256,
        max_decision_units=2,
        walls="off",
        initial_plan_action=initial_plan,
    )
    receipt = ClosedLoopRunner(config, components).run()
    return SimpleNamespace(
        receipt=receipt, components=components, run_dir=tmp_path / run_id, traci=traci
    )


def _rows(path):
    return e2e_mod.read_jsonl(path)


def _adapter():
    from traffic_shepherd.config import load_config
    from traffic_shepherd.planner.paths import research_dense_root

    return load_config(research_dense_root() / "baselines" / "circles" / "adapter.json")


class TestPhysicsRowSchema:
    """(a) the step_physics row schema matches the campaign keys."""

    def test_rows_match_campaign_key_structure(self, tmp_path, monkeypatch):
        spec = json.loads(FIXTURE.read_text(encoding="utf-8"))["row"]
        out = _run(tmp_path, monkeypatch)
        rows = _rows(out.run_dir / "physics_steps.jsonl")
        assert rows
        ((_, state),) = list(spec["vehicles"].items())
        ((_, action),) = list(spec["cav_actions"].items())
        for row in rows:
            assert set(row) - RUNNER_EXTRA_ROW_KEYS == set(spec)
            assert RUNNER_EXTRA_ROW_KEYS <= set(row)  # the runner keys are kept
            for key, placeholder in spec.items():  # value TYPES match too
                assert isinstance(row[key], type(placeholder)), key
            assert row["time_s"] == row["t_s"]
            assert set(row["boundary_crossings"]) == {
                "mainline_entry", "mainline_exit", "ramp_entry"
            }
            for vehicle in row["vehicles"].values():
                assert set(vehicle) - RUNNER_EXTRA_VEHICLE_KEYS == set(state)
                assert RUNNER_EXTRA_VEHICLE_KEYS <= set(vehicle)
                for key, placeholder in state.items():
                    assert isinstance(vehicle[key], type(placeholder)), key
                assert vehicle["lane_position_m"] == vehicle["s_m"]
                assert vehicle["speed_mps"] == vehicle["v_mps"]
                assert vehicle["acceleration_mps2"] == vehicle["accel_mps2"]
                assert vehicle["road_id"] == PRE_MERGE_LANE.rsplit("_", 1)[0]
                assert vehicle["lane_id"] == PRE_MERGE_LANE
            for entry in row["cav_actions"].values():
                assert set(entry) == set(action)
            assert row["cav_ids"] == list(e2e_mod.CAVS)
            assert set(row["live_ids"]) == set(row["vehicles"])

    def test_fixture_holds_the_twelve_campaign_keys(self):
        spec = json.loads(FIXTURE.read_text(encoding="utf-8"))["row"]
        assert sorted(spec) == sorted([
            "time_s", "live_ids", "departed_ids", "arrived_ids", "colliding_ids",
            "cav_ids", "speed_modes", "cav_actions", "boundary_crossings",
            "vehicles", "starting_teleports", "ending_teleports",
        ])


class TestPiSessionWiring:
    """(b) the real runner binds a REAL PI session; pi-external-u executes."""

    def test_planner_without_initial_plan_evaluates_pi_candidates(
        self, tmp_path, monkeypatch
    ):
        built = []
        plans = []
        real_build = cl.build_mixed_session
        real_plan = cl.mixed_assignment_plan

        def recording_build(*args, **kwargs):
            session = real_build(*args, **kwargs)
            built.append(session)
            return session

        def recording_plan(*args, **kwargs):
            plan = real_plan(*args, **kwargs)
            plans.append(plan)
            return plan

        monkeypatch.setattr(cl, "build_mixed_session", recording_build)
        monkeypatch.setattr(cl, "mixed_assignment_plan", recording_plan)
        out = _run(tmp_path, monkeypatch)  # NO --initial-plan
        assert out.receipt["exit_status"] == "complete"
        assert out.receipt["run_purpose"] == "planner"
        assert out.components.planner_config.actions == cl.ACTIONS  # no pin
        decision = _rows(out.run_dir / "decisions.jsonl")[0]
        assert decision["abstain"] is False
        assert decision["n_candidates_evaluated"] >= 2
        # the SAME construction site built the master session AND every branch's
        assert len(built) >= 2
        assert all(type(s).__name__ == "PlannerMixedAssignmentSession" for s in built)
        assert all(type(s.pi_session).__name__ == "PISession" for s in built)
        assert all(s.pi_session.identity["u_mps"] == 31.29 for s in built)
        # pi-external-u candidates were converted AND applied to branch sessions
        # without the PI refusal (the run completed)
        assert any("pi-external-u" in p.assignments.values() for p in plans)

    def test_master_and_branch_control_rows_carry_pi_commands(self, tmp_path, monkeypatch):
        out = _run(tmp_path, monkeypatch, initial_plan="pi-external-u", near_merge=False)
        assert out.receipt["exit_status"] == "complete"
        control = [
            r for r in _rows(out.run_dir / "dispatcher_rows.jsonl")
            if r.get("source") == "master_control_tick"
        ]
        # ``command`` is the record owner (damping for both damping and PI
        # issues); ``assigned_action`` distinguishes the PI-issued commands
        pi_rows = [r for r in control if r["assigned_action"] == "pi-external-u"]
        assert any(r["fallback"] is None and r["target_speed_mps"] is not None for r in pi_rows)
        branch = _rows(out.run_dir / "branch_control_rows.jsonl")
        assert any(
            r["assigned_action"] == "pi-external-u" and r["fallback"] is None
            and r["issued_target_speed_mps"] is not None
            for r in branch
        )

    def test_actions_option_keeps_the_restriction_explicit(self, tmp_path, monkeypatch):
        base = e2e_mod.make_base_run_pl04i(tmp_path)
        monkeypatch.setenv("SUMO_BINARY", sys.executable)
        monkeypatch.delenv("SUMO_HOME", raising=False)
        monkeypatch.setitem(sys.modules, "traci", e2e_mod._TinyTraci())
        args = e2e_mod._real_args(base)
        args.actions = "native,damping"
        assert build_real_components(args).planner_config.actions == ("native", "damping")
        args.actions = None
        assert build_real_components(args).planner_config.actions == cl.ACTIONS
        parsed = cl.make_parser().parse_args([
            "--out", "x", "--run-id", "r", "--world", "w", "--horizon-s", "1",
            "--predictor", "perfect_sim", "--max-sim-s", "1", "--actions", "native",
        ])
        assert parsed.actions == "native"

    def test_pi_plan_without_a_pi_session_is_a_named_refusal(self):
        from traffic_shepherd.mixed_assignment import AssignmentPlan

        adapter = _adapter()
        session = cl.build_mixed_session(adapter, ["c0", "c1"], with_pi=False)
        assert session.pi_session is None
        plan = AssignmentPlan(
            assignments={"c0": "pi-external-u", "c1": "damping"},
            plan_id="p",
            commit_horizon_s=0.0,
        )
        with pytest.raises(cl.PiSessionMissingError, match="no PI session"):
            session.apply_plan(plan, now_s=0.0)
        assert issubclass(cl.PiSessionMissingError, cl.ClosedLoopError)
        # with the PI session bound the very same plan is accepted
        cl.build_mixed_session(adapter, ["c0", "c1"]).apply_plan(plan, now_s=0.0)
