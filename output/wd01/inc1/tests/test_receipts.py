"""Receipt schema + a fake-traci warm-up/arm run (no SUMO)."""
from __future__ import annotations

import json

import pytest

from inc1 import constants as C
from inc1 import gates as G
from inc1 import plan as P
from inc1 import runner as R
from inc1 import schedules as S
from inc1.tests.fake_traci import FakeTraci


class FakeOwnership:
    def __init__(self):
        self.applied = []

    def apply(self, plan, *, now_s, live_ids, states, api):
        self.applied.append((now_s, dict(plan.wall), tuple(plan.native)))
        return []


def _world():
    return P.build_plan(seeds=C.seed_block(),
                        include_scenario_binding=False)["worlds"][0]


def _api():
    api = FakeTraci(time_s=0.0)
    api.set_lane("1191554636_0", length=2644.2)
    api.set_lane("1191554636_1", length=2644.2)
    api.set_vehicle("ramp.0", lane_id="1191554636_0", position=1000.0,
                    speed=15.0, type_id="cautious_car")
    api.set_vehicle("cav.0", lane_id="1191554636_0", position=500.0,
                    speed=20.0, type_id="cav.0")
    api.set_vehicle("cav.1", lane_id="1191554636_1", position=480.0,
                    speed=20.0, type_id="cav.0")
    return api


def _runner(tmp_path, *, api=None):
    gate_info = {"registration_lf_sha256": C.REGISTRATION_LF_SHA}
    return R.IncidentRunner(_plan(), tmp_path, machine_id="testbox",
                            platform="linux", gate_info=gate_info,
                            budget=G.BudgetCounter(tmp_path / "budget.jsonl"))


def _plan():
    return P.build_plan(seeds=C.seed_block(), include_scenario_binding=False)


def test_receipt_requires_all_fields():
    with pytest.raises(R.RunnerError):
        R.validate_receipt({"schema": C.RECEIPT_SCHEMA})


def test_receipt_write_once(tmp_path):
    world = _world()
    receipt = R.build_receipt(
        world=world, arm="R0", machine_id="m", platform="linux",
        sumo_version="SUMO 1.27.1", saved_state_digest="d" * 64,
        obstacle=None, schedule_check={"arm": "R0"}, collisions=[],
        teleports=[], executed_time_s=300.0, physics_rows=600)
    R.validate_receipt(receipt)
    path = R.write_receipt(tmp_path / "run", receipt)
    assert path.is_file()
    with pytest.raises(R.RunnerError):
        R.write_receipt(tmp_path / "run", receipt)


def test_warmup_saves_state_and_writes_receipt(tmp_path):
    runner = _runner(tmp_path)
    world = _world()
    api = _api()
    receipt = runner.warm_up(world, api=api)
    assert receipt["arm"] == "warmup"
    assert receipt["saved_state_digest"]
    assert api.saved_states
    state = tmp_path / world["warmup_run"] / "run" / "state.xml"
    assert state.is_file()


def test_run_arm_restores_inserts_and_records(tmp_path):
    runner = _runner(tmp_path)
    world = _world()
    state = tmp_path / world["warmup_run"] / "run" / "state.xml"
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text("<state/>\n", encoding="utf-8")

    api = _api()
    ownership = FakeOwnership()
    states_at_t0 = [
        S.CavState("cav.0", 20.0, -2144.22, 1),
        S.CavState("cav.1", 20.0, -2164.22, 0),
    ]
    receipt = runner.run_arm(
        world, "R2", api=api, ownership=ownership,
        states_at_t0=states_at_t0,
        obstacle={"obstacle_id": "incident.obstacle", "position_m": 40.0,
                  "shift_m": 0.0, "hold_s": C.OBSTACLE_HOLD_S},
        restore_digest="d" * 64)
    assert receipt["arm"] == "R2"
    assert receipt["obstacle"]["obstacle_id"] == "incident.obstacle"
    assert receipt["arm_schedule_check"]["arm"] == "R2"
    assert receipt["physics_rows"] == 600
    assert receipt["executed_time_s"] == pytest.approx(300.0, abs=1e-6)
    assert api.load_calls                      # restored exactly once
    log = (tmp_path / f"{world['world']}__r2" / "run" / "physics_steps.jsonl")
    rows = [json.loads(line) for line in log.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 600
    assert "time_s" in rows[0] and "vehicles" in rows[0]
    assert ownership.applied                   # CommandOwnership drove the arm


def test_run_r5_arm_issues_multi_cav_commands(tmp_path):
    """ADDENDUM_2: R5 runs through the shared generator + CommandOwnership."""
    runner = _runner(tmp_path)
    world = _world()
    state = tmp_path / world["warmup_run"] / "run" / "state.xml"
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text("<state/>\n", encoding="utf-8")
    obs = float(world["incident_s_down_m"])
    states_at_t0 = [
        S.CavState("cav.0", 20.0, obs - 200.0, 1, native_accel_mps2=1.0),
        S.CavState("cav.1", 20.0, obs - 100.0, 0, native_accel_mps2=1.0),
    ]
    ownership = FakeOwnership()
    receipt = runner.run_arm(
        world, "R5", api=_api(), ownership=ownership,
        states_at_t0=states_at_t0,
        obstacle={"obstacle_id": "incident.obstacle", "position_m": obs,
                  "shift_m": 0.0, "hold_s": C.OBSTACLE_HOLD_S},
        restore_digest="d" * 64)
    assert receipt["arm"] == "R5"
    assert receipt["arm_schedule_check"]["feasible"] is True
    assert receipt["arm_schedule_check"]["wall_members"]
    assert ownership.applied
    assert ownership.applied[0][1]                # non-empty wall command map


def test_dispatch_requires_api_factory_and_spends_nothing(tmp_path):
    """R4 is no longer blocked (ADDENDUM_1 A1.2): dispatch with no api_factory
    fails before any sim-s is spent."""
    runner = _runner(tmp_path)
    with pytest.raises(R.RunnerError):
        runner.run_all()
    assert not (tmp_path / "LAUNCH_SUMMARY.json").exists()
    assert not (tmp_path / "budget.jsonl").exists()
