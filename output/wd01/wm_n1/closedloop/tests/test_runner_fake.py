"""Closed-loop drive with a fake traci api / dispatcher (no SUMO)."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

_WM = Path(__file__).resolve().parents[2]
CODE_ROOT = Path("E:/research-dense")
MANIFEST = CODE_ROOT / "runs/campaign_stage1_chunk4/w8676m2700p20t360__base/saved_state/STATE_MANIFEST.json"


class FakeApi:
    def __init__(self, row, t0):
        self.t = float(t0)
        self.row = row
        self.speed_calls = []
        self.vehicle = SimpleNamespace(
            getIDList=lambda: list(self.row["vehicles"]),
            setSpeed=lambda v, s: self.speed_calls.append((v, s)),
            getSpeed=lambda v: 10.0)
        self.simulation = SimpleNamespace(
            getTime=lambda: self.t, getArrivedIDList=lambda: [],
            getStartingTeleportIDList=lambda: [], getEndingTeleportIDList=lambda: [])

    def simulationStep(self):
        self.t += 0.5


class FakeSession:
    def __init__(self):
        self.plans = []

    def apply_plan(self, plan, now_s):
        self.plans.append((now_s, plan))

    def step(self, ownership, frames, now_s, live_ids):
        return []


@pytest.fixture()
def wired(monkeypatch):
    if not MANIFEST.is_file():
        pytest.skip("TRAIN manifest not on disk")
    from closedloop.runner import setup_paths

    setup_paths(CODE_ROOT)
    import campaign_pilot_v1.branch_types as BT
    import campaign_pilot_v1.suffix_runner as SR

    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    row0 = man["physics_row_at_t0"]

    def fake_row(api, after, previous_row=None, **kw):
        r = copy.deepcopy(row0)
        r["time_s"] = after
        r["departed_ids"] = []
        return r

    monkeypatch.setattr(SR, "build_physics_row", fake_row)
    monkeypatch.setattr(BT, "make_plan", lambda assignments, plan_id, **kw: dict(
        assignments=dict(assignments), plan_id=plan_id))
    monkeypatch.setattr(BT, "dispatcher_rows_from_telemetry", lambda *a, **k: [])
    restore = {"manifest": man, "saved_state_id": "fake"}
    return man, row0, restore


@pytest.mark.parametrize("policy_name", ["always_native", "random"])
def test_drive_closed_loop_fake(tmp_path, wired, policy_name):
    from closedloop import policies as P
    from closedloop.runner import drive_closed_loop

    man, row0, restore = wired
    api = FakeApi(row0, 360.0)
    session = FakeSession()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    world = {"world": "wFAKE", "seed": 1}
    res = drive_closed_loop(
        api, world, "wFAKE__cl", run_dir, restore, session=session,
        ownership=SimpleNamespace(), sensor=None, policy=P.make_policy(policy_name),
        route_departures={}, t0_s=360.0, horizon_s=360.0 + 130.0, decision_period_s=60.0,
        base_world_dir=None, log=lambda m: None)
    assert res["rows_written"] == 260 and res["decisions"] == 3
    assert [p[0] for p in session.plans] == [360.0, 420.0, 480.0]
    roster = set(man["harness_state"]["roster"]["cav_ids"])
    for _t, plan in session.plans:
        assert set(plan["assignments"]) == roster
    decs = [json.loads(x) for x in (run_dir / "decisions.jsonl").read_text().splitlines()]
    assert [d["time_s"] for d in decs] == [360.0, 420.0, 480.0]
    assert decs[0]["n_live_cavs"] == len(roster & set(row0["vehicles"]))
    phys = (run_dir / "physics_steps.jsonl").read_text().splitlines()
    assert len(phys) == 260
    assert (run_dir / "closed_loop_schedule.json").is_file()
    assert (run_dir / "branch_record.json").is_file()
    # t0 release of every live roster CAV
    assert {v for v, s in api.speed_calls if s == -1} == roster & set(row0["vehicles"])
