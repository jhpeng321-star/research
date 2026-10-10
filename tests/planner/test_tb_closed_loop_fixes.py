"""T-B (PL-EXP1b correctness, part 2): restore parity, branch controller-state
carry, ownership write in every branch, t0 release, apply-timing, the
decision-unit reversion, the margin floor, absolute-TTC veto pair logging,
--resume refusal, id-reuse ids, receipts and the PL-02a sweep inheritance.

Fakes only (zero SUMO). The bounded real equivalence checks live in the
opt-in ``test_tb_equivalence_real.py``.
"""

import hashlib
import json
import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from tests.planner import test_pl04i_e2e_real_run as e2e_mod
from tests.planner.test_closed_loop import CAV, VALIDATED_META, Rig
from traffic_shepherd.mixed_assignment import AssignmentPlan, MixedAssignmentSession
from traffic_shepherd.ownership import CommandOwnership
from traffic_shepherd.planner import PlannerConfig, PlanCandidate
from traffic_shepherd.planner import harness_state as hs
from traffic_shepherd.planner.closed_loop import (
    BaseRunResumeRefusedError,
    ClosedLoopConfig,
    ClosedLoopRunner,
    FakeMasterHarness,
    SessionControlWrite,
    aggregate_decisions,
    build_real_components,
    fixed_policy_for_roster,
    resolve_base_run,
)
from traffic_shepherd.planner.interface import plan_key
from traffic_shepherd.planner.loop import NO_DECISION_UNITS, Planner
from traffic_shepherd.planner.mock import MockWorldModel

MASTER = "pl04"


# =========================================================== harness_state
class _Api:
    """Minimal traci surface for a real CommandOwnership."""

    def __init__(self, modes):
        self.modes = dict(modes)
        self.calls = []
        self.vehicle = SimpleNamespace(
            getSpeedMode=lambda v: self.modes[v],
            setSpeedMode=self._set_mode,
            slowDown=lambda v, s, d: self.calls.append(("slowDown", v, s)),
            setSpeed=lambda v, s: self.calls.append(("setSpeed", v, s)),
        )

    def _set_mode(self, v, m):
        self.modes[v] = m
        self.calls.append(("setSpeedMode", v, m))


def _controller():
    cfg = SimpleNamespace(control_dt_s=1.0)

    class Ctl:
        config = cfg

        def decide(self, frame, *, now_s):
            from traffic_shepherd.damping import DampingDecision

            return DampingDecision(fallback_reason="missing_observation")

    return Ctl()


BASE_OWNERSHIP = {
    "cav_speed_mode": 31,  # the campaign manifests record 31 (fleet runs 27)
    "mode_configured": [],
    "owned": {"c0": "damping", "c1": "damping"},
    "last": {
        "c0": {
            "vehicle_id": "c0", "timestamp_s": 358.0, "owner": "damping",
            "network_output_mps2": 0.1, "wall_request_mps2": None,
            "issued_acceleration_mps2": 0.1, "issued_target_speed_mps": 27.0,
            "fallback_reason": None, "released_to_native": False,
            "actor_input_si": [27.0, 27.5, 80.0],
        },
    },
    "time_s": 359.0,
}


class TestOwnershipRestore:
    def test_rebuild_from_a_campaign_manifest_block(self):
        owner = CommandOwnership(_Api({"c0": 27, "c1": 27}), {"c0", "c1"}, _controller(),
                                 cav_speed_mode=27)
        hs.rebuild_ownership(owner, BASE_OWNERSHIP)
        assert owner._owned == {"c0": "damping", "c1": "damping"}
        assert owner._time == 359.0
        assert owner._last["c0"].actor_input_si == (27.0, 27.5, 80.0)
        # captured mode_configured is deliberately NOT restored (re-verified
        # on the first tick); the base's cav_speed_mode 31 is not enforced
        assert owner._mode_configured == set()
        assert owner.cav_speed_mode == 27

    def test_capture_rebuild_round_trip_is_json_safe(self):
        owner = CommandOwnership(_Api({"c0": 27, "c1": 27}), {"c0", "c1"}, _controller(),
                                 cav_speed_mode=27)
        hs.rebuild_ownership(owner, BASE_OWNERSHIP)
        state = hs.capture_ownership(owner)
        assert json.loads(json.dumps(state)) == state
        again = CommandOwnership(_Api({"c0": 27, "c1": 27}), {"c0", "c1"}, _controller(),
                                 cav_speed_mode=27)
        hs.rebuild_ownership(again, state)
        assert hs.capture_ownership(again) == state


class TestSessionRestore:
    def _plan(self, mode, plan_id="p"):
        return AssignmentPlan(assignments={"c0": mode, "c1": "native"}, plan_id=plan_id,
                              commit_horizon_s=0.0, strict_roster=True)

    def test_committed_assignments_and_clocks_round_trip(self):
        ctl = _controller()
        a = MixedAssignmentSession(circles_controller=ctl, pi_session=None, cav_ids={"c0", "c1"})
        a.apply_plan(self._plan("damping"), now_s=10.0)
        a._last_tick = 11.0
        state = hs.capture_session(a)
        assert json.loads(json.dumps(state)) == state
        b = MixedAssignmentSession(circles_controller=ctl, pi_session=None, cav_ids={"c0", "c1"})
        hs.restore_session(b, state)
        assert hs.capture_session(b) == state
        assert b._committed == {"c0": "damping", "c1": "native"}
        assert b._first_plan_applied is True and b._last_tick == 11.0
        # the restored session accepts the next plan and the NEXT tick only
        b.apply_plan(self._plan("native", "q"), now_s=12.0)

    def test_roster_mismatch_refused(self):
        ctl = _controller()
        a = MixedAssignmentSession(circles_controller=ctl, pi_session=None, cav_ids={"c0", "c1"})
        state = hs.capture_session(a)
        b = MixedAssignmentSession(circles_controller=ctl, pi_session=None, cav_ids={"c0", "c2"})
        with pytest.raises(hs.HarnessStateError, match="roster"):
            hs.restore_session(b, state)

    def test_pi_state_round_trip_and_no_pi(self):
        ctl = _controller()
        s = MixedAssignmentSession(circles_controller=ctl, pi_session=None, cav_ids={"c0"})
        assert hs.capture_pi(s) is None
        hs.restore_pi(s, None)  # nothing to do
        # a PI-shaped stand-in: adapter carrying the recurrence state
        inner = SimpleNamespace(_state={"c0": (27.5, float("nan")), "c1": (20.0, 5.0)},
                                _generation={"c0": 2, "c1": 1}, _pending={})
        adapter = SimpleNamespace(pi=inner, _initialized={"c0"}, _tick=5.0, _pending={},
                                  stopped=False)
        s.pi_session = SimpleNamespace(adapter=adapter)
        state = hs.capture_pi(s)
        assert json.loads(json.dumps(state)) == state
        assert state["state"]["c0"][1] is None  # NaN encoded as None
        inner._state, inner._generation, adapter._initialized, adapter._tick = {}, {}, set(), None
        hs.restore_pi(s, state)
        assert np.isnan(inner._state["c0"][1]) and inner._state["c1"] == (20.0, 5.0)
        assert inner._generation == {"c0": 2, "c1": 1} and adapter._initialized == {"c0"}
        assert adapter._tick == 5.0


class TestT0Release:
    """#4: an owner whose restored owner is damping, assigned NATIVE at t0,
    is released via setSpeed(-1) on the FIRST tick (CommandOwnership does it
    for _owned vehicles) -- and the ownership write runs for native too."""

    def test_restored_damping_owner_released_on_first_native_tick(self):
        api = _Api({"c0": 27, "c1": 27})
        ctl = _controller()
        owner = CommandOwnership(api, {"c0", "c1"}, ctl, cav_speed_mode=27)
        hs.rebuild_ownership(owner, BASE_OWNERSHIP)
        session = MixedAssignmentSession(circles_controller=ctl, pi_session=None,
                                         cav_ids={"c0", "c1"})
        session.apply_plan(AssignmentPlan(assignments={"c0": "native", "c1": "native"},
                                          plan_id="n", commit_horizon_s=0.0,
                                          strict_roster=True), now_s=360.0)
        records = session.step(owner, {}, now_s=360.0, live_ids={"c0", "c1"})
        assert [c for c in api.calls if c[0] == "setSpeed"] == [
            ("setSpeed", "c0", -1), ("setSpeed", "c1", -1)]
        assert all(r.owner == "native" and r.released_to_native for r in records)
        assert owner._owned == {}


# ============================================ SessionControlWrite lifecycle
class _Conn:
    def __init__(self, lives, arrived):
        self._lives, self._arrived, self.i = lives, arrived, -1
        self.vehicle = SimpleNamespace(getIDList=lambda: self._lives[self.i])
        self.simulation = SimpleNamespace(getArrivedIDList=lambda: self._arrived[self.i])

    def advance(self):
        self.i += 1


class TestIdReuseIds:
    def test_disappeared_and_reappeared_reach_the_session(self, monkeypatch):
        class Sensor:
            def __init__(self, *a, **k):
                pass

            def capture(self, cav, now_s):
                return SimpleNamespace(vehicle_id=cav)

        monkeypatch.setattr("traffic_shepherd.sensing.SumoLocalSensor", Sensor)
        seen = []

        class Session:
            circles_controller = object()

            def step(self, owner, frames, *, now_s, live_ids, disappeared_ids=(),
                     reappeared_ids=()):
                seen.append((now_s, tuple(disappeared_ids), tuple(reappeared_ids)))
                return []

        conn = _Conn(
            lives=[("a", "b", "h"), ("a", "h"), ("a", "b", "h"), ("h",)],
            arrived=[(), (), (), ("a",)],
        )
        write = SessionControlWrite(conn, Session(), roster=("a", "b"), adapter_config=None,
                                    owner=object())
        for t in (0.0, 1.0, 2.0, 3.0):
            conn.advance()
            write.tick(t)
        assert seen == [
            (0.0, (), ()),
            (1.0, ("b",), ()),      # b left between ticks 0 and 1
            (2.0, (), ("b",)),      # b re-entered: id reuse
            (3.0, ("a", "b"), ()),  # a arrived, b left again
        ]
        assert write.first_control_tick_s == 0.0
        assert set(write.loop_state["retired_cavs"]) == {"a", "b"}

    def test_loop_state_restore_seeds_previous_live(self, monkeypatch):
        class Sensor:
            def __init__(self, *a, **k):
                pass

            def capture(self, cav, now_s):
                return None

        monkeypatch.setattr("traffic_shepherd.sensing.SumoLocalSensor", Sensor)
        seen = []
        session = SimpleNamespace(
            circles_controller=object(),
            step=lambda owner, frames, **kw: seen.append(kw) or [],
        )
        conn = _Conn(lives=[("h",)], arrived=[()])
        write = SessionControlWrite(conn, session, roster=("a",), adapter_config=None,
                                    owner=SimpleNamespace(cav_ids=None))
        write.loop_state = {"previous_live": ["a", "h"], "retired_cavs": []}
        conn.advance()
        write.tick(360.0)
        assert seen[0]["disappeared_ids"] == ("a",)  # seeded history is used


# ====================================================== master / branches
def _e2e(tmp_path, monkeypatch, *, owned=None, horizon_s=10.0, period_s=10.0,
         fixed_policy=None, run_id="tb", drop_ownership=False):
    """The REAL-wiring rig (pl04l pattern) over a base run whose saved
    ownership block lists ``owned`` (id -> owner)."""
    base = e2e_mod.make_base_run_pl04j(tmp_path / "fixture")
    manifest_path = base / "saved_state" / "STATE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if owned is not None:
        manifest["harness_state"]["ownership"]["owned"] = dict(owned)
    if drop_ownership:
        del manifest["harness_state"]["ownership"]
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    resolved = resolve_base_run(base)
    monkeypatch.setenv("SUMO_BINARY", sys.executable)
    monkeypatch.delenv("SUMO_HOME", raising=False)
    traci = e2e_mod._TinyTraci(lane=e2e_mod.PRE_MERGE_LANE)
    monkeypatch.setitem(sys.modules, "traci", traci)
    components = build_real_components(e2e_mod._real_args(base))
    components.planner_config = PlannerConfig(actions=("native", "damping"),
                                              margin_floor_s=0.0)
    made = {}
    real_harness, real_model = components.make_harness, components.make_world_model

    def rec_harness(ctx):
        made["harness"] = real_harness(ctx)
        return made["harness"]

    def rec_model(ctx):
        made["model"] = real_model(ctx)
        return made["model"]

    components.make_harness, components.make_world_model = rec_harness, rec_model
    if fixed_policy is not None:
        components.fixed_plan = fixed_policy_for_roster(components.roster, fixed_policy)
    config = ClosedLoopConfig(
        out_root=tmp_path, run_id=run_id, world="w_tb", seed=e2e_mod.SEED,
        horizon_s=horizon_s, warmup_s=0.0, decision_period_s=period_s,
        predictor="perfect_sim", lookahead_horizon_s=5.0, lookahead_commit_s=None,
        k_samples=1, max_sim_s=800.0, base_run=resolved.base_run,
        base_t0_s=resolved.t0_s, base_state_sha256=resolved.state_sha256,
        max_decision_units=2, walls="off", fixed_policy=fixed_policy,
    )
    return SimpleNamespace(base=base, traci=traci, components=components, config=config,
                           run_dir=tmp_path / run_id, made=made, resolved=resolved)


def _ops_between(ops, start_op, stop_op, *, after=0):
    i = ops.index(start_op, after)
    j = ops.index(stop_op, i + 1)
    return ops[i + 1:j]


class TestMasterRestoreParity:
    def test_first_control_tick_at_t0_before_the_first_step_releases_native(
        self, tmp_path, monkeypatch
    ):
        rig = _e2e(tmp_path, monkeypatch, owned={e2e_mod.CAVS[0]: "damping"},
                   fixed_policy="native", horizon_s=2.0)
        receipt = ClosedLoopRunner(rig.config, rig.components).run()
        master = rig.traci.connections[MASTER]
        ops = master.calls
        first_step = ops.index("simulationStep")
        before_step = ops[:first_step]
        # the tick ran AT t0: sensor + ownership ops and the t0 release
        assert "vehicle.getSpeedMode" in before_step
        assert "vehicle.setSpeed" in before_step  # setSpeed(-1) of the restored owner
        assert receipt["harness_state_restored"] is True
        assert receipt["first_control_tick_s"] == e2e_mod.BASE_T0_S
        digest = hashlib.sha256(
            (rig.base / "saved_state" / "STATE_MANIFEST.json").read_bytes()
        ).hexdigest()
        assert receipt["restore_source_manifest_sha256"] == digest
        assert rig.made["harness"].restore_evidence["owned"] == 1

    def test_no_release_when_the_restored_owner_stays_damping(self, tmp_path, monkeypatch):
        rig = _e2e(tmp_path, monkeypatch, owned={e2e_mod.CAVS[0]: "damping"},
                   fixed_policy="damping", horizon_s=2.0, run_id="tbd")
        ClosedLoopRunner(rig.config, rig.components).run()
        ops = rig.traci.connections[MASTER].calls
        before_step = ops[: ops.index("simulationStep")]
        assert "vehicle.slowDown" in before_step  # commanded AT t0, not t0+1
        assert "vehicle.setSpeed" not in before_step

    def test_a_manifest_without_ownership_is_refused_loudly(self, tmp_path, monkeypatch):
        from traffic_shepherd.planner.closed_loop import BaseRunRefusedError

        rig = _e2e(tmp_path, monkeypatch, fixed_policy="native", horizon_s=2.0,
                   drop_ownership=True)
        with pytest.raises(BaseRunRefusedError, match="harness_state.ownership"):
            ClosedLoopRunner(rig.config, rig.components).run()


class TestBranchStateCarryAndTiming:
    """Two decisions (t0 and t0+10): the second snapshot carries the master's
    ticked state; every branch restores it, runs the ownership write in
    EVERY branch (native included) and ticks AT t0 before its first step."""

    def _run(self, tmp_path, monkeypatch):
        rig = _e2e(tmp_path, monkeypatch, owned={e2e_mod.CAVS[0]: "damping"},
                   horizon_s=20.0, period_s=10.0)
        rig.components.initial_plan = None
        receipt = ClosedLoopRunner(rig.config, rig.components).run()
        return rig, receipt

    def test_snapshot_carries_master_state_and_branches_restore_it(self, tmp_path, monkeypatch):
        rig, receipt = self._run(tmp_path, monkeypatch)
        assert receipt["exit_status"] == "complete" and receipt["decisions"] == 2
        manifests = sorted(rig.run_dir.glob("lookahead_state/**/STATE_MANIFEST.json"))
        by_t = {}
        for path in manifests:
            data = json.loads(path.read_text(encoding="utf-8"))
            by_t.setdefault(data["branch_point_t0_s"], []).append(data)
        first, second = by_t[e2e_mod.BASE_T0_S], by_t[e2e_mod.BASE_T0_S + 10.0]
        h1, h2 = first[0]["harness_state"], second[0]["harness_state"]
        assert set(h2) >= {"ownership", "loop_state", "session", "pi", "roster"}
        # decision 1 (t0): nothing ticked yet -> the restored base ownership
        assert h1["ownership"]["owned"] == {e2e_mod.CAVS[0]: "damping"}
        assert h1["ownership"]["time_s"] == e2e_mod.BASE_T0_S - 1.0
        # decision 2 (t0+10): the master ticked 240..249 -> its own clock
        assert h2["ownership"]["time_s"] == e2e_mod.BASE_T0_S + 9.0
        assert h2["session"]["_last_tick"] == e2e_mod.BASE_T0_S + 9.0
        assert h2["loop_state"]["previous_live"]
        model = rig.made["model"]
        assert model.branch_runner.carried_state_restores >= 4  # 2 decisions x >=2 branches

    def test_every_branch_runs_the_ownership_write_at_t0_before_its_first_step(
        self, tmp_path, monkeypatch
    ):
        rig, _ = self._run(tmp_path, monkeypatch)
        labels = [c["label"] for c in rig.traci.init_calls if c["label"] != MASTER]
        assert labels
        owned_first = 0
        for label in labels:
            ops = rig.traci.connections[label].calls
            load = ops.index("simulation.loadState")
            first_step = ops.index("simulationStep", load)
            window = ops[load + 1:first_step]
            # #3/#5: the ownership write (speed-mode check) ran at t0, before
            # the first physics step, in EVERY branch -- native ones included
            assert "vehicle.getSpeedMode" in window, label
            owned_first += "vehicle.setSpeed" in window
        # #4: the restored damping owner is released at t0 in the native
        # branches (the first decision's native reference at least)
        assert owned_first >= 1

    def test_master_applies_the_chosen_plan_at_its_control_tick_before_the_step(
        self, tmp_path, monkeypatch
    ):
        rig, _ = self._run(tmp_path, monkeypatch)
        ops = rig.traci.connections[MASTER].calls
        saves = [i for i, op in enumerate(ops) if op == "simulation.saveState"]
        last_save = saves[-1]
        next_step = ops.index("simulationStep", last_save)
        # after the last lookahead save and BEFORE the next physics step the
        # master ran the control tick of the decision time (plan-before-step)
        assert "vehicle.getSpeedMode" in ops[last_save:next_step]


class TestBranchPlanFactoryOrdering:
    def test_decision_time_is_the_ticks_time(self, tmp_path, monkeypatch):
        rig = _e2e(tmp_path, monkeypatch, fixed_policy="damping", horizon_s=3.0, run_id="tbo")
        ClosedLoopRunner(rig.config, rig.components).run()
        rows = [
            json.loads(line)
            for line in (rig.run_dir / "dispatcher_rows.jsonl").read_text("utf-8").splitlines()
            if line.strip()
        ]
        control = [r for r in rows if r.get("source") == "master_control_tick"]
        times = sorted({r["t_s"] for r in control})
        assert times == [240.0, 241.0, 242.0]  # t0 first, contiguous


# ===================================================== loop-level: #10 / #7
def _sample(distances, time_s=0.0):
    return SimpleNamespace(
        hybrid=None, arrivals={"seed": 0},
        meta={"sample_id": "s0", "time_s": time_s, "merge_observed": True, "coverage": 1.0},
        frames={c: time_s for c in distances}, frames_distance_to_merge_m=dict(distances),
    )


def _state(cavs):
    return {c: {"action": "native", "wall_id": None, "wall_phase": None,
                "time_in_phase_s": 0.0, "wall_speed_mps": None, "locked": False} for c in cavs}


class RecModel(MockWorldModel):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.plans_seen = []

    def predict_many(self, samples, plans, *a, **kw):
        self.plans_seen.extend(list(plans))
        return super().predict_many(samples, plans, *a, **kw)


class PairModel(MockWorldModel):
    """Mock whose non-native plans breach the absolute TTC floor with a
    scripted triggering pair."""

    PAIR = {"follower_id": "hdv.1", "leader_id": "hdv.2", "ttc_s": 1.0, "time_s": 3.0,
            "pair_class": "hdv_hdv"}

    def predict_many(self, samples, plans, *a, **kw):
        preds = super().predict_many(samples, plans, *a, **kw)
        for plan, pred in zip(plans, preds):
            if any(v != "native" for v in plan.per_cav.values()):
                pred.min_ttc_s[...] = 1.0
                pred.meta["min_ttc_pairs"] = [
                    dict(self.PAIR) for _ in range(pred.min_ttc_s.shape[1])
                ]
        return preds


def _model(cls=MockWorldModel, cavs=CAV):
    model = cls(cavs, part_a_horizon_s=30.0, contract_version="wd01_predict/2",
                model_id="mock/tb")
    model.meta_script.update(VALIDATED_META)
    model.action_delay = {c: {"damping": -80.0} for c in cavs}
    return model


def _cfg(**over):
    base = dict(mode="closed_loop", decide_every_s=1.0, trajectory_s=None,
                margin_floor_s=0.0)
    base.update(over)
    return PlannerConfig(**base)


class TestMarginFloorDefault:
    def test_default_is_60_and_a_small_saving_does_not_switch(self):
        assert PlannerConfig().margin_floor_s == 60.0
        model = _model(cavs=("c0",))
        model.action_delay = {"c0": {"damping": -30.0}}  # 30 < 60 veh*s
        planner = Planner(model, PlannerConfig(mode="closed_loop", trajectory_s=None),
                          current_plan=PlanCandidate(per_cav={"c0": "native"}))
        decision = planner.decide([_sample({"c0": 5.0})], 0.0, controller_state=_state(("c0",)))
        assert decision.plan.per_cav == {"c0": "native"}
        # a saving above the floor switches (K=1 => std 0 => the floor alone)
        model.action_delay = {"c0": {"damping": -80.0}}
        planner = Planner(model, PlannerConfig(mode="closed_loop", trajectory_s=None),
                          current_plan=PlanCandidate(per_cav={"c0": "native"}))
        decision = planner.decide([_sample({"c0": 5.0})], 0.0, controller_state=_state(("c0",)))
        assert decision.plan.per_cav == {"c0": "damping"}


class TestNonSelectedRevertToNative:
    def test_held_cavs_revert_and_candidates_reflect_it(self):
        model = _model(RecModel)
        start = PlanCandidate(per_cav={"c0": "damping", "c1": "damping", "c2": "native"})
        planner = Planner(model, _cfg(max_decision_units=1), current_plan=start)
        decision = planner.decide(
            [_sample({"c0": 300.0, "c1": 50.0, "c2": 120.0})], 0.0,
            controller_state={c: {**s, "action": start.per_cav[c]}
                              for c, s in _state(CAV).items()},
        )
        # c1 is the only decision unit; c0 (was damping) reverts to native
        assert decision.plan.per_cav["c0"] == "native"
        assert decision.plan.per_cav["c2"] == "native"
        assert decision.switched is True
        assert any(r.candidate == "revert_non_decision_units" for r in decision.reasons)
        # EVERY plan the lookahead evaluated carries the reversion (the
        # current plan, the alternatives and the combined plan alike)
        evaluated = model.plans_seen
        assert evaluated
        # the reversion runs BEFORE the probe: the probe, the current plan,
        # every alternative and the combined plan all have c0 native
        assert all(p.per_cav["c0"] == "native" for p in evaluated)

    def test_zero_units_is_an_abstain_named_no_decision_units(self):
        planner = Planner(_model(), _cfg(max_decision_units=1),
                          current_plan=PlanCandidate(per_cav={c: "damping" for c in CAV}))
        decision = planner.decide([SimpleNamespace(
            hybrid=None, arrivals={"seed": 0},
            meta={"sample_id": "s", "time_s": 0.0, "merge_observed": True, "coverage": 1.0},
        )], 0.0, controller_state={c: {**s, "action": "damping"}
                                    for c, s in _state(CAV).items()})
        assert decision.abstain is True
        assert any(r.step == "abstain" and r.candidate == NO_DECISION_UNITS
                   for r in decision.reasons)
        assert set(decision.plan.per_cav.values()) == {"native"}  # everyone reverted

    def test_runner_records_the_abstain_reason(self, tmp_path):
        rig = Rig(tmp_path, run_id="nu", max_decision_units=1, horizon_s=60.0)
        receipt = rig.run()
        rows = rig.read_log("decisions.jsonl")
        assert rows and all(r["abstain"] and r["abstain_reason"] == "no_decision_units"
                            for r in rows)
        assert receipt["abstains"] == len(rows)
        assert receipt["abstain_reasons"] == {"no_decision_units": len(rows)}


class TestAbsoluteTtcVetoPairs:
    def test_planner_logs_the_pair_of_every_absolute_breach(self):
        model = _model(PairModel)
        planner = Planner(model, _cfg(max_decision_units=1),
                          current_plan=PlanCandidate(per_cav={c: "native" for c in CAV}))
        decision = planner.decide([_sample({"c0": 10.0, "c1": 50.0, "c2": 90.0})], 0.0,
                                  controller_state=_state(CAV))
        assert decision.ttc_vetoes
        entry = decision.ttc_vetoes[0]
        assert (entry["follower_id"], entry["leader_id"]) == ("hdv.1", "hdv.2")
        assert entry["pair_class"] == "hdv_hdv" and entry["ttc_s"] == 1.0
        assert entry["plan"] == "c0=damping"
        # the rule itself is unchanged: the damping plan is still vetoed
        assert decision.plan.per_cav["c0"] == "native"

    def test_class_cav_involved_and_unknown_pair(self):
        class CavPair(PairModel):
            PAIR = {**PairModel.PAIR, "follower_id": "c1", "pair_class": "cav_involved"}

        for cls, expected in ((CavPair, "cav_involved"), (PairModel, "hdv_hdv")):
            planner = Planner(_model(cls), _cfg(max_decision_units=1),
                              current_plan=PlanCandidate(per_cav={c: "native" for c in CAV}))
            decision = planner.decide([_sample({"c0": 10.0})], 0.0,
                                      controller_state=_state(CAV))
            assert {e["pair_class"] for e in decision.ttc_vetoes} == {expected}

        class NoPair(MockWorldModel):
            def predict_many(self, samples, plans, *a, **kw):
                preds = super().predict_many(samples, plans, *a, **kw)
                for plan, pred in zip(plans, preds):
                    if any(v != "native" for v in plan.per_cav.values()):
                        pred.min_ttc_s[...] = 0.5
                return preds

        planner = Planner(_model(NoPair), _cfg(max_decision_units=1),
                          current_plan=PlanCandidate(per_cav={c: "native" for c in CAV}))
        decision = planner.decide([_sample({"c0": 10.0})], 0.0, controller_state=_state(CAV))
        assert {e["pair_class"] for e in decision.ttc_vetoes} == {"unknown_pair"}

    def test_row_and_receipt_carry_the_pairs_and_counts(self, tmp_path):
        harness = FakeMasterHarness(CAV, distances_to_merge_m={"c0": 10.0, "c1": 20.0,
                                                               "c2": 30.0})
        rig = Rig(tmp_path, run_id="tv", model=_model(PairModel), harness=harness,
                  max_decision_units=1, horizon_s=30.0)
        receipt = rig.run()
        rows = rig.read_log("decisions.jsonl")
        assert rows[0]["abs_ttc_vetoes"]
        assert rows[0]["abs_ttc_vetoes"][0]["pair_class"] == "hdv_hdv"
        counts = receipt["abs_ttc_veto_counts"]
        assert counts == {"hdv_hdv": sum(len(r["abs_ttc_vetoes"]) for r in rows)}
        assert aggregate_decisions(rows)["abs_ttc_veto_counts"] == counts


# ====================================================== resume refusal (#11)
class TestResumeRefusal:
    def test_base_run_runs_are_never_resumed(self, tmp_path):
        rig = Rig(tmp_path, run_id="rr", base_run=tmp_path / "some__base", base_t0_s=10.0,
                  base_state_sha256="a" * 64)
        with pytest.raises(BaseRunResumeRefusedError, match="NEW run dir"):
            rig.run(resume=True)
        assert not rig.run_dir.exists()  # refused before anything was created

    def test_non_base_resume_still_reports_nothing_to_resume(self, tmp_path):
        from traffic_shepherd.planner.closed_loop import NothingToResumeError

        rig = Rig(tmp_path, run_id="rr2")
        with pytest.raises(NothingToResumeError):
            rig.run(resume=True)


# ============================================= receipts of non-real starts
class TestReceiptFieldsOnFakes:
    def test_fake_master_reports_not_restored(self, tmp_path):
        rig = Rig(tmp_path, run_id="rf")
        receipt = rig.run()
        assert receipt["harness_state_restored"] is False
        assert receipt["first_control_tick_s"] is None
        assert receipt["restore_source_manifest_sha256"] is None


# ================================================= PL-02a sweep inheritance
class TestPl02aSweepInheritsTheCallersConfig:
    def test_tolerance_and_margin_sweeps_vary_only_their_field(self):
        pytest.importorskip("output.pl02a_offline_v1.run")
        from output.pl02a_offline_v1 import run as pl02a

        caller = PlannerConfig(relative_ttc_threshold_s=3.0, margin_floor_s=7.0,
                               ttc_floor_s=2.0, trajectory_s=12.0, rel_safety_tol_s=1.0,
                               margin_k=3.0)
        tol = pl02a.tolerance_sweep_config(caller, 5.0)
        assert tol == replace(caller, rel_safety_tol_s=5.0)
        assert tol.relative_ttc_threshold_s == 3.0 and tol.margin_floor_s == 7.0
        assert tol.ttc_floor_s == 2.0 and tol.margin_k == 3.0
        margin = pl02a.margin_sweep_config(caller, 30.0)
        assert margin == replace(caller, margin_floor_s=30.0)
        assert margin.relative_ttc_threshold_s == 3.0 and margin.rel_safety_tol_s == 1.0


# ===================================== launcher argv + budget projection
class TestLauncherArgvStillParses:
    def test_planner_and_reference_argv_parse_and_project_within_budget(self, tmp_path):
        from traffic_shepherd.planner.closed_loop import (
            config_from_args,
            make_parser,
            project_run_cost,
        )

        base = e2e_mod.make_base_run_pl04i(tmp_path)
        resolved = resolve_base_run(base)
        common = [
            "--components", "real", "--world", "W", "--split", "train",
            "--seed", str(resolved.seed), "--base-run", str(base),
            "--horizon-s", "600", "--walls", "off", "--process-start-s", "0.688",
            "--out", "E:/rd-wt/pl39/tmp_tb/argv_out", "--predictor", "perfect_sim",
        ]
        parser = make_parser()
        ref = config_from_args(parser.parse_args(
            common + ["--fixed-policy", "native", "--max-sim-s", "700", "--run-id", "r1"]))
        assert ref.fixed_policy == "native"
        plan = config_from_args(parser.parse_args(
            common + ["--decision-period-s", "60", "--lookahead-horizon-s", "60",
                      "--lookahead-commit-s", "60", "--k-samples", "1",
                      "--max-decision-units", "2", "--max-sim-s", "9000",
                      "--run-id", "p1"]))
        components = build_real_components(SimpleNamespace(
            predictor="perfect_sim", seed=resolved.seed, base_run=str(base),
            decision_period_s=60.0, lookahead_horizon_s=60.0, lookahead_commit_s=60.0,
        ))
        planner_cfg = ClosedLoopRunner(plan, components)._planner_config()
        assert planner_cfg.margin_floor_s == 60.0  # owner default reaches the run
        projection = project_run_cost(plan, n_cavs=59, planner_config=planner_cfg)
        assert projection["projected_sim_s"] <= 9000.0
        assert plan_key(PlanCandidate(per_cav={"a": "native"}))  # smoke import use
