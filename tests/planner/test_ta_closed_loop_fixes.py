"""T-A closed-loop correctness fixes (zero SUMO in this file).

(1) #1: the perfect-sim lookahead delay is the p4 EVALUATION delay --
    ALL corridor vehicles (CAVs and HDVs), advance-based free flow, entry
    waits incl. the insertion backlog; the branch extractor equals
    ``recorded.p4_delay`` on the same rows and window (within 1e-9) and
    Part A is the committed window. Old-code evidence: the pristine
    extractor's CAV-presence value on the same fixture.
(2) #6: the branch TTC uses the PHYSICAL gap (200 m ``getLeader`` gap PLUS
    the follower's ``getMinGap``), matching ``metrics.ttc_exposure``.
(3) PlannerConfig.relative_ttc_threshold_s (1.5 default, {1.5, 3.0}): the
    relative rule selects the TTC exposure field from the config and fails
    CLOSED with a named error when the selected field is absent; the 3 s
    fields keep their frozen meaning and carry a report-only SHADOW
    verdict; decision rows + receipt log BOTH exposures.
(4) #16: the NATIVE reference branch is never early-stopped (an in-window
    collision still runs its full window); a plan branch may stop early.
(5) --fixed-policy native|damping|pi: a REFERENCE run (run_purpose
    'reference') with no decision points and no branch launches, command
    rows for every live roster CAV every control tick, the master horizon
    as the whole budget projection, and mutual exclusion with
    --initial-plan. The launcher argv of
    output/plexp1b_oracle_v1_launch/run_plexp1b.py is accepted as-is.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from traffic_shepherd.metrics import closing_ttc
from traffic_shepherd.planner import MockWorldModel, PlannerConfig, plan_key
from traffic_shepherd.planner import closed_loop as cl
from traffic_shepherd.planner.closed_loop import (
    BudgetRefusedError,
    CAV_SPEED_MODE,
    ClosedLoopConfig,
    ClosedLoopRunner,
    FakeMasterHarness,
    RUN_PURPOSE_PLANNER,
    RUN_PURPOSE_REFERENCE,
    fixed_policy_for_roster,
    make_parser,
    config_from_args,
    project_run_cost,
)
from traffic_shepherd.planner.interface import PlanCandidate
from traffic_shepherd.planner.perfect_sim import (
    BranchRecord,
    BranchRunner,
    MetricsExtractor,
    PerfectSimWorldModel,
    SimpleArrivalDraws,
    StepRecord,
    TraciBranchRunner,
    VehicleStep,
)
from traffic_shepherd.planner.perfect_sim import P4DelayEvaluator
from traffic_shepherd.planner.recorded import (
    Layout,
    _p4_net_path,
    load_p4_delay,
    p4_delay,
)
from traffic_shepherd.planner.paths import rd_wt_root
from tests.planner._helpers import RecordingProcessFactory, make_samples, plan

pytestmark = pytest.mark.skipif(
    not (rd_wt_root() / "wd01c" / "output" / "wd01").is_dir(),
    reason="WD-01 p4 delay library absent (E:/rd-wt/wd01c/output/wd01)",
)


# ----------------------------------------------------------------- fixture
T0 = 360.0
WINDOW = 120.0
DT = 0.5
CHAIN_ROAD = "1191554636"
CHAIN_LANE = "1191554636_0"  # WM-30 main_right chain lane (limit 31.29 m/s)
VALIDATED_META = {
    "future_min_ttc_validated": True,
    "future_min_ttc_validation_id": "ta-validated",
    "relative_safety_validated": True,
    "relative_safety_validation_id": "ta-validated",
}


def _fixture_rows(hdv_speed=15.0, backlog_first_seen=390.0):
    """Campaign-schema physics rows on the WM-30 chain lane: a CAV and an
    HDV present at t0 plus a vehicle with a scheduled depart BEFORE its
    first sighting (the insertion backlog)."""
    rows = []
    positions = {"cav0": 100.0, "hdv1": 60.0}
    speeds = {"cav0": 20.0, "hdv1": hdv_speed, "backlog1": 20.0}
    t = T0
    while t < T0 + WINDOW + 1e-9:
        vehicles = {
            vid: {
                "road_id": CHAIN_ROAD,
                "lane_id": CHAIN_LANE,
                "lane_position_m": positions[vid],
                "speed_mps": speeds[vid],
                "acceleration_mps2": 0.0,
            }
            for vid in positions
        }
        if t >= backlog_first_seen - 1e-9:
            positions.setdefault("backlog1", 0.0)
            vehicles["backlog1"] = {
                "road_id": CHAIN_ROAD,
                "lane_id": CHAIN_LANE,
                "lane_position_m": positions["backlog1"],
                "speed_mps": 20.0,
                "acceleration_mps2": 0.0,
            }
        rows.append(
            {
                "time_s": t,
                "vehicles": vehicles,
                "colliding_ids": [],
                "cav_ids": ["cav0"],
            }
        )
        for vid in positions:
            positions[vid] += speeds.get(vid, 20.0) * DT
        t += DT
    departs = {"cav0": 100.0, "hdv1": 110.0, "backlog1": backlog_first_seen - 10.0}
    t0_present = ("cav0", "hdv1")
    return rows, departs, t0_present


def _record_from_rows(rows, t0_present):
    """The BranchRecord exactly as TraciBranchRunner records the fixture
    (campaign road identity on every vehicle; HDVs included)."""
    steps = []
    for row in rows:
        vehicles = {
            vid: VehicleStep(
                s_m=float(v["lane_position_m"]),
                lane_id=1.0,
                speed_mps=float(v["speed_mps"]),
                accel_mps2=float(v["acceleration_mps2"]),
                road_id=str(v["road_id"]),
                sumo_lane_id=str(v["lane_id"]),
            )
            for vid, v in row["vehicles"].items()
        }
        steps.append(StepRecord(time_s=float(row["time_s"]), dt_s=DT, vehicles=vehicles))
    return BranchRecord(
        plan_label="native",
        draw_id="k0",
        t0_s=T0,
        horizon_s=WINDOW,
        dt_s=DT,
        free_flow_s={"cav0": 0.0},  # the OLD runner default: free_flow_of = 0
        steps=tuple(steps),
        tracked=("cav0",),
        cav_ids=("cav0",),
        t0_present_ids=tuple(t0_present),
    )


@pytest.fixture(scope="module")
def p4_lib():
    core, geometry = load_p4_delay()
    geom = geometry.wm30_geometry(net_path=_p4_net_path())
    chains = getattr(geometry, "WM30_CHAINS", None)
    return core, geom, chains


def _direct_p4(p4_lib, rows, departs, t0_present, window_s):
    core, geom, chains = p4_lib
    delay, waits, n_vehicles, _diag = p4_delay(
        rows,
        T0,
        window_s,
        Layout(),
        departs,
        core,
        geom,
        t0_present_ids=frozenset(t0_present),
        lane_chains=chains,
    )
    return delay, waits, n_vehicles


# --------------------------------------------- (1) the p4 evaluation delay
class TestItem1P4DelayObjective:
    def test_branch_delay_equals_p4_on_same_rows_and_window(self, p4_lib):
        rows, departs, t0_present = _fixture_rows()
        core, geom, chains = p4_lib
        evaluator = P4DelayEvaluator(
            departs=departs, layout=Layout(), core=core, geom=geom, lane_chains=chains
        )
        extractor = MetricsExtractor(delay_evaluator=evaluator)
        record = _record_from_rows(rows, t0_present)
        out = extractor.extract(record, part_a_horizon_s=30.0, trajectory_s=0.0)
        expected_full, waits, n_vehicles = _direct_p4(p4_lib, rows, departs, t0_present, WINDOW)
        assert out["delay_s"] == pytest.approx(expected_full, abs=1e-9)
        # ALL corridor vehicles are billed (CAV + HDV + backlog arrival)
        assert n_vehicles == 3
        # Part A IS the committed window, from the same evaluator
        expected_a, _w, _n = _direct_p4(p4_lib, rows, departs, t0_present, 30.0)
        assert out["delay_part_a_s"] == pytest.approx(expected_a, abs=1e-9)
        assert out["delay_part_a_s"] + out["delay_part_b_s"] == pytest.approx(
            out["delay_s"], abs=1e-9
        )

    def test_hdvs_and_entry_waits_now_count(self, p4_lib):
        core, geom, chains = p4_lib
        rows, departs, t0_present = _fixture_rows()
        evaluator = P4DelayEvaluator(
            departs=departs, layout=Layout(), core=core, geom=geom, lane_chains=chains
        )
        record = _record_from_rows(rows, t0_present)
        out = MetricsExtractor(delay_evaluator=evaluator).extract(
            record, part_a_horizon_s=30.0, trajectory_s=0.0
        )
        # the insertion-backlog wait (scheduled 380.0, first seen 390.0) is
        # IN the delay: removing it drops the delay by exactly 10.0 s
        rows_punctual, departs_punctual, _ = _fixture_rows(backlog_first_seen=390.0)
        departs_punctual = dict(departs_punctual)
        departs_punctual["backlog1"] = 390.0
        evaluator2 = P4DelayEvaluator(
            departs=departs_punctual, layout=Layout(), core=core, geom=geom, lane_chains=chains
        )
        out2 = MetricsExtractor(delay_evaluator=evaluator2).extract(
            _record_from_rows(rows_punctual, t0_present), part_a_horizon_s=30.0, trajectory_s=0.0
        )
        assert out["delay_s"] - out2["delay_s"] == pytest.approx(10.0, abs=1e-9)
        # the HDV counts: slowing it changes the total delay
        rows_slow_hdv, departs2, t0p2 = _fixture_rows(hdv_speed=10.0)
        evaluator3 = P4DelayEvaluator(
            departs=departs2, layout=Layout(), core=core, geom=geom, lane_chains=chains
        )
        out3 = MetricsExtractor(delay_evaluator=evaluator3).extract(
            _record_from_rows(rows_slow_hdv, t0p2), part_a_horizon_s=30.0, trajectory_s=0.0
        )
        assert out3["delay_s"] > out["delay_s"]

    def test_old_extractor_gave_the_cav_presence_value(self, p4_lib):
        # OLD-CODE EVIDENCE (pristine extractor, 2026-09-30): the same
        # fixture gave delay_s == 331.5 (CAV-presence seconds: free_flow_of
        # = 0 for the CAV, HDV/backlog transit billed raw, entry waits
        # absent) while the p4 evaluation delay is 148.24544582933913 s.
        rows, departs, t0_present = _fixture_rows()
        legacy = MetricsExtractor().extract(
            _record_from_rows(rows, t0_present), part_a_horizon_s=30.0, trajectory_s=0.0
        )
        assert legacy["delay_s"] == pytest.approx(331.5, abs=1e-9)
        expected_full, _w, _n = _direct_p4(p4_lib, rows, departs, t0_present, WINDOW)
        assert expected_full == pytest.approx(148.24544582933913, abs=1e-6)
        assert legacy["delay_s"] != pytest.approx(expected_full, abs=1.0)


# ------------------------------------------------- (2) the true-gap TTC
class _FakeVehicleApi:
    def getIDList(self):
        return ("c0",)

    def getLanePosition(self, vid):
        return 100.0

    def getLaneIndex(self, vid):
        return 0

    def getSpeed(self, vid):
        return 10.0 if vid == "lead" else 20.0

    def getAcceleration(self, vid):
        return 0.0

    def getLeader(self, vid, dist):
        # raw query gap 17.5 + minGap 2.5 -> the PHYSICAL gap 20.0
        return ("lead", 17.5)

    def getMinGap(self, vid):
        return 2.5

    def getRoadID(self, vid):
        return "r0"

    def getLaneID(self, vid):
        return "r0_0"


class _FakeApi:
    vehicle = _FakeVehicleApi()


class TestItem2TrueGapTtc:
    def test_gap_includes_follower_mingap_and_ttc_matches_metrics(self):
        runner = TraciBranchRunner(cav_ids=("c0",))
        step = runner._vehicle_step(1.0, _FakeApi())
        vs = step.vehicles["c0"]
        # the PHYSICAL bumper-to-bumper gap: 17.5 raw + 2.5 minGap
        assert vs.leader_gap_m == pytest.approx(20.0)
        ttc = closing_ttc(vs.leader_gap_m, vs.speed_mps, vs.leader_speed_mps)
        assert ttc == pytest.approx(closing_ttc(20.0, 20.0, 10.0))
        assert ttc == pytest.approx(2.0)
        # OLD-CODE EVIDENCE (pristine runner, 2026-09-30): the biased-low
        # value from the raw getLeader gap was 1.75 s.
        assert closing_ttc(17.5, 20.0, 10.0) == pytest.approx(1.75)

    def test_extractor_safety_uses_the_physical_gap(self):
        runner = TraciBranchRunner(cav_ids=("c0",))
        step = runner._vehicle_step(1.0, _FakeApi())
        minimum, exposure3, exposure15, _brake, has_pair = MetricsExtractor()._safety([step])
        # TTC 2.0 s on the PHYSICAL gap: inside the 3 s window, outside the
        # 1.5 s one (the biased-low 1.75 s raw-gap value is also < 3)
        assert minimum == pytest.approx(2.0)
        assert has_pair is True
        assert exposure3 == pytest.approx(step.dt_s)
        assert exposure15 == pytest.approx(0.0)


# --------------------------------- (3) the relative-TTC threshold (1.5/3)
class TestItem3RelativeTtcThreshold:
    def test_config_default_and_validation(self):
        assert PlannerConfig().relative_ttc_threshold_s == 1.5
        PlannerConfig(relative_ttc_threshold_s=3.0)
        with pytest.raises(ValueError, match="relative_ttc_threshold_s"):
            PlannerConfig(relative_ttc_threshold_s=2.0)
        with pytest.raises(ValueError, match="relative_ttc_threshold_s"):
            PlannerConfig(relative_ttc_threshold_s=1.0)

    @staticmethod
    def _pred_with_lt3_rise():
        """A prediction whose TTC<3 exposure rises vs native (+2.0 veh*s)
        while its TTC<1.5 exposure does not (both 0)."""
        world = MockWorldModel(("c0",), part_a_horizon_s=10.0)
        candidate = plan({"c0": "damping"})
        world.safety_script[plan_key(candidate)] = {"ttc_lt3_exposure_s": 2.0}
        world.native_safety["ttc_lt3_exposure_s"] = 0.0
        return world.predict_many(make_samples(1, time_s=0.0), [candidate])[0]

    def test_a_lt3_rise_alone_not_vetoed_by_default_vetoed_at_3(self):
        from traffic_shepherd.planner.safety import safety_check

        pred = self._pred_with_lt3_rise()
        default = safety_check(pred, PlannerConfig())
        assert default.ok
        assert default.relative[0].metric == "ttc_lt1p5_exposure_s"
        assert default.relative[0].ok
        # the 3 s SHADOW verdict (report-only) shows what the old rule
        # would have said: a +2.0 rise with tol 0 fails
        assert [v.metric for v in default.shadow_relative_3s] == ["ttc_lt3_exposure_s"]
        assert default.shadow_relative_3s[0].ok is False
        strict = safety_check(pred, PlannerConfig(relative_ttc_threshold_s=3.0))
        assert not strict.ok
        assert strict.relative[0].metric == "ttc_lt3_exposure_s"
        assert not strict.relative[0].ok
        # at threshold 3.0 the primary verdict IS the 3 s one: no shadow
        assert strict.shadow_relative_3s == ()

    def test_b_missing_lt1p5_field_fails_closed_named(self):
        from dataclasses import replace

        from traffic_shepherd.planner.safety import (
            RelativeSafetyFieldMissingError,
            safety_check,
        )

        world = MockWorldModel(("c0",), part_a_horizon_s=10.0)
        pred = world.predict_many(make_samples(1, time_s=0.0), [plan({"c0": "damping"})])[0]
        stripped = replace(pred, ttc_lt1p5_exposure_s=None)
        with pytest.raises(RelativeSafetyFieldMissingError, match="ttc_lt1p5_exposure_s"):
            safety_check(stripped, PlannerConfig())
        v2 = replace(
            pred,
            ttc_lt1p5_exposure_s=None,
            meta={**pred.meta, "contract_version": "wd01_predict/2"},
        )
        with pytest.raises(RelativeSafetyFieldMissingError, match="ttc_lt1p5_exposure_s"):
            safety_check(v2, PlannerConfig())
        # a missing future twin on a v2 prediction fails closed too
        v2_future = replace(
            v2, ttc_lt1p5_exposure_s=np.zeros((1, 1)), future_ttc_lt1p5_exposure_s=None
        )
        with pytest.raises(RelativeSafetyFieldMissingError, match="future_ttc_lt1p5"):
            safety_check(v2_future, PlannerConfig())

    @staticmethod
    def _rig_run(tmp_path, run_id, script, native=None):
        cav = ("c0", "c1", "c2")
        model = MockWorldModel(
            cav, part_a_horizon_s=30.0, contract_version="wd01_predict/2", model_id="mock/ta"
        )
        model.meta_script.update(VALIDATED_META)
        # the CHOSEN plan with the coupling + saving is the all-damping
        # group candidate: script ITS exposure block
        candidate = PlanCandidate(per_cav={c: "damping" for c in cav})
        model.safety_script[plan_key(candidate)] = script
        if native:
            model.native_safety.update(native)
        model.traj_coupling_pos[("c0", "c1")] = {"damping": 10.0}
        # make the damping candidate the CHOSEN plan (a real saving), so the
        # row's exposure block reads the candidate's prediction
        model.action_delay = {c: {"damping": -5.0} for c in cav}
        config = ClosedLoopConfig(
            out_root=tmp_path,
            run_id=run_id,
            world="w-ta",
            horizon_s=40.0,
            warmup_s=0.0,
            decision_period_s=30.0,
            predictor="mock",
            lookahead_horizon_s=30.0,
            max_sim_s=1e9,
        )
        components = SimpleNamespace(
            make_harness=lambda context: FakeMasterHarness(cav),
            make_world_model=lambda context: model,
            roster=cav,
            planner_config=PlannerConfig(margin_floor_s=0.0),
            initial_plan=None,
            sim_seconds_fn=None,
        )
        ClosedLoopRunner(config, components).run()
        text = (config.run_dir / "decisions.jsonl").read_text(encoding="utf-8")
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        receipt = json.loads(
            (config.run_dir / "RUN_RECEIPT.json").read_text(encoding="utf-8")
        )
        return rows, receipt

    def test_decision_row_logs_both_exposures_and_the_shadow(self, tmp_path):
        rows, receipt = self._rig_run(
            tmp_path,
            "ta3-row",
            script={"ttc_lt3_exposure_s": 2.0},
            native={"ttc_lt3_exposure_s": 0.0},
        )
        assert rows
        for row in rows:
            block = row["ttc_exposure"]
            assert block is not None
            assert block["relative_ttc_threshold_s"] == 1.5
            # BOTH exposures (plan + native, window) are logged; the 3 s
            # shadow verdict names the frozen metric and fails here
            assert block["plan"]["ttc_lt1p5_exposure_s"] is not None
            assert block["plan"]["native_ttc_lt1p5_exposure_s"] is not None
            assert block["plan"]["ttc_lt3_exposure_s"][0][0] == pytest.approx(2.0)
            assert block["plan"]["native_ttc_lt3_exposure_s"][0][0] == pytest.approx(0.0)
            # the v2 future twins are logged when the predictor reports them
            assert block["future"]["ttc_lt3_exposure_s"] is not None
            assert block["future"]["ttc_lt1p5_exposure_s"] is not None
            assert block["future"]["native_ttc_lt3_exposure_s"] is not None
            shadow = block["shadow_relative_3s"]
            assert shadow and shadow[0]["metric"] == "ttc_lt3_exposure_s"
            assert shadow[0]["ok"] is False
        assert receipt["ttc_exposure_summary"]["rows_with_exposures"] == len(rows)
        assert receipt["ttc_exposure_summary"]["shadow_relative_3s_fail"] >= 1
        assert receipt["ttc_exposure_summary"]["thresholds"] == [1.5]


# ------------------------------ (4) the native reference never stops early
class _FlagAwareRunner(BranchRunner):
    """Scripted branches that honour the runner-level stop flag: a branch
    with an in-window collision stops early ONLY when
    ``stop_on_window_collision`` is True (the model hands native False)."""

    def __init__(self, *, collision_keys=(), **kwargs):
        super().__init__(**kwargs)
        self.collision_keys = set(collision_keys)
        self.calls = []

    def run(
        self,
        plan,
        *,
        connection,
        manifest,
        draw_id,
        departures,
        horizon_s,
        safety_window_s,
        controller_state,
        kept_wall_ids,
        stop_on_window_collision=True,
        commit_s=None,
        continuation=None,
    ):
        key = "native" if getattr(plan, "label", None) == "native" else plan_key(plan)
        collides = key in self.collision_keys
        stopped = collides and stop_on_window_collision
        self.calls.append({"plan": key, "stop": bool(stop_on_window_collision)})
        t0 = float(manifest["t0_s"])
        steps = tuple(
            StepRecord(
                time_s=t0 + i,
                dt_s=1.0,
                vehicles={
                    "c0": VehicleStep(
                        s_m=20.0 * i, lane_id=0.0, speed_mps=20.0, accel_mps2=0.0
                    )
                },
            )
            for i in range(int(horizon_s) + 1)
        )
        return BranchRecord(
            plan_label=str(key),
            draw_id=draw_id,
            t0_s=t0,
            horizon_s=float(horizon_s),
            dt_s=1.0,
            free_flow_s={"c0": 30.0},
            steps=steps,
            collision_times=(t0 + 5.0,) if collides else (),
            tracked=("c0",),
            stopped_early=stopped,
            stopped_at_s=t0 + 5.0 if stopped else None,
        )


def _flag_model(runner):
    if getattr(runner, "process_factory", None) is None:
        runner.process_factory = RecordingProcessFactory()
    snapshotter = SimpleNamespace(
        save=lambda *, t0_s, controller_state, kept_wall_ids: {
            "saved_state_dir": "<fake>",
            "t0_s": float(t0_s),
            "sumo_version": "1.27.1",
            "network_sha256": "f00d",
        },
        restore=lambda manifest, *, plan, draw_id, api: None,
    )
    return (
        PerfectSimWorldModel(
            snapshotter, SimpleArrivalDraws(), runner, part_a_horizon_s=5.0
        ),
        snapshotter,
    )


def _ta_sample():
    return [
        SimpleNamespace(
            hybrid=None,
            # one explicit option-A draw: the model declares no scenario,
            # so the PL-10c guard requires non-empty explicit departures
            arrivals=((70.0, "main", 20.0, False, "passenger"),),
            meta={"sample_id": "k0", "time_s": 60.0, "merge_observed": True, "coverage": 1.0},
        )
    ]


class TestItem4NativeNeverEarlyStop:
    def test_native_reference_runs_full_window_with_in_window_collision(self):
        plan_a = plan({"c0": "damping"})
        runner = _FlagAwareRunner(collision_keys={"native"})
        model, _snapshotter = _flag_model(runner)
        pred = model.predict_many(_ta_sample(), [plan_a], None, horizon_s=10.0)[0]
        # the native branch collided INSIDE the safety window yet ran the
        # FULL window: its metrics are not truncated (PL-06 option A stays
        # operative), and the collision stays recorded
        assert bool(pred.native_collision[0, 0])
        assert np.isfinite(pred.native_delay_s[0, 0])
        assert pred.meta["early_stop_cells"] == []
        (native_call,) = [c for c in runner.calls if c["plan"] == "native"]
        assert native_call["stop"] is False

    def test_plan_branch_may_still_stop_early(self):
        plan_a = plan({"c0": "damping"})
        runner = _FlagAwareRunner(collision_keys={plan_key(plan_a)})
        model, _snapshotter = _flag_model(runner)
        pred = model.predict_many(_ta_sample(), [plan_a], None, horizon_s=10.0)[0]
        assert bool(pred.collision[0, 0])
        assert np.all(np.isnan(pred.plan_delay_s))
        assert np.all(np.isnan(pred.delay_vs_native_s))
        assert pred.meta["early_stop_cells"] == [0]
        (plan_call,) = [c for c in runner.calls if c["plan"] == plan_key(plan_a)]
        assert plan_call["stop"] is True


# -------------------------------------------- (5) --fixed-policy reference
def _cli_args(tmp_path, base, *extra):
    return make_parser().parse_args(
        [
            "--components", "real",
            "--world", "w8568m2600p05t360",
            "--split", "train",
            "--seed", "8468",
            "--base-run", str(base),
            "--horizon-s", "600",
            "--walls", "off",
            "--process-start-s", "0.688",
            "--out", "E:/rd-wt/pl38/tmp_ta/cli_out",
            "--predictor", "perfect_sim",
            "--fixed-policy", "native",
            "--max-sim-s", "700",
            "--run-id", "w__ref_native",
            *extra,
        ]
    )


class TestItem5FixedPolicyCli:
    def test_launcher_argv_accepted_as_is(self, tmp_path):
        from tests.planner._helpers import make_base_run

        base = make_base_run(tmp_path)
        config = config_from_args(_cli_args(tmp_path, base))
        assert config.fixed_policy == "native"
        assert config.run_purpose == RUN_PURPOSE_REFERENCE
        assert config.horizon_s == 600.0
        assert config.max_sim_s == 700.0
        assert config.process_start_s == 0.688
        assert config.walls == "off"
        assert config.predictor == "perfect_sim"
        pi_args = make_parser().parse_args(
            [
                "--components", "real",
                "--world", "w8568m2600p05t360",
                "--split", "train",
                "--seed", "8468",
                "--base-run", str(base),
                "--horizon-s", "600",
                "--walls", "off",
                "--process-start-s", "0.688",
                "--out", "E:/rd-wt/pl38/tmp_ta/cli_out",
                "--predictor", "perfect_sim",
                "--fixed-policy", "pi",
                "--max-sim-s", "700",
                "--run-id", "w__ref_pi",
            ]
        )
        pi_config = config_from_args(pi_args)
        assert pi_config.fixed_policy == "pi"
        assert pi_config.run_purpose == RUN_PURPOSE_REFERENCE

    def test_mutual_exclusion_with_initial_plan(self, tmp_path):
        from tests.planner._helpers import make_base_run

        base = make_base_run(tmp_path)
        with pytest.raises(cl.BuildFailedError, match="mutually exclusive"):
            config_from_args(_cli_args(tmp_path, base, "--initial-plan", "damping"))
        with pytest.raises(ValueError, match="mutually exclusive"):
            ClosedLoopConfig(
                out_root=tmp_path,
                run_id="x",
                world="w",
                fixed_policy="native",
                initial_plan_action="damping",
            )

    def test_fixed_policy_validation_and_planner_default(self):
        with pytest.raises(ValueError, match="fixed_policy"):
            ClosedLoopConfig(out_root=Path("E:/x"), run_id="x", world="w", fixed_policy="bogus")
        # planner runs are unaffected
        cfg = ClosedLoopConfig(out_root=Path("E:/x"), run_id="x", world="w")
        assert cfg.fixed_policy is None and cfg.run_purpose == RUN_PURPOSE_PLANNER

    def test_fixed_policy_for_roster_maps_the_action(self):
        assert fixed_policy_for_roster(("b", "a"), "native").per_cav == {
            "a": "native",
            "b": "native",
        }
        assert fixed_policy_for_roster(("c0",), "pi").per_cav == {"c0": "pi-external-u"}
        assert fixed_policy_for_roster(("c0",), "damping").per_cav == {"c0": "damping"}
        with pytest.raises(ValueError, match="fixed policy"):
            fixed_policy_for_roster(("c0",), "walls")


class TestItem5FixedPolicyReferenceRun:
    @staticmethod
    def _run(tmp_path, policy, run_id, *, horizon_s=20.0, max_sim_s=1e9):
        """One REFERENCE run on the FAKE world: no decisions, no lookahead,
        the fixed plan applied at t0, per-tick command rows."""
        cav = ("c0", "c1", "c2")
        model = MockWorldModel(
            cav, part_a_horizon_s=30.0, contract_version="wd01_predict/2", model_id="mock/ref"
        )
        model.meta_script.update(VALIDATED_META)
        predict_calls = []
        original = model.predict_many

        def counting(*args, **kwargs):
            predict_calls.append(1)
            return original(*args, **kwargs)

        model.predict_many = counting
        harness = FakeMasterHarness(cav, emit_control_rows=True)
        config = ClosedLoopConfig(
            out_root=tmp_path,
            run_id=run_id,
            world="w-ta",
            horizon_s=horizon_s,
            warmup_s=0.0,
            decision_period_s=30.0,
            predictor="mock",
            lookahead_horizon_s=30.0,
            max_sim_s=max_sim_s,
            fixed_policy=policy,
        )
        components = SimpleNamespace(
            make_harness=lambda context: harness,
            make_world_model=lambda context: model,
            roster=cav,
            planner_config=PlannerConfig(margin_floor_s=0.0),
            fixed_plan=fixed_policy_for_roster(cav, policy),
            initial_plan=None,
            sim_seconds_fn=None,
        )
        receipt = ClosedLoopRunner(config, components).run()

        def read_log(name):
            text = (config.run_dir / name).read_text(encoding="utf-8")
            return [json.loads(line) for line in text.splitlines() if line.strip()]

        return receipt, read_log, predict_calls, config, cav

    @pytest.mark.parametrize("policy", ["native", "damping", "pi"])
    def test_reference_run_semantics_per_policy(self, tmp_path, policy):
        receipt, read_log, predict_calls, config, cav = self._run(
            tmp_path, policy, f"ref-{policy}"
        )
        # receipt identity: run_purpose 'reference' + fixed_policy; the
        # predictor_class is unchanged (mock); no lookahead was spent
        assert receipt["run_purpose"] == RUN_PURPOSE_REFERENCE
        assert receipt["fixed_policy"] == policy
        assert receipt["predictor_class"] == "mock"
        assert receipt["sim_s"]["lookahead_s"] == 0.0
        assert receipt["decisions"] == 0
        # no decision rows, no lookahead branch launches (the model was
        # never asked to predict)
        assert read_log("decisions.jsonl") == []
        assert predict_calls == []
        # the fixed plan reached every roster CAV at t0 ...
        apply_rows = [
            row for row in read_log("dispatcher_rows.jsonl") if row.get("note") == "fixed_policy"
        ]
        assert {row["cav"] for row in apply_rows} == set(cav)
        # ... and the ownership control write ran EVERY control tick for
        # every live roster CAV: native rows are the selected_native
        # ownership record; damping/pi rows are commanded.
        control_rows = [row for row in read_log("dispatcher_rows.jsonl") if row.get("kind") == "control"]
        ticks = sorted({row["t_s"] for row in control_rows})
        assert ticks and ticks[0] == pytest.approx(1.0)
        for tick in ticks:
            rows_at_tick = [row for row in control_rows if row["t_s"] == tick]
            assert {row["cav"] for row in rows_at_tick} == set(cav)
            for row in rows_at_tick:
                if policy == "native":
                    assert row["command"] == "native"
                    assert row["fallback"] == "selected_native"
                    assert row["assigned_action"] == "native"
                else:
                    assert row["command"] == "damping"
                    assert row["fallback"] is None
                    expected = "damping" if policy == "damping" else "pi-external-u"
                    assert row["assigned_action"] == expected
        # mode 27 on every live roster CAV in all three modes
        physics = read_log("physics_steps.jsonl")
        assert physics
        for row in physics:
            assert row["cav_speed_mode"] == CAV_SPEED_MODE == 27

    def test_projection_is_the_master_horizon_only(self, tmp_path):
        config = ClosedLoopConfig(
            out_root=tmp_path,
            run_id="proj",
            world="w-ta",
            horizon_s=600.0,
            predictor="perfect_sim",
            max_sim_s=1e9,
            fixed_policy="damping",
        )
        projection = project_run_cost(config, n_cavs=8, planner_config=PlannerConfig())
        assert projection["decision_points"] == 0
        assert projection["lookahead_sim_s"] == 0.0
        assert projection["branches"] == 0
        assert projection["projected_sim_s"] == pytest.approx(600.0)
        assert projection["cap_s"] == pytest.approx(1e9)
        # a reference run above --max-sim-s is refused up front
        capped = ClosedLoopConfig(
            out_root=tmp_path,
            run_id="proj2",
            world="w-ta",
            horizon_s=600.0,
            predictor="perfect_sim",
            max_sim_s=599.0,
            fixed_policy="damping",
        )
        with pytest.raises(BudgetRefusedError, match="max-sim-s cap"):
            project_run_cost(capped, n_cavs=8, planner_config=PlannerConfig())

    def test_reference_run_refuses_an_initial_plan_component(self, tmp_path):
        cav = ("c0",)
        config = ClosedLoopConfig(
            out_root=tmp_path,
            run_id="ref-guard",
            world="w-ta",
            horizon_s=10.0,
            predictor="mock",
            max_sim_s=1e9,
            fixed_policy="native",
        )
        components = SimpleNamespace(
            make_harness=lambda context: FakeMasterHarness(cav),
            make_world_model=lambda context: MockWorldModel(cav),
            roster=cav,
            planner_config=PlannerConfig(margin_floor_s=0.0),
            fixed_plan=fixed_policy_for_roster(cav, "native"),
            initial_plan=fixed_policy_for_roster(cav, "damping"),
            sim_seconds_fn=None,
        )
        with pytest.raises(cl.BuildFailedError, match="mutually exclusive"):
            ClosedLoopRunner(config, components).run()
