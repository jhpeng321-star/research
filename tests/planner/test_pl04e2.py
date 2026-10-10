"""PL-04e2 unit tests (fakes only, zero SUMO): the --base-run start, the
--max-decision-units cap, and the --walls off switch of the closed-loop
runner. All SUMO surfaces are the established faked Popen/traci/sensor/
ownership rigs from test_closed_loop.py."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.planner._helpers import make_base_run
from tests.planner.test_closed_loop import (  # the established rig fakes
    _FakeBranchSensor,
    _FakeBranchVehicle,
    _FakeConnection,
    _FakeTraciModule,
    _RecordingCommandOwnership,
    _RecordingPopen,
    Rig,
)
from traffic_shepherd.planner import PlannerConfig, PlanCandidate
from traffic_shepherd.planner.closed_loop import (
    BaseRunRefusedError,
    BuildFailedError,
    BuildContext,
    ClosedLoopConfig,
    ClosedLoopRunner,
    RosterSourceError,
    WallsOnRefusedError,
    build_real_components,
    config_from_args,
    make_parser,
    project_run_cost,
    resolve_base_run,
    roster_from_base_run,
)
from traffic_shepherd.planner.loop import Planner, select_decision_units
from traffic_shepherd.planner.mock import MockWorldModel

CAV = ("c0", "c1", "c2")
VALIDATED_META = {
    "future_min_ttc_validated": True,
    "future_min_ttc_validation_id": "test-validated",
    "relative_safety_validated": True,
    "relative_safety_validation_id": "test-validated",
}


def _cli_args(*extra):
    return make_parser().parse_args(
        [
            "--out", "E:/rd-wt/pl16/output",
            "--run-id", "x",
            "--world", "w-ok",
            "--horizon-s", "100",
            "--predictor", "mock",
            "--max-sim-s", "10",
            *extra,
        ]
    )


def _sample(time_s, distances):
    return SimpleNamespace(
        hybrid=None,
        arrivals={"seed": 0},
        meta={"sample_id": "s0", "time_s": time_s, "merge_observed": True, "coverage": 1.0},
        frames={cav: time_s for cav in distances},
        frames_distance_to_merge_m=dict(distances),
    )


def _native_state(cavs):
    return {
        cav: {
            "action": "native",
            "wall_id": None,
            "wall_phase": None,
            "time_in_phase_s": 0.0,
            "wall_speed_mps": None,
            "locked": False,
        }
        for cav in cavs
    }


def _damping_model(cavs):
    model = MockWorldModel(
        cavs, part_a_horizon_s=30.0, contract_version="wd01_predict/2", model_id="mock/e2"
    )
    model.meta_script.update(VALIDATED_META)
    model.action_delay = {cav: {"damping": -5.0} for cav in cavs}
    return model


# ------------------------------------------------- decision-unit cap (loop)
class TestPL04e2DecisionUnits:
    def test_select_ranks_by_distance_then_id(self):
        samples = [_sample(0.0, {"c0": 300.0, "c1": 50.0, "c2": 50.0})]
        units, held = select_decision_units(("c2", "c0", "c1"), samples, 2)
        assert units == ("c1", "c2")  # 50 m tie -> id order; c0 (300 m) held
        assert held == ("c0",)

    def test_select_uses_the_freshest_sample_and_finite_distances(self):
        stale = _sample(10.0, {"c0": 400.0, "c1": 10.0})
        fresh = _sample(20.0, {"c0": 30.0, "c1": float("nan")})
        units, held = select_decision_units(("c0", "c1"), [stale, fresh], 2)
        assert units == ("c0",)  # freshest sample wins; NaN is not rankable
        assert held == ("c1",)

    def test_select_with_no_distance_info_holds_everyone(self):
        samples = [SimpleNamespace(meta={"time_s": 0.0})]
        units, held = select_decision_units(("c1", "c0"), samples, 2)
        assert units == ()
        assert held == ("c0", "c1")

    def test_individual_stage_only_considers_the_capped_units(self):
        samples = [_sample(0.0, {"c0": 300.0, "c1": 50.0, "c2": 120.0})]
        planner = Planner(
            _damping_model(CAV),
            PlannerConfig(
                mode="closed_loop", decide_every_s=1.0, max_decision_units=1,
                trajectory_s=None, margin_floor_s=0.0,
            ),
            current_plan=PlanCandidate(per_cav={c: "native" for c in CAV}),
        )
        decision = planner.decide(samples, 0.0, controller_state=_native_state(CAV))
        # only c1 (the nearest decision unit) may move; the others keep native
        assert decision.plan.per_cav == {"c0": "native", "c1": "damping", "c2": "native"}
        cap_rows = [r for r in decision.reasons if r.action == "cap"]
        assert cap_rows and "decision_units=c1" in cap_rows[0].candidate
        assert "c0+c2" in cap_rows[0].detail  # held units are named

    def test_uncapped_stage_considers_everyone(self):
        samples = [_sample(0.0, {"c0": 300.0, "c1": 50.0, "c2": 120.0})]
        planner = Planner(
            _damping_model(CAV),
            PlannerConfig(
                mode="closed_loop", decide_every_s=1.0, trajectory_s=None, margin_floor_s=0.0
            ),
            current_plan=PlanCandidate(per_cav={c: "native" for c in CAV}),
        )
        decision = planner.decide(samples, 0.0, controller_state=_native_state(CAV))
        assert decision.plan.per_cav == {c: "damping" for c in CAV}

    def test_projection_uses_capped_units_and_prices_no_wall(self):
        config = ClosedLoopConfig(
            out_root=Path("E:/rd-wt/pl16/output"), run_id="p", world="w-ok",
            horizon_s=60.0, decision_period_s=30.0, predictor="perfect_sim",
            lookahead_horizon_s=10.0, max_sim_s=1e9,
        )
        capped = project_run_cost(
            config,
            n_cavs=66,
            planner_config=PlannerConfig(max_walls=0, max_decision_units=2),
        )
        assert capped["decision_units"] == 2
        uncapped = project_run_cost(
            config, n_cavs=66, planner_config=PlannerConfig(max_walls=0)
        )
        assert uncapped["decision_units"] == 66
        assert capped["lookahead_sim_s"] < uncapped["lookahead_sim_s"]
        # walls off: W=0 -- no wall candidate is priced
        walls_on = project_run_cost(config, n_cavs=2, planner_config=PlannerConfig())
        assert walls_on["lookahead_sim_s"] > capped["lookahead_sim_s"]

    def test_config_validation(self):
        with pytest.raises(ValueError, match="max_decision_units"):
            PlannerConfig(max_decision_units=0)
        assert PlannerConfig(max_decision_units=2).max_decision_units == 2
        assert PlannerConfig(max_walls=0).max_walls == 0  # walls off is legal now
        with pytest.raises(ValueError, match="max_walls"):
            PlannerConfig(max_walls=-1)


# --------------------------------------------------------- --walls off/on
class TestPL04e2Walls:
    def test_real_default_is_off_mock_default_is_on(self, tmp_path):
        base = make_base_run(tmp_path, seed=2718)
        assert config_from_args(
            _cli_args("--components", "real", "--seed", "2718", "--base-run", str(base))
        ).walls == "off"
        assert config_from_args(_cli_args("--components", "mock")).walls == "on"
        assert config_from_args(
            _cli_args("--components", "mock", "--walls", "off")
        ).walls == "off"

    def test_real_walls_on_refused_until_pl04e2_walls(self, tmp_path):
        base = make_base_run(tmp_path, seed=2718)
        with pytest.raises(WallsOnRefusedError, match="PL-04e2-walls"):
            config_from_args(
                _cli_args(
                    "--components", "real", "--seed", "2718",
                    "--base-run", str(base), "--walls", "on",
                )
            )

    def test_walls_off_pins_max_walls_zero_and_skips_proposals(self, tmp_path):
        rig = Rig(tmp_path, run_id="walls_off", walls="off")
        runner = ClosedLoopRunner(rig.config, rig.components)
        assert runner._planner_config().max_walls == 0
        rig.model.action_delay = {c: {"damping": -5.0} for c in rig.cav}
        rig.run()
        rows = rig.read_log("decisions.jsonl")
        assert rows
        assert all(row["walls"] == [] for row in rows)
        vetoed = [v for row in rows for v in row["vetoes"] if "walls off" in v["detail"]]
        assert vetoed


# ------------------------------------------------------------- --base-run
class TestPL04e2BaseRun:
    def test_resolve_accepts_a_valid_base_run(self, tmp_path):
        base = make_base_run(tmp_path, seed=8468)
        resolved = resolve_base_run(base)
        assert resolved.t0_s == 240.0
        assert resolved.seed == 8468
        assert len(resolved.state_sha256) == 64
        assert resolved.scenario_dir.name == "scenario"

    def test_resolve_refuses_a_dir_without_saved_state(self, tmp_path):
        empty = tmp_path / "w0__base"
        empty.mkdir()
        with pytest.raises(BaseRunRefusedError, match="no saved_state"):
            resolve_base_run(empty)

    def test_resolve_refuses_state_digest_drift(self, tmp_path):
        base = make_base_run(tmp_path)
        state = base / "saved_state" / "state.xml"
        state.write_text(state.read_text(encoding="utf-8").replace("28.14", "99.99"),
                         encoding="utf-8")
        with pytest.raises(BaseRunRefusedError, match="sha256"):
            resolve_base_run(base)

    def test_resolve_refuses_scenario_digest_drift(self, tmp_path):
        base = make_base_run(tmp_path)
        route = base / "run" / "scenario" / "requests.rou.xml"
        route.write_text(
            route.read_text(encoding="utf-8").replace('depart="0.0"', 'depart="0.5"'),
            encoding="utf-8",
        )
        with pytest.raises(BaseRunRefusedError, match="scenario digest drift"):
            resolve_base_run(base)

    def test_roster_is_the_full_base_run_physics_cav_ids(self, tmp_path):
        base = make_base_run(
            tmp_path,
            damping_cavs=("mainline.6", "mainline.7"),
            other_owner="mainline.5",
            late_cav="mainline.99",
        )
        roster, evidence = roster_from_base_run(base, 240.0)
        # PL-04m: the FULL designated set from the base run's constant
        # physics cav_ids -- the post-t0 late_cav is IN (it was excluded
        # before PL-04m), the non-designated other_owner is NOT (it is not
        # in the physics cav_ids list at all)
        assert roster == ("mainline.6", "mainline.7", "mainline.99")
        assert evidence["roster_source"] == "base_run_physics_cav_ids"
        assert evidence["roster_size"] == 3
        assert evidence["roster_ids"] == ["mainline.6", "mainline.7", "mainline.99"]
        assert evidence["n_inserted_le_t0"] == 2
        assert evidence["n_inserted_after_t0_or_never"] == 1
        with pytest.raises(RosterSourceError, match="no usable cav_ids"):
            roster_from_base_run(make_base_run(tmp_path / "empty", cav_ids=[]), 4.0)

    def test_cli_real_requires_base_run(self):
        with pytest.raises(BuildFailedError, match="--base-run"):
            config_from_args(_cli_args("--components", "real", "--seed", "1"))

    def test_cli_real_refuses_cav_ids(self, tmp_path):
        base = make_base_run(tmp_path, seed=1)
        with pytest.raises(BuildFailedError, match="--cav-ids is mock-only"):
            config_from_args(
                _cli_args(
                    "--components", "real", "--seed", "1",
                    "--base-run", str(base), "--cav-ids", "c0,c1",
                )
            )

    def test_cli_real_refuses_scenario_alongside_base_run(self, tmp_path):
        base = make_base_run(tmp_path, seed=1)
        with pytest.raises(BuildFailedError, match="--scenario is not used"):
            config_from_args(
                _cli_args(
                    "--components", "real", "--seed", "1",
                    "--base-run", str(base), "--scenario", "s.cfg",
                )
            )

    def test_cli_refuses_seed_mismatch_vs_base_state(self, tmp_path):
        base = make_base_run(tmp_path, seed=8468)
        with pytest.raises(BaseRunRefusedError, match="does not match the base state's seed"):
            config_from_args(
                _cli_args("--components", "real", "--seed", "999", "--base-run", str(base))
            )

    def test_cli_mock_refuses_base_run(self, tmp_path):
        base = make_base_run(tmp_path)
        with pytest.raises(BuildFailedError, match="--base-run is a real-run option"):
            config_from_args(_cli_args("--components", "mock", "--base-run", str(base)))

    def test_cli_refuses_max_decision_units_below_one(self, tmp_path):
        base = make_base_run(tmp_path, seed=1)
        with pytest.raises(BuildFailedError, match="--max-decision-units"):
            config_from_args(
                _cli_args(
                    "--components", "real", "--seed", "1",
                    "--base-run", str(base), "--max-decision-units", "0",
                )
            )


# ------------------------------------------- base-run start over real parts
class _MainlineVehicleSurface(_FakeBranchVehicle):
    """Branch vehicle surface whose live ids are the base run's roster."""

    def getIDList(self):
        return ("mainline.6", "mainline.7")


class _MainlineConnection(_FakeConnection):
    def __init__(self, label):
        super().__init__(label)
        self.vehicle = _MainlineVehicleSurface()


class _MainlineTraci(_FakeTraciModule):
    """The traci fake with the campaign roster as the live vehicle ids."""

    def __init__(self, **kwargs):
        kwargs.setdefault("vehicle_ids", ("mainline.6", "mainline.7"))
        super().__init__(**kwargs)

    def init(self, port=None, host=None, numRetries=None, proc=None, label=None,
             doSwitch=True):
        self.init_calls.append({"label": label, "port": int(port), "pid": proc.pid})
        if self._master_label is None:
            self._master_label = label  # getConnection(master) -> the module
        self.connections[label] = _MainlineConnection(label)
        return self.getVersion()


class TestPL04e2BaseRunStartReal:
    """The base-run start over the REAL harness/components with faked
    Popen/traci/sensor/ownership (zero SUMO): ONE guarded loadState, the
    scenario COPY inside the run dir, the branches launched from the copy,
    and the receipt evidence callables."""

    @pytest.fixture()
    def rig(self, monkeypatch, tmp_path):
        base = make_base_run(tmp_path, world="w8468chunk1train", seed=8468)
        popen = _RecordingPopen()
        traci = _MainlineTraci(allow_master_loadstate=True)
        monkeypatch.setattr("subprocess.Popen", popen)
        monkeypatch.setitem(sys.modules, "traci", traci)
        monkeypatch.setattr("traffic_shepherd.sensing.SumoLocalSensor", _FakeBranchSensor)
        monkeypatch.setattr(
            "traffic_shepherd.ownership.CommandOwnership", _RecordingCommandOwnership
        )
        del _RecordingCommandOwnership.created[:]
        monkeypatch.delenv("SUMO_BINARY", raising=False)
        monkeypatch.setenv("SUMO_HOME", "E:/sumo_home_fake_for_test")
        args = SimpleNamespace(
            scenario=None,
            base_run=str(base),
            predictor="perfect_sim",
            cav_ids=None,
            decision_period_s=30.0,
            lookahead_horizon_s=30.0,
            lookahead_commit_s=10.0,
            seed=8468,
        )
        components = build_real_components(args)
        context = BuildContext(run_dir=tmp_path / "run1", resume=False, process_seq=1)
        return SimpleNamespace(
            base=base, popen=popen, traci=traci,
            components=components, context=context, tmp_path=tmp_path,
        )

    def test_start_lands_on_t0_with_one_guarded_loadstate(self, rig):
        harness = rig.components.make_harness(rig.context)
        harness.start()
        # the copied scenario is what the master launched with...
        copied_cfg = rig.context.run_dir / "scenario" / "scenario.sumocfg"
        assert copied_cfg.is_file()
        master_cmd = rig.popen.calls[0].cmd
        assert master_cmd[master_cmd.index("-c") + 1] == str(copied_cfg)
        # the net-file of the COPY was rewritten to the E: assets network
        cfg_text = copied_cfg.read_text(encoding="utf-8")
        assert "assets/research/highway_merge_shepherd_long/osm.net.xml.gz" in cfg_text
        assert "legacy-mount" not in cfg_text
        # ...and the fresh process received exactly ONE loadState: the base
        # state (guarded restore), landing the clock on the base t0
        assert len(rig.traci.master_loads) == 1
        assert rig.traci.master_loads[0].startswith(str(rig.base / "saved_state"))
        assert harness.now_s() == 240.0
        # the base state travelled with the harness (the receipt's source)
        assert harness.base_state["t0_s"] == 240.0
        harness.close()

    def test_branches_launch_from_the_copied_cfg(self, rig):
        harness = rig.components.make_harness(rig.context)
        harness.start()
        model = rig.components.make_world_model(rig.context)
        sample = SimpleNamespace(
            hybrid={"tracked": {}},
            arrivals={"sample_id": "k0", "departures": ()},
            meta={"sample_id": "k0", "time_s": 240.0, "merge_observed": True, "coverage": 1.0},
        )
        plan = PlanCandidate(
            per_cav={
                "mainline.6": "damping",
                "mainline.7": "native",
                "mainline.99": "native",  # PL-04m: the roster includes the post-t0 CAV
            }
        )
        (model.predict_many(
            [sample], [plan], {}, horizon_s=30.0, commit_s=10.0,
            continuation="native", trajectory_s=None,
        ))
        copied_cfg = str(rig.context.run_dir / "scenario" / "scenario.sumocfg")
        branch_cmds = [proc.cmd for proc in rig.popen.calls[1:]]
        assert branch_cmds
        for cmd in branch_cmds:
            assert cmd[cmd.index("-c") + 1] == copied_cfg
        for cmd in [proc.cmd for proc in rig.popen.calls]:
            assert cmd[cmd.index("--seed") + 1] == "8468"
        harness.close()

    def test_branch_control_rows_and_launch_records_are_recorded(self, rig):
        harness = rig.components.make_harness(rig.context)
        harness.start()
        model = rig.components.make_world_model(rig.context)
        sample = SimpleNamespace(
            hybrid={"tracked": {}},
            arrivals={"sample_id": "k0", "departures": ()},
            meta={"sample_id": "k0", "time_s": 240.0, "merge_observed": True, "coverage": 1.0},
        )
        plan = PlanCandidate(
            per_cav={
                "mainline.6": "damping",
                "mainline.7": "native",
                "mainline.99": "native",  # PL-04m: the roster includes the post-t0 CAV
            }
        )
        model.predict_many(
            [sample], [plan], {}, horizon_s=30.0, commit_s=10.0,
            continuation="native", trajectory_s=None,
        )
        summary = rig.components.branch_control_summary()
        assert summary["count"] > 0  # criterion (b) evidence exists
        assert summary["by_branch"]
        rows_file = rig.context.run_dir / "branch_control_rows.jsonl"
        first_row = json.loads(rows_file.read_text(encoding="utf-8").splitlines()[0])
        assert first_row["kind"] == "branch_control"
        assert first_row["command"] in ("damping", "native")
        records = rig.components.launch_records()
        kinds = {record["kind"] for record in records}
        assert kinds == {"master", "branch"}
        assert len(records) == len(rig.popen.calls)
        for record in records:
            assert record["cmd"][record["cmd"].index("--seed") + 1] == "8468"
        harness.close()


# ---------------------------------------------- receipt + base-t0 offsets
class TestPL04e2RunnerReceipt:
    def test_base_t0_offsets_horizon_and_receipt(self, tmp_path):
        from traffic_shepherd.planner.closed_loop import FakeMasterHarness

        base = make_base_run(tmp_path, seed=8468)
        harness = FakeMasterHarness(("mainline.6", "mainline.7"))
        harness.time = 240.0  # the base-state t0 (a fresh state IS at t0)
        config = ClosedLoopConfig(
            out_root=tmp_path, run_id="base1", world="w-ok", seed=8468,
            horizon_s=30.0, warmup_s=0.0, decision_period_s=15.0,
            predictor="mock", lookahead_horizon_s=30.0, max_sim_s=1e9,
            base_run=base, base_t0_s=240.0, base_state_sha256="a" * 64,
            max_decision_units=2, walls="off",
        )
        components = SimpleNamespace(
            make_harness=lambda context: harness,
            make_world_model=lambda context: _damping_model(harness.cav_ids),
            roster=harness.cav_ids,
            planner_config=PlannerConfig(margin_floor_s=0.0),
            initial_plan=None,
            sim_seconds_fn=None,
        )
        receipt = ClosedLoopRunner(config, components).run()
        # decisions at t0 and t0+15; the horizon counts FROM t0
        rows = [
            json.loads(line)
            for line in (tmp_path / "base1" / "decisions.jsonl")
            .read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert [row["t_s"] for row in rows] == [240.0, 255.0]
        assert receipt["exit_status"] == "complete"
        assert receipt["start_mode"] == "base_state"
        assert receipt["base_t0_s"] == 240.0
        assert receipt["base_state_sha256"] == "a" * 64
        assert receipt["cav_roster"] == ["mainline.6", "mainline.7"]
        assert receipt["max_decision_units"] == 2
        assert receipt["walls"] == "off"
        assert receipt["sumo_launches"] is None  # no launch evidence wired here

    def test_fresh_receipt_start_mode(self, tmp_path):
        rig = Rig(tmp_path, run_id="fresh1")
        receipt = rig.run()
        assert receipt["start_mode"] == "fresh"
        assert receipt["base_run"] is None
        assert receipt["cav_roster"] == list(rig.cav)
        assert receipt["walls"] == "on"
