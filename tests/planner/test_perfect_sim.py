"""PL01-C: PerfectSimWorldModel skeleton against fake collaborators.

Zero simulation: no SUMO/traci/libsumo anywhere; the BranchRunner,
StateSnapshotter and ArrivalDraws are fakes, and a subprocess check pins the
no-SUMO-at-import-time rule for the module itself.
"""

import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from traffic_shepherd.planner import PlanCandidate, WallSpec, plan_key
from traffic_shepherd.planner.perfect_sim import (
    BranchRecord,
    BranchRunner,
    MetricsExtractor,
    PerfectSimWorldModel,
    SimpleArrivalDraws,
    StepRecord,
    VehicleStep,
    WallEventRecord,
    estimate_cost,
)
from tests.planner._helpers import RecordingProcessFactory

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

T0 = 60.0


class FakeSnapshotter:
    """Records save/restore; manifest cites SUMO version + network digest."""

    def __init__(self):
        self.saved = []
        self.restored = []

    def save(self, *, t0_s, controller_state, kept_wall_ids):
        self.saved.append({"t0_s": t0_s, "kept": list(kept_wall_ids)})
        return {
            "saved_state_dir": "<fake>/saved_state",
            "sumo_version": "1.27.1",
            "network_sha256": "f00d",
        }

    def restore(self, manifest, *, plan, draw_id, api):
        self.restored.append((getattr(plan, "label", "plan"), draw_id))


class FakeRunner(BranchRunner):
    """Scripted branches: straight traffic on one lane, per-plan knobs.

    PL-10d: every run() call records the fresh connection the model handed
    it (make_model injects a RecordingProcessFactory, so a predict_many
    call opens exactly one process per branch and closes each).

    ``wait_s`` (per plan key) adds entry-queue waiting each step so plan
    delays differ deterministically; ``collision_keys`` collide at t0+5 (in
    the safety window) and stop the branch; ``exit_after_step`` makes c0
    leave the section after that step (NaN trajectories from the exit on).
    """

    def __init__(
        self,
        *,
        process_factory=None,
        t0_s=T0,
        wait_s=None,
        collision_keys=(),
        collision_times_override=None,
        exit_after_step=None,
        wall_ids=("w0",),
        gap_m=8.0,
        leader_speed_mps=10.0,
    ):
        super().__init__(process_factory=process_factory)
        self.t0_s = t0_s
        self.wait_s = dict(wait_s or {})
        self.collision_keys = set(collision_keys)
        self.collision_times_override = collision_times_override
        self.exit_after_step = exit_after_step
        self.wall_ids = tuple(wall_ids)
        self.gap_m = gap_m
        self.leader_speed_mps = leader_speed_mps
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
        self.calls.append(
            {
                "plan": key,
                "connection": connection,
                "draw_id": draw_id,
                "departures": tuple(departures),
                "horizon_s": float(horizon_s),
                "stop_on_window_collision": bool(stop_on_window_collision),
                "commit_s": commit_s,
                "continuation": continuation,
            }
        )
        n_steps = int(horizon_s) + 1
        steps = []
        for i in range(n_steps):
            gone = self.exit_after_step is not None and i > self.exit_after_step
            vehicles = {
                "c0": VehicleStep(
                    s_m=20.0 * i,
                    lane_id=0.0,
                    speed_mps=20.0,
                    accel_mps2=0.0,
                    entry_wait_s=float(self.wait_s.get(key, 0.0)),
                    on_section=not gone,
                    leader_gap_m=self.gap_m,
                    leader_speed_mps=self.leader_speed_mps,
                )
            }
            steps.append(StepRecord(time_s=self.t0_s + i, dt_s=1.0, vehicles=vehicles))
        collides = key in self.collision_keys
        if self.collision_times_override is not None:
            collision_times = (float(self.collision_times_override),)
            # an override collision is deliberately outside the safety
            # window: the branch is not early-stopped for it
            stopped = False
        else:
            collision_times = (self.t0_s + 5.0,) if collides else ()
            stopped = collides
        walls = {}
        walls_spec = getattr(plan, "walls", ())
        for wall in walls_spec:
            if wall.wall_id in self.wall_ids:
                walls[wall.wall_id] = WallEventRecord(assembled_at_s=T0 + 10.0)
        return BranchRecord(
            plan_label=str(key),
            draw_id=draw_id,
            t0_s=self.t0_s,
            horizon_s=float(horizon_s),
            dt_s=1.0,
            free_flow_s={"c0": 30.0},
            steps=tuple(steps),
            collision_times=collision_times,
            wall_events=walls,
            tracked=("c0",),
            stopped_early=stopped,
            stopped_at_s=self.t0_s + 5.0 if stopped else None,
        )


def make_sample(k=2):
    return [
        SimpleNamespace(
            hybrid=None,
            arrivals={
                "sample_id": f"k{i}",
                "departures": ((60.0 + 10 * i, "main", 20.0, False, "passenger"),),
            },
            meta={"sample_id": f"k{i}", "time_s": T0, "merge_observed": True, "coverage": 1.0},
        )
        for i in range(k)
    ]


def make_model(runner, **over):
    if getattr(runner, "process_factory", None) is None:
        # PL-10d: every model needs a launcher; tests share one recorder
        runner.process_factory = RecordingProcessFactory()
    snapshotter = FakeSnapshotter()
    model = PerfectSimWorldModel(
        snapshotter, SimpleArrivalDraws(), runner, part_a_horizon_s=5.0, **over
    )
    return model, snapshotter


PLAN_A = PlanCandidate(per_cav={"c0": "damping"})
PLAN_B = PlanCandidate(per_cav={"c0": "pi-external-u"})
PLAN_C = PlanCandidate(per_cav={"c0": "native"})


class TestCommonRandomNumbers:
    def test_same_draw_id_and_departures_for_every_plan(self):
        runner = FakeRunner()
        model, _ = make_model(runner)
        model.predict_many(make_sample(2), [PLAN_A, PLAN_B], None, horizon_s=10.0)
        by_plan = {}
        for call in runner.calls:
            by_plan.setdefault(call["plan"], []).append(call)
        for plan_key_ in by_plan:
            draws = [c["draw_id"] for c in by_plan[plan_key_]]
            deps = [c["departures"] for c in by_plan[plan_key_]]
            assert draws == ["k0", "k1"]  # identical draw ids across plans
            assert deps[0] != () and deps[1] != ()  # explicit lists
        # the whole (draw id, departures) sequence is identical across
        # native and every plan: CRN by construction
        cross = {
            tuple((c["draw_id"], c["departures"]) for c in calls) for calls in by_plan.values()
        }
        assert len(cross) == 1

    def test_explicit_departures_are_passed_through(self):
        runner = FakeRunner()
        model, _ = make_model(runner)
        model.predict_many(make_sample(1), [PLAN_A], None, horizon_s=10.0)
        expected = ((60.0, "main", 20.0, False, "passenger"),)
        assert all(c["departures"] == expected for c in runner.calls)


class TestNativeSemanticsN1:
    def test_assemble_walls_kept_for_native(self):
        runner = FakeRunner()
        model, snapshotter = make_model(runner)
        state = {
            "c0": {
                "action": "wall",
                "wall_id": "w0",
                "wall_phase": "assemble",
                "locked": True,
                "time_in_phase_s": 1.0,
                "wall_speed_mps": 16.0,
            }
        }
        pred = model.predict_many(make_sample(1), [PLAN_A], state, horizon_s=10.0)[0]
        assert snapshotter.saved[0]["kept"] == ["w0"]
        assert pred.meta["native_kept_walls"] == ["w0"]

    def test_hold_and_release_walls_dissolved_for_native(self):
        runner = FakeRunner()
        model, snapshotter = make_model(runner)
        for phase in ("hold", "release"):
            state = {
                "c0": {
                    "action": "wall",
                    "wall_id": "w0",
                    "wall_phase": phase,
                    "locked": True,
                    "time_in_phase_s": 1.0,
                    "wall_speed_mps": 16.0,
                }
            }
            model.predict_many(make_sample(1), [PLAN_A], state, horizon_s=10.0)
            assert snapshotter.saved[-1]["kept"] == []
            pred = model.predict_many(make_sample(1), [PLAN_A], state, horizon_s=10.0)[0]
            assert pred.meta["native_kept_walls"] == []

    def test_no_controller_state_no_kept_walls(self):
        runner = FakeRunner()
        model, snapshotter = make_model(runner)
        model.predict_many(make_sample(1), [], None, horizon_s=10.0)
        assert snapshotter.saved[0]["kept"] == []
        assert snapshotter.saved[0]["t0_s"] == T0


class TestContractShapes:
    def test_arrays_and_delay_identity(self):
        runner = FakeRunner()
        model, _ = make_model(runner)
        out = model.predict_many(make_sample(2), [PLAN_A, PLAN_B], None, horizon_s=10.0)
        assert [p.meta["sample_ids"] for p in out] == [["k0", "k1"]] * 2
        assert out[0].meta["member_ids"] == ["m0"]
        for pred in out:
            for name in (
                "delay_vs_native_s",
                "plan_delay_s",
                "native_delay_s",
                "delay_part_a_s",
                "delay_part_b_s",
                "min_ttc_s",
                "ttc_lt3_exposure_s",
                "native_ttc_lt3_exposure_s",
                "hard_brake_s",
                "native_hard_brake_s",
            ):
                assert getattr(pred, name).shape == (1, 2), name
            assert pred.collision.shape == (1, 2)
            assert pred.collision.dtype == np.bool_
            assert pred.native_collision.dtype == np.bool_
            finite = ~np.isnan(pred.plan_delay_s[0])
            assert np.allclose(
                pred.plan_delay_s[0][finite] - pred.native_delay_s[0][finite],
                pred.delay_vs_native_s[0][finite],
            )
        assert out[0].native_delay_s is out[1].native_delay_s  # shared

    def test_trajectories_shape_nan_after_exit(self):
        runner = FakeRunner(exit_after_step=2)
        model, _ = make_model(runner)
        pred = model.predict_many(make_sample(1), [PLAN_A], None, horizon_s=10.0, trajectory_s=5.0)[
            0
        ]
        traj = pred.trajectories_tracked["c0"]
        assert traj.shape == (1, 1, 6, 4)  # T = trajectory_s + 1 incl. t0
        assert traj[0, 0, 0].tolist() == [0.0, 0.0, 0.0, 20.0]
        assert traj[0, 0, 2, 1] == pytest.approx(40.0)
        assert np.all(np.isnan(traj[0, 0, 3:]))  # NaN from the exit step on
        native = pred.native_trajectories_tracked["c0"]
        assert native.shape == (1, 1, 6, 4)

    def test_wall_events_block(self):
        wall = WallSpec(
            wall_id="w0", members=("c0",), speed_mps=16.0, phase="assemble", hold_s=60.0
        )
        plan = PlanCandidate(per_cav={}, walls=(wall,))
        runner = FakeRunner()
        model, _ = make_model(runner)
        pred = model.predict_many(make_sample(1), [plan], None, horizon_s=10.0)[0]
        block = pred.walls["w0"]
        assert block.assembled.dtype == np.bool_
        assert block.assembled.all() and block.safety_abort.dtype == np.bool_
        assert not block.safety_abort.any()
        assert block.abort_reason.dtype == np.int8
        assert int(block.abort_reason[0, 0]) == 0
        assert np.isfinite(block.assembled_at_s).all()
        assert block.assembled_at_s[0, 0] == pytest.approx(T0 + 10.0)

    def test_meta_contract_fields(self):
        runner = FakeRunner()
        model, _ = make_model(
            runner, model_id="perfect_sim/test", sumo_version="1.27.1", network_sha256="f00d"
        )
        pred = model.predict_many(make_sample(1), [PLAN_A], None, horizon_s=10.0)[0]
        meta = pred.meta
        assert meta["contract_version"] == "wd01_predict/1"
        assert meta["model_id"] == "perfect_sim/test"
        assert meta["weights_sha256"] == ["simulator:sumo-1.27.1+f00d"]
        assert meta["part_a_horizon_s"] == 5.0
        assert meta["safety_window_s"] == 5.0
        assert meta["horizon_s"] == 10.0
        assert meta["native_unlocked_available"] is False
        assert meta["device"] == "sumo-traci"
        assert meta["wall_clock_s"] >= 0.0
        assert meta["screened_out"] is False
        assert meta["evaluated_horizon_s"] == 10.0

    def test_safety_metrics_semantics(self):
        runner = FakeRunner(gap_m=8.0, leader_speed_mps=10.0)
        model, _ = make_model(runner)
        pred = model.predict_many(make_sample(1), [PLAN_A], None, horizon_s=10.0)[0]
        # closing = 20 - 10 = 10 m/s over an 8 m gap -> TTC 0.8 s < 3 s for
        # every one of the 6 window steps: exposure 6.0, min TTC 0.8
        assert pred.min_ttc_s[0, 0] == pytest.approx(0.8)
        assert pred.ttc_lt3_exposure_s[0, 0] == pytest.approx(6.0)
        assert not pred.collision[0, 0]
        assert pred.hard_brake_s[0, 0] == 0.0  # accel stays at 0

    def test_delay_arithmetic_split_parts(self):
        runner = FakeRunner(wait_s={plan_key(PLAN_A): 0.5})
        model, _ = make_model(runner)
        pred = model.predict_many(make_sample(1), [PLAN_A], None, horizon_s=10.0)[0]
        # native: 11 steps on section (10.0 transit), wait 0, free flow 30
        native_delay = 11.0 - 30.0
        # plan A adds 0.5 s waiting at each of the 11 steps
        plan_delay = native_delay + 11.0 * 0.5
        assert pred.native_delay_s[0, 0] == pytest.approx(native_delay)
        assert pred.plan_delay_s[0, 0] == pytest.approx(plan_delay)
        assert pred.delay_vs_native_s[0, 0] == pytest.approx(5.5)
        assert pred.delay_part_a_s[0, 0] == pytest.approx(
            6.0 - 30.0 + 6 * 0.5
        )  # steps t0..t0+5 = 6 steps in window
        assert pred.delay_part_a_s[0, 0] + pred.delay_part_b_s[0, 0] == pytest.approx(plan_delay)


class TestEarlyStop:
    def test_window_collision_nans_delay_and_flags_meta(self):
        runner = FakeRunner(collision_keys={plan_key(PLAN_A)})
        model, _ = make_model(runner)
        out = model.predict_many(make_sample(1), [PLAN_A, PLAN_B], None, horizon_s=10.0)
        unsafe, safe = out
        assert np.all(np.isnan(unsafe.plan_delay_s))
        assert np.all(np.isnan(unsafe.delay_vs_native_s))
        assert np.all(np.isnan(unsafe.delay_part_a_s))
        assert np.all(np.isnan(unsafe.delay_part_b_s))
        assert unsafe.meta["early_stop_cells"] == [0]
        assert unsafe.collision[0, 0]  # window collision recorded
        assert np.all(np.isfinite(safe.plan_delay_s))
        assert safe.meta["early_stop_cells"] == []
        # T-A #16: PLAN branches early-stop on an in-window collision; the
        # NATIVE reference branch never does (its run flag is False).
        assert all(
            c["stop_on_window_collision"] == (c["plan"] != "native") for c in runner.calls
        )

    def test_late_collision_outside_window_does_not_stop(self):
        runner = FakeRunner(collision_keys={plan_key(PLAN_A)}, collision_times_override=T0 + 8.0)
        model, _ = make_model(runner)
        pred = model.predict_many(make_sample(1), [PLAN_A], None, horizon_s=10.0)[0]
        assert pred.meta["early_stop_cells"] == []
        assert np.isfinite(pred.plan_delay_s[0, 0])
        assert not pred.collision[0, 0]  # window [t0, t0+5] excludes t0+8


class TestScreening:
    def test_selection_top_n_plus_always_screen_out_rest(self):
        waits = {plan_key(PLAN_A): 1.0, plan_key(PLAN_B): 0.0, plan_key(PLAN_C): 3.0}
        runner = FakeRunner(wait_s=waits)
        model, _ = make_model(runner, screen_horizon_s=8.0, rescore_top_n=1, rescore_always=(0,))
        out = model.predict_many(make_sample(1), [PLAN_A, PLAN_B, PLAN_C], None, horizon_s=10.0)
        # ranking at the screen horizon: B best (0 wait), A middle,
        # C worst -> rescore {B (top 1), A (always index 0)}; C screened out
        assert [p.meta["screened_out"] for p in out] == [False, False, True]
        assert [p.meta["evaluated_horizon_s"] for p in out] == [10.0, 10.0, 8.0]
        assert np.all(np.isfinite(out[0].plan_delay_s))
        assert np.all(np.isfinite(out[1].plan_delay_s))
        assert np.all(np.isnan(out[2].plan_delay_s))
        assert np.all(np.isnan(out[2].delay_vs_native_s))
        assert not np.all(np.isnan(out[2].min_ttc_s))  # safety still valid
        horizons = {c["plan"]: c["horizon_s"] for c in runner.calls}
        assert horizons[plan_key(PLAN_C)] == 8.0  # never run at full horizon
        full_b = [
            c for c in runner.calls if c["plan"] == plan_key(PLAN_B) and c["horizon_s"] == 10.0
        ]
        assert full_b  # B was rescored at the full horizon

    def test_screen_horizon_must_cover_safety_window(self):
        runner = FakeRunner()
        model, _ = make_model(runner, screen_horizon_s=3.0)
        with pytest.raises(ValueError, match="safety window"):
            model.predict_many(make_sample(1), [PLAN_A], None, horizon_s=10.0)

    def test_no_plans_no_screening(self):
        runner = FakeRunner()
        model, _ = make_model(runner, screen_horizon_s=8.0)
        model.predict_many(make_sample(1), [], None, horizon_s=10.0)
        assert {c["horizon_s"] for c in runner.calls} == {10.0}


class TestValidation:
    def test_empty_samples_rejected(self):
        runner = FakeRunner()
        model, _ = make_model(runner)
        with pytest.raises(ValueError, match="non-empty"):
            model.predict_many([], [PLAN_A], None, horizon_s=10.0)

    def test_mismatched_sample_times_rejected(self):
        runner = FakeRunner()
        model, _ = make_model(runner)
        samples = make_sample(2)
        samples[1].meta["time_s"] = 61.0
        with pytest.raises(ValueError, match="same time_s"):
            model.predict_many(samples, [PLAN_A], None, horizon_s=10.0)

    def test_trajectory_above_part_a_rejected(self):
        runner = FakeRunner()
        model, _ = make_model(runner)
        with pytest.raises(ValueError, match="part_a_horizon_s"):
            model.predict_many(make_sample(1), [PLAN_A], None, horizon_s=10.0, trajectory_s=6.0)

    def test_snapshot_taken_once_per_call(self):
        runner = FakeRunner()
        model, snapshotter = make_model(runner)
        model.predict_many(make_sample(2), [PLAN_A, PLAN_B], None, horizon_s=10.0)
        assert len(snapshotter.saved) == 1  # one t0 snapshot, six branches
        assert len(runner.calls) == 6  # (native + 2 plans) x K=2
        # PL-10d: exactly ONE fresh process per branch, each closed exactly
        # once, no connection reused across branches (native included)
        factory = runner.process_factory
        assert len(factory.started) == 6
        assert len({id(c) for c in factory.started}) == 6
        assert factory.closed == factory.started
        assert [c["connection"] for c in runner.calls] == factory.started


class TestNoSumoImport:
    def test_module_imports_without_sumo_stack(self):
        code = (
            "import sys; import traffic_shepherd.planner.perfect_sim"
            " as m; "
            "assert not [x for x in sys.modules if "
            "x.startswith(('traci', 'sumo', 'libsumo'))], "
            "sorted(sys.modules)[:3]; print('clean')"
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
            cwd=REPO_ROOT,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr
        assert "clean" in result.stdout


class TestEstimateCost:
    def test_full_horizon_arithmetic(self):
        # P = 2*2 + (2*2+2) + 1*2*2 + 1 = 15; B = 4*16 = 64; sim = 38400 s
        est = estimate_cost(
            N=2, K=4, W=1, n_speeds=2, n_releases=2, H=600.0, r=0.1, workers=2, group_sizes=(2,)
        )
        assert est.plans_p == 15
        assert est.branches == 64
        assert est.sim_seconds == pytest.approx(38400.0)
        assert est.cpu_hours == pytest.approx(38400.0 * 0.1 / 3600.0)
        assert est.wall_hours == pytest.approx(est.cpu_hours / 2.0)  # 0.5333 h

    def test_screened_arithmetic(self):
        # P = 20 + (8 + 10) + 8 + 1 = 47; screen 4*48 branches x 300 s,
        # rescore 4*(5+2) x 600 s
        est = estimate_cost(
            N=10,
            K=4,
            W=2,
            n_speeds=2,
            n_releases=2,
            H=600.0,
            r=0.05,
            workers=4,
            group_sizes=(3, 4),
            screen=(300.0, 600.0, 5),
        )
        assert est.plans_p == 47
        assert est.branches == 192 + 28
        assert est.sim_seconds == pytest.approx(192 * 300.0 + 28 * 600.0)
        assert est.cpu_hours == pytest.approx(74400.0 * 0.05 / 3600.0)
        assert est.wall_hours == pytest.approx(est.cpu_hours / 4.0)

    def test_design_doc_sizing_row_full_spec(self):
        # design table "full spec": N=10, K=4, 3 walls x 4 speeds x 4
        # releases, P ~= 80 with ~two groups; with groups (3, 4):
        est = estimate_cost(
            N=10, K=4, W=3, n_speeds=4, n_releases=4, H=600.0, r=0.05, workers=1, group_sizes=(3, 4)
        )
        assert est.plans_p == 87
        assert est.cpu_hours == pytest.approx(352 * 600.0 * 0.05 / 3600.0, rel=1e-9)

    def test_validation(self):
        with pytest.raises(ValueError):
            estimate_cost(N=-1, K=1, W=0, n_speeds=1, n_releases=1, H=600.0, r=0.1, workers=1)
        with pytest.raises(ValueError):
            estimate_cost(N=1, K=0, W=0, n_speeds=1, n_releases=1, H=600.0, r=0.1, workers=1)
        with pytest.raises(ValueError):
            estimate_cost(N=1, K=1, W=0, n_speeds=1, n_releases=1, H=600.0, r=0.1, workers=0)
        with pytest.raises(ValueError):
            estimate_cost(
                N=1, K=1, W=0, n_speeds=1, n_releases=1, H=600.0, r=0.1, workers=1, group_sizes=(0,)
            )
        with pytest.raises(ValueError):
            estimate_cost(
                N=1,
                K=1,
                W=0,
                n_speeds=1,
                n_releases=1,
                H=600.0,
                r=0.1,
                workers=1,
                screen=(600.0, 300.0, 1),
            )
        with pytest.raises(ValueError):
            estimate_cost(
                N=1,
                K=1,
                W=0,
                n_speeds=1,
                n_releases=1,
                H=600.0,
                r=0.1,
                workers=1,
                screen=(300.0, 600.0, 0),
            )


def test_extractor_skips_nonfinite_ttc_state():
    step = StepRecord(
        time_s=T0,
        dt_s=1.0,
        vehicles={
            "c0": VehicleStep(
                s_m=0.0,
                lane_id=0.0,
                speed_mps=float("nan"),
                accel_mps2=0.0,
                leader_gap_m=8.0,
                leader_speed_mps=10.0,
            )
        },
    )
    record = BranchRecord(
        plan_label="native",
        draw_id="k0",
        t0_s=T0,
        horizon_s=1.0,
        dt_s=1.0,
        free_flow_s={"c0": 30.0},
        steps=(step,),
        tracked=("c0",),
    )
    extracted = MetricsExtractor().extract(record, part_a_horizon_s=5.0, trajectory_s=1.0)
    assert extracted["min_ttc_s"] is None  # closing_ttc skips nonfinite


class TestEstimateCostNewKnobs:
    """PL01-C2: decision_points and early_stop_fraction accounting."""

    def test_decision_points_multiplies_branches_and_sim_seconds(self):
        base = estimate_cost(
            N=2, K=4, W=1, n_speeds=2, n_releases=2, H=600.0, r=0.1, workers=2, group_sizes=(2,)
        )
        swept = estimate_cost(
            N=2,
            K=4,
            W=1,
            n_speeds=2,
            n_releases=2,
            H=600.0,
            r=0.1,
            workers=2,
            group_sizes=(2,),
            decision_points=3,
        )
        assert swept.plans_p == base.plans_p
        assert swept.branches == 3 * base.branches == 192
        assert swept.sim_seconds == pytest.approx(3 * base.sim_seconds)
        assert swept.cpu_hours == pytest.approx(3 * base.cpu_hours)
        assert swept.wall_hours == pytest.approx(3 * base.wall_hours)

    def test_early_stop_fraction_truncates_unsafe_branch_share(self):
        # P = 1 (no N, W, groups; combined check), B = K*(1+P) = 2; stage at
        # 600 s with f = 0.5 stopping at the 60 s window:
        # 2 * (600*0.5 + 60*0.5) = 660 sim-seconds
        est = estimate_cost(
            N=0,
            K=1,
            W=0,
            n_speeds=1,
            n_releases=1,
            H=600.0,
            r=0.1,
            workers=1,
            early_stop_fraction=0.5,
            safety_window_s=60.0,
        )
        assert est.branches == 2
        assert est.sim_seconds == pytest.approx(660.0)
        assert est.cpu_hours == pytest.approx(660.0 * 0.1 / 3600.0)

    def test_early_stop_with_screening_applies_to_both_stages(self):
        plain = estimate_cost(
            N=10,
            K=4,
            W=2,
            n_speeds=2,
            n_releases=2,
            H=600.0,
            r=0.05,
            workers=6,
            group_sizes=(3, 4),
            screen=(300.0, 600.0, 5),
        )
        stopped = estimate_cost(
            N=10,
            K=4,
            W=2,
            n_speeds=2,
            n_releases=2,
            H=600.0,
            r=0.05,
            workers=6,
            group_sizes=(3, 4),
            screen=(300.0, 600.0, 5),
            early_stop_fraction=0.25,
            safety_window_s=60.0,
        )
        # screen stage: 192*(300*0.75 + 60*0.25) = 192*240 = 46080 s
        # rescore stage: 28*(600*0.75 + 60*0.25) = 28*465 = 13020 s
        assert stopped.branches == plain.branches
        assert stopped.sim_seconds == pytest.approx(46080.0 + 13020.0)

    def test_new_knob_validation(self):
        common = dict(N=1, K=1, W=0, n_speeds=1, n_releases=1, H=600.0, r=0.1, workers=1)
        with pytest.raises(ValueError, match="decision_points"):
            estimate_cost(**common, decision_points=0)
        with pytest.raises(ValueError, match="early_stop_fraction"):
            estimate_cost(**common, early_stop_fraction=1.0)
        with pytest.raises(ValueError, match="safety_window_s"):
            estimate_cost(**common, early_stop_fraction=0.1)


class TestEstimateCostProcessStart:
    """PL-10d: one fresh SUMO process per branch -> per-branch launch cost."""

    def test_process_start_overhead_scales_with_branch_count(self):
        kw = dict(
            N=2, K=4, W=1, n_speeds=2, n_releases=2, H=600.0, r=0.1, workers=2, group_sizes=(2,)
        )
        base = estimate_cost(**kw)  # default 0.0 = TBD, no overhead booked
        priced = estimate_cost(**kw, process_start_s=10.0)
        assert base.process_start_seconds == 0.0
        assert priced.process_start_seconds == pytest.approx(10.0 * base.branches)
        assert priced.cpu_hours == pytest.approx(
            base.cpu_hours + priced.process_start_seconds / 3600.0
        )
        assert priced.wall_hours == pytest.approx(priced.cpu_hours / 2.0)
        # screening and decision sweeps multiply BRANCHES, so the overhead
        # must scale with them, not with sim-seconds
        screened = estimate_cost(
            N=10,
            K=4,
            W=2,
            n_speeds=2,
            n_releases=2,
            H=600.0,
            r=0.05,
            workers=6,
            group_sizes=(3, 4),
            screen=(300.0, 600.0, 5),
            process_start_s=10.0,
        )
        plain_screened = estimate_cost(
            N=10,
            K=4,
            W=2,
            n_speeds=2,
            n_releases=2,
            H=600.0,
            r=0.05,
            workers=6,
            group_sizes=(3, 4),
            screen=(300.0, 600.0, 5),
        )
        assert screened.branches == 220
        assert screened.process_start_seconds == pytest.approx(10.0 * 220)
        assert screened.cpu_hours == pytest.approx(plain_screened.cpu_hours + 2200.0 / 3600.0)
        swept = estimate_cost(**kw, decision_points=3, process_start_s=10.0)
        assert swept.process_start_seconds == pytest.approx(3 * priced.process_start_seconds)

    def test_process_start_default_is_zero_tbd(self):
        est = estimate_cost(N=2, K=4, W=1, n_speeds=2, n_releases=2, H=600.0, r=0.1, workers=2)
        assert est.process_start_seconds == 0.0
        assert est.cpu_hours == pytest.approx(est.sim_seconds * 0.1 / 3600.0)

    def test_process_start_validation(self):
        with pytest.raises(ValueError, match="process_start_s"):
            estimate_cost(
                N=1,
                K=1,
                W=0,
                n_speeds=1,
                n_releases=1,
                H=600.0,
                r=0.1,
                workers=1,
                process_start_s=-1.0,
            )
