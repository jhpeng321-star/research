"""PL-02a tests: offline planner analysis on synthetic campaign fixtures.

The fixtures mimic the AMENDED campaign layout (BRIEF_PL02a.md): base runs
keep the frozen evaluate layout (physics/scenario under <run>/run/), restored
arms keep everything at <run>/, and WORLD_PLAN.json is the root->branches
index. Zero simulation; no real data is required.
"""

import dataclasses
import json
import shutil
import sys
import tempfile

import numpy as np
import pytest
from pathlib import Path

from traffic_shepherd.planner import PlanCandidate, PlannerConfig, safety_check
from traffic_shepherd.planner.recorded import (
    BranchOutcome,
    CampaignIndex,
    Layout,
    RecordedBranchWorldModel,
    branch_outcome,
    default_delay,
)

ROW_DT = 0.5
T0 = 60.0


def veh(road_id="e0", lane=0.0, pos=0.0, speed=20.0, accel=0.0):
    return {
        "road_id": road_id,
        "lane_id": lane,
        "lane_position_m": pos,
        "speed_mps": speed,
        "acceleration_mps2": accel,
    }


def steady_rows(start, end, ids, speed=20.0, cav_ids=()):
    """Constant-speed cruising rows; ``cav_ids`` are emitted in every row's
    ``cav_actions`` (native) so recorded constant plans can be derived."""
    rows = []
    t = start
    positions = {vid: 0.0 for vid in ids}
    cav = {vid: {"action": "native"} for vid in cav_ids}
    while t < end - 1e-9:
        vehicles = {vid: veh(pos=positions[vid], speed=speed) for vid in ids}
        rows.append(
            {
                "time_s": t,
                "vehicles": vehicles,
                "colliding_ids": [],
                "cav_actions": dict(cav),
                "boundary_crossings": [],
            }
        )
        for vid in positions:
            positions[vid] += speed * ROW_DT
        t += ROW_DT
    return rows


def write_run(
    runs_root,
    name,
    *,
    arm,
    is_base,
    rows,
    complete=True,
    arrivals=None,
    outcome=None,
    schedule=None,
    dispatcher=None,
    branch_record=None,
    rou_vehicles=None,
):
    layout = Layout()
    run_dir = runs_root / name
    physics_dir = run_dir / ("run" if is_base else ".")
    physics_dir.mkdir(parents=True, exist_ok=True)
    receipt = {
        "run": name,
        layout.receipt_status_key: (layout.receipt_complete_value if complete else "failed"),
    }
    (run_dir / layout.receipt_name).write_text(json.dumps(receipt))
    (physics_dir / layout.physics_name).write_text("".join(json.dumps(row) + "\n" for row in rows))
    if rou_vehicles is not None:
        scenario_dir = run_dir / "run" / "scenario"
        scenario_dir.mkdir(parents=True, exist_ok=True)
        body = "".join(
            f'  <vehicle id="{vid}" depart="{depart:g}"/>\n' for vid, depart in rou_vehicles
        )
        (scenario_dir / "requests.rou.xml").write_text("<routes>\n" + body + "</routes>")
    if arrivals is not None:
        (run_dir / layout.arrivals_name).write_text(
            "".join(
                json.dumps({"vehicle_id": vid, "arrival_time_s": t}) + "\n" for vid, t in arrivals
            )
        )
    if outcome is not None:
        (run_dir / layout.outcome_name).write_text(json.dumps(outcome))
    if schedule is not None:
        (run_dir / layout.switch_schedule_name).write_text(json.dumps(schedule))
    if dispatcher is not None:
        (run_dir / layout.dispatcher_rows_name).write_text(
            "".join(json.dumps(row) + "\n" for row in dispatcher)
        )
    if branch_record is not None:
        (run_dir / layout.branch_record_name).write_text(json.dumps(branch_record))
    return run_dir


def write_world_plan(path, worlds):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema": "wm44_chunk1_world_plan/1", "worlds": worlds}))


def world_entry(
    label, demand=2600, penetration=0.2, split="TRAIN", with_native=True, arms=("fixed",)
):
    base = {
        "arm": "base",
        "run": f"{label}__base",
        "controller_name": "damping",
        "saves_state": {"at_s": T0},
        "duration_s": 602.0,
    }
    branches = []
    if with_native:
        branches.append(
            {
                "arm": "native",
                "run": f"{label}__native",
                "suffix_controller": "native",
                "duration_s": 600.0,
                "mode": "restore_suffix",
                "restores_state": {"from_run": f"{label}__base", "at_s": T0},
                "native_sibling_run": f"{label}__native",
                "branch_seed": 111,
            }
        )
    for arm in arms:
        branches.append(
            {
                "arm": arm,
                "run": f"{label}__{arm}",
                "suffix_controller": "mixed-assignment",
                "duration_s": 600.0,
                "mode": "restore_suffix",
                "restores_state": {"from_run": f"{label}__base", "at_s": T0},
                "native_sibling_run": (f"{label}__native" if with_native else None),
                "branch_seed": 222,
                "assignment_mode": "fixed",
            }
        )
    return {
        "world": label,
        "seed": 1,
        "loading_veh_h": demand,
        "penetration": penetration,
        "split": split,
        "base_arm": base,
        "branches": branches,
    }


def _runner_module():
    """Guarded import: the PL-02a runner package lives in the planner
    worktree only; tests that need it skip with this precise reason."""
    return pytest.importorskip(
        "output.pl02a_offline_v1.run",
        reason="PL-02a runner package (output/pl02a_offline_v1) lives in the planner worktree only",
    )


def _sample():
    from types import SimpleNamespace

    return SimpleNamespace(
        hybrid=None,
        arrivals=(),
        meta={"sample_id": "k0", "time_s": T0, "merge_observed": True, "coverage": 1.0},
    )


@pytest.fixture()
def campaign(tmp_path):
    """w1 complete (base+native+fixed); w2 has no native sibling (partial)."""
    runs_root = tmp_path / "runs"
    window = 542.0
    write_run(
        runs_root,
        "w1__base",
        arm="base",
        is_base=True,
        rows=steady_rows(0.0, T0 + window, ("v1", "v2")),
        rou_vehicles=[("v_late", 60.0), ("v_never", 70.0)],
    )
    write_run(
        runs_root,
        "w1__native",
        arm="native",
        is_base=False,
        rows=steady_rows(T0, T0 + 600.0, ("v1", "v2")),
        arrivals=[("v_late", 160.0)],
        branch_record={"arm": "native", "root_id": "w1", "branch_point_t0_s": T0, "seed": 1},
    )
    schedule = {"events": [{"t_s": T0, "per_cav": {"c0": "damping", "c1": "pi-external-u"}}]}
    dispatcher = [
        {"time_s": T0 + t, "actions": {"c0": "damping", "c1": "pi-external-u"}}
        for t in (0.0, 5.0, 10.0)
    ]
    write_run(
        runs_root,
        "w1__fixed",
        arm="fixed",
        is_base=False,
        rows=steady_rows(T0, T0 + 600.0, ("v1", "v2")),
        schedule=schedule,
        dispatcher=dispatcher,
        branch_record={"arm": "fixed", "root_id": "w1", "branch_point_t0_s": T0, "seed": 1},
    )
    write_run(
        runs_root,
        "w2__base",
        arm="base",
        is_base=True,
        rows=steady_rows(0.0, T0 + window, ("v1", "v2")),
        rou_vehicles=[("v_late", 60.0)],
    )
    write_run(
        runs_root,
        "w2__fixed",
        arm="fixed",
        is_base=False,
        rows=steady_rows(T0, T0 + 600.0, ("v1", "v2")),
        schedule=schedule,
        dispatcher=dispatcher,
        branch_record={"arm": "fixed", "root_id": "w2", "branch_point_t0_s": T0, "seed": 2},
    )
    plan_path = tmp_path / "world_plan" / "WORLD_PLAN.json"
    write_world_plan(
        plan_path, [world_entry("w1"), world_entry("w2", with_native=False, split="HOLDOUT")]
    )
    return runs_root, plan_path, tmp_path / "out"


def index_campaign(campaign, **layout_over):
    runs_root, plan_path, _ = campaign
    layout = dataclasses.replace(Layout(), **layout_over) if layout_over else Layout()
    return CampaignIndex(runs_root, plan_path, layout).build()


class TestCampaignIndex:
    def test_groups_roots_and_counts_partial_data(self, campaign):
        index = index_campaign(campaign)
        assert set(index.roots) == {"w1__base"}
        root = index.roots["w1__base"]
        assert root.base.arm == "base"
        assert set(root.branches) == {"native", "fixed"}
        assert root.t0_s == T0
        assert root.meta["loading_veh_h"] == 2600
        # the root without its native sibling is skipped, with the reason
        assert any("w2__fixed" in reason and "native" in reason for reason in index.skipped)

    def test_incomplete_receipt_excluded(self, campaign):
        runs_root = campaign[0]
        layout = Layout()
        (runs_root / "w1__pi").mkdir()
        (runs_root / "w1__pi" / layout.receipt_name).write_text(
            json.dumps({"run": "w1__pi", layout.receipt_status_key: "failed"})
        )
        index = index_campaign(campaign)
        assert "pi" not in index.roots["w1__base"].branches

    def test_run_prefixes_scopes_the_scan(self, campaign):
        """PL-02a-S4: run_prefixes limits the scan to matching run dirs;
        out-of-scope dirs are neither indexed nor counted as skipped (a
        live stage may be writing them)."""
        runs_root, plan_path, _ = campaign
        layout = Layout(require_native_sibling=False)
        scoped = CampaignIndex(runs_root, plan_path, layout, run_prefixes=("w1__base",)).build()
        assert set(scoped.roots) == {"w1__base"}
        assert set(scoped.roots["w1__base"].branches) == set()
        assert scoped.skipped == []
        full = CampaignIndex(runs_root, plan_path, layout).build()
        assert set(full.roots["w1__base"].branches) == {"native", "fixed"}


class TestDelayFormula:
    def test_delay_formula_hand_computed(self, campaign):
        index = index_campaign(campaign)
        root = index.roots["w1__base"]
        from traffic_shepherd.planner.recorded import default_delay as builtin

        model = RecordedBranchWorldModel(
            root,
            layout=Layout(),
            horizon_s=100.0,
            delay_provider=builtin,
            delay_provider_name="builtin",
        )
        # v1 and v2 cruise the whole 100 s window at 20 m/s: presence
        # 200 s total; ADVANCE-BASED free-flow = 3980 m travelled / 30 m/s
        # = 132.667 s; PLUS the §8.14 entry waits of the two scheduled
        # vehicles never inserted in this arm (100 + 90 = 190 s).
        arm = model.outcomes["native"]
        assert arm.delay_s == pytest.approx(200.0 - 132.6666667 + 190.0, abs=1e-4)
        assert arm.delay_provider == "builtin"

    def test_base_arm_windowed_from_t0(self, campaign):
        index = index_campaign(campaign)
        root = index.roots["w1__base"]
        model = RecordedBranchWorldModel(root, layout=Layout(), horizon_s=100.0)
        # base rows start at t=0 but only [t0, t0+100) counts
        assert model.outcomes["base"].delay_s == pytest.approx(
            model.outcomes["native"].delay_s, abs=1e-6
        )

    def test_common_window_is_min_coverage(self, campaign):
        index = index_campaign(campaign)
        root = index.roots["w1__base"]
        model = RecordedBranchWorldModel(root, layout=Layout())
        # base covers 542 s post-t0; restored arms 600 -> common = 542
        assert model.coverage_s == pytest.approx(542.0)
        assert model.window_s == pytest.approx(542.0)
        assert model.outcomes["native"].window_s == pytest.approx(542.0)


class TestEntryWait:
    def test_late_insert_and_never_inserted(self, campaign):
        index = index_campaign(campaign)
        root = index.roots["w1__base"]
        model = RecordedBranchWorldModel(root, layout=Layout(), horizon_s=100.0)
        outcome = model.outcomes["native"]
        # neither scheduled vehicle is inserted in the recorded rows:
        # v_late (depart t0) waits the full window (100 s) and v_never
        # (depart t0+10) waits 90 s -> 190 s total (declared policy).
        assert outcome.entry_wait_s == pytest.approx(190.0)

    def test_default_delay_entry_wait_arithmetic(self):
        rows = steady_rows(60.0, 70.0, ("late",))
        rows_late = [row for row in rows if row["time_s"] >= 62.0]
        departs = {"late": 61.0, "ghost": 65.0}
        delay, waits, n, diag = default_delay(rows_late, 60.0, 10.0, Layout(), departs)
        assert waits == pytest.approx((62.0 - 61.0) + (70.0 - 65.0))
        assert n == 1
        # delay folds waits: presence 8 - advance/30 (150/30 = 5) + waits 6
        assert delay == pytest.approx(9.0)


class TestTtcAndPair:
    def test_closing_pair_ttc(self, tmp_path):
        runs_root = tmp_path / "runs"

        def present(t):
            return {"lead": veh(pos=100.0, speed=10.0), "fol": veh(pos=95.0, speed=20.0)}

        rows = [
            {
                "time_s": t,
                "vehicles": present(t),
                "colliding_ids": [],
                "cav_actions": {},
                "boundary_crossings": [],
            }
            for t in np.arange(T0, T0 + 20.0, ROW_DT)
        ]
        write_run(runs_root, "r__base", arm="base", is_base=True, rows=rows)
        write_run(runs_root, "r__native", arm="native", is_base=False, rows=rows)
        index = CampaignIndex(runs_root).build()
        model = RecordedBranchWorldModel(index.roots["r__base"], layout=Layout())
        outcome = model.outcomes["native"]
        # gap 5 m, closing 10 m/s -> TTC 0.5 s (< 3 s exposure accrues)
        assert outcome.min_ttc_s == pytest.approx(0.5)
        assert outcome.has_closing_pair is True
        assert outcome.ttc_lt3_exposure_s > 0.0

    def test_no_pair_min_ttc_none_cell_valid(self):
        outcome = BranchOutcome(
            arm="native",
            window_s=10.0,
            coverage_s=10.0,
            t0_s=T0,
            delay_s=0.0,
            entry_wait_s=0.0,
            n_vehicles=1,
            collision=False,
            min_ttc_s=None,
            has_closing_pair=False,
            ttc_lt3_exposure_s=0.0,
            hard_brake_s=0.0,
            n_teleports=0,
        )
        assert outcome.min_ttc_s is None
        assert not outcome.has_closing_pair
        assert outcome.cell_valid


class TestFailClosedRecorded:
    def test_collision_invalidates_and_vetoes(self, campaign):
        runs_root, plan_path, _ = campaign
        rows = steady_rows(T0, T0 + 100.0, ("v1",))
        rows[10]["colliding_ids"] = ["v9"]
        write_run(
            runs_root,
            "w1__crash",
            arm="fixed",
            is_base=False,
            rows=rows,
            schedule={"events": [{"t_s": T0, "per_cav": {"c0": "damping"}}]},
            dispatcher=[{"time_s": T0, "actions": {"c0": "damping"}}],
            branch_record={"arm": "fixed", "root_id": "w1"},
            outcome={"collisions": ["v9"]},
        )
        index = index_campaign(campaign)
        model = RecordedBranchWorldModel(index.roots["w1__base"], layout=Layout())
        assert not model.outcomes["crash"].cell_valid
        pred = model.predict_many([_sample()], [PlanCandidate(per_cav={"c0": "damping"})])[0]
        assert not pred.cell_valid[0, 0]
        assert not safety_check(pred, PlannerConfig()).ok  # fails closed

    def test_unknown_plan_fails_closed(self, campaign):
        index = index_campaign(campaign)
        model = RecordedBranchWorldModel(index.roots["w1__base"], layout=Layout())
        pred = model.predict_many([_sample()], [PlanCandidate(per_cav={"ghost": "pi-external-u"})])[
            0
        ]
        assert not pred.cell_valid[0, 0]
        assert np.isnan(pred.delay_vs_native_s[0, 0])
        assert not safety_check(pred, PlannerConfig()).ok

    def test_crosscheck_mismatch_invalidates(self, campaign):
        runs_root, plan_path, _ = campaign
        rows = steady_rows(T0, T0 + 100.0, ("v1",))
        write_run(
            runs_root,
            "w1__fixed2",
            arm="fixed",
            is_base=False,
            rows=rows,
            schedule={"events": [{"t_s": T0, "per_cav": {"c0": "native"}}]},
            dispatcher=[{"time_s": T0, "actions": {"c0": "native"}}],
            branch_record={"arm": "fixed", "root_id": "w1"},
            outcome={"time_on_section_s": 99999.0},
        )
        index = index_campaign(campaign)
        model = RecordedBranchWorldModel(index.roots["w1__base"], layout=Layout())
        reasons = model.outcomes["fixed2"].invalid_reasons
        assert any("cross-check" in reason for reason in reasons)
        assert not model.outcomes["fixed2"].cell_valid

    def test_realised_action_mismatch_invalidates(self, campaign):
        runs_root, plan_path, _ = campaign
        rows = steady_rows(T0, T0 + 100.0, ("v1",))
        write_run(
            runs_root,
            "w1__fixed3",
            arm="fixed",
            is_base=False,
            rows=rows,
            schedule={"events": [{"t_s": T0, "per_cav": {"c0": "damping"}}]},
            dispatcher=[{"time_s": T0, "actions": {"c0": "native"}}],
            branch_record={"arm": "fixed", "root_id": "w1"},
        )
        index = index_campaign(campaign)
        model = RecordedBranchWorldModel(index.roots["w1__base"], layout=Layout())
        reasons = model.outcomes["fixed3"].invalid_reasons
        assert any("realised" in reason for reason in reasons), reasons

    def test_teleport_detector_invalidates(self, tmp_path):
        runs_root = tmp_path / "runs"

        def present(t):
            jump = 500.0 if t >= 70.0 else 0.0
            return {"v1": veh(pos=20.0 * (t - T0) + jump, speed=20.0)}

        rows = [
            {
                "time_s": t,
                "vehicles": present(t),
                "colliding_ids": [],
                "cav_actions": {},
                "boundary_crossings": [],
            }
            for t in np.arange(T0, T0 + 50.0, ROW_DT)
        ]
        write_run(runs_root, "r__base", arm="base", is_base=True, rows=rows)
        write_run(runs_root, "r__native", arm="native", is_base=False, rows=rows)
        index = CampaignIndex(runs_root).build()
        model = RecordedBranchWorldModel(index.roots["r__base"], layout=Layout())
        assert model.outcomes["native"].n_teleports == 1
        assert not model.outcomes["native"].cell_valid

    def test_teleport_fail_closed_all_policy(self, tmp_path):
        runs_root = tmp_path / "runs"
        rows = steady_rows(T0, T0 + 50.0, ("v1",))
        write_run(runs_root, "r__base", arm="base", is_base=True, rows=rows)
        write_run(runs_root, "r__native", arm="native", is_base=False, rows=rows)
        index = CampaignIndex(runs_root).build()
        model = RecordedBranchWorldModel(
            index.roots["r__base"],
            layout=dataclasses.replace(Layout(), teleport_policy="fail_closed_all"),
        )
        assert not model.outcomes["native"].cell_valid


class TestOfflinePlannerDecisions:
    def test_planner_answers_recorded_constant_plans(self, campaign):
        index = index_campaign(campaign)
        model = RecordedBranchWorldModel(index.roots["w1__base"], layout=Layout())
        pred = model.predict_many(
            [_sample()], [PlanCandidate(per_cav={"c0": "damping", "c1": "pi-external-u"})]
        )[0]
        assert pred.cell_valid[0, 0]
        assert pred.meta["recorded_arm"] == "fixed"
        assert pred.meta["contract_version"] == "wd01_predict/1.1"
        assert pred.delay_vs_native_s[0, 0] == pytest.approx(0.0)
        assert pred.native_delay_s[0, 0] == pytest.approx(model.outcomes["native"].delay_s)

    def test_unknown_or_switching_plan_vetoed(self, campaign):
        index = index_campaign(campaign)
        model = RecordedBranchWorldModel(index.roots["w1__base"], layout=Layout())
        other = PlanCandidate(per_cav={"c0": "pi-external-u", "c1": "damping"})
        vetoed = model.predict_many([_sample()], [other])[0]
        assert not vetoed.cell_valid[0, 0]
        assert not safety_check(vetoed, PlannerConfig()).ok


class TestP4DelayProvider:
    """PL-02a2: the WD-01 p4 delay library is the default provider and
    agrees with the built-in cross-check on fixtures."""

    def test_default_provider_is_p4(self, campaign):
        index = index_campaign(campaign)
        root = index.roots["w1__base"]
        model = RecordedBranchWorldModel(root, layout=Layout(), horizon_s=100.0)
        assert model.delay_provider_name == "p4"
        for outcome in model.outcomes.values():
            assert outcome.delay_provider == "p4"

    def test_p4_absent_falls_back_with_precise_reason(self, campaign, monkeypatch):
        from traffic_shepherd.planner import recorded as recorded_module

        # a NON-EXISTENT off-repo root (never D:): the configured absence
        # must be exercised, not a stale kit tree
        absent_p4_root = str(Path(tempfile.gettempdir()) / "nonexistent-p4-path")
        monkeypatch.setattr(recorded_module, "P4_SEARCH_PATHS", (absent_p4_root,))
        # purge cached part4 modules AND quarantine every sys.path entry
        # that could still resolve them (earlier kit runs may have inserted
        # the WD-01 trees); then the configured absence is really exercised
        for key in [k for k in sys.modules if k.startswith("part4")]:
            monkeypatch.delitem(sys.modules, key, raising=False)
        isolated = [p for p in sys.path if "wd01" not in p.lower()]
        monkeypatch.setattr(sys, "path", isolated, raising=False)
        with pytest.raises(recorded_module.P4DelayUnavailable, match="nonexistent-p4-path"):
            recorded_module.load_p4_delay(recorded_module.P4_SEARCH_PATHS)
        # model-level default falls back to the built-in with a recorded note
        index = index_campaign(campaign)
        runs_root, plan_path, _ = campaign
        model = RecordedBranchWorldModel(
            index.roots["w1__base"],
            layout=Layout(),
            horizon_s=100.0,
            p4_search_paths=(absent_p4_root,),
        )
        assert model.delay_provider_name == "builtin"
        assert "nonexistent-p4-path" in model.delay_provider_fallback_reason

    def test_p4_and_builtin_agree_within_declared_tolerance(self, campaign):
        from traffic_shepherd.planner.recorded import (
            default_delay,
            p4_delay_provider,
        )

        # align the built-in constant limit with the WM-30 lane limit so
        # the free-flow references match (31.29 m/s mainline)
        layout = dataclasses.replace(Layout(), free_flow_mps=31.29)
        index = index_campaign(campaign)
        root = index.roots["w1__base"]
        p4_model = RecordedBranchWorldModel(
            root, layout=layout, horizon_s=100.0, delay_provider=p4_delay_provider()
        )
        builtin_model = RecordedBranchWorldModel(
            root,
            layout=layout,
            horizon_s=100.0,
            delay_provider=default_delay,
            delay_provider_name="builtin",
        )
        for arm in ("native", "fixed"):
            p4_out = p4_model.outcomes[arm]
            in_out = builtin_model.outcomes[arm]
            # §8.14 waits agree exactly (same clip arithmetic)
            assert p4_out.entry_wait_s == pytest.approx(in_out.entry_wait_s, abs=0.51)
            # delay agrees within one physics step per vehicle + 1 s
            tolerance = 2 * 0.5 + 1.0
            assert abs(p4_out.delay_s - in_out.delay_s) <= tolerance


class TestRunnerEndToEnd:
    def test_runner_writes_outputs_and_is_idempotent(self, campaign):
        runner = _runner_module()

        runs_root, plan_path, out_dir = campaign
        cache_dir = out_dir / "cache"
        first = runner.run(runs_root, plan_path, out_dir, cache_dir=cache_dir)
        results_one = (out_dir / "RESULTS.json").read_bytes()
        report_one = (out_dir / "REPORT.md").read_text(encoding="utf-8")
        runner.run(runs_root, plan_path, out_dir, cache_dir=cache_dir)
        # PL-02a-C1: the deterministic payload stays byte-identical across
        # invocations; the per-invocation timing/cache provenance blocks are
        # excluded (first run: cache misses; second: all hits).
        payload_one = json.loads(results_one)
        payload_two = json.loads((out_dir / "RESULTS.json").read_bytes())
        assert payload_one["timing"]["cache_misses"] > 0
        assert payload_two["timing"]["cache_hits"] > 0
        payload_one.pop("timing", None)
        payload_one.pop("cache", None)
        payload_two.pop("timing", None)
        payload_two.pop("cache", None)
        assert payload_one == payload_two
        marker = "\n## TIMING & CACHE"
        deterministic_one = report_one.split(marker)[0]
        deterministic_two = (out_dir / "REPORT.md").read_text(encoding="utf-8").split(marker)[0]
        assert deterministic_one == deterministic_two
        # one complete root used; the partial root counted, not used
        assert len(first["per_root"]) == 1
        assert first["index"].skipped
        assert (out_dir / "roots.csv").is_file()
        results = json.loads(results_one)
        assert results["n_roots_used"] == 1
        assert results["n_roots_total"] == 2
        # provenance: every arm's delay number records its provider (p4)
        arms = results["per_root"][0]["arms"]
        assert {a["delay_provider"] for a in arms.values()} == {"p4"}

    def test_report_states_declarations_and_limits(self, campaign):
        runner = _runner_module()

        runs_root, plan_path, out_dir = campaign
        runner.run(runs_root, plan_path, out_dir, cache_dir=out_dir / "cache")
        report = (out_dir / "REPORT.md").read_text(encoding="utf-8")
        assert "DECLARED DEFINITIONS" in report
        assert "LIMITS" in report
        assert "onset" in report.lower()
        assert "teleport" in report.lower()
        assert "perfect foresight" in report
        assert "K = 1" in report

    def test_recorded_payload_has_no_wall_clock_nondeterminism(self, campaign, monkeypatch):
        """Root cause of the intermittent idempotence failure.

        ``RecordedBranchWorldModel.predict_many`` used to stamp each prediction
        with ``wall_clock_s = clock() - clock()`` measured from
        ``time.monotonic``. That value is serialized (via ``str(Prediction)``)
        into ``RESULTS.json``'s ``per_root[*]["predictions"]``. On Windows
        ``time.monotonic`` has ~15.6 ms resolution and the recorded predict
        loop is far faster, so the two reads normally agree and the field is
        ``0.0`` -- but under CPU load the reads can straddle a tick, giving the
        FIRST and SECOND runner invocations different ``wall_clock_s`` values,
        so ``payload_one == payload_two`` fails intermittently.

        This forces the straddle with a drifting clock: the old code wrote a
        run-dependent ``wall_clock_s`` into the deterministic payload (payloads
        differ); the fix keeps the recorded replay's ``wall_clock_s`` at the
        deterministic ``0.0`` (payloads identical).
        """
        runner = _runner_module()
        runs_root, plan_path, out_dir = campaign
        cache_dir = out_dir / "cache"

        class _DriftingClock:
            """Strictly increasing clock with a growing step: every consecutive
            read pair straddles a different interval, exactly the load-induced
            condition, but deterministic."""

            def __init__(self):
                self._t = 0.0
                self._step = 0.0

            def __call__(self):
                self._step += 1.0
                self._t += self._step
                return self._t

        # Patch the clock source the recorded model used to consult by
        # default; the fixed model never reads it unless a caller injects a
        # clock explicitly (the runner never does), so the recorded payload
        # stays deterministic.
        monkeypatch.setattr("time.monotonic", _DriftingClock())

        runner.run(runs_root, plan_path, out_dir, cache_dir=cache_dir)
        payload_one = json.loads((out_dir / "RESULTS.json").read_bytes())
        runner.run(runs_root, plan_path, out_dir, cache_dir=cache_dir)
        payload_two = json.loads((out_dir / "RESULTS.json").read_bytes())
        for payload in (payload_one, payload_two):
            payload.pop("timing", None)
            payload.pop("cache", None)
        assert payload_one == payload_two


class TestOfflineDecisionRules:
    def test_switch_picks_best_safe_and_regret_arithmetic(self, tmp_path):
        runner = _runner_module()

        runs_root = tmp_path / "runs"
        # native: two cruisers present the whole 600 s (delay 1133.3);
        # fixed1: both exit at t0+300 (delay 413.3, best safe);
        # fixed2: full presence like native (no saving; margin-blockable)
        tr_write = write_run
        tr_write(
            runs_root,
            "w3__base",
            arm="base",
            is_base=True,
            rows=steady_rows(0.0, T0 + 600.0, ("v1", "v2")),
        )
        tr_write(
            runs_root,
            "w3__native",
            arm="native",
            is_base=False,
            rows=steady_rows(T0, T0 + 600.0, ("v1", "v2"), cav_ids=("v1",)),
        )
        tr_write(
            runs_root,
            "w3__fixed1",
            arm="fixed",
            is_base=False,
            rows=steady_rows(T0, T0 + 300.0, ("v1", "v2")),
            schedule={"events": [{"t_s": T0, "per_cav": {"c0": "damping"}}]},
            dispatcher=[{"time_s": T0, "actions": {"c0": "damping"}}],
            branch_record={"arm": "fixed", "root_id": "w3"},
        )
        tr_write(
            runs_root,
            "w3__fixed2",
            arm="fixed",
            is_base=False,
            rows=steady_rows(T0, T0 + 600.0, ("v1", "v2")),
            schedule={"events": [{"t_s": T0, "per_cav": {"c0": "native"}}]},
            dispatcher=[{"time_s": T0, "actions": {"c0": "native"}}],
            branch_record={"arm": "fixed", "root_id": "w3"},
        )
        index = CampaignIndex(runs_root).build()
        root = index.roots["w3__base"]
        record = runner.analyse_root(root, Layout(), 600.0, PlannerConfig(margin_floor_s=30.0))
        assert record["best_safe_arm"] == "fixed1"
        decision = record["decisions"]["native@margin30"]
        assert decision["chosen"] == "fixed1"
        assert decision["regret_vs_best_safe_s"] == pytest.approx(0.0)
        # the margin-blocked alternative (fixed2, saving 10 < 30) is not chosen
        assert decision["chosen"] != "fixed2"

    def test_margin_blocks_marginal_best_and_regret_counts(self, tmp_path, monkeypatch):
        runner = _runner_module()

        monkeypatch.setattr(runner, "MARGIN_SWEEP_S", (0.0, 200.0))

        runs_root = tmp_path / "runs"
        tr_write = write_run
        tr_write(
            runs_root,
            "w4__base",
            arm="base",
            is_base=True,
            rows=steady_rows(0.0, T0 + 600.0, ("v1", "v2")),
        )
        tr_write(
            runs_root,
            "w4__native",
            arm="native",
            is_base=False,
            rows=steady_rows(T0, T0 + 600.0, ("v1", "v2"), cav_ids=("v1",)),
        )
        # fixed1: both cruisers leave at t0+590 -> presence 590 s each vs
        # native 600: total saving 2*10 = 20 s, blocked by margin 200
        trimmed = [
            row for row in steady_rows(T0, T0 + 600.0, ("v1", "v2")) if row["time_s"] < T0 + 590.0
        ]
        tr_write(
            runs_root,
            "w4__fixed1",
            arm="fixed",
            is_base=False,
            rows=trimmed,
            schedule={"events": [{"t_s": T0, "per_cav": {"c0": "native"}}]},
            dispatcher=[{"time_s": T0, "actions": {"c0": "native"}}],
            branch_record={"arm": "fixed", "root_id": "w4"},
        )
        index = CampaignIndex(runs_root).build()
        root = index.roots["w4__base"]
        record = runner.analyse_root(root, Layout(), 600.0, PlannerConfig(margin_floor_s=200.0))
        decision = record["decisions"]["native@margin200"]
        # median saving 20 < margin 200: the rule keeps native; the regret
        # equals the forgone 20 s (best safe saving, 2 veh x 10 s)
        assert decision["chosen"] == "native"
        assert decision["regret_vs_best_safe_s"] == pytest.approx(
            record["arms"]["native"]["delay_s"] - record["arms"][record["best_safe_arm"]]["delay_s"]
        )
        assert decision["regret_vs_best_safe_s"] > 0.0
        switch_decision = record["decisions"]["native@margin0"]
        assert switch_decision["chosen"] == "fixed1"  # margin 0 switches
        best_safe = record["arms"][record["best_safe_arm"]]
        assert record["best_safe_arm"] == "fixed1"
        # the p4 advance-based accounting gives the exact saving; assert
        # self-consistency against the recorded arm delays
        assert record["native_delay_s"] - best_safe["delay_s"] == pytest.approx(
            decision["regret_vs_best_safe_s"]
        )


class TestEdgeCrossingAdvance:
    """PL-02a3: lane_position resets at edge boundaries must not count as
    travel; the built-in provider bridges an edge change from declared edge
    lengths, else skips and counts the step in the diagnostics."""

    def _rows(self, road_b_pos):
        """Rows 60..69.5: v cruises on e0 at 990 m until t<61, then the
        edge changes to e1 at road_b_pos."""
        rows = []
        t = T0
        while t < 70.0 - 1e-9:
            if t < 61.0 - 1e-9:
                vehicles = {"v": veh(road_id="e0", pos=990.0, speed=20.0)}
            else:
                vehicles = {"v": veh(road_id="e1", pos=road_b_pos, speed=20.0)}
            rows.append(
                {
                    "time_s": t,
                    "vehicles": vehicles,
                    "colliding_ids": [],
                    "cav_actions": {},
                    "boundary_crossings": [],
                }
            )
            t += ROW_DT
        return rows

    def _run(self, road_lengths):
        layout = dataclasses.replace(Layout(), road_length_by_id=road_lengths)
        rows = self._rows(0.0)
        delay, waits, n, diag = default_delay(rows, T0, 10.0, layout, {})
        return delay, waits, n, diag, layout

    def test_edge_change_bridged_from_edge_lengths(self):
        delay, waits, n, diag, _ = self._run(road_lengths={"e0": 1000.0, "e1": 1000.0})
        # presence 10.0 s (20 rows x 0.5); the boundary step bridges
        # (1000-990)+0 = 10 m of advance; free-flow 10/30 = 0.333 s
        assert diag["edge_change_steps_bridged"] >= 1
        assert diag["edge_change_steps_skipped"] == 0
        assert delay == pytest.approx(10.0 - 10.0 / 30.0)

    def test_edge_change_without_lengths_skips_and_counts(self):
        delay, waits, n, diag, _ = self._run(road_lengths=None)
        assert diag["edge_change_steps_skipped"] >= 1
        assert diag["edge_change_steps_bridged"] == 0
        # the boundary jump contributes no advance: delay is presence-only
        assert delay == pytest.approx(10.0)

    def test_same_road_lane_change_still_counts_advance(self):
        layout = dataclasses.replace(Layout(), road_length_by_id=None)
        rows = [
            {
                "time_s": 60.0,
                "vehicles": {"v": veh(lane=0.0, pos=100.0)},
                "colliding_ids": [],
                "cav_actions": {},
                "boundary_crossings": [],
            },
            {
                "time_s": 60.5,
                "vehicles": {"v": veh(lane=1.0, pos=110.0)},
                "colliding_ids": [],
                "cav_actions": {},
                "boundary_crossings": [],
            },
        ]
        delay, waits, n, diag = default_delay(rows, T0, 10.0, layout, {})
        assert diag["edge_change_steps_skipped"] == 0
        assert waits == 0.0


class TestPilotR3ExcerptRegression:
    """PL-02a-S: the REAL pilot-r3 rows (small excerpt) must parse and the
    base/verify same-action continuation must produce identical outcomes."""

    FIXTURE = Path(__file__).parent / "fixtures" / "pilot_r3_excerpt"

    def test_excerpt_base_equals_verify(self):
        layout = Layout(
            restored_file_subdir="run",
            receipt_status_ok_values=("complete", "failed"),
            require_native_sibling=False,
            free_flow_mps=31.29,
        )
        index = CampaignIndex(self.FIXTURE, None, layout).build()
        assert set(index.roots) == {"f8401m2700__base"}
        root = index.roots["f8401m2700__base"]
        assert set(root.branches) == {"verify"}
        base = branch_outcome(
            root.base,
            t0_s=60.0,
            window_s=3.0,
            layout=layout,
            delay_provider=default_delay,
            provider_name="builtin",
            departs={"mainline.8": 0.4, "mainline.70": 30.0},
            window_start_s=60.5,
            exit_ids=set(),
        )
        verify = branch_outcome(
            root.branches["verify"],
            t0_s=60.0,
            window_s=3.0,
            layout=layout,
            delay_provider=default_delay,
            provider_name="verify",
            departs={"mainline.8": 0.4, "mainline.70": 30.0},
            window_start_s=60.5,
            exit_ids=set(),
        )
        assert base.delay_s == pytest.approx(verify.delay_s, abs=1e-9)
        assert base.entry_wait_s == verify.entry_wait_s
        assert base.min_ttc_s == verify.min_ttc_s
        assert base.has_closing_pair == verify.has_closing_pair
        assert base.collision == verify.collision


FIXTURE = Path(__file__).parent / "fixtures" / "pilot_r3_excerpt"

R3_LAYOUT = Layout(
    restored_file_subdir="run",
    require_native_sibling=False,
    free_flow_mps=31.29,
    section_len_m=3199.0,
    accept_failed_receipt_if_exact=True,
)


def vanish_rows(vanish_after_s):
    """Rows for one vehicle 'x' that vanishes mid-section (no exit)."""
    rows = []
    for t in np.arange(T0, T0 + 5.0, ROW_DT):
        vehicles = (
            {"x": veh(road_id="e0", lane=0, pos=20.0 * (t - T0), speed=20.0)}
            if t <= vanish_after_s + 1e-9
            else {}
        )
        rows.append(
            {
                "time_s": float(t),
                "vehicles": vehicles,
                "colliding_ids": [],
                "cav_actions": {},
                "arrived_ids": [],
                "boundary_crossings": {},
            }
        )
    return rows


class TestPerArmExitIdentity:
    """PL-02a-S3 (1): teleport detection uses THAT arm's own exits.  The
    base arm (no arrivals file) derives them from its own physics rows; a
    sibling's exit list is never borrowed."""

    def test_excerpt_base_derives_exits_verify_reads_own_arrivals(self):
        index = CampaignIndex(FIXTURE, None, R3_LAYOUT).build()
        root = index.roots["f8401m2700__base"]
        # window ends exactly at the last excerpt row: nobody vanishes
        base = branch_outcome(
            root.base,
            t0_s=60.0,
            window_s=2.5,
            layout=R3_LAYOUT,
            delay_provider=default_delay,
            provider_name="builtin",
            departs={"mainline.8": 0.4},
            window_start_s=60.5,
        )
        verify = branch_outcome(
            root.branches["verify"],
            t0_s=60.0,
            window_s=2.5,
            layout=R3_LAYOUT,
            delay_provider=default_delay,
            provider_name="builtin",
            departs={"mainline.8": 0.4},
            window_start_s=60.5,
        )
        assert base.exits_source == "none"  # no exit info in this window
        assert verify.exits_source == "arrivals_file"
        assert base.n_teleports == verify.n_teleports == 0
        assert base.delay_s == pytest.approx(verify.delay_s, abs=1e-9)

    def test_named_exit_not_teleport_midsection_vanish_is(self, tmp_path):
        runs_root = tmp_path / "runs"
        rows = []
        positions = {"ex": 0.0, "w": 0.0}
        for t in np.arange(T0, T0 + 5.0, ROW_DT):
            vehicles = {}
            arrived = []
            for vid, pos in positions.items():
                if vid == "ex" and t >= T0 + 4.0:
                    arrived.append("ex")  # exits the network at this row
                    continue
                if vid == "w" and t >= T0 + 2.5:
                    continue  # vanishes mid-section, not an exit
                vehicles[vid] = veh(road_id="e0", lane=0, pos=pos, speed=20.0)
                positions[vid] += 10.0
            rows.append(
                {
                    "time_s": float(t),
                    "vehicles": vehicles,
                    "colliding_ids": [],
                    "cav_actions": {},
                    "arrived_ids": arrived,
                    "boundary_crossings": {},
                }
            )
        write_run(runs_root, "r__base", arm="base", is_base=True, rows=rows)
        index = CampaignIndex(runs_root, None, Layout(require_native_sibling=False)).build()
        root = index.roots["r__base"]
        outcome = branch_outcome(
            root.base,
            t0_s=T0,
            window_s=5.0,
            layout=Layout(),
            delay_provider=default_delay,
            departs={},
        )
        assert outcome.exits_source == "physics_rows"
        assert outcome.n_teleports == 1
        assert "w" in outcome.invalid_reasons[0]
        assert "ex" not in outcome.invalid_reasons[0]

    def test_base_arm_does_not_borrow_sibling_exits(self, tmp_path):
        runs_root = tmp_path / "runs"
        # base rows: "ok" exits via its own rows' arrived_ids, "x" vanishes
        base_rows = []
        positions = {"ok": 0.0, "x": 0.0}
        for t in np.arange(T0, T0 + 5.0, ROW_DT):
            vehicles = {}
            arrived = []
            for vid, pos in positions.items():
                if vid == "ok" and t >= T0 + 4.0:
                    arrived.append("ok")
                    continue
                if vid == "x" and t >= T0 + 2.5:
                    continue  # vanishes mid-section
                vehicles[vid] = veh(road_id="e0", lane=0, pos=pos, speed=20.0)
                positions[vid] += 10.0
            base_rows.append(
                {
                    "time_s": float(t),
                    "vehicles": vehicles,
                    "colliding_ids": [],
                    "cav_actions": {},
                    "arrived_ids": arrived,
                    "boundary_crossings": {},
                }
            )
        # verify rows: only "x", and the verify arm's OWN arrivals file
        # names "x" as an exit -- but the base must not borrow that list
        verify_rows = [row for row in vanish_rows(vanish_after_s=T0 + 2.5)]
        write_run(runs_root, "r__base", arm="base", is_base=True, rows=base_rows)
        write_run(
            runs_root,
            "r__verify",
            arm="verify",
            is_base=False,
            rows=verify_rows,
            arrivals=[("x", 500.0)],
        )
        layout = Layout(require_native_sibling=False)
        index = CampaignIndex(runs_root, None, layout).build()
        root = index.roots["r__base"]
        base = branch_outcome(
            root.base,
            t0_s=T0,
            window_s=5.0,
            layout=layout,
            delay_provider=default_delay,
            departs={},
        )
        assert base.exits_source == "physics_rows"
        assert base.n_teleports == 1  # "x" flagged DESPITE the sibling's list
        assert "x" in base.invalid_reasons[0]
        verify = branch_outcome(
            root.branches["verify"],
            t0_s=T0,
            window_s=5.0,
            layout=layout,
            delay_provider=default_delay,
            departs={},
        )
        assert verify.exits_source == "arrivals_file"
        assert verify.n_teleports == 0  # its OWN list names the exit

    def test_exit_zone_last_sighting_counts_as_exit(self, tmp_path):
        runs_root = tmp_path / "runs"
        rows = []
        for t in np.arange(T0, T0 + 5.0, ROW_DT):
            vehicles = (
                {"z": veh(road_id="e0", lane=0, pos=980.0, speed=20.0)}
                if t <= T0 + 2.5 + 1e-9
                else {}
            )
            rows.append(
                {
                    "time_s": float(t),
                    "vehicles": vehicles,
                    "colliding_ids": [],
                    "cav_actions": {},
                    "arrived_ids": [],
                    "boundary_crossings": {},
                }
            )
        write_run(runs_root, "r__base", arm="base", is_base=True, rows=rows)
        index = CampaignIndex(runs_root, None, Layout(require_native_sibling=False)).build()
        root = index.roots["r__base"]
        outcome = branch_outcome(
            root.base,
            t0_s=T0,
            window_s=5.0,
            layout=Layout(section_len_m=1000.0, exit_band_m=30.0),  # zone >= 970 m
            delay_provider=default_delay,
            departs={},
        )
        assert outcome.exits_source == "derived_positions"
        assert outcome.n_teleports == 0


class TestT0InsertionFromManifest:
    """PL-02a-S3 (2): t0 and the t0-present set come from the base's
    saved-state manifest (branch_point_t0_s / physics_row_at_t0.live_ids),
    never from a hard-coded 60.0; vehicles present at t0 get ZERO entry
    wait in every arm (§8.14), pending ones wait from t0."""

    def test_manifest_supplies_t0_and_t0_present_set(self):
        index = CampaignIndex(FIXTURE, None, R3_LAYOUT).build()
        root = index.roots["f8401m2700__base"]
        assert root.t0_s == 60.0  # from branch_point_t0_s
        assert root.t0_present_ids == frozenset({"mainline.8"})

    def test_t0_comes_from_manifest_not_hardcoded(self, tmp_path):
        copied = shutil.copytree(FIXTURE, tmp_path / "fixture")
        manifest_path = copied / "f8401m2700__base" / "saved_state" / "STATE_MANIFEST.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["branch_point_t0_s"] = 90.0
        manifest_path.write_text(json.dumps(manifest))
        index = CampaignIndex(copied, None, R3_LAYOUT).build()
        root = index.roots["f8401m2700__base"]
        assert root.t0_s == 90.0
        assert root.t0_present_ids == frozenset({"mainline.8"})

    def test_t0_present_zero_wait_in_restored_arm_pending_waits_from_t0(self):
        index = CampaignIndex(FIXTURE, None, R3_LAYOUT).build()
        root = index.roots["f8401m2700__base"]
        departs = {"mainline.8": 0.4, "mainline.70": 30.0}
        verify = root.branches["verify"]
        kwargs = dict(
            t0_s=root.t0_s,
            window_s=3.5,
            layout=R3_LAYOUT,
            delay_provider=default_delay,
            departs=departs,
            window_start_s=root.t0_s,
        )
        with_ids = branch_outcome(verify, t0_present_ids=root.t0_present_ids, **kwargs)
        without = branch_outcome(verify, t0_present_ids=None, **kwargs)
        # mainline.8 is present at t0 (manifest) but the restored arm's
        # first row is t0+0.5: with the manifest it contributes ZERO wait,
        # without it a phantom 0.5 s accrues (floor t0, first sight t0+0.5).
        # The pending vehicle (scheduled 30 s, never inserted) waits FROM
        # t0 to the window end (63.5): 3.5 s.
        assert with_ids.entry_wait_s == pytest.approx(3.5)
        assert without.entry_wait_s == pytest.approx(4.0)
        base = branch_outcome(root.base, t0_present_ids=root.t0_present_ids, **kwargs)
        assert base.entry_wait_s == pytest.approx(3.5)


class TestFailedReceiptExactGate:
    """PL-02a-S3 (4): a 'failed' receipt passes only when
    accept_failed_receipt_if_exact is set AND the run's EXACTNESS_VERDICT
    shows divergence_count == 0 with the t0 tick as the ONLY missing one;
    otherwise the run is excluded and counted."""

    def _index(self, root, **layout_kw):
        layout = Layout(restored_file_subdir="run", require_native_sibling=False, **layout_kw)
        return CampaignIndex(root, None, layout).build()

    def test_default_gate_excludes_failed_receipt(self):
        index = self._index(FIXTURE)
        assert set(index.roots["f8401m2700__base"].branches) == set()
        assert any("f8401m2700__verify" in s for s in index.skipped)

    def test_gate_accepts_exact_t0_boundary_only_failure(self):
        index = self._index(FIXTURE, accept_failed_receipt_if_exact=True)
        assert set(index.roots["f8401m2700__base"].branches) == {"verify"}
        assert index.skipped == []

    def test_gate_rejects_divergence_extra_beyond_t0_and_missing_verdict(self, tmp_path):
        def verdict_mutator(mutate):
            def apply(fixture_root):
                path = fixture_root / "f8401m2700__verify" / "run" / "EXACTNESS_VERDICT.json"
                verdict = json.loads(path.read_text())
                mutate(verdict)
                path.write_text(json.dumps(verdict))

            return apply

        cases = [
            ("divergent", verdict_mutator(lambda v: v.update(divergence_count=1))),
            ("extra", verdict_mutator(lambda v: v.update(extra_in_verify=[60.5]))),
            (
                "missing_beyond_t0",
                verdict_mutator(lambda v: v.update(missing_in_verify=[60.0, 60.5])),
            ),
            (
                "noverdict",
                lambda root: (
                    root / "f8401m2700__verify" / "run" / "EXACTNESS_VERDICT.json"
                ).unlink(),
            ),
        ]
        for name, mutate in cases:
            copied = shutil.copytree(FIXTURE, tmp_path / name)
            mutate(copied)
            index = self._index(copied, accept_failed_receipt_if_exact=True)
            assert set(index.roots["f8401m2700__base"].branches) == set(), name
            assert index.skipped, name


class TestRecordedPilotExtensions:
    """PL-02a PILOT reader/runner extensions: (a) the constant plan of an
    arm WITHOUT a switch_schedule is DECLARED by branch_record
    (suffix_controller) -- physics cav_actions log control activity, not
    the plan payload, and must never be mistaken for it; (b) the model
    takes an explicit common-window start (the restore does not re-record
    the t0 tick); (c) the runner's verify-vs-base integrity gate flags and
    excludes mismatching worlds."""

    def test_declared_plan_from_branch_record_without_schedule(self, tmp_path):
        runs_root = tmp_path / "runs"
        # the pi arm's cav_actions VARY per tick (control activity); with no
        # schedule the plan must come from branch_record.suffix_controller
        rows = []
        for i, row in enumerate(steady_rows(T0, T0 + 20.0, ("v1",), cav_ids=("v1",))):
            row = dict(row)
            action = "damping" if i % 2 else "native"
            row["cav_actions"] = {"v1": {"owner": "dispatcher", "action": action}}
            rows.append(row)
        write_run(
            runs_root,
            "w9__base",
            arm="base",
            is_base=True,
            rows=steady_rows(0.0, T0 + 20.0, ("v1",)),
        )
        write_run(
            runs_root,
            "w9__native",
            arm="native",
            is_base=False,
            rows=steady_rows(T0, T0 + 20.0, ("v1",), cav_ids=("v1",)),
        )
        write_run(
            runs_root,
            "w9__pi",
            arm="pi",
            is_base=False,
            rows=rows,
            branch_record={"arm": "pi", "root_id": "w9", "suffix_controller": "pi-external-u"},
        )
        index = CampaignIndex(runs_root).build()
        root = index.roots["w9__base"]
        outcome = branch_outcome(
            root.branches["pi"],
            t0_s=T0,
            window_s=20.0,
            layout=Layout(),
            departs={},
        )
        assert outcome.realised_plan == {"v1": "pi-external-u"}
        assert outcome.is_constant and outcome.realised_source == "declared"
        # horizon ends at the last row (T0+19.5): the vanish rule would
        # otherwise flag v1's end-of-rows as a teleport
        model = RecordedBranchWorldModel(root, layout=Layout(), horizon_s=19.5)
        key = tuple(sorted({"v1": "pi-external-u"}.items()))
        assert model.plans[key] == "pi"
        prediction = model.predict_many(
            [_sample()], [PlanCandidate(per_cav={"v1": "pi-external-u"})]
        )[0]
        assert prediction.cell_valid[0, 0]
        assert prediction.meta["recorded_arm"] == "pi"

    def test_verify_arm_is_integrity_not_candidate(self, tmp_path):
        runs_root = tmp_path / "runs"
        rows = steady_rows(T0, T0 + 20.0, ("v1",))
        write_run(
            runs_root,
            "w10__base",
            arm="base",
            is_base=True,
            rows=steady_rows(0.0, T0 + 20.0, ("v1",)),
        )
        write_run(
            runs_root,
            "w10__native",
            arm="native",
            is_base=False,
            rows=steady_rows(T0, T0 + 20.0, ("v1",), cav_ids=("v1",)),
        )
        write_run(
            runs_root,
            "w10__verify",
            arm="verify",
            is_base=False,
            rows=rows,
            branch_record={"arm": "verify", "root_id": "w10", "suffix_controller": "damping"},
        )
        index = CampaignIndex(runs_root).build()
        root = index.roots["w10__base"]
        assert "verify" in [run.arm for run in root.branches.values()]
        model = RecordedBranchWorldModel(root, layout=Layout(), horizon_s=20.0)
        assert "verify" in model.outcomes
        for arm in model.plans.values():
            assert arm != "verify"  # integrity arm: never a candidate

    def test_model_window_start_s_common_window(self):
        """The r3-excerpt fixture: base keeps its t0 anchor row, the restore
        starts at t0+dt.  With the common start (60.5) base == verify;
        from t0 the base's anchor row breaks the equality."""
        index = CampaignIndex(FIXTURE, None, R3_LAYOUT).build()
        root = index.roots["f8401m2700__base"]
        common = RecordedBranchWorldModel(root, layout=R3_LAYOUT, window_start_s=60.5)
        assert common.window_start_s == 60.5
        assert common.outcomes["base"].delay_s == pytest.approx(
            common.outcomes["verify"].delay_s, abs=1e-9
        )
        from_t0 = RecordedBranchWorldModel(root, layout=R3_LAYOUT, window_start_s=60.0)
        assert from_t0.outcomes["base"].delay_s != from_t0.outcomes["verify"].delay_s

    def test_runner_verify_integrity_gate(self, tmp_path):
        runner = _runner_module()
        runs_root = tmp_path / "runs"
        # the verify arm's rows are the BASE's own suffix rows (same-action
        # continuation); the bad world drops one row mid-window
        base_rows = steady_rows(0.0, T0 + 20.0, ("v1",))
        good = [row for row in base_rows if row["time_s"] >= T0]
        bad = [row for row in good if row["time_s"] != T0 + 5.0]
        write_run(runs_root, "w11__base", arm="base", is_base=True, rows=base_rows)
        write_run(
            runs_root,
            "w11__native",
            arm="native",
            is_base=False,
            rows=steady_rows(T0, T0 + 20.0, ("v1",), cav_ids=("v1",)),
        )
        write_run(
            runs_root,
            "w11__verify",
            arm="verify",
            is_base=False,
            rows=good,
            branch_record={"arm": "verify", "root_id": "w11", "suffix_controller": "damping"},
        )
        # a second world whose verify arm does NOT match its base
        write_run(runs_root, "w12__base", arm="base", is_base=True, rows=base_rows)
        write_run(
            runs_root,
            "w12__native",
            arm="native",
            is_base=False,
            rows=steady_rows(T0, T0 + 20.0, ("v1",), cav_ids=("v1",)),
        )
        write_run(
            runs_root,
            "w12__verify",
            arm="verify",
            is_base=False,
            rows=bad,
            branch_record={"arm": "verify", "root_id": "w12", "suffix_controller": "damping"},
        )
        index = CampaignIndex(runs_root).build()
        good_record = runner.analyse_root(index.roots["w11__base"], Layout(), 20.0, PlannerConfig())
        assert good_record["integrity"]["present"] and good_record["integrity_ok"]
        assert not good_record["integrity_failed"]
        bad_record = runner.analyse_root(index.roots["w12__base"], Layout(), 20.0, PlannerConfig())
        assert bad_record["integrity"]["present"] and not bad_record["integrity_ok"]
        assert bad_record["integrity_failed"]
        assert any(
            "delay_equal" in name
            for name, ok in bad_record["integrity"]["checks"].items()
            if not ok
        )
        summary = runner.summarise([good_record, bad_record], PlannerConfig())
        assert summary["headroom"]["n_roots_excluded_integrity"] == 1
        assert summary["headroom"]["roots_excluded_integrity"] == ["w12__base"]
        assert summary["headroom"]["n_roots_intact"] == 1


class TestOwnerRuleReview:
    """PL-02a-P2: 'safe' means the planner's own safety_check (absolute +
    relative vs native).  A faster-but-relatively-unsafe arm must NOT be
    best-safe; the absolute-only variant is reported separately, the
    best-safe-saving median counts no-safe-improvement roots as 0, and the
    rel_safety_tol_s sensitivity table flips the arm at the owner's
    tolerance."""

    @staticmethod
    def _closing_rows(start, end, fast_rows, fast_speed=25.0):
        """v1 cruises at 20; v2 starts 30 m behind, gains at fast_speed for
        ``fast_rows`` rows (min TTC 2.5 s, TTC<3 exposure 0.5 s), then
        matches speed."""
        rows = []
        p1, p2 = 100.0, 70.0
        t = start
        k = 0
        while t < end - 1e-9:
            speed = fast_speed if k < fast_rows else 20.0
            rows.append(
                {
                    "time_s": float(t),
                    "vehicles": {
                        "v1": veh(road_id="e0", lane=0, pos=p1, speed=20.0),
                        "v2": veh(road_id="e0", lane=0, pos=p2, speed=speed),
                    },
                    "colliding_ids": [],
                    "cav_actions": {},
                    "arrived_ids": [],
                    "boundary_crossings": {},
                }
            )
            p1 += 20.0 * ROW_DT
            p2 += speed * ROW_DT
            t += ROW_DT
            k += 1
        return rows

    def test_owner_rule_best_safe_and_absolute_variant(self, tmp_path):
        runner = _runner_module()
        runs_root = tmp_path / "runs"
        # base: both vehicles slightly slower than native -> strictly higher
        # delay, so the absolute-variant best safe arm is unambiguous
        base_rows = steady_rows(0.0, T0 + 15.0, ("v1", "v2"), speed=19.0)
        native_rows = steady_rows(T0, T0 + 15.0, ("v1", "v2"), cav_ids=("v1",))
        fixed_rows = self._closing_rows(T0, T0 + 15.0, fast_rows=8)
        write_run(runs_root, "w13__base", arm="base", is_base=True, rows=base_rows)
        write_run(
            runs_root,
            "w13__native",
            arm="native",
            is_base=False,
            rows=native_rows,
        )
        write_run(
            runs_root,
            "w13__fixed",
            arm="fixed",
            is_base=False,
            rows=fixed_rows,
            schedule={"events": [{"t_s": T0, "per_cav": {"v1": "pi-external-u"}}]},
            branch_record={"arm": "fixed", "root_id": "w13"},
        )
        index = CampaignIndex(runs_root).build()
        # T-A: this fixture vets the 3 s SECONDARY exposure rule (min TTC
        # 2.5 s sits between the 1.5 and 3.0 thresholds); the 3 s selection
        # is explicit now that the config default is the primary 1.5 s.
        record = runner.analyse_root(
            index.roots["w13__base"], Layout(), 14.5, PlannerConfig(relative_ttc_threshold_s=3.0)
        )
        fixed = record["arms"]["fixed"]
        # fixed is FASTER than native and outcome-safe (min ttc 2.5 >= 1.5),
        # but its TTC<3 exposure increase (+0.5 s vs native 0) vetoes it
        # under the owner rule
        assert fixed["delay_s"] < record["native_delay_s"]
        assert fixed["safe_outcome"] is True
        assert fixed["safe_planner"] is False
        rel = {r["metric"]: r for r in fixed["safety_verdict"]["relative"]}
        assert rel["ttc_lt3_exposure_s"]["mean_increase_s"] == pytest.approx(0.5)
        assert rel["ttc_lt3_exposure_s"]["ok"] is False
        # owner rule keeps native; the absolute-only variant picks fixed
        assert record["best_safe_arm"] == "native"
        assert record["anything_beats_native_safely"] is False
        assert record["best_safe_arm_absolute"] == "fixed"
        assert record["anything_beats_native_safely_absolute"] is True
        # rel_safety_tol_s sensitivity: fixed becomes planner-safe at 5 veh-s
        # T-B (#12): the sensitivity sweep now INHERITS the caller's config
        # (threshold 3.0 here), so the 3 s veto shows at tolerance 0 and the
        # flip to planner-safe happens by 5 veh-s -- the sweep used to
        # rebuild DEFAULT (1.5 s primary) configs and hide the veto.
        assert record["tolerance_sensitivity"]["0.0"]["fixed"] is False
        assert record["tolerance_sensitivity"]["5.0"]["fixed"] is True
        # both providers agree on the savings' sign in this world
        assert record["provider_signs_agree"] is True

    def test_owner_rule_median_counts_zero_and_sensitivity(self, tmp_path):
        runner = _runner_module()
        runs_root = tmp_path / "runs"
        base_rows = steady_rows(0.0, T0 + 15.0, ("v1", "v2"), speed=19.0)
        write_run(runs_root, "w14__base", arm="base", is_base=True, rows=base_rows)
        write_run(
            runs_root,
            "w14__native",
            arm="native",
            is_base=False,
            rows=steady_rows(T0, T0 + 15.0, ("v1", "v2"), cav_ids=("v1",)),
        )
        write_run(
            runs_root,
            "w14__fixed",
            arm="fixed",
            is_base=False,
            rows=self._closing_rows(T0, T0 + 15.0, fast_rows=8),
            schedule={"events": [{"t_s": T0, "per_cav": {"v1": "pi-external-u"}}]},
            branch_record={"arm": "fixed", "root_id": "w14"},
        )
        index = CampaignIndex(runs_root).build()
        record = runner.analyse_root(
            index.roots["w14__base"], Layout(), 14.5, PlannerConfig(relative_ttc_threshold_s=3.0)
        )
        summary = runner.summarise([record], PlannerConfig(relative_ttc_threshold_s=3.0))
        headroom = summary["headroom"]
        # no planner-safe improver -> counts as 0; absolute variant improves
        assert headroom["median_best_safe_saving_s"] == 0.0
        assert headroom["fraction_roots_beat_native_safely"] == 0.0
        assert headroom["median_best_safe_saving_absolute_s"] > 0.0
        assert headroom["fraction_roots_beat_native_safely_absolute"] == 1.0
        sensitivity = headroom["tolerance_sensitivity"]
        # T-B (#12): the sweep inherits the caller's 3 s threshold: the
        # TTC-2.5 fixture is vetoed at tolerance 0 and planner-safe by 5.
        assert sensitivity["0.0"]["n_roots_beat_native_safely"] == 0
        assert sensitivity["5.0"]["n_roots_beat_native_safely"] == 1


class TestChunk1Layout:
    """PL-02a chunk-1 layout: the switch-schedule payload is NESTED under a
    container key, events are per-VEHICLE rows at time_s with the t0 payload
    as an assignments map, the dispatcher telemetry is per-vehicle, and a
    schedule is constant iff its events carry exactly one distinct time."""

    LAYOUT = Layout(
        require_native_sibling=False,
        schedule_container_key="schedule",
        schedule_plan_key="assignments",
        schedule_time_key="time_s",
        dispatcher_rows_format="per_vehicle",
    )

    def _fixed_schedule(self, payload, extra_events=()):
        events = [
            {"vehicle_id": cav, "time_s": T0, "action": action} for cav, action in payload.items()
        ] + list(extra_events)
        return {
            "run": "w21__fixed1",
            "schedule": {"events": events, "assignments": dict(payload), "t0_s": T0},
        }

    def test_nested_assignments_and_per_vehicle_dispatcher(self, tmp_path):
        runs_root = tmp_path / "runs"
        rows = steady_rows(T0, T0 + 20.0, ("c0", "c1"))
        payload = {"c0": "damping", "c1": "pi-external-u"}
        write_run(
            runs_root,
            "w21__base",
            arm="base",
            is_base=True,
            rows=steady_rows(0.0, T0 + 20.0, ("c0", "c1")),
        )
        write_run(
            runs_root,
            "w21__fixed1",
            arm="fixed1",
            is_base=False,
            rows=rows,
            schedule=self._fixed_schedule(payload),
            dispatcher=[
                {"timestamp_s": T0 + 1, "vehicle_id": "c0", "action": "damping"},
                {"timestamp_s": T0 + 1, "vehicle_id": "c1", "action": "pi-external-u"},
            ],
        )
        index = CampaignIndex(runs_root, None, self.LAYOUT).build()
        root = index.roots["w21__base"]
        outcome = branch_outcome(
            root.branches["fixed1"],
            t0_s=T0,
            window_s=19.5,
            layout=self.LAYOUT,
            delay_provider=default_delay,
            departs={},
        )
        assert outcome.realised_plan == payload
        assert outcome.is_constant and outcome.realised_source == "schedule"
        assert outcome.cell_valid

    def test_per_vehicle_dispatcher_mismatch_invalidates(self, tmp_path):
        runs_root = tmp_path / "runs"
        rows = steady_rows(T0, T0 + 20.0, ("c0", "c1"))
        payload = {"c0": "damping", "c1": "pi-external-u"}
        write_run(
            runs_root,
            "w21__base",
            arm="base",
            is_base=True,
            rows=steady_rows(0.0, T0 + 20.0, ("c0", "c1")),
        )
        write_run(
            runs_root,
            "w21__fixed1",
            arm="fixed1",
            is_base=False,
            rows=rows,
            schedule=self._fixed_schedule(payload),
            dispatcher=[
                {"timestamp_s": T0 + 1, "vehicle_id": "c0", "action": "native"},
            ],
        )
        index = CampaignIndex(runs_root, None, self.LAYOUT).build()
        root = index.roots["w21__base"]
        outcome = branch_outcome(
            root.branches["fixed1"],
            t0_s=T0,
            window_s=19.5,
            layout=self.LAYOUT,
            delay_provider=default_delay,
            departs={},
        )
        assert not outcome.cell_valid
        assert any("realised action differs" in reason for reason in outcome.invalid_reasons)

    def test_multi_time_schedule_is_not_constant(self, tmp_path):
        runs_root = tmp_path / "runs"
        rows = steady_rows(T0, T0 + 20.0, ("c0", "c1"))
        payload = {"c0": "damping", "c1": "pi-external-u"}
        write_run(
            runs_root,
            "w21__base",
            arm="base",
            is_base=True,
            rows=steady_rows(0.0, T0 + 20.0, ("c0", "c1")),
        )
        write_run(
            runs_root,
            "w21__switching1",
            arm="switching1",
            is_base=False,
            rows=rows,
            schedule=self._fixed_schedule(
                payload,
                extra_events=[{"vehicle_id": "c0", "time_s": T0 + 10.0, "action": "native"}],
            ),
        )
        index = CampaignIndex(runs_root, None, self.LAYOUT).build()
        root = index.roots["w21__base"]
        outcome = branch_outcome(
            root.branches["switching1"],
            t0_s=T0,
            window_s=19.5,
            layout=self.LAYOUT,
            delay_provider=default_delay,
            departs={},
        )
        assert not outcome.is_constant
        assert outcome.realised_source == "schedule"


class TestChunk1ArmIdentity:
    """PL-02a chunk-1: the run-dir suffix is the arm identity.  The chunk-1
    world plan labels fixed1/fixed2 both as branch type "fixed" (and
    switching1-3 as "switching"); deriving arms from the plan meta would
    collide them in root.branches.  Regression: two 'fixed' branches attach
    as DISTINCT arms fixed1/fixed2."""

    def test_fixed1_fixed2_attach_as_distinct_arms(self, tmp_path):
        runs_root = tmp_path / "runs"
        write_run(
            runs_root,
            "w30__base",
            arm="base",
            is_base=True,
            rows=steady_rows(0.0, T0 + 20.0, ("c0", "c1")),
        )
        for arm in ("fixed1", "fixed2"):
            write_run(
                runs_root,
                f"w30__{arm}",
                arm=arm,
                is_base=False,
                rows=steady_rows(T0, T0 + 20.0, ("c0", "c1")),
                branch_record={"branch_type": "fixed", "root_id": "w30", "t0_s": T0},
            )
        layout = Layout(require_native_sibling=False)
        index = CampaignIndex(runs_root, None, layout).build()
        assert set(index.roots["w30__base"].branches) == {"fixed1", "fixed2"}


class TestChunk1DeclaredPlans:
    """PL-02a chunk-1: native/pi arms declare their constant plan via
    branch_record branch_type (mapped to the arm's action), and their CAV
    universe comes from the rows' cav_ids lists (all-native arms log no
    cav_actions).  Schedules/dispatcher files live under the restored file
    subdir."""

    LAYOUT = Layout(
        require_native_sibling=False,
        restored_file_subdir="run",
        branch_type_actions={"native": "native", "pi": "pi-external-u"},
    )

    def _rows(self, cav_ids=("c0", "c1"), actions=()):
        rows = []
        t = T0
        while t < T0 + 20.0 - 1e-9:
            rows.append(
                {
                    "time_s": float(t),
                    "vehicles": {
                        cav: veh(road_id="e0", lane=0, pos=10.0 * (t - T0), speed=20.0)
                        for cav in cav_ids
                    },
                    "colliding_ids": [],
                    "cav_actions": dict(actions),
                    "cav_ids": list(cav_ids),
                    "arrived_ids": [],
                    "boundary_crossings": {},
                }
            )
            t += ROW_DT
        return rows

    def test_native_and_pi_declared_from_branch_type(self, tmp_path):
        runs_root = tmp_path / "runs"
        write_run(runs_root, "w31__base", arm="base", is_base=True, rows=self._rows())
        for arm, record_type, root_dir in (
            ("native", "native", "w31__native"),
            ("pi", "pi", "w31__pi"),
        ):
            run_dir = runs_root / root_dir
            (run_dir / "run").mkdir(parents=True, exist_ok=True)
            (run_dir / layout_receipt_name()).write_text(
                json.dumps({"run": run_dir.name, "exit_status": "complete"})
            )
            physics_text = "".join(json.dumps(row) + chr(10) for row in self._rows())
            (run_dir / "run" / layout_physics()).write_text(physics_text)
            (run_dir / "run" / "branch_record.json").write_text(
                json.dumps({"root_id": "w31", "branch_type": record_type, "t0_s": T0})
            )
        index = CampaignIndex(runs_root, None, self.LAYOUT).build()
        root = index.roots["w31__base"]
        model = RecordedBranchWorldModel(root, layout=self.LAYOUT, horizon_s=19.5)
        arms = sorted(set(model.plans.values()))
        assert "native" in arms and "pi" in arms
        native_key = tuple(sorted({"c0": "native", "c1": "native"}.items()))
        assert model.plans[native_key] == "native"
        pi_key = tuple(sorted({"c0": "pi-external-u", "c1": "pi-external-u"}.items()))
        assert model.plans[pi_key] == "pi"
        prediction = model.predict_many(
            [_sample()], [PlanCandidate(per_cav={"c0": "pi-external-u", "c1": "pi-external-u"})]
        )[0]
        assert prediction.cell_valid[0, 0]
        assert prediction.meta["recorded_arm"] == "pi"

    def test_base_damping_candidate_uses_cav_id_lists(self, tmp_path):
        """PL-02a-C1 fix: the recorded base/damping candidate key comes from
        the base run's CAV universe (cav_actions keys PLUS cav_ids lists);
        chunk-1 base rows log empty cav_actions, so without the cav_ids
        lists the current=damping analysis would have no candidate."""
        runs_root = tmp_path / "runs"
        write_run(runs_root, "w32__base", arm="base", is_base=True, rows=self._rows())
        index = CampaignIndex(runs_root, None, self.LAYOUT).build()
        model = RecordedBranchWorldModel(
            index.roots["w32__base"], layout=self.LAYOUT, horizon_s=19.5
        )
        damping_key = tuple(sorted({"c0": "damping", "c1": "damping"}.items()))
        assert model.plans.get(damping_key) == "base"


def layout_physics():
    from traffic_shepherd.planner.recorded import Layout

    return Layout().physics_name


def layout_receipt_name():
    from traffic_shepherd.planner.recorded import Layout

    return Layout().receipt_name


class TestStreamingScanParity:
    """PL-02a-C1: the single streaming scan + vectorised metrics must
    reproduce the dict-row reader's numbers exactly (regression on the real
    pilot-r3 fixture excerpt and on hand-computed synthetic rows)."""

    FIXTURES = Path(__file__).parent / "fixtures" / "pilot_r3_excerpt"
    LAYOUT = Layout(
        restored_file_subdir="run",
        require_native_sibling=False,
        accept_failed_receipt_if_exact=True,
    )

    @staticmethod
    def _all_rows(run):
        from traffic_shepherd.planner.recorded import load_jsonl

        rows = []
        for path in run.physics_paths:
            rows.extend(load_jsonl(path))
        return rows

    def test_scan_equals_old_reader_on_pilot_r3_excerpt(self):
        from traffic_shepherd.planner.recorded import (
            _exits_from_rows,
            _onset_s,
            _safety_from_rows,
            _teleports,
            load_jsonl,
        )

        index = CampaignIndex(self.FIXTURES, None, self.LAYOUT).build()
        root = index.roots["f8401m2700__base"]
        window_s = 100.0
        for run in [root.base] + list(root.branches.values()):
            rows_all = self._all_rows(run)
            rows = [
                row
                for row in rows_all
                if root.t0_s - 1e-9 <= float(row["time_s"]) <= root.t0_s + window_s + 1e-9
            ]
            delay, waits, n_vehicles, _diag = default_delay(
                rows, t0_s=root.t0_s, window_s=window_s, layout=self.LAYOUT, departs={}
            )
            safety = _safety_from_rows(rows, self.LAYOUT)
            # exit identity: the run's own arrivals file wins (pilot layout),
            # else the row-derived exits (arrived_ids + exit-zone rule)
            if run.arrivals_path is not None:
                exits = {
                    str(row.get("vehicle_id"))
                    for row in load_jsonl(run.arrivals_path)
                    if row.get("vehicle_id")
                }
                exits_source = "arrivals_file"
            else:
                exits, exits_source = _exits_from_rows(rows, self.LAYOUT)
            n_teleports, _cands = _teleports(rows, self.LAYOUT, exits, root.t0_s + window_s)
            onset = _onset_s(rows, self.LAYOUT)
            outcome = branch_outcome(
                run,
                t0_s=root.t0_s,
                window_s=window_s,
                layout=self.LAYOUT,
                delay_provider=default_delay,
                departs={},
                provider_name="builtin",
            )
            assert outcome.delay_s == delay
            assert outcome.entry_wait_s == waits
            assert outcome.n_vehicles == n_vehicles
            assert outcome.min_ttc_s == safety["min_ttc_s"]
            assert outcome.has_closing_pair == safety["has_closing_pair"]
            assert outcome.ttc_lt3_exposure_s == safety["ttc_lt3_exposure_s"]
            assert outcome.hard_brake_s == safety["hard_brake_s"]
            assert outcome.n_teleports == n_teleports
            assert outcome.onset_s == onset
            assert outcome.exits_source == exits_source
            assert outcome.n_exits == len(exits)

    def test_vectorised_ttc_hand_computed(self):
        """Two lanes, one hard closing pair (ttc 1.0), one receding pair
        (no ttc), one hard brake: the numpy per-tick pass (vehicles sorted
        by lane position once per tick) must produce exactly these numbers."""
        from traffic_shepherd.planner.recorded import _safety_from_rows

        layout = Layout()
        rows = []
        t = T0
        for _ in range(4):  # 4 rows x 0.5 s
            rows.append(
                {
                    "time_s": t,
                    "vehicles": {
                        "lead_fast": veh(lane=0, pos=100.0, speed=10.0, accel=0.0),
                        "follower": veh(lane=0, pos=90.0, speed=20.0, accel=0.0),
                        "tail_braking": veh(lane=1, pos=90.0, speed=18.0, accel=-4.0),
                        "lead_slow": veh(lane=1, pos=100.0, speed=20.0, accel=0.0),
                        "receding_tail": veh(lane=2, pos=90.0, speed=1.0, accel=0.0),
                        "receding_lead": veh(lane=2, pos=100.0, speed=5.0, accel=0.0),
                    },
                    "colliding_ids": [],
                    "cav_actions": {},
                    "cav_ids": [],
                    "arrived_ids": [],
                }
            )
            t += ROW_DT
        safety = _safety_from_rows(rows, layout)
        # lane 0: gap 10, closing 20-10=10 -> ttc 1.0 (exposed, 4 x 0.5 s)
        # lane 1: gap 10, closing 18-20=-2 -> no ttc; hard brake -4 < -3
        # lane 2: gap 10, closing 1-5=-4 -> no ttc
        assert safety["min_ttc_s"] == 1.0
        assert safety["has_closing_pair"] is True
        assert safety["ttc_lt3_exposure_s"] == 4 * ROW_DT
        assert safety["hard_brake_s"] == 4 * ROW_DT
        assert safety["collisions"] == []

    def test_probe_run_times_matches_scan_and_falls_back(self, tmp_path):
        from traffic_shepherd.planner.recorded import (
            RunPaths,
            probe_run_times,
            scan_run_physics,
        )

        layout = Layout()
        monotone = tmp_path / layout.physics_name
        monotone.write_text(
            "".join(json.dumps({"time_s": 10.0 * i, "vehicles": {}}) + "\n" for i in range(5))
        )
        run = RunPaths(directory=tmp_path, arm="x", is_base=True, receipt={})
        run.physics_paths = (monotone,)
        assert probe_run_times(run, layout) == (0.0, 40.0)
        # non-monotone file: the regex probe refuses, the exact scan answers
        scrambled = tmp_path / "scrambled.jsonl"
        scrambled.write_text(
            "".join(
                json.dumps({"time_s": ts, "vehicles": {}}) + "\n"
                for ts in (0.0, 30.0, 10.0, 40.0, 20.0)
            )
        )
        run2 = RunPaths(directory=tmp_path, arm="y", is_base=True, receipt={})
        run2.physics_paths = (scrambled,)
        first, last = probe_run_times(run2, layout)
        scan = scan_run_physics(run2, layout)
        assert (first, last) == (0.0, 40.0) == (scan.first_time, scan.last_time)


class TestRunCache:
    """PL-02a-C1 cache: hit/miss/stale-by-receipt/stale-by-code/stale-by-
    window, atomic writes, per-run exception isolation and progress log."""

    def test_second_invocation_is_all_cache_hits(self, campaign):
        runner = _runner_module()
        runs_root, plan_path, out_dir = campaign
        cache_dir = out_dir / "cache"
        first = runner.run(runs_root, plan_path, out_dir, cache_dir=cache_dir)
        second = runner.run(runs_root, plan_path, out_dir, cache_dir=cache_dir)
        assert first["timing"]["cache_misses"] == 3
        assert second["timing"]["cache_hits"] == 3
        assert second["timing"]["cache_misses"] == 0
        entries = sorted(
            p
            for p in (cache_dir / runs_root.name).glob("*.json")
            if p.name != "compute_timing.json"
        )
        assert len(entries) == 3  # base, native, fixed
        log = (cache_dir / runs_root.name / "progress.log").read_text(encoding="utf-8")
        assert log.count("|START|") == 6 and log.count("cache=miss|status=ok") == 3
        assert log.count("cache=hit|status=ok") == 3

    def test_stale_by_receipt_recomputes_only_that_run(self, campaign):
        runner = _runner_module()
        runs_root, plan_path, out_dir = campaign
        cache_dir = out_dir / "cache"
        runner.run(runs_root, plan_path, out_dir, cache_dir=cache_dir)
        receipt = runs_root / "w1__native" / "RUN_RECEIPT.json"
        payload = json.loads(receipt.read_text())
        payload["note"] = "rewritten"  # same status, different bytes
        receipt.write_text(json.dumps(payload))
        again = runner.run(runs_root, plan_path, out_dir, cache_dir=cache_dir)
        assert again["timing"]["cache_misses"] == 1
        assert again["timing"]["cache_hits"] == 2

    def test_stale_by_code_recomputes_everything(self, campaign, monkeypatch):
        runner = _runner_module()
        runs_root, plan_path, out_dir = campaign
        cache_dir = out_dir / "cache"
        runner.run(runs_root, plan_path, out_dir, cache_dir=cache_dir)
        monkeypatch.setattr(runner, "analysis_code_sha", lambda: "stale-code")
        again = runner.run(runs_root, plan_path, out_dir, cache_dir=cache_dir)
        assert again["timing"]["cache_misses"] == 3
        assert again["timing"]["cache_hits"] == 0

    def test_stale_by_window_recomputes_everything(self, campaign):
        runner = _runner_module()
        runs_root, plan_path, out_dir = campaign
        cache_dir = out_dir / "cache"
        runner.run(runs_root, plan_path, out_dir, cache_dir=cache_dir)
        again = runner.run(runs_root, plan_path, out_dir, cache_dir=cache_dir, horizon_s=100.0)
        assert again["timing"]["cache_misses"] == 3

    def test_cached_report_matches_fresh_report(self, campaign):
        runner = _runner_module()
        runs_root, plan_path, out_dir = campaign
        cache_dir = out_dir / "cache"
        first = runner.run(runs_root, plan_path, out_dir, cache_dir=cache_dir)
        second = runner.run(runs_root, plan_path, out_dir, cache_dir=cache_dir)
        assert first["per_root"][0]["arms"] == second["per_root"][0]["arms"]

    def test_atomic_write_no_tmp_leftover_and_safe_on_failure(self, tmp_path, monkeypatch):
        import os

        from traffic_shepherd.planner.recorded import atomic_write_json

        target = tmp_path / "entry.json"
        atomic_write_json(target, {"n": 1})
        assert json.loads(target.read_text()) == {"n": 1}
        assert not list(tmp_path.glob("*.tmp*"))
        # a failing os.replace leaves the previous entry intact, tmp cleaned

        def boom(src, dst):
            raise OSError("simulated crash")

        monkeypatch.setattr(os, "replace", boom)
        with pytest.raises(OSError):
            atomic_write_json(target, {"n": 2})
        monkeypatch.setattr(os, "replace", os.replace)
        assert json.loads(target.read_text()) == {"n": 1}
        assert not list(tmp_path.glob("*.tmp*"))

    def _two_root_campaign(self, tmp_path):
        runs_root = tmp_path / "runs"
        write_run(
            runs_root,
            "wa__base",
            arm="base",
            is_base=True,
            rows=steady_rows(0.0, T0 + 60.0, ("v1", "v2")),
        )
        write_run(
            runs_root,
            "wa__native",
            arm="native",
            is_base=False,
            rows=steady_rows(T0, T0 + 60.0, ("v1", "v2"), cav_ids=("v1", "v2")),
            branch_record={
                "suffix_controller": "native",
                "root_id": "wa",
                "branch_point_t0_s": T0,
            },
        )
        write_run(
            runs_root,
            "wb__base",
            arm="base",
            is_base=True,
            rows=steady_rows(0.0, T0 + 60.0, ("v1", "v2")),
        )
        write_run(
            runs_root,
            "wb__native",
            arm="native",
            is_base=False,
            rows=steady_rows(T0, T0 + 60.0, ("v1", "v2"), cav_ids=("v1", "v2")),
            branch_record={
                "suffix_controller": "native",
                "root_id": "wb",
                "branch_point_t0_s": T0,
            },
        )
        plan_path = tmp_path / "WORLD_PLAN.json"
        write_world_plan(
            plan_path,
            [
                world_entry("wa", split="TRAIN"),
                world_entry("wb", split="CALIBRATION"),
            ],
        )
        return runs_root, plan_path

    def test_per_run_error_is_fail_closed_and_logged(self, tmp_path):
        runner = _runner_module()
        runs_root, plan_path = self._two_root_campaign(tmp_path)
        out_dir = tmp_path / "out"
        cache_dir = out_dir / "cache"
        # corrupt one arm of root wa only
        physics = runs_root / "wa__native" / "physics_steps.jsonl"
        physics.write_text(physics.read_text() + "{corrupt\n")
        payload = runner.run(runs_root, plan_path, out_dir, cache_dir=cache_dir)
        by_root = {record["root"]: record for record in payload["per_root"]}
        assert "wa__base" in by_root and "wb__base" in by_root
        assert not by_root["wa__base"]["usable"]
        assert "run computation failed" in by_root["wa__base"]["unusable_reason"]
        assert by_root["wb__base"]["usable"]  # the other root still analysed
        log = (cache_dir / runs_root.name / "progress.log").read_text(encoding="utf-8")
        assert "run=wa__native" in log and "status=error" in log
        assert "JSONDecodeError" in log

    def test_cache_roundtrip_preserves_branch_outcome(self):
        from traffic_shepherd.planner.recorded import (
            branch_outcome_from_dict,
            branch_outcome_to_dict,
        )

        outcome = BranchOutcome(
            arm="native",
            window_s=600.0,
            coverage_s=600.0,
            t0_s=60.0,
            delay_s=12.5,
            entry_wait_s=1.5,
            n_vehicles=3,
            collision=False,
            min_ttc_s=None,
            has_closing_pair=False,
            ttc_lt3_exposure_s=0.0,
            hard_brake_s=0.0,
            n_teleports=0,
            realised_plan={"c0": "native"},
            is_constant=True,
            realised_source="declared",
            invalid_reasons=[],
            cav_ids_all=("c0", "c1"),
        )
        payload = json.loads(json.dumps(branch_outcome_to_dict(outcome)))
        assert branch_outcome_from_dict(payload) == outcome


# ============================================================== PL-02a-C2
# Trimmed REAL excerpt: the chunk-1 BASE arm outcome record of
# w8407m2400p02t360__base (wm44 base schema). Its ``collisions`` field is a
# scalar COUNT (0 = no collision), NOT an id list -- the old reader wrapped
# any non-None scalar as a one-element id list and flagged every base run
# collision=True ("0" became a collision id) even though the run's physics
# rows carry no colliding_ids occurrence anywhere and its min closing TTC
# is 2.20 s. Regression: a zero count must NOT flag a collision.
CHUNK1_BASE_OUTCOME_EXCERPT = {
    "schema": "wm44_base_outcome/1",
    "run": "w8407m2400p02t360__base",
    "window_s": [360.0, 960.0],
    "hard_brake_seconds": 209.0,
    "collisions": 0,
    "declared_gaps": [
        "ttc_below_3_s",
        "min_ttc_s",
        "teleports (frozen base loop exposes no per-tick teleport hook; see registration)",
    ],
}


class TestOutcomeCollisionCount:
    """PL-02a-C2: the outcome-record ``collisions`` scalar COUNT forms."""

    def _model(self, tmp_path, outcome):
        runs_root = tmp_path / "runs"
        rows = steady_rows(T0, T0 + 100.0, ("v1", "v2"))
        write_run(runs_root, "r__base", arm="base", is_base=True, rows=rows, outcome=outcome)
        write_run(runs_root, "r__native", arm="native", is_base=False, rows=rows)
        index = CampaignIndex(runs_root).build()
        return RecordedBranchWorldModel(
            index.roots["r__base"],
            layout=Layout(),
            delay_provider=lambda *a, **k: (0.0, 0.0, 2, {}),
            delay_provider_name="builtin",
        )

    def test_real_base_outcome_zero_count_is_no_collision(self, tmp_path):
        """Trimmed real excerpt (w8407m2400p02t360__base): count 0 -> no
        collision, cell stays valid."""
        outcome = self._model(tmp_path, dict(CHUNK1_BASE_OUTCOME_EXCERPT)).outcomes["base"]
        assert outcome.collision is False
        assert outcome.cell_valid
        assert outcome.invalid_reasons == []

    def test_positive_count_fails_closed(self, tmp_path):
        """A POSITIVE collision count means collisions with unrecorded ids:
        collision=True + a precise fail-closed reason."""
        excerpt = dict(CHUNK1_BASE_OUTCOME_EXCERPT)
        excerpt["collisions"] = 2
        outcome = self._model(tmp_path, excerpt).outcomes["base"]
        assert outcome.collision is True
        assert not outcome.cell_valid
        assert any("collision count 2 > 0" in reason for reason in outcome.invalid_reasons)

    def test_id_list_still_flags(self, tmp_path):
        """wm44 branch-outcome id lists keep flagging (old behaviour kept)."""
        excerpt = dict(CHUNK1_BASE_OUTCOME_EXCERPT)
        excerpt["collisions"] = ["v9"]
        outcome = self._model(tmp_path, excerpt).outcomes["base"]
        assert outcome.collision is True
        assert not outcome.cell_valid
        assert any("collision in window" in reason for reason in outcome.invalid_reasons)


class TestTrainOnlyGuard:
    """PL-02a-C2: --roots/--scan-prefixes (smoke/dev) select TRAIN only."""

    def test_non_train_worlds_from_plan(self, campaign):
        runner = _runner_module()

        _runs_root, plan_path, _ = campaign
        assert runner.non_train_worlds(plan_path, roots_scope=["w1__base"]) == []
        # bare world labels are accepted via their <world>__base run
        assert runner.non_train_worlds(plan_path, roots_scope=["w1"]) == []
        # offenders are reported at WORLD granularity
        assert runner.non_train_worlds(plan_path, roots_scope=["w2__base"]) == [
            "w2 (HOLDOUT)"
        ]
        assert runner.non_train_worlds(plan_path, scan_prefixes=["w2"]) == ["w2 (HOLDOUT)"]
        assert runner.non_train_worlds(plan_path, scan_prefixes=["w1__"]) == []

    def test_scoped_run_refuses_holdout(self, campaign):
        runner = _runner_module()

        runs_root, plan_path, out_dir = campaign
        with pytest.raises(SystemExit, match="HOLDOUT"):
            runner.run(
                runs_root, plan_path, out_dir, roots_scope=["w2__base"], cache_dir=out_dir / "cache"
            )

    def test_scoped_run_allow_non_train_logs_and_proceeds(self, campaign, capsys):
        runner = _runner_module()

        runs_root, plan_path, out_dir = campaign
        runner.run(
            runs_root,
            plan_path,
            out_dir,
            roots_scope=["w2__base"],
            allow_non_train=True,
            cache_dir=out_dir / "cache",
        )
        out = capsys.readouterr().out
        assert "[allow-non-train]" in out
        assert "w2 (HOLDOUT)" in out

    def test_unscoped_run_is_not_gated(self, campaign):
        """A full (unscoped) run is not a smoke/dev invocation: no gate."""
        runner = _runner_module()

        runs_root, plan_path, out_dir = campaign
        payload = runner.run(runs_root, plan_path, out_dir, cache_dir=out_dir / "cache")
        assert payload["per_root"]


class TestOnsetBeforeT0:
    """PL-02a-C2: per-world t0 onset buckets; a pre-t0 onset is its own
    category (chunk-1 bases record the pre-t0 prefix; the old smoke root's
    onset 131.5 s precedes its t0 = 360 s and was lumped into <=60 s)."""

    @staticmethod
    def _record(t0_s, onset_s):
        return {
            "usable": True,
            "root": "w__base",
            "meta": {"split": "TRAIN", "penetration": 0.02, "loading_veh_h": 2400},
            "t0_s": t0_s,
            "window_s": 600.0,
            "onset_s": onset_s,
            "arms": {},
            "decisions": {},
            "native_delay_s": 100.0,
            "best_safe_delay_s": None,
            "best_safe_delay_s_absolute": None,
            "anything_beats_native_safely": False,
            "anything_beats_native_safely_absolute": False,
            "integrity": {"present": False, "ok": True, "checks": {}},
        }

    def test_onset_before_t0_is_own_category(self):
        runner = _runner_module()

        summary = runner.summarise([self._record(360.0, 131.5)], PlannerConfig())
        assert "onset before t0" in summary["per_onset"]
        assert "onset<=60s" not in summary["per_onset"]  # no longer lumped in

    def test_onset_after_t0_keeps_age_buckets(self):
        runner = _runner_module()

        summary = runner.summarise(
            [self._record(360.0, 420.0), self._record(360.0, 9999.0)], PlannerConfig()
        )
        assert set(summary["per_onset"]) == {"onset<=60s", "onset>180s"}

    def test_no_onset_bucket(self):
        runner = _runner_module()

        summary = runner.summarise([self._record(360.0, None)], PlannerConfig())
        assert set(summary["per_onset"]) == {"no onset"}


class TestReportArmTables:
    """PL-02a-C2: the safety table covers EVERY arm (fixed1/fixed2 too);
    schedule-only switching arms go to their own headroom-only table."""

    @staticmethod
    def _arm(is_constant=True, safe_planner=True):
        return {
            "delay_s": 100.0,
            "min_ttc_s": 2.2,
            "has_closing_pair": True,
            "ttc_lt3_exposure_s": 1.0,
            "hard_brake_s": 2.0,
            "collision": False,
            "n_teleports": 0,
            "is_constant": is_constant,
            "safe_outcome": True,
            "safe_planner": safe_planner,
            "saving_vs_native_s": 5.0,
            "saving_builtin_vs_native_s": 4.0,
        }

    def test_all_arms_rendered_and_switching_separated(self):
        runner = _runner_module()

        record = {
            "usable": True,
            "root": "w__base",
            "meta": {"split": "TRAIN", "penetration": 0.02, "loading_veh_h": 2400},
            "t0_s": 240.0,
            "window_s": 600.0,
            "onset_s": None,
            "arms": {
                "base": self._arm(),
                "native": self._arm(safe_planner=None),
                "pi": self._arm(),
                "fixed1": self._arm(),
                "fixed2": self._arm(),
                "switching1": self._arm(is_constant=False, safe_planner=None),
            },
            "decisions": {},
            "native_delay_s": 105.0,
            "best_safe_arm": "pi",
            "best_safe_delay_s": 100.0,
            "best_safe_arm_absolute": "pi",
            "best_safe_delay_s_absolute": 100.0,
            "anything_beats_native_safely": True,
            "anything_beats_native_safely_absolute": True,
            "provider_signs_agree": True,
            "integrity_ok": True,
            "integrity": {"present": False, "ok": True, "checks": {}},
        }
        summary = runner.summarise([record], PlannerConfig())
        from types import SimpleNamespace

        report = runner.render_report(
            runner.TEMPLATE_PATH.read_text(encoding="utf-8"),
            "runs",
            "plan",
            SimpleNamespace(roots={}, skipped=[]),
            [record],
            summary,
            "csv",
        )
        switching_section = report.split("### Switching arms")[1].split("###")[0]
        # constant arms (incl. fixed1/fixed2) are in the safety table...
        for arm_name in ("base", "fixed1", "fixed2", "native", "pi"):
            row = [line for line in report.splitlines() if f"| w__base | {arm_name} |" in line]
            assert row, arm_name
            # ... and NOT in the switching section
            assert not any(line in switching_section for line in row), arm_name
        # the switching arm appears ONLY in the headroom-only section
        rows = [line for line in report.splitlines() if "| w__base | switching1 |" in line]
        assert len(rows) == 1
        assert rows[0] in switching_section


# ---------------------------------------------------------------------------
# PL-02a chunk-1 FULL: collision-pair scan + owner-tally reconciliation
# (BRIEF_PL02a.md FULL CHUNK-1 RUN requirements; the reader rule is
# validated per-pair against the 22-row COLLISION_TALLY_1320Z.txt)
# ---------------------------------------------------------------------------


def colliding_rows(t0, entries, *, colliding, cav_ids=()):
    """Rows around ``t0`` where one tick carries non-empty colliding_ids.

    ``entries``: (vid, lane, pos, speed, accel, vtype) per vehicle; every
    row carries every vehicle (the scanner reads states from the same row
    as the colliding_ids)."""
    rows = []
    for offset in (-1.0, -0.5, 0.0, 0.5):
        t = t0 + offset
        vehicles = {
            vid: {
                "road_id": "r0",
                "lane_id": lane,
                "lane_position_m": pos,
                "speed_mps": speed,
                "acceleration_mps2": accel,
                "type_id": vtype,
            }
            for vid, lane, pos, speed, accel, vtype in entries
        }
        rows.append(
            {
                "time_s": t,
                "vehicles": vehicles,
                "colliding_ids": (list(colliding) if offset == 0.0 else []),
                "cav_ids": list(cav_ids),
                "arrived_ids": [],
            }
        )
    return rows


@pytest.fixture()
def collisions_module():
    return pytest.importorskip(
        "output.pl02a_offline_v1.collisions",
        reason="PL-02a chunk-1 collision stage lives in the planner worktree only",
    )


def _one_world_plan(path, base_run, branches, t0=480.0, duration=None):
    write_world_plan(
        path,
        [
            {
                "world": "w1",
                "seed": 1,
                "loading_veh_h": 2400,
                "penetration": 0.4,
                "split": "TRAIN",
                "t0_s": t0,
                "base_arm": {
                    "arm": "base",
                    "run": base_run,
                    "controller_name": "damping",
                    "saves_state": {"at_s": t0},
                    "duration_s": (duration if duration is not None else t0 + 600.0),
                },
                "branches": branches,
            }
        ],
    )


TWO_CAR_ENTRIES = [
    ("a", "l0", 50.0, 0.0, 0.0, "aggressive_car"),
    ("b", "l0", 40.0, 3.0, -3.0, "cautious_car"),
]


def _write_run_rows(runs_root, name, rows, *, complete=True):
    run_dir = runs_root / name
    (run_dir / "run").mkdir(parents=True, exist_ok=True)
    (run_dir / "run" / "physics_steps.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    if complete:
        (run_dir / "RUN_RECEIPT.json").write_text(
            json.dumps({"run": name, "exit_status": "complete"})
        )


class TestCollisionPairScan:
    def test_pair_fields_follower_behind_and_kind_from_cav_ids(
        self, tmp_path, collisions_module
    ):
        entries = [
            ("lead", "l0", 1958.8, 0.12, -0.25, "aggressive_car"),
            ("fol", "l0", 1953.9, 3.82, -3.0, "cautious_car"),
        ]
        rows = colliding_rows(
            480.0, entries, colliding=["lead", "fol"], cav_ids=["fol"]
        )
        path = tmp_path / "physics_steps.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        pairs, last_t = collisions_module.scan_run_pairs(
            path, t0_s=480.0, layout=Layout()
        )
        assert last_t == 480.5
        assert len(pairs) == 1
        pair = pairs[0]
        assert (pair["follower_id"], pair["leader_id"]) == ("fol", "lead")
        assert pair["t_s"] == 480.0
        assert pair["follower_type"] == "cautious_car"
        assert pair["leader_type"] == "aggressive_car"
        assert pair["follower_speed_mps"] == pytest.approx(3.82)
        assert pair["follower_accel_mps2"] == pytest.approx(-3.0)
        assert pair["leader_speed_mps"] == pytest.approx(0.12)
        assert pair["lane_id"] == "l0"
        assert pair["follower_position_m"] == pytest.approx(1953.9)
        # CAV membership comes from the row's cav_ids, NOT the vtype
        assert pair["pair_kind"] == "CAV-behind-human"

    def test_pair_kind_cav_behind_cav_when_leader_is_cav(
        self, tmp_path, collisions_module
    ):
        rows = colliding_rows(
            100.0, TWO_CAR_ENTRIES, colliding=["a", "b"], cav_ids=["a", "b"]
        )
        path = tmp_path / "physics_steps.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        pairs, _ = collisions_module.scan_run_pairs(path, t0_s=60.0, layout=Layout())
        assert pairs[0]["pair_kind"] == "CAV-behind-CAV"

    def test_id_pair_reports_first_tick_only(self, tmp_path, collisions_module):
        rows = colliding_rows(
            100.0, TWO_CAR_ENTRIES, colliding=["a", "b"], cav_ids=["b"]
        )
        path = tmp_path / "physics_steps.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        pairs, _ = collisions_module.scan_run_pairs(path, t0_s=60.0, layout=Layout())
        assert len(pairs) == 1
        assert pairs[0]["t_s"] == 100.0  # first tick, not the last

    def test_multi_vehicle_lane_groups_into_consecutive_pairs(
        self, tmp_path, collisions_module
    ):
        entries = TWO_CAR_ENTRIES + [
            ("mid", "l0", 45.0, 0.0, 0.0, "cautious_car"),
            ("other_lane", "l1", 99.0, 0.0, 0.0, "cautious_car"),
        ]
        rows = colliding_rows(
            100.0,
            entries,
            colliding=["a", "b", "mid", "other_lane"],
            cav_ids=["b", "mid", "other_lane"],
        )
        path = tmp_path / "physics_steps.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        pairs, _ = collisions_module.scan_run_pairs(path, t0_s=60.0, layout=Layout())
        by_ids = {(p["follower_id"], p["leader_id"]) for p in pairs}
        # consecutive pairs front-first per lane; a lone colliding vehicle
        # on its own lane has no leader and forms no pair
        assert by_ids == {("b", "mid"), ("mid", "a")}


class TestCollisionClassification:
    def test_pre_t0_vs_in_window_tally_rule(self, tmp_path, collisions_module):
        """The tally rule: t < t0 is pre_t0 (base prefix); t >= t0 in-window."""
        module = collisions_module
        runs_root = tmp_path / "runs"
        runs_root.mkdir()
        plan = tmp_path / "plan.json"
        branches = [
            {
                "arm": name,
                "run": f"w1__{name}",
                "restores_state": {"from_run": "w1__base", "at_s": 100.0},
            }
            for name in ("early", "boundary", "late")
        ]
        _one_world_plan(plan, "w1__base", branches, t0=100.0)
        for name, tick_t in (("early", 99.5), ("boundary", 100.0), ("late", 150.0)):
            rows = colliding_rows(
                tick_t, TWO_CAR_ENTRIES, colliding=["a", "b"], cav_ids=["b"]
            )
            _write_run_rows(runs_root, f"w1__{name}", rows)
        pairs, reports, unplanned = collisions_module.scan_runs_root(
            runs_root, plan, layout=Layout()
        )
        assert unplanned == []
        by_run = {pair.run: pair.classification for pair in pairs}
        assert by_run == {
            "w1__early": module.PAIR_PRE_T0,
            "w1__boundary": module.PAIR_IN_WINDOW,  # t == t0 counts in-window
            "w1__late": module.PAIR_IN_WINDOW,
        }
        assert all(report.crash_note is None for report in reports)

    def test_analysis_window_flag_uses_root_window(self, tmp_path, collisions_module):
        runs_root = tmp_path / "runs"
        runs_root.mkdir()
        plan = tmp_path / "plan.json"
        _one_world_plan(plan, "w1__base", [])
        rows = colliding_rows(
            150.0, TWO_CAR_ENTRIES, colliding=["a", "b"], cav_ids=["b"]
        )
        _write_run_rows(runs_root, "w1__base", rows)
        pairs, _, _ = collisions_module.scan_runs_root(
            runs_root, plan, layout=Layout(), analysis_windows={"w1": (100.5, 600.0)}
        )
        assert pairs[0].in_analysis_window is True
        outside, _, _ = collisions_module.scan_runs_root(
            runs_root, plan, layout=Layout(), analysis_windows={"w1": (200.0, 600.0)}
        )
        assert outside[0].in_analysis_window is False
        unknown, _, _ = collisions_module.scan_runs_root(runs_root, plan, layout=Layout())
        assert unknown[0].in_analysis_window is None


class TestCrashedRunScan:
    def _plan_with_branch(self, tmp_path):
        plan = tmp_path / "plan.json"
        _one_world_plan(
            plan,
            "w1__base",
            [
                {
                    "arm": "fixed",
                    "run": "w1__fixed2",
                    "suffix_controller": "mixed-assignment",
                    "duration_s": 600.0,
                    "mode": "restore_suffix",
                    "restores_state": {"from_run": "w1__base", "at_s": 420.0},
                }
            ],
            t0=420.0,
            duration=1020.0,
        )
        return plan

    def test_no_receipt_run_with_collision_is_reported_with_reason(
        self, tmp_path, collisions_module
    ):
        module = collisions_module
        runs_root = tmp_path / "runs"
        runs_root.mkdir()
        rows = colliding_rows(447.0, TWO_CAR_ENTRIES, colliding=["a", "b"], cav_ids=["b"])
        _write_run_rows(runs_root, "w1__base", rows)
        crashed_rows = colliding_rows(
            696.5, TWO_CAR_ENTRIES, colliding=["a", "b"], cav_ids=["b"]
        )
        _write_run_rows(runs_root, "w1__fixed2", crashed_rows, complete=False)
        pairs, reports, _ = module.scan_runs_root(
            runs_root, self._plan_with_branch(tmp_path), layout=Layout()
        )
        by_run = {report.run: report for report in reports}
        crashed = by_run["w1__fixed2"]
        assert crashed.receipt_status == "missing"
        assert crashed.n_pairs == 1
        assert "physics ends at 697" in crashed.crash_note
        assert "short of the planned 1020" in crashed.crash_note
        # the pair is still counted (a collision never disappears through
        # exclusion -- brief rule)
        crashed_pairs = [pair for pair in pairs if pair.run == "w1__fixed2"]
        assert len(crashed_pairs) == 1
        assert crashed_pairs[0].classification == module.PAIR_IN_WINDOW
        assert by_run["w1__base"].crash_note is None

    def test_no_physics_run_gets_reason(self, tmp_path, collisions_module):
        runs_root = tmp_path / "runs"
        (runs_root / "w1__base").mkdir(parents=True)
        (runs_root / "w1__fixed2").mkdir(parents=True)  # created, nothing written
        _, reports, _ = collisions_module.scan_runs_root(
            runs_root, self._plan_with_branch(tmp_path), layout=Layout()
        )
        crashed = {report.run: report for report in reports}["w1__fixed2"]
        assert crashed.n_pairs == 0
        assert "no receipt and no physics_steps.jsonl" in crashed.crash_note

    def test_summary_json_involvements_cross_check(self, tmp_path, collisions_module):
        runs_root = tmp_path / "runs"
        runs_root.mkdir()
        rows = colliding_rows(447.0, TWO_CAR_ENTRIES, colliding=["a", "b"], cav_ids=["b"])
        _write_run_rows(runs_root, "w1__base", rows)
        summary = runs_root / "w1__base" / "run" / "summary.json"
        summary.write_text(json.dumps({"totals": {"collision_involvements": 2}}))
        _, reports, _ = collisions_module.scan_runs_root(
            runs_root, self._plan_with_branch(tmp_path), layout=Layout()
        )
        assert reports[0].summary_involvements == 2
        assert reports[0].involvements_match is True

    def test_no_collision_run_with_summary_does_not_crash(
        self, tmp_path, collisions_module
    ):
        """Regression: the involvements cross-check crashed on runs with no
        collisions (comprehension scoping) and mis-counted distinct ids for
        multi-pair runs."""
        runs_root = tmp_path / "runs"
        runs_root.mkdir()
        plan = self._plan_with_branch(tmp_path)
        rows = colliding_rows(500.0, TWO_CAR_ENTRIES, colliding=["a", "b"], cav_ids=["b"])
        for row in rows:
            row["colliding_ids"] = []  # the run never collides
        _write_run_rows(runs_root, "w1__base", rows)
        summary = runs_root / "w1__base" / "run" / "summary.json"
        summary.write_text(json.dumps({"totals": {"collision_involvements": 0}}))
        pairs, reports, _ = collisions_module.scan_runs_root(
            runs_root, plan, layout=Layout()
        )
        assert pairs == []
        assert reports[0].summary_involvements == 0
        assert reports[0].involvements_match is True  # 0 involvements, 0 pairs

        # a second run colliding in TWO different pairs: 4 distinct ids
        runs_root2 = tmp_path / "runs2"
        runs_root2.mkdir()
        entries = TWO_CAR_ENTRIES + [
            ("c", "l1", 70.0, 0.0, 0.0, "aggressive_car"),
            ("d", "l1", 60.0, 2.0, -3.0, "cautious_car"),
        ]
        both = []
        for tick_t, colliding in ((500.0, ["a", "b"]), (600.0, ["c", "d"])):
            both.extend(colliding_rows(tick_t, entries, colliding=colliding, cav_ids=["b", "d"]))
        _write_run_rows(runs_root2, "w1__base", both)
        summary2 = runs_root2 / "w1__base" / "run" / "summary.json"
        summary2.write_text(json.dumps({"totals": {"collision_involvements": 4}}))
        pairs2, reports2, _ = collisions_module.scan_runs_root(
            runs_root2, plan, layout=Layout()
        )
        assert len(pairs2) == 2
        assert reports2[0].involvements_match is True  # 4 == 4 distinct ids


class TestTallyReconciliation:
    TALLY = (
        "# comment\n"
        "w1__base | p40 | t0=480 | t=447.0 | pre_t0 | CAV-behind-CAV "
        "| follower b cautious_car v=3.82 a=-3.00 | leader a aggressive_car v=0.00 "
        "| l0 1953.9\n"
        "w1__base | p40 | t0=480 | t=1027.5 | in_window | CAV-behind-CAV "
        "| follower b cautious_car v=1.40 a=-3.00 | leader a aggressive_car v=0.00 "
        "| l0 1953.9\n"
    )

    def _scan_one(self, module, tmp_path, tag, tick_t, follower_v, leader_v):
        runs_root = tmp_path / f"runs_{tag}"
        runs_root.mkdir()
        entries = [
            ("a", "l0", 1958.8, leader_v, 0.0, "aggressive_car"),
            ("b", "l0", 1953.9, follower_v, -3.0, "cautious_car"),
        ]
        rows = colliding_rows(tick_t, entries, colliding=["a", "b"], cav_ids=["a", "b"])
        _write_run_rows(runs_root, "w1__base", rows)
        plan = tmp_path / f"plan_{tag}.json"
        _one_world_plan(plan, "w1__base", [])
        pairs, _, _ = module.scan_runs_root(runs_root, plan, layout=Layout())
        assert len(pairs) == 1
        return pairs[0]

    def test_matched_missing_and_explained_extra(self, tmp_path, collisions_module):
        module = collisions_module
        tally_path = tmp_path / "tally.txt"
        tally_path.write_text(self.TALLY)
        tally = module.parse_tally(tally_path)
        assert len(tally) == 2
        assert tally[0].key == ("w1__base", 447.0, "b", "a")
        first = self._scan_one(module, tmp_path, "a", 447.0, 3.82, 0.0)
        second = self._scan_one(module, tmp_path, "b", 1027.5, 1.40, 0.0)
        fresh_extra = self._scan_one(module, tmp_path, "c", 1500.0, 2.0, 0.1)
        stale_extra = self._scan_one(module, tmp_path, "d", 1600.0, 2.0, 0.1)
        all_pairs = [first, second, fresh_extra, stale_extra]
        tally_mtime = 1_000_000.0
        fresh_extra.data_mtime = tally_mtime + 10.0
        stale_extra.data_mtime = tally_mtime - 10.0
        recon = module.reconcile(all_pairs, tally, tally_mtime=tally_mtime)
        assert len(recon.matched) == 2
        assert recon.field_mismatches == []
        assert recon.missing == []
        assert [pair.key for pair in recon.extra_explained] == [
            ("w1__base", 1500.0, "b", "a")
        ]
        assert [pair.key for pair in recon.extra_unexplained] == [
            ("w1__base", 1600.0, "b", "a")
        ]
        assert recon.ok is False  # the unexplained extra fails the gate
        assert first.tally_note == "matched tally"
        assert "explained" in fresh_extra.tally_note
        assert "UNEXPLAINED" in stale_extra.tally_note

    def test_tally_row_not_found_is_missing(self, tmp_path, collisions_module):
        module = collisions_module
        tally_path = tmp_path / "tally.txt"
        tally_path.write_text(self.TALLY)
        tally = module.parse_tally(tally_path)
        first = self._scan_one(module, tmp_path, "only", 447.0, 3.82, 0.0)
        recon = module.reconcile([first], tally, tally_mtime=None)
        assert len(recon.matched) == 1
        assert [row.key for row in recon.missing] == [("w1__base", 1027.5, "b", "a")]

    def test_field_mismatch_is_reported(self, tmp_path, collisions_module):
        module = collisions_module
        tally_path = tmp_path / "tally.txt"
        tally_path.write_text(self.TALLY.replace("v=3.82", "v=9.99"))
        tally = module.parse_tally(tally_path)
        first = self._scan_one(module, tmp_path, "bad", 447.0, 3.82, 0.0)
        recon = module.reconcile([first], tally, tally_mtime=None)
        assert len(recon.matched) == 1
        assert len(recon.field_mismatches) == 1
        assert "follower v" in recon.field_mismatches[0]

    def test_duplicate_scan_key_raises(self, tmp_path, collisions_module):
        first = self._scan_one(collisions_module, tmp_path, "x", 447.0, 3.82, 0.0)
        second = self._scan_one(collisions_module, tmp_path, "y", 447.0, 3.82, 0.0)
        with pytest.raises(ValueError, match="duplicate collision pair key"):
            collisions_module.reconcile([first, second], [])


def pair_record(module, **overrides):
    """A representative matched pair (w1__base, pre-t0, CAV-behind-CAV)."""
    fields = dict(
        run="w1__base",
        root="w1",
        arm="base",
        is_base=True,
        receipt_status="complete",
        t_s=447.0,
        t0_s=480.0,
        classification=module.PAIR_PRE_T0,
        follower_id="b",
        follower_type="cautious_car",
        follower_speed_mps=3.82,
        follower_accel_mps2=-3.0,
        leader_id="a",
        leader_type="aggressive_car",
        leader_speed_mps=0.0,
        lane_id="l0",
        follower_position_m=1953.9,
        pair_kind="CAV-behind-CAV",
    )
    fields.update(overrides)
    return module.CollisionPair(**fields)


class TestCollisionOutputs:
    def test_csv_rows_and_results_annotation(self, tmp_path, collisions_module):
        module = collisions_module
        pair = pair_record(module)
        report = module.RunReport(
            run="w1__base",
            root="w1",
            arm="base",
            is_base=True,
            receipt_status="complete",
            n_pairs=1,
            n_in_window=0,
            n_pre_t0=1,
            summary_involvements=2,
            involvements_match=True,
            crash_note=None,
        )
        csv_path = tmp_path / "collisions.csv"
        module.write_collisions_csv(csv_path, [pair])
        lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
        assert lines[0].startswith("run,root,arm,is_base,receipt_status,t_s")
        assert len(lines) == 2
        assert "w1__base" in lines[1] and "pre_t0" in lines[1]
        counts = {
            "n_roots_planned": 70,
            "n_roots_complete": 60,
            "by_split": {"TRAIN": 41},
        }
        results_path = tmp_path / "RESULTS.json"
        results_path.write_text(json.dumps({"schema": "pl02a_results/1", "per_root": []}))
        module.annotate_results_json(
            results_path,
            [pair],
            [report],
            module.Reconciliation(),
            counts,
            "tally",
            ["stray"],
        )
        block = json.loads(results_path.read_text(encoding="utf-8"))["chunk1_collisions"]
        assert block["pairs_total"] == 1
        assert block["pairs_pre_t0"] == 1
        assert block["tally_rows"] == 0
        assert block["pre_t0_base_collision_roots"] == ["w1"]
        assert block["roots_complete_at_run_time"]["n_roots_complete"] == 60
        assert block["unplanned_run_dirs_ignored"] == ["stray"]

    def test_complete_root_counts_needs_every_arm(self, tmp_path, collisions_module):
        runs_root = tmp_path / "runs"
        runs_root.mkdir()
        plan = tmp_path / "plan.json"
        worlds = []
        for label, runs in (
            ("w_complete", ["c__base", "c__native"]),
            ("w_partial", ["p__base", "p__native"]),
        ):
            worlds.append(
                {
                    "world": label,
                    "seed": 1,
                    "loading_veh_h": 2400,
                    "penetration": 0.4,
                    "split": "TRAIN",
                    "t0_s": 100.0,
                    "base_arm": {
                        "arm": "base",
                        "run": runs[0],
                        "saves_state": {"at_s": 100.0},
                        "duration_s": 700.0,
                    },
                    "branches": [
                        {
                            "arm": "native",
                            "run": runs[1],
                            "restores_state": {"from_run": runs[0], "at_s": 100.0},
                        }
                    ],
                }
            )
        write_world_plan(plan, worlds)
        for run in ("c__base", "c__native", "p__base"):
            (runs_root / run).mkdir(parents=True)
            (runs_root / run / "RUN_RECEIPT.json").write_text(
                json.dumps({"run": run, "exit_status": "complete"})
            )
        counts = collisions_module.complete_root_counts(runs_root, plan)
        assert counts["n_roots_planned"] == 2
        assert counts["n_roots_complete"] == 1
        assert counts["by_split"] == {"TRAIN": 1}
        assert counts["incomplete_roots"] == ["w_partial"]


class TestRenderedCollisionSections:
    def test_sections_cover_counts_reconciliation_crash_and_pre_t0(
        self, tmp_path, collisions_module
    ):
        module = collisions_module
        tally = module.TallyRow(
            run="w1__base",
            penetration="p40",
            t0_s=480.0,
            t_s=447.0,
            classification=module.PAIR_PRE_T0,
            pair_kind="CAV-behind-CAV",
            follower_id="b",
            follower_type="cautious_car",
            follower_speed_mps=3.82,
            follower_accel_mps2=-3.0,
            leader_id="a",
            leader_type="aggressive_car",
            leader_speed_mps=0.0,
            lane_id="l0",
            follower_position_m=1953.9,
        )
        pair = pair_record(module)
        crashed_pair = pair_record(
            module,
            run="w1__fixed2",
            arm="fixed2",
            is_base=False,
            receipt_status="missing",
            t_s=696.5,
            t0_s=420.0,
            classification=module.PAIR_IN_WINDOW,
            follower_id="x",
            follower_type="cautious_car",
            follower_speed_mps=3.5,
            leader_id="y",
            leader_type="normal_car",
            leader_speed_mps=0.17,
            follower_position_m=1989.9,
            pair_kind="CAV-behind-human",
        )
        reports = [
            module.RunReport(
                run="w1__base",
                root="w1",
                arm="base",
                is_base=True,
                receipt_status="complete",
                n_pairs=1,
                n_in_window=0,
                n_pre_t0=1,
            ),
            module.RunReport(
                run="w1__fixed2",
                root="w1",
                arm="fixed2",
                is_base=False,
                receipt_status="missing",
                n_pairs=1,
                n_in_window=1,
                n_pre_t0=0,
                crash_note=(
                    "receipt 'missing': physics ends at 697 s, 323 s short of "
                    "the planned 1020 s (stale, age 292 min)"
                ),
            ),
        ]
        recon = module.Reconciliation()
        recon.matched.append((tally, pair))
        sections = module.render_report_sections(
            [pair, crashed_pair],
            reports,
            recon,
            {"n_roots_planned": 70, "n_roots_complete": 60, "by_split": {"TRAIN": 41}},
            "tally.txt",
        )
        assert "1/1" in sections  # tally reconciliation fraction
        assert "| w1__base | w1 | base | True | complete | 1 | 0 | 1 |" in sections
        assert "w1__fixed2" in sections and "physics ends at 697 s" in sections
        assert "every branch of" in sections  # pre-t0 root flag note
        assert "never best-safe" in sections  # owner rule statement


class TestRootLabelMappingAndCrosscheck:
    def test_runner_root_to_plan_root(self, collisions_module):
        module = collisions_module
        assert module.runner_root_to_plan_root("w8476m2600p40t480__base") == (
            "w8476m2600p40t480"
        )
        assert module.runner_root_to_plan_root("w8476m2600p40t480") == (
            "w8476m2600p40t480"
        )

    def test_owner_rule_crosscheck_flags_disagreement(self, tmp_path, collisions_module):
        module = collisions_module
        pair = pair_record(module, in_analysis_window=True)
        results = {
            "schema": "pl02a_results/1",
            "per_root": [
                {
                    "root": "w1__base",  # runner label (world + __base)
                    "usable": True,
                    "window_start_s": 480.5,
                    "window_s": 600.0,
                    "arms": {
                        "base": {"collision": True},
                        "native": {"collision": False},
                    },
                }
            ],
        }
        results_path = tmp_path / "RESULTS.json"
        results_path.write_text(json.dumps(results))
        reports = [
            module.RunReport(
                run="w1__base",
                root="w1",
                arm="base",
                is_base=True,
                receipt_status="complete",
                n_pairs=1,
                n_in_window=1,
                n_pre_t0=0,
            )
        ]
        module.annotate_results_json(
            results_path, [pair], reports, module.Reconciliation(), {}, "tally", []
        )
        block = json.loads(results_path.read_text(encoding="utf-8"))["chunk1_collisions"]
        cross = block["owner_rule_crosscheck"]
        # base: pair in window AND collision True -> agree; native: no pair
        # AND False -> agree
        assert cross["n_arms_compared"] == 2
        assert cross["n_disagreements"] == 0

        # flip the native arm's flag: the scan contradicts the outcome
        results["per_root"][0]["arms"]["native"]["collision"] = True
        results_path.write_text(json.dumps(results))
        module.annotate_results_json(
            results_path, [pair], reports, module.Reconciliation(), {}, "tally", []
        )
        cross = json.loads(results_path.read_text(encoding="utf-8"))[
            "chunk1_collisions"
        ]["owner_rule_crosscheck"]
        assert cross["n_disagreements"] == 1
        assert cross["disagreements"][0]["arm"] == "native"

    def test_branches_of_analysed_roots_counted(self, tmp_path, collisions_module):
        module = collisions_module
        results_path = tmp_path / "RESULTS.json"
        results_path.write_text(
            json.dumps(
                {
                    "per_root": [
                        {"root": "w1__base", "usable": True, "arms": {"base": {}, "native": {}, "pi": {}}}
                    ]
                }
            )
        )
        module.annotate_results_json(
            results_path,
            [],
            [],
            module.Reconciliation(),
            {},
            "tally",
            [],
        )
        block = json.loads(results_path.read_text(encoding="utf-8"))["chunk1_collisions"]
        assert block["n_branches_of_analysed_roots"] == 2


def _final_report_module():
    """Guarded import: the PL-02a FINAL report module lives in the planner
    worktree only."""
    return pytest.importorskip(
        "output.pl02a_offline_v1.report_chunk1_final",
        reason="PL-02a FINAL report module (output/pl02a_offline_v1) "
        "lives in the planner worktree only",
    )


class TestChunk1FinalReport:
    """PL-02a FINAL chunk-1 report: the --splits scope guard (HOLDOUT
    refused without --owner-allow-holdout), TRAIN/CALIBRATION-scoped output
    (HOLDOUT ids nowhere except the single pair count), crashed constant
    arms as unsafe candidates, and the FINAL-tally reconciliation."""

    WINDOW = 20.0

    @staticmethod
    def _colliding_row(t, follower="cav_f", leader="lead"):
        return {
            "time_s": t,
            "vehicles": {
                follower: veh(
                    lane="lane0", pos=90.0, speed=3.0, accel=-3.0
                )
                | {"type_id": "cautious_car"},
                leader: veh(lane="lane0", pos=100.0, speed=0.0, accel=0.0)
                | {"type_id": "normal_car"},
            },
            "colliding_ids": [follower, leader],
            "cav_actions": {follower: {"action": "native"}},
            "cav_ids": [follower],
            "arrived_ids": [],
        }

    def _world_runs(self, runs_root, label, *, base_collide_t=None, fixed_collide_t=None,
                    native_collide_t=None, fixed_speed=20.0, fixed_arrivals=False,
                    fixed_hard_brake=False):
        schedule = {"events": [{"t_s": T0, "per_cav": {"c0": "pi-external-u"}}]}
        base_rows = steady_rows(0.0, T0 + self.WINDOW, ("v1",), cav_ids=("c0",))
        if base_collide_t is not None:
            # mid-file insert: the run keeps full coverage, the pair at
            # base_collide_t is pre-t0 when base_collide_t < T0
            base_rows = (
                [row for row in base_rows if row["time_s"] < base_collide_t]
                + [self._colliding_row(base_collide_t)]
                + [
                    row
                    for row in steady_rows(
                        base_collide_t + ROW_DT, T0 + self.WINDOW, ("v1",), cav_ids=("c0",)
                    )
                ]
            )
        write_run(
            runs_root,
            f"{label}__base",
            arm="base",
            is_base=True,
            rows=base_rows,
            rou_vehicles=[("v_late", T0)],
        )
        native_rows = steady_rows(T0, T0 + self.WINDOW, ("v1",), cav_ids=("c0",))
        if native_collide_t is not None:
            native_rows = [row for row in native_rows if row["time_s"] < native_collide_t] + [
                self._colliding_row(native_collide_t)
            ]
        write_run(
            runs_root,
            f"{label}__native",
            arm="native",
            is_base=False,
            rows=native_rows,
            branch_record={"arm": "native", "root_id": label, "branch_point_t0_s": T0, "seed": 1},
        )
        fixed_rows = steady_rows(
            T0, T0 + self.WINDOW, ("v1",), speed=fixed_speed, cav_ids=("c0",)
        )
        if fixed_hard_brake:
            # v1 logs accel -4 m/s^2 for the first 2 s: hard-brake
            # vehicle-seconds that fail the truncated relative rule at
            # tol 0 while its OWN absolute gate stays clean
            for row in fixed_rows[:4]:
                row["vehicles"]["v1"]["acceleration_mps2"] = -4.0
        if fixed_collide_t is not None:
            fixed_rows = [row for row in fixed_rows if row["time_s"] < fixed_collide_t] + [
                self._colliding_row(fixed_collide_t)
            ]
        write_run(
            runs_root,
            f"{label}__native",
            arm="native",
            is_base=False,
            rows=native_rows,
            branch_record={"arm": "native", "root_id": label, "branch_point_t0_s": T0, "seed": 1},
        )
        write_run(
            runs_root,
            f"{label}__fixed1",
            arm="fixed1",
            is_base=False,
            rows=fixed_rows,
            arrivals=[("v1", T0 + 15.0)] if fixed_arrivals else None,
            schedule=schedule,
            dispatcher=[
                {"t_s": T0 + t, "actions": {"c0": "pi-external-u"}} for t in (0.0, 5.0)
            ],
            branch_record={
                "arm": "fixed1",
                "root_id": label,
                "branch_point_t0_s": T0,
                "seed": 1,
            },
        )

    @pytest.fixture()
    def final_campaign(self, tmp_path):
        """w1 TRAIN clean (p20); w3 CALIBRATION at p40 (pre-fix label);
        w2 HOLDOUT whose BASE carries an in-window collision pair (count
        only); w4 TRAIN whose fixed1 arm crashed after an in-window
        collision (no receipt, physics kept); w5 TRAIN whose NATIVE arm
        crashed likewise (root recovered, fail-closed savings)."""
        runs_root = tmp_path / "runs"
        self._world_runs(runs_root, "w1")
        self._world_runs(runs_root, "w3")
        self._world_runs(runs_root, "w2", base_collide_t=T0 + 10.0)
        self._world_runs(runs_root, "w4", fixed_collide_t=T0 + 5.0)
        self._world_runs(
            runs_root,
            "w5",
            native_collide_t=T0 + 7.0,
            fixed_speed=25.0,
            fixed_arrivals=True,
            fixed_hard_brake=True,
        )
        self._world_runs(runs_root, "w6", base_collide_t=T0 - 5.0)  # pre-t0
        layout = Layout()
        (runs_root / "w4__fixed1" / layout.receipt_name).unlink()  # crashed
        (runs_root / "w5__native" / layout.receipt_name).unlink()  # crashed
        plan_path = tmp_path / "world_plan" / "WORLD_PLAN.json"
        write_world_plan(
            plan_path,
            [
                world_entry("w1", arms=("fixed1",)),
                world_entry("w2", split="HOLDOUT", arms=("fixed1",)),
                world_entry(
                    "w3", penetration=0.4, split="CALIBRATION", arms=("fixed1",)
                ),
                world_entry("w4", arms=("fixed1",)),
                world_entry("w5", arms=("fixed1",)),
                world_entry("w6", split="CALIBRATION", arms=("fixed1",)),
            ],
        )
        return runs_root, plan_path, tmp_path / "out"

    def _tally(self, tmp_path, rows):
        path = tmp_path / "TALLY_FINAL.txt"
        body = [
            "# test FINAL tally",
            "# run | penetration | arm | t0 | first tick | class | kind "
            "| follower | leader | receipt",
        ] + rows
        path.write_text("\n".join(body) + "\n")
        return path

    def _tally_rows(self):
        return [
            "w2__base | p20 | base | t0=60 | t=70.0 | in_window "
            "| CAV-behind-human | follower cav_f cautious_car v=3.00 a=-3.00 "
            "| leader lead normal_car v=0.00 | receipt=yes",
            "w4__fixed1 | p20 | fixed1 | t0=60 | t=65.0 | in_window "
            "| CAV-behind-human | follower cav_f cautious_car v=3.00 a=-3.00 "
            "| leader lead normal_car v=0.00 | receipt=NO",
            "w5__native | p20 | native | t0=60 | t=67.0 | in_window "
            "| CAV-behind-human | follower cav_f cautious_car v=3.00 a=-3.00 "
            "| leader lead normal_car v=0.00 | receipt=NO",
            "w6__base | p20 | base | t0=60 | t=55.0 | pre_t0 "
            "| CAV-behind-human | follower cav_f cautious_car v=3.00 a=-3.00 "
            "| leader lead normal_car v=0.00 | receipt=yes",
        ]

    def _build(self, final_campaign, tally_path, splits=("TRAIN", "CALIBRATION")):
        reporter = _final_report_module()
        runs_root, plan_path, out_dir = final_campaign
        return reporter.build_report(
            runs_root=runs_root,
            world_plan=plan_path,
            tally_path=tally_path,
            chunk_summary=None,
            out_dir=out_dir / "final",
            cache_dir=out_dir / "cache",
            splits=tuple(splits),
            layout=Layout(),
        )

    def test_resolve_splits_guard(self):
        reporter = _final_report_module()
        assert reporter.resolve_splits("TRAIN,CALIBRATION", False) == (
            "TRAIN",
            "CALIBRATION",
        )
        assert reporter.resolve_splits("train", False) == ("TRAIN",)
        assert reporter.resolve_splits("HOLDOUT", True) == ("HOLDOUT",)
        with pytest.raises(SystemExit, match="owner-allow-holdout"):
            reporter.resolve_splits("TRAIN,HOLDOUT", False)
        with pytest.raises(SystemExit, match="unknown split"):
            reporter.resolve_splits("BOGUS", False)
        with pytest.raises(SystemExit, match="at least one"):
            reporter.resolve_splits("  ", False)

    def test_main_refuses_holdout_without_owner_flag(self, final_campaign):
        reporter = _final_report_module()
        runs_root, plan_path, out_dir = final_campaign
        with pytest.raises(SystemExit, match="owner-allow-holdout"):
            reporter.main(
                [
                    "--runs-root",
                    str(runs_root),
                    "--plan",
                    str(plan_path),
                    "--out",
                    str(out_dir / "final"),
                    "--cache",
                    str(out_dir / "cache"),
                    "--tally",
                    str(out_dir / "TALLY_FINAL.txt"),
                    "--splits",
                    "TRAIN,HOLDOUT",
                ]
            )

    def test_final_report_scope_outputs_and_reconciliation(self, final_campaign, tmp_path):
        reporter = _final_report_module()
        tally_path = self._tally(tmp_path, self._tally_rows())
        results = self._build(final_campaign, tally_path)
        out_dir = final_campaign[2] / "final"
        report = (out_dir / "REPORT.md").read_text(encoding="utf-8")
        roots_csv = (out_dir / "roots.csv").read_text(encoding="utf-8")
        collisions_csv = (out_dir / "collisions.csv").read_text(encoding="utf-8")
        # HOLDOUT root ids absent everywhere except the single pair count
        assert reporter.REPORT_HOLDOUT_LINE + ": 1" in report
        assert "w2__base" not in report
        assert "w2__base" not in roots_csv
        assert "w2" not in collisions_csv
        assert results["guard"]["holdout_analysed"] is False
        assert results["collisions"]["holdout_collision_pairs_count_only"] == 1
        # TRAIN + CALIBRATION sections; in-scope roots all present
        assert "## TRAIN HEADLINE (TRAIN roots only)" in report
        assert "## CALIBRATION (reported separately; NEVER a tuning input)" in report
        assert "w1__base" in report and "w3__base" in report and "w4__base" in report
        # reconciliation matches the (4-row) tally per pair
        assert results["collisions"]["tally_matched_total"] == 4
        assert results["collisions"]["pairs_in_scope"] == 3  # + w6's pre-t0 base pair
        assert results["collisions"]["reconciliation_ok"] is True
        # the crashed arms' pairs are listed with their crash reasons
        assert "w4__fixed1" in report
        assert "crashed (no receipt)" in report
        # crashed constant arm: included, unsafe, delay unavailable
        w4 = next(r for r in results["per_root"] if r["root"] == "w4__base")
        assert w4["usable"] and "fixed1" in w4["crashed_arms"]
        assert w4["crashed_arms"]["fixed1"]["collision"] is True
        assert w4["crashed_arms"]["fixed1"]["delay_s"] is None
        assert "fixed1" not in w4["arms"]
        assert w4["best_safe_arm"] != "fixed1"
        # crashed NATIVE: the strict index dropped the root; it is RECOVERED
        # (native = unsafe candidate), and its savings basis fails closed
        w5 = next(r for r in results["per_root"] if r["root"] == "w5__base")
        assert w5["usable"] and "native" in w5["crashed_arms"]
        assert w5["crashed_arms"]["native"]["collision"] is True
        assert "native" not in w5["arms"]
        assert w5["native_delay_s"] is None
        assert w5["best_safe_arm"] == "base"  # option A pick (truncated window)
        assert w5["best_safe_delay_s"] is None
        assert w5["anything_beats_native_safely"] is False
        assert "w5__base" in results["inclusion"]["recovered_roots_without_complete_native"]
        # p40 result carries the pre-fix label (w3 is CALIBRATION p40)
        assert reporter.P40_PREFIX_NOTE in report
        w3_row = [line for line in roots_csv.splitlines() if line.startswith("w3__base")][0]
        assert "p40 pre-fix" in w3_row
        # pre-t0 base collision: flagged, listed, EXCLUDED from aggregates
        assert "w6__base" in report and "pre-t0 base collision" in report
        w6_row = [line for line in roots_csv.splitlines() if line.startswith("w6__base")][0]
        assert "pre-t0 base collision (excluded from aggregates)" in w6_row
        assert (
            "EXCLUDED from savings aggregates: 1 [pre-t0-flagged: w6__base; "
            "native-collided: (none)] -> aggregates over 1 roots" in report
        )
        # native-collided TRAIN root excluded too: w1+w4-w5 -> aggregates over 2
        assert (
            "EXCLUDED from savings aggregates: 1 [pre-t0-flagged: (none); "
            "native-collided: w5__base] -> aggregates over 2 roots" in report
        )
        assert (
            "native collided: 1 TRAIN roots - option A picks base / option B "
            "keeps the crashing native" in report
        )
        w5_row = [line for line in roots_csv.splitlines() if line.startswith("w5__base")][0]
        assert ",yes," in w5_row and ",base," in w5_row
        assert "option A picks base; option B keeps the crashing native" in w5_row
        assert "w6__base" in results["inclusion"]["pre_t0_base_collision_roots"]
        # inclusion status covers every in-scope root
        assert results["inclusion"]["roots_analysed"] == 5
        assert sorted(p.name for p in out_dir.glob("*")) == [
            "REPORT.md",
            "RESULTS.json",
            "collisions.csv",
            "roots.csv",
        ]

    def test_native_collided_decision_and_aggregate_exclusion(
        self, final_campaign, tmp_path
    ):
        """NATIVE COLLIDED: per-arm own absolute gates, relative checks
        truncated at the native crash (declared), option A picks the
        absolutely-safe arm with the smallest OWN delay while option B
        keeps the crashing native; the root is excluded from the savings
        aggregates."""
        tally_path = self._tally(tmp_path, self._tally_rows())
        results = self._build(final_campaign, tally_path)
        out_dir = final_campaign[2] / "final"
        report = (out_dir / "REPORT.md").read_text(encoding="utf-8")
        w5 = next(r for r in results["per_root"] if r["root"] == "w5__base")
        nc = w5["native_collided"]
        # declared truncation at the native crash (t=67 last row, +dt)
        assert nc["trunc_end_s"] == T0 + 7.5 and nc["trunc_window_s"] == 7.5
        assert "truncated window" in nc["declared_truncation"]
        # own absolute gates: base and fixed1 are collision-free, native is not
        assert nc["arms"]["base"]["own_absolute_gate"] is True
        assert nc["arms"]["fixed1"]["own_absolute_gate"] is True
        # the lowest-delay arm (fixed1) FAILS the truncated relative rule
        # (hard brake delta > tol 0), so option A skips it
        assert nc["arms"]["fixed1"]["own_delay_s"] < nc["arms"]["base"]["own_delay_s"]
        assert nc["arms"]["fixed1"]["rel_trunc_ok"] is False
        assert nc["arms"]["base"]["rel_trunc_ok"] is True
        assert nc["decision_a_pick"] == "base"
        # option B: every plan vetoed -> stays native, which collides
        assert nc["decision_b"] == "every plan vetoed -> stays native, which collides"
        assert w5["best_safe_arm"] == "base" and w5["best_safe_arm_option_b"] is None
        # savings vs the truncated native are UNAVAILABLE on every arm
        assert all(
            arm_record["saving_vs_native_s"] is None
            for arm_record in w5["arms"].values()
        )
        # MIXED WINDOWS: the root re-enters under option A with its own row
        assert "## MIXED WINDOWS (n = 3 = 2 primary + 1 native-collided INCLUDED under option A)" in report
        assert "NOT on the full-window scale" in report
        mixed_rows = [
            line
            for line in report.splitlines()
            if line.startswith("| w5__base | TRAIN | 20 ")
        ]
        assert len(mixed_rows) == 1 and "| base |" in mixed_rows[0]
        # the dedicated section renders gates + both decisions
        assert "## NATIVE COLLIDED" in report
        assert "saving vs native is UNAVAILABLE" in report
        assert "picks **base**" in report
        assert "truncated relative rule vs native" in report
        assert "stays native, which collides" in report
        # aggregate exclusion: TRAIN aggregates over 2 roots (w1, w4)
        assert results["train_summary"]["headroom"]["n_roots_intact"] == 2
        assert results["inclusion"]["native_collided_roots"] == ["w5__base"]

    def test_native_collided_headline_line_per_root_picks(self):
        """The headline one-liner names each root when the option A picks
        differ (regression for 'option A picks fixed1 (w8469), pi (w8473)'
        formatting)."""
        reporter = _final_report_module()

        def _record(root, pick):
            return {"root": root, "native_collided": {"decision_a_pick": pick}}

        line = reporter.native_collided_headline_line(
            [
                _record("w8469m2400p40t420__base", "fixed1"),
                _record("w8473m2700p40t540__base", "pi"),
            ],
            "TRAIN",
        )
        assert line == (
            "native collided: 2 TRAIN roots - option A picks "
            "fixed1 (w8469m2400p40t420), pi (w8473m2700p40t540) / "
            "option B keeps the crashing native"
        )
        same = reporter.native_collided_headline_line(
            [
                _record("w8469m2400p40t420__base", "fixed1"),
                _record("w8473m2700p40t540__base", "fixed1"),
            ],
            "TRAIN",
        )
        assert same == (
            "native collided: 2 TRAIN roots - option A picks fixed1 / "
            "option B keeps the crashing native"
        )

    def test_holdout_count_mismatch_fails_as_reader_bug(self, final_campaign, tmp_path):
        rows = [row for row in self._tally_rows() if not row.startswith("w2__base")]
        tally_path = self._tally(tmp_path, rows)  # HOLDOUT pair missing
        with pytest.raises(SystemExit, match="reader bug"):
            self._build(final_campaign, tally_path)

    def test_field_mismatch_reported_as_reader_bug(self, final_campaign, tmp_path):
        rows = self._tally_rows()
        rows[1] = rows[1].replace("v=3.00", "v=9.99")  # corrupt w4's pair
        tally_path = self._tally(tmp_path, rows)
        results = self._build(final_campaign, tally_path)
        out_dir = final_campaign[2] / "final"
        report = (out_dir / "REPORT.md").read_text(encoding="utf-8")
        assert "FIELD MISMATCH (reader bug)" in report
        assert results["collisions"]["reconciliation_ok"] is False
        assert results["collisions"]["field_mismatches_in_scope"]

    def test_owner_allow_holdout_analyses_holdout(self, final_campaign, tmp_path):
        tally_path = self._tally(tmp_path, self._tally_rows())
        results = self._build(final_campaign, tally_path, splits=("HOLDOUT",))
        out_dir = final_campaign[2] / "final"
        report = (out_dir / "REPORT.md").read_text(encoding="utf-8")
        assert results["guard"]["holdout_analysed"] is True
        assert "w2__base" in report  # owner-approved: HOLDOUT is analysed
        assert any(r["root"] == "w2__base" for r in results["per_root"])

    def test_report_reruns_from_cache_alone(self, final_campaign, tmp_path):
        tally_path = self._tally(tmp_path, self._tally_rows())
        first = self._build(final_campaign, tally_path)
        first_timing = first["cache"]["hits_misses_errors"]
        assert first_timing["cache_misses"] > 0  # runs computed into the cache
        # the report stage reads ONLY the cache: the re-invocation must hit
        # every entry (no recompute) and reproduce the same numbers
        second = self._build(final_campaign, tally_path)
        second_timing = second["cache"]["hits_misses_errors"]
        assert second_timing["cache_misses"] == 0 and second_timing["errors"] == 0
        assert second_timing["cache_hits"] == first_timing["cache_misses"]
        by_root = {r["root"]: r for r in first["per_root"] if r.get("usable")}
        for record in second["per_root"]:
            if record.get("usable"):
                assert record["arms"] == by_root[record["root"]]["arms"]

    def test_parse_tally_final_format_with_arm_column(self, collisions_module, tmp_path):
        """Reader-bug regression: the FINAL tally inserts the arm column
        after penetration and replaces the lane column with receipt -- the
        parser must read it (and the scan-time format unchanged)."""
        parse_tally = collisions_module.parse_tally

        final_path = tmp_path / "FINAL.txt"
        final_path.write_text(
            "# run | penetration | arm | t0 | first tick | class | kind "
            "| follower | leader | receipt\n"
            "w2__base | p20 | base | t0=60 | t=70.0 | in_window "
            "| CAV-behind-human | follower cav_f cautious_car v=3.00 a=-3.00 "
            "| leader lead normal_car v=0.00 | receipt=yes\n",
            encoding="utf-8",
        )
        rows = parse_tally(final_path)
        assert len(rows) == 1
        assert rows[0].arm == "base"
        assert rows[0].key == ("w2__base", 70.0, "cav_f", "lead")
        assert rows[0].follower_speed_mps == 3.0
        old_path = tmp_path / "OLD.txt"
        old_path.write_text(
            "# run | penetration | t0 | t | class | kind | follower | leader "
            "| lane pos\n"
            "w2__base | p20 | t0=60 | t=70.0 | in_window "
            "| CAV-behind-human | follower cav_f cautious_car v=3.00 a=-3.00 "
            "| leader lead normal_car v=0.00 | lane0 90.0\n",
            encoding="utf-8",
        )
        old_rows = parse_tally(old_path)
        assert old_rows[0].arm == ""
        assert old_rows[0].lane_id == "lane0"
        assert old_rows[0].follower_position_m == 90.0

