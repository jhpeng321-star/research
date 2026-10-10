"""Per-candidate logging (candidates.jsonl) and the zero-noise replay.

Fakes only (ZERO SUMO). Proves: (1) logging on/off leaves every decision
artifact byte-identical, (2) candidates.jsonl has one row per priced
candidate equal to the Predictions the planner used, chosen flag == the
decision, (3) the pure replay helper re-applies the selection rule to the
logged values and reproduces the logged decisions exactly.
"""

import json

import numpy as np

from tests.planner.test_closed_loop import Rig, UnvalidatedFrom30, start_plan
from traffic_shepherd.planner import PlannerConfig
from traffic_shepherd.planner.candidate_replay import (
    CANDIDATE_ROW_SCHEMA,
    UnpricedPlanError,
    group_rows_by_decision,
    prediction_from_row,
    replay_decision,
)
from traffic_shepherd.planner.closed_loop import make_parser, config_from_args
from traffic_shepherd.planner.interface import plan_key, with_actions

CAV = ("c0", "c1", "c2")
BYTE_LOGS = ("physics_steps.jsonl", "dispatcher_rows.jsonl", "decisions.jsonl", "walls.jsonl")


def scenario(tmp_path, *, name, log_candidates, run_id="cand", **over):
    """Richest fake scenario: a c0-c1 coupling (a size-2 group with flips), a
    scripted window-relative veto and an absolute (collision) veto."""
    rig = Rig(tmp_path / name, run_id=run_id, log_candidates=log_candidates, **over)
    rig.model.action_delay = {c: {"damping": -5.0, "pi-external-u": -3.0} for c in rig.cav}
    rig.model.traj_coupling_pos[("c0", "c1")] = {"damping": 10.0}
    base = start_plan(rig.cav)
    bad_window = with_actions(base, {"c1": "damping", "c0": "native"})
    rig.model.safety_script[plan_key(bad_window)] = {"ttc_lt1p5_exposure_s": 5.0}
    bad_abs = with_actions(base, {"c2": "pi-external-u"})
    rig.model.safety_script[plan_key(bad_abs)] = {"collision": True, "min_ttc_s": 0.5}
    return rig


def spy(rig):
    """Record (decision time, plans, returned predictions) of every model call."""
    calls = []
    original = rig.model.predict_many

    def wrapped(state_samples, plans, *args, **kwargs):
        plans = list(plans)
        out = original(state_samples, plans, *args, **kwargs)
        calls.append((float(state_samples[0].meta["time_s"]), plans, list(out)))
        return out

    rig.model.predict_many = wrapped
    return calls


def read_jsonl(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


class TestLoggingDoesNotChangeDecisions:
    def test_on_off_runs_are_byte_identical(self, tmp_path):
        on = scenario(tmp_path, name="on", log_candidates=True)
        off = scenario(tmp_path, name="off", log_candidates=False)
        receipt_on, receipt_off = on.run(), off.run()
        for name in BYTE_LOGS:
            assert (on.run_dir / name).read_bytes() == (off.run_dir / name).read_bytes(), name
        assert (on.run_dir / "candidates.jsonl").exists()
        assert not (off.run_dir / "candidates.jsonl").exists()
        assert receipt_on["candidate_log"]["enabled"] is True
        assert receipt_on["candidate_log"]["file"] == "candidates.jsonl"
        assert receipt_on["candidate_log"]["rows"] == len(
            read_jsonl(on.run_dir / "candidates.jsonl")
        )
        assert receipt_off["candidate_log"] == {"enabled": False, "file": None, "rows": 0}
        strip_on = {k: v for k, v in receipt_on.items() if k != "candidate_log"}
        strip_off = {k: v for k, v in receipt_off.items() if k != "candidate_log"}
        assert json.dumps(strip_on, sort_keys=True) == json.dumps(strip_off, sort_keys=True)
        # the decisions are non-trivial in this scenario
        rows = on.read_log("decisions.jsonl")
        assert any(r["switched"] for r in rows)

    def test_config_sha_ignores_the_switch(self, tmp_path):
        on = scenario(tmp_path, name="a", log_candidates=True)
        off = scenario(tmp_path, name="b", log_candidates=False)
        assert on.config.to_canonical_json() == off.config.to_canonical_json()

    def test_cli_flag_default_on_and_off(self):
        parser = make_parser()
        base = ["--out", "E:/x", "--run-id", "r", "--world", "w1", "--horizon-s", "60",
                "--predictor", "mock", "--components", "mock", "--max-sim-s", "1e9"]
        assert config_from_args(parser.parse_args(base)).log_candidates is True
        off = config_from_args(parser.parse_args(base + ["--log-candidates", "off"]))
        assert off.log_candidates is False


class TestCandidateRows:
    def test_one_row_per_priced_candidate_equal_to_used_predictions(self, tmp_path):
        rig = scenario(tmp_path, name="rows", log_candidates=True)
        calls = spy(rig)
        receipt = rig.run()
        rows = read_jsonl(rig.run_dir / "candidates.jsonl")
        decisions = rig.read_log("decisions.jsonl")
        by_decision = group_rows_by_decision(rows)
        assert sorted(by_decision) == [d["decision_index"] for d in decisions]
        assert receipt["candidate_log"]["rows"] == len(rows)
        for decision in decisions:
            t = decision["t_s"]
            priced = [(p, pr) for (tc, plans, outs) in calls if tc == t for p, pr in zip(plans, outs)]
            logged = by_decision[decision["decision_index"]]
            assert len(logged) == len(priced) == decision["n_candidates_evaluated"]
            assert [r["cand_index"] for r in logged] == list(range(len(logged)))
            for row, (plan, pred) in zip(logged, priced):
                assert row["schema"] == CANDIDATE_ROW_SCHEMA
                assert row["t_s"] == t
                assert row["actions"] == dict(plan.per_cav)
                rebuilt = prediction_from_row(row)
                for name in ("delay_vs_native_s", "plan_delay_s", "native_delay_s",
                             "delay_part_a_s", "delay_part_b_s", "collision", "min_ttc_s",
                             "ttc_lt1p5_exposure_s", "ttc_lt3_exposure_s", "hard_brake_s",
                             "native_min_ttc_s", "has_closing_pair", "future_min_ttc_q05_s",
                             "p_future_collision"):
                    want = getattr(pred, name)
                    got = getattr(rebuilt, name)
                    if want is None:
                        assert got is None, name
                    else:
                        np.testing.assert_array_equal(
                            np.asarray(got), np.asarray(want), err_msg=name
                        )
                for key in ("safety", "veto", "roles", "stage", "subject", "label", "chosen"):
                    assert key in row
            chosen = [r for r in logged if r["chosen"]]
            assert len(chosen) == 1
            assert chosen[0]["actions"] == decision["actions"]
        # veto + switch quantities are populated where the planner reached them
        assert any(r["veto"]["vetoed"] for r in rows)
        assert any(r["veto"]["rules"] for r in rows)
        assert any(r["switch"] is not None and r["switch"]["passes"] for r in rows)
        assert {r["stage"] for r in rows} >= {"probe", "individual", "group"}

    def test_gate_refusal_decision_logs_rows_without_a_chosen_flag(self, tmp_path):
        model = UnvalidatedFrom30(CAV)
        model.action_delay = {c: {"damping": -5.0} for c in CAV}
        rig = Rig(tmp_path / "refuse", run_id="ref", model=model, log_candidates=True)
        rig.run()
        rows = read_jsonl(rig.run_dir / "candidates.jsonl")
        refused = [d for d in rig.read_log("decisions.jsonl") if d["gate"]["refusal"]]
        assert refused
        for d in refused:
            mine = [r for r in rows if r["decision_index"] == d["decision_index"]]
            assert mine and not any(r["chosen"] for r in mine)


class TestReplayReproducesDecisions:
    def _run(self, tmp_path, **over):
        rig = scenario(tmp_path, name="replay", log_candidates=True, **over)
        rig.run()
        cfg = rig.last_runner._planner_config()
        rows = read_jsonl(rig.run_dir / "candidates.jsonl")
        return rig, cfg, group_rows_by_decision(rows)

    def test_zero_noise_replay_matches_logged_decisions(self, tmp_path):
        rig, cfg, by_decision = self._run(tmp_path)
        decisions = {d["decision_index"]: d for d in rig.read_log("decisions.jsonl")}
        assert len(decisions) >= 4
        n_full = 0
        for index, rows in by_decision.items():
            result = replay_decision(rows, cfg)
            logged_chosen = [r for r in rows if r["chosen"]]
            assert len(logged_chosen) == 1
            assert result.plan_key == logged_chosen[0]["plan_key"]
            assert dict(result.plan.per_cav) == decisions[index]["actions"]
            n_full += result.kind == "full"
        assert n_full >= 3

    def test_replay_with_zero_noise_hook_is_identical_and_big_noise_is_handled(self, tmp_path):
        rig, cfg, by_decision = self._run(tmp_path)
        for rows in by_decision.values():
            base = replay_decision(rows, cfg)
            zero = replay_decision(rows, cfg, noise_fn=lambda row: np.zeros((1, 1)))
            assert zero.plan_key == base.plan_key
        # noise that flips the rule either gives a decision or names the
        # unpriced plan it would need; it never crashes otherwise
        rng = np.random.default_rng(0)
        for rows in by_decision.values():
            try:
                replay_decision(rows, cfg, noise_fn=lambda row: rng.normal(0, 50.0, (1, 1)))
            except UnpricedPlanError:
                pass

    def test_cadence_and_abstain_decisions_replay_trivially(self, tmp_path):
        rig = Rig(tmp_path / "abst", run_id="abs", k_samples=3, merge_observed_fraction=0.34,
                  log_candidates=True)
        rig.run()
        cfg = rig.last_runner._planner_config()
        by_decision = group_rows_by_decision(read_jsonl(rig.run_dir / "candidates.jsonl"))
        assert by_decision
        for rows in by_decision.values():
            result = replay_decision(rows, cfg)
            assert result.kind == "trivial"
            assert result.plan_key == [r for r in rows if r["chosen"]][0]["plan_key"]


def test_planner_level_logging_is_off_by_default():
    from traffic_shepherd.planner import Planner
    from traffic_shepherd.planner.mock import MockWorldModel

    planner = Planner(MockWorldModel(CAV), PlannerConfig())
    assert planner.log_candidates is False
    assert planner.last_candidates == ()


class TestWallLifecycleReplay:
    """Wall add, assemble->hold and release-now decisions replay exactly."""

    def test_wall_add_hold_and_release_decisions_replay(self):
        from tests.planner.test_walls import NATIVE2, two_lane_setup
        from traffic_shepherd.planner import Planner
        from traffic_shepherd.planner.closed_loop import _dumps

        world, cfg, samples, wid = two_lane_setup()
        planner = Planner(world, cfg, log_candidates=True)

        def locked(phase, time_in_phase):
            return {c: {"action": "wall", "wall_id": wid, "wall_phase": phase,
                        "locked": True, "time_in_phase_s": time_in_phase}
                    for c in ("c0", "c1")}

        steps = []
        world.wall_delay[wid] = -8.0
        steps.append((0.0, NATIVE2))
        steps.append((1.0, locked("assemble", 10.0)))
        steps.append((2.0, locked("hold", 11.0)))
        chosen_walls = []
        for t, state in steps:
            if t == 2.0:
                world.wall_delay[wid] = 5.0  # hold now costs more than release
            decision = planner.decide(samples, t, controller_state=state)
            rows = [json.loads(_dumps(r)) for r in planner.last_candidates]
            for i, row in enumerate(rows):
                row["decision_index"] = int(t)
            assert rows and sum(r["chosen"] for r in rows) == 1
            result = replay_decision(rows, cfg)
            assert result.plan_key == [r for r in rows if r["chosen"]][0]["plan_key"]
            assert plan_key(result.plan) == plan_key(decision.plan)
            chosen_walls.append([(w.wall_id, w.phase) for w in decision.plan.walls])
        assert chosen_walls[0] == [(wid, "assemble")]
        assert chosen_walls[2] == [(wid, "release")]
        stages = {r["stage"] for r in rows}
        assert "wall_release" in stages
        assert any(r["assembly_bar"] is None or r["assembly_bar"]["ok"] for r in rows)
