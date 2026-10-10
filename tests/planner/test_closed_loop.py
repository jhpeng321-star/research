"""PL-04 closed-loop runner tests (fakes + the kit only, ZERO SUMO).

Covers output/planner_v1/PL04_CLOSED_LOOP_DESIGN.md end to end:

(a) the fake world runs N decision points; decisions.jsonl has one row per
    decision with the required fields; executed dispatcher commands match the
    chosen plan;
(b) every veto rule and abstain appear when triggered;
(c) an unvalidated prediction in closed_loop -> gate_refusal row + all
    non-locked CAVs native; a locked assembling wall keeps its lock;
(d) the owner gate-override FILE: valid accepted + flagged per decision and
    in the receipt with its sha256; missing/non-owner approved_by, an
    expired expires_utc and a missing or empty expires_utc refused at load;
    no override without a file;
(e) predictor_class oracle/learned in every row + the receipt; the
    aggregation helper refuses mixed classes;
(f) perfect-sim lookahead opens ONE FRESH process per branch (fake factory)
    and the master never loadStates (except on resume);
(g) the budget projection refuses over-cap configs; a run hitting the cap
    ends with exit_status budget_exhausted and a receipt;
(h) resume equivalence: straight-through vs interrupt-at-decision-k + resume
    produce byte-identical logs and an equal receipt except resumed_from;
(i) no receipt for an interrupted run;
(j) HOLDOUT/TEST worlds refused without the explicit owner flag;
(k) the write-once run dir is refused if it exists without --resume;
(l) build_real_components wires the REAL perfect-sim lookahead (master-bound
    save-only snapshotter, real branch process factory, real branch runner):
    subprocess.Popen and the traci module are FAKES, so zero SUMO starts.
(m) PL-04c: the runner FAILS CLOSED without a control write; build_real_
    components wires the REAL per-branch control write (faked sensor and
    ownership): a damping branch steps the session every control tick and
    issues commands, a native branch issues none, release_all runs even
    when the branch raises; the REQUIRED --seed reaches the master and
    every branch launch (no default for --components real).
(n) PL-04d: the REAL master path -- the master executes its current
    dispatcher plan on every control tick through the SAME shared control
    write as the branches (commands for a damping plan, none for native),
    the executed rows drain into dispatcher_rows.jsonl, release_all runs
    at normal end and on exception (finally), observe() builds the K=1
    sample from the sensor frames only (real frames/coverage/
    merge_observed per contract 8.15, ground truth absent, receipt age +
    missingness), the learned predictor fails closed without a hybrid
    builder, resume is the ONLY master loadState, and the audit test
    refuses NotImplementedError / 'placeholder' / 'marker-level' in any
    RealMasterHarness / build_real_components body.

Zero simulation: every traci surface here is a fake; the real harness is
constructed and started only over faked Popen/traci/sensor/ownership.
"""

import hashlib
import inspect
import json
import subprocess
import sys
import tempfile
from dataclasses import replace as dataclasses_replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.planner._helpers import make_base_run
from traffic_shepherd.planner import PlannerConfig, PlanCandidate, WallSpec
from traffic_shepherd.planner.closed_loop import (
    BELOW_NORMAL_PRIORITY_CLASS,
    BranchControlWrite,
    BudgetRefusedError,
    BuildContext,
    BuildFailedError,
    CAV_SPEED_MODE,
    CHECKPOINT_SCHEMA,
    ClosedLoopConfig,
    ClosedLoopRunner,
    ControlPathProofAggregationRefusedError,
    FakeMasterHarness,
    INITIAL_PLAN_ACTIONS,
    LearnedObservationUnavailableError,
    MasterBoundSnapshotter,
    NothingToResumeError,
    OverrideRefusedError,
    PathGuardError,
    RECEIPT_SCHEMA,
    RUN_PURPOSE_CONTROL_PATH_PROOF,
    RUN_PURPOSE_PLANNER,
    RealBranchProcessFactory,
    RealMasterHarness,
    RunDirExistsError,
    SessionControlWrite,
    SplitRefusedError,
    aggregate_decisions,
    build_real_components,
    classify_veto,
    config_from_args,
    initial_plan_for_roster,
    load_gate_override_file,
    main,
    make_parser,
    mixed_assignment_plan,
    project_run_cost,
    refuse_control_path_proof,
)
from traffic_shepherd.planner.dispatch import WallAssignmentError, to_dispatcher_v3
from traffic_shepherd.ownership import CommandOwnership
from traffic_shepherd.planner.interface import plan_key, with_actions
from traffic_shepherd.planner.loop import DecisionReason
from traffic_shepherd.planner.mock import MockWorldModel
from traffic_shepherd.planner.perfect_sim import (
    BranchProcessFactory,
    PerfectSimWorldModel,
    SimpleArrivalDraws,
    TraciBranchRunner,
    TraciStateSnapshotter,
)

CAV = ("c0", "c1", "c2")
TWO_CAV = ("c0", "c1")
LOGS = ("physics_steps.jsonl", "dispatcher_rows.jsonl", "decisions.jsonl", "walls.jsonl")
VALIDATED_META = {
    "future_min_ttc_validated": True,
    "future_min_ttc_validation_id": "test-validated",
    "relative_safety_validated": True,
    "relative_safety_validation_id": "test-validated",
}
UNVALIDATED_META = {
    "future_min_ttc_validated": False,
    "future_min_ttc_validation_id": None,
    "relative_safety_validated": False,
    "relative_safety_validation_id": None,
}


# ------------------------------------------------------------------ helpers
def make_model(cav=CAV, *, meta=None, model_id="mock/test"):
    model = MockWorldModel(
        cav,
        part_a_horizon_s=30.0,
        contract_version="wd01_predict/2",
        model_id=model_id,
    )
    model.meta_script.update(VALIDATED_META if meta is None else meta)
    return model


class UnvalidatedFrom30(MockWorldModel):
    """Validated v2 producer until t=30, unvalidated afterwards (test (c))."""

    def __init__(self, cav, **kwargs):
        kwargs.setdefault("contract_version", "wd01_predict/2")
        kwargs.setdefault("part_a_horizon_s", 30.0)
        super().__init__(cav, **kwargs)
        self.meta_script.update(VALIDATED_META)

    def predict_many(self, state_samples, plans, *args, **kwargs):
        if float(state_samples[0].meta["time_s"]) >= 30.0:
            self.meta_script.update(UNVALIDATED_META)
        return super().predict_many(state_samples, plans, *args, **kwargs)


class Rig:
    """Config + fake components over one shared scripted model."""

    def __init__(
        self,
        tmp_path,
        *,
        run_id="run",
        cav=CAV,
        model=None,
        harness=None,
        initial_plan=None,
        sim_seconds_fn=None,
        merge_observed_fraction=1.0,
        k_samples=1,
        **config_over,
    ):
        self.cav = tuple(cav)
        self.model = model if model is not None else make_model(self.cav)
        self.harness_proto = harness
        self.merge_observed_fraction = merge_observed_fraction
        self.k_samples = int(k_samples)
        self.harnesses = []
        self.last_runner = None
        config = dict(
            out_root=tmp_path,
            run_id=run_id,
            world="w8468chunk1train",
            horizon_s=100.0,
            warmup_s=0.0,
            decision_period_s=30.0,
            predictor="mock",
            lookahead_horizon_s=30.0,
            k_samples=self.k_samples,
            max_sim_s=1e9,
        )
        config.update(config_over)
        self.config = ClosedLoopConfig(**config)
        self.components = SimpleNamespace(
            make_harness=self._make_harness,
            make_world_model=lambda context: self.model,
            roster=self.cav,
            planner_config=PlannerConfig(margin_floor_s=0.0),
            initial_plan=initial_plan,
            sim_seconds_fn=sim_seconds_fn,
        )

    def _make_harness(self, context):
        if self.harness_proto is not None:
            harness = self.harness_proto
        else:
            harness = FakeMasterHarness(
                self.cav,
                k_samples=self.k_samples,
                merge_observed_fraction=self.merge_observed_fraction,
            )
        self.harnesses.append(harness)
        return harness

    @property
    def run_dir(self) -> Path:
        return self.config.run_dir

    def read_log(self, name: str) -> list:
        text = (self.run_dir / name).read_text(encoding="utf-8")
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    def run(self, resume=False):
        self.last_runner = ClosedLoopRunner(self.config, self.components)
        return self.last_runner.run(resume=resume)


class BombHarness(FakeMasterHarness):
    """Fake master that raises mid-run (an interrupted process)."""

    def __init__(self, *args, explode_at_s=30.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.explode_at_s = float(explode_at_s)

    def step_physics(self):
        if self.time >= self.explode_at_s:
            raise RuntimeError("interrupted mid-run (test bomb)")
        return super().step_physics()


def start_plan(cav=CAV, action="native"):
    return PlanCandidate(per_cav={c: action for c in cav})


def override_file(tmp_path, payload, name="gate_override.json"):
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ------------------------------------------------------------- (a) the loop
class TestDecisionLoopAndLogs:
    def test_n_decision_points_rows_and_matching_commands(self, tmp_path):
        rig = Rig(tmp_path, run_id="a1")
        rig.model.action_delay = {c: {"damping": -5.0} for c in rig.cav}
        receipt = rig.run()
        assert receipt["exit_status"] == "complete"
        rows = rig.read_log("decisions.jsonl")
        # horizon 100, period 30 -> decision points at 0, 30, 60, 90
        assert [r["t_s"] for r in rows] == [0.0, 30.0, 60.0, 90.0]
        assert receipt["decisions"] == len(rows) == 4
        required = (
            "t_s",
            "sample_ids",
            "coverage",
            "predictor",
            "predictor_class",
            "predictor_id",
            "predictor_sha256",
            "n_candidates_evaluated",
            "groups",
            "plan_id",
            "actions",
            "vetoes",
            "abstain",
            "switch_checks",
            "gate",
            "lookahead_sim_s",
        )
        for row in rows:
            for field_name in required:
                assert field_name in row, f"decision row missing {field_name}"
            assert row["sample_ids"] == ["k0"]
            assert row["coverage"] == [1.0]
            assert row["predictor"] == "mock"
            assert row["predictor_id"] == "mock/test"
            assert row["predictor_sha256"]
            assert row["n_candidates_evaluated"] >= 1
            assert row["gate"]["mode"] == "closed_loop"
        # decision 0 switches all CAVs to damping (scripted saving) ...
        assert rows[0]["actions"] == {c: "damping" for c in rig.cav}
        assert rows[0]["switched"] is True
        # ... and the executed dispatcher commands at t=0 match the chosen plan
        commands = {
            r["cav"]: r["command"]
            for r in rig.read_log("dispatcher_rows.jsonl")
            if r["t_s"] == 0.0
        }
        assert commands == rows[0]["actions"]
        # the world executes them: physics speeds follow the damping command
        first = rig.read_log("physics_steps.jsonl")[0]
        assert all(v["v_mps"] == 19.0 for v in first["vehicles"].values())
        # the winning switch rule's margin/agreement is logged
        assert any(
            check["agreement"] is not None and check["agreement"] >= 0.8
            for check in rows[0]["switch_checks"]
        )
        for name in LOGS:
            assert (rig.run_dir / name).exists()
        assert receipt["schema"] == RECEIPT_SCHEMA

    def test_executed_commands_match_plan_at_every_decision(self, tmp_path):
        rig = Rig(tmp_path, run_id="a2")
        receipt = rig.run()
        rows = rig.read_log("decisions.jsonl")
        dispatcher = rig.read_log("dispatcher_rows.jsonl")
        for row in rows:
            commands = {
                r["cav"]: r["command"] for r in dispatcher if r["t_s"] == row["t_s"]
            }
            assert commands == row["actions"]
        assert receipt["fallbacks"] == 0


# ---------------------------------------------- (b) veto rules and abstain
class TestVetoRulesAndAbstain:
    def test_abstain_when_merge_underobserved(self, tmp_path):
        rig = Rig(tmp_path, run_id="b1", k_samples=3, merge_observed_fraction=0.34)
        receipt = rig.run()
        rows = rig.read_log("decisions.jsonl")
        assert rows, "decisions were made"
        assert rows[0]["abstain"] is True
        assert rows[0]["merge_observed"] == [True, False, False]
        assert receipt["abstains"] == len(rows)
        assert aggregate_decisions(rows)["abstains"] == len(rows)

    def test_no_safe_action_veto_rule(self, tmp_path):
        rig = Rig(tmp_path, run_id="b2")
        base = start_plan(rig.cav)
        keys = [plan_key(base)] + [
            plan_key(with_actions(base, {c: a}))
            for c in rig.cav
            for a in ("damping", "pi-external-u")
        ]
        for key in keys:
            rig.model.safety_script[key] = {"collision": True}
        rig.run()
        rules = {v["rule"] for row in rig.read_log("decisions.jsonl") for v in row["vetoes"]}
        assert "no_safe_action" in rules

    def test_window_relative_veto_rule(self, tmp_path):
        rig = Rig(tmp_path, run_id="b3")
        # a c0->damping coupling groups c0 with c1, so the flipped plan is a
        # step-3 group candidate (singleton groups never produce flips)
        rig.model.traj_coupling_pos[("c0", "c1")] = {"damping": 10.0}
        bad = with_actions(start_plan(rig.cav), {"c0": "damping"})
        # T-A: scripted on the PRIMARY 1.5 s exposure (the config default
        # selects it for the relative rule)
        rig.model.safety_script[plan_key(bad)] = {"ttc_lt1p5_exposure_s": 5.0}
        rig.run()
        vetoes = [
            v for row in rig.read_log("decisions.jsonl") for v in row["vetoes"]
        ]
        windowed = [v for v in vetoes if v["rule"] == "window"]
        assert windowed
        assert any("window-only relative" in v["detail"] for v in windowed)

    def test_absolute_safety_veto_rule(self, tmp_path):
        rig = Rig(tmp_path, run_id="b4")
        rig.model.traj_coupling_pos[("c0", "c1")] = {"damping": 10.0}
        bad = with_actions(start_plan(rig.cav), {"c0": "damping"})
        rig.model.safety_script[plan_key(bad)] = {"collision": True, "min_ttc_s": 0.5}
        rig.run()
        rules = {v["rule"] for row in rig.read_log("decisions.jsonl") for v in row["vetoes"]}
        assert "absolute" in rules

    def test_classify_veto_covers_the_design_rule_names(self):
        def reason(detail, step="group"):
            return DecisionReason(step, "s", "c", "veto", detail=detail)

        assert classify_veto(reason("no safe action; keep current")) == "no_safe_action"
        assert classify_veto(reason("x; window-only relative: ttc_lt3_exposure_s")) == "window"
        assert classify_veto(reason("wall w; v2 assembly bar failed: median")) == "wall_bar"
        assert (
            classify_veto(reason("p_future_collision exceeds the absolute max"))
            == "future_collision"
        )
        assert classify_veto(reason("relative rule: ttc_lt3_exposure_s increase")) == "relative"
        assert classify_veto(reason("plan predicted unsafe (min TTC below floor)")) == "absolute"
        assert classify_veto(reason("simultaneous wall cap 2 reached", step="wall")) == "wall_cap"
        assert (
            classify_veto(reason("proposer feasibility filter excluded: stale", step="wall"))
            == "wall_feasibility"
        )
        assert classify_veto(reason("misc")) == "safety"


# ----------------------------------------------------- (c) gate refusal path
WALL = WallSpec(
    wall_id="wall:c0+c1@16.0",
    members=("c0", "c1"),
    speed_mps=16.0,
    phase="assemble",
    hold_s=60.0,
)
LOCKED_INITIAL_PLAN = PlanCandidate(per_cav={"c2": "damping"}, walls=(WALL,))


class TestGateRefusal:
    def test_unvalidated_prediction_gate_refusal_native_fallback_keeps_lock(self, tmp_path):
        model = UnvalidatedFrom30(CAV)
        rig = Rig(tmp_path, run_id="c1", model=model, initial_plan=LOCKED_INITIAL_PLAN)
        receipt = rig.run()
        rows = rig.read_log("decisions.jsonl")
        # decision 0 (validated) keeps the wall; refusals start at t=30
        assert not any(v["rule"] == "gate_refusal" for v in rows[0]["vetoes"])
        refusals = [
            row for row in rows if any(v["rule"] == "gate_refusal" for v in row["vetoes"])
        ]
        assert [row["t_s"] for row in refusals] == [30.0, 60.0, 90.0]
        for row in refusals:
            assert row["gate"]["refusal"], "the refusal message is logged"
            assert row["gate"]["override"] is False
            # ALL non-locked CAVs revert to native in the executed fallback
            assert row["actions"]["c2"] == "native"
            assert "c0" not in row["actions"] and "c1" not in row["actions"]
        refusal_ts = {row["t_s"] for row in refusals}
        # the locked assembling wall keeps its lock after the fallback apply
        commands = {}
        for r in rig.read_log("dispatcher_rows.jsonl"):
            if r["note"] == "gate_refusal" and r["t_s"] in refusal_ts:
                commands[r["cav"]] = r["command"]
        assert commands["c0"] == "wall-v2" and commands["c1"] == "wall-v2"
        assert commands["c2"] == "native"
        entries = rig.harnesses[-1].observe()[1]
        for member in ("c0", "c1"):
            assert entries[member]["locked"] is True
            assert entries[member]["wall_id"] == WALL.wall_id
            assert entries[member]["wall_phase"] == "assemble"
        assert entries["c2"]["locked"] is False
        assert entries["c2"]["action"] == "native"
        assert receipt["vetoes_by_rule"].get("gate_refusal") == len(refusals)
        assert receipt["gate_refusals"] == len(refusals)

    def test_first_decision_refusal_with_no_walls_all_native(self, tmp_path):
        model = make_model(meta=UNVALIDATED_META)
        rig = Rig(tmp_path, run_id="c2", model=model)
        receipt = rig.run()
        rows = rig.read_log("decisions.jsonl")
        assert rows
        for row in rows:
            assert any(v["rule"] == "gate_refusal" for v in row["vetoes"])
            assert row["actions"] == {c: "native" for c in rig.cav}
        commands = {r["cav"]: r["command"] for r in rig.read_log("dispatcher_rows.jsonl")}
        assert set(commands.values()) == {"native"}
        assert receipt["exit_status"] == "complete"


# -------------------------------------------------- (d) owner override file
class TestGateOverrideFile:
    PAYLOAD = {
        "reason": "owner-approved unvalidated probe",
        "approved_by": "owner:pl-owner",
        "scope": "both",
        "expires_utc": "2099-01-01T00:00:00+00:00",
    }

    def test_valid_override_accepted_and_logged_per_decision_and_receipt(self, tmp_path):
        path = override_file(tmp_path, self.PAYLOAD)
        loaded = load_gate_override_file(path)
        assert loaded.override.approved_by == "owner:pl-owner"
        assert loaded.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
        model = make_model(meta=UNVALIDATED_META)
        rig = Rig(tmp_path, run_id="d1", model=model, gate_override=loaded)
        receipt = rig.run()
        rows = rig.read_log("decisions.jsonl")
        assert rows
        for row in rows:
            assert row["gate"]["override"] is True
            assert row["gate"]["override_reason"] == self.PAYLOAD["reason"]
            assert row["gate"]["override_file"] == str(path)
            assert row["gate"]["override_sha256"] == loaded.sha256
            assert not any(v["rule"] == "gate_refusal" for v in row["vetoes"])
        assert receipt["overrides"]["count"] == len(rows)
        assert receipt["overrides"]["approved_by"] == "owner:pl-owner"
        assert receipt["overrides"]["sha256"] == loaded.sha256
        assert receipt["overrides"]["file"] == str(path)

    def test_scope_limited_override_still_refuses_the_other_axis(self, tmp_path):
        payload = dict(self.PAYLOAD, scope="future_ttc")
        loaded = load_gate_override_file(override_file(tmp_path, payload, "scope.json"))
        model = make_model(meta=UNVALIDATED_META)  # BOTH axes unvalidated
        rig = Rig(tmp_path, run_id="d2", model=model, gate_override=loaded)
        receipt = rig.run()
        rows = rig.read_log("decisions.jsonl")
        refusals = [r for r in rows if any(v["rule"] == "gate_refusal" for v in r["vetoes"])]
        assert refusals, "the uncovered G-2 axis must still refuse"
        assert all(r["gate"]["override"] is True for r in refusals)
        assert "relative-safety" in refusals[0]["gate"]["refusal"]
        assert receipt["overrides"]["count"] == len(refusals)

    def test_missing_approved_by_refused(self, tmp_path):
        payload = {k: v for k, v in self.PAYLOAD.items() if k != "approved_by"}
        with pytest.raises(OverrideRefusedError, match="approved_by"):
            load_gate_override_file(override_file(tmp_path, payload, "no_owner.json"))

    def test_non_owner_approved_by_refused(self, tmp_path):
        payload = dict(self.PAYLOAD, approved_by="pi-session:agent-7")
        with pytest.raises(OverrideRefusedError, match="owner:"):
            load_gate_override_file(override_file(tmp_path, payload, "pi_owner.json"))

    def test_expired_expires_utc_refused(self, tmp_path):
        payload = dict(self.PAYLOAD, expires_utc="2020-01-01T00:00:00+00:00")
        with pytest.raises(OverrideRefusedError, match="expired"):
            load_gate_override_file(override_file(tmp_path, payload, "stale.json"))

    def test_missing_expires_utc_refused(self, tmp_path):
        payload = {k: v for k, v in self.PAYLOAD.items() if k != "expires_utc"}
        with pytest.raises(OverrideRefusedError, match="expires_utc"):
            load_gate_override_file(override_file(tmp_path, payload, "no_expiry.json"))

    def test_empty_expires_utc_refused(self, tmp_path):
        payload = dict(self.PAYLOAD, expires_utc="")
        with pytest.raises(OverrideRefusedError, match="expires_utc"):
            load_gate_override_file(override_file(tmp_path, payload, "empty_expiry.json"))

    def test_missing_override_file_refused(self, tmp_path):
        with pytest.raises(OverrideRefusedError, match="no such file"):
            load_gate_override_file(tmp_path / "does_not_exist.json")

    def test_no_override_without_a_file(self, tmp_path):
        rig = Rig(tmp_path, run_id="d3")
        receipt = rig.run()
        rows = rig.read_log("decisions.jsonl")
        for row in rows:
            assert row["gate"]["override"] is False
            assert row["gate"]["override_file"] is None
        assert receipt["overrides"]["count"] == 0
        assert receipt["overrides"]["sha256"] is None
        args = _cli_args("--components", "mock")
        assert config_from_args(args).gate_override is None

    def test_cli_loads_the_override_file_only_from_the_flag(self, tmp_path):
        path = override_file(tmp_path, self.PAYLOAD, "cli.json")
        args = _cli_args("--components", "mock", "--gate-override-file", str(path))
        config = config_from_args(args)
        assert config.gate_override is not None
        assert config.gate_override.override.scope == "both"
        assert config.gate_override.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def _cli_args(*extra):
    return make_parser().parse_args(
        [
            "--out",
            "E:/rd-wt/pl16/output",
            "--run-id",
            "x",
            "--world",
            "w-ok",
            "--horizon-s",
            "100",
            "--predictor",
            "mock",
            "--max-sim-s",
            "10",
            *extra,
        ]
    )


# ------------------------------------------------- (e) predictor_class rules
class TestPredictorClassLabelling:
    def test_mock_and_learned_classes_in_rows_and_receipt(self, tmp_path):
        mock_rig = Rig(tmp_path, run_id="e-mock")
        mock_receipt = mock_rig.run()
        rows = mock_rig.read_log("decisions.jsonl")
        assert rows
        assert {row["predictor_class"] for row in rows} == {"mock"}
        assert mock_receipt["predictor_class"] == "mock"
        learned_rig = Rig(
            tmp_path,
            run_id="e-learned",
            model=make_model(model_id="learned/test"),
            predictor="learned",
        )
        learned_receipt = learned_rig.run()
        learned_rows = learned_rig.read_log("decisions.jsonl")
        assert {row["predictor_class"] for row in learned_rows} == {"learned"}
        assert learned_receipt["predictor_class"] == "learned"

    def test_aggregation_refuses_mixed_classes(self):
        mock_row = {"predictor_class": "mock", "vetoes": [], "abstain": False, "gate": {}}
        learned_row = {"predictor_class": "learned", "vetoes": [], "abstain": False, "gate": {}}
        with pytest.raises(ValueError, match="mixed predictor_class"):
            aggregate_decisions([mock_row, learned_row])
        assert aggregate_decisions([mock_row, mock_row])["decisions"] == 2
        assert aggregate_decisions([]) == {
            "decisions": 0,
            "predictor_class": None,
            "vetoes_by_rule": {},
            "abstains": 0,
            "abstain_reasons": {},
            "abs_ttc_veto_counts": {},
            "gate_refusals": 0,
            "override_decisions": 0,
            "lookahead_sim_s": 0.0,
        }


# ------------------------ (PL-04f) control-path proof: flag, labels, refusal
class TestControlPathProof:
    """PL-04f: --initial-plan {damping,pi} PRE-SETs a non-native initial plan
    on the roster at t0 and labels the run run_purpose='control_path_proof'
    (receipt + EVERY decisions.jsonl row); the aggregation helpers refuse
    such rows/receipts with the named error, so a control-path proof is
    never pooled with planner results. Code-only: fakes, zero SUMO."""

    @staticmethod
    def _native_choosing_rig(tmp_path, run_id):
        """A rig whose PLANNER keeps native at every decision (the damping /
        PI candidates look worse) -- the proof mode must still show the
        pre-set initial plan on the REAL rows."""
        rig = Rig(
            tmp_path,
            run_id=run_id,
            initial_plan=start_plan(CAV, action="damping"),
            initial_plan_action="damping",
        )
        rig.model.action_delay = {c: {"damping": 5.0, "pi-external-u": 5.0} for c in rig.cav}
        return rig

    def test_flag_labels_rows_and_receipt_while_planner_stays_native(self, tmp_path):
        rig = self._native_choosing_rig(tmp_path, "f-proof")
        receipt = rig.run()
        rows = rig.read_log("decisions.jsonl")
        assert rows
        # the planner itself chose native at every decision ...
        assert {row["actions"]["c0"] for row in rows} == {"native"}
        # ... yet EVERY row and the receipt carry the control-path-proof label
        assert {row["run_purpose"] for row in rows} == {RUN_PURPOSE_CONTROL_PATH_PROOF}
        assert receipt["run_purpose"] == RUN_PURPOSE_CONTROL_PATH_PROOF
        assert receipt["initial_plan"] == "damping"
        applied = [
            row
            for row in rig.read_log("dispatcher_rows.jsonl")
            if row.get("note") == "initial_plan"
        ]
        # the pre-set damping plan was applied to EVERY roster CAV at t0
        assert applied and {row["t_s"] for row in applied} == {0.0}
        assert {row["cav"]: row["command"] for row in applied} == {
            c: "damping" for c in rig.cav
        }

    def test_default_purpose_is_planner(self, tmp_path):
        rig = Rig(tmp_path, run_id="f-planner")
        receipt = rig.run()
        rows = rig.read_log("decisions.jsonl")
        assert rows
        assert {row["run_purpose"] for row in rows} == {RUN_PURPOSE_PLANNER}
        assert receipt["run_purpose"] == RUN_PURPOSE_PLANNER
        assert receipt["initial_plan"] == "native"

    def test_programmatic_initial_plan_without_the_flag_stays_planner(self, tmp_path):
        # the LABEL follows the CLI/config flag, not the components field
        # (the locked-wall tests' pre-set plans stay 'planner' runs)
        rig = Rig(
            tmp_path, run_id="f-prog", initial_plan=start_plan(CAV, action="damping")
        )
        receipt = rig.run()
        rows = rig.read_log("decisions.jsonl")
        assert rows
        assert {row["run_purpose"] for row in rows} == {RUN_PURPOSE_PLANNER}
        assert receipt["run_purpose"] == RUN_PURPOSE_PLANNER
        assert receipt["initial_plan"] == "native"

    def test_aggregation_refuses_proof_rows_and_receipts(self):
        proof_row = {
            "predictor_class": "mock",
            "vetoes": [],
            "abstain": False,
            "gate": {},
            "run_purpose": RUN_PURPOSE_CONTROL_PATH_PROOF,
        }
        planner_row = {"predictor_class": "mock", "vetoes": [], "abstain": False, "gate": {}}
        with pytest.raises(ControlPathProofAggregationRefusedError):
            aggregate_decisions([proof_row])
        with pytest.raises(ControlPathProofAggregationRefusedError):
            aggregate_decisions([planner_row, proof_row])  # mixing purposes
        with pytest.raises(ControlPathProofAggregationRefusedError):
            refuse_control_path_proof(
                {"schema": RECEIPT_SCHEMA, "run_purpose": RUN_PURPOSE_CONTROL_PATH_PROOF}
            )
        # the guard passes planner/legacy payloads through
        refuse_control_path_proof({"schema": RECEIPT_SCHEMA})
        refuse_control_path_proof([planner_row, {"schema": RECEIPT_SCHEMA}])
        # a run's OWN receipt aggregation still works (one run = one purpose)
        assert aggregate_decisions(
            [proof_row], accept_run_purpose=RUN_PURPOSE_CONTROL_PATH_PROOF
        )["decisions"] == 1
        # planner rows aggregate exactly as before (legacy rows: no key)
        assert aggregate_decisions([planner_row, planner_row])["decisions"] == 2

    def test_initial_plan_flag_maps_choices(self):
        assert INITIAL_PLAN_ACTIONS == {"native": None, "damping": "damping", "pi": "pi-external-u"}
        assert config_from_args(_cli_args("--components", "mock")).initial_plan_action is None
        assert config_from_args(
            _cli_args("--components", "mock", "--initial-plan", "damping")
        ).initial_plan_action == "damping"
        pi = config_from_args(_cli_args("--components", "mock", "--initial-plan", "pi"))
        assert pi.initial_plan_action == "pi-external-u"
        assert pi.run_purpose == RUN_PURPOSE_CONTROL_PATH_PROOF
        assert (
            config_from_args(_cli_args("--components", "mock")).run_purpose
            == RUN_PURPOSE_PLANNER
        )
        with pytest.raises(SystemExit):  # argparse refuses unknown choices
            make_parser().parse_args(
                ["--initial-plan", "walls", "--out", "E:/x", "--run-id", "r",
                 "--world", "w", "--horizon-s", "100", "--predictor", "mock",
                 "--max-sim-s", "10"]
            )

    def test_initial_plan_for_roster_guards(self):
        plan = initial_plan_for_roster(("c1", "c0"), "pi-external-u")
        assert plan.per_cav == {"c1": "pi-external-u", "c0": "pi-external-u"}
        with pytest.raises(ValueError):
            initial_plan_for_roster(("c0",), "native")
        with pytest.raises(BuildFailedError):
            initial_plan_for_roster((), "damping")

    def test_cli_end_to_end_proof_labels_receipt_rows_and_commands(self):
        # the E:-paths-only guard makes tmp_path (system temp) unusable here;
        # this test's transient run dir lives inside the workspace instead
        with tempfile.TemporaryDirectory(dir="E:/rd-wt/pl16/output") as scratch:
            out = str(Path(scratch).resolve())
            code = main(
                [
                    "--out", out,
                    "--run-id", "cli-proof",
                    "--world", "w-cliproof",
                    "--horizon-s", "100",
                    "--decision-period-s", "30",
                    "--predictor", "mock",
                    "--lookahead-horizon-s", "30",
                    "--max-sim-s", "100000",
                    "--components", "mock",
                    "--initial-plan", "pi",
                ]
            )
            assert code == 0
            run_dir = Path(out) / "cli-proof"
            receipt = json.loads(
                (run_dir / "RUN_RECEIPT.json").read_text(encoding="utf-8")
            )
            assert receipt["schema"] == RECEIPT_SCHEMA
            assert receipt["exit_status"] == "complete"
            assert receipt["run_purpose"] == RUN_PURPOSE_CONTROL_PATH_PROOF
            assert receipt["initial_plan"] == "pi-external-u"
            rows = [
                json.loads(line)
                for line in (run_dir / "decisions.jsonl").read_text("utf-8").splitlines()
                if line.strip()
            ]
            assert rows
            assert {row["run_purpose"] for row in rows} == {RUN_PURPOSE_CONTROL_PATH_PROOF}
            applied = [
                json.loads(line)
                for line in (run_dir / "dispatcher_rows.jsonl").read_text("utf-8").splitlines()
                if line.strip() and json.loads(line).get("note") == "initial_plan"
            ]
            assert {row["cav"]: row["command"] for row in applied} == {
                "c0": "pi-external-u",
                "c1": "pi-external-u",
            }


# --------------------------------- (f) perfect-sim lookahead, fresh processes
class _BranchSim:
    def __init__(self, t0, dt):
        self._time = float(t0)
        self._dt = float(dt)
        self.loaded = []

    def loadState(self, path):
        self.loaded.append(str(path))

    def getTime(self):
        return self._time

    def simulationStep(self):
        self._time += self._dt

    def getArrivedIDList(self):
        return []  # T-B: the control write reads arrivals (run_suffix lifecycle)

    def getStartingTeleportIDList(self):
        return ()

    def getCollidingVehiclesIDList(self):
        return ()


class _BranchVehicle:
    def __init__(self):
        self.added = []

    def add(self, rid, route, **kw):
        self.added.append((str(rid), str(route), kw))

    def getIDList(self):
        return ("c0", "c1")

    def getLeader(self, rid, range_m):
        return ("lead", 30.0)

    def getLanePosition(self, rid):
        return 12.5

    def getLaneIndex(self, rid):
        return 0

    def getSpeed(self, rid):
        return 20.0

    def getAcceleration(self, rid):
        return 0.0


class _BranchConnection:
    def __init__(self, label=None, t0=0.0, dt=0.5):
        self.label = label
        self.simulation = _BranchSim(t0, dt)
        self.vehicle = _BranchVehicle()

    def simulationStep(self):
        self.simulation.simulationStep()


class FreshProcessFactory:
    """PL-10d fake launcher: one fresh connection per start_process.

    The fake ``loadState`` cannot move the branch clock, so each fresh
    process is created at the CURRENT master time -- the emulation of
    launch + loadState landing on the snapshot time."""

    def __init__(self, t0_provider=None):
        self.started = []
        self.closed = []
        self.labels = []
        self._t0_provider = t0_provider

    def start_process(self, *, label=None):
        t0 = float(self._t0_provider()) if self._t0_provider is not None else 0.0
        connection = _BranchConnection(label, t0=t0)
        self.started.append(connection)
        self.labels.append(label)
        return connection

    def close_process(self, connection):
        assert any(connection is c for c in self.started), "closed a never-started process"
        assert not any(connection is c for c in self.closed), "a process was closed twice"
        self.closed.append(connection)


class _FakeDispatcherSession:
    def __init__(self):
        self.applied = []

    wall_records = ()

    def apply_plan(self, plan, now_s=0.0):
        self.applied.append((plan, float(now_s)))


def _no_control_write(session, now):
    """PL-04c fail-closed opt-out: these rigs' fake plan/dispatcher
    factories write no control, and the runner now refuses a non-native
    branch without a control write -- the stub declares that (the fake
    sessions have no step method, so a real write is impossible here)."""
    del session, now


class PerfectSimRig:
    """Closed-loop runner whose lookahead IS the real PerfectSimWorldModel
    over fake branch processes and the REAL guarded snapshotter."""

    def __init__(self, tmp_path, *, run_id="f1", cav=TWO_CAV, commit_s=10.0, horizon=20.0):
        self.cav = tuple(cav)
        self.factory = FreshProcessFactory(t0_provider=lambda: self.made["harness"].time)
        self.made = {}
        self.last_runner = None
        clean = tmp_path / f"clean_{run_id}.rou.xml"
        clean.write_text(
            '<routes><vType id="passenger" speedDev="0"/></routes>', encoding="utf-8"
        )
        self.clean_route = clean
        self.config = ClosedLoopConfig(
            out_root=tmp_path,
            run_id=run_id,
            world="w8468chunk1train",
            horizon_s=100.0,
            decision_period_s=30.0,
            predictor="perfect_sim",
            lookahead_horizon_s=float(horizon),
            lookahead_commit_s=float(commit_s),
            k_samples=1,
            max_sim_s=1e9,
        )
        self.components = SimpleNamespace(
            make_harness=self._make_harness,
            make_world_model=self._make_model,
            roster=self.cav,
            planner_config=PlannerConfig(margin_floor_s=0.0),
            initial_plan=None,
            sim_seconds_fn=None,
        )

    def _make_harness(self, context):
        harness = FakeMasterHarness(self.cav)
        self.made["harness"] = harness
        return harness

    def _make_model(self, context):
        harness = self.made["harness"]
        state_root = context.run_dir / "lookahead_state" / f"p{context.process_seq}"
        snapshotter = TraciStateSnapshotter(
            harness.master_api, state_root, wm34_output_root="no-such-wm34-root (vendored)"
        )
        runner = TraciBranchRunner(
            harness.master_api,
            snapshotter=snapshotter,
            cav_ids=self.cav,
            dispatcher_factory=lambda controller_state, kept: _FakeDispatcherSession(),
            plan_factory=lambda plan, plan_id, controller_state, kept: SimpleNamespace(
                plan_id=plan_id
            ),
            tick_hook=_no_control_write,
            process_factory=self.factory,
        )
        return PerfectSimWorldModel(
            snapshotter,
            SimpleArrivalDraws(),
            runner,
            part_a_horizon_s=20.0,
            contract_version="wd01_predict/2",
            model_id="perfect_sim/test",
            scenario_route_files=[str(self.clean_route)],
        )

    @property
    def run_dir(self) -> Path:
        return self.config.run_dir

    def read_log(self, name: str) -> list:
        text = (self.run_dir / name).read_text(encoding="utf-8")
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    def run(self, resume=False):
        self.last_runner = ClosedLoopRunner(self.config, self.components)
        return self.last_runner.run(resume=resume)


class TestPerfectSimLookahead:
    def test_one_fresh_process_per_branch_and_master_never_loads(self, tmp_path):
        rig = PerfectSimRig(tmp_path)
        receipt = rig.run()
        harness = rig.made["harness"]
        rows = rig.read_log("decisions.jsonl")
        assert rows, "decisions were made"
        # every decisions row carries the oracle class and the priced lookahead
        for row in rows:
            assert row["predictor"] == "perfect_sim"
            assert row["predictor_class"] == "oracle"
            assert row["lookahead_sim_s"] > 0.0
        assert receipt["predictor_class"] == "oracle"
        # ONE fresh process per (plan, draw) branch, priced exactly
        k = rig.config.k_samples
        expected_branches = sum(
            (1 + c["n_plans"]) * k for c in rig.last_runner.meter.calls
        )
        assert expected_branches > 0
        assert len(rig.factory.started) == expected_branches
        assert len({id(c) for c in rig.factory.started}) == len(rig.factory.started)
        assert rig.factory.closed == rig.factory.started  # every branch retired
        for connection in rig.factory.started:
            assert len(connection.simulation.loaded) == 1  # one loadState per branch
        # the MASTER never loadStates (no resume here)
        assert harness.simulation.loaded == []
        assert harness.load_state_calls == []
        # the lookahead sim-s on the rows equals the branch pricing
        per_decision = [row["lookahead_sim_s"] for row in rows]
        assert receipt["sim_s"]["lookahead_s"] == pytest.approx(sum(per_decision))
        assert receipt["sim_s"]["lookahead_s"] > 0.0

    def test_oracle_predictions_need_no_override(self, tmp_path):
        rig = PerfectSimRig(tmp_path, run_id="f2")
        receipt = rig.run()
        rows = rig.read_log("decisions.jsonl")
        # perfect-sim predictions are validated by construction (G-1/G-2 pass)
        for row in rows:
            assert not any(v["rule"] == "gate_refusal" for v in row["vetoes"])
            assert row["gate"]["override"] is False
        assert receipt["exit_status"] == "complete"


# --------------------------------------------------------------- (g) budget
class TestBudget:
    def test_projection_refuses_over_cap_config_before_starting(self, tmp_path):
        rig = Rig(tmp_path, run_id="g1", max_sim_s=10.0)  # the horizon alone is 100 s
        with pytest.raises(BudgetRefusedError, match="refusing to start"):
            rig.run()
        assert not rig.run_dir.exists(), "a refused run never creates the run dir"

    def test_perfect_sim_projection_includes_lookahead(self, tmp_path):
        config = ClosedLoopConfig(
            out_root=tmp_path,
            run_id="g2",
            world="w-ok",
            horizon_s=100.0,
            predictor="perfect_sim",
            lookahead_horizon_s=30.0,
            lookahead_commit_s=10.0,
            max_sim_s=1e12,
        )
        projection = project_run_cost(config, n_cavs=3, planner_config=PlannerConfig())
        assert projection["decision_points"] == 4
        assert projection["lookahead_sim_s"] > 0
        assert projection["projected_sim_s"] > projection["master_sim_s"]
        small = dataclasses_replace(config, max_sim_s=projection["projected_sim_s"] - 1.0)
        with pytest.raises(BudgetRefusedError):
            project_run_cost(small, n_cavs=3, planner_config=PlannerConfig())

    def test_cap_hit_mid_run_ends_budget_exhausted_with_receipt(self, tmp_path):
        def flat_price(n_plans, k, horizon_s):
            del n_plans, k, horizon_s
            return 25.0

        rig = Rig(tmp_path, run_id="g3", max_sim_s=220.0, sim_seconds_fn=flat_price)
        receipt = rig.run()
        assert receipt["exit_status"] == "budget_exhausted"
        assert (rig.run_dir / "RUN_RECEIPT.json").exists()
        physics = rig.read_log("physics_steps.jsonl")
        assert physics, "the run stopped after stepping the world"
        assert physics[-1]["t_s"] < rig.config.horizon_s, "the cap stopped the run early"
        assert receipt["sim_s"]["total_s"] >= rig.config.max_sim_s - 90.0
        assert receipt["sim_s"]["total_s"] <= rig.config.max_sim_s + 90.0


# -------------------------------------------------- (h) + (i) resume / crash
class TestResumeEquivalence:
    @staticmethod
    def _run_logs(run_dir: Path) -> dict:
        return {name: (run_dir / name).read_bytes() for name in LOGS}

    def test_interrupt_and_resume_gives_byte_identical_logs_and_equal_receipt(
        self, tmp_path
    ):
        # the straight run and the resumed run share run_id (so the receipts
        # are comparable) but live in separate out roots (write-once dirs)
        straight = Rig(tmp_path / "straight", run_id="h_resume")
        straight.model.action_delay = {c: {"damping": -5.0} for c in straight.cav}
        straight_receipt = straight.run()
        assert straight_receipt["exit_status"] == "complete"
        assert straight_receipt["resumed_from"] == []

        # interrupted run: bombs right after the t=30 decision checkpoint
        model_b = make_model()
        model_b.action_delay = {c: {"damping": -5.0} for c in CAV}
        bomb = BombHarness(CAV, explode_at_s=30.5)
        interrupted = Rig(tmp_path / "resumed", run_id="h_resume", model=model_b, harness=bomb)
        with pytest.raises(RuntimeError, match="interrupted"):
            interrupted.run()
        # (i) an interrupted run has NO receipt
        assert not (interrupted.run_dir / "RUN_RECEIPT.json").exists()
        ckpts = sorted((interrupted.run_dir / "checkpoints").glob("ckpt_*.json"))
        assert [p.name for p in ckpts] == ["ckpt_000000.json", "ckpt_000001.json"]
        payload = json.loads(ckpts[1].read_text(encoding="utf-8"))
        assert payload["schema"] == CHECKPOINT_SCHEMA
        assert payload["t_s"] == 30.0

        # resume: a FRESH runner process (fresh harness, fresh model) continues
        model_r = make_model()
        model_r.action_delay = {c: {"damping": -5.0} for c in CAV}
        resumed = Rig(tmp_path / "resumed", run_id="h_resume", model=model_r)
        resumed_receipt = resumed.run(resume=True)

        assert resumed_receipt["exit_status"] == "complete"
        # byte-identical logs
        assert self._run_logs(straight.run_dir) == self._run_logs(resumed.run_dir)
        # equal receipt except resumed_from
        differences = {
            key for key in straight_receipt if straight_receipt[key] != resumed_receipt[key]
        }
        assert differences == {"resumed_from"}
        assert resumed_receipt["resumed_from"] == ["ckpt_000001"]
        # the resumed master loaded its checkpoint state exactly once
        assert len(resumed.harnesses) == 1
        assert resumed.harnesses[0].load_state_calls == [
            payload["master_state"]["saved_state_dir"]
        ]
        assert resumed.harnesses[0].simulation.loaded == [
            payload["master_state"]["saved_state_dir"]
        ]

    def test_checkpoint_captures_python_state_and_log_offsets(self, tmp_path):
        rig = Rig(tmp_path, run_id="h2")
        rig.model.action_delay = {c: {"damping": -5.0} for c in rig.cav}
        rig.run()
        payload = json.loads(
            (rig.run_dir / "checkpoints" / "ckpt_000001.json").read_text(encoding="utf-8")
        )
        state = payload["python_state"]
        assert state["decision_index"] == 1
        assert state["next_decision_s"] == 30.0
        assert state["current_plan"]["per_cav"] == {c: "damping" for c in rig.cav}
        assert set(state["log_offsets"]) == set(LOGS)
        assert state["log_offsets"]["decisions.jsonl"] > 0
        assert payload["master_state"]["manifest"]["schema"] == FakeMasterHarness.STATE_SCHEMA


# ------------------------------------------------------- (j) + (k) guards
class TestGuards:
    def test_holdout_world_refused_without_owner_flag(self):
        with pytest.raises(SplitRefusedError, match="HOLDOUT"):
            ClosedLoopConfig(
                out_root=Path("E:/rd-wt/pl16/output"),
                run_id="j1",
                world="chunk2_HOLDOUT",
                max_sim_s=1e9,
            )
        with pytest.raises(SplitRefusedError, match="TEST"):
            ClosedLoopConfig(
                out_root=Path("E:/rd-wt/pl16/output"),
                run_id="j2",
                world="w-ok",
                split="TEST",
                max_sim_s=1e9,
            )

    def test_owner_flag_allows_holdout_construction(self):
        config = ClosedLoopConfig(
            out_root=Path("E:/rd-wt/pl16/output"),
            run_id="j3",
            world="chunk2_HOLDOUT",
            allow_holdout=True,
            max_sim_s=1e9,
        )
        assert config.world == "chunk2_HOLDOUT"

    def test_decision_period_range_enforced(self):
        with pytest.raises(ValueError, match="decision_period_s"):
            ClosedLoopConfig(
                out_root=Path("E:/rd-wt/pl16/output"),
                run_id="j4",
                world="w-ok",
                decision_period_s=5.0,
                max_sim_s=1e9,
            )

    def test_write_once_run_dir_refused_without_resume(self, tmp_path):
        rig = Rig(tmp_path, run_id="k1")
        first = rig.run()
        assert first["exit_status"] == "complete"
        before = {name: (rig.run_dir / name).stat().st_size for name in LOGS}
        with pytest.raises(RunDirExistsError, match="write-once"):
            ClosedLoopRunner(rig.config, rig.components).run()
        after = {name: (rig.run_dir / name).stat().st_size for name in LOGS}
        assert before == after, "a refused command must not touch the run dir"

    def test_resume_without_dir_or_checkpoint_refused(self, tmp_path):
        rig = Rig(tmp_path, run_id="k2")
        with pytest.raises(NothingToResumeError, match="does not exist"):
            rig.run(resume=True)
        rig.run_dir.mkdir(parents=True)
        with pytest.raises(NothingToResumeError, match="no checkpoint"):
            rig.run(resume=True)


# ------------------------------------------------------------------- CLI
class TestCli:
    def test_cli_end_to_end_with_mock_components(self):
        # the E:-paths-only guard makes tmp_path (system temp) unusable here;
        # this test's transient run dir lives inside the workspace instead
        with tempfile.TemporaryDirectory(dir="E:/rd-wt/pl16/output") as scratch:
            out = str(Path(scratch).resolve())
            code = main(
                [
                    "--out",
                    out,
                    "--run-id",
                    "cli1",
                    "--world",
                    "w-clitest",
                    "--horizon-s",
                    "100",
                    "--decision-period-s",
                    "30",
                    "--predictor",
                    "mock",
                    "--lookahead-horizon-s",
                    "30",
                    "--max-sim-s",
                    "100000",
                    "--components",
                    "mock",
                ]
            )
            assert code == 0
            receipt = json.loads(
                (Path(out) / "cli1" / "RUN_RECEIPT.json").read_text(encoding="utf-8")
            )
            assert receipt["schema"] == RECEIPT_SCHEMA
            assert receipt["exit_status"] == "complete"
            assert receipt["decisions"] == 4

    def test_cli_refuses_d_drive_out_root(self, capsys):
        # built by concatenation: the PL-13 grep forbids D: path literals in
        # planner test sources; the runtime guard refuses the path either way
        doomed = "D" + ":/somewhere"
        code = main(
            [
                "--out",
                doomed,
                "--run-id",
                "x",
                "--world",
                "w-ok",
                "--horizon-s",
                "100",
                "--predictor",
                "mock",
                "--max-sim-s",
                "10",
                "--components",
                "mock",
            ]
        )
        assert code == 2
        assert "E:" in capsys.readouterr().err
        with pytest.raises(PathGuardError, match="E:"):
            config_from_args(_cli_args("--out", doomed))

    def test_cli_refuses_expired_override_file(self, tmp_path, capsys):
        path = override_file(
            tmp_path,
            {
                "reason": "r",
                "approved_by": "owner:x",
                "scope": "both",
                "expires_utc": "2020-01-01T00:00:00+00:00",
            },
            "expired_cli.json",
        )
        code = main(
            [
                "--out",
                "E:/rd-wt/pl16/output",
                "--run-id",
                "x",
                "--world",
                "w-ok",
                "--horizon-s",
                "100",
                "--predictor",
                "mock",
                "--max-sim-s",
                "10",
                "--components",
                "mock",
                "--gate-override-file",
                str(path),
            ]
        )
        assert code == 2
        assert "expired" in capsys.readouterr().err


# ---------------------------- (l) real components wiring (faked Popen/traci)
class _FakeBranchSimulation:
    """Branch-process simulation surface: loadState lands on the snapshot
    t0 (read from the sibling STATE_MANIFEST.json, like the real server)."""

    def __init__(self, connection):
        self._connection = connection

    def loadState(self, path):
        manifest = json.loads(
            (Path(path).parent / "STATE_MANIFEST.json").read_text(encoding="utf-8")
        )
        self._connection.time = float(manifest["branch_point_t0_s"])
        self._connection.loaded.append(str(path))

    def getTime(self):
        return self._connection.time

    def simulationStep(self):
        self._connection.time += 0.5

    def getArrivedIDList(self):
        return []  # T-B: the control write reads arrivals (run_suffix lifecycle)

    def getStartingTeleportIDList(self):
        return ()

    def getCollidingVehiclesIDList(self):
        return ()


class _FakeBranchVehicle:
    """The branch connection's vehicle surface. The REAL CommandOwnership
    write goes through it: getSpeedMode/setSpeedMode (the speedMode 27
    configuration check), slowDown (every issued longitudinal command) and
    setSpeed(-1) (native handback releases) -- all recorded."""

    def __init__(self):
        self.speed_modes = {}
        self.slow_downs = []
        self.speed_resets = []

    def getIDList(self):
        return ("c0", "c1")

    def getLeader(self, vid, range_m):
        return ("lead", 30.0)

    def getLanePosition(self, vid):
        return 12.5

    def getLaneIndex(self, vid):
        return 0

    def getSpeed(self, vid):
        return 20.0

    def getAcceleration(self, vid):
        return 0.0

    # T-A: the campaign physics-row identity (the real p4 delay evaluator
    # consumes these through the branch steps)
    def getLaneID(self, vid):
        return "L0_0"

    def getRoadID(self, vid):
        return "L0"

    def getSpeedMode(self, vid):
        return self.speed_modes.get(str(vid), 31)

    def setSpeedMode(self, vid, mode):
        self.speed_modes[str(vid)] = int(mode)

    def slowDown(self, vid, target_speed_mps, duration_s):
        self.slow_downs.append((str(vid), float(target_speed_mps), float(duration_s)))

    def setSpeed(self, vid, speed_mps):
        self.speed_resets.append((str(vid), float(speed_mps)))


class _FakeEdgeSurface:
    """The edge surface ``scenarios.observation_map`` reads (PL-04j): one
    lane per adapter edge id resolves a non-empty merge map on the fakes --
    the real ``RealMasterHarness.start`` / branch tick hook call it on every
    connection."""

    def getLaneNumber(self, edge_id):
        del edge_id
        return 1


class _FakeLaneLengthSurface:
    """The lane surface ``observation_map`` reads: a length for any lane id
    (the map values shift, the map stays non-empty and monotone)."""

    def getLength(self, lane_id):
        del lane_id
        return 500.0


class _FakeConnection:
    """One labeled traci connection: the surface TraciBranchRunner drives."""

    def __init__(self, label):
        self.label = label
        self.time = 0.0
        self.loaded = []
        self.closed = []
        self.vehicle = _FakeBranchVehicle()
        self.simulation = _FakeBranchSimulation(self)
        # PL-04j: the per-branch control write resolves the merge-region
        # map on ITS connection (observation_map)
        self.edge = _FakeEdgeSurface()
        self.lane = _FakeLaneLengthSurface()

    def simulationStep(self):
        self.simulation.simulationStep()

    def close(self, wait=True):
        self.closed.append(bool(wait))


class _MasterSimulationSurface:
    """The MASTER's module-level surface: saveState writes a REAL
    17-decimal + rngState state file (the exactness guard must pass on it);
    loadState is recorded and REFUSED -- the master never receives one
    (except on an opted-in resume, where it restores the snapshot clock)."""

    def __init__(self, traci_fake):
        self._traci = traci_fake

    # PL-04k: the campaign physics-row fields read these per step
    def getDepartedIDList(self):
        return ()

    def getArrivedIDList(self):
        return ()

    def getCollidingVehiclesIDList(self):
        return ()

    def getStartingTeleportIDList(self):
        return ()

    def getEndingTeleportIDList(self):
        return ()

    def saveState(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            '<state><rngState seed="0"/>'
            '<vehicle id="c0" pos="12.50000000000000000" speed="20.00000000000000000"/>'
            '<vehicle id="c1" pos="30.50000000000000000" speed="20.00000000000000000"/>'
            "</state>",
            encoding="utf-8",
        )
        self._traci.saved.append(str(path))

    def loadState(self, path):
        self._traci.master_loads.append(str(path))
        if not self._traci.allow_master_loadstate:
            raise AssertionError("the MASTER connection received loadState")
        manifest = json.loads(
            (Path(path).parent / "STATE_MANIFEST.json").read_text(encoding="utf-8")
        )
        self._traci.master_time = float(manifest["branch_point_t0_s"])

    def getTime(self):
        return self._traci.master_time


class _MasterVehicleSurface:
    """The MASTER module-level vehicle surface. The REAL CommandOwnership
    write goes through it: getSpeedMode/setSpeedMode (the speedMode 27
    configuration check), slowDown (every issued longitudinal command) and
    setSpeed(-1) (native handback releases) -- all recorded on the traci
    fake."""

    def __init__(self, traci_fake):
        self._traci = traci_fake

    def getIDList(self):
        return self._traci.vehicle_ids

    def setSpeedMode(self, vid, mode):
        self._traci.speed_modes[vid] = mode

    def getSpeedMode(self, vid):
        return self._traci.speed_modes.get(vid, 31)

    def slowDown(self, vid, target_speed_mps, duration_s):
        self._traci.master_slow_downs.append(
            (str(vid), float(target_speed_mps), float(duration_s))
        )

    def setSpeed(self, vid, speed_mps):
        self._traci.master_speed_resets.append((str(vid), float(speed_mps)))

    # PL-04d: the master physics snapshot reads these per live vehicle
    def getLanePosition(self, vid):
        return 12.5

    def getSpeed(self, vid):
        return 20.0

    def getAcceleration(self, vid):
        return 0.0

    def getLaneIndex(self, vid):
        return 0

    # PL-04k: the campaign per-vehicle physics fields
    def getLaneID(self, vid):
        return "L0_0"

    def getRoadID(self, vid):
        return "L0"

    def getTypeID(self, vid):
        return "normal_car"


class _FakeTraciModule:
    """The traci MODULE surface with labeled connections (zero SUMO).

    ``init`` registers a fresh connection under its label and
    ``getConnection`` hands it back -- the exact shapes the real
    RealMasterHarness._launch and RealBranchProcessFactory use (PL-04l:
    including the ``doSwitch`` keyword; both launchers now pass False).
    The FIRST init is the master, and ``getConnection(<master label>)``
    returns the module ITSELF: this fake's master surfaces (simulation/
    vehicle/lane/edge + master_time/saved/master_loads tripwires) live on
    the module object, so the harness's own master Connection is the module
    here. This fake does NOT model module switching -- the e2e-grade fake
    that does (tests/planner/test_pl04i_e2e_real_run._TinyTraci) is the one
    that proves the PL-04l routing; after the fix no production path uses
    the module surface at all."""

    def __init__(self, master_time=5.0, vehicle_ids=("c0", "c1"),
                 allow_master_loadstate=False):
        self.master_time = float(master_time)
        self.vehicle_ids = tuple(vehicle_ids)
        # the tripwire: the master connection never receives loadState,
        # EXCEPT on a resumed run (the PL-04d resume test opts in; the real
        # resumed process launches fresh and loads its checkpoint once)
        self.allow_master_loadstate = bool(allow_master_loadstate)
        self.connections = {}
        self.init_calls = []
        self._master_label = None
        self.closed = []
        self.saved = []
        self.master_loads = []
        self.speed_modes = {}
        self.master_slow_downs = []
        self.master_speed_resets = []
        self.simulation = _MasterSimulationSurface(self)
        self.vehicle = _MasterVehicleSurface(self)
        # PL-04j: the master's start() resolves the merge-region map on the
        # module-level (master) surface via observation_map
        self.edge = _FakeEdgeSurface()
        self.lane = _FakeLaneLengthSurface()

    def getVersion(self):
        return (27, "SUMO 1.27.1")

    def simulationStep(self):
        # the module-level step the master harness drives (0.5 s physics)
        self.master_time += 0.5

    def init(self, port=None, host=None, numRetries=None, proc=None, label=None,
             doSwitch=True):
        self.init_calls.append(
            {"label": label, "port": int(port), "pid": proc.pid}
        )
        if self._master_label is None:
            self._master_label = label
        self.connections[label] = _FakeConnection(label)
        return self.getVersion()

    def getConnection(self, label):
        # the master drives the module-bound master surface (the module IS
        # the master connection in this fake); every other label gets its
        # own fresh connection object
        if label == self._master_label:
            return self
        return self.connections[label]

    def close(self, wait=True):
        self.closed.append(bool(wait))


class _FakeProcess:
    def __init__(self, cmd, creationflags, pid):
        self.cmd = [str(part) for part in cmd]
        self.creationflags = creationflags
        self.pid = pid
        self.wait_calls = 0
        self.killed = False

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        self.wait_calls += 1
        return 0


class _RecordingPopen:
    """The faked subprocess.Popen: records every launch, starts NOTHING
    (no real SUMO can ever start while it is installed)."""

    def __init__(self):
        self.calls = []

    def __call__(self, cmd, creationflags=0, *args, **kwargs):
        proc = _FakeProcess(cmd, creationflags, pid=4200 + len(self.calls))
        self.calls.append(proc)
        return proc


class _FakeBranchSensor:
    """The faked SumoLocalSensor (PL-04c wiring): records captures;
    ``ValueError`` for ``unavailable_ids`` (the -> None frame path),
    ``RuntimeError`` for ``boom_ids`` (the branch raises).

    Frames must satisfy the REAL consumers downstream of the shared control
    write: ``CommandOwnership.step`` (``vehicle_id`` match, keys within the
    live roster) and ``CirclesController.decide``/``wall_group`` (current
    ``timestamp_s``, finite nonnegative ``speed_mps``/``allowed_speed_mps``
    coverages within the sensor range, and a neighbor within its coverage),
    so the defaults carry a contract-valid damping frame. Tests override
    per-CAV fields through ``frames_by_id``/``unavailable_ids``."""

    unavailable_ids: frozenset = frozenset()
    boom_ids: frozenset = frozenset()

    def __init__(self, api, contract=None, lane_end_distance_to_merge=None):
        self.api = api
        self.captures = []

    def capture(self, cav_id, *, now_s, previous_command_delta=0):
        self.captures.append((str(cav_id), float(now_s)))
        if str(cav_id) in type(self).boom_ids:
            raise RuntimeError(f"sensor exploded for {cav_id}")
        if str(cav_id) in type(self).unavailable_ids:
            raise ValueError("sensor unavailable (fake)")
        return SimpleNamespace(
            vehicle_id=str(cav_id),
            timestamp_s=float(now_s),
            speed_mps=20.0,
            allowed_speed_mps=30.0,
            front_coverage_m=50.0,
            rear_coverage_m=50.0,
            front=SimpleNamespace(gap_m=30.0, relative_speed_mps=0.0),
            rear=None,
        )


class _RecordingCommandOwnership(CommandOwnership):
    """The REAL ``CommandOwnership`` with test recording ONLY.

    Every attribute the /3 ``MixedAssignmentSession`` touches on its owner
    is the REAL implementation: ``cav_ids`` and ``controller`` (identity
    checked at the first ``step``), ``step(...)`` (validation + the sole
    longitudinal write), ``_last`` (read by ``_controller_state`` for the
    damping target telemetry) and ``release_all``. The subclass adds the
    recording surfaces the assertions read: ``steps`` (per-tick now_s /
    frames / live_ids / native_ids / wall_commands / per-CAV record owners
    and issued targets), ``release_calls``, and the derived ``issued`` list
    (CAVs written by a non-native owner, the old fake's recording surface).
    """

    created: list = []

    def __init__(self, api, cav_ids, controller, *, cav_speed_mode=31):
        super().__init__(api, cav_ids, controller, cav_speed_mode=cav_speed_mode)
        self.steps = []
        self.release_calls = []
        type(self).created.append(self)

    def step(self, frames, *, now_s, live_ids, wall_commands=None, native_ids=(),
             **kwargs):
        records = super().step(
            frames, now_s=now_s, live_ids=live_ids,
            wall_commands=wall_commands, native_ids=native_ids, **kwargs,
        )
        self.steps.append(
            {
                "now_s": float(now_s),
                "frames": dict(frames),
                "live_ids": {str(v) for v in live_ids},
                "native_ids": {str(v) for v in native_ids},
                "wall_commands": dict(wall_commands or {}),
                "owners": {str(r.vehicle_id): str(r.owner) for r in records},
                "targets": {
                    str(r.vehicle_id): r.issued_target_speed_mps for r in records
                },
            }
        )
        return records

    @property
    def issued(self) -> list:
        return [
            v
            for step in self.steps
            for v, owner in sorted(step["owners"].items())
            if owner != "native"
        ]

    def release_all(self, *, live_ids):
        super().release_all(live_ids=live_ids)
        self.release_calls.append({str(v) for v in live_ids})


def _campaign_scenario(tmp_path):
    """A campaign-format scenario cfg (explicit <vehicle> list, fixed vType,
    speedDev="0"): the PL-10c guard must pass on exactly this cfg."""
    (tmp_path / "routes.rou.xml").write_text(
        '<routes><vType id="passenger" speedDev="0"/>'
        '<vehicle id="v0" type="passenger" depart="0"/>'
        "</routes>",
        encoding="utf-8",
    )
    cfg = tmp_path / "scenario.sumocfg"
    cfg.write_text(
        "<configuration><input>"
        '<route-files value="routes.rou.xml"/>'
        "</input></configuration>",
        encoding="utf-8",
    )
    return cfg


class RealComponentsRig:
    """build_real_components with faked Popen + faked traci (zero SUMO).

    PL-04c: the sensor is faked and the ownership class is the REAL
    ``CommandOwnership`` with a recording subclass -- both installed at
    their source modules, so the wired per-branch control write runs end
    to end without SUMO and its calls are all recorded."""

    def __init__(self, monkeypatch, tmp_path, *, run_id="real1", seed=1729):
        self.cfg = _campaign_scenario(tmp_path)
        self.popen = _RecordingPopen()
        self.traci = _FakeTraciModule()
        monkeypatch.setattr(subprocess, "Popen", self.popen)
        monkeypatch.setitem(sys.modules, "traci", self.traci)
        monkeypatch.setattr("traffic_shepherd.sensing.SumoLocalSensor", _FakeBranchSensor)
        monkeypatch.setattr(
            "traffic_shepherd.ownership.CommandOwnership", _RecordingCommandOwnership
        )
        _FakeBranchSensor.unavailable_ids = frozenset()
        _FakeBranchSensor.boom_ids = frozenset()
        del _RecordingCommandOwnership.created[:]
        monkeypatch.delenv("SUMO_BINARY", raising=False)
        monkeypatch.setenv("SUMO_HOME", "E:/sumo_home_fake_for_test")
        self.args = SimpleNamespace(
            scenario=str(self.cfg),
            predictor="perfect_sim",
            cav_ids="c0,c1",
            decision_period_s=30.0,
            lookahead_horizon_s=30.0,
            lookahead_commit_s=10.0,
            seed=seed,
        )
        self.components = build_real_components(self.args)
        self.context = BuildContext(run_dir=tmp_path / run_id, resume=False, process_seq=1)

    def start_harness(self):
        harness = self.components.make_harness(self.context)
        harness.start()
        return harness

    def predict_once(self, model, t0_s=5.0):
        sample = SimpleNamespace(
            hybrid={"tracked": {}},
            arrivals={"sample_id": "k0", "departures": ()},
            meta={
                "sample_id": "k0",
                "time_s": float(t0_s),
                "merge_observed": True,
                "coverage": 1.0,
            },
        )
        plan = PlanCandidate(per_cav={"c0": "damping", "c1": "native"})
        (prediction,) = model.predict_many(
            [sample],
            [plan],
            controller_state={},
            horizon_s=30.0,
            commit_s=10.0,
            continuation="native",
            trajectory_s=None,
        )
        return prediction


class TestRealComponentsWiring:
    """PL-04b: build_real_components wires the REAL perfect-sim lookahead.

    ZERO SUMO: subprocess.Popen and the traci module are fakes; the Popen
    fake records every launch and starts nothing."""

    def test_make_world_model_wires_the_real_perfect_sim_lookahead(
        self, tmp_path, monkeypatch
    ):
        rig = RealComponentsRig(monkeypatch, tmp_path, run_id="l1")
        harness = rig.start_harness()
        assert isinstance(harness, RealMasterHarness)
        model = rig.components.make_world_model(rig.context)
        # (a) a PerfectSimWorldModel whose branch runner has a REAL factory
        assert isinstance(model, PerfectSimWorldModel)
        assert isinstance(model.branch_runner, TraciBranchRunner)
        assert isinstance(model.branch_runner.process_factory, BranchProcessFactory)
        assert type(model.branch_runner.process_factory) is RealBranchProcessFactory
        assert isinstance(model.snapshotter, MasterBoundSnapshotter)
        # the lookahead state root lives under run_dir/lookahead_state/p<seq>
        assert model.snapshotter.state_root == rig.context.run_dir / "lookahead_state" / "p1"
        # the scenario cfg IS the declared route file (the PL-10c guard input)
        assert model.scenario_route_files == (str(rig.cfg),)
        # the master harness received the SAME scenario cfg
        assert harness.scenario_cfg == rig.cfg
        assert harness.scenario_routes == [rig.cfg]
        harness.close()

    def test_predict_many_runs_one_fresh_fake_process_per_branch(
        self, tmp_path, monkeypatch
    ):
        rig = RealComponentsRig(monkeypatch, tmp_path, run_id="l2")
        harness = rig.start_harness()
        model = rig.components.make_world_model(rig.context)
        prediction = rig.predict_once(model)
        # (e) no real SUMO: the Popen fake recorded EVERY launch (1 master
        # + 2 branches: one plan + the shared native reference)
        assert len(rig.popen.calls) == 3
        # (b) one fake process per branch, each closed exactly once
        branch_inits = rig.traci.init_calls[1:]
        assert len(branch_inits) == 2
        assert len({c["label"] for c in branch_inits}) == 2
        for init in branch_inits:
            connection = rig.traci.connections[init["label"]]
            assert len(connection.loaded) == 1  # one loadState per fresh branch
            assert connection.closed == [False]  # traci close on its OWN connection
            assert init["pid"] != rig.popen.calls[0].pid
        for proc in rig.popen.calls[1:]:
            assert proc.wait_calls == 1  # waited on its OWN Popen child only
        assert prediction.meta["model_id"] == "perfect_sim/pl04-closed-loop"
        assert prediction.meta["continuation"] == "native"
        harness.close()

    def test_master_connection_only_ever_receives_savestate(self, tmp_path, monkeypatch):
        rig = RealComponentsRig(monkeypatch, tmp_path, run_id="l3")
        harness = rig.start_harness()
        model = rig.components.make_world_model(rig.context)
        rig.predict_once(model)
        # (c) the master saved the branch point through its own surface...
        assert len(rig.traci.saved) == 1
        # ...and NEVER received loadState (the fake master loadState refuses)
        assert rig.traci.master_loads == []
        harness.close()

    def test_branch_command_line_matches_the_master_launch_contract(
        self, tmp_path, monkeypatch
    ):
        rig = RealComponentsRig(monkeypatch, tmp_path, run_id="l4")
        harness = rig.start_harness()
        model = rig.components.make_world_model(rig.context)
        rig.predict_once(model)

        def flagged(cmd, flag):
            return cmd[cmd.index(flag) + 1]

        # (d) every launch (master and branches alike): -c <cfg>, precision
        # 17 + rng true, its OWN port, same seed/step-length, BELOW_NORMAL,
        # SUMO_HOME's sumo.exe as the binary (SUMO_BINARY unset)
        expected_binary = str(Path("E:/sumo_home_fake_for_test") / "bin" / "sumo.exe")
        ports = []
        for proc in rig.popen.calls:
            cmd = proc.cmd
            assert cmd[0] == expected_binary
            assert flagged(cmd, "-c") == str(rig.cfg)
            assert flagged(cmd, "--save-state.precision") == "17"
            assert flagged(cmd, "--save-state.rng") == "true"
            assert flagged(cmd, "--seed") == "1729"  # the rig's configured seed
            assert flagged(cmd, "--step-length") == "0.5"
            assert proc.creationflags == BELOW_NORMAL_PRIORITY_CLASS
            ports.append(flagged(cmd, "--remote-port"))
        assert len(ports) == 3
        assert len(set(ports)) == 3  # every process got its OWN port
        harness.close()

    def test_sumo_binary_default_is_sumo_home_else_sumo_binary(self, tmp_path, monkeypatch):
        args = SimpleNamespace(
            scenario="unused.sumocfg",
            predictor="perfect_sim",
            cav_ids="c0",
            decision_period_s=30.0,
            lookahead_horizon_s=30.0,
            lookahead_commit_s=None,
            seed=7,
        )
        monkeypatch.setenv("SUMO_BINARY", "E:/explicit/sumo.exe")
        monkeypatch.setenv("SUMO_HOME", "E:/sumo_home_fake_for_test")
        components = build_real_components(args)
        harness = components.make_harness(BuildContext(tmp_path / "l5", False, 1))
        assert harness.sumo_binary == "E:/explicit/sumo.exe"
        # with SUMO_BINARY unset, SUMO_HOME's sumo.exe is the default
        monkeypatch.delenv("SUMO_BINARY")
        components = build_real_components(args)
        harness = components.make_harness(BuildContext(tmp_path / "l5b", False, 1))
        assert harness.sumo_binary == str(Path("E:/sumo_home_fake_for_test") / "bin" / "sumo.exe")

    def test_make_world_model_before_make_harness_refused(self, tmp_path, monkeypatch):
        rig = RealComponentsRig(monkeypatch, tmp_path, run_id="l6")
        with pytest.raises(BuildFailedError, match="make_harness"):
            rig.components.make_world_model(rig.context)


class TestPL04cBranchControlWrite:
    """PL-04c: the REAL per-branch control write runs in every non-native
    branch and the run seed reaches every SUMO launch.

    Faked Popen/traci/sensor/ownership (the PL-04b rig pattern): zero
    SUMO; the real TraciBranchRunner fail-closed rule, the real /3
    dispatcher session and the real BranchControlWrite plumbing are all
    exercised end to end."""

    def test_damping_branch_steps_control_every_tick_and_issues_commands(
        self, tmp_path, monkeypatch
    ):
        rig = RealComponentsRig(monkeypatch, tmp_path, run_id="m1")
        harness = rig.start_harness()
        model = rig.components.make_world_model(rig.context)
        rig.predict_once(model)  # PlanCandidate: c0 damping, c1 native
        # branch owners bind BRANCH CONNECTIONS; the master's own ownership
        # (created at harness.start) binds the traci module surface
        owners = [o for o in _RecordingCommandOwnership.created if o.api is not rig.traci]
        # T-B (#4): the ownership write runs in EVERY branch, the all-native
        # reference included (it must configure speedMode 27 and release any
        # inherited override); ONLY the plan branch issues commands
        assert len(owners) == 2
        issuing = [o for o in owners if o.issued]
        assert len(issuing) == 1
        owner = issuing[0]
        # the CAV speed mode matches the master (speedMode 27)
        assert owner.cav_speed_mode == CAV_SPEED_MODE == 27
        assert owner.controller is not None
        # the tick hook ran at EVERY control tick of the 30 s branch,
        # starting at the restore t0, on the 1 s grid, contiguous
        ticks = [step["now_s"] for step in owner.steps]
        assert ticks == [5.0 + i for i in range(30)]
        for step in owner.steps:
            assert step["live_ids"] == {"c0", "c1"}
            # a captured frame for every live roster CAV, damping CAV
            # included; the damping CAV is issued a command every tick
            assert step["frames"]["c0"] is not None
            assert step["frames"]["c1"] is not None
        assert set(owner.issued) == {"c0"}  # native CAV never commanded
        harness.close()

    def test_valueerror_capture_becomes_a_none_frame(self, tmp_path, monkeypatch):
        rig = RealComponentsRig(monkeypatch, tmp_path, run_id="m2")
        harness = rig.start_harness()
        _FakeBranchSensor.unavailable_ids = frozenset({"c1"})
        model = rig.components.make_world_model(rig.context)
        rig.predict_once(model)
        owner = next(o for o in _RecordingCommandOwnership.created if o.api is not rig.traci)
        assert owner.steps
        for step in owner.steps:
            # the unavailable CAV's frame is EXPLICITLY None (never an
            # invented frame); the healthy CAV is captured normally
            assert step["frames"]["c1"] is None
            assert step["frames"]["c0"] is not None
        harness.close()

    def test_native_plan_runs_the_ownership_write_and_issues_nothing(self, tmp_path, monkeypatch):
        rig = RealComponentsRig(monkeypatch, tmp_path, run_id="m3")
        harness = rig.start_harness()
        model = rig.components.make_world_model(rig.context)
        sample = SimpleNamespace(
            hybrid={"tracked": {}},
            arrivals={"sample_id": "k0", "departures": ()},
            meta={
                "sample_id": "k0",
                "time_s": 5.0,
                "merge_observed": True,
                "coverage": 1.0,
            },
        )
        plan = PlanCandidate(per_cav={"c0": "native", "c1": "native"})
        (prediction,) = model.predict_many(
            [sample],
            [plan],
            controller_state={},
            horizon_s=30.0,
            commit_s=10.0,
            continuation="native",
            trajectory_s=None,
        )
        assert prediction.meta["model_id"] == "perfect_sim/pl04-closed-loop"
        # T-B (#4): the native reference AND the all-native plan branch BOTH
        # run the ownership control write (speedMode 27 configured, overrides
        # released) yet issue not a single command
        branch_owners = [o for o in _RecordingCommandOwnership.created if o.api is not rig.traci]
        assert len(branch_owners) == 2
        assert all(o.steps and o.issued == [] for o in branch_owners)
        assert all(s["native_ids"] == {"c0", "c1"} for o in branch_owners for s in o.steps)
        harness.close()

    def test_release_all_runs_even_when_the_branch_raises(self, tmp_path, monkeypatch):
        rig = RealComponentsRig(monkeypatch, tmp_path, run_id="m4")
        harness = rig.start_harness()
        _FakeBranchSensor.boom_ids = frozenset({"c0"})
        model = rig.components.make_world_model(rig.context)
        with pytest.raises(RuntimeError, match="sensor exploded"):
            rig.predict_once(model)
        owner = next(o for o in _RecordingCommandOwnership.created if o.api is not rig.traci)
        # the end-of-branch ownership handback ran DESPITE the raise
        # (TraciBranchRunner's finally), and the branch process was retired
        assert len(owner.release_calls) == 1
        assert owner.release_calls[0] == {"c0", "c1"}
        branch_procs = rig.popen.calls[1:]
        # T-B (#4): the native reference branch now runs the ownership write
        # too, so the FIRST branch (the native reference) already raises
        assert len(branch_procs) == 1
        assert all(proc.wait_calls == 1 for proc in branch_procs)
        harness.close()

    def test_seed_reaches_master_snapshotter_and_every_branch_launch(
        self, tmp_path, monkeypatch
    ):
        rig = RealComponentsRig(monkeypatch, tmp_path, run_id="m5", seed=314)
        harness = rig.start_harness()
        model = rig.components.make_world_model(rig.context)
        rig.predict_once(model)
        # the configured seed -- not a hard-coded constant -- is what the
        # master, the snapshotter manifests and EVERY branch launch carry
        assert harness.seed == 314
        assert model.snapshotter.seed == 314
        assert model.branch_runner.process_factory.seed == 314
        seeds = {
            proc.cmd[proc.cmd.index("--seed") + 1] for proc in rig.popen.calls
        }
        assert seeds == {"314"}
        harness.close()

    def test_real_components_without_seed_refused(self, tmp_path):
        args = SimpleNamespace(
            scenario="unused.sumocfg",
            predictor="perfect_sim",
            cav_ids="c0,c1",
            decision_period_s=30.0,
            lookahead_horizon_s=30.0,
            lookahead_commit_s=None,
            seed=None,
        )
        with pytest.raises(BuildFailedError, match="--seed"):
            build_real_components(args)

    def test_cli_components_real_without_seed_refused(self):
        # --components defaults to real: no --seed means refused at config
        with pytest.raises(BuildFailedError, match="--seed"):
            config_from_args(_cli_args("--scenario", "unused.sumocfg"))

    def test_cli_components_real_seed_lands_in_the_run_config(self, tmp_path):
        # PL-04e2 contract: a real config comes from --base-run (validated,
        # t0 + state sha recorded); the seed must equal the base state's
        base = make_base_run(tmp_path, seed=2718)
        config = config_from_args(
            _cli_args(
                "--components", "real", "--seed", "2718", "--base-run", str(base)
            )
        )
        assert config.seed == 2718  # the receipt records this seed
        assert config.base_run == base
        assert config.base_t0_s == 240.0
        assert config.base_state_sha256
        # the old --scenario-only real contract is refused (PL-04e2)
        with pytest.raises(BuildFailedError, match="--base-run"):
            config_from_args(
                _cli_args("--components", "real", "--seed", "2718", "--scenario", "s.cfg")
            )
        config = config_from_args(_cli_args("--components", "mock"))
        assert config.seed == 0  # the mock path keeps the historical default

    def test_main_refuses_components_real_without_seed(self, capsys):
        code = main(
            [
                "--out",
                "E:/rd-wt/pl18/output/pl04c_seed_refusal_probe",
                "--run-id",
                "no-seed",
                "--world",
                "w-ok",
                "--horizon-s",
                "100",
                "--predictor",
                "perfect_sim",
                "--max-sim-s",
                "10",
                "--components",
                "real",
                "--scenario",
                "unused.sumocfg",
            ]
        )
        assert code == 2
        assert "--seed" in capsys.readouterr().err


# ---------------------------------- (n) PL-04d: the real closed-loop master
class _MasterSensor:
    """Sensor fake for the PL-04d master tests: per-CAV frame specs with
    merge-distance fields; ``ValueError`` for unavailable_ids (the -> None
    frame path); a time-gated RuntimeError for boom_ids (an interrupted
    run). Tests set the instance attributes AFTER harness.start() built the
    sensor (RealMasterHarness._control_write.sensor).

    Frames must satisfy the REAL consumers downstream of the shared control
    write (``CommandOwnership.step`` validation, ``CirclesController.decide``
    and the /3 wall-group session), so the defaults carry a contract-valid
    damping frame; ``frames_by_id`` overrides per-CAV fields (lane_index,
    distance_to_merge_m, ...)."""

    def __init__(self, api, contract=None, lane_end_distance_to_merge=None):
        self.api = api
        self.lane_end_distance_to_merge = dict(lane_end_distance_to_merge or {})
        self.captures = []
        self.frames_by_id = {}
        self.unavailable_ids = frozenset()
        self.boom_ids = frozenset()
        self.boom_from_s = float("inf")

    def capture(self, cav_id, *, now_s, previous_command_delta=0):
        self.captures.append((str(cav_id), float(now_s)))
        if str(cav_id) in self.boom_ids and float(now_s) >= self.boom_from_s:
            raise RuntimeError(f"sensor exploded for {cav_id}")
        if str(cav_id) in self.unavailable_ids:
            raise ValueError("sensor unavailable (fake)")
        spec = dict(self.frames_by_id.get(str(cav_id), {}))
        spec.setdefault("vehicle_id", str(cav_id))
        spec.setdefault("timestamp_s", float(now_s))
        spec.setdefault("speed_mps", 20.0)
        spec.setdefault("allowed_speed_mps", 30.0)
        spec.setdefault("front_coverage_m", 50.0)
        spec.setdefault("rear_coverage_m", 50.0)
        spec.setdefault("front", SimpleNamespace(gap_m=30.0, relative_speed_mps=0.0))
        spec.setdefault("rear", None)
        return SimpleNamespace(**spec)


class MasterRig:
    """A RealMasterHarness over faked Popen/traci/sensor/ownership (the
    PL-04b/04c rig pattern, master-side). Zero SUMO."""

    def __init__(self, monkeypatch, tmp_path, *, run_id="m", master_time=0.0,
                 vehicle_ids=("c0", "c1"), seed=1729, allow_master_loadstate=False):
        self.cfg = _campaign_scenario(tmp_path)
        self.popen = _RecordingPopen()
        self.traci = _FakeTraciModule(
            master_time=master_time,
            vehicle_ids=vehicle_ids,
            allow_master_loadstate=allow_master_loadstate,
        )
        monkeypatch.setattr(subprocess, "Popen", self.popen)
        monkeypatch.setitem(sys.modules, "traci", self.traci)
        monkeypatch.setattr("traffic_shepherd.sensing.SumoLocalSensor", _MasterSensor)
        monkeypatch.setattr("traffic_shepherd.ownership.CommandOwnership", _RecordingCommandOwnership)
        del _RecordingCommandOwnership.created[:]
        monkeypatch.delenv("SUMO_BINARY", raising=False)
        monkeypatch.setenv("SUMO_HOME", "E:/sumo_home_fake_for_test")
        self.args = SimpleNamespace(
            scenario=str(self.cfg),
            predictor="perfect_sim",
            cav_ids="c0,c1",
            decision_period_s=30.0,
            lookahead_horizon_s=30.0,
            lookahead_commit_s=10.0,
            seed=seed,
        )
        self.components = build_real_components(self.args)
        self.context = BuildContext(run_dir=tmp_path / run_id, resume=False, process_seq=1)

    def make_harness(self, predictor="perfect_sim", *, start=True):
        self.args.predictor = predictor
        harness = self.components.make_harness(self.context)
        self._harness = harness
        if start:
            harness.start()
        return harness

    @property
    def sensor(self):
        return self._harness._control_write.sensor

    @property
    def master_owner(self):
        return next(o for o in _RecordingCommandOwnership.created if o.api is self.traci)

    def v3_plan(self, actions):
        return to_dispatcher_v3(SimpleNamespace(plan=PlanCandidate(per_cav=dict(actions))), t0_s=0.0)


class TestPL04dMasterControlWrite:
    """PL-04d (a): the master executes its CURRENT dispatcher plan on every
    1 s control tick through the SAME shared write the branches use."""

    def test_branch_control_write_is_the_shared_class(self):
        # one implementation, two users (master tick + branch tick hook)
        assert BranchControlWrite is SessionControlWrite

    def test_damping_plan_issues_commands_every_control_tick(self, tmp_path, monkeypatch):
        rig = MasterRig(monkeypatch, tmp_path, run_id="d1")
        harness = rig.make_harness()
        harness.apply_plan(rig.v3_plan({"c0": "damping", "c1": "damping"}), 0.0)
        for _ in range(4):  # 0.5, 1.0, 1.5, 2.0 s (0.5 s physics)
            harness.step_physics()
        owner = rig.master_owner
        # control ticks only, contiguous on the 1 s grid
        # T-B (#5): the tick runs BEFORE each physics step -- the first one
        # AT t0=0.0 (never one second late)
        assert [s["now_s"] for s in owner.steps] == [0.0, 1.0]
        # every live roster CAV captured (frames from the sensor) and
        # commanded EVERY control tick
        assert owner.issued == ["c0", "c1", "c0", "c1"]
        for step in owner.steps:
            assert step["frames"]["c0"] is not None
            assert step["frames"]["c1"] is not None
            assert step["live_ids"] == {"c0", "c1"}
        # speedMode 27 on the master
        assert rig.traci.speed_modes == {"c0": CAV_SPEED_MODE, "c1": CAV_SPEED_MODE}
        # the executed rows drain into dispatcher_rows.jsonl (the runner
        # emits whatever the harness drains after every physics step)
        rows = harness.drain_control_rows()
        assert [(r["t_s"], r["cav"], r["command"]) for r in rows] == [
            (0.0, "c0", "damping"), (0.0, "c1", "damping"),
            (1.0, "c0", "damping"), (1.0, "c1", "damping"),
        ]
        assert all(r["source"] == "master_control_tick" for r in rows)
        assert harness.drain_control_rows() == []
        # (e) outside resume the master never loadStates
        assert rig.traci.master_loads == []
        harness.close()

    def test_native_plan_steps_but_issues_no_commands(self, tmp_path, monkeypatch):
        rig = MasterRig(monkeypatch, tmp_path, run_id="d2")
        harness = rig.make_harness()
        harness.apply_plan(rig.v3_plan({"c0": "native", "c1": "native"}), 0.0)
        for _ in range(4):
            harness.step_physics()
        owner = rig.master_owner
        assert [s["now_s"] for s in owner.steps] == [0.0, 1.0]
        assert owner.issued == []  # native: the session steps, nothing written
        assert all(s["native_ids"] == {"c0", "c1"} for s in owner.steps)
        harness.close()

    def test_wall_plan_executes_through_the_v3_wall_group_session(
        self, tmp_path, monkeypatch
    ):
        rig = MasterRig(monkeypatch, tmp_path, run_id="d3")
        harness = rig.make_harness()
        plan = PlanCandidate(
            per_cav={},
            walls=(
                WallSpec(
                    wall_id="wall:c0+c1@5", members=("c0", "c1"), speed_mps=5.0,
                    phase="assemble", hold_s=60.0,
                ),
            ),
        )
        rows = harness.apply_plan(
            to_dispatcher_v3(SimpleNamespace(plan=plan), t0_s=0.0), 0.0
        )
        # NO refusal: the members stage as wall-v2 under /3 (walls execute)
        assert {r["cav"]: r["command"] for r in rows} == {
            "c0": "wall-v2", "c1": "wall-v2",
        }
        # the /3 wall session reads member lane + merge coordinates from the
        # frames (one member per lane; the s_m = -distance_to_merge_m span)
        rig.sensor.frames_by_id = {
            "c0": {"lane_index": 0, "distance_to_merge_m": 40.0},
            "c1": {"lane_index": 1, "distance_to_merge_m": 30.0},
        }
        harness.step_physics()  # 0.5 s: physics tick
        harness.step_physics()  # 1.0 s: the first control tick assembles
        # the members are under wall ownership: the session recorded the
        # wall entering ASSEMBLY, and the owner wrote the bounded pacing
        # commands through the real ownership path (slowDown on the traci
        # surface, record owner "wall")
        assert any(
            e["event"] == "assembly_start" and e["wall_id"] == "wall:c0+c1@5"
            for e in harness.session.wall_records
        )
        owner = rig.master_owner
        step = owner.steps[-1]
        assert step["wall_commands"] == {"c0": -3.0, "c1": -3.0}  # clip(-15, -3, 1.5)
        assert step["owners"] == {"c0": "wall", "c1": "wall"}
        assert {v for v, _, _ in rig.traci.master_slow_downs} == {"c0", "c1"}
        # observe() reports the members locked to their wall (the raw /3
        # session phase vocabulary; walls.py maps it on ingestion)
        state = harness.observe()[1]
        for member in ("c0", "c1"):
            assert state[member]["locked"] is True
            assert state[member]["wall_id"] == "wall:c0+c1@5"
            assert state[member]["wall_phase"] == "assembly"
            assert state[member]["action"] == "wall"
        harness.close()

    def test_mixed_assignment_plan_builds_the_real_v3_plan_with_walls(self):
        from traffic_shepherd.mixed_assignment import (  # noqa: PLC0415
            AssignmentPlan,
            WallAssignment,
        )

        # the ONE shared conversion: the REAL /3 AssignmentPlan with walls
        # EXECUTED (WallAssignment entries), strict_roster=True and the
        # WM-40 planner horizon 0 (the planner re-decides, dwell 0)
        v3 = to_dispatcher_v3(
            SimpleNamespace(
                plan=PlanCandidate(
                    per_cav={"c2": "damping"},
                    walls=(
                        WallSpec(
                            wall_id="wall:c0+c1@16", members=("c0", "c1"),
                            speed_mps=16.0, phase="assemble", hold_s=60.0,
                        ),
                    ),
                ),
            ),
            t0_s=0.0,
        )
        plan3 = mixed_assignment_plan(v3, plan_id="branch-id")
        assert isinstance(plan3, AssignmentPlan)
        assert plan3.plan_id == "branch-id"
        assert plan3.commit_horizon_s == 0.0
        assert plan3.strict_roster is True
        assert plan3.assignments == {"c0": "wall-v2", "c1": "wall-v2", "c2": "damping"}
        (wall,) = plan3.walls
        assert isinstance(wall, WallAssignment)
        assert wall.wall_id == "wall:c0+c1@16"
        assert wall.members == ("c0", "c1")
        assert wall.wall_speed_mps == 16.0
        assert wall.phase == "assemble"

    def test_mixed_assignment_plan_wall_free_plan_passthrough(self):
        # the same ONE conversion for wall-free plans: identity fields pass
        # through, strict_roster on, horizon 0, walls empty
        v3 = to_dispatcher_v3(
            SimpleNamespace(plan=PlanCandidate(per_cav={"c0": "damping", "c1": "native"})),
            t0_s=0.0,
        )
        plan3 = mixed_assignment_plan(v3)
        assert plan3.plan_id == v3.plan_id
        assert plan3.assignments == {"c0": "damping", "c1": "native"}
        assert plan3.walls == ()
        assert plan3.strict_roster is True
        assert plan3.commit_horizon_s == 0.0

    def test_version_guard_refuses_a_non_v3_installed_module(self, monkeypatch):
        # a stub installed module reporting /2 -> the NAMED error at plan
        # build time (no /2 fallback path exists)
        monkeypatch.setitem(
            sys.modules,
            "traffic_shepherd.mixed_assignment",
            SimpleNamespace(VERSION="mixed_assignment/2"),
        )
        v3 = to_dispatcher_v3(
            SimpleNamespace(plan=PlanCandidate(per_cav={"c0": "native", "c1": "native"})),
            t0_s=0.0,
        )
        with pytest.raises(WallAssignmentError, match="mixed_assignment/3"):
            mixed_assignment_plan(v3)


class TestPL04e1WallRunEndToEnd:
    """PL-04e1: a wall initial plan executes end to end through the full
    ClosedLoopRunner over the REAL harness (faked surfaces): the /3 wall
    session assembles, the wall commands land in dispatcher_rows.jsonl, and
    the run-level release hands the members back."""

    def test_wall_initial_plan_runs_wall_commands_and_release(
        self, tmp_path, monkeypatch
    ):
        rig = MasterRig(monkeypatch, tmp_path, run_id="e1wall")
        harness = rig.make_harness(start=False)  # the runner starts it
        # the /3 wall session reads member lane + merge coordinates from the
        # frames (one member per lane; s_m = -distance_to_merge_m): install
        # them when the RUNNER-triggered start() has built the sensor
        orig_start = harness.start

        def start_with_wall_frames():
            orig_start()
            rig.sensor.frames_by_id = {
                "c0": {"lane_index": 0, "distance_to_merge_m": 40.0},
                "c1": {"lane_index": 1, "distance_to_merge_m": 30.0},
            }

        harness.start = start_with_wall_frames
        components = SimpleNamespace(
            make_harness=lambda context: harness,
            make_world_model=lambda context: make_model(TWO_CAV),
            roster=TWO_CAV,
            planner_config=PlannerConfig(margin_floor_s=0.0),
            initial_plan=PlanCandidate(
                per_cav={},
                walls=(
                    WallSpec(
                        wall_id="wall:c0+c1@5", members=("c0", "c1"), speed_mps=5.0,
                        phase="assemble", hold_s=60.0,
                    ),
                ),
            ),
            sim_seconds_fn=None,
        )
        config = ClosedLoopConfig(
            out_root=tmp_path,
            run_id="e1wall",
            world="w-ok",
            horizon_s=2.5,
            warmup_s=0.0,
            decision_period_s=30.0,
            predictor="mock",
            lookahead_horizon_s=30.0,
            max_sim_s=1e9,
        )
        receipt = ClosedLoopRunner(config, components).run()
        assert receipt["exit_status"] == "complete"
        rows = [
            json.loads(line)
            for line in (tmp_path / "e1wall" / "dispatcher_rows.jsonl")
            .read_text("utf-8")
            .splitlines()
            if line.strip()
        ]
        # the applied plans stage the members as wall-v2 (no refusal)
        assert {
            r["cav"]: r["command"]
            for r in rows
            if r["t_s"] == 0.0 and r.get("source") != "master_control_tick"
        } == {"c0": "wall-v2", "c1": "wall-v2"}
        # the control ticks WRITE wall ownership through the real path
        control = [r for r in rows if r.get("source") == "master_control_tick"]
        # T-B (#5): ticks run BEFORE each step -- at t0 (0.0) too
        assert [(r["t_s"], r["cav"], r["command"]) for r in control] == [
            (0.0, "c0", "wall"), (0.0, "c1", "wall"),
            (1.0, "c0", "wall"), (1.0, "c1", "wall"),
            (2.0, "c0", "wall"), (2.0, "c1", "wall"),
        ]
        assert {v for v, _, _ in rig.traci.master_slow_downs} == {"c0", "c1"}
        # the /3 wall session assembled (its event ledger carries it)
        assert any(
            e["event"] == "assembly_start" for e in harness.session.wall_records
        )
        # the run-level release handed the members back (runner finally)
        assert rig.master_owner.release_calls == [{"c0", "c1"}]
        assert harness.proc is None


class TestPL04dMasterRelease:
    """PL-04d (b): owner.release_all runs when the run ends, is interrupted
    or raises (the runner's finally), and on the real harness directly."""

    def test_runner_releases_control_on_normal_end(self, tmp_path):
        rig = Rig(tmp_path, run_id="rel1")
        receipt = rig.run()
        assert receipt["exit_status"] == "complete"
        assert len(rig.harnesses[0].release_control_calls) == 1

    def test_runner_releases_control_when_the_run_raises(self, tmp_path):
        rig = Rig(tmp_path, run_id="rel2", harness=BombHarness(CAV, explode_at_s=30.5))
        with pytest.raises(RuntimeError, match="interrupted"):
            rig.run()
        assert len(rig.harnesses[0].release_control_calls) == 1

    def test_real_master_release_all_at_normal_end(self, tmp_path, monkeypatch):
        rig = MasterRig(monkeypatch, tmp_path, run_id="rel3")
        harness = rig.make_harness()
        for _ in range(4):
            harness.step_physics()
        harness.release_control()
        assert rig.master_owner.release_calls == [{"c0", "c1"}]
        harness.close()

    def test_real_master_release_all_when_the_write_raises(self, tmp_path, monkeypatch):
        rig = MasterRig(monkeypatch, tmp_path, run_id="rel4")
        harness = rig.make_harness()
        rig.sensor.boom_ids = frozenset({"c0"})
        rig.sensor.boom_from_s = 1.0
        with pytest.raises(RuntimeError, match="sensor exploded"):
            harness.step_physics()  # tick at 0.0 (t0), then to 0.5 s
            harness.step_physics()  # 0.5 s: physics tick, no write
            harness.step_physics()  # 1.0 s: the control tick raises (T-B: pre-step)
        # the runner's finally ordering: release BEFORE close, even after
        # the control write blew up mid-tick
        harness.release_control()
        assert rig.master_owner.release_calls == [{"c0", "c1"}]
        harness.close()


class TestPL04dObserve:
    """PL-04d (c): observe() builds the K=1 sample from the sensor frames
    ONLY -- real frames/coverage/merge_observed (contract 8.15), receipt
    age + missingness, ground truth absent."""

    def test_frames_coverage_and_merge_observed_from_sensor_frames(
        self, tmp_path, monkeypatch
    ):
        rig = MasterRig(monkeypatch, tmp_path, run_id="o1", vehicle_ids=("c0", "c1", "hdv9"))
        harness = rig.make_harness()
        rig.sensor.frames_by_id = {
            "c0": {"distance_to_merge_m": 40.0, "front_coverage_m": 50.0},
            "c1": {"distance_to_merge_m": 30.0, "front_coverage_m": 45.0},
        }
        samples, controller_state = harness.observe()
        assert len(samples) == 1
        sample = samples[0]
        # frames come from the sensor fake at the observation time
        assert sample.frames == {"c0": 0.0, "c1": 0.0}
        # ground truth (the traci vehicle list, incl. hdv9 beyond sensor
        # range) never enters the sample or its hybrid marker
        assert "hdv9" not in sample.frames
        assert "hdv9" not in json.dumps(sample.meta)
        assert set(controller_state) == {"c0", "c1"}
        # coverage-based 8.15 verdict from the frames, never hard-coded
        assert sample.meta["coverage"] == 1.0
        assert sample.meta["merge_observed"] is True
        assert sample.meta["merge_observed_reason"] is None
        assert sample.meta["frame_age_s"] == {"c0": 0.0, "c1": 0.0}
        assert sample.meta["frame_missing"] == []
        # controller_state = the session's committed reality (the /3 native
        # default; no plan applied, nothing locked)
        assert controller_state["c0"] == {
            "action": "native", "wall_id": None, "wall_phase": None,
            "time_in_phase_s": 0.0, "wall_speed_mps": None, "locked": False,
        }
        # perfect_sim: the documented opaque oracle marker (the oracle
        # reads truth through its own branch snapshot)
        assert "opaque" in sample.hybrid
        harness.close()

    def test_missing_frame_lowers_coverage_and_fails_merge_closed(
        self, tmp_path, monkeypatch
    ):
        rig = MasterRig(monkeypatch, tmp_path, run_id="o2")
        harness = rig.make_harness()
        rig.sensor.unavailable_ids = frozenset({"c1"})  # in range, no frame
        rig.sensor.frames_by_id = {
            "c0": {"distance_to_merge_m": 40.0, "front_coverage_m": 10.0},
        }
        (sample,), controller_state = harness.observe()
        # the unreporting CAV has NO frame (never invented) although the
        # ground-truth list carries it
        assert sample.frames == {"c0": 0.0}
        assert "c1" not in sample.frames
        assert sample.meta["coverage"] == 0.5
        assert sample.meta["frame_age_s"] == {"c0": 0.0}
        assert sample.meta["frame_missing"] == ["c1"]
        # low coverage case: no frame reaches the merge -> False + reason
        assert sample.meta["merge_observed"] is False
        assert "merge region" in sample.meta["merge_observed_reason"]
        assert set(controller_state) == {"c0", "c1"}  # roster reality, not frames
        harness.close()

    def test_no_merge_inputs_resolves_none_to_false(self, tmp_path, monkeypatch):
        rig = MasterRig(monkeypatch, tmp_path, run_id="o3")
        harness = rig.make_harness()
        # frames WITHOUT merge-distance fields (e.g. no RSU lane mapping)
        (sample,), _ = harness.observe()
        assert sample.meta["coverage"] == 1.0
        assert sample.meta["merge_observed"] is False
        assert "no RSU merge-region inputs" in sample.meta["merge_observed_reason"]
        harness.close()


class TestPL04dLearnedHybrid:
    """PL-04d (d): the learned predictor fails closed without a hybrid
    builder and builds through it when one is wired."""

    def test_learned_without_hybrid_builder_raises_the_named_error(
        self, tmp_path, monkeypatch
    ):
        rig = MasterRig(monkeypatch, tmp_path, run_id="h1")
        harness = rig.make_harness(predictor="learned")
        with pytest.raises(LearnedObservationUnavailableError, match="hybrid_builder"):
            harness.observe()
        harness.close()

    def test_learned_with_builder_builds_the_hybrid_from_sensor_frames(
        self, tmp_path, monkeypatch
    ):
        rig = MasterRig(monkeypatch, tmp_path, run_id="h2")
        harness = rig.make_harness(predictor="learned")
        rig.sensor.unavailable_ids = frozenset({"c1"})
        seen = {}

        def builder(*, frames, now_s, roster, coverage, merge_observed):
            seen.update(
                frames=dict(frames), now_s=now_s, roster=tuple(roster),
                coverage=coverage, merge_observed=merge_observed,
            )
            return {"tracked": {c: f.timestamp_s for c, f in frames.items()}}

        harness._hybrid_builder = builder
        (sample,), _ = harness.observe()
        assert sample.hybrid == {"tracked": {"c0": 0.0}}
        assert seen["roster"] == ("c0", "c1")
        assert seen["coverage"] == 0.5
        assert seen["merge_observed"] is False
        assert set(seen["frames"]) == {"c0"}
        # frames/coverage stay real regardless of the predictor
        assert sample.frames == {"c0": 0.0}
        assert sample.meta["coverage"] == 0.5
        harness.close()


class TestPL04dResumeLoadState:
    """PL-04d (e): resume is the ONLY master loadState, and the resumed
    process launches fresh, loads ONCE, and can execute control ticks."""

    def test_resume_launches_and_loadstates_exactly_once(self, tmp_path, monkeypatch):
        rig = MasterRig(monkeypatch, tmp_path, run_id="rs1")
        harness = rig.make_harness()
        for _ in range(4):  # t -> 2.0
            harness.step_physics()
        saved = harness.save_state(tmp_path / "rs_state")
        t_saved = harness.now_s()
        harness.close()

        # the resumed run: a FRESH process (no start() call) whose
        # load_state launches, wires the control write, and loads ONCE
        rig2 = MasterRig(
            monkeypatch, tmp_path, run_id="rs2", allow_master_loadstate=True
        )
        harness2 = rig2.make_harness(start=False)
        assert rig2.traci.master_loads == []
        harness2.load_state(saved)
        assert len(rig2.traci.master_loads) == 1
        assert harness2.now_s() == t_saved
        assert harness2._control_write is not None
        harness2.step_physics()  # T-B: the first resumed control tick AT t_saved
        harness2.step_physics()  # 2.5 s: physics tick, no write
        owner = rig2.master_owner
        assert [s["now_s"] for s in owner.steps] == [t_saved]
        harness2.close()


class TestPL04dRealMasterRunner:
    """The full ClosedLoopRunner over the REAL harness (faked surfaces):
    the control rows land in dispatcher_rows.jsonl, the damping plan is
    executed every control tick, release runs at the end, and the master
    never loadStates outside resume."""

    def test_full_run_control_rows_release_and_no_loadstate(self, tmp_path, monkeypatch):
        rig = MasterRig(monkeypatch, tmp_path, run_id="rr")
        harness = rig.make_harness(start=False)  # the runner starts it
        components = SimpleNamespace(
            make_harness=lambda context: harness,
            make_world_model=lambda context: make_model(TWO_CAV),
            roster=TWO_CAV,
            planner_config=PlannerConfig(margin_floor_s=0.0),
            initial_plan=None,
            sim_seconds_fn=None,
        )
        config = ClosedLoopConfig(
            out_root=tmp_path,
            run_id="rr",
            world="w-ok",
            horizon_s=4.0,
            warmup_s=0.0,
            decision_period_s=30.0,
            predictor="mock",
            lookahead_horizon_s=30.0,
            max_sim_s=1e9,
        )
        receipt = ClosedLoopRunner(config, components).run()
        assert receipt["exit_status"] == "complete"
        owner = rig.master_owner
        # the t=0 decision keeps the committed /3 default (native -- the
        # sensor fake reports no merge inputs -> abstain), so the session
        # steps onto every control tick but the owner writes NO command
        # (native selection; the loop steps onto the horizon tick before
        # the exit check)
        assert [s["now_s"] for s in owner.steps] == [0.0, 1.0, 2.0, 3.0]
        assert owner.issued == []
        assert all(s["native_ids"] == {"c0", "c1"} for s in owner.steps)
        rows = [
            json.loads(line)
            for line in (tmp_path / "rr" / "dispatcher_rows.jsonl").read_text("utf-8").splitlines()
            if line.strip()
        ]
        control = [r for r in rows if r.get("source") == "master_control_tick"]
        assert [(r["t_s"], r["cav"]) for r in control] == [
            (t, c) for t in (0.0, 1.0, 2.0, 3.0) for c in ("c0", "c1")
        ]
        # the observation the decision ran on was REAL: full frame coverage
        # from the sensor fake, merge unobserved (no RSU inputs) -> abstain
        decision = next(
            json.loads(line)
            for line in (tmp_path / "rr" / "decisions.jsonl").read_text("utf-8").splitlines()
            if line.strip()
        )
        assert decision["coverage"] == [1.0]
        assert decision["merge_observed"] == [False]
        assert decision["abstain"] is True
        # (b) release ran at the normal end, then the harness closed
        assert len(owner.release_calls) == 1
        assert harness.proc is None
        # (e) a straight-through run never loadStates the master
        assert rig.traci.master_loads == []
        # speedMode 27 was configured on the master roster
        assert rig.traci.speed_modes == {"c0": CAV_SPEED_MODE, "c1": CAV_SPEED_MODE}


class TestPL04dRealPathAudit:
    """PL-04d: the explicit real-path audit -- no stub words in any
    RealMasterHarness / build_real_components body (docstrings included)."""

    def test_no_notimplementederror_placeholder_or_marker_level_in_real_path(self):
        sources = []
        for name, member in vars(RealMasterHarness).items():
            if isinstance(member, (classmethod, staticmethod)):
                member = member.__func__
            elif isinstance(member, property):
                member = member.fget
            if inspect.isfunction(member):
                sources.append((f"RealMasterHarness.{name}", inspect.getsource(member)))
        sources.append(("SessionControlWrite", inspect.getsource(SessionControlWrite)))
        sources.append(("build_real_components", inspect.getsource(build_real_components)))
        audited = {label for label, _ in sources}
        # the audit really did inspect every real-master-path body
        for method in (
            "__init__", "_launch", "start", "close", "now_s", "step_physics",
            "_run_control_tick", "drain_control_rows", "release_control",
            "observe", "apply_plan", "save_state", "load_state",
        ):
            assert f"RealMasterHarness.{method}" in audited
        for banned in ("notimplementederror", "placeholder", "marker-level"):
            for label, src in sources:
                assert banned not in src.lower(), f"{label} contains {banned!r}"

    def test_real_path_checklist_covers_the_real_components(self):
        text = (
            Path(__file__).parents[2] / "output" / "planner_v1" / "PL04_CLOSED_LOOP_DESIGN.md"
        ).read_text(encoding="utf-8")
        section = text.split("## REAL_PATH_CHECKLIST", 1)[1]
        for component in (
            "master launch", "seed", "master control write", "observe",
            "checkpoint save/restore", "lookahead factory",
            "branch control write", "receipt",
        ):
            assert component in section, f"REAL_PATH_CHECKLIST misses {component!r}"



# ------------ (PL-04g) real-subprocess launcher seams over a python child
class TestRealSubprocessLaunchers:
    """PL-04g: the REAL Popen paths of the closed-loop master and branch
    launchers are exercised with a harmless ``python -c`` child -- the SUMO
    binary/command is monkeypatched, NEVER Popen itself (faked Popen hid a
    real launcher crash, so these seams now prove: a real child starts,
    exits 0, and the launcher's own start/close lifecycle reaps it). No
    SUMO; only the launchers' OWN Popen children are ever waited."""

    CHILD = "import sys; sys.stdout.write('ok'); sys.stderr.write('err')"

    @staticmethod
    def _install_traci_stub(monkeypatch):
        """A sys.modules['traci'] stub: init records + getConnection hands
        out one dummy connection (Popen itself is NEVER patched)."""
        import types

        conn = types.SimpleNamespace(close=lambda wait=True: None)

        class _TraciStub:
            calls: list = []

            @staticmethod
            def init(**kwargs):
                _TraciStub.calls.append(kwargs)

            @staticmethod
            def getConnection(label):
                return conn

        monkeypatch.setitem(sys.modules, "traci", _TraciStub)
        return conn, _TraciStub

    def test_real_master_harness_launch_starts_and_reaps_a_real_child(
        self, tmp_path, monkeypatch
    ):
        conn, stub = self._install_traci_stub(monkeypatch)
        harness = RealMasterHarness(
            sumo_binary=sys.executable,
            scenario_cfg="<payload below>",  # replaced by the -c payload
            scenario_routes=["unused"],
            roster=("c0",),
            adapter_config=None,
            seed=7,
        )
        harness.scenario_cfg = self.CHILD  # the command payload, not Popen
        returned = harness._launch()
        # PL-04l: _launch returns the master's OWN Connection object from
        # getConnection('pl04') -- never the traci module -- and the init
        # does NOT switch the module's current connection
        assert returned is conn
        assert stub.calls[0]["label"] == "pl04"
        assert stub.calls[0]["doSwitch"] is False
        proc = harness.proc
        assert proc is not None and proc.pid > 0  # a REAL child started
        record = harness.launch_record
        assert record["kind"] == "master" and record["pid"] == proc.pid
        assert "--seed" in record["cmd"] and record["cmd"][0] == sys.executable
        assert stub.calls and stub.calls[0]["proc"] is proc
        harness._traci = conn
        harness.close()  # own child only: traci stub close + wait
        assert proc.wait(timeout=0) == 0  # exits 0, reaped by close
        assert harness.proc is None

    def test_real_branch_factory_launch_starts_and_reaps_a_real_child(
        self, tmp_path, monkeypatch
    ):
        conn, _stub = self._install_traci_stub(monkeypatch)
        factory = RealBranchProcessFactory(
            sumo_binary=sys.executable,
            scenario_cfg="<payload below>",
            seed=7,
        )
        factory.scenario_cfg = self.CHILD  # the command payload, not Popen
        got = factory.start_process(label="pl04g")
        assert got is conn
        # PL-04l: a branch init never switches the module's current
        # connection, and the branch is used only via its own Connection
        assert _stub.calls[0]["doSwitch"] is False
        assert _stub.calls[0]["label"] == "pl04_branch1_pl04g"
        assert len(factory.launched) == 1
        record = factory.launched[0]
        assert record["kind"] == "branch" and record["cmd"][0] == sys.executable
        assert "--seed" in record["cmd"]
        (_label, proc) = factory._children[id(got)]
        assert proc.pid > 0  # a REAL child started
        factory.close_process(got)  # traci stub close + wait on its own child
        assert proc.wait(timeout=0) == 0  # exits 0, reaped by close_process
