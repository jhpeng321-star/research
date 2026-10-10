"""PL-11: planner adoption of the frozen predict contract v2.1 (Part A v2
general).

Spec (read-only, WD-01 worktree): PREDICT_CONTRACT_v2_1.md, sha256
ba52820a6559cfe766daa1be4aa29c479ec7b5c9e9760c1b5ac21106b2b361d2
(verified 2026-09-29; raw == LF-normalized). Planner obligations covered
here, mapped by letter:

(a) an M = 1 v2-general-shaped prediction (``part_a_scope`` 'tracked', the
    new per-cell fields present) flows through ``Planner.decide``: abstain
    counts over K, the switch rule pairs over the M*K cells, and the
    documented M = 1 margin behaviour (single cell -> std 0 -> margin ==
    margin_floor_s exactly; no invented uncertainty);
(b) a prediction WITHOUT the new fields still works, and perfect_sim's
    meta declares ``part_a_scope='all_vehicles'``;
(c) a cross-scope ``delay_part_a_s`` comparison raises (or skips) BY NAME
    (``PartAScopeMismatchError`` / ``part_a_scope_mismatch_reason``);
    totals always compare;
(d) closed-loop decisions.jsonl + RUN_RECEIPT carry the new fields when
    present and null them when absent -- the real-runner e2e helpers
    (tests/planner/test_pl04i_e2e_real_run.py) drive the REAL wiring with
    a mock predictor;
(e) arrivals None / empty are accepted (contract v2.1 section 2).

Zero simulation: mocks, fakes and the tiny fake traci world only. The WD
kit validator is NOT edited (WD-owned); the existing conformance test
already proves it accepts M = 1 (``report.M == 1``,
tests/planner/test_contract_conformance.py).
"""

import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from tests.planner import test_pl04i_e2e_real_run as e2e_mod
from tests.planner.test_perfect_sim_v2 import PLAN_A, Setup, make_sample
from traffic_shepherd.planner import PlannerConfig
from traffic_shepherd.planner.interface import (
    PART_A_V21_CELL_FIELDS,
    PlanCandidate,
    Prediction,
    part_a_scope,
)
from traffic_shepherd.planner.loop import Planner
from traffic_shepherd.planner.mock import MockWorldModel
from traffic_shepherd.planner.perfect_sim import SimpleArrivalDraws
from traffic_shepherd.planner.rules import (
    PartAScopeMismatchError,
    compare_part_a,
    paired_saving,
    part_a_scope_mismatch_reason,
    switch_check,
)

V2 = "wd01_predict/2"
CAVS = ("c0", "c1")
VALIDATED_META = {
    "future_min_ttc_validated": True,
    "future_min_ttc_validation_id": "pl11-validated",
    "relative_safety_validated": True,
    "relative_safety_validation_id": "pl11-validated",
}
V21_META = {
    "part_a_scope": "tracked",
    "part_a_model": "v2_general",
    "param_set_id": "sha256:pl11-params",
    "recon_schema_version": "wm30_hybrid_state/1",
}
V21_FIELDS = {
    "n_tracked": 12,
    "n_exited_tracked": 2,
    "n_free_road_assumed": 1,
    "untracked_arrivals_expected": 3,
}


# ------------------------------------------------------------------ helpers
def samples(k=3, observed=(True, False, False), arrivals="none", time_s=0.0):
    """K state samples; ``arrivals='none'`` carries the None payload
    (contract v2.1 section 2), 'empty' the empty tuple, anything else the
    option-A mapping payload."""
    payload = {"sample_id": "k0", "departures": ()}
    if arrivals == "none":
        payload = None
    elif arrivals == "empty":
        payload = ()
    out = []
    for i in range(k):
        out.append(
            SimpleNamespace(
                hybrid=None,
                arrivals=payload,
                meta={
                    "sample_id": f"k{i}",
                    "time_s": time_s,
                    "merge_observed": bool(observed[min(i, len(observed) - 1)]),
                    "coverage": 1.0,
                },
            )
        )
    return out


def v21_model(*, fields=V21_FIELDS, meta=None, cav=CAVS):
    """A validated contract-v2 mock (M = 1 by default) optionally emitting
    the v2.1 per-cell fields and meta."""
    model = MockWorldModel(
        cav,
        part_a_horizon_s=60.0,
        contract_version=V2,
        model_id="mock/pl11",
    )
    script = dict(VALIDATED_META)
    if meta is not None:
        script.update(meta)
    model.meta_script.update(script)
    if fields:
        model.part_a_v21_fields = dict(fields)
    return model


def v21_prediction(dvn, *, part_a=None, scope="tracked", k=3):
    """A hand-built M = 1 v2-general-shaped planner Prediction ([1, K]
    arrays, the four optional per-cell fields, meta part_a_scope)."""
    shape = (1, k)
    dvn_arr = np.full(shape, float(dvn), dtype=float)
    part_a = (
        np.full(shape, float(part_a))
        if part_a is not None
        else dvn_arr * 0.5
    )
    plan_total = dvn_arr + 100.0
    meta = {
        "contract_version": V2,
        "model_id": "v21_general/test",
        "member_ids": ["m0"],
        "sample_ids": [f"k{i}" for i in range(k)],
        "part_a_horizon_s": 30.0,
        "safety_window_s": 30.0,
        "horizon_s": 600.0,
        "future_min_ttc_validated": True,
        "future_min_ttc_validation_id": "pl11-validated",
        "relative_safety_validated": True,
        "relative_safety_validation_id": "pl11-validated",
    }
    if scope is not None:
        meta["part_a_scope"] = scope
    kwargs = (
        {name: np.full(shape, float(value)) for name, value in V21_FIELDS.items()}
        if scope == "tracked"
        else {}
    )
    return Prediction(
        delay_vs_native_s=dvn_arr,
        plan_delay_s=plan_total,
        native_delay_s=np.full(shape, 100.0),
        delay_part_a_s=part_a,
        delay_part_b_s=plan_total - part_a,
        collision=np.zeros(shape, dtype=bool),
        min_ttc_s=np.full(shape, 10.0),
        ttc_lt3_exposure_s=np.zeros(shape),
        native_ttc_lt3_exposure_s=np.zeros(shape),
        hard_brake_s=np.zeros(shape),
        native_hard_brake_s=np.zeros(shape),
        native_collision=np.zeros(shape, dtype=bool),
        native_min_ttc_s=np.full(shape, 10.0),
        meta=meta,
        **kwargs,
    )


# ------------------------------------------------------- (a) M = 1 through
class TestA_M1V2GeneralThroughPlanner:
    def test_prediction_datum_carries_v21_fields_m1(self):
        pred = v21_prediction(-10.0)
        assert pred.delay_vs_native_s.shape == (1, 3)
        for name, value in V21_FIELDS.items():
            assert getattr(pred, name).shape == (1, 3)
            assert getattr(pred, name)[0, 0] == float(value)
        assert part_a_scope(pred) == "tracked"
        assert set(PART_A_V21_CELL_FIELDS) == set(V21_FIELDS)

    def test_decide_switches_on_m1_v21_prediction(self):
        model = v21_model(meta=V21_META)
        model.action_delay = {c: {"damping": -10.0} for c in CAVS}
        planner = Planner(
            model,
            PlannerConfig(mode="closed_loop", decide_every_s=1.0, margin_floor_s=0.0),
        )
        decision = planner.decide(samples(k=3, observed=(True,) * 3), 0.0,
                                  controller_state={c: {"action": "native"} for c in CAVS})
        assert set(decision.plan.per_cav.values()) == {"damping"}
        # the chosen plan's own prediction rides along (logging-only) with
        # the v2.1 fields on the [1, K] grid
        chosen = decision.chosen_prediction
        assert chosen is not None
        assert chosen.n_tracked.shape == (1, 3)
        assert chosen.n_tracked[0, 0] == 12.0
        assert part_a_scope(chosen) == "tracked"
        # the switch verdicts paired over the M*K cells (median/agreement)
        switch_reasons = [r for r in decision.reasons if r.action == "switch"]
        assert switch_reasons and all(r.margin_s is not None for r in switch_reasons)

    def test_abstain_counts_over_k_not_mk(self):
        from traffic_shepherd.planner.rules import abstain_check

        model = v21_model()
        planner = Planner(model, PlannerConfig(mode="closed_loop"))
        # 1 of 3 samples observed -> fraction 1/3 < 0.8 -> abstain
        decision = planner.decide(samples(k=3, observed=(True, False, False)), 0.0,
                                  controller_state={c: {"action": "native"} for c in CAVS})
        assert decision.abstain is True
        verdict = abstain_check(samples(k=3, observed=(True, False, False)),
                                PlannerConfig())
        assert verdict.observed_fraction == pytest.approx(1.0 / 3.0)
        assert verdict.should_abstain is True
        # the abstain decision still carries the chosen (current) prediction
        assert decision.chosen_prediction is not None
        assert decision.chosen_prediction.n_tracked.shape == (1, 3)

    def test_documented_m1_margin_behaviour(self):
        cfg = PlannerConfig(margin_floor_s=0.0)  # margin_k 1.0
        current = v21_prediction(0.0)
        # constant saving across the K cells: std = 0 -> margin == floor
        constant = v21_prediction(-10.0)
        verdict = switch_check(current, constant, cfg)
        assert verdict.margin_s == pytest.approx(cfg.margin_floor_s)
        assert np.std(np.asarray(constant.delay_vs_native_s, dtype=float)) == 0.0
        assert verdict.passes is True
        # spread saving: margin == floor + margin_k * std (std over the
        # M*K cells; M = 1 means the K axis alone)
        saving = np.array([[10.0, 4.0, -2.0]])
        spread = replace(v21_prediction(0.0), delay_vs_native_s=-saving)
        expected_std = float(np.std(saving))
        verdict = switch_check(current, spread, cfg)
        assert verdict.median_saving_s == pytest.approx(4.0)
        assert verdict.margin_s == pytest.approx(
            cfg.margin_floor_s + cfg.margin_k * expected_std
        )
        # M = K = 1: a single deterministic cell, margin == floor EXACTLY
        one = v21_prediction(0.0, k=1)
        better = v21_prediction(-7.0, k=1)
        verdict = switch_check(one, better, cfg)
        assert verdict.margin_s == cfg.margin_floor_s
        assert verdict.passes is True


# ------------------------------------------- (b) without the fields / scope
class TestB_WithoutNewFieldsAndPerfectSimScope:
    def test_prediction_without_v21_fields_still_works(self):
        model = v21_model(fields=None, meta=None)
        planner = Planner(model, PlannerConfig(mode="closed_loop"))
        decision = planner.decide(samples(k=2, observed=(True, True)), 0.0,
                                  controller_state={c: {"action": "native"} for c in CAVS})
        assert decision.abstain is False
        chosen = decision.chosen_prediction
        assert chosen is not None
        for name in PART_A_V21_CELL_FIELDS:
            assert getattr(chosen, name) is None
        assert part_a_scope(chosen) is None

    def test_perfect_sim_meta_declares_all_vehicles(self, tmp_path):
        rig = Setup(tmp_path / "rig")
        (pred,), _ = rig.v2_predict(samples=make_sample(k=1))
        assert pred.meta["part_a_scope"] == "all_vehicles"
        assert part_a_scope(pred) == "all_vehicles"
        # the perfect simulator emits none of the optional per-cell fields
        for name in PART_A_V21_CELL_FIELDS:
            assert getattr(pred, name) is None


# --------------------------------------------------- (c) scope-safe compare
class TestC_ScopeSafePartAComparison:
    def _pair(self):
        tracked = v21_prediction(-4.0, part_a=10.0, scope="tracked")
        everyone = v21_prediction(-2.0, part_a=40.0, scope="all_vehicles")
        return tracked, everyone

    def test_cross_scope_raises_the_named_error(self):
        tracked, everyone = self._pair()
        with pytest.raises(
            PartAScopeMismatchError,
            match=r"part_a_scope_mismatch: tracked vs all_vehicles",
        ):
            compare_part_a(tracked, everyone)
        with pytest.raises(PartAScopeMismatchError, match="compare TOTAL delay"):
            compare_part_a(everyone, tracked)

    def test_cross_scope_skip_is_named_and_returns_none(self):
        tracked, everyone = self._pair()
        assert (
            part_a_scope_mismatch_reason(tracked, everyone)
            == "part_a_scope_mismatch: tracked vs all_vehicles"
        )
        assert compare_part_a(tracked, everyone, on_mismatch="skip") is None
        # one side unreported is still a mismatch (named)
        unreported = v21_prediction(-1.0, scope=None)
        assert (
            part_a_scope_mismatch_reason(unreported, tracked)
            == "part_a_scope_mismatch: unreported vs tracked"
        )
        with pytest.raises(ValueError, match="on_mismatch"):
            compare_part_a(tracked, everyone, on_mismatch="ignore")

    def test_equal_scopes_compare_and_totals_always_do(self):
        tracked, other = v21_prediction(-4.0, part_a=10.0), v21_prediction(
            -4.0, part_a=7.0
        )
        delta = compare_part_a(tracked, other)
        assert np.allclose(delta, 3.0)
        # both unreported: nothing distinguishes them -> comparable
        assert compare_part_a(v21_prediction(-1.0, scope=None),
                              v21_prediction(-1.0, scope=None)) is not None
        # TOTALS compare across ANY scopes (the planner's decision rule)
        tracked, everyone = self._pair()
        saving = paired_saving(tracked, everyone)
        assert np.allclose(saving, -2.0)


# --------------------------------- (d) closed-loop logging of the new fields
class TestD_ClosedLoopLogging:
    def _run_mock_real_path(self, tmp_path, monkeypatch, *, model, run_id):
        """One REAL-wiring closed-loop run (e2e helpers by import) whose
        lookahead predictor is the injected mock model."""
        base = e2e_mod.make_base_run_pl04j(tmp_path)
        from traffic_shepherd.planner.closed_loop import (
            ClosedLoopConfig,
            ClosedLoopRunner,
            build_real_components,
            resolve_base_run,
        )

        resolved = resolve_base_run(base)
        monkeypatch.setenv("SUMO_BINARY", sys.executable)
        monkeypatch.delenv("SUMO_HOME", raising=False)
        traci = e2e_mod._TinyTraci(lane=e2e_mod.PRE_MERGE_LANE)
        monkeypatch.setitem(sys.modules, "traci", traci)
        components = build_real_components(e2e_mod._real_args(base))
        components.make_world_model = lambda context: model
        config = ClosedLoopConfig(
            out_root=tmp_path,
            run_id=run_id,
            world="w_pl11",
            seed=e2e_mod.SEED,
            horizon_s=10.0,
            warmup_s=0.0,
            decision_period_s=10.0,
            predictor="mock",
            lookahead_horizon_s=5.0,
            lookahead_commit_s=None,
            k_samples=1,
            max_sim_s=500.0,
            base_run=resolved.base_run,
            base_t0_s=resolved.t0_s,
            base_state_sha256=resolved.state_sha256,
            max_decision_units=2,
            walls="off",
        )
        receipt = ClosedLoopRunner(config, components).run()
        return receipt, tmp_path / run_id

    def test_rows_and_receipt_carry_fields_when_present(self, tmp_path, monkeypatch):
        model = v21_model(meta=V21_META, cav=e2e_mod.CAVS)
        receipt, run_dir = self._run_mock_real_path(
            tmp_path, monkeypatch, model=model, run_id="pl11_present"
        )
        assert receipt["exit_status"] == "complete"
        assert receipt["decisions"] >= 1
        row = e2e_mod.read_jsonl(run_dir / "decisions.jsonl")[0]
        assert row["abstain"] is False  # near-merge fixture: the planner decides
        block = row["part_a_v21"]
        assert block["part_a_scope"] == "tracked"
        assert block["part_a_model"] == "v2_general"
        assert block["param_set_id"] == "sha256:pl11-params"
        assert block["recon_schema_version"] == "wm30_hybrid_state/1"
        assert block["plan"]["n_tracked"] == [[12.0]]
        assert block["plan"]["n_exited_tracked"] == [[2.0]]
        assert block["plan"]["n_free_road_assumed"] == [[1.0]]
        assert block["plan"]["untracked_arrivals_expected"] == [[3.0]]
        # the plan/native delay breakdown rides along (M = 1, K = 1); the
        # mock splits the plan total 50/50 between the parts
        assert block["plan"]["delay_part_a_s"][0][0] == pytest.approx(
            0.5 * block["plan"]["delay_total_s"][0][0]
        )
        assert block["native"]["delay_total_s"] == [[100.0]]
        meta_block = receipt["part_a_v21_meta"]
        assert meta_block == {
            "part_a_scope": "tracked",
            "part_a_model": "v2_general",
            "param_set_id": "sha256:pl11-params",
            "recon_schema_version": "wm30_hybrid_state/1",
        }

    def test_rows_and_receipt_null_fields_when_absent(self, tmp_path, monkeypatch):
        model = v21_model(fields=None, meta=None, cav=e2e_mod.CAVS)
        receipt, run_dir = self._run_mock_real_path(
            tmp_path, monkeypatch, model=model, run_id="pl11_absent"
        )
        assert receipt["exit_status"] == "complete"
        row = e2e_mod.read_jsonl(run_dir / "decisions.jsonl")[0]
        block = row["part_a_v21"]
        assert block is not None
        assert block["part_a_scope"] is None
        assert block["part_a_model"] is None
        assert block["param_set_id"] is None
        assert block["recon_schema_version"] is None
        for name in PART_A_V21_CELL_FIELDS:
            assert block["plan"][name] is None
        # the delay breakdown (v2 fields, not v2.1-optional) is still there
        assert block["plan"]["delay_part_a_s"] is not None
        assert block["plan"]["delay_total_s"] is not None
        assert block["native"]["delay_total_s"] == [[100.0]]
        assert receipt["part_a_v21_meta"] == {
            "part_a_scope": None,
            "part_a_model": None,
            "param_set_id": None,
            "recon_schema_version": None,
        }


# ------------------------------------------------ (e) arrivals None / empty
class TestE_OptionalArrivals:
    def test_mock_accepts_none_and_empty_arrivals(self):
        model = v21_model()
        plan = PlanCandidate(per_cav={c: "native" for c in CAVS})
        for payload in ("none", "empty"):
            preds = model.predict_many(samples(k=2, arrivals=payload), [plan])
            assert len(preds) == 1
            assert preds[0].delay_vs_native_s.shape == (1, 2)

    def test_planner_decide_with_none_arrivals(self):
        planner = Planner(v21_model(), PlannerConfig(mode="closed_loop"))
        decision = planner.decide(samples(k=2, arrivals="none"), 0.0,
                                  controller_state={c: {"action": "native"} for c in CAVS})
        assert decision.chosen_prediction is not None

    def test_simple_arrival_draws_none_payload(self):
        draws = SimpleArrivalDraws()
        assert draws.draw_id(None) == SimpleArrivalDraws.NO_ARRIVALS_ID
        assert tuple(draws.departures(None)) == ()
        assert tuple(draws.departures(())) == ()

    def test_perfect_sim_runs_with_none_arrivals(self, tmp_path):
        rig = Setup(tmp_path / "rig")
        none_samples = make_sample(k=1, arrivals=lambda i: None)
        (pred,), _ = rig.v2_predict(samples=none_samples, plans=(PLAN_A,))
        assert pred.delay_vs_native_s.shape == (1, 1)
        assert rig.runner.calls, "branches ran"
        assert all(call["draw_id"] == SimpleArrivalDraws.NO_ARRIVALS_ID
                   for call in rig.runner.calls)
