"""Safety RULE C (owner, PL-EXP1c): the absolute TTC floor is
min(ttc_floor_s, NATIVE min TTC of the same cell); ``absolute_ttc_rule``
"absolute" reproduces the old rule A exactly. Zero SUMO."""

import dataclasses

import numpy as np
import pytest

from tests.planner._helpers import make_samples, plan, plan_key
from tests.planner.test_ta_closed_loop_fixes import (
    VALIDATED_META,
    TestItem5FixedPolicyReferenceRun,
)
from traffic_shepherd.planner import MockWorldModel, PlannerConfig, safety_check
from traffic_shepherd.planner import closed_loop as cl
from traffic_shepherd.planner import loop as loop_mod
from traffic_shepherd.planner import safety as safety_mod
from traffic_shepherd.planner.safety import absolute_safety_mask

PLAN = plan({"c0": "damping"})


def _pred(script=None, native=None, k=3, v2=False):
    world = MockWorldModel(
        cav_ids=("c0",),
        **({"contract_version": "wd01_predict/2", "part_a_horizon_s": 30.0} if v2 else {}),
    )
    if v2:
        world.meta_script.update(VALIDATED_META)
    if script:
        world.safety_script[plan_key(PLAN)] = script
    if native:
        world.native_safety.update(native)
    return world.predict_many(make_samples(k), [PLAN], trajectory_s=10)[0]


def _cfg(**over):
    return PlannerConfig(trajectory_s=10.0, **over)


class TestConfig:
    def test_default_is_relative_to_native(self):
        assert PlannerConfig().absolute_ttc_rule == "relative_to_native"

    def test_invalid_rule_rejected(self):
        with pytest.raises(ValueError, match="absolute_ttc_rule"):
            PlannerConfig(absolute_ttc_rule="bogus")

    def test_invalid_tol_rejected(self):
        with pytest.raises(ValueError, match="ttc_rule_tol_s"):
            PlannerConfig(ttc_rule_tol_s=-1.0)


class TestWindowRuleC:
    def test_plan_below_1p5_but_at_native_is_admitted(self):
        pred = _pred({"min_ttc_s": 1.0}, {"min_ttc_s": 1.0})
        assert safety_check(pred, _cfg()).abs_ok
        assert not safety_check(pred, _cfg(absolute_ttc_rule="absolute")).abs_ok

    def test_plan_above_native_below_1p5_admitted(self):
        pred = _pred({"min_ttc_s": 1.2}, {"min_ttc_s": 1.0})
        assert safety_check(pred, _cfg()).abs_ok

    def test_plan_below_native_when_native_below_floor_vetoes(self):
        pred = _pred({"min_ttc_s": 0.9}, {"min_ttc_s": 1.0})
        verdict = safety_check(pred, _cfg())
        assert not verdict.abs_ok and not verdict.ok

    def test_native_at_or_above_floor_uses_1p5(self):
        pred = _pred({"min_ttc_s": 1.4}, {"min_ttc_s": 3.0})
        assert not safety_check(pred, _cfg()).abs_ok
        pred = _pred({"min_ttc_s": 1.5}, {"min_ttc_s": 3.0})
        assert safety_check(pred, _cfg()).abs_ok

    def test_native_nan_uses_1p5(self):
        pred = _pred({"min_ttc_s": 1.0}, {"min_ttc_s": np.nan})
        floors = safety_mod.native_floor_cells(pred, _cfg())
        assert np.allclose(floors, 1.5)

    def test_per_cell_floor(self):
        pred = _pred(
            {"min_ttc_s": np.array([1.0, 1.0, 1.0])},
            {"min_ttc_s": np.array([0.8, 1.0, 2.0])},
        )
        mask = absolute_safety_mask(pred, _cfg())
        assert mask.ravel().tolist() == [True, True, False]

    def test_fail_closed_rules_unchanged(self):
        nat = {"min_ttc_s": 1.0}
        pred = _pred({"min_ttc_s": 1.0, "collision": np.array([False, True, False])}, nat)
        assert safety_check(pred, _cfg()).abs_safe_cells == 2
        pred = _pred(
            {"min_ttc_s": np.array([2.0, np.nan, 2.0]), "has_closing_pair": True}, nat
        )
        assert not safety_check(pred, _cfg()).abs_ok
        pred = _pred(
            {"min_ttc_s": np.array([2.0, np.inf, 2.0]), "has_closing_pair": False}, nat
        )
        assert not safety_check(pred, _cfg()).abs_ok
        pred = _pred({"cell_valid": np.array([True, False, True])}, nat)
        assert not safety_check(pred, _cfg()).abs_ok

    def test_absolute_reproduces_rule_a_exactly(self):
        for ttc in (np.array([2.0, 1.49, 9.0]), 1.5, 1.0):
            pred = _pred({"min_ttc_s": ttc}, {"min_ttc_s": 0.5})
            a = safety_check(pred, _cfg(absolute_ttc_rule="absolute"))
            expect = np.all(np.asarray(pred.min_ttc_s) >= 1.5)
            assert a.abs_ok == bool(expect)
            assert a.native_floor_min_s is None


class TestFutureRuleC:
    def test_future_twin_same_behaviour(self):
        pred = _pred({"future_min_ttc_q05_s": 0.9}, {"native_future_min_ttc_q05_s": 0.8}, v2=True)
        assert not safety_mod.g1_future_floor_veto(pred, _cfg())
        assert safety_mod.g1_future_floor_veto(pred, _cfg(absolute_ttc_rule="absolute"))
        pred = _pred({"future_min_ttc_q05_s": 0.7}, {"native_future_min_ttc_q05_s": 0.8}, v2=True)
        assert safety_mod.g1_future_floor_veto(pred, _cfg())
        pred = _pred({"future_min_ttc_q05_s": 1.4}, {"native_future_min_ttc_q05_s": 2.5}, v2=True)
        assert safety_mod.g1_future_floor_veto(pred, _cfg())

    def test_future_native_nan_floor_is_1p5(self):
        pred = _pred(
            {"future_min_ttc_q05_s": 1.0},
            {"native_future_min_ttc_q05_s": np.nan, "native_future_has_closing_pair": False},
            v2=True,
        )
        assert safety_mod.g1_future_floor_veto(pred, _cfg())

    def test_no_native_twin_keeps_absolute(self):
        pred = _pred({"future_min_ttc_q05_s": 1.0}, {"native_future_min_ttc_q05_s": 1.0}, v2=True)
        stripped = dataclasses.replace(pred, native_future_min_ttc_q05_s=None)
        assert safety_mod.g1_future_floor_veto(stripped, _cfg())

    def test_unvalidated_future_head_no_floor(self):
        world = MockWorldModel(
            cav_ids=("c0",), contract_version="wd01_predict/2", part_a_horizon_s=30.0
        )
        world.safety_script[plan_key(PLAN)] = {"future_min_ttc_q05_s": 0.1}
        pred = world.predict_many(make_samples(3), [PLAN], trajectory_s=10)[0]
        assert not safety_mod.g1_future_floor_veto(pred, _cfg())


class TestRuleAShadow:
    def test_shadow_logged_and_differs_from_primary(self):
        pred = _pred({"min_ttc_s": 1.0}, {"min_ttc_s": 1.0})
        v = safety_check(pred, _cfg())
        assert v.absolute_ttc_rule == "relative_to_native"
        assert v.abs_ok and v.ok
        assert v.shadow_rule_a_abs_ok is False and v.shadow_rule_a_ok is False
        assert v.native_floor_min_s == pytest.approx(1.0)

    def test_shadow_matches_primary_under_absolute(self):
        pred = _pred({"min_ttc_s": 1.0}, {"min_ttc_s": 1.0})
        v = safety_check(pred, _cfg(absolute_ttc_rule="absolute"))
        assert v.ok is False and v.shadow_rule_a_ok is False

    def test_shadow_agrees_when_safe(self):
        pred = _pred({"min_ttc_s": 3.0}, {"min_ttc_s": 3.0})
        v = safety_check(pred, _cfg())
        assert v.ok and v.shadow_rule_a_ok

    def test_v2_future_shadow(self):
        pred = _pred({"future_min_ttc_q05_s": 0.9}, {"native_future_min_ttc_q05_s": 0.8}, v2=True)
        v = safety_check(pred, _cfg())
        assert v.future_ok and v.shadow_rule_a_future_ok is False


class TestVetoLogRule:
    def test_ttc_veto_entry_records_rule_and_native_floor(self):
        pred = _pred({"min_ttc_s": 1.0}, {"min_ttc_s": 1.2})
        stub = type("S", (), {"config": _cfg(), "_ttc_veto_log": {}})()
        loop_mod.Planner._note_ttc_vetoes(stub, [PLAN], [pred])
        (entry,) = stub._ttc_veto_log.values()
        assert entry["absolute_ttc_rule"] == "relative_to_native"
        assert entry["native_min_ttc_s"] == pytest.approx(1.2)
        assert entry["floor_applied_s"] == pytest.approx(1.2)
        assert entry["vetoed_under_rule"] is True
        assert entry["pair_class"] == "unknown_pair"


class TestFixedPolicyNeverCallsSafety:
    @pytest.mark.parametrize("policy", ["native", "damping", "pi"])
    def test_fixed_policy_run_never_evaluates_safety(self, tmp_path, monkeypatch, policy):
        calls = []

        def spy(name):
            def _f(*a, **k):
                calls.append(name)
                raise AssertionError(f"safety entry point {name} called in a fixed-policy run")

            return _f

        for mod in (safety_mod, loop_mod, cl):
            for name in ("safety_check", "absolute_safety_mask", "fail_closed_mask"):
                if hasattr(mod, name):
                    monkeypatch.setattr(mod, name, spy(f"{mod.__name__}.{name}"))
        receipt, _read_log, _predict, _config, _cav = TestItem5FixedPolicyReferenceRun._run(
            tmp_path, policy, f"rule-c-ref-{policy}"
        )
        assert calls == []
        assert receipt["decisions"] == 0
