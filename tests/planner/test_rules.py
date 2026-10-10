"""Decision rules: switch (margin + agreement), native-return, abstain, keys."""
import numpy as np
import pytest

from traffic_shepherd.planner import (
    MockWorldModel,
    PlannerConfig,
    abstain_check,
    candidate_sort_key,
    native_return_check,
    paired_saving,
    safety_check,
    switch_check,
)
from tests.planner._helpers import make_samples, plan, plan_key

K100 = make_samples(100)


def two_predictions(current_script, candidate_script, k=100):
    """Predictions for current=c0:native and candidate=c0:damping, K=k.

    Scripts are delay_vs_native_s values (negative = saving vs native).
    """
    world = MockWorldModel(cav_ids=("c0",))
    world.plan_dvn_override[plan_key(plan({"c0": "native"}))] = current_script
    world.plan_dvn_override[plan_key(plan({"c0": "damping"}))] = candidate_script
    out = world.predict_many(make_samples(k), [plan({"c0": "native"}),
                                               plan({"c0": "damping"})],
                             trajectory_s=10)
    return out[0], out[1]


def base_config(**over):
    return PlannerConfig(margin_floor_s=0.0, trajectory_s=10.0, **over)


class TestPairedSaving:
    def test_per_cell_difference(self):
        current, candidate = two_predictions(10.0, np.full(100, 4.0))
        saving = paired_saving(current, candidate)
        assert saving.shape == (1, 100)
        assert saving == pytest.approx(6.0)


class TestSwitchRule:
    def test_margin_spread_scaled_pass_side(self):
        # saving = [2]*9 + [-2]: median 2, std 1.2 -> margin 1.2 -> passes
        saving = np.array([2.0] * 9 + [-2.0])
        current, candidate = two_predictions(0.0, -saving, k=10)
        verdict = switch_check(current, candidate, base_config())
        assert verdict.median_saving_s == pytest.approx(2.0)
        assert verdict.margin_s == pytest.approx(1.2)
        assert verdict.margin_ok and verdict.agreement_ok and verdict.passes

    def test_margin_spread_scaled_fail_side(self):
        # saving = [2]*9 + [-8]: median 2, std 3 -> margin 3 -> not > -> fail
        saving = np.array([2.0] * 9 + [-8.0])
        current, candidate = two_predictions(0.0, -saving, k=10)
        verdict = switch_check(current, candidate, base_config())
        assert verdict.median_saving_s == pytest.approx(2.0)
        assert verdict.margin_s == pytest.approx(3.0)
        assert not verdict.margin_ok and verdict.agreement_ok
        assert not verdict.passes

    def test_agreement_79_fails_80_passes(self):
        # 80/100 cells with saving +1, rest -0.01: median 1 > margin(std) ok,
        # so the agreement fraction decides the verdict.
        saving = np.array([1.0] * 80 + [-0.01] * 20)
        current, candidate = two_predictions(0.0, -saving)
        verdict = switch_check(current, candidate, base_config())
        assert verdict.agreement == pytest.approx(0.80)
        assert verdict.passes

        saving = np.array([1.0] * 79 + [-0.01] * 21)
        current, candidate = two_predictions(0.0, -saving)
        verdict = switch_check(current, candidate, base_config())
        assert verdict.agreement == pytest.approx(0.79)
        assert verdict.agreement_ok is False and not verdict.passes

    def test_candidate_safety_folds_into_passes(self):
        world = MockWorldModel(cav_ids=("c0",))
        world.plan_dvn_override[plan_key(plan({"c0": "damping"}))] = -1.0
        world.safety_script[plan_key(plan({"c0": "damping"}))] = {"collision": True}
        out = world.predict_many(make_samples(2),
                                 [plan({"c0": "native"}), plan({"c0": "damping"})],
                                 trajectory_s=10)
        # without the safety verdict the margin+agreement rule alone passes...
        assert switch_check(out[0], out[1], base_config()).passes
        # ...and folding the candidate's safety veto in blocks the switch
        verdict = switch_check(out[0], out[1], base_config(),
                               safety=safety_check(out[1], base_config()))
        assert verdict.margin_ok and verdict.agreement_ok and not verdict.passes

    def test_wall_bar_is_higher(self):
        # agreement exactly 5/6, spread tiny: passes default bar, fails wall bar
        saving = np.array([1.0] * 5 + [-0.01])
        current, candidate = two_predictions(0.0, -saving, k=6)
        plain = switch_check(current, candidate, base_config())
        assert plain.passes
        walled = switch_check(current, candidate, base_config(), wall=True)
        assert not walled.agreement_ok or not walled.margin_ok
        assert not walled.passes


class TestNativeReturn:
    def _current(self, script, unsafe=False):
        world = MockWorldModel(cav_ids=("c0",))
        world.plan_dvn_override[plan_key(plan({"c0": "damping"}))] = script
        if unsafe:
            world.safety_script[plan_key(plan({"c0": "damping"}))] = \
                {"collision": True}
        return world.predict_many(make_samples(2), [plan({"c0": "damping"})],
                                  trajectory_s=10)[0]

    def test_worse_than_native_triggers(self):
        verdict = native_return_check(self._current(3.0), base_config())
        assert verdict.should_return
        assert verdict.reason == "worse_than_native"
        assert verdict.median_delay_vs_native_s == pytest.approx(3.0)

    def test_exactly_native_does_not_trigger(self):
        verdict = native_return_check(self._current(0.0), base_config())
        assert not verdict.should_return and verdict.reason is None

    def test_better_than_native_does_not_trigger(self):
        verdict = native_return_check(self._current(-5.0), base_config())
        assert not verdict.should_return

    def test_unsafe_triggers_even_when_better(self):
        verdict = native_return_check(self._current(-5.0, unsafe=True),
                                      base_config())
        assert verdict.should_return and verdict.reason == "unsafe"


class TestAbstain:
    def test_fraction_boundary(self):
        samples = make_samples(10, observed=8)
        assert abstain_check(samples, base_config()).should_abstain is False
        assert abstain_check(samples, base_config()).observed_fraction == \
            pytest.approx(0.8)
        samples = make_samples(10, observed=7)
        assert abstain_check(samples, base_config()).should_abstain is True

    def test_threshold_is_strictly_less(self):
        assert abstain_check(make_samples(4, observed=4),
                             base_config(abstain_min_observed=1.0)).should_abstain \
            is False
        assert abstain_check(make_samples(4, observed=3),
                             base_config(abstain_min_observed=1.0)).should_abstain \
            is True

    def test_empty_samples_raise(self):
        with pytest.raises(ValueError):
            abstain_check([], base_config())


class TestCandidateSortKey:
    def test_median_decides_then_action_order(self):
        world = MockWorldModel(cav_ids=("c0",))
        world.plan_dvn_override[plan_key(plan({"c0": "native"}))] = -2.0
        world.plan_dvn_override[plan_key(plan({"c0": "damping"}))] = -2.0
        world.plan_dvn_override[plan_key(plan({"c0": "pi-external-u"}))] = -1.0
        out = world.predict_many(
            make_samples(1),
            [plan({"c0": "native"}), plan({"c0": "damping"}),
             plan({"c0": "pi-external-u"})], trajectory_s=10)
        keys = {a: candidate_sort_key(p, plan({"c0": a}))
                for a, p in zip(("native", "damping", "pi-external-u"), out)}
        # native and damping tie on median; native has the lower action rank
        assert keys["native"] < keys["damping"] < keys["pi-external-u"]

    def test_ids_sorted_before_action_comparison(self):
        world = MockWorldModel(cav_ids=("a", "b"))
        world.plan_dvn_override[plan_key(plan({"a": "damping", "b": "native"}))] = 0.0
        world.plan_dvn_override[plan_key(plan({"a": "native", "b": "damping"}))] = 0.0
        out = world.predict_many(
            make_samples(1),
            [plan({"a": "damping", "b": "native"}),
             plan({"a": "native", "b": "damping"})], trajectory_s=10)
        key_a = candidate_sort_key(out[0], plan({"a": "damping", "b": "native"}))
        key_b = candidate_sort_key(out[1], plan({"a": "native", "b": "damping"}))
        # same median: at the first sorted id ('a'), the lower-ranked action
        # (native) wins, so the plan with damping on 'a' loses
        assert key_b < key_a


def test_safety_check_used_by_rules_matches_direct_call():
    current, candidate = two_predictions(0.0, np.full(100, -1.0))
    verdict = switch_check(current, candidate, base_config(),
                           safety=safety_check(candidate, base_config()))
    assert verdict.passes
