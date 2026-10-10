"""Safety checks: absolute (collision / TTC floor) and relative owner rule."""
import dataclasses  # noqa: E402  (see the kit-path note below)
import sys  # noqa: E402

import numpy as np
import pytest

from traffic_shepherd.planner import (  # noqa: E402
    MockWorldModel,
    PlannerConfig,
    WallSpec,
    safety_check,
)
from traffic_shepherd.planner.safety import (  # noqa: E402
    fail_closed_mask,
    native_collision_note,
)
from traffic_shepherd.planner.paths import rd_wt_root  # noqa: E402
from tests.planner._helpers import make_samples, plan, plan_key  # noqa: E402

# The WD-01 conformance kit lives in a read-only sibling workspace; its tree
# goes on sys.path before the imports that need it (fail-closed agreement).
KIT_ROOT = rd_wt_root() / "wd01b" / "output" / "wd01"
KIT_AVAILABLE = KIT_ROOT.is_dir()
if KIT_AVAILABLE:
    if str(KIT_ROOT) not in sys.path:
        sys.path.insert(0, str(KIT_ROOT))
    try:
        from predict import contract as kit_contract
        from predict.safety import safety_veto as kit_safety_veto
    except ImportError:  # pragma: no cover - kit present but broken
        KIT_AVAILABLE = False


def make_pred(script=None, native=None, k=3, world=None):
    world = world or MockWorldModel(cav_ids=("c0",))
    if script:
        world.safety_script[plan_key(plan({"c0": "damping"}))] = script
    if native:
        world.native_safety = native
    return world.predict_many(
        make_samples(k), [plan({"c0": "damping"})], trajectory_s=10)[0]


def rel_config(**over):
    return PlannerConfig(trajectory_s=10.0,
                         rel_safety_tol_s=over.pop("rel_safety_tol_s", 0.0),
                         rel_safety_quantile=over.pop("rel_safety_quantile", 0.90),
                         **over)


class TestAbsoluteSafety:
    def test_default_mock_plan_is_safe(self):
        verdict = safety_check(make_pred(), rel_config())
        assert verdict.ok and verdict.abs_ok and verdict.rel_ok
        assert verdict.abs_safe_cells == verdict.abs_total_cells == 3

    def test_collision_in_any_cell_vetoes(self):
        pred = make_pred({"collision": np.array([False, True, False])})
        verdict = safety_check(pred, rel_config())
        assert not verdict.ok
        assert not verdict.abs_ok
        assert verdict.abs_safe_cells == 2

    def test_min_ttc_below_floor_in_any_cell_vetoes(self):
        pred = make_pred({"min_ttc_s": np.array([2.0, 1.49, 9.0])})
        assert not safety_check(pred, rel_config()).abs_ok

    def test_min_ttc_equal_to_floor_is_safe(self):
        pred = make_pred({"min_ttc_s": 1.5})
        assert safety_check(pred, rel_config()).abs_ok

    def test_nan_min_ttc_is_no_pair_and_passes_floor(self):
        # WD-01 v1.1 clarification 7/§9: NaN min_ttc_s = no closing pair in a
        # VALID cell -> the TTC floor passes there.
        pred = make_pred({"min_ttc_s": np.array([2.0, np.nan, 2.0]),
                          "has_closing_pair": np.array([True, False, True])})
        verdict = safety_check(pred, rel_config())
        assert verdict.abs_ok and verdict.ok
        assert verdict.abs_safe_cells == 3
        assert verdict.vetoed_cells == 0

    def test_nan_min_ttc_with_pair_vetoes(self):
        # NaN must never paper over a real closing pair
        pred = make_pred({"min_ttc_s": np.array([2.0, np.nan, 2.0]),
                          "has_closing_pair": np.array([True, True, True])})
        assert not safety_check(pred, rel_config()).abs_ok

    def test_nan_min_ttc_with_collision_still_vetoes(self):
        pred = make_pred({"min_ttc_s": np.array([2.0, np.nan, 2.0]),
                          "has_closing_pair": np.array([True, False, True]),
                          "collision": np.array([False, True, False])})
        verdict = safety_check(pred, rel_config())
        assert not verdict.abs_ok and verdict.abs_safe_cells == 2

    def test_inf_min_ttc_remains_a_hard_veto(self):
        pred = make_pred({"min_ttc_s": np.array([2.0, np.inf, 2.0]),
                          "has_closing_pair": np.array([True, False, True])})
        assert not safety_check(pred, rel_config()).abs_ok

    def test_abs_quantile_relaxes_the_all_cells_rule(self):
        pred = make_pred({"min_ttc_s": np.array([9.0, 9.0, 9.0, 0.1])}, k=4)
        strict = rel_config(abs_safety_quantile=1.0)
        relaxed = rel_config(abs_safety_quantile=0.75)
        assert not safety_check(pred, strict).abs_ok
        assert safety_check(pred, relaxed).abs_ok


class TestRelativeSafety:
    def test_equal_to_native_passes(self):
        native = {"ttc_lt3_exposure_s": 2.0, "hard_brake_s": 1.0}
        pred = make_pred({"ttc_lt3_exposure_s": 2.0, "hard_brake_s": 1.0},
                         native=native)
        verdict = safety_check(pred, rel_config())
        assert verdict.rel_ok and verdict.ok
        assert verdict.relative[0].mean_increase_s == pytest.approx(0.0)

    def test_mean_increase_above_tol_vetoes_even_with_97pct_within(self):
        native = {"ttc_lt1p5_exposure_s": 0.0}
        # 3 of 100 cells +100: fraction within tol 0.97 >= 0.90, but mean
        # increase 3.0 > tol 1.0 -> the MEAN rule vetoes. (T-A: scripted on
        # the PRIMARY 1.5 s metric -- the config default.)
        pred = make_pred({"ttc_lt1p5_exposure_s": np.array([0.0] * 97 + [100.0] * 3)},
                         native=native, k=100)
        verdict = safety_check(pred, rel_config(rel_safety_tol_s=1.0))
        assert verdict.relative[0].fraction_within_tol == pytest.approx(0.97)
        assert verdict.relative[0].mean_increase_s == pytest.approx(3.0)
        assert not verdict.relative[0].ok and not verdict.rel_ok

    def test_mean_exactly_at_tol_passes(self):
        native = {"ttc_lt1p5_exposure_s": 0.0}
        # one +100 cell: mean increase exactly 1.0 == tol; 99% within tol.
        pred = make_pred({"ttc_lt1p5_exposure_s": np.array([0.0] * 99 + [100.0])},
                         native=native, k=100)
        verdict = safety_check(pred, rel_config(rel_safety_tol_s=1.0))
        assert verdict.relative[0].mean_increase_s == pytest.approx(1.0)
        assert verdict.relative[0].ok

    def test_cell_fraction_boundary_89_vs_90_percent(self):
        # K=100 cells; increase 5.0 in the tail, 0 elsewhere; tol 1.0 keeps the
        # mean rule satisfied (0.55 / 0.5 <= 1) so the fraction rule decides.
        def pred_with_failures(n_bad):
            script = {"ttc_lt1p5_exposure_s": np.array(
                [0.0] * (100 - n_bad) + [5.0] * n_bad)}
            return make_pred(script, k=100)

        verdict_89 = safety_check(pred_with_failures(11),
                                  rel_config(rel_safety_tol_s=1.0))
        assert verdict_89.relative[0].fraction_within_tol == pytest.approx(0.89)
        assert not verdict_89.rel_ok
        verdict_90 = safety_check(pred_with_failures(10),
                                  rel_config(rel_safety_tol_s=1.0))
        assert verdict_90.relative[0].fraction_within_tol == pytest.approx(0.90)
        assert verdict_90.rel_ok

    def test_hard_brake_gated_independently(self):
        native = {"ttc_lt3_exposure_s": 0.0, "hard_brake_s": 0.0}
        pred = make_pred({
            "ttc_lt3_exposure_s": np.zeros(3),
            "hard_brake_s": np.array([0.0, 0.0, 5.0]),
        }, native=native)
        verdict = safety_check(pred, rel_config())
        assert verdict.relative[0].ok  # exposure fine
        assert not verdict.relative[1].ok  # hard brake vetoed
        assert not verdict.ok

    def test_nan_increase_vetoes(self):
        native = {"ttc_lt1p5_exposure_s": 0.0}
        pred = make_pred({"ttc_lt1p5_exposure_s": np.array([1.0, np.nan, 1.0])},
                         native=native)
        assert not safety_check(pred, rel_config()).rel_ok

    def test_unsafe_native_does_not_mask_plan_veto(self):
        # If NATIVE itself collides, the plan's own absolute check still uses
        # the plan's cells; relative deltas stay zero here.
        pred = make_pred({"collision": True}, native={"collision": True})
        assert not safety_check(pred, rel_config()).abs_ok


# --- PL01-E2: contract v1.1 FAIL CLOSED (§8.7/§9) -------------------


def _v11(pred):
    """A prediction labelled as a contract-1.1 output."""
    meta = {**pred.meta, "contract_version": "wd01_predict/1.1"}
    return dataclasses.replace(pred, meta=meta)


def _wall_plan(world, script):
    wall = WallSpec(wall_id="w0", members=("c0",), speed_mps=16.0,
                    phase="assemble", hold_s=60.0)
    candidate = plan({}, walls=[wall])
    world.wall_script["w0"] = script
    return world.predict_many(
        make_samples(3), [candidate],
        controller_state={"c0": {"action": "wall", "wall_id": "w0",
                                 "wall_phase": "assemble", "locked": True,
                                 "time_in_phase_s": 20.0,
                                 "wall_speed_mps": 16.0}},
        trajectory_s=10)[0]


class TestFailClosed:
    """§9: the vetoes fire BEFORE min_ttc is ever read."""

    def test_invalid_cell_with_nan_min_ttc_vetoes(self):
        pred = make_pred({"min_ttc_s": np.array([2.0, np.nan, 2.0]),
                          "cell_valid": np.array([True, False, True]),
                          "has_closing_pair": np.array([True, False, True])})
        verdict = safety_check(pred, rel_config())
        assert not verdict.abs_ok and not verdict.ok
        assert verdict.vetoed_cells == 1

    def test_mixed_valid_invalid_cells_veto(self):
        pred = make_pred({"cell_valid": np.array([True, False, True])})
        verdict = safety_check(pred, rel_config())
        assert not verdict.abs_ok
        assert verdict.vetoed_cells == 1
        assert verdict.abs_safe_cells == 2

    def test_native_cell_invalid_vetoes(self):
        world = MockWorldModel(cav_ids=("c0",))
        world.native_safety = {
            "native_cell_valid": np.array([True, True, False])}
        pred = world.predict_many(make_samples(3),
                                  [plan({"c0": "damping"})],
                                  trajectory_s=10)[0]
        verdict = safety_check(pred, rel_config())
        assert not verdict.abs_ok
        assert verdict.vetoed_cells == 1

    def test_nan_plan_delay_vetoes(self):
        world = MockWorldModel(cav_ids=("c0",))
        world.plan_delay_override[plan_key(plan({"c0": "damping"}))] = \
            np.array([np.nan, 100.0, 100.0])
        pred = world.predict_many(make_samples(3),
                                  [plan({"c0": "damping"})],
                                  trajectory_s=10)[0]
        verdict = safety_check(pred, rel_config())
        assert not verdict.abs_ok
        assert verdict.vetoed_cells >= 1

    def test_nan_native_delay_vetoes(self):
        world = MockWorldModel(cav_ids=("c0",))
        world.plan_dvn_override[plan_key(plan({"c0": "damping"}))] = \
            np.array([np.nan, 0.0, 0.0])
        pred = world.predict_many(make_samples(3),
                                  [plan({"c0": "damping"})],
                                  trajectory_s=10)[0]
        verdict = safety_check(pred, rel_config())
        assert not verdict.abs_ok

    def test_wall_safety_abort_vetoes_plan(self):
        world = MockWorldModel(cav_ids=("c0",))
        pred = _wall_plan(world, {"safety_abort": True, "abort_reason": 1})
        verdict = safety_check(pred, rel_config())
        assert not verdict.ok
        assert verdict.vetoed_cells == 3

    def test_v11_prediction_missing_diagnostics_vetoes_everything(self):
        pred = _v11(make_pred())  # healthy numbers, but fields stripped
        pred = dataclasses.replace(pred, cell_valid=None,
                                   has_closing_pair=None)
        verdict = safety_check(pred, rel_config())
        assert not verdict.abs_ok and not verdict.ok
        assert verdict.vetoed_cells == 3  # every cell, never defaulted True

    def test_v10_style_nan_min_ttc_vetoes_strict_legacy(self):
        pred = make_pred({"min_ttc_s": np.array([2.0, np.nan, 2.0])})
        pred = dataclasses.replace(pred, cell_valid=None,
                                   has_closing_pair=None)  # 1.0: no diagnostics
        assert not safety_check(pred, rel_config()).abs_ok


# --- PL-06: native_collision_vetoes_plan (owner decision 2026-09-27) --


class TestNativeCollisionFlag:
    """Option A (False, the owner default since 2026-09-27): a native
    collision does NOT veto -- the plan's own absolute gate (no plan
    collision, min TTC >= floor in all cells) and the relative rule vs
    native decide, and the native collision is recorded as a logged
    'native_collision' diagnostic. Option B (True) restores the
    pre-2026-09-27 veto via fail_closed_mask. Every other native
    fail-closed term stays fail-closed."""

    @staticmethod
    def _native_collision_pred():
        # plan itself is clean (no collision, min_ttc 10.0 everywhere);
        # the NATIVE reference collides in cell 1 only
        return make_pred(native={"collision": np.array([False, True, False])})

    def test_default_false_collision_free_plan_passes_and_is_logged(self):
        # owner decision 2026-09-27 (option A): the DEFAULT lets a plan
        # whose own gates pass through despite the native collision, and
        # the collision is the logged diagnostic
        pred = self._native_collision_pred()
        cfg = rel_config()  # flag defaults False since 2026-09-27
        assert cfg.native_collision_vetoes_plan is False
        assert not bool(fail_closed_mask(pred, cfg).any())
        verdict = safety_check(pred, cfg)
        assert verdict.ok and verdict.abs_ok and verdict.rel_ok
        assert verdict.vetoed_cells == 0
        assert verdict.native_collision_cells == 1

    def test_explicit_true_restores_option_b_veto(self):
        pred = self._native_collision_pred()
        cfg = rel_config(native_collision_vetoes_plan=True)
        assert bool(fail_closed_mask(pred, cfg)[0, 1])
        verdict = safety_check(pred, cfg)
        assert not verdict.ok and not verdict.abs_ok
        assert verdict.vetoed_cells == 1
        assert verdict.native_collision_cells == 1

    def test_false_flag_same_plan_passes_and_collision_is_logged(self):
        pred = self._native_collision_pred()
        cfg = rel_config(native_collision_vetoes_plan=False)
        assert not bool(fail_closed_mask(pred, cfg).any())
        verdict = safety_check(pred, cfg)
        assert verdict.ok and verdict.abs_ok and verdict.rel_ok
        assert verdict.vetoed_cells == 0
        # the native collision is still recorded as the diagnostic
        assert verdict.native_collision_cells == 1
        assert native_collision_note(pred) == "native_collision cells=1"

    def test_false_flag_plan_collision_still_vetoes(self):
        pred = make_pred({"collision": np.array([False, True, False])},
                         native={"collision": np.array([False, True, False])})
        cfg = rel_config(native_collision_vetoes_plan=False)
        verdict = safety_check(pred, cfg)
        assert not verdict.ok and not verdict.abs_ok
        assert verdict.vetoed_cells == 1  # the PLAN's collision cell

    def test_false_flag_plan_ttc_below_floor_still_vetoes(self):
        pred = make_pred({"min_ttc_s": np.array([2.0, 1.0, 2.0])},
                         native={"collision": np.array([False, True, False])})
        cfg = rel_config(native_collision_vetoes_plan=False)
        assert not safety_check(pred, cfg).abs_ok

    def test_false_flag_relative_rule_vs_native_still_applies(self):
        pred = make_pred({"ttc_lt1p5_exposure_s": np.array([1.0, 1.0, 1.0])},
                         native={"collision": np.array([False, True, False]),
                                 "ttc_lt1p5_exposure_s": 0.0})
        cfg = rel_config(native_collision_vetoes_plan=False)
        verdict = safety_check(pred, cfg)
        assert verdict.abs_ok  # the native collision no longer vetoes
        assert not verdict.rel_ok and not verdict.ok

    def test_false_flag_native_invalid_cell_still_fails_closed(self):
        world = MockWorldModel(cav_ids=("c0",))
        world.native_safety = {
            "native_cell_valid": np.array([True, False, True]),
            "collision": np.array([False, True, False]),
        }
        pred = world.predict_many(make_samples(3),
                                  [plan({"c0": "damping"})],
                                  trajectory_s=10)[0]
        cfg = rel_config(native_collision_vetoes_plan=False)
        verdict = safety_check(pred, cfg)
        assert not verdict.ok
        assert verdict.vetoed_cells == 1  # the invalid NATIVE cell
        assert verdict.native_collision_cells == 1

    def test_false_flag_nan_native_delay_still_fails_closed(self):
        world = MockWorldModel(cav_ids=("c0",))
        world.plan_dvn_override[plan_key(plan({"c0": "damping"}))] = \
            np.array([np.nan, 0.0, 0.0])
        world.native_safety = {"collision": np.array([False, True, False])}
        pred = world.predict_many(make_samples(3),
                                  [plan({"c0": "damping"})],
                                  trajectory_s=10)[0]
        assert not safety_check(pred, rel_config(native_collision_vetoes_plan=False)).ok

    def test_mask_without_config_keeps_native_collision_veto(self):
        # the historical/kit-reference signature: no config -> the native
        # collision vetoes exactly as before PL-06
        pred = self._native_collision_pred()
        assert bool(fail_closed_mask(pred)[0, 1])


@pytest.mark.skipif(not KIT_AVAILABLE,
                    reason=f"WD-01 kit not present at {KIT_ROOT}")
class TestKitSafetyVetoAgreement:
    """The planner's fail-closed mask must agree with the kit's reference
    safety_veto on the same (converted) predictions."""

    @staticmethod
    def _converted(pred):
        from traffic_shepherd.planner.contract_adapter import (
            ContractPredictorAdapter,
        )

        adapter = ContractPredictorAdapter(contract_version="wd01_predict/1.1")
        return adapter._prediction_out(_v11(pred), kit_contract, 0.0)

    def test_agreement_on_scenarios(self):
        scenarios = {
            "default": make_pred(),
            "invalid_cell": make_pred({"cell_valid": np.array(
                [True, False, True])}),
            "native_invalid_cell": None,  # built below
            "collision": make_pred({"collision": np.array(
                [False, True, False])}),
            "nan_delay": None,  # built below
            "safety_abort": None,  # built below
        }
        native_world = MockWorldModel(cav_ids=("c0",))
        native_world.native_safety = {
            "native_cell_valid": np.array([True, False, True])}
        scenarios["native_invalid_cell"] = native_world.predict_many(
            make_samples(3), [plan({"c0": "damping"})], trajectory_s=10)[0]

        nan_world = MockWorldModel(cav_ids=("c0",))
        nan_world.plan_delay_override[plan_key(plan({"c0": "damping"}))] = \
            np.array([np.nan, 100.0, 100.0])
        scenarios["nan_delay"] = nan_world.predict_many(
            make_samples(3), [plan({"c0": "damping"})], trajectory_s=10)[0]

        abort_world = MockWorldModel(cav_ids=("c0",))
        scenarios["safety_abort"] = _wall_plan(
            abort_world, {"safety_abort": True, "abort_reason": 1})

        for name, pred in scenarios.items():
            kit_pred = self._converted(pred)
            reference = kit_safety_veto(kit_pred)
            planner = fail_closed_mask(pred)
            assert np.array_equal(reference, planner), name

    def test_plus_inf_plan_delay_vetoes(self):
        world = MockWorldModel(cav_ids=("c0",))
        world.plan_delay_override[plan_key(plan({"c0": "damping"}))] = \
            np.array([np.inf, 100.0, 100.0])
        pred = world.predict_many(make_samples(3),
                                  [plan({"c0": "damping"})],
                                  trajectory_s=10)[0]
        verdict = safety_check(pred, rel_config())
        assert not verdict.abs_ok and not verdict.ok
        assert verdict.vetoed_cells == 1

    def test_minus_inf_native_delay_vetoes(self):
        world = MockWorldModel(cav_ids=("c0",))
        world.plan_dvn_override[plan_key(plan({"c0": "damping"}))] = \
            np.array([-np.inf, 0.0, 0.0])
        pred = world.predict_many(make_samples(3),
                                  [plan({"c0": "damping"})],
                                  trajectory_s=10)[0]
        verdict = safety_check(pred, rel_config())
        assert not verdict.abs_ok
        assert verdict.vetoed_cells >= 1
