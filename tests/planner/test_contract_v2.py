"""PL-05 tests: the planner side of predict contract v2 (wd01_predict/2).

Covers, per the brief's acceptance criteria: fail-closed future validity,
the p_future_collision absolute and paired-relative vetoes, the relative
rule extended to window+future, the G-1 refusal/override gate and the
conditional future q05 floor (with NaN handling and
future_has_closing_pair), the v2 wall assembly bar, the commit/continuation
dispatch, and v1/v1.1 behaviour unchanged.

Contract text (read-only): the WD-01 worktree's PREDICT_CONTRACT_v2.md,
sha256 db475371...0775 (post-review text: future_has_closing_pair is
REQUIRED, not optional). Interpretations are declared at
:func:`_interpretations`.
"""

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

from traffic_shepherd.planner import PlanCandidate, PlannerConfig
from traffic_shepherd.planner.interface import plan_key
from traffic_shepherd.planner.mock import MockWorldModel
from traffic_shepherd.planner.safety import (
    FutureSafetyRefusal,
    enforce_future_gate,
    fail_closed_mask,
    future_collision_veto,
    future_relative_verdicts,
    g1_future_floor_veto,
    safety_check,
    window_relative_note,
)
from traffic_shepherd.planner.loop import Planner

V2 = "wd01_predict/2"


def _samples(k=2, time_s=0.0):
    return [
        SimpleNamespace(
            hybrid=None,
            arrivals={"seed": i},
            meta={"sample_id": f"s{i}", "time_s": time_s, "merge_observed": True, "coverage": 1.0},
        )
        for i in range(k)
    ]


def _v2_world(**kw):
    world = MockWorldModel(("c0", "c1"), part_a_horizon_s=60.0, contract_version=V2, **kw)
    return world


_PLAN = {"c0": "damping", "c1": "native"}


def _v2_predict(
    world,
    *,
    script=None,
    native=None,
    meta=None,
    commit_s=30.0,
    continuation="planner",
    horizon_s=600.0,
):
    plan = PlanCandidate(per_cav=dict(_PLAN))
    if script:
        # safety_script is keyed by plan key: per-field values for THIS plan
        world.safety_script[plan_key(plan)] = dict(script)
    if native:
        world.native_safety.update(native)
    if meta:
        world.meta_script.update(meta)
    return world.predict_many(
        _samples(), [plan], horizon_s=horizon_s, commit_s=commit_s, continuation=continuation
    )[0]


def _interpretations():
    """Declared interpretations (PL-05).

    1. The on-disk contract sha256 is db475371...0775, NOT the dc2589... in
       the brief: the WM-40 review note records the contract UPDATE that
       makes future_has_closing_pair REQUIRED and supersedes the brief's
       optional support. We implement REQUIRED (missing -> fail closed);
       'with/without future_has_closing_pair' is exercised on the VALUE
       (NaN q05 passes where the pair flag is False, fails where True).
    2. The future floor (G-1, validated) applies to the PLAN's
       future_min_ttc_q05_s only; the native q05 has no floor (the contract's
       consumer rules name only the plan head), but the native NaN-with-pair
       validity rule IS enforced in the fail-closed mask.
    3. The future relative rule tests window diff + future diff per metric
       with the same mean and >= rel_safety_quantile cell tests; the verdict
       metric is labelled '<metric>+future'.
    4. p_future_collision NaN/inf fails closed (not a legible probability).
    5. The mode gate probes the CURRENT plan's prediction at decide() entry,
       before any decision reason is produced; the probe is reused (never
       recomputed). A controlled-experiment override is prepended to the
       decision's reasons.
    6. The v2 wall assembly bar uses the MEDIAN over M*K cells for both
       heads; NaN medians fail closed; predictions without v2 wall heads
       (v1/v1.1) pass vacuously.
    7. ContractPredictorAdapter is unchanged: the WD-01 conformance kit is
       v1/v1.1-frozen, so v2 conformance is exercised through the mock's v2
       mode; the adapter keeps refusing v2 until the kit validates it.
    8. PL-07 (owner decision 2026-09-27): window_relative_nonincreasing
       True (the default) re-arms the 30 s window difference ALONE as an
       ADDITIVE v2 veto on top of the summed window+future rule ('window-
       only relative'); the window verdicts are the SafetyVerdict.relative
       entries and window_relative_veto carries the veto. False, and every
       v1/v1.1 prediction, leave the relative behaviour unchanged.
    """


# --------------------------------------------------------------- fail closed
class TestV2FailClosed:
    def test_missing_future_cell_valid_fails_closed(self):
        world = _v2_world()
        prediction = _v2_predict(world)
        prediction = dataclasses.replace(prediction, future_cell_valid=None)
        assert bool(fail_closed_mask(prediction).all())
        assert not safety_check(prediction, PlannerConfig()).ok

    def test_false_future_cell_valid_vetoes_cells(self):
        world = _v2_world()
        prediction = _v2_predict(world, script={"future_cell_valid": [[True, False]]})
        mask = fail_closed_mask(prediction)
        assert not bool(mask[0, 0])
        assert bool(mask[0, 1])
        verdict = safety_check(prediction, PlannerConfig())
        assert not verdict.ok

    def test_false_native_future_cell_valid_vetoes(self):
        world = _v2_world()
        prediction = _v2_predict(world, native={"native_future_cell_valid": [[False, True]]})
        assert bool(fail_closed_mask(prediction)[0, 0])
        assert not safety_check(prediction, PlannerConfig()).ok

    def test_missing_future_has_closing_pair_fails_closed(self):
        world = _v2_world()
        prediction = _v2_predict(world)
        prediction = dataclasses.replace(prediction, future_has_closing_pair=None)
        assert bool(fail_closed_mask(prediction).all())

    def test_nan_q05_with_closing_pair_fails_closed(self):
        world = _v2_world()
        prediction = _v2_predict(world, script={"future_min_ttc_q05_s": [[np.nan, np.nan]]})
        # pair True by default: a NaN q05 in a valid future cell with a
        # closing pair is an invalid signal, validated or not
        assert bool(fail_closed_mask(prediction).all())
        assert not safety_check(prediction, PlannerConfig()).ok

    def test_nan_q05_without_pair_is_valid(self):
        world = _v2_world()
        prediction = _v2_predict(
            world,
            script={
                "future_min_ttc_q05_s": np.nan,
                "future_has_closing_pair": False,
            },
        )
        assert not bool(fail_closed_mask(prediction).any())
        assert safety_check(prediction, PlannerConfig()).ok


# --------------------------------------------------------- p_future_collision
class TestV2FutureCollision:
    def test_absolute_gate_0099_passes_0101_vetoes(self):
        world = _v2_world()
        # native p equals the plan's so the PAIRED rule is quiet and the
        # ABSOLUTE gate alone decides
        ok = _v2_predict(
            world,
            script={"p_future_collision": 0.0099},
            native={"native_p_future_collision": 0.0099},
        )
        bad = _v2_predict(
            world,
            script={"p_future_collision": 0.0101},
            native={"native_p_future_collision": 0.0099},
        )
        assert not future_collision_veto(ok, PlannerConfig())
        assert future_collision_veto(bad, PlannerConfig())
        assert safety_check(ok, PlannerConfig()).ok
        vetoed = safety_check(bad, PlannerConfig())
        assert not vetoed.ok and vetoed.p_future_collision_veto

    def test_paired_relative_veto_at_zero_tolerance(self):
        world = _v2_world()
        prediction = _v2_predict(
            world,
            script={
                "p_future_collision": [[0.005, 0.005]],
            },
            native={"native_p_future_collision": 0.0},
        )
        # 0.005 < abs max 0.01, but the plan exceeds its native pair (0.0)
        # by more than p_future_collision_rel_tol = 0 -> veto
        assert future_collision_veto(prediction, PlannerConfig())
        verdict = safety_check(prediction, PlannerConfig())
        assert not verdict.ok and verdict.p_future_collision_veto

    def test_paired_within_tolerance_passes(self):
        world = _v2_world()
        prediction = _v2_predict(
            world,
            script={
                "p_future_collision": [[0.008, 0.008]],
            },
            native={"native_p_future_collision": 0.008},
        )
        assert not future_collision_veto(prediction, PlannerConfig())
        assert safety_check(prediction, PlannerConfig()).ok


# --------------------------------------------------- relative window + future
class TestV2RelativeWindowPlusFuture:
    @staticmethod
    def _cfg3(**over):
        """These tests exercise the FROZEN 3 s exposure fields' rule; T-A
        moved the config default to the 1.5 s primary metric, so they opt
        back onto the 3 s selection explicitly."""
        return PlannerConfig(relative_ttc_threshold_s=3.0, **over)

    def test_window_plus_future_veto(self):
        world = _v2_world()
        # window exposure equal, future +1.0 veh*s: the combined mean
        # increase is +1.0 > rel_safety_tol_s = 0 -> veto
        prediction = _v2_predict(
            world,
            script={
                "ttc_lt3_exposure_s": 5.0,
                "future_ttc_lt3_exposure_s": 6.0,
            },
            native={"ttc_lt3_exposure_s": 5.0, "native_future_ttc_lt3_exposure_s": 5.0},
        )
        verdicts = future_relative_verdicts(prediction, self._cfg3())
        exposure = {v.metric: v for v in verdicts}
        assert exposure["ttc_lt3_exposure_s+future"].mean_increase_s == pytest.approx(1.0)
        assert not exposure["ttc_lt3_exposure_s+future"].ok
        assert not safety_check(prediction, self._cfg3()).ok

    def test_window_rise_with_future_saving_true_vetoes_false_allows(self):
        """PL-07 (owner 2026-09-27): window exposure +1.0 veh*s and a future
        -1.0 make the summed delta 0 (passes the summed rule either way).
        window_relative_nonincreasing=True (the default) ALSO requires the
        30 s window difference alone to pass: the plan is vetoed and the
        verdict names 'window-only relative'. False restores the summed
        rule alone and allows the plan."""
        world = _v2_world()
        prediction = _v2_predict(
            world,
            script={
                "ttc_lt3_exposure_s": 6.0,
                "future_ttc_lt3_exposure_s": 4.0,
            },
            native={"ttc_lt3_exposure_s": 5.0, "native_future_ttc_lt3_exposure_s": 5.0},
        )
        # the summed window+future rule passes on both flag values
        assert all(v.ok for v in future_relative_verdicts(prediction, self._cfg3()))
        allowed = safety_check(
            prediction, self._cfg3(window_relative_nonincreasing=False)
        )
        assert allowed.ok and allowed.rel_ok and not allowed.window_relative_veto
        vetoed = safety_check(prediction, self._cfg3())  # default True
        assert not vetoed.ok
        assert vetoed.rel_ok and vetoed.future_ok  # only the window rule vetoes
        assert vetoed.window_relative_veto
        assert [v.metric for v in vetoed.relative if not v.ok] == ["ttc_lt3_exposure_s"]
        assert window_relative_note(vetoed) == "window-only relative: ttc_lt3_exposure_s"
        assert window_relative_note(allowed) == ""

    def test_window_hard_brake_rise_vetoes_by_default(self):
        """PL-07 covers hard_brake_s too: window +1.0, future -1.0, summed
        delta 0 -> the default (True) vetoes window-only; False allows."""
        world = _v2_world()
        prediction = _v2_predict(
            world,
            script={
                "hard_brake_s": 2.0,
                "future_hard_brake_s": 0.0,
            },
            native={"hard_brake_s": 1.0, "native_future_hard_brake_s": 1.0},
        )
        assert all(v.ok for v in future_relative_verdicts(prediction, PlannerConfig()))
        assert not safety_check(prediction, PlannerConfig()).ok
        vetoed = safety_check(prediction, PlannerConfig())
        assert vetoed.window_relative_veto
        assert [v.metric for v in vetoed.relative if not v.ok] == ["hard_brake_s"]
        assert safety_check(
            prediction, PlannerConfig(window_relative_nonincreasing=False)
        ).ok

    def test_loop_veto_reason_names_window_only_relative(self):
        """PL-07: the group veto reason carries the 'window-only relative'
        note when the window rule (not the summed rule) vetoes."""
        world = _v2_world()
        start = PlanCandidate(per_cav=dict(_PLAN))
        # window +1.0 vs native, future -1.0: summed rule quiet, window
        # rule vetoes -> the keep-current candidate is unsafe with the
        # window-only why-vetoed note
        world.safety_script[plan_key(start)] = {
            "ttc_lt3_exposure_s": 1.0,
            "future_ttc_lt3_exposure_s": -1.0,
        }
        world.native_safety["ttc_lt3_exposure_s"] = 0.0
        # a measured interaction folds both CAVs into one group so the
        # keep-current candidate (the unsafe start plan) is evaluated
        world.traj_coupling_pos[("c0", "c1")] = {"damping": 3.0}
        planner = Planner(world, self._cfg3(), current_plan=start)
        decision = planner.decide(_samples(), now_s=0.0)
        vetoes = [
            r
            for r in decision.reasons
            if r.action == "veto" and "window-only relative" in r.detail
        ]
        assert vetoes and "ttc_lt3_exposure_s" in vetoes[0].detail


# ------------------------------------------------------------------- G-1
class TestV2G1FutureFloor:
    def test_floor_applies_only_when_validated(self):
        world = _v2_world()
        # q05 = 1.2 below the 1.5 floor, closing pair True
        unvalidated = _v2_predict(world, script={"future_min_ttc_q05_s": 1.2})
        assert not g1_future_floor_veto(unvalidated, PlannerConfig())
        assert safety_check(unvalidated, PlannerConfig()).ok
        validated = _v2_predict(
            world,
            script={"future_min_ttc_q05_s": 1.2},
            meta={"future_min_ttc_validated": True, "future_min_ttc_validation_id": "val-1"},
        )
        assert g1_future_floor_veto(validated, PlannerConfig())
        verdict = safety_check(validated, PlannerConfig())
        assert not verdict.ok and verdict.g1_future_floor_veto

    def test_validated_q05_at_floor_passes_and_nan_without_pair_passes(self):
        world = _v2_world()
        at_floor = _v2_predict(
            world,
            script={"future_min_ttc_q05_s": 1.5},
            meta={"future_min_ttc_validated": True, "future_min_ttc_validation_id": "val-1"},
        )
        assert not g1_future_floor_veto(at_floor, PlannerConfig())
        nan_no_pair = _v2_predict(
            world,
            script={"future_min_ttc_q05_s": np.nan, "future_has_closing_pair": False},
            meta={"future_min_ttc_validated": True, "future_min_ttc_validation_id": "val-1"},
        )
        assert not g1_future_floor_veto(nan_no_pair, PlannerConfig())
        assert safety_check(nan_no_pair, PlannerConfig()).ok


# --------------------------------------------------------- G-1 mode gate
class TestV2ModeGate:
    def test_live_refuses_v1(self):
        world = MockWorldModel(("c0", "c1"), part_a_horizon_s=60.0)
        planner = Planner(
            world, PlannerConfig(mode="live"), current_plan=PlanCandidate(per_cav=dict(_PLAN))
        )
        with pytest.raises(FutureSafetyRefusal, match="live"):
            planner.decide(_samples(), now_s=0.0)

    def test_closed_loop_refuses_v1_1(self):
        world = MockWorldModel(
            ("c0", "c1"), part_a_horizon_s=60.0, contract_version="wd01_predict/1.1"
        )
        planner = Planner(
            world,
            PlannerConfig(mode="closed_loop"),
            current_plan=PlanCandidate(per_cav=dict(_PLAN)),
        )
        with pytest.raises(FutureSafetyRefusal, match="closed_loop"):
            planner.decide(_samples(), now_s=0.0)

    def test_live_refuses_unvalidated_v2(self):
        planner = Planner(
            _v2_world(), PlannerConfig(mode="live"), current_plan=PlanCandidate(per_cav=dict(_PLAN))
        )
        with pytest.raises(FutureSafetyRefusal, match="not validated"):
            planner.decide(_samples(), now_s=0.0)

    def test_live_allows_validated_v2(self):
        world = _v2_world()
        world.meta_script.update(
            {
                "future_min_ttc_validated": True,
                "future_min_ttc_validation_id": "val-42",
                # G-2 (2026-09-27): live requires BOTH gates validated
                "relative_safety_validated": True,
                "relative_safety_validation_id": "rel-42",
            }
        )
        planner = Planner(
            world, PlannerConfig(mode="live"), current_plan=PlanCandidate(per_cav=dict(_PLAN))
        )
        decision = planner.decide(_samples(), now_s=0.0)
        assert decision.plan is not None
        assert not [r for r in decision.reasons if r.step == "gate"]

    def test_validated_requires_validation_id(self):
        world = _v2_world()
        world.meta_script.update({"future_min_ttc_validated": True})
        planner = Planner(
            world, PlannerConfig(mode="live"), current_plan=PlanCandidate(per_cav=dict(_PLAN))
        )
        with pytest.raises(FutureSafetyRefusal, match="validation_id"):
            planner.decide(_samples(), now_s=0.0)

    def test_controlled_experiment_override_allowed_and_logged(self):
        world = _v2_world()
        # G-2 quiet (relative validated) so this test isolates the G-1
        # override; the G-2-only behaviour lives in test_contract_v2_g2.py
        world.meta_script.update(
            {
                "relative_safety_validated": True,
                "relative_safety_validation_id": "rel-1",
            }
        )
        planner = Planner(
            world,
            PlannerConfig(mode="controlled_experiment", allow_unvalidated_future_ttc=True),
            current_plan=PlanCandidate(per_cav=dict(_PLAN)),
        )
        decision = planner.decide(_samples(), now_s=0.0)
        overrides = [r for r in decision.reasons if r.action == "override"]
        assert overrides and "unvalidated" in overrides[0].detail
        # without the flag the same experiment refuses
        strict = Planner(
            world,
            PlannerConfig(mode="controlled_experiment"),
            current_plan=PlanCandidate(per_cav=dict(_PLAN)),
        )
        with pytest.raises(FutureSafetyRefusal, match="allow_unvalidated_future_ttc"):
            strict.decide(_samples(), now_s=0.0)

    def test_offline_never_refuses(self):
        planner = Planner(
            _v2_world(),
            PlannerConfig(mode="offline"),
            current_plan=PlanCandidate(per_cav=dict(_PLAN)),
        )
        assert planner.decide(_samples(), now_s=0.0).plan is not None
        legacy = Planner(
            MockWorldModel(("c0", "c1")),
            PlannerConfig(mode="offline"),
            current_plan=PlanCandidate(per_cav=dict(_PLAN)),
        )
        assert legacy.decide(_samples(), now_s=0.0).plan is not None


# ------------------------------------------ PL-09 gate hardening (mode + G-1)
class TestPL09GateHardening:
    """PL-09 (owner 2026-09-27): the G-1/G-2 mode gate fails closed on an
    unknown/typo mode (only 'offline' passes OPEN) and on NON-BOOL
    validation flags (only the literal True counts as validated)."""

    @staticmethod
    def _config_with_doctored_mode(bad_mode):
        """A frozen PlannerConfig carrying an invalid mode, injected past
        the __post_init__ validation via object.__setattr__."""
        config = PlannerConfig(mode="live")
        object.__setattr__(config, "mode", bad_mode)
        assert config.mode == bad_mode
        return config

    @pytest.mark.parametrize("bad_mode", ["Live", "closedloop", "simulation", ""])
    def test_unknown_mode_refuses_not_passes_open(self, bad_mode):
        """The gate returns None ONLY for 'offline'; any other non-gated
        mode -- a typo like 'Live'/'closedloop' injected past the frozen
        dataclass validation -- is a FutureSafetyRefusal, never an
        implicit pass."""
        prediction = _v2_predict(_v2_world())
        with pytest.raises(FutureSafetyRefusal, match="unknown mode"):
            enforce_future_gate(prediction, self._config_with_doctored_mode(bad_mode))

    def test_offline_remains_the_only_nongated_mode(self):
        prediction = _v2_predict(_v2_world())
        assert enforce_future_gate(prediction, PlannerConfig(mode="offline")) is None
        for mode in ("live", "closed_loop", "controlled_experiment"):
            with pytest.raises(FutureSafetyRefusal):
                enforce_future_gate(prediction, PlannerConfig(mode=mode))

    @pytest.mark.parametrize("bad", ["False", "false", "True", 1, 0, None])
    def test_live_g1_non_bool_validated_is_not_validated(self, bad):
        """G-1 (PL-09): the mock writes meta_script values verbatim
        (meta.update after its own bool() defaults), so a producer emitting
        'False'/'True'/1/None instead of a bool is NOT validated -- live
        refuses instead of consuming the head (pre-PL-09 the truthy strings
        and 1 PASSED the gate)."""
        prediction = _v2_predict(
            _v2_world(),
            meta={
                "future_min_ttc_validated": bad,
                "future_min_ttc_validation_id": "val-1",
                "relative_safety_validated": True,  # G-2 quiet: isolate G-1
                "relative_safety_validation_id": "rel-1",
            },
        )
        assert prediction.meta["future_min_ttc_validated"] == bad
        with pytest.raises(
            FutureSafetyRefusal, match="future_min_ttc_q05_s head is not validated"
        ):
            enforce_future_gate(prediction, PlannerConfig(mode="live"))

    def test_controlled_experiment_g1_non_bool_validated_requires_override(self):
        """The string 'True' is UNVALIDATED for the gate: without the
        override flag controlled_experiment refuses; with it the head is
        consumed and the override note is returned for the decision
        record."""
        prediction = _v2_predict(
            _v2_world(),
            meta={
                "future_min_ttc_validated": "True",
                "future_min_ttc_validation_id": "val-1",
                "relative_safety_validated": True,
                "relative_safety_validation_id": "rel-1",
            },
        )
        with pytest.raises(FutureSafetyRefusal, match="allow_unvalidated_future_ttc"):
            enforce_future_gate(prediction, PlannerConfig(mode="controlled_experiment"))
        note = enforce_future_gate(
            prediction,
            PlannerConfig(mode="controlled_experiment", allow_unvalidated_future_ttc=True),
        )
        assert note and "future_min_ttc_q05" in note


# ------------------------------------------------------- commit / continuation
class TestV2CommitDispatch:
    def test_loop_passes_commit_and_continuation_from_config(self):
        world = _v2_world()
        config = PlannerConfig(commit_s=45.0, continuation="native")
        planner = Planner(world, config, current_plan=PlanCandidate(per_cav=dict(_PLAN)))
        planner.decide(_samples(), now_s=0.0)
        assert world.calls and world.calls[0]["commit_s"] == 45.0
        assert world.calls[0]["continuation"] == "native"

    def test_v1_model_is_called_without_v2_kwargs(self):
        world = MockWorldModel(("c0", "c1"), part_a_horizon_s=60.0)
        planner = Planner(
            world, PlannerConfig(commit_s=45.0), current_plan=PlanCandidate(per_cav=dict(_PLAN))
        )
        planner.decide(_samples(), now_s=0.0)
        assert "commit_s" not in world.calls[0]

    def test_v2_meta_fields(self):
        world = _v2_world()
        world.meta_script.update(
            {
                "future_min_ttc_validated": True,
                "future_min_ttc_validation_id": "val-7",
            }
        )
        prediction = _v2_predict(world)
        assert prediction.meta["contract_version"] == V2
        assert prediction.meta["commit_s"] == 30.0
        assert prediction.meta["continuation"] == "planner"
        assert prediction.meta["continuation_policy_id"]
        assert prediction.meta["value_model_id"]
        assert prediction.meta["value_model_sha256"]
        assert prediction.meta["future_min_ttc_validated"] is True
        assert prediction.meta["future_min_ttc_validation_id"] == "val-7"
        assert prediction.meta["safety_window_s"] == 30.0

    def test_native_continuation_has_null_policy_id(self):
        world = _v2_world()
        prediction = _v2_predict(world, continuation="native")
        assert prediction.meta["continuation"] == "native"
        assert prediction.meta["continuation_policy_id"] is None

    def test_v1_prediction_has_no_future_fields(self):
        world = MockWorldModel(("c0", "c1"), part_a_horizon_s=60.0)
        prediction = world.predict_many(
            _samples(), [PlanCandidate(per_cav={"c0": "damping", "c1": "native"})]
        )[0]
        assert prediction.future_cell_valid is None
        assert prediction.p_future_collision is None
        assert prediction.meta["contract_version"] == "wd01_predict/1"


# ------------------------------------------------------------ v2 wall heads
class TestV2WallAssemblyBar:
    def _prediction_with_wall(self, completes, abort):
        from traffic_shepherd.planner import PlanCandidate, WallSpec

        world = _v2_world()
        wall = WallSpec(
            wall_id="w0", members=("c0", "c1"), speed_mps=16.0, phase="assemble", hold_s=60.0
        )
        plan = PlanCandidate(per_cav={}, walls=(wall,))
        world.wall_script["w0"] = {
            "p_assembly_completes": completes,
            "p_assembly_abort": abort,
        }
        return world.predict_many(_samples(), [plan], horizon_s=600.0, commit_s=30.0)[0]

    def test_wall_heads_reported_and_bar_passes(self):
        from traffic_shepherd.planner.walls import assembly_bar_ok

        prediction = self._prediction_with_wall(0.95, 0.05)
        block = prediction.walls["w0"]
        assert float(block.p_assembly_completes[0, 0]) == pytest.approx(0.95)
        assert float(block.p_assembly_abort[0, 0]) == pytest.approx(0.05)
        ok, detail = assembly_bar_ok(prediction, "w0", PlannerConfig())
        assert ok and "passed" in detail

    def test_bar_vetoes_low_completion_and_high_abort(self):
        from traffic_shepherd.planner.walls import assembly_bar_ok

        low = self._prediction_with_wall(0.85, 0.05)
        ok, detail = assembly_bar_ok(low, "w0", PlannerConfig())
        assert not ok and "completes" in detail
        aborting = self._prediction_with_wall(0.95, 0.20)
        ok, detail = assembly_bar_ok(aborting, "w0", PlannerConfig())
        assert not ok and "abort" in detail

    def test_v1_wall_block_has_no_v2_heads(self):
        from traffic_shepherd.planner.walls import assembly_bar_ok

        world = MockWorldModel(("c0", "c1"), part_a_horizon_s=60.0)
        from traffic_shepherd.planner import PlanCandidate, WallSpec

        plan = PlanCandidate(
            per_cav={},
            walls=(
                WallSpec(
                    wall_id="w0",
                    members=("c0", "c1"),
                    speed_mps=16.0,
                    phase="assemble",
                    hold_s=60.0,
                ),
            ),
        )
        prediction = world.predict_many(_samples(), [plan])[0]
        assert prediction.walls["w0"].p_assembly_completes is None
        ok, detail = assembly_bar_ok(prediction, "w0", PlannerConfig())
        assert ok and "not applicable" in detail


# --------------------------------------------------------- v1/v1.1 unchanged
class TestPreV2Unchanged:
    def test_v1_1_safety_verdict_identical_fields(self):
        world = MockWorldModel(
            ("c0", "c1"), part_a_horizon_s=60.0, contract_version="wd01_predict/1.1"
        )
        prediction = world.predict_many(
            _samples(), [PlanCandidate(per_cav={"c0": "damping", "c1": "native"})]
        )[0]
        verdict = safety_check(prediction, PlannerConfig())
        assert verdict.ok
        assert verdict.future_ok is True
        assert verdict.future_relative == ()
        assert verdict.p_future_collision_veto is False
        assert verdict.g1_future_floor_veto is False

    def test_v1_prediction_without_diagnostics_keeps_v1_behaviour(self):
        world = MockWorldModel(("c0", "c1"), part_a_horizon_s=60.0)
        prediction = world.predict_many(
            _samples(), [PlanCandidate(per_cav={"c0": "damping", "c1": "native"})]
        )[0]
        assert not bool(fail_closed_mask(prediction).any())
        assert safety_check(prediction, PlannerConfig()).ok

    def test_v1_1_window_flag_unaffected(self):
        """PL-07: on v1/v1.1 (no future terms) the window rule is the ONLY
        relative test either way -- window_relative_nonincreasing must not
        change the verdict (it adds the additive v2 veto nowhere)."""
        rising_world = MockWorldModel(
            ("c0", "c1"), part_a_horizon_s=60.0, contract_version="wd01_predict/1.1"
        )
        plan = PlanCandidate(per_cav=dict(_PLAN))
        rising_world.safety_script[plan_key(plan)] = {"ttc_lt1p5_exposure_s": 1.0}
        rising_world.native_safety["ttc_lt1p5_exposure_s"] = 0.0
        rising = rising_world.predict_many(_samples(), [plan])[0]
        clean_world = MockWorldModel(
            ("c0", "c1"), part_a_horizon_s=60.0, contract_version="wd01_predict/1.1"
        )
        clean = clean_world.predict_many(
            _samples(), [PlanCandidate(per_cav=dict(_PLAN))]
        )[0]
        for cfg in (
            PlannerConfig(window_relative_nonincreasing=True),
            PlannerConfig(window_relative_nonincreasing=False),
        ):
            verdict = safety_check(rising, cfg)
            assert not verdict.ok and not verdict.rel_ok  # the window rule vetoes
            assert verdict.window_relative_veto is False  # no v2 additive flag
            assert safety_check(clean, cfg).ok


# ------------------------------------------- PL-06 native-collision flag
class TestNativeCollisionFlag:
    """PL-06 (owner decision 2026-09-27): native_collision_vetoes_plan
    defaults to False (option A) -- a native collision does not veto a plan
    whose own gates pass; the collision is a logged 'native_collision'
    diagnostic (SafetyVerdict.native_collision_cells + a decision reason).
    Explicit True restores option B (the pre-2026-09-27 veto).
    Exercised on v1.1 AND v2 predictions."""

    @staticmethod
    def _world_with_native_collision(contract_version):
        world = MockWorldModel(
            ("c0", "c1"), part_a_horizon_s=60.0, contract_version=contract_version
        )
        # shape [M=1, K=2]: the native collides in sample cell 1 only
        world.native_safety["collision"] = [[False, True]]
        return world

    def test_v1_1_default_false_plan_passes_and_collision_logged(self):
        world = self._world_with_native_collision("wd01_predict/1.1")
        prediction = world.predict_many(
            _samples(), [PlanCandidate(per_cav=dict(_PLAN))]
        )[0]
        cfg = PlannerConfig()  # default False since 2026-09-27
        assert cfg.native_collision_vetoes_plan is False
        assert not bool(fail_closed_mask(prediction, cfg).any())
        verdict = safety_check(prediction, cfg)
        assert verdict.ok and verdict.abs_ok and verdict.rel_ok
        assert verdict.vetoed_cells == 0
        assert verdict.native_collision_cells == 1

    def test_v1_1_explicit_true_restores_option_b_veto(self):
        world = self._world_with_native_collision("wd01_predict/1.1")
        prediction = world.predict_many(
            _samples(), [PlanCandidate(per_cav=dict(_PLAN))]
        )[0]
        cfg = PlannerConfig(native_collision_vetoes_plan=True)
        verdict = safety_check(prediction, cfg)
        assert not verdict.ok and not verdict.abs_ok
        assert verdict.vetoed_cells == 1
        assert verdict.native_collision_cells == 1

    def test_v2_default_false_plan_passes_and_collision_logged(self):
        world = self._world_with_native_collision(V2)
        prediction = _v2_predict(world)
        cfg = PlannerConfig()  # default False since 2026-09-27
        assert not bool(fail_closed_mask(prediction, cfg).any())
        verdict = safety_check(prediction, cfg)
        # the plan's own v2 gates decide: p_future 0.0 <= abs max, no
        # relative increase, q05 floor quiet (unvalidated)
        assert verdict.ok and verdict.abs_ok and verdict.rel_ok and verdict.future_ok
        assert verdict.vetoed_cells == 0
        assert verdict.native_collision_cells == 1

    def test_v2_explicit_true_restores_option_b_veto(self):
        world = self._world_with_native_collision(V2)
        prediction = _v2_predict(world)
        cfg = PlannerConfig(native_collision_vetoes_plan=True)
        assert not safety_check(prediction, cfg).ok
        assert bool(fail_closed_mask(prediction, cfg)[0, 1])

    def test_v1_1_loop_default_false_logs_native_collision_reason(self):
        # owner decision 2026-09-27 (option A): the DEFAULT logs the
        # 'native_collision' diagnostic into the decision reasons
        world = self._world_with_native_collision("wd01_predict/1.1")
        planner = Planner(
            world, PlannerConfig(), current_plan=PlanCandidate(per_cav=dict(_PLAN))
        )
        decision = planner.decide(_samples(), now_s=0.0)
        assert decision.plan is not None
        hits = [r for r in decision.reasons if r.action == "native_collision"]
        assert len(hits) == 1
        assert "native_collision cells=1" in hits[0].detail
        assert "not a veto" in hits[0].detail

    def test_v1_1_loop_explicit_true_no_diagnostic(self):
        world = self._world_with_native_collision("wd01_predict/1.1")
        planner = Planner(
            world,
            PlannerConfig(native_collision_vetoes_plan=True),
            current_plan=PlanCandidate(per_cav=dict(_PLAN)),
        )
        decision = planner.decide(_samples(), now_s=0.0)
        assert not [r for r in decision.reasons if r.action == "native_collision"]

    def test_v1_1_loop_false_flag_logs_native_collision_reason(self):
        world = self._world_with_native_collision("wd01_predict/1.1")
        planner = Planner(
            world,
            PlannerConfig(native_collision_vetoes_plan=False),
            current_plan=PlanCandidate(per_cav=dict(_PLAN)),
        )
        decision = planner.decide(_samples(), now_s=0.0)
        assert decision.plan is not None
        hits = [r for r in decision.reasons if r.action == "native_collision"]
        assert len(hits) == 1
        assert "native_collision cells=1" in hits[0].detail
        assert "not a veto" in hits[0].detail

    def test_v2_loop_false_flag_logs_native_collision_reason(self):
        world = self._world_with_native_collision(V2)
        planner = Planner(
            world,
            PlannerConfig(native_collision_vetoes_plan=False),
            current_plan=PlanCandidate(per_cav=dict(_PLAN)),
        )
        decision = planner.decide(_samples(), now_s=0.0)
        assert decision.plan is not None
        hits = [r for r in decision.reasons if r.action == "native_collision"]
        assert len(hits) == 1
        assert "native_collision cells=1" in hits[0].detail

    def test_v2_loop_default_false_logs_native_collision_reason(self):
        world = self._world_with_native_collision(V2)
        planner = Planner(
            world, PlannerConfig(), current_plan=PlanCandidate(per_cav=dict(_PLAN))
        )
        decision = planner.decide(_samples(), now_s=0.0)
        assert decision.plan is not None
        hits = [r for r in decision.reasons if r.action == "native_collision"]
        assert len(hits) == 1
        assert "native_collision cells=1" in hits[0].detail

    def test_v2_loop_explicit_true_no_diagnostic(self):
        world = self._world_with_native_collision(V2)
        planner = Planner(
            world,
            PlannerConfig(native_collision_vetoes_plan=True),
            current_plan=PlanCandidate(per_cav=dict(_PLAN)),
        )
        decision = planner.decide(_samples(), now_s=0.0)
        assert not [r for r in decision.reasons if r.action == "native_collision"]
