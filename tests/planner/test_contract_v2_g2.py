"""G-2 tests (contract v2 §8, owner decision 2026-09-27): the planner-side
relative-safety validation gate, mirrored next to G-1.

Contract §8 (read-only text: the WD-01 worktree's PREDICT_CONTRACT_v2.md,
sha256
3a0a5f20...ac37): meta ``relative_safety_validated: bool`` and
``relative_safety_validation_id`` (non-None iff True; a MISSING field on a
v2 prediction is treated as False -- fail closed). While False, the
relative-safety verdicts (TTC<3 exposure, hard braking) of that prediction
are UNVALIDATED: live/closed_loop REFUSE (like G-1; the two gates combine,
BOTH must pass), controlled_experiment proceeds only with
``PlannerConfig.allow_unvalidated_relative_safety`` set (the override is
logged in the decision record), and offline never refuses but logs the
``relative_safety_validated=False`` diagnostic into the decision record.
A validated producer -- the perfect simulator -- sets True with its own id
(exercised here through the mock's scriptable v2 meta).
"""

import dataclasses
from types import SimpleNamespace

import pytest

from traffic_shepherd.planner.config import PlannerConfig
from traffic_shepherd.planner.interface import PlanCandidate
from traffic_shepherd.planner.loop import Planner
from traffic_shepherd.planner.mock import MockWorldModel
from traffic_shepherd.planner.safety import (
    FutureSafetyRefusal,
    enforce_future_gate,
    relative_safety_diagnostic,
)

V2 = "wd01_predict/2"

_PLAN = {"c0": "damping", "c1": "native"}

_G1_OK = {
    "future_min_ttc_validated": True,
    "future_min_ttc_validation_id": "val-1",
}

_G2_OK = {
    "relative_safety_validated": True,
    "relative_safety_validation_id": "rel-1",
}


def _samples(k=2, time_s=0.0):
    return [
        SimpleNamespace(
            hybrid=None,
            arrivals={"seed": i},
            meta={"sample_id": f"s{i}", "time_s": time_s, "merge_observed": True, "coverage": 1.0},
        )
        for i in range(k)
    ]


def _v2_world():
    return MockWorldModel(("c0", "c1"), part_a_horizon_s=60.0, contract_version=V2)


def _planner(world, config):
    return Planner(world, config, current_plan=PlanCandidate(per_cav=dict(_PLAN)))


# ------------------------------------------------------------ live/closed_loop
class TestG2LiveRefuses:
    def test_live_refuses_unvalidated_relative_safety(self):
        """G-1 satisfied, G-2 default False -> live refuses (missing field
        on the mock's meta is emitted as False; the stripped-field case is
        test_missing_meta_field_treated_as_false below)."""
        world = _v2_world()
        world.meta_script.update(dict(_G1_OK))
        with pytest.raises(FutureSafetyRefusal, match=r"relative-safety.*G-2"):
            _planner(world, PlannerConfig(mode="live")).decide(_samples(), now_s=0.0)

    def test_closed_loop_refuses_unvalidated_relative_safety(self):
        world = _v2_world()
        world.meta_script.update(dict(_G1_OK))
        with pytest.raises(FutureSafetyRefusal, match="closed_loop"):
            _planner(world, PlannerConfig(mode="closed_loop")).decide(_samples(), now_s=0.0)

    def test_live_allows_validated_relative_safety(self):
        world = _v2_world()
        world.meta_script.update({**_G1_OK, **_G2_OK})
        decision = _planner(world, PlannerConfig(mode="live")).decide(_samples(), now_s=0.0)
        assert decision.plan is not None
        assert not [r for r in decision.reasons if r.step == "gate"]

    def test_closed_loop_allows_validated_relative_safety(self):
        world = _v2_world()
        world.meta_script.update({**_G1_OK, **_G2_OK})
        decision = _planner(
            world, PlannerConfig(mode="closed_loop")
        ).decide(_samples(), now_s=0.0)
        assert decision.plan is not None

    def test_live_refuses_validated_true_without_id(self):
        world = _v2_world()
        world.meta_script.update({**_G1_OK, "relative_safety_validated": True})
        with pytest.raises(FutureSafetyRefusal, match="relative_safety_validation_id"):
            _planner(world, PlannerConfig(mode="live")).decide(_samples(), now_s=0.0)

    def test_closed_loop_refuses_validated_true_without_id(self):
        world = _v2_world()
        world.meta_script.update({**_G1_OK, "relative_safety_validated": True})
        with pytest.raises(FutureSafetyRefusal, match="relative_safety_validation_id"):
            _planner(world, PlannerConfig(mode="closed_loop")).decide(_samples(), now_s=0.0)

    def test_missing_meta_field_treated_as_false(self):
        """A v2 prediction whose meta OMITS the G-2 fields fails closed:
        the mock emits them by default, so strip them from the prediction."""
        world = _v2_world()
        world.meta_script.update(dict(_G1_OK))
        plan = PlanCandidate(per_cav=dict(_PLAN))
        prediction = world.predict_many(_samples(), [plan], commit_s=30.0)[0]
        stripped = dataclasses.replace(
            prediction,
            meta={
                k: v
                for k, v in prediction.meta.items()
                if not k.startswith("relative_safety")
            },
        )
        assert not bool(stripped.meta.get("relative_safety_validated", False))
        with pytest.raises(FutureSafetyRefusal, match="G-2"):
            enforce_future_gate(stripped, PlannerConfig(mode="live"))
        assert relative_safety_diagnostic(stripped)  # offline notes it too


# ---------------------------------------------- PL-09 non-bool G-2 meta
class TestG2NonBoolMetaFailClosed:
    """PL-09 (owner 2026-09-27): only the literal True counts as
    validated. The mock writes meta_script values verbatim (meta.update
    after its own bool() defaults), so a producer emitting 'False',
    'True', 1 or None instead of a bool is NOT validated: live refuses
    (pre-PL-09 the truthy strings and 1 PASSED the gate)."""

    @pytest.mark.parametrize("bad", ["False", "false", "True", 1, 0, None])
    def test_live_g2_non_bool_validated_refuses(self, bad):
        world = _v2_world()
        world.meta_script.update(
            {
                **_G1_OK,  # G-1 quiet: isolate the G-2 read
                "relative_safety_validated": bad,
                "relative_safety_validation_id": "rel-1",
            }
        )
        prediction = world.predict_many(
            _samples(), [PlanCandidate(per_cav=dict(_PLAN))], commit_s=30.0
        )[0]
        assert prediction.meta["relative_safety_validated"] == bad
        with pytest.raises(FutureSafetyRefusal, match=r"relative-safety.*G-2"):
            enforce_future_gate(prediction, PlannerConfig(mode="live"))

    def test_controlled_experiment_g2_non_bool_validated_requires_override(self):
        world = _v2_world()
        world.meta_script.update({**_G1_OK, "relative_safety_validated": "True"})
        prediction = world.predict_many(
            _samples(), [PlanCandidate(per_cav=dict(_PLAN))], commit_s=30.0
        )[0]
        with pytest.raises(FutureSafetyRefusal, match="allow_unvalidated_relative_safety"):
            enforce_future_gate(prediction, PlannerConfig(mode="controlled_experiment"))
        note = enforce_future_gate(
            prediction,
            PlannerConfig(
                mode="controlled_experiment", allow_unvalidated_relative_safety=True
            ),
        )
        assert note and "relative-safety" in note


# ------------------------------------------------------- controlled_experiment
class TestG2ControlledExperiment:
    def test_refuses_without_override(self):
        world = _v2_world()
        world.meta_script.update(dict(_G1_OK))
        with pytest.raises(FutureSafetyRefusal, match="allow_unvalidated_relative_safety"):
            _planner(
                world, PlannerConfig(mode="controlled_experiment")
            ).decide(_samples(), now_s=0.0)

    def test_override_allows_and_logs(self):
        world = _v2_world()
        world.meta_script.update(dict(_G1_OK))
        decision = _planner(
            world,
            PlannerConfig(mode="controlled_experiment", allow_unvalidated_relative_safety=True),
        ).decide(_samples(), now_s=0.0)
        assert decision.plan is not None
        overrides = [r for r in decision.reasons if r.action == "override"]
        assert len(overrides) == 1
        assert "relative-safety" in overrides[0].detail
        assert "allow_unvalidated_relative_safety=True" in overrides[0].detail

    def test_both_unvalidated_one_flag_still_refuses(self):
        world = _v2_world()  # G-1 and G-2 both default False
        with pytest.raises(FutureSafetyRefusal, match="allow_unvalidated_relative_safety"):
            _planner(
                world,
                PlannerConfig(mode="controlled_experiment", allow_unvalidated_future_ttc=True),
            ).decide(_samples(), now_s=0.0)

    def test_both_unvalidated_both_flags_log_combined_override(self):
        world = _v2_world()
        decision = _planner(
            world,
            PlannerConfig(
                mode="controlled_experiment",
                allow_unvalidated_future_ttc=True,
                allow_unvalidated_relative_safety=True,
            ),
        ).decide(_samples(), now_s=0.0)
        assert decision.plan is not None
        overrides = [r for r in decision.reasons if r.action == "override"]
        assert len(overrides) == 1
        assert "future_min_ttc_q05" in overrides[0].detail
        assert "relative-safety" in overrides[0].detail


# --------------------------------------------------------------------- offline
class TestG2Offline:
    def test_offline_never_refuses_and_logs_diagnostic(self):
        world = _v2_world()  # relative_safety_validated defaults False
        decision = _planner(world, PlannerConfig()).decide(_samples(), now_s=0.0)
        assert decision.plan is not None
        diagnostics = [r for r in decision.reasons if r.action == "relative_safety_unvalidated"]
        assert len(diagnostics) == 1
        assert diagnostics[0].step == "gate"
        assert "relative_safety_validated=False" in diagnostics[0].detail
        assert "G-2" in diagnostics[0].detail

    def test_offline_validated_carries_no_diagnostic(self):
        world = _v2_world()
        world.meta_script.update({**_G1_OK, **_G2_OK})  # e.g. perfect-sim truth
        decision = _planner(world, PlannerConfig()).decide(_samples(), now_s=0.0)
        assert decision.plan is not None
        assert not [r for r in decision.reasons if r.action == "relative_safety_unvalidated"]

    def test_v1_prediction_carries_no_g2_diagnostic(self):
        """Additive-only: the G-2 diagnostic never fires on v1/v1.1."""
        world = MockWorldModel(("c0", "c1"), part_a_horizon_s=60.0)
        decision = _planner(world, PlannerConfig()).decide(_samples(), now_s=0.0)
        assert decision.plan is not None
        assert not [r for r in decision.reasons if r.step == "gate"]

    def test_diagnostic_note_helper(self):
        assert relative_safety_diagnostic(
            _v2_world().predict_many(_samples(), [PlanCandidate(per_cav=dict(_PLAN))])[0]
        )
        assert not relative_safety_diagnostic(
            MockWorldModel(("c0", "c1")).predict_many(
                _samples(), [PlanCandidate(per_cav=dict(_PLAN))]
            )[0]
        )


# ------------------------------------------------------- G-1 x G-2 combination
class TestG2CombinesWithG1:
    def test_live_refusal_names_g1_when_only_future_unvalidated(self):
        world = _v2_world()
        world.meta_script.update(dict(_G2_OK))
        with pytest.raises(FutureSafetyRefusal, match="not validated"):
            _planner(world, PlannerConfig(mode="live")).decide(_samples(), now_s=0.0)

    def test_live_refusal_names_g2_when_only_relative_unvalidated(self):
        world = _v2_world()
        world.meta_script.update(dict(_G1_OK))
        with pytest.raises(FutureSafetyRefusal, match=r"relative-safety.*G-2"):
            _planner(world, PlannerConfig(mode="live")).decide(_samples(), now_s=0.0)

    def test_live_requires_both_gates(self):
        world = _v2_world()  # both unvalidated: G-1 raises first
        with pytest.raises(FutureSafetyRefusal, match="not validated"):
            _planner(world, PlannerConfig(mode="live")).decide(_samples(), now_s=0.0)

    def test_both_validated_passes_live(self):
        world = _v2_world()
        world.meta_script.update({**_G1_OK, **_G2_OK})
        assert (
            _planner(world, PlannerConfig(mode="live")).decide(_samples(), now_s=0.0).plan
            is not None
        )


# --------------------------------------------------------- meta + config plumbing
class TestG2MetaAndConfig:
    def test_mock_v2_emits_scriptable_g2_meta(self):
        world = _v2_world()
        prediction = world.predict_many(
            _samples(), [PlanCandidate(per_cav=dict(_PLAN))], commit_s=30.0
        )[0]
        assert prediction.meta["relative_safety_validated"] is False
        assert prediction.meta["relative_safety_validation_id"] is None
        world.meta_script.update({**_G2_OK, "relative_safety_validation_id": "sim-truth-7"})
        prediction = world.predict_many(
            _samples(), [PlanCandidate(per_cav=dict(_PLAN))], commit_s=30.0
        )[0]
        assert prediction.meta["relative_safety_validated"] is True
        assert prediction.meta["relative_safety_validation_id"] == "sim-truth-7"

    def test_v1_meta_has_no_g2_fields(self):
        prediction = MockWorldModel(("c0", "c1")).predict_many(
            _samples(), [PlanCandidate(per_cav=dict(_PLAN))]
        )[0]
        assert "relative_safety_validated" not in prediction.meta
        assert "relative_safety_validation_id" not in prediction.meta

    def test_config_flag_default_false(self):
        assert PlannerConfig().allow_unvalidated_relative_safety is False
        assert (
            PlannerConfig(allow_unvalidated_relative_safety=True).allow_unvalidated_relative_safety
            is True
        )

    def test_g1_id_refusal_precedes_g2_when_both_ids_missing(self):
        """Ordering fidelity: G-1's validated-True-without-id refusal still
        fires (unchanged message) even though G-2 is unvalidated too."""
        world = _v2_world()
        world.meta_script.update({"future_min_ttc_validated": True})
        with pytest.raises(FutureSafetyRefusal, match="future_min_ttc_validation_id"):
            _planner(world, PlannerConfig(mode="live")).decide(_samples(), now_s=0.0)


# ------------------------------------------------------- offline probe reuse
class TestG2ProbeReuse:
    def test_gate_prediction_reused_not_recomputed(self):
        """The G-2 diagnostic runs on the SAME probe prediction the mode
        gate used: no extra world-model call for the gate itself."""
        world = _v2_world()
        _planner(world, PlannerConfig()).decide(_samples(), now_s=0.0)
        assert world.calls and world.calls[0]["n_plans"] == 1
