"""Interface types mirror the frozen wd01_predict/1 contract validation."""
from types import SimpleNamespace

import numpy as np
import pytest

from traffic_shepherd.planner import (
    ACTIONS,
    PlanCandidate,
    Prediction,
    WallSpec,
    WorldModel,
    locked_entries,
    plan_key,
    with_actions,
)
from tests.planner._helpers import plan, wall


class TestWallSpec:
    def test_exactly_one_of_release_or_hold(self):
        WallSpec("w", ("c0",), 16.0, "hold", hold_s=60.0)
        WallSpec("w", ("c0",), 16.0, "hold", release_at_s=30.0)
        with pytest.raises(ValueError):
            WallSpec("w", ("c0",), 16.0, "hold")
        with pytest.raises(ValueError):
            WallSpec("w", ("c0",), 16.0, "hold", hold_s=60.0, release_at_s=30.0)

    def test_phase_vocabulary(self):
        with pytest.raises(ValueError):
            wall("w", ("c0",), phase="cruise")

    def test_members_unique_and_nonempty(self):
        with pytest.raises(ValueError):
            wall("w", ())
        with pytest.raises(ValueError):
            wall("w", ("c0", "c0"))


class TestPlanCandidate:
    def test_action_vocabulary(self):
        assert ACTIONS == ("native", "damping", "pi-external-u")
        with pytest.raises(ValueError):
            plan({"c0": "emergency-stop"})

    def test_plan_version_is_1(self):
        with pytest.raises(ValueError):
            PlanCandidate(per_cav={"c0": "native"}, plan_version=2)

    def test_wall_members_never_in_per_cav(self):
        with pytest.raises(ValueError):
            plan({"c0": "native", "c1": "damping"}, walls=[wall("w", ("c1",))])

    def test_wall_membership_disjoint(self):
        with pytest.raises(ValueError):
            plan({"c0": "native"},
                 walls=[wall("w1", ("c1",)), wall("w2", ("c1",))])

    def test_duplicate_wall_id_rejected(self):
        with pytest.raises(ValueError):
            plan({"c0": "native"},
                 walls=[wall("w1", ("c1",)), wall("w1", ("c2",))])

    def test_free_cavs_sorted(self):
        assert plan({"b": "native", "a": "damping"}).free_cavs == ("a", "b")


class TestPlanHelpers:
    def test_plan_key_insertion_order_independent(self):
        assert plan_key(plan({"a": "native", "b": "damping"})) == \
            plan_key(plan({"b": "damping", "a": "native"}))

    def test_plan_key_distinguishes_walls(self):
        plain = plan_key(plan({"a": "native", "b": "damping"}))
        walled = plan_key(plan({"a": "native"}, walls=[wall("w", ("b",))]))
        assert plain != walled

    def test_with_actions_keeps_walls_and_original(self):
        base = plan({"a": "native", "b": "native"}, walls=[wall("w", ("c",))])
        updated = with_actions(base, {"a": "damping"})
        assert updated.per_cav == {"a": "damping", "b": "native"}
        assert updated.walls == base.walls
        assert base.per_cav["a"] == "native"

    def test_with_actions_rejects_wall_members(self):
        base = plan({"a": "native"}, walls=[wall("w", ("b",))])
        with pytest.raises(ValueError):
            with_actions(base, {"b": "damping"})

    def test_locked_entries(self):
        state = {
            "a": {"action": "damping", "locked": False},
            "b": {"action": "wall", "wall_id": "w", "locked": True},
        }
        assert set(locked_entries(state)) == {"b"}
        assert locked_entries(None) == {}
        with pytest.raises(ValueError):
            locked_entries({"b": {"action": "wall", "locked": True}})


class TestWorldModelProtocol:
    def test_predict_delegates_to_predict_many(self):
        marker = Prediction(
            delay_vs_native_s=np.zeros((1, 1)), plan_delay_s=np.zeros((1, 1)),
            native_delay_s=np.zeros((1, 1)), delay_part_a_s=np.zeros((1, 1)),
            delay_part_b_s=np.zeros((1, 1)), collision=np.zeros((1, 1), dtype=bool),
            min_ttc_s=np.zeros((1, 1)), ttc_lt3_exposure_s=np.zeros((1, 1)),
            native_ttc_lt3_exposure_s=np.zeros((1, 1)), hard_brake_s=np.zeros((1, 1)),
            native_hard_brake_s=np.zeros((1, 1)),
            native_collision=np.zeros((1, 1), dtype=bool),
            native_min_ttc_s=np.zeros((1, 1)))

        class Fake(WorldModel):
            seen = []

            def predict_many(self, state_samples, plans, *, controller_state=None,
                             horizon_s=600.0, trajectory_s=None):
                self.seen.append([p.per_cav for p in plans])
                return [marker for _ in plans]

        fake = Fake()
        assert fake.predict([], plan({"c0": "native"}), None) is marker
        assert fake.seen == [[{"c0": "native"}]]

    def test_state_sample_is_structural(self):
        # StateSample is a typing Protocol: any object with the attributes
        # satisfies it; SimpleNamespace is the test stand-in.
        sample = SimpleNamespace(hybrid=None, arrivals=(), meta={})
        assert sample.hybrid is None and sample.meta == {}
