"""PL-03a tests: KitWorldModel + real-state builders (WD-01 p2C, read-only).

Zero simulation: the ConstSpeedPredictor is a pure-numpy reference. Real
dataset tests skip precisely when the WD-01 kit's partA_data (root from
paths.rd_wt_root()) is absent.
"""

import sys
from pathlib import Path

import pytest

from traffic_shepherd.planner import Planner, PlannerConfig, PlanCandidate
from traffic_shepherd.planner.kit_world import (
    controller_state_out,
    plan_out,
    P2C_DATA_ROOT,
    PREDICT_KIT_ROOT,
    KitWorldModel,
    assert_train_only,
    build_real_state,
    load_p2c_loader,
    load_predict_kit,
)

DATA_READY = Path(P2C_DATA_ROOT).is_dir()


def _predictor(contract_version="wd01_predict/1.1"):
    contract, _ = load_predict_kit()
    sys.path.insert(0, PREDICT_KIT_ROOT)
    from predict.reference_const_speed import ConstSpeedPredictor

    return ConstSpeedPredictor(contract_version=contract_version)


def _hybrid_sample(sample_id="k0", t0=60.0):
    contract, _ = load_predict_kit()
    tracked = {"c0": (100.0, 0, 20.0), "c1": (102.0, 1, 20.0)}
    arrivals = (
        contract.Arrival(
            t_s=t0 + 10.0, entrance="mainline", speed_mps=25.0, is_cav=False, vehicle_type="car"
        ),
    )
    return contract.StateSample(
        hybrid={"tracked": tracked, "section_len_m": 3199.0, "t0": t0},
        arrivals=arrivals,
        meta=contract.SampleMeta(sample_id, t0, True, 1.0),
    )


class TestTrainOnlyGuard:
    def test_holdout_roots_rejected(self):
        with pytest.raises(AssertionError, match="holdout"):
            assert_train_only(
                ["a", "seed_7012m2700"], {"train": ["a"], "holdout": ["seed_7012m2700"]}
            )

    def test_train_roots_accepted(self):
        assert_train_only(["a"], {"train": ["a"], "holdout": ["h"]})


class TestConversions:
    def test_plan_out_maps_fields(self):
        contract, _ = load_predict_kit()
        plan = PlanCandidate(per_cav={"c0": "damping"})
        kit_plan = plan_out(plan, contract)
        assert kit_plan.plan_version == 1
        assert kit_plan.per_cav == {"c0": "damping"}
        assert kit_plan.walls == ()

    def test_plan_out_maps_walls(self):
        contract, _ = load_predict_kit()
        from traffic_shepherd.planner import WallSpec

        plan = PlanCandidate(
            per_cav={},
            walls=[
                WallSpec(
                    wall_id="w0",
                    members=("c0", "c1"),
                    speed_mps=16.0,
                    phase="assemble",
                    hold_s=60.0,
                )
            ],
        )
        kit_plan = plan_out(plan, contract)
        assert kit_plan.walls[0].wall_id == "w0"
        assert kit_plan.walls[0].members == ("c0", "c1")
        assert kit_plan.walls[0].hold_s == 60.0

    def test_controller_state_out_fills_defaults(self):
        contract, _ = load_predict_kit()
        state = controller_state_out({"c0": {"action": "damping", "locked": False}}, contract)
        entry = state["c0"]
        assert entry.action == "damping"
        assert entry.wall_id is None and entry.locked is False
        assert entry.time_in_phase_s == 0.0


class TestKitWorldModel:
    def test_predict_many_returns_planner_predictions(self):
        predictor = _predictor()
        world = KitWorldModel(predictor)
        samples = [_hybrid_sample()]
        plan = PlanCandidate(per_cav={"c0": "damping", "c1": "damping"})
        preds = world.predict_many(
            samples,
            [plan],
            {
                "c0": {"action": "damping", "locked": False},
                "c1": {"action": "damping", "locked": False},
            },
            horizon_s=60.0,
        )
        assert len(preds) == 1
        pred = preds[0]
        assert pred.meta["contract_version"] == "wd01_predict/1.1"
        assert pred.cell_valid.shape == (1, 1)
        assert pred.delay_vs_native_s.shape == (1, 1)
        assert world.calls[-1]["n_plans"] == 1

    def test_boundary_rejections_recorded(self):
        predictor = _predictor()
        world = KitWorldModel(predictor)
        # locked CAV moved into per_cav: kit validate_plan must reject
        bad = PlanCandidate(per_cav={"c0": "damping"})
        with pytest.raises(Exception) as excinfo:
            world.predict_many(
                [_hybrid_sample()],
                [bad],
                {
                    "c0": {
                        "action": "wall",
                        "wall_id": "w9",
                        "wall_phase": "assemble",
                        "locked": True,
                        "time_in_phase_s": 1.0,
                        "wall_speed_mps": 16.0,
                    }
                },
                horizon_s=60.0,
            )
        assert "locked" in str(excinfo.value).lower() or "per_cav" in str(excinfo.value).lower()
        assert world.rejections


class TestRealState:
    @pytest.mark.skipif(
        not Path(P2C_DATA_ROOT).is_dir(), reason=f"p2C dataset not present at {P2C_DATA_ROOT}"
    )
    def test_real_state_build_and_validate(self):
        loader = load_p2c_loader()
        train = sorted(loader.list_roots_by_split()["train"])
        root = train[0]
        bundle = build_real_state(root, 240.0, loader=loader)
        assert bundle.t0_s == 240.0
        assert bundle.cav_ids, "expected CAVs on physical lanes"
        assert set(bundle.frames) == set(bundle.cav_ids)
        assert all(t == 240.0 for t in bundle.frames.values())
        contract, validate = load_predict_kit()
        validate.validate_state_samples(bundle.state_samples)

    @pytest.mark.skipif(
        not Path(P2C_DATA_ROOT).is_dir(), reason=f"p2C dataset not present at {P2C_DATA_ROOT}"
    )
    def test_dry_run_decision_on_real_state(self):
        loader = load_p2c_loader()
        train = sorted(loader.list_roots_by_split()["train"])
        root = train[0]
        bundle = build_real_state(root, 240.0, loader=loader)
        world = KitWorldModel(_predictor())
        planner = Planner = __import__("traffic_shepherd.planner", fromlist=["Planner"]).Planner
        # the FROZEN kit v2.1 fields carry the 3 s exposure only: the
        # T-A fail-closed rule selects the 3 s metric explicitly
        planner = Planner(world, PlannerConfig(relative_ttc_threshold_s=3.0))
        decision = planner.decide(
            bundle.state_samples, 240.0, controller_state=bundle.controller_state
        )
        assert decision.plan.per_cav  # a plan was chosen on the real state
        assert not decision.plan.walls  # stage-1 data has no walls
        for record in decision.reasons:
            assert record.step


class TestMergeObservedSection815:
    """§8.15: merge_observed is True IFF the whole merge region is within
    sensor coverage, REGARDLESS of occupancy; never derived from presence."""

    class _FakeLoader:
        """Synthetic p2C window: three vehicles on the mainline lanes, an
        EMPTY ramp/accel lane (no lane-2 vehicle, no arrivals)."""

        DT = 0.5

        def windows(self, root, t0, horizon_s):
            state = {
                "id": ["v0", "v1", "v2"],
                "vid": [0, 1, 2],
                "lane": [0, 1, 1],
                "s": [100.0, 300.0, 500.0],
                "v": [20.0, 20.0, 20.0],
                "a": [0.0, 0.0, 0.0],
                "is_cav": [1, 0, 1],
                "in_junction": [0, 0, 0],
            }
            arrivals = {
                "t_s": [],
                "entrance": [],
                "speed_mps": [],
                "is_cav": [],
                "vehicle_type": [],
            }
            return {"t0": t0, "horizon_s": horizon_s, "state": state, "arrivals": arrivals}

    def test_empty_ramp_is_observed(self):
        bundle = build_real_state("synthetic", 240.0, loader=self._FakeLoader())
        # no vehicle on the accel/ramp lane at all: occupancy-based logic
        # would say False; §8.15 says the region is COVERED -> True
        assert bundle.merge_observed is True

    def test_rsu_report_missing_sets_false(self):
        bundle = build_real_state(
            "synthetic", 240.0, loader=self._FakeLoader(), rsu_report_missing=True
        )
        assert bundle.merge_observed is False
        assert bundle.rsu_report_missing is True

    def test_abstain_only_when_rsu_report_missing(self):
        from traffic_shepherd.planner import Planner, PlannerConfig

        healthy = build_real_state("synthetic", 240.0, loader=self._FakeLoader())
        missing = build_real_state(
            "synthetic", 240.0, loader=self._FakeLoader(), rsu_report_missing=True
        )
        # the FROZEN kit v2.1 fields carry the 3 s exposure only: the
        # T-A fail-closed rule selects the 3 s metric explicitly
        healthy_decision = Planner(
            KitWorldModel(_predictor()), PlannerConfig(relative_ttc_threshold_s=3.0)
        ).decide(healthy.state_samples, 240.0, controller_state=healthy.controller_state)
        missing_decision = Planner(
            KitWorldModel(_predictor()), PlannerConfig(relative_ttc_threshold_s=3.0)
        ).decide(missing.state_samples, 240.0, controller_state=missing.controller_state)
        assert healthy_decision.abstain is False
        assert missing_decision.abstain is True


class TestHybridMetaAddendum:
    """Contract §8.15 ADDENDUM: hybrid_state/2 may carry merge_observed=None
    (no RSU inputs) plus merge_observed_reason; the builder maps None to
    False (fail closed) and the planner abstains with the reason logged."""

    class _FakeLoader:
        DT = 0.5

        def windows(self, root, t0, horizon_s):
            state = {
                "id": ["v0", "v1", "v2"],
                "vid": [0, 1, 2],
                "lane": [0, 1, 1],
                "s": [100.0, 300.0, 500.0],
                "v": [20.0, 20.0, 20.0],
                "a": [0.0, 0.0, 0.0],
                "is_cav": [1, 0, 1],
                "in_junction": [0, 0, 0],
            }
            return {
                "t0": t0,
                "horizon_s": horizon_s,
                "state": state,
                "arrivals": {
                    "t_s": [],
                    "entrance": [],
                    "speed_mps": [],
                    "is_cav": [],
                    "vehicle_type": [],
                },
            }

    def test_none_merge_observed_maps_to_false_and_abstains_with_reason(self):
        bundle = build_real_state(
            "synthetic",
            240.0,
            loader=self._FakeLoader(),
            merge_observed=None,
            merge_observed_reason="no RSU inputs for t0",
        )
        assert bundle.merge_observed is False  # fail closed
        assert bundle.merge_observed_reason == "no RSU inputs for t0"
        # the hybrid meta carries the RAW addendum values (None stays None)
        assert bundle.state_samples[0].hybrid["meta"]["merge_observed"] is None
        # and Planner.decide abstains with the reason logged
        world = KitWorldModel(_predictor())
        # the FROZEN kit v2.1 fields carry the 3 s exposure only: the
        # T-A fail-closed rule selects the 3 s metric explicitly
        planner = Planner(world, PlannerConfig(relative_ttc_threshold_s=3.0))
        decision = planner.decide(
            bundle.state_samples, 240.0, controller_state=bundle.controller_state
        )
        assert decision.abstain is True
        abstains = [r for r in decision.reasons if r.step == "abstain"]
        assert abstains and "no RSU inputs for t0" in abstains[0].detail

    def test_true_stays_true(self):
        bundle = build_real_state(
            "synthetic",
            240.0,
            loader=self._FakeLoader(),
            merge_observed=True,
            merge_observed_reason="RSU coverage confirmed",
        )
        assert bundle.merge_observed is True
        assert bundle.merge_observed_reason == "RSU coverage confirmed"

    def test_false_with_reason_is_carried(self):
        bundle = build_real_state(
            "synthetic",
            240.0,
            loader=self._FakeLoader(),
            merge_observed=False,
            merge_observed_reason="RSU offline at t0",
        )
        assert bundle.merge_observed is False
        assert bundle.merge_observed_reason == "RSU offline at t0"

    def test_hybrid_meta_carries_addendum_fields(self):
        bundle = build_real_state(
            "synthetic",
            240.0,
            loader=self._FakeLoader(),
            merge_observed=True,
            merge_observed_reason=None,
            rsu_staleness_bound_s=0.5,
            rsu_report_age_s={"ramp": 0.2, "merge": 0.1, "main": 0.3, "down": 0.4},
        )
        hybrid_meta = bundle.state_samples[0].hybrid["meta"]
        assert hybrid_meta["rsu_staleness_bound_s"] == 0.5
        assert hybrid_meta["rsu_report_age_s"]["ramp"] == 0.2


class TestHoldoutNeverTouched:
    def test_real_splits_guard(self):
        loader = load_p2c_loader()
        splits = loader.list_roots_by_split()
        assert_train_only(splits.get("train", ()), splits)
        # and the guard really rejects holdout selections
        with pytest.raises(AssertionError):
            assert_train_only(splits.get("holdout", ()), splits)
