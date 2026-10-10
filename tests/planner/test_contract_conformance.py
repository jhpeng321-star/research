"""PL01-E: the planner's MockWorldModel behind the official WD-01 kit.

The kit lives in a read-only sibling workspace; its tree is inserted into
sys.path here (module-level, so the kit import happens once). If the kit is
absent the module skips with the precise path; if present,
run_conformance(ContractPredictorAdapter(...)) must PASS.

Zero simulation: the kit is pure numpy; the adapter wraps MockWorldModel.
"""

import sys

import numpy as np
import pytest

from traffic_shepherd.planner.paths import rd_wt_root  # noqa: E402

KIT_ROOT = rd_wt_root() / "wd01b" / "output" / "wd01"

if not KIT_ROOT.is_dir():
    pytest.skip(
        f"WD-01 predict conformance kit not found at {KIT_ROOT} "
        "(read-only sibling workspace not present)",
        allow_module_level=True,
    )
if str(KIT_ROOT) not in sys.path:
    sys.path.insert(0, str(KIT_ROOT))

try:
    from predict import contract as kit
    from predict.conformance import ConformanceError, run_conformance
except ImportError as exc:  # pragma: no cover - kit present but broken
    pytest.skip(
        f"WD-01 predict kit is not importable from {KIT_ROOT}: {exc!r}", allow_module_level=True
    )

import traffic_shepherd.planner.contract_adapter as contract_adapter  # noqa: E402
from traffic_shepherd.planner.contract_adapter import ContractPredictorAdapter  # noqa: E402
from traffic_shepherd.planner.mock import MockWorldModel  # noqa: E402
from traffic_shepherd.planner.interface import PlanCandidate  # noqa: E402


@pytest.fixture(scope="module")
def report():
    """run_conformance on the adapter; PASS or a precise failure list."""
    try:
        return run_conformance(ContractPredictorAdapter())
    except ConformanceError as err:
        pytest.fail("conformance failed: " + "; ".join(err.failures))


def test_conformance_pass(report):
    print("CONFORMANCE", report)
    assert report.n_plans == 4 and report.n_samples == 3
    assert report.M == 1 and report.K == 3
    assert len(report.checks) == 8
    assert report.predictor_id == ContractPredictorAdapter().model_id


class TestInboundConversions:
    """kit types -> planner types."""

    def test_plan_with_walls_carries_over(self):
        plan = kit.PlanCandidate.from_raw(
            {
                "plan_version": 1,
                "per_cav": {"cav_n1": "native"},
                "walls": [
                    {
                        "wall_id": "W1",
                        "members": ("cav_a0", "cav_a1"),
                        "speed_mps": 5.0,
                        "phase": "assemble",
                        "release_at_s": None,
                        "hold_s": 30.0,
                    }
                ],
            }
        )
        inner = ContractPredictorAdapter._plan_in(plan, kit)
        assert isinstance(inner, PlanCandidate)
        assert inner.per_cav == {"cav_n1": "native"}
        assert len(inner.walls) == 1
        wall = inner.walls[0]
        assert (wall.wall_id, wall.members, wall.speed_mps, wall.phase) == (
            "W1",
            ("cav_a0", "cav_a1"),
            5.0,
            "assemble",
        )
        assert wall.hold_s == 30.0 and wall.release_at_s is None

    def test_release_now_encoding_release_at_s_zero(self):
        # kit §5 dissolution encoding: phase="release", release_at_s=0.0
        wall = kit.WallSpec(
            wall_id="W2",
            members=("cav_h0",),
            speed_mps=6.0,
            phase="release",
            release_at_s=0.0,
            hold_s=None,
        )
        plan = kit.PlanCandidate(plan_version=1, per_cav={}, walls=(wall,))
        inner = ContractPredictorAdapter._plan_in(plan, kit)
        assert inner.walls[0].phase == "release"
        assert inner.walls[0].release_at_s == 0.0
        assert inner.walls[0].hold_s is None

    def test_native_kept_assemble_wall_hold_s_inf_accepted(self):
        # kit §5: ASSEMBLE walls are kept with hold_s=math.inf ("only a
        # safety abort ends it"); the planner WallSpec accepts it as-is.
        wall = kit.WallSpec(
            wall_id="W1",
            members=("cav_a0", "cav_a1"),
            speed_mps=5.0,
            phase="assemble",
            release_at_s=None,
            hold_s=float("inf"),
        )
        inner = ContractPredictorAdapter._plan_in(
            kit.PlanCandidate(plan_version=1, per_cav={}, walls=(wall,)), kit
        )
        assert inner.walls[0].hold_s == float("inf")
        assert inner.walls[0].phase == "assemble"

    def test_controller_state_wall_action_and_native_entry(self):
        state = {
            "cav_a0": kit.ControllerStateEntry("wall", "W1", "assemble", 12.0, 5.0, True),
            "cav_n0": kit.ControllerStateEntry("native"),
        }
        inner = {cav: contract_adapter._as_mapping_entry(entry) for cav, entry in state.items()}
        assert inner["cav_a0"]["action"] == "wall"
        assert inner["cav_a0"]["wall_id"] == "W1"
        assert inner["cav_a0"]["wall_phase"] == "assemble"
        assert inner["cav_a0"]["locked"] is True
        assert inner["cav_a0"]["time_in_phase_s"] == 12.0
        assert inner["cav_a0"]["wall_speed_mps"] == 5.0
        assert inner["cav_n0"]["locked"] is False
        assert inner["cav_n0"]["wall_id"] is None

    def test_state_sample_meta_becomes_mapping(self):
        sample = kit.StateSample(
            hybrid={"tracked": {}},
            arrivals=(kit.Arrival(1.0, "mainline", 28.0, False, "car"),),
            meta=kit.SampleMeta("s0", 1000.0, True, 0.75),
        )
        inner = ContractPredictorAdapter._sample_in(sample)
        assert inner.meta == {
            "sample_id": "s0",
            "time_s": 1000.0,
            "merge_observed": True,
            "coverage": 0.75,
        }
        assert inner.hybrid == {"tracked": {}}


class TestOutboundConversions:
    """planner Prediction -> kit Prediction."""

    def _mock_prediction(self):
        state = {
            "c0": {
                "action": "wall",
                "wall_id": "w0",
                "wall_phase": "assemble",
                "locked": True,
                "time_in_phase_s": 20.0,
                "wall_speed_mps": 16.0,
            }
        }
        wall = kit.WallSpec(
            wall_id="w0",
            members=("c0", "c1"),
            speed_mps=16.0,
            phase="assemble",
            release_at_s=None,
            hold_s=30.0,
        )
        plan = ContractPredictorAdapter._plan_in(
            kit.PlanCandidate(plan_version=1, per_cav={}, walls=(wall,)), kit
        )
        mock = MockWorldModel(cav_ids=("c0", "c1"), part_a_horizon_s=10.0)
        samples = ContractPredictorAdapter._sample_in(
            kit.StateSample(hybrid={}, arrivals=(), meta=kit.SampleMeta("k0", 1000.0, True, 1.0))
        )
        mock.assembly_duration_s = 10.0
        return mock.predict_many(
            [samples],
            [plan],
            controller_state={"c0": state["c0"], "c1": dict(state["c0"], wall_id="w0")},
            trajectory_s=10,
        )[0], wall

    def test_prediction_and_meta_fields(self):
        pred, wall = self._mock_prediction()
        out = ContractPredictorAdapter()._prediction_out(pred, kit, 0.5)
        assert isinstance(out, kit.Prediction)
        assert isinstance(out.meta, kit.Meta)
        assert out.meta.contract_version == "wd01_predict/1"
        assert out.meta.weights_sha256 == ("mock:mock/1",)  # one per member
        assert out.meta.member_ids == ("m0",) and out.meta.sample_ids == ("k0",)
        assert out.meta.safety_window_s == out.meta.part_a_horizon_s
        assert out.meta.native_unlocked_available is False
        assert out.meta.wall_clock_s == 0.5
        assert list(out.meta.native_kept_walls) == ["w0"]  # locked + assemble
        # arrays pass through as numpy with the same values
        assert np.array_equal(out.delay_vs_native_s, pred.delay_vs_native_s)
        assert out.collision.dtype == np.bool_
        # unlocked diagnostics stay unset when unavailable
        assert out.trajectories_hidden is None
        assert out.native_unlocked_delay_s is None
        assert out.trajectories_tracked["c0"].shape[-1] == 4

    def test_wall_outcome_block(self):
        pred, wall = self._mock_prediction()
        out = ContractPredictorAdapter()._prediction_out(pred, kit, 0.0)
        block = out.walls[wall.wall_id]
        assert isinstance(block, kit.WallOutcome)
        assert block.assembled.dtype == np.bool_
        assert block.abort_reason.dtype == np.int8
        # assembled_at_s is NaN exactly where not assembled (validator rule)
        assert np.array_equal(np.isnan(block.assembled_at_s), ~block.assembled)
        assert np.array_equal(block.abort_reason != 0, block.safety_abort)

    def test_assembled_wall_reports_absolute_time(self):
        pred, wall = self._mock_prediction()
        out = ContractPredictorAdapter()._prediction_out(pred, kit, 0.0)
        block = out.walls[wall.wall_id]
        # time_in_phase_s 20 >= assembly_duration 10 -> assembled at t0
        assert block.assembled[0, 0]
        assert block.assembled_at_s[0, 0] == pytest.approx(1000.0)


def test_conformance_both_versions():
    """§9: the adapter passes the kit for BOTH supported contract versions
    (v1 omits the diagnostics; 1.1 carries cell_valid and the pair flags)."""
    from predict.conformance import run_conformance_both_versions

    reports = run_conformance_both_versions(
        lambda version: ContractPredictorAdapter(contract_version=version)
    )
    assert sorted(reports) == ["wd01_predict/1", "wd01_predict/1.1"]
    for version, version_report in reports.items():
        print("CONFORMANCE", version, "->", version_report)
        assert len(version_report.checks) == 8
        assert version_report.n_plans == 4 and version_report.n_samples == 3


def test_v11_requires_diagnostics_mode():
    """A 1.1-labelled adapter output without the arrays is kit-invalid
    (consumers must reject a missing cell_valid, never default it True)."""
    from predict.conformance import (
        build_controller_state,
        build_plans,
        build_state_samples,
    )
    from predict.validate import PredictionViolation, validate_prediction
    from dataclasses import replace

    adapter = ContractPredictorAdapter(contract_version="wd01_predict/1.1")
    state = build_controller_state()
    samples = build_state_samples(k=3)
    plans = build_plans(state)
    pred = adapter.predict_many(samples, [plans[0]], state)[0]
    assert pred.meta.contract_version == "wd01_predict/1.1"
    assert pred.cell_valid.shape == (1, 3) and pred.cell_valid.all()
    assert pred.has_closing_pair.shape == (1, 3)
    stripped = replace(pred, cell_valid=None)
    with pytest.raises(PredictionViolation, match="cell_valid"):
        validate_prediction(stripped, 1, 3, 1, None)
