"""The conformance kit: reference predictor passes, broken predictors fail."""
import dataclasses

import numpy as np
import pytest

from predict import (
    ConformanceError,
    ConformanceReport,
    Predictor,
    build_controller_state,
    build_plans,
    build_state_samples,
    run_conformance,
)
from predict.conformance import CHECKS, _pred_equal
from predict.reference_const_speed import ConstSpeedPredictor


def test_reference_predictor_passes_conformance():
    report = run_conformance(ConstSpeedPredictor())
    assert isinstance(report, ConformanceReport)
    assert report.checks == CHECKS
    assert report.predictor_id == "reference_const_speed/1"
    assert (report.n_plans, report.n_samples, report.M, report.K) == (4, 3, 1, 3)


def test_reference_passes_with_explicit_trajectory_s():
    report = run_conformance(ConstSpeedPredictor(), trajectory_s=60.0)
    assert report.trajectory_s == 60.0


def test_protocol_is_runtime_checkable():
    assert isinstance(ConstSpeedPredictor(), Predictor)


def test_k_samples_must_be_at_least_three():
    with pytest.raises(ValueError, match="k >= 3"):
        run_conformance(ConstSpeedPredictor(), k_samples=2)


# --- deliberately broken predictors; each must be caught -----------------------

class ReversedOrder(ConstSpeedPredictor):
    def predict_many(self, state_samples, plans, controller_state, **kw):
        out = super().predict_many(state_samples, plans, controller_state, **kw)
        return list(reversed(out))


class Nondeterministic(ConstSpeedPredictor):
    def predict_many(self, state_samples, plans, controller_state, **kw):
        out = super().predict_many(state_samples, plans, controller_state, **kw)
        rng = np.random.default_rng()
        return [
            dataclasses.replace(
                p, plan_delay_s=p.plan_delay_s + rng.random(p.plan_delay_s.shape)
            )
            for p in out
        ]


class NativeDrift(ConstSpeedPredictor):
    """native_* arrays differ across Predictions of one call (§2 violation)."""

    def predict_many(self, state_samples, plans, controller_state, **kw):
        out = super().predict_many(state_samples, plans, controller_state, **kw)
        return [
            dataclasses.replace(p, native_delay_s=p.native_delay_s + i * 7.0)
            for i, p in enumerate(out)
        ]


class BadShape(ConstSpeedPredictor):
    def predict_many(self, state_samples, plans, controller_state, **kw):
        out = super().predict_many(state_samples, plans, controller_state, **kw)
        return [dataclasses.replace(p, plan_delay_s=p.plan_delay_s[:, :-1]) for p in out]


class BrokenPredict(ConstSpeedPredictor):
    """predict() deviates from predict_many(...)[0] (§2 identity violation)."""

    def predict(self, state_samples, plan, controller_state, **kw):
        pred = super().predict(state_samples, plan, controller_state, **kw)
        return dataclasses.replace(pred, plan_delay_s=pred.plan_delay_s + 1.0)


class WrongSampleIds(ConstSpeedPredictor):
    def predict_many(self, state_samples, plans, controller_state, **kw):
        out = super().predict_many(state_samples, plans, controller_state, **kw)
        return [
            dataclasses.replace(
                p,
                meta=dataclasses.replace(
                    p.meta, sample_ids=tuple(p.meta.sample_ids[:-1])
                ),
            )
            for p in out
        ]


@pytest.mark.parametrize(
    "cls,fragment",
    [
        (ReversedOrder, "output_order_matches_plans"),
        (Nondeterministic, "deterministic_on_repeat_call"),
        (NativeDrift, "native_shared_within_call"),
        (BadShape, "validators_pass"),
        (BrokenPredict, "predict_matches_predict_many"),
        (WrongSampleIds, "validators_pass"),
    ],
)
def test_broken_predictors_are_caught(cls, fragment):
    with pytest.raises(ConformanceError) as err:
        run_conformance(cls())
    assert fragment in str(err.value)


# --- kit equality helper --------------------------------------------------------

def test_pred_equal_ignores_wall_clock_s():
    cs = build_controller_state()
    samples = build_state_samples(k=3)
    plans = build_plans(cs)
    predictor = ConstSpeedPredictor()
    a = predictor.predict_many(samples, plans[:2], cs)[0]
    b = predictor.predict_many(samples, plans[:2], cs)[0]
    assert _pred_equal(a, b)
    changed = dataclasses.replace(
        a, meta=dataclasses.replace(a.meta, wall_clock_s=a.meta.wall_clock_s + 5.0)
    )
    assert _pred_equal(changed, b)
    assert not _pred_equal(
        dataclasses.replace(a, plan_delay_s=a.plan_delay_s + 0.5), b
    )
