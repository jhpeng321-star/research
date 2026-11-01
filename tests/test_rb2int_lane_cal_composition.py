"""LC attachment must not silently remove the calibrated physical latent wrapper."""
from types import SimpleNamespace

import torch

from tools.rbint import recipe, observations, motion_posterior
from tools.rblane import runtime_history
from tools.rb2int.audit import ROOT
from wms.control.fitted_hidden import FittedHiddenState
from wms.control.spread_calibration import SpreadCalibration, CalibratedHidden
from wms.physics.lane_continuation import LaneContinuation


def test_causal_lc_attachment_preserves_calibrated_entrant_mixture(monkeypatch):
    base = FittedHiddenState.from_directory(ROOT/'artifacts/rbfit/shrinkage_v1')
    calibration = SpreadCalibration.from_file(ROOT/'docs/paper_results/rb2cal/v1/fit/calibration.json')
    fitted = calibration.hidden(base)
    lane = LaneContinuation.from_directory(ROOT/'artifacts/rbfit/lane_raw_c5602102891e')
    monkeypatch.setattr(lane, 'condition', lambda *a, **k: ({}, dict(lc_test=True)))
    monkeypatch.setattr(observations, 'load_motion_history', lambda s: ({}, {}))
    monkeypatch.setattr(runtime_history, 'lane_arrays', lambda *a: {})
    monkeypatch.setattr(motion_posterior, 'condition_population', lambda f, *a:
                        (f.with_motion_posteriors({}, t0_s=300.), dict(causal_test=True)))
    model = SimpleNamespace(_wmphys_fitted_state=fitted, _wmphys_condition_motion=True,
        _wmphys_conditioned_cache={}, _wmphys_lane_continuation=lane,
        _wmphys_motion_fit=None, _wmphys_shrinkage_policy=None, physics=None)
    conditioned, receipt = recipe.fitted_state_at_start(model, 969420)
    assert isinstance(conditioned, CalibratedHidden)
    assert conditioned.calibration is calibration
    assert conditioned.lane_population is not None
    assert receipt['lc_test'] is True
    mixture = conditioned.entrant_weights(969420, seed=190)
    assert torch.isclose(mixture.sum(), torch.tensor(1., dtype=mixture.dtype))
    draw = conditioned.draw(969420, None, generator=torch.Generator().manual_seed(1), lane=True)
    assert 'lcKeepRight' in draw
