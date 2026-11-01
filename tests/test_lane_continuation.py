import numpy as np
import pytest
import torch

from wms.control.fitted_hidden import LC_ORDER
from wms.physics.config import PhysicsConfig
from wms.physics.lane_continuation import LaneContinuation, warm_lane_memory
from wms.physics.lc2013 import lane_change
from tests.test_lc2013 import scene
from tests.unit.test_fitted_hidden import fitted


def continuation():
    pop = dict(schema='wm_phys.lc2013_population.v1', parameter_order=LC_ORDER,
        fit_interval_s=[240, 300], fit_worlds=list(range(969400, 969432)), snapshot_version='v2',
        estimation='physical categorical one-hot least squares; causal symmetric-likelihood driver Bayes',
        support=[[1., 1., 1., 1., 1.], [0., 0., 0., 0., 1.]], weights=[.5, .5],
        idm_parameters=dict(v0=30., accel=1., decel=2., time_headway=1., min_gap=2.),
        lc2013_source_sha256='test', observation_emission=dict(probabilities=
            (np.tile((np.eye(3)+.5*(1-np.eye(3)))[None], (3, 1, 1))/2).tolist()))
    return LaneContinuation(pop)


def history(vid=41):
    times = np.arange(121)*.5+240.
    kin = np.zeros((121, 1, 7), dtype=np.float64)
    kin[:, 0, 0], kin[:, 0, 1], kin[:, 0, 4], kin[:, 0, 6] = -1000., 1., 25., 5.
    return dict(time_s=times, veh_kin=kin, veh_mask=np.ones((121, 1), dtype=bool),
        vid=np.full((121, 1), vid, dtype=np.int64), is_cav=np.zeros((121, 1), dtype=bool))


def test_no_event_history_selects_persistent_disabled_profile_without_event_noise():
    fit, physics = continuation(), PhysicsConfig(sumo_fidelity=True)
    records, receipt = fit.condition(history(), world=969400, t0_s=300., physics=physics)
    record = records[969400, 41]
    assert record['weights'][1] > .99
    assert record['n_observations'] == 120
    reader = fit.attach(fitted(), records, t0_s=300., receipt=receipt)
    assert reader.posterior == fitted().posterior  # IDM block unchanged
    assert reader.provenance['saved_lc_driver_id_lookup'] is False
    assert reader.lane_posterior['drivers']['969400:41']['weights'] == record['weights']
    assert reader.draw(969400, 999, lane=True, generator=torch.Generator().manual_seed(1))


def test_future_motion_and_duplicate_identity_rejected_before_profile_evaluation():
    fit, physics = continuation(), PhysicsConfig(sumo_fidelity=True)
    a = history()
    a['time_s'][-1] = 300.5
    with pytest.raises(ValueError, match='causal'):
        fit.condition(a, world=969400, t0_s=300., physics=physics)
    a = history()
    for key in ('veh_kin', 'veh_mask', 'vid', 'is_cav'):
        a[key] = np.repeat(a[key], 2, axis=1)
    with pytest.raises(ValueError, match='unique'):
        fit.condition(a, world=969400, t0_s=300., physics=physics)


def test_relabelled_observed_identity_conditions_identically_without_saved_id_lookup():
    fit, physics = continuation(), PhysicsConfig(sumo_fidelity=True)
    a, _ = fit.condition(history(41), world=969400, t0_s=300., physics=physics)
    b, _ = fit.condition(history(500), world=969431, t0_s=300., physics=physics)
    assert a[969400, 41] == b[969431, 500]


def test_warm_memory_is_packed_by_identity_and_reset_on_last_observed_change():
    a, physics = history(), PhysicsConfig(sumo_fidelity=True)
    ids = torch.tensor([[999, 41, -1]])
    params = {name: torch.ones((1, 3)) for name in LC_ORDER}
    result = warm_lane_memory(a, t0_s=300., physics=physics, driver_params=params, vehicle_ids=ids)
    assert result['lc2013_present'].tolist() == [[False, True, False]]
    assert result['lc2013_keep_right'][0, 1] < -20_000_000
    assert result['lc2013_keep_right'][0, 0] == 0
    a['veh_kin'][-1, 0, 1:3] = [0., 1.]
    changed = warm_lane_memory(a, t0_s=300., physics=physics, driver_params=params, vehicle_ids=ids)
    assert changed['lc2013_keep_right'][0, 1] == 0


def test_non_action_step_freezes_all_wishes_and_lookahead():
    kin, mask = scene([(0., 0, 25.)])
    p = PhysicsConfig(sumo_fidelity=True)
    memory = lane_change(kin, mask, p)
    kin[..., 4] = 5.
    frozen = lane_change(kin, mask, p, memory=memory, decision_mask=torch.zeros_like(mask))
    for name in ('lc2013_left', 'lc2013_right', 'lc2013_keep_right', 'lc2013_lookahead'):
        assert torch.equal(frozen[name], memory[name])
    assert not frozen['lc2013_wish'].any()


def test_longer_action_step_uses_source_reaction_time_secure_gap():
    kin, mask = scene([(40., 2, 10.), (59., 1, 10.)])
    kin[0, 0, 5], kin[0, 1, 5] = 1., -1.
    p = PhysicsConfig()
    ordinary = lane_change(kin, mask, p)
    slower = lane_change(kin, mask, p, driver_params={'actionStepLength': torch.full((1, 2), 1.)})
    assert ordinary['lc2013_safe'][0, 0]
    assert not slower['lc2013_safe'][0, 0]


def test_recipe_attaches_lc_after_causal_idm_conditioning(monkeypatch):
    from types import SimpleNamespace
    from tools.rbint import recipe, observations, motion_posterior
    from tools.rblane import runtime_history
    fit, physics = continuation(), PhysicsConfig(sumo_fidelity=True)
    start, raw = {'synthetic': True}, {'synthetic': True}
    monkeypatch.setattr(observations, 'load_motion_history', lambda seed:(start, raw))
    monkeypatch.setattr(runtime_history, 'lane_arrays', lambda s,r:history())
    monkeypatch.setattr(motion_posterior, 'condition_population', lambda *args:(fitted(), {'idm_causal': True}))
    model = SimpleNamespace(_wmphys_fitted_state=fitted(), _wmphys_condition_motion=True,
        _wmphys_conditioned_cache={}, _wmphys_motion_fit=None, _wmphys_shrinkage_policy=None,
        _wmphys_lane_continuation=fit, physics=physics)
    result, receipt = recipe.fitted_state_at_start(model, 969400)
    assert result.lane_population is not None and receipt['idm_causal']
    assert set(result.lane_posterior['drivers']) == {'969400:41'}
    assert receipt['saved_lc_driver_id_lookup'] is False
    again, _ = recipe.fitted_state_at_start(model, 969400)
    assert again is result


def test_explicit_recipe_changes_only_lc_opt_in_and_artifact():
    import tomllib
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    baseline = tomllib.loads((root/'configs/wm_phys.toml').read_text())
    candidate = tomllib.loads((root/'configs/wm_phys_rb2_lane.toml').read_text())
    full = candidate['variants']['FULL']
    assert full.pop('lane_parameter_directory') == 'artifacts/rbfit/lane_raw_c5602102891e'
    assert full['fitted_lc'] is True
    full['fitted_lc'] = False
    assert candidate == baseline
