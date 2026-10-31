import hashlib
import json
import math

import numpy as np
import pytest
import torch

from tools.rbint.motion_posterior import chain_gaps, condition_population, fit_kernels, profile_next_speed
from wms.control.fitted_hidden import FittedHiddenState, IDM_ORDER
from wms.physics.motion_fit import MotionFit


def fixture(tmp_path):
    pop = dict(schema='wm_phys.idm_population.v1', parameter_order=list(IDM_ORDER),
        distribution='finite_support_mixture', support=[[30., 1., 1., 4., 2.],
        [30., 1., 2., 4., 2.]], weights=[.5, .5], sigma_accel_mps2=1., delta=4)
    post = dict(schema='wm_phys.idm_posteriors.v1', parameter_order=list(IDM_ORDER),
        conditioning_end_s=300., drivers={f'969432:{v}': dict(support_indices=[1], weights=[1.])
                                        for v in (51, 52)})
    raw = json.dumps(pop).encode()
    (tmp_path/'idm_population.json').write_bytes(raw)
    (tmp_path/'idm_posteriors.json').write_text(json.dumps(post))
    policy = tmp_path/'policy.json'
    policy.write_text(json.dumps(dict(schema='wm_phys.idm_shrinkage.v1', prior_count=2.,
        population_only=False, idm_population_sha256=hashlib.sha256(raw).hexdigest())))
    start = dict(t0=300., seed=969432, state=dict(
        present=torch.ones(1, 3, dtype=torch.bool), is_cav=torch.tensor([[False, False, True]])))
    history = dict(time_s=np.array([298., 299., 300.]), kin=np.zeros((3, 3, 7)),
        mask=np.ones((3, 3), dtype=bool), vid_index=np.array([[51, 52, 99]]*3))
    return FittedHiddenState(pop, post), MotionFit(tmp_path), policy, start, history


def test_unseen_world_conditions_motion_in_local_id_space_without_saved_lookup(tmp_path):
    base, fit, policy, start, history = fixture(tmp_path)
    def extract(arrays, start, stop):
        assert (start, stop) == (240., 300.)
        assert arrays['is_cav'].tolist() == [[False, False, True]]*3
        rows = np.zeros((2, 11))
        rows[:, 0], rows[:, 1], rows[:, 4], rows[:, 9] = [298., 299.], 51, 1., 1.
        rows[:, 5], rows[:, 7] = np.inf, -1  # observed free-road intervals
        return rows, {}, {}
    def predict(before, gap, leader, support, dt):
        return np.array([[1., 3.], [1., 3.]])
    state, receipt = condition_population(base, fit, policy, start, history,
        kernels=(extract, predict, 'synthetic-selector-binding'))
    # Two independent Gaussian acceleration residuals: SSE0 vs8, prior1/2.
    # Learned synthetic prior_count2 yields half posterior +half population.
    expected = .5/(1+math.exp(-4))+.25
    driver = state.posterior['drivers']['969432:51']
    assert driver['weights'][0] == pytest.approx(expected)
    assert driver['posterior_fraction'] == .5
    assert '969432:52' not in state.posterior['drivers']  # no evidence -> prior
    assert base.posterior['drivers']['969432:51']['weights'] == [1.]  # not mutated/used
    assert receipt['conditioned_drivers'] == receipt['population_fallbacks'] == 1
    assert receipt['saved_driver_id_lookup'] is False


def test_future_frames_and_noncausal_selected_intervals_fail_before_inference(tmp_path):
    base, fit, policy, start, history = fixture(tmp_path)
    history['time_s'][-1] = 300.5
    with pytest.raises(ValueError, match='causal'):
        condition_population(base, fit, policy, start, history)
    history['time_s'][-1] = 300.
    def future_row(arrays, **kwargs):
        row = np.zeros((1, 11))
        row[0, [0, 1, 9]] = [299.75, 51., .5]
        return row, {}, {}
    with pytest.raises(ValueError, match='noncausal intervals'):
        condition_population(base, fit, policy, start, history,
            kernels=(future_row, lambda *a: pytest.fail('must not predict'), 'synthetic'))


def test_fit_selector_and_active_cf_core_need_no_optional_estimation_dependencies():
    extract, predict, source_sha = fit_kernels()
    params = np.array([[30., 1., 1., 4., 2.], [30., 1., 2., 4., 2.]])
    speeds = predict(np.array([[0.], [10.]]), np.full((2, 1), np.inf),
                     np.zeros((2, 1)), params, .5)
    assert speeds.shape == (2, 2) and np.isfinite(speeds).all()
    assert speeds[0].tolist() == pytest.approx([.5, 1.])
    assert callable(extract) and len(source_sha) == 64


def test_runtime_profiles_use_active_sumo_lane_cap_not_the_older_fit_approximation():
    # At31m/s with30m/s desired speed and b4.5, planMove laneCap is30,
    # not the gently decelerating raw IDM safe speed (~30.93).
    profiles = np.array([[30., 1., 1., 4.5, 2.], [30., 1., 2., 4.5, 2.]])
    predicted = profile_next_speed(np.array([[31.]]), np.array([[np.inf]]),
                                   np.zeros((1, 1)), profiles, .5)
    assert predicted.tolist() == [[30., 30.]]


def test_conditioning_leader_gap_includes_the_same_registered_link_as_rollout():
    kin = np.zeros((1, 2, 7), dtype=np.float32)
    kin[0, :, 0], kin[0, :, 1], kin[0, :, 6] = [-8., 5.], 1., 5.
    arrays = dict(time_s=np.array([299.5]), veh_kin=kin,
        vid=np.array([[51, 52]]), veh_mask=np.ones((1, 2), dtype=bool))
    row = np.zeros((1, 11))
    row[0, [0, 1, 5, 7]] = [299.5, 51., 8., 52.]
    converted = chain_gaps(row, arrays)
    assert converted[0, 5] == 11.  # 13m raw separation+3m link−5m leader length
    assert row[0, 5] == 8.  # original observed rows are not mutated
