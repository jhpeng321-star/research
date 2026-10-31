import numpy as np
import pytest
import torch

from tools.rbint.endpoint_targets import causal_lane_carry, match_targets, match_targets_with_carry


def test_factual_mapping_uses_exact_observed_state_then_persistent_id_after_slot_recycling():
    start = dict(state=dict(present=torch.tensor([[True, True, False]]),
        junction=torch.tensor([[False, True, False]]), lane=torch.tensor([[0, 2, 0]]),
        s_m=torch.tensor([[-8., 74., 0.]]), v_mps=torch.tensor([[12., 5., 0.]]),
        is_cav=torch.zeros(1, 3, dtype=torch.bool)),
        vehicle_ids=['unused']*8)
    start['vehicle_ids'][7], start['vehicle_ids'][2] = 'mainline.41', 'ramp.77'
    initial = np.zeros((2, 7), dtype=np.float32)
    initial[:, 0], initial[:, 4], initial[1, 1] = [74., -8.], [5., 12.], 1.
    future = np.zeros((4, 7), dtype=np.float32)
    future[:, 0], future[:, 2] = [10., 80., 0., 300.], 1.
    rows = match_targets(start, np.array([7, 2, -1]), initial, np.ones(2, bool),
        np.array([311, 88]), np.zeros(2, bool), future, np.array([True, True, False, True]),
        np.array([777, 311, -1, 88]), np.zeros(4, bool))
    assert [(r['vehicle_id'], r['s_m']) for r in rows] == [('mainline.41', 300.), ('ramp.77', 80.)]


def test_ambiguous_or_not_exact_t0_matching_never_uses_nearest_position():
    start = dict(state=dict(present=torch.tensor([[True]]), lane=torch.tensor([[0]]),
        s_m=torch.tensor([[10.]]), v_mps=torch.tensor([[0.]]), is_cav=torch.tensor([[False]])),
        vehicle_ids=['ego'])
    kin = np.zeros((2, 7), dtype=np.float32)
    kin[:, 0], kin[:, 1] = 10., 1.
    with pytest.raises(ValueError, match='EXACT and unique'):
        match_targets(start, np.array([0]), kin, np.ones(2, bool), np.array([90, 91]),
            np.zeros(2, bool), kin, np.ones(2, bool), np.array([90, 91]), np.zeros(2, bool))
    kin = kin[:1].copy()
    kin[0, 0] = np.nextafter(np.float32(10.), np.float32(11.))
    with pytest.raises(ValueError, match='EXACT and unique'):
        match_targets(start, np.array([0]), kin, np.ones(1, bool), np.array([90]),
            np.zeros(1, bool), kin, np.ones(1, bool), np.array([90]), np.zeros(1, bool))


def test_future_junction_carried_position_is_censored_not_a_false_endpoint():
    start = dict(state=dict(present=torch.tensor([[True]]), lane=torch.tensor([[0]]),
        s_m=torch.tensor([[10.]]), v_mps=torch.tensor([[0.]]), is_cav=torch.tensor([[False]])),
        vehicle_ids=['ego'])
    initial = np.zeros((1, 7), dtype=np.float32)
    initial[0, 0], initial[0, 1] = 10., 1.
    future = initial.copy(); future[:, 1:4] = 0
    rows = match_targets(start, np.array([0]), initial, np.ones(1, bool), np.array([90]),
        np.zeros(1, bool), future, np.ones(1, bool), np.array([90]), np.zeros(1, bool))
    assert rows[0]['s_m'] is None and rows[0]['censoring'] == 'junction_carried_position'


def test_parallel_junction_rows_are_disambiguated_only_by_causal_last_observed_lane():
    kin = np.zeros((2, 2, 7), dtype=np.float32)
    kin[..., 0], kin[..., 4] = 75.44, 31.
    kin[0, 0, 1], kin[0, 1, 2] = 1., 1.
    vids = np.array([[100, 101], [100, 101]])
    carry = causal_lane_carry(kin, np.ones((2, 2), bool), vids)
    assert carry.tolist() == [0, 1]
    start = dict(state=dict(present=torch.ones(1, 2, dtype=torch.bool),
        junction=torch.ones(1, 2, dtype=torch.bool), lane=torch.tensor([[0, 1]]),
        s_m=torch.full((1, 2), 75.44), v_mps=torch.full((1, 2), 31.),
        is_cav=torch.zeros(1, 2, dtype=torch.bool)), vehicle_ids=['left', 'right'])
    future = np.zeros((2, 7), dtype=np.float32)
    future[:, 0], future[:, 1] = [200., 300.], 1.
    rows = match_targets_with_carry(start, np.array([0, 1]), kin[-1], np.ones(2, bool),
        vids[-1], np.zeros(2, bool), future, np.ones(2, bool), np.array([101, 100]),
        np.zeros(2, bool), carry)
    assert [r['s_m'] for r in rows] == [300., 200.]


def test_causal_lane_carry_does_not_inherit_a_recycled_slots_former_lane():
    kin = np.zeros((2, 1, 7), dtype=np.float32)
    kin[0, 0, 2] = 1.
    carry = causal_lane_carry(kin, np.ones((2, 1), bool), np.array([[99], [100]]))
    assert carry.tolist() == [-1]
