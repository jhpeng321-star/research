"""V2V intent-sharing observation block (tools/coopmwm/v2v_obs, c4/v2v-intent part 1).

Covers: fixed shapes, presence mask + zero padding, distance ordering, the flag-off
contract (zeros + byte-identical legacy observation against the pinned pristine
golden), last-command plumbing and the slot-aligned env block.  CPU only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tools.coopmwm.v2v_obs import (V2V_DIM, V2V_K, V2V_PARTNER_FEATURES, V2V_RANGE_M,
                                   V2VConfig, v2v_features, v2v_slot_block)

REPO = Path(__file__).resolve().parents[2]
GOLDEN = REPO / "tests" / "fixtures" / "v2v_golden.npz"


@pytest.fixture(autouse=True)
def small_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def ego_state():
    """Four other CAVs + ego + one HDV + one absent slot; distances 30/60/120/90 m."""
    return {
        "ego":  {"s_m": -100.0, "lane": 1, "v_mps": 10.0, "a_mps2": 0.0},
        "near": {"s_m": -70.0, "lane": 1, "v_mps": 8.0, "a_mps2": -1.5},
        "mid":  {"s_m": -160.0, "lane": 0, "v_mps": 11.0, "a_mps2": 0.5},
        "far":  {"s_m": -280.0, "lane": 1, "v_mps": 12.0, "a_mps2": 0.0},
        "rear": {"s_m": -10.0, "lane": 1, "v_mps": 9.0, "a_mps2": 1.0},
        "hdv":  {"s_m": -110.0, "lane": 1, "v_mps": 7.0, "a_mps2": 0.0, "is_cav": False},
        "gone": {"s_m": -400.0, "lane": 2, "v_mps": 6.0, "a_mps2": 0.0},
    }


def cfg(**kw):
    return V2VConfig(v2v=True, **kw)


def test_v2v_dim_and_layout_constants():
    assert V2V_K == 4 and len(V2V_PARTNER_FEATURES) == 6
    assert V2V_DIM == V2V_K * 6 == 24
    assert V2VConfig().v2v is False and V2VConfig().k == V2V_K
    from wms.control.contract import SENSOR_RANGE_M
    assert V2VConfig().range_m == V2V_RANGE_M == SENSOR_RANGE_M


def test_flag_off_returns_zeros_and_default_config():
    state = ego_state()
    out = v2v_features(state, "ego", {}, V2VConfig())
    assert out.shape == (V2V_DIM,) and out.dtype == np.float32
    assert not out.any()
    assert not v2v_slot_block(np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3),
                              np.ones(3, bool), np.ones(3, bool), ["a", "b", "c"],
                              {}, V2VConfig()).any()


def test_shapes_values_mask_and_zero_padding():
    out = v2v_features(ego_state(), "ego", {"near": -1.5}, cfg())
    assert out.shape == (V2V_DIM,)
    blocks = out.reshape(V2V_K, 6)
    # nearest first by |gap|: near (30 m), mid (60 m), rear (90 m)
    order = [tuple(np.round(blocks[i, :5], 6)) for i in range(V2V_DIM // 6)]
    assert order[0] == (30.0, 0.0, 8.0, -1.5, -1.5)          # near, gap + lane offset + v + a + cmd
    assert order[1] == (-60.0, -1.0, 11.0, 0.5, 0.0)         # mid
    assert order[2] == (90.0, 0.0, 9.0, 1.0, 0.0)            # rear
    # far is 180 m away: out of comm range; hdv excluded; only 3 partners -> 1 padded slot
    assert not blocks[3].any()
    assert blocks[3, 5] == 0.0
    assert [blocks[i, 5] for i in range(3)] == [1.0, 1.0, 1.0]


def test_ordering_by_distance_not_insertion():
    state = {
        "ego": {"s_m": 0.0, "lane": 1, "v_mps": 10.0, "a_mps2": 0.0},
        "first": {"s_m": 80.0, "lane": 1, "v_mps": 10.0, "a_mps2": 0.0},
        "second": {"s_m": 20.0, "lane": 1, "v_mps": 10.0, "a_mps2": 0.0},
    }
    blocks = v2v_features(state, "ego", {}, cfg(k=2)).reshape(2, 6)
    assert blocks[0, 0] == 20.0 and blocks[1, 0] == 80.0


def test_k_truncates_beyond_four_partners():
    state = {"ego": {"s_m": 0.0, "lane": 1, "v_mps": 10.0, "a_mps2": 0.0}}
    for i in range(6):
        state[f"p{i}"] = {"s_m": float(10 * (i + 1)), "lane": 1, "v_mps": 10.0, "a_mps2": 0.0}
    out = v2v_features(state, "ego", {}, cfg())     # default k=4
    gaps = out.reshape(V2V_K, 6)[:, 0]
    assert np.allclose(gaps, [10.0, 20.0, 30.0, 40.0])
    assert out.reshape(V2V_K, 6)[:, 5].tolist() == [1.0] * 4


def test_last_commands_record_mapping_and_missing():
    state = {
        "ego": {"s_m": 0.0, "lane": 1, "v_mps": 10.0, "a_mps2": 0.0},
        "a": {"s_m": 10.0, "lane": 1, "v_mps": 10.0, "a_mps2": 0.0},
        "b": {"s_m": 20.0, "lane": 1, "v_mps": 10.0, "a_mps2": 0.0},
        "c": {"s_m": 30.0, "lane": 1, "v_mps": 10.0, "a_mps2": 0.0},
    }
    last = {"a": 0.75, "b": {"issued_acceleration_mps2": -2.5, "owner": "direct"},
            "c": None}
    out = v2v_features(state, "ego", last, cfg(k=3)).reshape(3, 6)
    assert out[:, 4].tolist() == [0.75, -2.5, 0.0]
    assert v2v_features(state, "ego", None, cfg(k=3)).reshape(3, 6)[:, 4].tolist() == [0.0] * 3


def test_absent_ego_and_lane_defaults():
    out = v2v_features(ego_state(), "ghost", {}, cfg())
    assert not out.any()
    state = {"only": {"s_m": 5.0, "v_mps": 1.0, "a_mps2": 0.0}}
    out = v2v_features(state, "only", {}, cfg())
    assert not out.any()                            # no OTHER CAV within range


def test_slot_block_alignment_and_zero_rows():
    k = 6
    s = np.array([-100.0, -70.0, 0.0, -200.0, 5.0, 0.0], np.float64)
    lane = np.array([1, 1, 1, 0, 1, -1], np.int64)
    v = np.array([10.0, 8.0, 9.0, 11.0, 9.0, 0.0], np.float64)
    a = np.array([0.0, -1.5, 1.0, 0.5, 0.0, 0.0], np.float64)
    present = np.array([True, True, True, True, True, False])
    is_cav = np.array([True, True, True, True, False, False])
    ids = ["ego", "near", "rear", "mid", "hdv", None]
    block = v2v_slot_block(s, lane, v, a, present, is_cav, ids, {}, cfg())
    assert block.shape == (k, V2V_DIM) and block.dtype == np.float32
    per_cav = v2v_features({"ego": {"s_m": -100.0, "lane": 1, "v_mps": 10.0, "a_mps2": 0.0},
                            "near": {"s_m": -70.0, "lane": 1, "v_mps": 8.0, "a_mps2": -1.5},
                            "rear": {"s_m": 0.0, "lane": 1, "v_mps": 9.0, "a_mps2": 1.0},
                            "mid": {"s_m": -200.0, "lane": 0, "v_mps": 11.0, "a_mps2": 0.5}},
                           "ego", {}, cfg())
    assert np.array_equal(block[0], per_cav)
    assert not block[4].any() and not block[5].any()      # HDV + absent slots zeroed
    assert block[3].any()                                  # mid CAV row exists


def test_flag_off_state_is_byte_identical_against_pristine_golden():
    """Flag-off contract: no v2v key, and the legacy observation bytes are untouched."""
    golden = np.load(GOLDEN)
    state = {key.removeprefix("state_"): torch.tensor(value)
             for key, value in golden.items() if key.startswith("state_")}
    from wms.control.obs import build_observations
    obs = build_observations(state)
    assert torch.equal(obs.own, torch.tensor(golden["own"]))
    assert torch.equal(obs.reports, torch.tensor(golden["reports"]))
    assert torch.equal(obs.report_mask, torch.tensor(golden["report_mask"]))
    assert torch.equal(obs.cav_mask, torch.tensor(golden["cav_mask"]))
    # an extra numpy v2v key (flag on elsewhere) never leaks into the legacy observation
    augmented = {**state, "v2v": np.zeros((state["s_m"].shape[1], V2V_DIM), np.float32)}
    obs2 = build_observations(augmented)
    assert torch.equal(obs2.own, obs.own) and torch.equal(obs2.reports, obs.reports)
    assert torch.equal(obs2.report_mask, obs.report_mask) and torch.equal(obs2.cav_mask, obs.cav_mask)


def test_deploy_style_gather_survives_numpy_block():
    """deploy.py gathers every state key by slot index; on the [K, V2V_DIM] block the
    slot axis is 0, so consumers slice ``block[keep]`` -- assert that keeps rows aligned."""
    keep = np.array([0, 2])
    block = v2v_slot_block(np.arange(4.0), np.zeros(4), np.zeros(4), np.zeros(4),
                           np.ones(4, bool), np.ones(4, bool), ["a", "b", "c", "d"], {}, cfg())
    gathered = block[keep]
    assert gathered.shape == (2, V2V_DIM) and np.array_equal(gathered[0], block[0])


def test_config_validation():
    with pytest.raises(ValueError):
        V2VConfig(k=-1)
    with pytest.raises(ValueError):
        V2VConfig(range_m=0.0)
