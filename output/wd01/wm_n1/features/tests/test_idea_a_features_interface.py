"""P9 interface: kinematics-only view + new channel contract (implemented)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from features.idea_a import (CAV_PLAN_CH_F, HDV_BASE_F, HDV_BASE_FIELDS,
                             IDEA_A_CHANNELS, KIN_F, KIN_SOURCE_INDICES,
                             NOMINAL_PARAMS_LF_SHA256,
                             ROAD_END_REASON_RELABELLED,
                             ROAD_END_REASON_SOURCE, ROAD_END_RELABEL_FLAG,
                             IdeaAChannelSpecs, apply_road_end_relabel,
                             baseline_predictions, cav_plan_channels,
                             expected_bytes_idea_a, kinematics_view,
                             lf_sha256_bytes, load_nominal_params)


def test_kinematics_view_strips_cav_flag_and_actions():
    veh = torch.zeros(2, 4, 16)
    veh[..., 7] = 1.0                    # is_cav
    veh[..., 8:13] = 1.0                 # action one-hot
    kin = kinematics_view(veh)
    assert kin.shape == (2, 4, KIN_F) == (2, 4, 7)
    assert float(kin.abs().max()) == 0.0


def test_kinematics_view_schema_mapping():
    """veh_kin == exactly the 7 kinematics source columns; no CAV flag,
    no action/strategy channel, no wall-member/junction/present."""
    veh = torch.arange(2 * 3 * 16, dtype=torch.float32).reshape(2, 3, 16)
    kin = kinematics_view(veh)
    for j, src in enumerate(KIN_SOURCE_INDICES):
        assert torch.equal(kin[..., j], veh[..., src])
    assert 7 not in KIN_SOURCE_INDICES            # is_cav
    assert not set(range(8, 16)) & set(KIN_SOURCE_INDICES)
    # flipping every non-kinematics column cannot change the view
    veh2 = veh.clone()
    veh2[..., 7:16] = 42.0
    assert torch.equal(kinematics_view(veh2), kin)


def test_kinematics_view_numpy():
    veh = np.zeros(16, np.float16)
    veh[7] = 1.0
    veh[8] = 1.0
    kin = kinematics_view(veh)
    assert isinstance(kin, np.ndarray) and kin.shape == (KIN_F,)
    assert float(np.abs(kin).max()) == 0.0


def test_channel_names_and_flag():
    assert IDEA_A_CHANNELS == ("veh_kin", "hdv_base", "cav_plan_ch",
                               "event_tok", "event_mask")
    assert ROAD_END_RELABEL_FLAG == "amendment2_road_end_relabel"
    assert (ROAD_END_REASON_SOURCE, ROAD_END_REASON_RELABELLED) == \
        ("member_left_network", "road_end")


def test_baseline_predictions_stack():
    z = torch.zeros(2, 4)
    out = baseline_predictions(z + 1, z + 2, z + 3, z + 4,
                               (z > 0).int(), torch.zeros(2, 4, dtype=torch.bool))
    assert out.shape == (2, 4, HDV_BASE_F) == (2, 4, 6)
    assert out.dtype == torch.float32
    assert torch.equal(out[..., 0], torch.ones(2, 4))
    assert torch.equal(out[..., 4], torch.zeros(2, 4))   # bool/int -> 0/1
    arr = baseline_predictions(np.ones(3), np.ones(3), np.ones(3), np.ones(3),
                               np.ones(3, bool), np.zeros(3, np.int64))
    assert isinstance(arr, np.ndarray) and arr.shape == (3, 6)
    assert list(HDV_BASE_FIELDS) == ["idm_accel", "mobil_gain_left",
                                     "mobil_gain_right", "merge_urgency",
                                     "lc_accept_left", "lc_accept_right"]


def test_cav_plan_channels_encoding():
    out = cav_plan_channels(torch.full((2,), 40.0), torch.full((2,), 3.0),
                            torch.full((2,), 40.0), torch.tensor([0, -1]),
                            torch.full((2,), 600.0), torch.tensor([2, 2]))
    assert out.shape == (2, CAV_PLAN_CH_F) == (2, 8)
    assert torch.equal(out[:, 0], torch.ones(2))          # 40/40
    assert torch.equal(out[:, 1], torch.ones(2))          # 3/3
    assert torch.equal(out[:, 2], torch.ones(2))          # 40/40
    assert torch.equal(out[:, 3], torch.tensor([0.0, -0.25]))  # id/4, -1 none
    assert torch.equal(out[:, 4], torch.ones(2))          # 600/600
    assert torch.equal(out[:, 5:8], torch.tensor([[0., 0., 1.]] * 2))


def test_cav_plan_channels_dense_lane_target_and_numpy():
    one_hot = torch.tensor([[1.0, 0.0, 0.0]])
    out = cav_plan_channels(torch.zeros(1), torch.zeros(1), torch.zeros(1),
                            torch.zeros(1), torch.zeros(1), one_hot)
    assert torch.equal(out[:, 5:8], one_hot)
    arr = cav_plan_channels(np.zeros(1), np.zeros(1), np.zeros(1),
                            np.zeros(1), np.zeros(1), np.full(1, -1, np.int64))
    assert isinstance(arr, np.ndarray)
    assert np.allclose(arr.sum(-1), 0.0)                  # lane -1 -> zeros


def test_apply_road_end_relabel_rewrites_only_member_left_network():
    tokens = [
        {"phase": "aborted", "end_reason": "member_left_network"},
        {"phase": "aborted", "abort_reason": "safety_gap"},
        {"phase": "released"},
    ]
    labels = pd.DataFrame(
        {"end_reason": ["member_left_network", "member_left_network",
                        "other"]})
    meta = {"run": "j4"}
    out = apply_road_end_relabel(
        {"wall_tokens": tokens, "labels": labels, "meta": meta}, enabled=True)
    assert [t.get("end_reason") for t in out["wall_tokens"]] == \
        ["road_end", None, None]
    assert out["wall_tokens"][1]["abort_reason"] == "safety_gap"
    got = out["labels"]["end_reason"].tolist()
    assert got == ["road_end", "road_end", "other"]
    assert out["meta"][ROAD_END_RELABEL_FLAG] is True
    assert out["meta"]["road_end_relabelled_n"] == 3
    # inputs untouched (receipts / original records keep their labels)
    assert tokens[0]["end_reason"] == "member_left_network"
    assert labels["end_reason"].tolist()[0] == "member_left_network"
    assert meta == {"run": "j4"}


def test_apply_road_end_relabel_disabled_is_noop():
    rec = {"wall_tokens": [{"end_reason": "member_left_network"}],
           "meta": {}}
    out = apply_road_end_relabel(rec, enabled=False)
    assert out["wall_tokens"][0]["end_reason"] == "member_left_network"
    assert ROAD_END_RELABEL_FLAG not in out["meta"]
    out2 = apply_road_end_relabel({}, enabled=True)
    assert out2["meta"][ROAD_END_RELABEL_FLAG] is True
    assert out2["meta"]["road_end_relabelled_n"] == 0


def test_expected_bytes_idea_a_arithmetic():
    n, k = 11, 128
    specs = IdeaAChannelSpecs()
    shapes = specs.array_shapes(n, k)
    assert shapes["veh_kin"] == (n, k, 7)
    assert shapes["event_mask"] == (n, 8)
    dtypes = {"veh_kin": 2, "hdv_base": 2, "cav_plan_ch": 2, "event_tok": 2,
              "event_mask": 1}
    payload = sum(int(np.prod(s)) * dtypes[name]
                  for name, s in shapes.items())
    # npy headers are included, so strictly more than the raw payload
    assert expected_bytes_idea_a(n, k) > payload
    assert expected_bytes_idea_a(n, k) - payload < 2048
    assert expected_bytes_idea_a(n, k, base_bytes=123) == \
        expected_bytes_idea_a(n, k) + 123
    assert expected_bytes_idea_a(0, k) > 0          # headers only
    assert expected_bytes_idea_a(2 * n, k) > expected_bytes_idea_a(n, k)


def test_nominal_params_provenance():
    params = load_nominal_params()
    assert params["param_set_name"] == "parta_v2g_nominal_treiber"
    raw = open(_params_path(), "rb").read()
    assert lf_sha256_bytes(raw) == NOMINAL_PARAMS_LF_SHA256


def _params_path():
    import sys

    from features.idea_a import _engine

    eng = _engine()
    return eng.PARAMS_PATH


@pytest.mark.parametrize("field", ["cooldown_s", "lc_min_speed_mps",
                                   "ego_gain_floor_mps2"])
def test_non_nominal_guards_refused(field):
    import copy

    from features.idea_a import NominalParamsError, scene_baseline_channels

    params = load_nominal_params()
    params = copy.deepcopy(params)
    params["mobil"][field] = 2.0
    with pytest.raises(NominalParamsError):
        scene_baseline_channels(np.zeros(4, np.int64), np.zeros(4),
                                np.zeros(4), np.full(4, 5.0),
                                np.zeros(4, bool), params=params)
