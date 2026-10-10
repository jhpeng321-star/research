"""A5 K_max format rule (amendment 5 section 4, "K = 192").

CODE PREP for the K=192 option: the slot width K_max is a FORMAT parameter
(v2 = 128 frozen and byte-reproducible, v3 = 192), everywhere in the
featurizer.  Zero SUMO; synthetic physics runs only; nothing outside tmp is
written and no real data file is opened.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from features import constants as C
from features import featurizer as F
from features import writer as W
from features.idea_a import idea_a_channels_from_shard, load_nominal_params
from synth_features import plan_row, row, veh, write_run

MAIN_LEFT = "1191554636_1"
MAIN_RIGHT = "1191554636_0"
ACCEL = "11553048#1_0"

# ------------------------------------------------- frozen v2 fixture (HEAD) --
# sha256 of every path-independent output of the synthetic ``wfix`` run below,
# produced by the PRE-CHANGE (git HEAD) featurizer+writer on branch
# zcode/kfix192 @ 148439f.  The parameterised default (v2) path must reproduce
# them exactly (byte-reproducibility of the frozen v2 format).
V2_FIXTURE_SHAS = {
    "veh_step": "de120c68a033079cd5b8faba8854180ddd06233777e64d5afcb0c12565400c8a",
    "veh_mask": "adccea3ef82ee8381850ccb4900c2b57cdd929b37d0a6584de0d1d6845c24ceb",
    "slot_index": "c4059e148c3da1af2fcca2228acdbf17c3e51b446afeb6939e7cdfec0f27676d",
    "plan": "d5541844f130f66e565b9ed14c9710794e40a3b8adbb78fe9d436e2d5f295f35",
    "decision_t": "f09fc9d5c34e0806bb122a86ce7284b0857741a79b7f6ad1aef9d3ce71958da1",
    "wall": "cbad7d0c88f457c82e477bc878f882be6d13f74a839912cc5ac229592d2bdda3",
    "wall_mask": "ee3160d4bc5d95480954274d1bc08eeca4fde40c8ee4fc3eb44543e3e45d5a2a",
    "counter": "20f2d83c3615099e5113dc52792503414a42a09a97da2904e635af32ee30f774",
    "accel_next": "fe8e3783c27dacedd56bec00c0eee6715bf0368e2c897e4241cc5beec1962bfe",
    "target_mask": "32a439d5e92a88bbeb60319f1794c33b83a91aaff509062179149ec475de9d89",
    "lane_next": "2c8d8af49bd6c9ff9f9c7070b7915babf0230d89a7c79af538b49caa78ae4635",
    "arrivals": "cb31e19e1a4353654c6cca20f4678bbf5eebc748b6e06f831ba8de7bc2f5bba0",
    "slot_vid.json": "9dbbd7ad085cf1f678d38c80375694ad0fbf2292b77274dd1969f6e40fd43928",
}


def _sha(path) -> str:
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def _v2_fixture_run(root):
    """The exact synthetic run the frozen fixture was built from."""
    ml, mr, ac = MAIN_LEFT, MAIN_RIGHT, ACCEL
    main = {"main0": veh(mr, 10.0, 30.0), "main1": veh(mr, 42.5, 28.0, -0.5),
            "left0": veh(ml, 20.0, 31.0), "accel0": veh(ac, 380.0, 22.0, 1.5)}
    tick1 = {"main0": veh(mr, 25.0, 30.0), "main1": veh(mr, 56.0, 27.5, -0.5),
             "left0": veh(ml, 35.5, 31.0), "accel0": veh(ac, 391.0, 23.5, 1.0)}
    tick2 = {"main0": veh(mr, 40.0, 30.0), "left0": veh(ml, 51.0, 31.0),
             "accel0": veh(ac, 402.5, 24.0, 0.5)}
    rows = [row(0.5, main, departed=["main0", "main1", "left0", "accel0"],
                cav_ids=["main0"]),
            row(1.0, tick1, cav_ids=["main0"]),
            row(1.5, tick2, departed=["main1"], cav_ids=["main0"])]
    return write_run(root, rows)


def _dense_run(root, n_vehicles: int, *, t0: float = 0.0):
    """A synthetic run with ``n_vehicles`` concurrent vehicles per tick."""
    lanes = (MAIN_RIGHT, MAIN_LEFT, ACCEL)
    vehicles = {}
    for i in range(n_vehicles):
        lane_id = lanes[i % 3]
        pos = 20.0 + 7.0 * i if lane_id != ACCEL else 381.0 + 0.4 * i
        vehicles[f"v{i}"] = veh(lane_id, pos, 25.0 + (i % 7) * 0.5,
                                0.1 * (i % 3 - 1))
    rows = [row(t0 + 0.5, vehicles, departed=sorted(vehicles),
                cav_ids=["v0", "v1"]),
            row(t0 + 1.0, vehicles, cav_ids=["v0", "v1"]),
            row(t0 + 1.5, vehicles, cav_ids=["v0", "v1"])]
    return write_run(root, rows)


def _build(root, **kw):
    return F.build_shard("chunk4", "wfix", "base", root, split="TRAIN",
                         plan_row=plan_row(0.0), outcomes_root=None, **kw)


# ------------------------------------------------------------ format logic --
def test_format_constants_and_helpers():
    assert (C.K_MAX_V2, C.K_MAX_V3) == (128, 192)
    assert C.FORMAT_K_MAX == {"v2": 128, "v3": 192}
    assert C.k_max_of_format("v3") == 192
    assert C.format_of_k_max(192) == "v3"
    assert C.format_of_k_max(128) == "v2"
    assert C.resolve_k_max() == ("v2", 128)
    assert C.resolve_k_max(format="v3") == ("v3", 192)
    assert C.resolve_k_max(k_max=192) == ("v3", 192)
    with pytest.raises(ValueError):
        C.k_max_of_format("v4")
    with pytest.raises(ValueError):
        C.format_of_k_max(64)
    with pytest.raises(ValueError):
        C.resolve_k_max(format="v3", k_max=128)


def test_v2_default_is_frozen_128(tmp_path):
    sh = _build(_v2_fixture_run(tmp_path / "run"))
    assert sh["veh_step"].shape[1] == 128
    assert sh["meta"]["n_dropped_slot_steps"] == 0
    assert "format" not in sh["meta"] and "k_max" not in sh["meta"]


def test_v3_featurizes_192_concurrent_vehicles(tmp_path):
    sh = _build(_dense_run(tmp_path / "run", 192), format="v3")
    n_ticks = sh["veh_step"].shape[0]
    assert sh["veh_step"].shape[1] == 192
    assert sh["veh_mask"].sum() == 192 * (n_ticks - 1)      # first tick empty
    assert sh["meta"]["n_dropped_slot_steps"] == 0          # nothing dropped
    assert sh["meta"]["format"] == "v3" and sh["meta"]["k_max"] == 192


def test_v2_still_drops_beyond_128_concurrent(tmp_path):
    sh = _build(_dense_run(tmp_path / "run", 192))
    n_ticks = sh["veh_step"].shape[0]
    assert sh["veh_step"].shape[1] == 128
    assert sh["meta"]["n_dropped_slot_steps"] == 64 * (n_ticks - 1)
    assert "format" not in sh["meta"] and "k_max" not in sh["meta"]


def test_v2_fixture_is_still_byte_identical(tmp_path):
    """The frozen v2 (K=128) outputs of the HEAD featurizer are unchanged."""
    sh = _build(_v2_fixture_run(tmp_path / "run"))
    data_root = tmp_path / "data"
    rec = W.write_shard(sh, data_root)
    assert rec["status"] == "written"
    dirp = W.shard_dir(data_root, "chunk4", "wfix", "base")
    for name, want in V2_FIXTURE_SHAS.items():
        fname = name if name.endswith(".json") else f"{name}.npy"
        assert _sha(dirp / fname) == want, name
    meta = json.loads((dirp / "meta.json").read_text(encoding="utf-8"))
    meta.pop("input_physics")
    assert meta["schema"] == C.FORMAT_SCHEMA_V2
    assert "format" not in meta and "k_max" not in meta
    assert meta["shapes"]["veh_step"] == [4, 128, 16]
    assert meta["slot_vid"] == ["main0", "main1", "left0", "accel0"]


def test_v3_shard_self_describes_in_meta_and_writer_schema(tmp_path):
    sh = _build(_dense_run(tmp_path / "run", 192), format="v3")
    data_root = tmp_path / "data"
    W.write_shard(sh, data_root)
    dirp = W.shard_dir(data_root, "chunk4", "wfix", "base")
    meta = json.loads((dirp / "meta.json").read_text(encoding="utf-8"))
    assert meta["schema"] == C.FORMAT_SCHEMA_V3
    assert meta["format"] == "v3" and meta["k_max"] == 192
    assert meta["shapes"]["veh_step"] == [4, 192, 16]


def test_expected_bytes_scale_with_format_k():
    k128 = W.expected_bytes(1000, 128, k_max=128)
    k192 = W.expected_bytes(1000, 192, k_max=192)
    assert k192 > k128
    # K-dependent share only: plan + slot arrays grow by 192/128, the rest not
    assert 1.4 < k192 / k128 < 1.6


def test_unregistered_formats_are_refused(tmp_path):
    root = _v2_fixture_run(tmp_path / "run")
    with pytest.raises(ValueError):
        _build(root, k_max=64)
    with pytest.raises(ValueError):
        _build(root, format="v4")
    with pytest.raises(ValueError):
        _build(root, format="v3", k_max=128)


# ------------------------------------------------- Idea A channels at K=192 --
def _k192_shard_dict():
    n, k = 6, 192
    veh_arr = np.zeros((n, k, 16), np.float16)
    mask = np.zeros((n, k), np.uint8)
    wall = np.zeros((n, 2, 11), np.float16)
    wall_mask = np.zeros((n, 2), np.uint8)
    rng = np.random.default_rng(20261001)
    for j in range(n):
        for s in range(k):
            li = s % 3
            pos = (rng.uniform(300.0, 2700.0) if li != 2
                   else rng.uniform(360.0, 460.0))
            veh_arr[j, s, 0] = pos
            veh_arr[j, s, 1 + li] = 1.0
            veh_arr[j, s, 4] = rng.uniform(3.0, 33.0)
            veh_arr[j, s, 5] = rng.uniform(-1.0, 1.0)
            veh_arr[j, s, 6] = 5.0
            veh_arr[j, s, 15] = 1.0
            mask[j, s] = 1
    wall[:, 0, 0] = 1.0
    wall_mask[:, 0] = 1
    return dict(veh_step=veh_arr, veh_mask=mask, wall=wall, wall_mask=wall_mask)


def test_idea_a_channels_fast_identical_to_slow_at_k192(params):
    shard = _k192_shard_dict()
    assert shard["veh_mask"].shape[1] == 192
    ref = idea_a_channels_from_shard(shard, params=params, fast=False)
    fast = idea_a_channels_from_shard(shard, params=params, fast=True)
    for name in ("veh_kin", "hdv_base", "cav_plan_ch", "event_tok",
                 "event_mask"):
        assert ref[name].shape == fast[name].shape
        assert np.array_equal(ref[name], fast[name]), name


@pytest.fixture(scope="module")
def params():
    return load_nominal_params()


# ------------------------------------------------------------------- CLI ----
def test_cli_accepts_format_flag():
    from features.cli import build_parser

    argv = ["featurize", "--chunk", "chunk4", "--data-root", "d",
            "--format", "v3"]
    args = build_parser().parse_args(argv)
    assert args.format == "v3"
    dflt = build_parser().parse_args(
        ["featurize", "--chunk", "chunk4", "--data-root", "d"])
    assert dflt.format == C.DEFAULT_FORMAT == "v2"
    est = build_parser().parse_args(
        ["estimate", "--chunk", "chunk4", "--format", "v3"])
    assert est.format == "v3"
