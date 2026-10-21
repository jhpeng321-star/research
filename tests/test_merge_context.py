"""Hand-built scene tests for :mod:`wms.model.merge_context` (section A prep)."""
from __future__ import annotations

import math

import torch

from wms.model.merge_context import (
    CONTEXT_FEATURES,
    EMA_TAU_S,
    MAX_NEIGHBOUR_GAP_M,
    MERGE_ZONE_END_M,
    RAMP_LANE,
    TARGET_LANE,
    MergeContextTracker,
    recorded_context,
)

DT = 0.5


def _scene(cars: list[dict | None], k: int | None = None):
    """Build ``[1, K]`` step tensors from a list of per-slot dicts (None = empty)."""
    if k is None:
        k = len(cars)
    assert len(cars) <= k
    rows = list(cars) + [None] * (k - len(cars))
    s, lane, v, a, present, is_cav, length = [], [], [], [], [], [], []
    for car in rows:
        if car is None:
            s.append(0.0)
            lane.append(-1)
            v.append(0.0)
            a.append(0.0)
            present.append(False)
            is_cav.append(False)
            length.append(0.0)
        else:
            s.append(float(car.get("s", 0.0)))
            lane.append(int(car.get("lane", -1)))
            v.append(float(car.get("v", 0.0)))
            a.append(float(car.get("a", 0.0)))
            present.append(bool(car.get("present", True)))
            is_cav.append(bool(car.get("is_cav", False)))
            length.append(float(car.get("length", 4.5)))
    return {
        "s": torch.tensor([s], dtype=torch.float32),
        "lane": torch.tensor([lane], dtype=torch.long),
        "v": torch.tensor([v], dtype=torch.float32),
        "a": torch.tensor([a], dtype=torch.float32),
        "present": torch.tensor([present], dtype=torch.bool),
        "is_cav": torch.tensor([is_cav], dtype=torch.bool),
        "length": torch.tensor([length], dtype=torch.float32),
    }


def _step(tracker: MergeContextTracker, cars: list[dict | None]):
    scene = _scene(cars)
    return tracker.step(scene["s"], scene["lane"], scene["v"], scene["a"],
                        scene["present"], scene["is_cav"], scene["length"])


def test_features_and_masks_present():
    expected = (
        "n_ramp_ahead", "ramp_lead_gap_m", "ramp_lead_dv_mps", "dist_to_lane_end_m",
        "wait_s", "t_since_ramp_insertion_s", "t_since_lane1_passage_s",
        "last_service_was_ramp", "foll_speed_mps", "foll_accel_now",
        "foll_accel_ema1_s", "foll_accel_ema3_s", "foll_dv_mps", "n_ramp_same_gap",
    )
    assert CONTEXT_FEATURES == expected
    tracker = MergeContextTracker(1, 1)
    out = _step(tracker, [{"s": 20.0, "lane": RAMP_LANE, "v": 10.0}])
    for name in CONTEXT_FEATURES:
        assert name in out
    for name in ("ramp_lead_gap_m", "ramp_lead_dv_mps", "foll_speed_mps",
                 "foll_accel_now", "foll_accel_ema1_s", "foll_accel_ema3_s",
                 "t_since_ramp_insertion_s", "last_service_was_ramp"):
        assert out[f"{name}_valid"].dtype == torch.bool
        assert out[f"{name}_valid"].shape == (1, 1)


def test_ramp_queue_counts_and_lead_gaps():
    cars = [
        {"s": 30.0, "lane": RAMP_LANE, "v": 10.0, "length": 4.5},  # front
        {"s": 20.0, "lane": RAMP_LANE, "v": 10.0, "length": 4.5},  # middle
        {"s": 10.0, "lane": RAMP_LANE, "v": 10.0, "length": 4.5},  # rear
    ]
    out = _step(MergeContextTracker(1, 3), cars)
    assert out["n_ramp_ahead"].tolist() == [[0.0, 1.0, 2.0]]
    gaps = out["ramp_lead_gap_m"][0]
    assert out["ramp_lead_gap_m_valid"][0].tolist() == [False, True, True]
    assert gaps[0].item() == 0.0
    assert torch.allclose(gaps[1:], torch.tensor([5.5, 5.5]))
    assert out["ramp_lead_dv_mps_valid"][0].tolist() == [False, True, True]
    assert torch.allclose(out["ramp_lead_dv_mps"], torch.zeros(1, 3))
    assert torch.allclose(out["dist_to_lane_end_m"][0],
                          torch.tensor([MERGE_ZONE_END_M - 30.0,
                                        MERGE_ZONE_END_M - 20.0,
                                        MERGE_ZONE_END_M - 10.0]))


def test_follower_ema_on_known_accel_series():
    tracker = MergeContextTracker(1, 2)
    series = [1.0, 2.0, 3.0, 4.0, 5.0]
    ema1, ema3, now = [], [], []
    for accel in series:
        out = _step(tracker, [
            {"s": 20.0, "lane": RAMP_LANE, "v": 10.0},
            {"s": 15.0, "lane": TARGET_LANE, "v": 12.0, "a": accel},
        ])
        assert out["foll_accel_now_valid"][0, 0].item()
        assert out["foll_accel_now"][0, 0].item() == accel
        assert out["foll_speed_mps"][0, 0].item() == 12.0
        assert out["foll_dv_mps"][0, 0].item() == 2.0
        ema1.append(out["foll_accel_ema1_s"][0, 0].item())
        ema3.append(out["foll_accel_ema3_s"][0, 0].item())
        now.append(out["foll_accel_now"][0, 0].item())

    a1 = 1.0 - math.exp(-DT / EMA_TAU_S[0])
    a3 = 1.0 - math.exp(-DT / EMA_TAU_S[1])
    want1, want3 = series[0], series[0]
    expected1, expected3 = [], []
    for i, accel in enumerate(series):
        if i:
            want1 = want1 + (accel - want1) * a1
            want3 = want3 + (accel - want3) * a3
        expected1.append(want1)
        expected3.append(want3)
    assert torch.allclose(torch.tensor(ema1), torch.tensor(expected1), atol=1e-6)
    assert torch.allclose(torch.tensor(ema3), torch.tensor(expected3), atol=1e-6)
    assert now == series
    # the new follower's first EMA observation is its instantaneous acceleration
    assert ema1[0] == series[0] and ema3[0] == series[0]


def test_wait_s_across_entry():
    tracker = MergeContextTracker(1, 1)
    values, valid = [], []
    for pos in (-5.0, 0.0, 2.0, 4.0):
        out = _step(tracker, [{"s": pos, "lane": RAMP_LANE, "v": 5.0}])
        values.append(out["wait_s"][0, 0].item())
        valid.append(out["wait_s_valid"][0, 0].item())
    assert torch.allclose(torch.tensor(values), torch.tensor([0.0, 0.0, 0.5, 1.0]))
    assert valid == [False, True, True, True]


def test_zipper_timers_and_last_service():
    tracker = MergeContextTracker(1, 2)
    # world has a lane-1 car to the left, independent of the ramp/merge car
    left = {"s": 20.0, "lane": TARGET_LANE, "v": 10.0}
    out = _step(tracker, [{"s": 60.0, "lane": RAMP_LANE, "v": 10.0}, left])
    assert not out["t_since_ramp_insertion_s_valid"][0, 0].item()
    assert not out["last_service_was_ramp_valid"][0, 0].item()

    # ramp car inserts into lane 1 -> ramp service, timer zero
    out = _step(tracker, [{"s": 65.0, "lane": TARGET_LANE, "v": 10.0}, left])
    assert out["t_since_ramp_insertion_s_valid"][0, 0].item()
    assert out["t_since_ramp_insertion_s"][0, 0].item() == 0.0
    assert out["last_service_was_ramp_valid"][0, 0].item()
    assert out["last_service_was_ramp"][0, 0].item() == 1.0

    # one step later the insertion timer has advanced
    out = _step(tracker, [{"s": 70.0, "lane": TARGET_LANE, "v": 10.0}, left])
    assert out["t_since_ramp_insertion_s"][0, 0].item() == DT
    assert out["last_service_was_ramp"][0, 0].item() == 1.0

    # a lane-1 car crosses the merge point -> lane-1 service, timer zero
    out = _step(tracker, [{"s": 80.0, "lane": TARGET_LANE, "v": 10.0}, left])
    assert out["t_since_lane1_passage_s_valid"][0, 0].item()
    assert out["t_since_lane1_passage_s"][0, 0].item() == 0.0
    assert out["last_service_was_ramp"][0, 0].item() == 0.0
    assert out["t_since_ramp_insertion_s"][0, 0].item() == 2.0 * DT + 0.0


def test_n_ramp_same_gap_two_ramp_cars():
    cars = [
        {"s": 30.0, "lane": RAMP_LANE, "v": 8.0},
        {"s": 25.0, "lane": RAMP_LANE, "v": 8.0},
        {"s": 60.0, "lane": TARGET_LANE, "v": 12.0},  # shared leader
        {"s": 10.0, "lane": TARGET_LANE, "v": 12.0},  # shared follower
    ]
    out = _step(MergeContextTracker(1, 4), cars)
    same = out["n_ramp_same_gap"][0]
    assert same[0].item() == 1.0
    assert same[1].item() == 1.0
    assert same[2].item() == 0.0 and same[3].item() == 0.0
    assert out["n_ramp_ahead"][0].tolist() == [0.0, 1.0, 0.0, 0.0]
    assert torch.allclose(out["ramp_lead_gap_m"][0, 1], torch.tensor(0.5))


def test_missing_neighbours_are_invalid():
    out = _step(MergeContextTracker(1, 1),
                [{"s": 20.0, "lane": RAMP_LANE, "v": 10.0}])
    assert out["n_ramp_ahead"].item() == 0.0
    assert out["n_ramp_same_gap"].item() == 0.0
    for name in ("ramp_lead_gap_m", "ramp_lead_dv_mps", "foll_speed_mps",
                 "foll_accel_now", "foll_accel_ema1_s", "foll_accel_ema3_s",
                 "foll_dv_mps", "t_since_ramp_insertion_s",
                 "t_since_lane1_passage_s", "last_service_was_ramp"):
        assert not out[f"{name}_valid"].item(), name
        assert out[name].item() == 0.0, name


def _random_sequence(b: int, n: int, k: int, dtype: torch.dtype, seed: int = 7):
    g = torch.Generator().manual_seed(seed)
    s = torch.randn(b, n, k, generator=g, dtype=dtype) * 20.0
    lane_idx = torch.randint(0, 3, (b, n, k), generator=g)
    onehot = torch.nn.functional.one_hot(lane_idx, 3).to(dtype)
    present = torch.rand(b, n, k, generator=g) > 0.25
    # absent slots carry garbage, as recorded states may
    s = torch.where(present, s, torch.full_like(s, 999.0))
    kin = torch.zeros(b, n, k, 7, dtype=dtype)
    kin[..., 0] = s
    kin[..., 1:4] = onehot
    kin[..., 4] = torch.rand(b, n, k, generator=g, dtype=dtype) * 20.0
    kin[..., 5] = torch.randn(b, n, k, generator=g, dtype=dtype)
    kin[..., 6] = torch.full((b, n, k), 4.5, dtype=dtype)
    is_cav = torch.rand(b, n, k, generator=g) > 0.7
    return kin, present, is_cav


def test_recorded_context_equals_stepping_tracker():
    for dtype in (torch.float32, torch.float64):
        b, n, k = 2, 7, 4
        kin, present, is_cav = _random_sequence(b, n, k, dtype)
        got = recorded_context(kin, present, is_cav, dt=DT)
        onehot = kin[..., 1:4]
        lanes = onehot.argmax(-1)
        lanes = torch.where(onehot.sum(-1) > 0, lanes, torch.full_like(lanes, -1))
        tracker = MergeContextTracker(b, k, dt=DT, device=kin.device, dtype=dtype)
        want: dict[str, list[torch.Tensor]] = {}
        for t in range(n):
            out = tracker.step(kin[:, t, :, 0], lanes[:, t], kin[:, t, :, 4],
                               kin[:, t, :, 5], present[:, t], is_cav[:, t],
                               kin[:, t, :, 6])
            for name, value in out.items():
                want.setdefault(name, []).append(value)
        assert set(got) == set(want)
        for name, values in want.items():
            expected = torch.stack(values, 1)
            assert got[name].dtype == expected.dtype, name
            assert got[name].shape == (b, n, k), name
            assert torch.allclose(got[name], expected), name


def test_slot_reuse_resets_per_car_state():
    tracker = MergeContextTracker(1, 2)
    ramp = {"s": 10.0, "lane": RAMP_LANE, "v": 6.0}
    follower = {"s": 5.0, "lane": TARGET_LANE, "v": 6.0, "a": 0.0}
    _step(tracker, [ramp, follower])
    out = _step(tracker, [ramp, follower])
    assert out["wait_s"][0, 0].item() == DT

    tracker.reset_slots(torch.tensor([[True, False]]))
    # same tensors, but slot 0 now holds a different car (and a different follower)
    new_ramp = {"s": 10.0, "lane": RAMP_LANE, "v": 6.0}
    new_follower = {"s": 5.0, "lane": TARGET_LANE, "v": 6.0, "a": 7.0}
    out = _step(tracker, [new_ramp, new_follower])
    assert out["wait_s"][0, 0].item() == 0.0        # fresh entry, not carried over
    assert out["wait_s_valid"][0, 0].item()
    assert out["foll_accel_ema1_s"][0, 0].item() == 7.0  # EMA restarted
    assert out["foll_accel_ema3_s"][0, 0].item() == 7.0


def test_batch_vectorisation_matches_single_rows():
    rows = [
        [
            {"s": 30.0, "lane": RAMP_LANE, "v": 9.0},
            {"s": 20.0, "lane": RAMP_LANE, "v": 9.0},
            {"s": 15.0, "lane": TARGET_LANE, "v": 11.0, "a": 0.5},
        ],
        [
            {"s": 5.0, "lane": TARGET_LANE, "v": 14.0, "a": -0.5},
            {"s": 12.0, "lane": RAMP_LANE, "v": 7.0},
            {"s": 40.0, "lane": TARGET_LANE, "v": 13.0, "a": 0.2},
        ],
    ]
    k = 3
    batch = {key: torch.stack([_scene(r, k)[key].squeeze(0) for r in rows], 0)
             for key in ("s", "lane", "v", "a", "present", "is_cav", "length")}
    tracker_b = MergeContextTracker(2, k)
    single = [MergeContextTracker(1, k) for _ in rows]
    for _ in range(3):
        out_b = tracker_b.step(batch["s"], batch["lane"], batch["v"], batch["a"],
                               batch["present"], batch["is_cav"], batch["length"])
        for b, row in enumerate(rows):
            one = {key: value[b:b + 1] for key, value in batch.items()}
            out_1 = single[b].step(one["s"], one["lane"], one["v"], one["a"],
                                   one["present"], one["is_cav"], one["length"])
            for name, value in out_1.items():
                assert torch.allclose(out_b[name][b:b + 1], value), (b, name)
        # next position of each row
        batch["s"] = batch["s"].clone()
        batch["s"][0, 0] += 1.0
        batch["s"][1, 1] += 1.5


def test_k_vectorisation_padding_is_inert():
    cars = [
        {"s": 30.0, "lane": RAMP_LANE, "v": 9.0},
        {"s": 20.0, "lane": RAMP_LANE, "v": 9.0},
        {"s": 15.0, "lane": TARGET_LANE, "v": 11.0, "a": 0.5},
    ]
    padded = list(cars) + [None, None, None]
    tracker_k = MergeContextTracker(1, 3)
    tracker_pad = MergeContextTracker(1, 6)
    for _ in range(2):
        out_k = _step(tracker_k, cars)
        out_pad = _step(tracker_pad, padded)
        for name, value in out_k.items():
            assert torch.allclose(out_pad[name][:, :3], value), name


def test_state_dict_roundtrip():
    tracker = MergeContextTracker(1, 2, dtype=torch.float64)
    cars = [{"s": 10.0, "lane": RAMP_LANE, "v": 6.0},
            {"s": 5.0, "lane": TARGET_LANE, "v": 6.0, "a": 1.0}]
    _step(tracker, cars)
    saved = tracker.state_dict()
    clone = MergeContextTracker(1, 2, dtype=torch.float64)
    clone.load_state_dict(saved)
    out_a = _step(tracker, cars)
    out_b = _step(clone, cars)
    for name in out_a:
        assert torch.allclose(out_a[name], out_b[name]), name


def test_detach_breaks_state_graph():
    tracker = MergeContextTracker(1, 2)
    scene = _scene([
        {"s": 20.0, "lane": RAMP_LANE, "v": 10.0},
        {"s": 15.0, "lane": TARGET_LANE, "v": 12.0, "a": 1.5},
    ])
    scene["a"] = scene["a"].clone().requires_grad_(True)
    out = tracker.step(scene["s"], scene["lane"], scene["v"], scene["a"],
                       scene["present"], scene["is_cav"], scene["length"])
    out["foll_accel_ema1_s"].sum().backward()
    assert scene["a"].grad is not None
    assert tracker.foll_ema1.requires_grad
    tracker.detach_()
    assert not tracker.foll_ema1.requires_grad
    assert not tracker.foll_ema3.requires_grad
    assert all(not tensor.requires_grad for tensor in tracker.state_dict().values())


def test_float32_and_float64():
    for dtype in (torch.float32, torch.float64):
        tracker = MergeContextTracker(1, 2, dtype=dtype)
        scene = _scene([
            {"s": 20.0, "lane": RAMP_LANE, "v": 10.0},
            {"s": 15.0, "lane": TARGET_LANE, "v": 12.0, "a": 1.5},
        ])
        scene = {key: value.to(dtype) if value.is_floating_point() else value
                 for key, value in scene.items()}
        out = tracker.step(scene["s"], scene["lane"], scene["v"], scene["a"],
                           scene["present"], scene["is_cav"], scene["length"])
        for name in CONTEXT_FEATURES:
            assert out[name].dtype == dtype, (dtype, name)
        assert torch.isfinite(out["foll_accel_ema1_s"]).all()


def test_max_neighbour_range_is_geometry_constant():
    # a lane-1 follower just beyond the association range is not the gap partner
    assert MAX_NEIGHBOUR_GAP_M == 150.0
    tracker = MergeContextTracker(1, 2)
    out = _step(tracker, [
        {"s": 200.0, "lane": RAMP_LANE, "v": 10.0},
        {"s": 10.0, "lane": TARGET_LANE, "v": 10.0},
    ])
    assert not out["foll_speed_mps_valid"][0, 0].item()
