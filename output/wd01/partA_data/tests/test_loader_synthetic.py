"""Loader tests on a synthetic scene written in the dataset's own npz
schema: windows with births/exits, the 6 neighbour slots incl. the accel
lane adjacency, junction slots, gap cap, and the vType exclusion rule."""
import numpy as np
import pytest

import pdl as L
import loader as LD

# ---------------------------------------------------------------- scene ----
# ticks k = 0..20 (t = 60 + 0.5k).  lane 1 main_right, 0 main_left, 2 accel.
OFF0, OFF1, OFF2 = 2645.67, 2644.22, 378.66  # main_left, main_right, accel


def _rows_for(vid_idx, spec):
    """spec: list of (tick, lane, s, v, a, cav, inj)."""
    return dict(vid=[vid_idx] * len(spec), tick=[r[0] for r in spec],
                lane=[r[1] for r in spec], s=[r[2] for r in spec],
                v=[r[3] for r in spec], a=[r[4] for r in spec],
                cav=[r[5] for r in spec], inj=[r[6] for r in spec])


def make_scene(tmp_path, monkeypatch):
    rows = {k: [] for k in ("vid", "tick", "lane", "s", "v", "a", "cav",
                            "inj")}

    def add(vid_idx, spec):
        r = _rows_for(vid_idx, spec)
        for k in rows:
            rows[k].extend(r[k])

    # main_right ego at s=100..104 from k0..k20 (d = OFF1 - s = 2543.67 @k10)
    add(0, [(k, 1, 100.0 + 0.2 * k, 30.0, 0.0, 1, 0) for k in range(21)])
    # own leader on main_right at s=140 (38 m ahead in d @k10), exits k12
    add(1, [(k, 1, 140.0, 28.0, 0.0, 0, 0) for k in range(13)])
    # own follower on main_right at s=80 (22 m behind in d), CAV
    add(2, [(k, 1, 80.0, 31.0, 0.0, 1, 0) for k in range(21)])
    # main_left vehicle at s=95 (5.55 m behind ego in d)
    add(3, [(k, 0, 95.0, 30.0, 0.0, 0, 0) for k in range(21)])
    # accel vehicle at s=300 (d=78.66, far AHEAD of ego in d -> capped gap)
    add(4, [(k, 2, 300.0, 25.0, 0.0, 0, 0) for k in range(21)])
    # vehicle crossing pre->merge through the junction: lane 1 below the
    # boundary, junction row at the boundary s, then lane 1 on the merge
    add(5, [(9, 1, 2644.7, 25.0, 0.0, 0, 0),
            (10, -1, OFF1, 25.0, 0.0, 0, 1),
            (11, 1, OFF1 + 12.5, 25.0, 0.0, 0, 0)])
    # accel vehicle merging to main_right at k13 (accel->main lane change)
    add(6, [(12, 2, 440.0, 24.0, 0.0, 0, 0),
            (13, 1, 446.0, 24.0, 0.0, 0, 0)])
    # ramp birth at k15
    add(7, [(k, 2, 5.0 + 5.0 * (k - 15), 22.0, 0.0, 0, 0)
            for k in range(15, 21)])

    n_rows = len(rows["vid"])
    vids = ["mainline.5", "mainline.6", "mainline.7", "mainline.9",
            "ramp.1", "mainline.8", "ramp.2", "ramp.3"]
    npz = dict(
        vid=np.asarray(rows["vid"], dtype=np.int32),
        tick=np.asarray(rows["tick"], dtype=np.int16),
        lane=np.asarray(rows["lane"], dtype=np.int8),
        s=np.asarray(rows["s"], dtype=np.float64),
        v=np.asarray(rows["v"], dtype=np.float32),
        a=np.asarray(rows["a"], dtype=np.float32),
        is_cav=np.asarray(rows["cav"], dtype=np.int8),
        in_junction=np.asarray(rows["inj"], dtype=np.int8),
        type_label=np.zeros(n_rows, dtype=np.int8),
        vid_table=np.asarray(vids),
        lc_vid=np.asarray([6], dtype=np.int32),
        lc_tick=np.asarray([13], dtype=np.int16),
        lc_from=np.asarray([2], dtype=np.int8),
        lc_to=np.asarray([1], dtype=np.int8),
        lc_s=np.asarray([446.0]),
        b_vid=np.asarray([7], dtype=np.int32),
        b_tick=np.asarray([15], dtype=np.int16),
        b_lane=np.asarray([2], dtype=np.int8),
        b_s=np.asarray([5.0]), b_v=np.asarray([22.0], dtype=np.float32),
        b_departed=np.asarray([1], dtype=np.int8),
        e_vid=np.asarray([1], dtype=np.int32),
        e_tick=np.asarray([12], dtype=np.int16),
        e_s=np.asarray([140.0]), e_arrived=np.asarray([1], dtype=np.int8),
        ar_tick=np.asarray([15], dtype=np.int16),
        ar_entrance=np.asarray([1], dtype=np.int8),
        ar_v=np.asarray([22.0], dtype=np.float32),
        ar_cav=np.asarray([0], dtype=np.int8),
        ar_type=np.asarray([1], dtype=np.int8),
    )
    monkeypatch.setattr(L, "DATA", tmp_path)
    np.savez_compressed(tmp_path / "root_syn.npz", **npz)
    LD.load_root.cache_clear()
    return "syn"


def test_windows_births_exits_arrivals(tmp_path, monkeypatch):
    root = make_scene(tmp_path, monkeypatch)
    t0 = L.WINDOW_T0 + 10 * L.DT                       # t = 65.0
    w = LD.windows(root, t0, 5.0)                      # ticks 11..20
    assert set(w["state"]) >= {"vid", "id", "lane", "s", "d", "v", "a",
                               "is_cav", "in_junction"}
    assert len(w["state"]["vid"]) == 6                 # all alive at t0
    assert len(w["truth"]) == 10
    # births inside the window: the merge (vid 6, first row k12) and the
    # ramp entrance (vid 7, first row k15)
    assert set(w["new_vid"].tolist()) == {6, 7}
    assert w["arrivals"]["t_s"].tolist() == [67.5]
    assert w["arrivals"]["entrance"].tolist() == [1]   # ramp
    # vehicle 1 exits at tick 12 -> trajectory ends inside the window
    assert 1 in w["gone_vid"].tolist()
    k_of = {int(r["tick"]): r for r in w["truth"]}
    assert (k_of[12]["vid"] == 1).any()
    assert not (k_of[13]["vid"] == 1).any()
    # the birth row appears from tick 15 on
    assert not (k_of[14]["vid"] == 7).any()
    assert (k_of[15]["vid"] == 7).any()
    # d coordinate: ego main_right d = OFF1 - s
    ego = w["state"]["vid"] == 0
    assert abs(w["state"]["d"][ego][0] - (OFF1 - 102.0)) < 1e-9


def test_neighbour_slots_six_and_accel_adjacency(tmp_path, monkeypatch):
    root = make_scene(tmp_path, monkeypatch)
    t0 = L.WINDOW_T0 + 10 * L.DT
    nb = LD.neighbour_features(root, t0)
    for k in ("present", "gap", "rel_speed", "is_cav"):
        assert nb[k].shape == (6, 6)          # 6 vehicles alive at t0
    cols = {n: j for j, n in enumerate(LD.SLOT_NAMES)}

    def row(vid_idx):
        return {name: (nb["present"][vid_idx, j], nb["gap"][vid_idx, j],
                       nb["rel_speed"][vid_idx, j], nb["is_cav"][vid_idx, j])
                for name, j in cols.items()}

    ego = row(0)
    # own leader: 38 m ahead in d, slower by 2
    ol = ego["own_leader"]
    assert ol[0] == 1.0 and abs(ol[1] - 38.0) < 1e-3
    assert abs(ol[2] - (28.0 - 30.0)) < 1e-3 and ol[3] == 0.0
    # own follower: 22 m behind in d, faster by 1, is_cav
    of = ego["own_follower"]
    assert of[0] == 1.0 and abs(of[1] - 22.0) < 1e-3 and of[3] == 1.0
    # left follower (main_left vehicle is 8.45 m behind in d:
    # OFF0-OFF1 = +1.45 and s is 7 m smaller)
    lf = ego["left_follower"]
    assert lf[0] == 1.0 and abs(lf[1] - 8.45) < 1e-3
    assert ego["left_leader"][0] == 0.0
    # accel is the right lane: that vehicle is far ahead in d -> capped gap
    rl = ego["right_leader"]
    assert rl[0] == 1.0 and rl[1] == L.GAP_MAX
    assert ego["right_follower"][0] == 0.0
    # accel ego (vehicle 4): left lane = main_right (all of it behind in
    # d, nearest follower capped), right lane = none
    acc = row(4)
    assert acc["left_leader"][0] == 0.0
    assert acc["left_follower"][0] == 1.0
    assert acc["left_follower"][1] == L.GAP_MAX
    assert acc["right_leader"][0] == 0.0 and acc["right_follower"][0] == 0.0
    # junction vehicle at t0: all six slots absent
    jn = row(5)
    assert jn["own_leader"][0] == 0.0 and jn["own_follower"][0] == 0.0
    assert nb["present"][5].sum() == 0.0


def test_vtype_label_not_in_input_features(tmp_path, monkeypatch):
    root = make_scene(tmp_path, monkeypatch)
    w = LD.windows(root, L.WINDOW_T0, 5.0)
    nb = LD.neighbour_features(root, L.WINDOW_T0)
    bad = ("type", "vtype", "class")
    for k in list(w["state"]) + list(nb):
        assert not any(b in k.lower() for b in bad), k
    for tick_rows in w["truth"]:
        for k in tick_rows:
            assert not any(b in k.lower() for b in bad), k
    for name in LD.EGO_FEATURES + LD.SLOT_FEATURES + LD.SLOT_NAMES:
        assert not any(b in name.lower() for b in bad), name
    # the raw stored label still exists in the arrays (supervision) ...
    assert "type_label" in LD.load_root(root)
    # ... and the contract-format arrivals list carries vehicle_type by
    # spec (truth-side draw, not a per-vehicle input feature)
    assert "vehicle_type" in w["arrivals"]
