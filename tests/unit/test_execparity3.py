"""S-EXECPARITY3: SUMO lane/link-based leader identification (default OFF).

These tests pin (a) bitwise flag-OFF identity of
:func:`wms.control.sumo_leader.leader_gap_dv_sumo` and of the
:mod:`wms.physics.neighbours` lane search with the frozen legacy bodies,
(b) the SUMO-rule fixtures measured on the 00:20 recordings (ramp +2.93 m
connector, mainline entry +3.00 m, exit +8.00 m, junction-transit candidates at
the connector-entry bound, no phantom corridor-projected leaders between the
ramp and the accel lane), (c) the offline replay tool wiring under the flag.
CPU only; no SUMO import.
"""
import json

import numpy as np
import pytest
import torch

from wms.control.contract import (MERGE_CONNECTOR_ENTRY_LEN_M,
                                  MERGE_CONNECTOR_EXIT_LEN_M,
                                  MERGE_CONNECTOR_RAMP_LEN_M, MERGE_S1_M)
from wms.control.sumo_leader import (chain_s, junction_chain_s,
                                     leader_gap_dv_sumo, leader_rule_enabled)

torch.set_num_threads(1)


def _state(rows, *, dtype=torch.float32):
    """rows = (s, v, lane, junction, carry[, present])."""
    present = torch.tensor([[r[5] if len(r) > 5 else True for r in rows]], dtype=torch.bool)
    return {"s_m": torch.tensor([[r[0] for r in rows]], dtype=dtype),
            "v_mps": torch.tensor([[r[1] for r in rows]], dtype=dtype),
            "a_mps2": torch.zeros(1, len(rows), dtype=dtype),
            "lane": torch.tensor([[r[2] for r in rows]], dtype=torch.long),
            "junction": torch.tensor([[r[3] for r in rows]], dtype=torch.bool),
            "lane_carry": torch.tensor([[r[4] for r in rows]], dtype=torch.long),
            "present": present,
            "is_cav": torch.zeros(1, len(rows), dtype=torch.bool),
            "length_m": torch.full((1, len(rows)), 5., dtype=dtype)}


@pytest.fixture(autouse=True)
def _flag_off(monkeypatch):
    monkeypatch.delenv("C4_SUMO_LEADER_RULE", raising=False)


# ------------------------------------------------------------------ flag OFF

def test_flag_off_is_bitwise_legacy_leader_rule():
    from wms.control.native_cav import leader_gap_dv
    rng = np.random.default_rng(20261031)
    n = 800
    lane_np = rng.integers(-1, 3, n)
    rows = [(float(rng.uniform(-200., 300.)), float(rng.uniform(0., 30.)),
             int(lane_np[i]), bool(rng.uniform() < .1), int(lane_np[i]),
             bool(rng.uniform() < .9)) for i in range(n)]
    state = _state(rows)
    assert not leader_rule_enabled()
    got = leader_gap_dv_sumo(state)
    want = leader_gap_dv(state)
    assert all(torch.equal(a, b) for a, b in zip(got, want))


def test_flag_off_lane_neighbour_is_bitwise_legacy_body():
    from wms.physics.config import PhysicsConfig
    from wms.physics.neighbours import lane_neighbour
    rng = np.random.default_rng(20261032)
    n = 500
    lane_np = rng.integers(0, 3, n)
    s_np = rng.uniform(-250., 350., n)
    kin = np.stack([s_np,
                    (lane_np == 0).astype(float), (lane_np == 1).astype(float),
                    (lane_np == 2).astype(float),
                    rng.uniform(0., 30., n), rng.uniform(-2., 2., n),
                    rng.uniform(3., 8., n)], -1)
    kin_t = torch.tensor(kin[None])
    # legacy body (pre-S-EXECPARITY3 neighbours.py), byte-for-byte
    present = torch.ones(1, n, dtype=torch.bool)
    physics = PhysicsConfig()
    for ahead in (True, False):
        s, v, length = kin_t[..., 0], kin_t[..., 4], kin_t[..., 6]
        lane = kin_t[..., 1:4].argmax(-1)
        ds = s.unsqueeze(-2) - s.unsqueeze(-1)
        other = present.unsqueeze(-2) & (lane.unsqueeze(-2) == lane.unsqueeze(-1))
        valid = other & ((ds > 0) if ahead else (ds < 0))
        if ahead:
            bumper = ds - length.unsqueeze(-2)
        else:
            bumper = -ds - length.unsqueeze(-1)
        from wms.physics.neighbours import MIN_GAP_M, _CLOSED_ROAD_FREE_GAP_M
        gaps = torch.where(valid, bumper, torch.full_like(ds, float("inf")))
        gap, index = gaps.min(dim=-1)
        has = torch.isfinite(gap)
        other_v = torch.gather(v, -1, index)
        free = physics.free_road_gap_m if physics.open_road else _CLOSED_ROAD_FREE_GAP_M
        want = (torch.where(has, gap.clamp_min(MIN_GAP_M), torch.full_like(gap, free)),
                torch.where(has, other_v, v))
        query = lane
        got = lane_neighbour(kin_t, present, query, physics, ahead=ahead)
        assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1])


def test_flag_off_alwayscmd_geometry_is_legacy(monkeypatch):
    from wms.control import alwayscmd, native_cav
    rows = [(-10., 12., 2, False, 2), (3., 8., 2, False, 2), (5., 9., 0, False, 0)]
    state = _state(rows)
    seen = {}
    real = native_cav.leader_gap_dv(state)
    monkeypatch.setattr(native_cav, "leader_gap_dv",
                        lambda st: seen.update(out=real) or real)
    v, gap, vl, has = alwayscmd._wm_geometry(state)
    assert gap is seen["out"][0] and has is seen["out"][2]


# ------------------------------------------------------------------ chain geometry

def test_chain_coordinates_add_registered_connector_lengths(monkeypatch):
    monkeypatch.setenv("C4_SUMO_LEADER_RULE", "1")
    lane = torch.tensor([[0, 1, 2, 2, 0, 1, 0]])
    s = torch.tensor([[-4., 10., -5., 10., 76., 76., -7.]], dtype=torch.float64)
    c = chain_s(lane, s)
    want = [-4., 10. + MERGE_CONNECTOR_ENTRY_LEN_M, -5.,
            10. + MERGE_CONNECTOR_RAMP_LEN_M,
            76. + MERGE_CONNECTOR_ENTRY_LEN_M + MERGE_CONNECTOR_EXIT_LEN_M,
            76. + MERGE_CONNECTOR_ENTRY_LEN_M + MERGE_CONNECTOR_EXIT_LEN_M, -7.]
    assert c[0].tolist() == want


def test_connector_lengths_are_network_attributes():
    # osm.net.xml.gz internal lanes: :1780739017_0_0 = 2.93 (ramp),
    # :1780739017_1_* = 3.00 (entry), :13638027693_0_* = 8.00 (exit)
    assert (MERGE_CONNECTOR_RAMP_LEN_M, MERGE_CONNECTOR_ENTRY_LEN_M,
            MERGE_CONNECTOR_EXIT_LEN_M) == (2.93, 3.00, 8.00)


def test_junction_chain_s_is_the_connector_entry_bound():
    got = junction_chain_s(torch.tensor([2, 0, 1]),
                           torch.tensor([-.1, -.4, 76.], dtype=torch.float64))
    want = [0., 0., MERGE_CONNECTOR_ENTRY_LEN_M + MERGE_S1_M]
    assert got.tolist() == want


# ------------------------------------------------------------------ flag ON fixtures

def test_ramp_ego_leader_gains_the_ramp_connector_length(monkeypatch):
    monkeypatch.setenv("C4_SUMO_LEADER_RULE", "1")
    """The measured 00:20 anatomy: SUMO leader gap = frame gap + 2.93 m."""
    rows = [(-10., 12., 2, False, 2), (3., 8., 2, False, 2)]
    state = _state(rows)
    from wms.control.native_cav import leader_gap_dv
    legacy = float(leader_gap_dv(state)[0][0, 0])
    assert legacy == pytest.approx(3. - (-10.) - 5.)       # 8.0 m corridor gap
    got = leader_gap_dv_sumo(state)
    assert bool(got[2][0, 0]) and float(got[0][0, 0]) == pytest.approx(
        legacy + MERGE_CONNECTOR_RAMP_LEN_M)
    assert float(got[1][0, 0]) == pytest.approx(12. - 8.)


def test_mainline_entry_phantom_is_pushed_across_the_connector(monkeypatch):
    monkeypatch.setenv("C4_SUMO_LEADER_RULE", "1")
    """Corridor-projected 1 m leader at the merge edge is 4 m away on the chain."""
    rows = [(-4., 12., 0, False, 0), (2., 8., 0, False, 0)]
    state = _state(rows)
    from wms.control.native_cav import leader_gap_dv
    legacy = float(leader_gap_dv(state)[0][0, 0])
    assert legacy == pytest.approx(1.0)
    got = leader_gap_dv_sumo(state)
    assert float(got[0][0, 0]) == pytest.approx(legacy + MERGE_CONNECTOR_ENTRY_LEN_M)


def test_mainline_exit_leader_gains_the_exit_connector(monkeypatch):
    monkeypatch.setenv("C4_SUMO_LEADER_RULE", "1")
    rows = [(74., 12., 1, False, 1), (76., 8., 1, False, 1)]
    state = _state(rows)
    got = leader_gap_dv_sumo(state)
    assert float(got[0][0, 0]) == pytest.approx(
        76. + MERGE_CONNECTOR_ENTRY_LEN_M + MERGE_CONNECTOR_EXIT_LEN_M
        - (74. + MERGE_CONNECTOR_ENTRY_LEN_M) - 5.)


def test_accel_lane_ego_does_not_see_upstream_ramp_vehicles(monkeypatch):
    monkeypatch.setenv("C4_SUMO_LEADER_RULE", "1")
    rows = [(5., 10., 2, False, 2), (-1., 10., 2, False, 2)]
    state = _state(rows)
    got = leader_gap_dv_sumo(state)
    assert not bool(got[2][0, 0]) and float(got[0][0, 0]) == 0.


def test_junction_transit_vehicle_leads_its_upstream_lane_only(monkeypatch):
    monkeypatch.setenv("C4_SUMO_LEADER_RULE", "1")
    # mid-merge ramp vehicle on the connector (junction, carry 2) at the entry bound
    rows = [(-4., 10., 2, False, 2),          # ramp ego
            (-.1, 4., -1, True, 2),           # on the ramp connector
            (-4., 10., 1, False, 1)]          # mainline-lane ego (must not see it)
    state = _state(rows)
    got = leader_gap_dv_sumo(state)
    assert bool(got[2][0, 0])
    assert float(got[0][0, 0]) == pytest.approx(0.1)       # bumper-gap floor
    assert float(got[1][0, 0]) == pytest.approx(10. - 4.)
    assert not bool(got[2][0, 2])


def test_junction_transit_without_carry_is_no_candidate(monkeypatch):
    monkeypatch.setenv("C4_SUMO_LEADER_RULE", "1")
    rows = [(-4., 10., 2, False, 2), (-.1, 4., -1, True, -1)]
    state = _state(rows)
    got = leader_gap_dv_sumo(state)
    assert not bool(got[2][0, 0])


def test_rule_uses_only_observable_frame_information(monkeypatch):
    monkeypatch.setenv("C4_SUMO_LEADER_RULE", "1")
    """lane_carry absent -> junction slots stay non-candidates (legacy behaviour)."""
    state = _state([(-4., 10., 2, False, 2), (-.1, 4., -1, True, -1)])
    state.pop("lane_carry")
    got = leader_gap_dv_sumo(state)
    assert not bool(got[2][0, 0])


def test_flag_parsing_and_per_call_read(monkeypatch):
    monkeypatch.setenv("C4_SUMO_LEADER_RULE", "1")
    assert leader_rule_enabled()
    monkeypatch.setenv("C4_SUMO_LEADER_RULE", "0")
    assert not leader_rule_enabled()
    assert leader_rule_enabled({}) is False
    assert leader_rule_enabled({"C4_SUMO_LEADER_RULE": "on"}) is True


# ------------------------------------------------------------------ neighbours

def test_neighbours_chain_rule_and_dead_end(monkeypatch):
    from wms.physics.config import PhysicsConfig
    from wms.physics import neighbours
    monkeypatch.setenv("C4_SUMO_LEADER_RULE", "1")
    # ramp ego at s=-4, accel-lane leader at s=3 (chain gap 2.93 m longer)
    kin = torch.tensor([[[-4., 0., 0., 1., 10., 0., 5.],
                         [3., 0., 0., 1., 8., 0., 5.]]])
    present = torch.ones(1, 2, dtype=torch.bool)
    physics = PhysicsConfig()
    gap, v = neighbours.lane_neighbour(kin, present, torch.tensor([[2]]),
                                       physics, ahead=True)
    assert float(gap[0, 0]) == pytest.approx(3. + MERGE_CONNECTOR_RAMP_LEN_M
                                             - (-4.) - 5., abs=1e-5)
    assert float(v[0, 0]) == pytest.approx(8.)
    # a junction-transit slot is neither an ego nor a candidate of a lane search
    kin_j = torch.tensor([[[-4., 0., 0., 1., 10., 0., 5.],
                           [-.1, 0., 0., 0., 4., 0., 5.],
                           [3., 0., 0., 1., 8., 0., 5.]]])
    present_j = torch.ones(1, 3, dtype=torch.bool)
    gap_j, _ = neighbours.lane_neighbour(kin_j, present_j, torch.tensor([[2, 2, 2]]),
                                         physics, ahead=True)
    assert float(gap_j[0, 1]) == float(physics.free_road_gap_m)  # junction ego: free
    assert float(gap_j[0, 0]) == pytest.approx(
        3. + MERGE_CONNECTOR_RAMP_LEN_M - (-4.) - 5., abs=1e-5)  # candidate skipped
    # the lane-2 dead end competes at the chain coordinate MERGE_S1 + connector
    kin_end = torch.tensor([[[70., 0., 0., 1., 9., 0., 5.],
                             [200., 0., 0., 1., 20., 0., 5.]]])
    gap_e, ve = neighbours.lead_neighbour(kin_end, present, physics)
    assert float(gap_e[0, 0]) == pytest.approx(
        float(physics.merge_zone_s1_m) + MERGE_CONNECTOR_RAMP_LEN_M
        - (70. + MERGE_CONNECTOR_RAMP_LEN_M), abs=1e-5)
    assert float(ve[0, 0]) == 0.


def test_neighbours_flag_off_dead_end_is_legacy(monkeypatch):
    from wms.physics.config import PhysicsConfig
    from wms.physics import neighbours
    kin_end = torch.tensor([[[70., 0., 0., 1., 9., 0., 5.],
                             [200., 0., 0., 1., 20., 0., 5.]]])
    present = torch.ones(1, 2, dtype=torch.bool)
    gap_e, ve = neighbours.lead_neighbour(kin_end, present, PhysicsConfig())
    assert float(gap_e[0, 0]) == pytest.approx(float(MERGE_S1_M) - 70., abs=1e-5)


# ------------------------------------------------------------------ parity tool

def test_replay_states_from_episode_carries_lane_carry(tmp_path):
    import sys
    from pathlib import Path
    sys.path[:0] = [str(Path(__file__).resolve().parents[2] / "src"),
                    str(Path(__file__).resolve().parents[2])]
    from tools.execparity.replay import states_from_episode
    n = 3
    z = {"s_m": np.array([[10., 20., 30.]] * n), "v_mps": np.array([[5., 6., 7.]] * n),
         "a_mps2": np.zeros((n, 3)), "present": np.ones((n, 3), dtype=bool),
         "is_cav": np.ones((n, 3), dtype=bool),
         "lane": np.array([[0, 2, -1]] * n, dtype=np.int8),
         "junction": np.array([[False, False, True]] * n, dtype=bool),
         "lane_carry": np.array([[0, 2, 2]] * n, dtype=np.int8),
         "length_m": np.full((n, 3), 5.)}
    st = states_from_episode(dict(z))(0)
    assert st["lane_carry"].tolist() == [[0, 2, 2]]
    z.pop("lane_carry")                                     # older tapes fall back
    st = states_from_episode(dict(z))(0)
    assert st["lane_carry"].tolist() == [[0, 2, -1]]


def test_replay_law_value_under_the_sumo_rule(tmp_path, monkeypatch):
    """The imagined executed accel uses the chain-coordinate leader gap."""
    import gzip
    import json
    import sys
    from pathlib import Path
    sys.path[:0] = [str(Path(__file__).resolve().parents[2] / "src"),
                    str(Path(__file__).resolve().parents[2])]
    from tools.execparity.alwayscmd_replay import world_rows
    from wms.control.alwayscmd import scalar_executed
    payload = {"schema": "execparity_learned_idm/1", "fit_controller": "direct",
               "evaluation_data_used": False, "fit_worlds": ["a"], "score_worlds": ["b"],
               "parameters": {"accel": 1.1, "decel": 2., "time_headway": 1.2,
                              "min_gap": 2., "v0": 28.}}
    driver = tmp_path / "driver.json"
    driver.write_text(json.dumps(payload))
    monkeypatch.setenv("C4_EXECUTION_DRIVER", str(driver))
    monkeypatch.setenv("C4_SUMO_LEADER_RULE", "1")
    n = 3
    # ramp ego at s=-10 (lane 2), chain leader at s=3 on the accel lane
    z = {"time_s": np.array([300. + .5 * i for i in range(n)]),
         "s_m": np.array([[-10., 3.] for i in range(n)]),
         "v_mps": np.array([[20., 20.] for i in range(n)]),
         "a_mps2": np.zeros((n, 2)), "present": np.ones((n, 2), dtype=bool),
         "is_cav": np.array([[True, False]] * n),
         "lane": np.array([[2, 2]] * n, dtype=np.int8),
         "junction": np.zeros((n, 2), dtype=bool),
         "lane_carry": np.array([[2, 2]] * n, dtype=np.int8),
         "vid_index": np.array([[0, 1]] * n, dtype=np.int32),
         "vehicle_ids": np.array(["cav1", "ldr1"]),
         "length_m": np.full((n, 2), 5.)}
    job = tmp_path / "s977001_alwaysx"
    (job / "deploy/episode").mkdir(parents=True)
    np.savez(job / "deploy/episode/episode.npz", **z)
    row = {"t_s": 300., "vehicle_id": "cav1", "residual_mps2": .1,
           "base_law_mps2": 0., "issued_mps2": .1, "speed_before_mps": 20.,
           "leader": {"gap_m": 10.93 - 2., "speed_mps": 20.}}  # SUMO getLeader (bumper - minGap)
    with gzip.open(job / "execution_log.jsonl.gz", "wt", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    out, _ = world_rows(job)
    chain_gap = 3. + MERGE_CONNECTOR_RAMP_LEN_M - (-10.) - 5.
    want = scalar_executed(20., 20., chain_gap, .1, None, lane=2, s_m=-10.)
    assert len(out) == 1
    assert out[0]["imagined"] == pytest.approx(want, abs=1e-5)


def test_committed_gate_receipt_satisfies_mode2_predicates():
    from pathlib import Path
    receipt = json.loads((Path(__file__).resolve().parents[2]
                          / "tools/execparity/alwayscmd3_execution_parity.json")
                         .read_text(encoding="utf-8"))
    assert receipt["schema"] == "execparity_always_commanded_acceleration/1"
    assert receipt["backend_contract"] == "always_commanded_ctrl0/1"
    assert receipt["measured_real_sumo"] is True and receipt["zero_ctrl0_identity"] is True
    assert receipt["passed"] is True and set(receipt["error"]) == {"ours"}
    assert receipt["error"]["ours"]["rmse"] <= receipt["native_arm_head_error"]["rmse"]
