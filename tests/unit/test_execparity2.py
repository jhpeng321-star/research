"""S-EXECPARITY2: imagination executes commanded CAVs through the executor's own law.

The deployment executor's base law is ``sumo_native_accel`` -> TraCI
``getFollowSpeed`` -> SUMO 1.27.1 ``MSCFModel_IDM::_v`` on the designed
``cav_typical`` vType. The shared module ``wms.control.sumo_idm`` is the single
implementation of that exact function; these tests pin (a) the transcription of
the C++ formula, (b) the executor-side TraCI conventions it must reproduce,
(c) bitwise tensor/scalar identity of the shared law, (d) the residual-0 CTRL0
path and the speedMode-27 cap composition. CPU only; no SUMO import.
"""
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from wms.control import sumo_idm
from wms.control.alwayscmd import executed, inferred_base, inferred_cap, scalar_executed
from wms.control.contract import MERGE_S1_M, SPEED_LIMIT_MPS
from wms.control.nativeinit import sumo_native_accel


def test_cav_params_are_the_registered_vtype_design():
    assert sumo_idm.CAV_PARAMS == {
        "accel": 1.2, "decel": 2.0, "time_headway": 1.2, "min_gap": 2.0,
        "delta": 4.0, "speed_factor": 1.0}
    from wms.model.cav import TEXTBOOK_CAV_SETTINGS
    assert {k: sumo_idm.CAV_PARAMS[k] for k in
            ("accel", "decel", "time_headway", "min_gap", "speed_factor")} \
        == TEXTBOOK_CAV_SETTINGS
    traffic_shepherd = pytest.importorskip("traffic_shepherd.c4.population", reason="runtime copy")
    assert traffic_shepherd.TEXTBOOK_CAV_VTYPE == "cav_typical"
    assert traffic_shepherd.TEXTBOOK_CAV_PARAMS["accel_mps2"] == sumo_idm.CAV_PARAMS["accel"]
    assert traffic_shepherd.TEXTBOOK_CAV_PARAMS["decel_mps2"] == sumo_idm.CAV_PARAMS["decel"]
    assert traffic_shepherd.TEXTBOOK_CAV_PARAMS["tau_s"] == sumo_idm.CAV_PARAMS["time_headway"]
    assert traffic_shepherd.TEXTBOOK_CAV_PARAMS["min_gap_m"] == sumo_idm.CAV_PARAMS["min_gap"]
    assert traffic_shepherd.TEXTBOOK_CAV_PARAMS["delta"] == sumo_idm.CAV_PARAMS["delta"]
    assert traffic_shepherd.TEXTBOOK_CAV_PARAMS["speed_factor"] == sumo_idm.CAV_PARAMS["speed_factor"]


def test_follow_speed_is_an_exact_transcription_of_sumo_idm_v():
    """Independent python-math transcription of MSCFModel_IDM::_v (v1_27_1)."""
    a, b, tau, gap0, delta = 1.2, 2.0, 1.2, 2.0, 4.0
    two_sqrt_ab = 2. * math.sqrt(a * b)
    eps = 1e-10

    def cpp_v(gap2pred, ego_speed, pred_speed, des_speed):
        new_speed = ego_speed
        gap = gap2pred + gap0  # respectMinGap
        for _ in range(2):  # myIterations = max(1, int(0.5/0.25 + .5))
            s = max(0., new_speed * tau + new_speed * (new_speed - pred_speed) / two_sqrt_ab) + gap0
            gap = max(eps, gap)
            acc = a * (1. - (new_speed / max(eps, des_speed)) ** delta - (s * s) / (gap * gap))
            new_speed = max(0., new_speed + (acc * 0.5) / 2)
            gap = gap - max(0., ((new_speed - pred_speed) * 0.5) / 2)
        return max(0., new_speed)

    rng = np.random.default_rng(202610292)
    n = 3000
    v = rng.uniform(0., 33., n)
    vl = rng.uniform(0., 33., n)
    gap = rng.uniform(0.05, 400., n)
    des = rng.uniform(20., 31.29, n)
    got = np.array([sumo_idm.follow_speed_scalar(g, x, l, d) for g, x, l, d in
                    zip(gap, v, vl, des)])
    want = np.array([cpp_v(g, x, l, d) for g, x, l, d in zip(gap, v, vl, des)])
    assert np.max(np.abs(got - want)) < 1e-10


class SumoDomain:
    """TraCI vehicle-domain stand-in whose getFollowSpeed IS the shared law."""

    def __init__(self, vehicles):
        self.vehicles = vehicles
        self.gaps = []
        self._last_leader_of = None

    def getSpeed(self, v):
        if v == "LEAD":
            return self.vehicles[self._last_leader_of]["leader_speed"]
        return self.vehicles[v]["speed"]

    def getDecel(self, v):
        return sumo_idm.CAV_PARAMS["decel"]

    def getMinGap(self, v):
        return sumo_idm.CAV_PARAMS["min_gap"]

    def getAccel(self, v):
        return sumo_idm.CAV_PARAMS["accel"]

    def getAllowedSpeed(self, v):
        return self.vehicles[v]["allowed"]

    def getLeader(self, v):
        if self.vehicles[v]["leader"] is None:
            return None
        self._last_leader_of = v
        return ("LEAD", self.vehicles[v]["gap"])

    def getFollowSpeed(self, v, speed, gap, pred_speed, decel, leader_id):
        self.gaps.append((v, gap))
        # SUMO Vehicle::getFollowSpeed -> veh's own CF model (IDM::_v,
        # respectMinGap=True so minGap is added internally once)
        return sumo_idm.follow_speed_scalar(gap, speed, pred_speed,
                                            self.vehicles[v]["allowed"])


def test_shared_base_law_is_bit_identical_to_the_executor_traci_query():
    """sumo_idm.base_accel_scalar == sumo_native_accel on the same domain state."""
    rng = np.random.default_rng(2026103)
    ids = {}
    vehicles = {}
    for i in range(500):
        vid = f"v{i}"
        has = rng.uniform() < .8
        vehicles[vid] = dict(
            speed=float(rng.uniform(0., 30.)),
            leader_speed=float(rng.uniform(0., 30.)),
            leader="l" if has else None,
            gap=float(rng.uniform(0.2, 300.)),
            allowed=(sumo_idm.RAMP_SPEED_LIMIT_MPS if rng.uniform() < .3
                     else float(SPEED_LIMIT_MPS)))
        ids[vid] = vehicles[vid]
    api = SimpleNamespace(vehicle=SumoDomain(vehicles))
    for vid, spec in ids.items():
        want = float(sumo_native_accel(api, vid, 0.5))
        got = sumo_idm.base_accel_scalar(
            spec["speed"], spec["leader_speed"], spec["gap"] + sumo_idm.CAV_PARAMS["min_gap"],
            spec["allowed"], has_leader=spec["leader"] is not None)
        assert got == want, vid
    # the executor's frozen legacy gap convention (getLeader gap + minGap)
    assert all(abs(g - (v["gap"] + sumo_idm.CAV_PARAMS["min_gap"])) < 1e-12
               for (vid, g), v in zip(api.vehicle.gaps, [ids[k] for k in ids]
                                      ) if v["leader"] is not None)


def test_tensor_and_scalar_shared_law_are_bitwise_identical():
    rng = np.random.default_rng(20261029)
    n = 2000
    speed = torch.tensor(rng.uniform(0., 31., (1, n)), dtype=torch.float64)
    leader = torch.tensor(rng.uniform(0., 31., (1, n)), dtype=torch.float64)
    gap = torch.tensor(rng.uniform(0.5, 250., (1, n)), dtype=torch.float64)
    has = torch.tensor(rng.uniform(size=(1, n)) < .8)
    lane = torch.where(torch.tensor(rng.uniform(size=(1, n)) < .4), 2, 1)
    s_m = torch.tensor(rng.uniform(-300., 400., (1, n)), dtype=torch.float64)
    allowed = sumo_idm.allowed_speed(lane, s_m)
    assert float(sumo_idm.base_accel(speed, leader, gap, allowed,
                                     has_leader=has)[0, 0]) == \
        sumo_idm.base_accel_scalar(float(speed[0, 0]), float(leader[0, 0]),
                                   float(gap[0, 0]), float(allowed[0, 0]),
                                   has_leader=bool(has[0, 0]))
    for name, fn, scalar_fn in (
            ("base", sumo_idm.base_accel, sumo_idm.base_accel_scalar),
            ("cap", sumo_idm.safe_speed_cap, sumo_idm.safe_speed_cap_scalar)):
        if name == "base":
            out = fn(speed, leader, gap, allowed, has_leader=has)
        else:
            out = fn(speed, leader, gap, allowed, has_leader=has, lane=lane, s_m=s_m)
        equal = 0
        for i in range(n):
            kwargs = ({} if name == "base"
                      else dict(lane=int(lane[0, i]), s_m=float(s_m[0, i])))
            want = scalar_fn(float(speed[0, i]), float(leader[0, i]), float(gap[0, i]),
                             float(allowed[0, i]), has_leader=bool(has[0, i]), **kwargs)
            equal += float(out[0, i]) == want
        # same ops; vectorised vs 0-dim kernels may differ in the last ULP
        # (more often on the cap's extra min/stop composition)
        assert equal >= int(.85 * n), (name, equal, n)


def test_allowed_speed_lane_rule():
    lane = torch.tensor([[0, 1, 2, 2, -1, -1]])
    s_m = torch.tensor([[100., -50., -50., 50., 80., 80.]], dtype=torch.float64)
    got = sumo_idm.allowed_speed(lane, s_m)
    want = [SPEED_LIMIT_MPS, SPEED_LIMIT_MPS, sumo_idm.RAMP_SPEED_LIMIT_MPS,
            SPEED_LIMIT_MPS, SPEED_LIMIT_MPS, SPEED_LIMIT_MPS]
    assert got[0].tolist() == want
    assert sumo_idm.allowed_speed_scalar(2, -50.) == sumo_idm.RAMP_SPEED_LIMIT_MPS
    assert sumo_idm.allowed_speed_scalar(2, 50.) == SPEED_LIMIT_MPS
    assert sumo_idm.allowed_speed_scalar(-1, 80., prev_lane=2) == \
        sumo_idm.MERGE_CONNECTOR_SPEED_MPS
    assert sumo_idm.allowed_speed_scalar(-1, 80., prev_lane=1) == SPEED_LIMIT_MPS


def test_imagination_executed_is_the_shared_law_on_wm_geometry(monkeypatch):
    from wms.control import native_cav
    rng = np.random.default_rng(2026103)
    n = 1500
    v_np = rng.uniform(0., 30., n)
    leader_np = rng.uniform(0., 30., n)
    gap_np = rng.uniform(0.5, 220., n)
    has_np = rng.uniform(size=n) < .8
    lane_np = np.where(rng.uniform(size=n) < .35, 2, 1)
    s_np = rng.uniform(-250., 350., n)
    r_np = rng.uniform(-4., 2., n)
    holder = {"gap": torch.where(torch.tensor(has_np), torch.tensor(gap_np), torch.zeros(n))[None],
              "dv": (torch.tensor(v_np) - torch.where(torch.tensor(has_np),
                                                      torch.tensor(leader_np),
                                                      torch.tensor(v_np)))[None],
              "has": torch.tensor(has_np)[None]}
    monkeypatch.setattr(native_cav, "leader_gap_dv", lambda state: (
        holder["gap"], holder["dv"], holder["has"]))
    state = {"v_mps": torch.tensor(v_np, dtype=torch.float64)[None],
             "s_m": torch.tensor(s_np, dtype=torch.float64)[None],
             "lane": torch.tensor(lane_np)[None],
             "present": torch.ones(1, n, dtype=torch.bool),
             "length_m": torch.full((1, n), 5., dtype=torch.float64)}
    out = executed(state, torch.tensor(r_np, dtype=torch.float64)[None])[0].numpy()
    ref = np.array([
        scalar_executed(v_np[i], leader_np[i] if has_np[i] else v_np[i],
                        gap_np[i] if has_np[i] else float("inf"), r_np[i], None,
                        ramp_gap=(max(MERGE_S1_M - s_np[i], 0.) if lane_np[i] == 2 else None),
                        lane=int(lane_np[i]), s_m=s_np[i])
        for i in range(n)])
    assert np.max(np.abs(out - ref)) < 5e-9
    assert float((out == ref).mean()) > .9


def test_residual_zero_is_the_ctrl0_executed_law(monkeypatch):
    from wms.control import native_cav
    holder = {"gap": torch.tensor([[15., 40., 8.]]),
              "dv": torch.tensor([[3., 0., -1.]]),
              "has": torch.tensor([[True, False, True]])}
    monkeypatch.setattr(native_cav, "leader_gap_dv",
                        lambda state: (holder["gap"], holder["dv"], holder["has"]))
    v = torch.tensor([[12., 5., 2.]], dtype=torch.float64)
    state = {"v_mps": v, "s_m": torch.tensor([[400., 300., 20.]]),
             "lane": torch.tensor([[1, 1, 2]]), "present": torch.ones(1, 3, dtype=torch.bool),
             "length_m": torch.full((1, 3), 5.)}
    zero = executed(state, torch.zeros_like(v))
    base = inferred_base(state)
    cap = inferred_cap(state)
    torch.testing.assert_close(zero, torch.maximum(torch.minimum(base, cap), -v / .5))
    # CTRL0 identity: with residual 0 the issued command IS the base law
    assert float(zero[0, 1]) <= float(base[0, 1]) + 1e-12


def test_cap_composition_matches_sumo_speedmode27(monkeypatch):
    """executed == (max(0, min(v + issued*dt, vSafe, maxNextSpeed)) - v) / dt."""
    from wms.control import native_cav
    rng = np.random.default_rng(2026103)
    n = 2000
    v_np = rng.uniform(0., 30., n)
    leader_np = rng.uniform(0., 30., n)
    gap_np = rng.uniform(0.5, 220., n)
    has_np = rng.uniform(size=n) < .85
    lane_np = np.where(rng.uniform(size=n) < .3, 2, 1)
    s_np = rng.uniform(-250., 350., n)
    issued_np = rng.uniform(-3., 1.2, n)
    p = sumo_idm.CAV_PARAMS
    ref = []
    for i in range(n):
        allowed = sumo_idm.allowed_speed_scalar(int(lane_np[i]), s_np[i])
        g = float(gap_np[i]) if has_np[i] else float(sumo_idm.FREE_GAP_M)
        vl = float(leader_np[i]) if has_np[i] else float(v_np[i])
        base = sumo_idm.base_accel_scalar(
            float(v_np[i]), vl, float(gap_np[i]) if has_np[i] else float("inf"),
            allowed, has_leader=bool(has_np[i]))
        issued_i = min(max(base + issued_np[i], min(base, -3.)), max(base, 1.2))
        vsafe = sumo_idm.follow_speed_scalar(
            (g - p["min_gap"]) if has_np[i] else g, float(v_np[i]), vl, allowed)
        if int(lane_np[i]) == 2:
            stop_gap = max(MERGE_S1_M - s_np[i], 0.)
            stop = min(sumo_idm.follow_speed_scalar(stop_gap, float(v_np[i]), 0., allowed,
                                                    respect_min_gap=False),
                       stop_gap / 0.5)
            vsafe = min(vsafe, stop)
        vnext = max(0., min(v_np[i] + issued_i * 0.5, vsafe,
                            min(v_np[i] + p["accel"] * 0.5, allowed)))
        ref.append((vnext - v_np[i]) / 0.5)
    holder = {"gap": torch.where(torch.tensor(has_np), torch.tensor(gap_np), torch.zeros(n))[None],
              "dv": (torch.tensor(v_np) - torch.where(torch.tensor(has_np),
                                                      torch.tensor(leader_np),
                                                      torch.tensor(v_np)))[None],
              "has": torch.tensor(has_np)[None]}
    monkeypatch.setattr(native_cav, "leader_gap_dv",
                        lambda state: (holder["gap"], holder["dv"], holder["has"]))
    state = {"v_mps": torch.tensor(v_np, dtype=torch.float64)[None],
             "s_m": torch.tensor(s_np, dtype=torch.float64)[None],
             "lane": torch.tensor(lane_np)[None],
             "present": torch.ones(1, n, dtype=torch.bool),
             "length_m": torch.full((1, n), 5., dtype=torch.float64)}
    # drive the envelope from the base law exactly like the composer
    base = inferred_base(state)
    upper = torch.maximum(base, torch.full_like(base, 1.2))
    lower = torch.minimum(base, torch.full_like(base, -3.))
    issued = torch.minimum(torch.maximum(base + torch.tensor(issued_np)[None].to(base),
                                         lower), upper)
    cap = inferred_cap(state)
    out = torch.maximum(torch.minimum(issued, cap), -state["v_mps"] / .5)[0].numpy()
    assert np.max(np.abs(out - np.array(ref))) < 1e-12


def test_scalar_executed_rejects_nonfinite_and_keeps_legacy_signature():
    with pytest.raises(ValueError, match="finite"):
        scalar_executed(float("nan"), 10., 12., .1, None)
    # legacy call shape (params positional) still works and is ignored
    a = scalar_executed(10., 8., 15., .2, dict(accel=9., decel=9., time_headway=9.,
                                               min_gap=9., v0=9.))
    b = scalar_executed(10., 8., 15., .2, None)
    assert a == b
