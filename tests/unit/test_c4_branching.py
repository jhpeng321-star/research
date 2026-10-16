"""SUMO saved-state branching (traffic_shepherd.c4.branching): fast parts on fakes,
one short SUMO smoke (skipped without ``sumo``)."""
from __future__ import annotations

import shutil
import types

import numpy as np
import pytest

from traffic_shepherd.c4 import branching as br
from traffic_shepherd.c4.state import FrameBuilder


# ------------------------------------------------------------------ groups / arms
def test_chain_groups_link_along_s_any_lane():
    s = {"a": 0.0, "b": 140.0, "c": 280.0, "d": 600.0, "e": 700.0, "f": 2000.0}
    assert br.chain_groups(s) == [["a", "b", "c"], ["d", "e"], ["f"]]


def test_select_group_prefers_largest_then_nearest_merge_and_cuts_to_six():
    assert br.select_group({"a": -900.0, "b": -800.0, "c": 10.0, "d": 60.0}) == ["c", "d"]
    assert br.select_group({"a": -1000.0, "b": 0.0}) is None
    chain = {f"v{i}": -1000.0 + 100.0 * i for i in range(8)}
    chain["v7"] = chain["v6"] + 5.0                 # tightest window of 6 is the tail
    g = br.select_group(chain)
    assert len(g) == 6 and g == [f"v{i}" for i in range(2, 8)]
    assert br.select_group({"a": 0.0, "b": 200.0}) is None      # 200 m > 150 m link


def test_plan_arms_covers_reference_all_only_group_flip():
    rng = np.random.default_rng(0)
    arms = br.plan_arms(["a", "b", "c", "x"], ["a", "b", "c"], rng=rng, extra_solo=1, n_flip=2)
    names = [a["name"] for a in arms]
    assert names[:3] == ["B", "B_repeat", "ALL"]
    assert {"ONLY:a", "ONLY:b", "ONLY:c", "ONLY:x", "GROUP"} <= set(names)
    flips = [a for a in arms if a["kind"] == "FLIP"]
    assert len(flips) == 2
    for a in flips:
        assert a["members"] == frozenset("abc") - {a["i"]}
    assert next(a for a in arms if a["name"] == "ALL")["members"] is None
    assert next(a for a in arms if a["name"] == "B")["members"] == frozenset()
    no_group = br.plan_arms(["a"], None, rng=rng)
    assert [a["name"] for a in no_group] == ["B", "B_repeat", "ALL"]


# ------------------------------------------------------------------ algebra
def test_improvement_sign_positive_is_better():
    b = {"team_delay_vs": 10.0, "reward": -5.0, "merge_crossings": 3}
    x = {"team_delay_vs": 7.0, "reward": -4.0, "merge_crossings": 4}
    d = br.improvement(x, b)
    assert d == {"team_delay_vs": 3.0, "reward": 1.0, "merge_crossings": 1.0}


def test_cooperation_gain_marginal_and_interaction():
    def m(v):
        return {"reward": v}
    d = {"B": m(0.0), "ALL": m(5.0), "ONLY:a": m(1.0), "ONLY:b": m(0.5), "GROUP": m(3.0),
         "FLIP:a": m(0.5)}
    c = br.cooperation(d, ["a", "b"])
    assert c["sum_D_i"]["reward"] == pytest.approx(1.5)
    assert c["C_G"]["reward"] == pytest.approx(1.5)            # D_G - sum D_i
    assert c["M_i"]["a"]["reward"] == pytest.approx(2.5)       # D_G - D_{G\a}
    assert c["I_i"]["a"]["reward"] == pytest.approx(1.5)       # M_a - D_a
    assert c["D_all"]["reward"] == 5.0
    missing = br.cooperation({"B": m(0), "ALL": m(1), "ONLY:a": m(1), "GROUP": m(2)}, ["a", "b"])
    assert missing["C_G"] is None                              # D_b missing -> no C_G


def test_cvar_and_bootstrap():
    assert br.cvar([1.0] * 9 + [11.0]) == pytest.approx(11.0)
    assert br.cvar(range(1, 21)) == pytest.approx(19.5)
    assert br.cvar([]) == 0.0
    ci = br.bootstrap_ci([1.0, 2.0, 3.0, 4.0])
    assert ci["n"] == 4 and ci["mean"] == pytest.approx(2.5) and ci["lo"] <= 2.5 <= ci["hi"]
    assert br.bootstrap_ci([None])["n"] == 0
    assert br.bootstrap_ci([2.0])["lo"] is None


def _kin(rows, k=4):
    kin = np.zeros((k, 7), np.float32)
    for slot, (s, lane, v, a) in rows.items():
        kin[slot] = [s, lane == 0, lane == 1, lane == 2, v, a, 5.0]
    return kin


def test_step_pair_carries_exits_and_splits_reused_slots():
    # slot 0: continuing vehicle crossing the merge end; slot 1: exits and is reused by a
    # new vehicle; slot 2: newly inserted into a free slot
    kin0 = _kin({0: (70.0, 1, 10.0, 0.0), 1: (900.0, 0, 20.0, 0.0)})
    kin1 = _kin({0: (80.0, 1, 12.0, 1.0), 1: (-2000.0, 0, 25.0, 0.0), 2: (-300.0, 2, 15.0, 0.0)})
    m0 = np.array([1, 1, 0, 0], bool)
    m1 = np.array([1, 1, 1, 0], bool)
    id0 = np.array([0, 1, -1, -1])
    id1 = np.array([0, 2, 3, -1])
    state, nxt, accel, info = br.step_pair(kin0, m0, id0, kin1, m1, id1)
    assert state["present"].tolist() == [True, True, False, False, False]
    assert nxt["present"].tolist() == [True, False, True, False, True]
    assert nxt["s_m"][1] == pytest.approx(900.0)          # exit carried, not the newcomer
    assert nxt["s_m"][4] == pytest.approx(-2000.0)        # newcomer on an appended slot
    assert accel.tolist()[:2] == [1.0, 0.0]
    assert info["merge_crossings"] == 1
    assert info["delay_by_slot"][0] == pytest.approx(0.5 * (1 - 12.0 / br.SPEED_LIMIT_MPS))
    assert info["delay_by_slot"][1] == pytest.approx(0.5 * (1 - 20.0 / br.SPEED_LIMIT_MPS))


def test_step_pair_feeds_wms_team_reward():
    kin0 = _kin({0: (0.0, 1, 10.0, 0.0), 1: (30.0, 1, 10.0, 0.0)})
    kin1 = _kin({0: (5.0, 1, 10.0, 0.0), 1: (35.0, 1, 10.0, 0.0)})
    m = np.array([1, 1, 0, 0], bool)
    ids = np.array([0, 1, -1, -1])
    state, nxt, accel, _ = br.step_pair(kin0, m, ids, kin1, m, ids)
    from traffic_shepherd.c4.wms_glue import team_reward_terms
    terms = team_reward_terms(state, nxt, accel)
    assert set(terms) == {"reward", "cost_ttc", "delay", "throughput", "smooth", "hard_brake",
                          "fairness_delay", "source_wait"}
    assert terms["delay"] == pytest.approx(2 * 0.5 * (1 - 10.0 / 31.29), rel=1e-5)


def test_trajectory_diff():
    a = [(1.0, {"x": (1.0, 2.0)}), (1.5, {"x": (2.0, 2.0)})]
    assert br.trajectory_diff(a, a)["identical"]
    b = [(1.0, {"x": (1.0, 2.0)}), (1.5, {"x": (2.5, 2.0), "y": (0.0, 0.0)})]
    d = br.trajectory_diff(a, b)
    assert d["max_abs_ds_m"] == pytest.approx(0.5) and d["mismatched_entries"] == 1
    assert not d["identical"]


# ------------------------------------------------------------------ controller plug-in
class _FakeEnv:
    def __init__(self):
        self.builder = FrameBuilder(4, {"c1", "c2"})
        table = {v: {"lane_id": "1191554636_0", "lane_position_m": p, "speed_mps": 20.0,
                     "acceleration_mps2": 0.0, "length_m": 5.0}
                 for v, p in (("c1", 10.0), ("c2", 50.0), ("h1", 30.0))}
        self.frame = self.builder.build(0.0, table)

    def step(self, accel_by_slot=None, *, fallback="native", fallback_ids=None):
        return None, {}


def test_split_output_forms():
    a, lane = br.split_output(np.zeros((1, 3)))
    assert lane is None and a.shape == (1, 3)
    a, lane = br.split_output({"accel": np.ones((1, 3)), "lane": np.array([[0, 1, -1]])})
    assert lane.tolist() == [[0, 1, -1]]
    a, lane = br.split_output((np.ones(3), np.zeros(3)))
    assert a.shape == (1, 3) and lane.shape == (1, 3)


def test_branch_controller_masks_members_and_named_fallbacks():
    env = _FakeEnv()
    ids = env.builder.slot_vehicle_ids(env.frame)

    def fn(state):
        return np.full((1, 4), 0.7)
    c = br.BranchController(fn, name="const")
    assert c.base_mode == "direct"
    out, kw = c.command(env, None, frozenset({"c2"}), fn)
    for slot, vid in enumerate(ids):
        assert (out[0, slot] == 0.7) == (vid == "c2")
    out, _ = c.command(env, None, None, fn)
    assert sorted(v for s, v in enumerate(ids) if out[0, s] == 0.7) == ["c1", "c2"]
    # lane commands: env without lane support -> counted as dropped, not applied
    c2 = br.BranchController(lambda s: {"accel": np.zeros((1, 4)), "lane": np.ones((1, 4))})
    c2.command(env, None, frozenset({"c1"}), c2.fn)
    assert c2.lane_dropped == 1
    damping = br.BranchController("damping")
    assert damping.base_mode == "direct" and damping.snapshot() is None
    acc, kw = damping.command(env, None, frozenset({"c1"}), None)
    assert acc is None and kw == {"fallback": "damping", "fallback_ids": frozenset({"c1"})}
    assert br.BranchController("native").command(env, None, None, None) == (None, {})
    with pytest.raises(ValueError):
        br.BranchController("wall")


def test_branch_controller_random_spec_is_a_callable():
    c = br.BranchController("random", seed=3)
    assert c.base_mode == "direct" and callable(c.fn)


def test_snapshot_shares_api_and_copies_python_state():
    api = types.ModuleType("fake_traci")      # modules cannot be deep-copied
    env = types.SimpleNamespace(api=api, _circles=None, recorder=None, counter=[1])
    snap = br.snapshot_env(env)
    env.counter.append(2)
    br.restore_env(env, snap)
    assert env.counter == [1] and env.api is api
    br.restore_env(env, snap)
    env.counter.append(5)
    assert snap["counter"] == [1]                 # restores never alias the snapshot


def test_cli_refuses_restricted_paths(tmp_path):
    with pytest.raises(ValueError):
        br.checked(tmp_path / "HOLDOUT_x")


# ------------------------------------------------------------------ SUMO smoke
@pytest.mark.skipif(shutil.which("sumo") is None, reason="SUMO not installed")
def test_sumo_branching_is_deterministic(tmp_path):
    from traffic_shepherd.c4.env import CavSumoEnv

    env = CavSumoEnv(controller="direct", extra_sumo_cli=br.SAVE_STATE_CLI, physics_log=False,
                     scenario_root=tmp_path / "sc")
    c = br.BranchController("damping")
    sink = []
    pts = br.measure_episode(env, c, seed=4107, demand_vph=2700, penetration=0.2,
                             warmup_s=60, decision_times=[70], horizon_s=4,
                             workdir=tmp_path / "states", rng=np.random.default_rng(0),
                             n_flip=1, branch_sink=sink.append)
    p = pts[0]
    assert p["determinism_B_vs_B_repeat"]["identical"] and p["metrics_B_equal_B_repeat"]
    assert all(v == 0.0 for v in p["D"]["B"].values())
    assert len(sink) == p["n_branches"]
    summary = br.summarize(pts)
    assert summary["determinism"]["all_identical"]
    # base episode (damping via the per-tick fallback path, no reload) == ALL arm (reloaded
    # into a fresh SUMO process)
    assert p["fidelity_ALL_vs_live"]["identical"]



# ------------------------------------------------------------------ save right after a load
class _FakeSim:
    """SUMO 1.27.1 quirk: a state saved right after ``loadState`` lacks ``loaderTime``."""

    def __init__(self):
        self.t, self.saved = 0.0, []

    def getTime(self):
        return self.t

    def loadState(self, path):
        self.t = float(open(path, encoding="utf-8").read().split('time="')[1].split('"')[0])

    def saveState(self, path):
        self.saved.append(path)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(f'<snapshot time="{self.t}">\n<delay number="1"/>\n')


class _LoadedEnv(types.SimpleNamespace):
    __hash__ = object.__hash__
    now = property(lambda self: self.api.simulation.getTime())


def test_save_right_after_load_copies_the_loaded_file(tmp_path):
    conn = object()                                # one TraCI connection = one SUMO process
    api = types.SimpleNamespace(getConnection=lambda: conn, simulation=_FakeSim(),
                                vehicle=types.SimpleNamespace(getIDList=lambda: []))
    env = _LoadedEnv(api=api, _circles=None, recorder=None,
                     reader=types.SimpleNamespace(_subscribed=set()),
                     ownership=types.SimpleNamespace(_mode_configured=set()))
    src = tmp_path / "t900_mid.xml"
    src.write_text('<snapshot time="900.0">\n<delay number="1" loaderTime="1007181"/>\n',
                   encoding="utf-8")
    snap = {"frame": types.SimpleNamespace(time_s=900.0)}
    br.load_state(env, src, snap, fresh=False)
    assert env.now == 900.0
    out = tmp_path / "t900.xml"
    br.save_state(env, out)                        # no step since the load: byte copy
    assert out.read_bytes() == src.read_bytes() and api.simulation.saved == []
    api.simulation.t = 900.5                       # after a step SUMO saves itself; a file
    bad = tmp_path / "t900_5.xml"                  # without loaderTime is refused both ways
    with pytest.raises(RuntimeError, match="loaderTime"):
        br.save_state(env, bad)
    with pytest.raises(RuntimeError, match="loaderTime"):
        br.load_state(env, bad, snap, fresh=False)


@pytest.mark.skipif(shutil.which("sumo") is None, reason="SUMO not installed")
def test_sumo_branching_spacing_equal_to_horizon(tmp_path):
    """Regression (COOP-MAPPO-H300): with spacing == horizon the base episode resumes from the
    live-continuation state exactly at the next decision point and saves it again with no step
    in between; that file used to lack loaderTime and its loads re-inserted arrived vehicles
    ("unknown or duplicate lifecycle event")."""
    from traffic_shepherd.c4.env import CavSumoEnv

    env = CavSumoEnv(controller="direct", extra_sumo_cli=br.SAVE_STATE_CLI, physics_log=False,
                     scenario_root=tmp_path / "sc")
    pts = br.measure_episode(env, br.BranchController("damping"), seed=4107, demand_vph=2700,
                             penetration=0.2, warmup_s=60, decision_times=[200, 204],
                             horizon_s=4, workdir=tmp_path / "states",
                             rng=np.random.default_rng(0), n_flip=0)
    assert [p["t_s"] for p in pts] == [200, 204]
    assert env.ledger.arrived                      # vehicles had arrived before t = 204
    for p in pts:
        assert p["determinism_B_vs_B_repeat"]["identical"] and p["metrics_B_equal_B_repeat"]
    assert (tmp_path / "states" / "s4107_t204.xml").read_bytes() == (
        tmp_path / "states" / "s4107_t200_mid.xml").read_bytes()
