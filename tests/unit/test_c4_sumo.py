"""C4 SUMO side (traffic_shepherd.c4): conversion parity, sensing, recorder, seeds, collector.

Fast tests use fake TraCI objects; the single SUMO smoke is skipped without ``sumo``.
"""
from __future__ import annotations

import gzip
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from traffic_shepherd.c4 import geometry as G  # noqa: E402
from traffic_shepherd.c4 import state as ST  # noqa: E402
from traffic_shepherd.c4.recorder import (CONTROLLERS, EpisodeRecorder,  # noqa: E402
                                          episode_windows, load_episode,
                                          write_trajectory_npz)
from traffic_shepherd.c4.sumo_io import build_physics_row, validate_physics_row  # noqa: E402

WM_N1 = ROOT / "output" / "wd01" / "wm_n1"


def _frozen():
    if not (WM_N1 / "features" / "featurizer.py").is_file():
        pytest.skip("frozen WM-N1 featurizer not present")
    from traffic_shepherd.c4.parity import frozen_featurizer
    return frozen_featurizer()


# ------------------------------------------------------------------ geometry
def test_lane_table_matches_frozen_featurizer_geometry():
    _, geometry = _frozen()
    assert G.LANE_TABLE == geometry.LANE_TABLE
    assert G.HERE_OFFSETS == geometry.HERE_OFFSETS
    for lane_id, pos in (("1191554636_0", 12.5), ("1487606179_0", 30.0),
                         ("1487606178_1", 4.0), ("11553048#1_0", 100.0)):
        assert G.lane_and_s(lane_id, pos) == pytest.approx(geometry.s_down_of(lane_id, pos))


def test_merge_zone_is_zero_to_75_44_on_the_acceleration_lane():
    assert G.lane_and_s("1487606179_0", 0.0) == (2, pytest.approx(0.0))
    assert G.lane_and_s("1487606179_0", 75.44) == (2, pytest.approx(75.44))
    assert G.lane_and_s("11553048#1_0", 0.0)[1] == pytest.approx(-378.66)
    assert G.lane_and_s("1487606178_0", 0.0) == (1, pytest.approx(75.44))
    with pytest.raises(G.GeometryError):
        G.lane_and_s("999_0", 1.0)


def _veh(lane_id, pos, v=20.0, a=0.0):
    return {"road_id": lane_id.rpartition("_")[0], "lane_id": lane_id,
            "lane_position_m": pos, "speed_mps": v, "acceleration_mps2": a,
            "type_id": "normal_car", "speed_mode": 31, "length_m": 5.0, "min_gap_m": 2.5,
            "allowed_speed_mps": 31.29, "lane_index": int(lane_id.rpartition("_")[2]),
            "leader": None}


def _synthetic_rows():
    rng = np.random.default_rng(3)
    lanes = ["1191554636_0", "1191554636_1", "11553048#1_0", "1487606179_1",
             ":J1_0_0", "1487606178_0"]
    rows, live = [], {}
    for tick in range(1, 60):
        t = tick * 0.5
        for vid in list(live):
            if rng.random() < 0.08:
                del live[vid]
        for _ in range(int(rng.integers(0, 4))):
            vid = f"{'ramp' if rng.random() < 0.3 else 'mainline'}.{tick}_{_}"
            live[vid] = lanes[int(rng.integers(0, len(lanes) - 2))]
        for vid in list(live):
            if rng.random() < 0.1:
                live[vid] = lanes[int(rng.integers(0, len(lanes)))]
        table = {vid: _veh(lane, float(rng.uniform(0, 70)), float(rng.uniform(0, 30)),
                           float(rng.uniform(-3, 1.5))) for vid, lane in live.items()}
        rows.append((t, table))
    return rows


def test_slot_pool_and_frames_match_frozen_parse_run(tmp_path):
    featurizer, _ = _frozen()
    rows = _synthetic_rows()
    k = 12                                        # small pool: exercises drop + reuse
    builder = ST.FrameBuilder(k, cav_ids={vid for _, tab in rows for vid in tab
                                          if vid.endswith("_0")})
    physics = tmp_path / "run" / "physics_steps.jsonl.gz"
    physics.parent.mkdir()
    frames, prev = [], None
    with gzip.open(physics, "wt", encoding="utf-8") as fh:
        for t, table in rows:
            frames.append(builder.build(t, table))
            row = build_physics_row(t, table, previous_row=prev, cav_ids=builder.cav_ids,
                                    cav_actions={}, edges=_edges(), departed=[], arrived=[],
                                    colliding=[])
            validate_physics_row(row)
            fh.write(json.dumps(row) + "\n")
            prev = row
    parsed = featurizer.parse_run(tmp_path, 0.0, k_max=k, physics_path=physics)
    assert parsed["n_dropped_concurrent"] == builder.pool.dropped_steps > 0
    for i, frame in enumerate(frames):
        tick = i + 1
        frozen_ids = [parsed["slot_vid"][j] if j >= 0 else None
                      for j in parsed["slot_index"][tick].tolist()]
        assert frozen_ids == builder.slot_vehicle_ids(frame)
    sel = parsed["rec_tick"] >= 1
    tick, slot = parsed["rec_tick"][sel], parsed["rec_slot"][sel]
    lanes = np.stack([f.lane for f in frames])[tick - 1, slot]
    np.testing.assert_array_equal(lanes, parsed["rec_lane"][sel])
    s = np.stack([f.s for f in frames])[tick - 1, slot]
    modelled = parsed["rec_junc"][sel] == 0
    np.testing.assert_allclose(s[modelled], parsed["rec_sdown"][sel][modelled], atol=1e-4)
    junction = np.stack([f.junction for f in frames])[tick - 1, slot]
    np.testing.assert_array_equal(junction, parsed["rec_junc"][sel] == 1)


def _edges():
    return {"mainline_entry": "1191554636", "mainline_exit": "1487606178",
            "ramp_entry": "11553048#1"}


def test_junction_rows_carry_last_position_and_have_no_lane():
    b = ST.FrameBuilder(4, cav_ids={"a"})
    f1 = b.build(0.5, {"a": _veh("1191554636_0", 2644.0)})
    f2 = b.build(1.0, {"a": _veh(":J1_0_0", 1.0)})
    slot = b.pool.slot_of("a")
    assert f1.lane[slot] == 1 and f2.lane[slot] == -1 and f2.junction[slot]
    assert f2.s[slot] == pytest.approx(f1.s[slot]) and f2.lane_carry[slot] == 1
    state = ST.frame_to_state(f2)
    assert state["lane"][0, slot] == -1 and bool(state["present"][0, slot])
    assert bool(state["is_cav"][0, slot]) and state["s_m"].shape == (1, 4)


# ------------------------------------------------------------------- RSU zones
def test_rsu_constants_match_contract_and_cells():
    from traffic_shepherd.c4.wms_glue import contract
    C = contract()
    assert ST.RSU_MAINLINE_S_M == C.RSU_MAINLINE_S_M
    assert ST.RSU_RAMP_S_M == C.RSU_RAMP_S_M and ST.RSU_CELL_M == C.RSU_CELL_M
    assert ST.n_rsu_cells() == 21 + 21 + 10
    assert ST.rsu_cell_index(0, -800.0) == 0 and ST.rsu_cell_index(1, -800.0) == 21
    assert ST.rsu_cell_index(2, -378.66) == 42 and ST.rsu_cell_index(2, 75.44) is None
    assert ST.rsu_cell_index(0, -800.01) is None and ST.rsu_cell_index(1, 225.43) == 41


class _FakeApi:
    """Minimal TraCI stand-in: lanes with lengths, vehicles with positions."""

    def __init__(self, vehicles):
        self.vehicles = vehicles            # vid -> dict(lane_id, pos, v, a, length)
        api = self
        lengths = {"1191554636_0": 2644.22, "1191554636_1": 2645.67, "1487606179_0": 75.44,
                   "1487606179_1": 75.44, "1487606179_2": 75.44, "1487606178_0": 500.0,
                   "1487606178_1": 500.0, "11553048#1_0": 378.66}

        class lane:
            @staticmethod
            def getLength(lane_id):
                return lengths[lane_id]

            @staticmethod
            def getLastStepVehicleIDs(lane_id):
                return [v for v, d in api.vehicles.items() if d["lane_id"] == lane_id]

        class edge:
            @staticmethod
            def getLaneNumber(edge_id):
                return {"1191554636": 2, "1487606179": 3, "1487606178": 2,
                        "11553048#1": 1}[edge_id]

        class vehicle:
            getLaneID = staticmethod(lambda v: api.vehicles[v]["lane_id"])
            getLanePosition = staticmethod(lambda v: api.vehicles[v]["pos"])
            getLength = staticmethod(lambda v: api.vehicles[v]["length"])
            getSpeed = staticmethod(lambda v: api.vehicles[v]["v"])
            getAcceleration = staticmethod(lambda v: api.vehicles[v]["a"])
            getAllowedSpeed = staticmethod(lambda v: 31.29)
            getLaneIndex = staticmethod(lambda v: int(api.vehicles[v]["lane_id"][-1]))
            getMinGap = staticmethod(lambda v: 2.5)
            getIDList = staticmethod(lambda: list(api.vehicles))

            @staticmethod
            def getLeader(vid, dist):
                me = api.vehicles[vid]
                best = None
                for o, d in api.vehicles.items():
                    if o != vid and d["lane_id"] == me["lane_id"] and d["pos"] > me["pos"]:
                        gap = d["pos"] - d["length"] - me["pos"] - 2.5
                        if gap <= dist and (best is None or gap < best[1]):
                            best = (o, gap)
                return best

        self.lane, self.edge, self.vehicle = lane, edge, vehicle


def _config():
    from traffic_shepherd.config import load_config
    return load_config()


def _fake_vehicles():
    rng = np.random.default_rng(5)
    lanes = ["1191554636_0", "1191554636_1", "1487606179_0", "1487606179_1",
             "1487606179_2", "1487606178_0", "11553048#1_0"]
    out = {}
    for i in range(80):
        lane = lanes[i % len(lanes)]
        top = {"1191554636_0": 2644.0, "1191554636_1": 2645.0,
               "11553048#1_0": 378.0}.get(lane, 75.0)
        out[f"v{i}"] = dict(lane_id=lane, pos=float(rng.uniform(5, top)),
                            v=float(rng.uniform(0, 30)), a=float(rng.uniform(-3, 1)),
                            length=5.0)
    return out


def test_reference_frame_and_cell_counts_agree_with_lane_table():
    api = _FakeApi(_fake_vehicles())
    ref = ST.ReferenceFrame(api, _config())
    table = {v: _veh(d["lane_id"], d["pos"], d["v"], d["a"]) for v, d in api.vehicles.items()}
    for v, d in api.vehicles.items():
        lane, s = G.lane_and_s(d["lane_id"], d["pos"])
        rl, rs = ref.lane_and_s(d["lane_id"], d["pos"])
        assert rl == lane and rs == pytest.approx(s, abs=1e-6)
    builder = ST.FrameBuilder(192, cav_ids={"v1", "v2"})
    frame = builder.build(1.0, table)
    np.testing.assert_array_equal(ref.cell_counts(table), ST.frame_cell_counts(frame))
    det = ST.rsu_detections(frame, builder.vehicle_ids)
    in_zone = sum(1 for v in table.values()
                  if ST.in_rsu_zone(*[np.asarray([x]) for x in
                                      G.lane_and_s(v["lane_id"], v["lane_position_m"])])[0])
    assert len(det) == in_zone > 0
    assert {"vehicle_id", "s_m", "lane", "v_mps", "is_cav", "timestamp_s"} <= set(det[0])


def test_table_sensor_matches_sumo_local_sensor():
    from traffic_shepherd.c4.sensing import TableSensor
    from traffic_shepherd.sensing import SumoLocalSensor
    api = _FakeApi(_fake_vehicles())
    cfg = _config()
    lane_map = {"1191554636_0": 0.0, "1191554636_1": 0.0, "11553048#1_0": 0.0}
    slow = SumoLocalSensor(api, cfg, lane_map)
    fast = TableSensor(api, cfg, lane_map)
    table = {v: dict(_veh(d["lane_id"], d["pos"], d["v"], d["a"]), length_m=d["length"])
             for v, d in api.vehicles.items()}
    fast.bind(table)

    def outcome(sensor, vid):
        try:
            return sensor.capture(vid, now_s=3.0)
        except ValueError as exc:            # overlaps are refused identically
            return repr(exc)

    results = [(outcome(fast, v), outcome(slow, v)) for v in api.vehicles]
    assert all(a == b for a, b in results)
    assert sum(not isinstance(a, str) for a, _ in results) > 40


def test_closing_ttc_from_table_matches_metrics_ttc_exposure():
    from traffic_shepherd.metrics import closing_ttc, ttc_exposure
    api = _FakeApi(_fake_vehicles())
    ids = sorted(api.vehicles)
    expected, minimum = ttc_exposure(api, ids, 0.5)
    speeds = {v: api.vehicles[v]["v"] for v in ids}
    got, low = 0.0, None
    for v in ids:
        leader = api.vehicle.getLeader(v, 200.0)
        if not leader or leader[0] not in speeds:
            continue
        ttc = closing_ttc(leader[1] + 2.5, speeds[v], speeds[leader[0]])
        if ttc is not None:
            low = ttc if low is None else min(low, ttc)
            got += 0.5 if ttc < 3.0 else 0.0
    assert got == expected and low == minimum


# ------------------------------------------------------- recorder + windows
def _fake_episode(tmp_path, n_ctrl=40, k=8):
    rec = EpisodeRecorder(tmp_path / "ep", k=k)
    rec.start({"seed": 4107})
    builder = ST.FrameBuilder(k, cav_ids={"c0", "c1"})
    frame = builder.build(0.0, {})
    rec.on_frame(frame, None)
    s = {"c0": 100.0, "c1": 80.0, "h0": 60.0}
    for c in range(n_ctrl):
        decisions = {}
        for vid in ("c0", "c1"):
            slot = builder.pool.slot_of(vid)
            if slot is not None:
                decisions[slot] = ("direct", 0.5, 20.5, 0.7)
        rec.on_control(float(c), frame, decisions, builder.vehicle_ids)
        for sub in (1, 2):
            for vid in s:
                s[vid] += 10.0
            table = {vid: _veh("1191554636_0", pos, 20.0) for vid, pos in s.items()}
            frame = builder.build(c + 0.5 * sub, table)
            rec.on_frame(frame, None)
    arrays = rec.finish({"ok": True}, vehicle_ids=builder.vehicle_ids,
                        cav_ids=builder.cav_ids, meta={"warmup_s": 5.0})
    return rec.out, arrays


def test_recorder_episode_and_trajectory_windows_load_in_wms(tmp_path):
    from traffic_shepherd.c4.wms_glue import ensure_wms_importable
    ensure_wms_importable()
    from wms.data.collate import collate
    from wms.data.dataset import TrajectoryDataset
    out, arrays = _fake_episode(tmp_path)
    assert json.loads((out / "manifest.json").read_text())["status"] == "complete"
    ep = load_episode(out)
    assert ep["present"].shape == (81, 8) and ep["controller"].shape == (40, 8)
    assert ep["meta"]["warmup_s"] == 5.0
    windows = episode_windows(ep, window_s=10, stride_s=10, history_s=4.0)
    assert windows["states"].shape == (3, 21, 8, 7)          # t0 = 5, 15, 25 s
    assert windows["controller_timeline"].shape == (3, 10, 8)
    np.testing.assert_array_equal(windows["history_kin"][:, -1], windows["states"][:, 0])
    path = write_trajectory_npz(tmp_path / "traj" / "trajectory.npz", windows)
    ds = TrajectoryDataset.from_pack(path, history_s=4.0)
    assert len(ds) == 3
    item = ds[0]
    assert item["driver_hist"].shape == (8, 5, 9)
    direct = CONTROLLERS.index("direct")
    cav_slots = np.flatnonzero(item["is_cav"][0])
    assert len(cav_slots) == 2
    assert (item["action_timeline"]["controller"][:, cav_slots] == direct).all()
    assert np.allclose(item["action_timeline"]["acceleration_mps2"][:, cav_slots], 0.5)
    batch = collate([ds[0], ds[1]])
    assert tuple(batch["states"].shape) == (2, 21, 8, 7)
    with pytest.raises(FileExistsError):
        write_trajectory_npz(path, windows)
    lines = gzip.open(out / "run" / "action_timeline.jsonl.gz", "rt").read().splitlines()
    first = json.loads(lines[0])
    assert first["owner"] == "direct" and first["issued_command"] == 0.5


def test_history_buffer_long_and_short_features():
    buf = ST.HistoryBuffer(max_ticks=30)
    b = ST.FrameBuilder(6, cav_ids=set())
    for i in range(25):
        table = {"a": _veh("1191554636_0", 100.0 + 10 * i), "b": _veh("1191554636_0", 50.0 + 10 * i)}
        if i > 20:
            table.pop("b")
        buf.append(b.build(0.5 * i, table))
    hist = buf.history(5.0)
    assert hist["kin"].shape == (11, 6, 7) and hist["target_mask"].shape == (10, 6)
    feat, mask = buf.driver_history(5.0, 1.0)
    assert feat.shape == (1, 6, 6, 9) and mask.shape == (1, 6, 6)
    short, smask = buf.driver_history(None, history_len=10)
    assert short.shape == (1, 6, 10, 6)
    slot_b = 1
    assert not bool(smask[0, slot_b, -1])                    # b left: not present now


# ------------------------------------------------------------------ policies
def test_random_policy_bounds_cav_only_and_deterministic():
    import torch
    from traffic_shepherd.c4.baselines import RandomAccelPolicy
    state = {"is_cav": torch.tensor([[True, False, True, True]]),
             "present": torch.tensor([[True, True, False, True]])}
    p1, p2 = RandomAccelPolicy(7), RandomAccelPolicy(7)
    for _ in range(30):
        a1, a2 = p1(state), p2(state)
        np.testing.assert_array_equal(a1, a2)
        assert np.isnan(a1[0, 1]) and np.isnan(a1[0, 2])
        assert -3.0 <= a1[0, 0] <= 1.5 and -3.0 <= a1[0, 3] <= 1.5


def test_wall_law_forms_one_pair_per_mainline_lane():
    from traffic_shepherd.c4.baselines import WallLaw
    b = ST.FrameBuilder(8, cav_ids={"c0", "c1", "c2"})
    table = {"c0": _veh("1191554636_1", 2645.67 - 800.0, 25.0),
             "c1": _veh("1191554636_0", 2644.22 - 810.0, 24.0),
             "c2": _veh("1191554636_0", 2644.22 - 1900.0, 24.0)}
    frame = b.build(10.0, table)
    law = WallLaw(8)
    cmds = law.commands(frame, b, now_s=10.0, live_cavs={"c0", "c1", "c2"})
    assert set(cmds) == {"c0", "c1"} and law.n_walls == 1
    assert all(-3.0 <= a <= 1.5 for a in cmds.values())


# --------------------------------------------------------------- seeds / CLI
def test_seed_guard_only_admits_exposed_development_seeds():
    from traffic_shepherd.c4.seeds import SeedRefusal, check_case, parse_seeds
    check_case(4107, 2700)
    check_case(4148, 2400)
    for seed in (8707, 4201, 5001, 920044, 930001, 9801, 7101, 8410, 4106):
        with pytest.raises((SeedRefusal, ValueError)):
            check_case(seed, 2700)
    assert parse_seeds("4107-4109,4120") == [4107, 4108, 4109, 4120]


def test_collector_plans_and_preserves_incomplete_runs(tmp_path):
    from traffic_shepherd.c4 import collect
    args = collect.argparse.Namespace(
        policy="random", policy_loader="x:y", k=192, warmup=10, horizon=10,
        check_parity=False, no_ttc=False, no_physics_log=False, seeds="4107-4108", demands="2400,2700",
        penetrations="0.05,0.4", out=str(tmp_path / "out"))
    jobs = collect.plan_jobs(args)
    assert len(jobs) == 8 and jobs[0]["job_id"] == "s4107_m2400_p0050"
    done_dir, partial_dir = Path(jobs[0]["episode_dir"]), Path(jobs[1]["episode_dir"])
    done_dir.mkdir(parents=True)
    (done_dir / "manifest.json").write_text(json.dumps({"status": "complete"}))
    (done_dir / "summary.json").write_text("{}")
    partial_dir.mkdir(parents=True)
    (partial_dir / "manifest.json").write_text(json.dumps({"status": "running"}))
    todo, done = collect.prepare(jobs)
    assert len(done) == 1 and len(todo) == 7
    assert (partial_dir.parent / f"{partial_dir.name}.incomplete-1").is_dir()
    assert todo[0]["attempt"] == 2
    with pytest.raises(SystemExit):
        collect.main(["--policy", "native", "--seeds", "4107", "--out",
                      str(tmp_path / "HOLDOUT_x"), "--dry-run"])


# ------------------------------------------------------------------ SUMO smoke
@pytest.mark.skipif(shutil.which("sumo") is None, reason="SUMO not installed")
def test_sumo_short_direct_episode_applies_commands(tmp_path):
    from traffic_shepherd.c4.baselines import RandomAccelPolicy, run_policy_episode
    from traffic_shepherd.c4.env import CavSumoEnv
    env = CavSumoEnv(check_parity=True, scenario_root=tmp_path / "scen")
    seen = []
    summary = run_policy_episode(env, RandomAccelPolicy(3), seed=4107, demand_vph=2700,
                                 penetration=0.4, warmup_s=20, horizon_s=10,
                                 episode_dir=tmp_path / "ep",
                                 on_step=lambda s, i: seen.append(i))
    assert summary["collisions"] == 0 and summary["teleports"] == 0
    assert summary["rsu_parity_failures"] == 0 and summary["sensor_parity_mismatches"] == 0
    assert summary["command_counts"].get("wall", 0) > 0        # direct commands issued
    assert len(seen) == 10 and seen[-1]["done"]
    assert isinstance(seen[0]["rsu_detections"], list)
    ep = load_episode(tmp_path / "ep")
    direct = ep["controller"][20:] == CONTROLLERS.index("direct")
    assert direct.any()
    if (WM_N1 / "features" / "featurizer.py").is_file():
        from traffic_shepherd.c4.parity import verify_episode
        assert verify_episode(tmp_path / "ep")["ok"]
    assert not math.isnan(summary["window_sim_s_per_wall_s"])
