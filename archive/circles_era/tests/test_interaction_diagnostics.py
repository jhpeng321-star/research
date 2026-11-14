"""Offline observer integrity and exact synthetic runner equivalence; no SUMO."""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import pytest

from traffic_shepherd.interaction_diagnostics import (
    FILENAME, OfflineInteractionObserver, capture_states, offline_read_cache, same_lane_links,
)
from traffic_shepherd.metrics import RequestLedger, braking_effort, ttc_exposure
from traffic_shepherd.scenarios import TripRequest


_spec = importlib.util.spec_from_file_location("paired_test_support", Path(__file__).with_name("test_paired_experiments.py"))
_support = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_support)
constant_policy, job, paired = _support.constant_policy, _support.job, _support.paired
registration, runtime = _support.registration, _support.runtime


def api_for(states, leaders=None):
    def field(name):
        return lambda vehicle: states[vehicle][name]
    return SimpleNamespace(vehicle=SimpleNamespace(getLaneID=field("lane"), getRoadID=field("edge"),
        getLanePosition=field("position"), getLength=field("length"), getSpeed=field("speed"),
        getAcceleration=field("acceleration"), getMinGap=lambda _: 2.,
        getLeader=lambda vehicle, _: (leaders or {}).get(vehicle)),
        lane=SimpleNamespace(getLength=lambda _: 1000.))


def test_same_lane_identity_matches_bruteforce_with_ties_lengths_and_overlap():
    rng = random.Random(71)
    states = [dict(vehicle_id=f"v{i:03}", lane_id=f"L{i%3}", lane_position_m=rng.randrange(12)*5.,
        length_m=float(rng.randrange(2, 30)), speed_mps=float(rng.randrange(30))) for i in range(90)]
    links = same_lane_links(states)
    for ego in states:
        same = [r for r in states if r["lane_id"] == ego["lane_id"] and r is not ego]
        fronts = sorted((r for r in same if r["lane_position_m"] >= ego["lane_position_m"]),
            key=lambda r: (r["lane_position_m"]-r["length_m"]-ego["lane_position_m"], r["vehicle_id"]))
        rears = sorted((r for r in same if r["lane_position_m"] < ego["lane_position_m"]),
            key=lambda r: (ego["lane_position_m"]-ego["length_m"]-r["lane_position_m"], r["vehicle_id"]))
        for side, candidates in (("front", fronts), ("rear", rears)):
            actual = links[ego["vehicle_id"]]
            assert actual[f"same_lane_{side}_id"] == (candidates[0]["vehicle_id"] if candidates else None)
            if candidates:
                other = candidates[0]
                expected_gap = (other["lane_position_m"]-other["length_m"]-ego["lane_position_m"]
                    if side == "front" else ego["lane_position_m"]-ego["length_m"]-other["lane_position_m"])
                assert actual[f"same_lane_{side}_gap_m"] == expected_gap
                assert actual[f"same_lane_{side}_relative_speed_mps"] == other["speed_mps"]-ego["speed_mps"]


def test_route_ttc_is_distinct_from_sensor_geometry_and_reuses_readonly_cache():
    states = dict(a=dict(lane="L0", edge="E0", position=100., length=5., speed=20., acceleration=-4.),
        b=dict(lane="L0", edge="E0", position=125., length=5., speed=19., acceleration=-3.),
        c=dict(lane="L1", edge="E1", position=200., length=5., speed=10., acceleration=2.))
    cache = offline_read_cache(api_for(states, {"a": ("c", 8.)}))
    expected = braking_effort((cache.vehicle.getAcceleration(v) for v in states), .5)
    exposure, minimum = ttc_exposure(cache, states, .5)
    before_queries = cache.vehicle.queries
    rows = capture_states(cache, states)
    assert sum(r["braking_vehicle_seconds"] for r in rows) == expected["braking_vehicle_seconds"]
    assert sum(r["acceleration_effort"] for r in rows) == expected["acceleration_energy"]
    assert sum(r["low_ttc_vehicle_seconds"] for r in rows) == exposure == .5
    assert minimum == rows[0]["closing_ttc_s"] == 1.
    assert rows[0]["sensor_front_candidate_id"] == "b" and rows[0]["same_lane_front_gap_m"] == 20.
    assert rows[0]["ttc_leader_id"] == "c" and rows[0]["ttc_gap_m"] == 10.
    assert cache.vehicle.queries-before_queries == 4*len(states)  # Only new geometry fields.
    with pytest.raises(AttributeError, match="read queries only"):
        cache.vehicle.setSpeed("a", 0.)


def test_compressed_stream_clock_events_service_and_partial_preservation(tmp_path):
    requests = [TripRequest("a", 0., "main", "normal_car", True),
        TripRequest("b", .25, "ramp", "normal_car", False)]
    states = dict(a=dict(lane="L0", edge="E0", position=100., length=5., speed=20., acceleration=-4.))
    api = api_for(states)
    path = tmp_path / FILENAME
    observer = OfflineInteractionObserver(path, requests=requests, horizon_s=2, edge_regions={"E0": "main"})
    observer.sample(api, now_s=0., live_ids=[], starting_teleport_ids=[])
    observer.sample(api, now_s=.5, live_ids=["a"], departed_ids=["a"], starting_teleport_ids=[])
    states["a"]["lane"] = "L1"
    observer.sample(api, now_s=1., live_ids=["a"], colliding_ids=["a"], new_colliding_ids=["a"],
        starting_teleports=1, starting_teleport_ids=["a"])
    observer.sample(api, now_s=1.5, live_ids=[], arrived_ids=["a"])
    observer.sample(api, now_s=2., live_ids=[])
    metadata = observer.close()
    assert observer.close() == metadata
    raw = gzip.decompress(path.read_bytes())
    records = [json.loads(line) for line in raw.splitlines()]
    assert metadata["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert metadata["uncompressed_sha256"] == hashlib.sha256(raw).hexdigest()
    assert metadata["complete"] and metadata["sample_count"] == 5
    samples = records[1:-1]
    assert [r["time_s"] for r in samples] == [0., .5, 1., 1.5, 2.]
    assert samples[2]["lane_transitions"][0]["kind"] == "within_edge_lane_change"
    assert samples[2]["starting_teleport_ids"] == ["a"]
    assert samples[3]["starting_teleport_ids"] is None and not samples[3]["starting_teleport_ids_available"]
    assert samples[3]["disappeared_ids"] == ["a"]
    ledger = RequestLedger(requests)
    for row in samples[1:]:
        ledger.advance(row["time_s"], row["departed_ids"], row["arrived_ids"])
        assert sum(v["system_vehicle_seconds"] for v in row["service_by_origin"].values()) == ledger.system_vehicle_seconds
        assert sum(v["source_wait_vehicle_seconds"] for v in row["service_by_origin"].values()) == ledger.source_vehicle_seconds
    assert samples[-1]["service_by_origin"]["ramp"]["source_waiting"] == 1
    partial = OfflineInteractionObserver(tmp_path / "partial.gz", requests=requests, horizon_s=2)
    partial.sample(api, now_s=0., live_ids=[])
    for kwargs in (dict(now_s=1.), dict(now_s=.5, departed_ids=["unknown"]),
                   dict(now_s=.5, starting_teleports=1, starting_teleport_ids=[])):
        with pytest.raises(ValueError):
            partial.sample(api, live_ids=[], **kwargs)
    assert not partial.close()["complete"]
    assert json.loads(gzip.decompress((tmp_path / "partial.gz").read_bytes()).splitlines()[-1])["complete"] is False


@pytest.mark.parametrize("arm", ["damping", "wall", "sham"])
def test_observer_preserves_every_control_input_command_and_physical_stream(tmp_path, runtime, arm):
    api = sys.modules["traci"]
    api.vehicle.getRoadID = lambda _: "E0"
    api.vehicle.getLength = lambda _: 5.
    api.lane = SimpleNamespace(getLength=lambda _: 1000.)
    api.simulation.getStartingTeleportIDList = lambda: []
    results, writes, contexts = [], [], []
    for enabled in (False, True):
        runtime.update(time=0., writes=[], closed=False, starts=0)
        data, observed = registration(), []
        if enabled:
            data["interaction_diagnostics"] = True
        policy = constant_policy(arm)
        def callback(context):
            observed.append(context)
            return policy(context)
        root = tmp_path / str(enabled)
        result = paired.evaluate_arm(job=job(arm), registration=data, output=root, policy=callback)
        paired.verified_arm(root)
        results.append({k: v for k, v in result.items() if k != "elapsed_wall_seconds"})
        writes.append(list(runtime["writes"]))
        contexts.append(observed)
    assert writes[0] == writes[1] and contexts[0] == contexts[1] and results[0] == results[1]
    for name in ("telemetry.jsonl", "intervals.jsonl", "traffic.jsonl", "policy.jsonl", "formation.jsonl",
                 "offline_state.jsonl", "decision_snapshots.json"):
        assert (tmp_path / "False" / name).read_bytes() == (tmp_path / "True" / name).read_bytes()
    manifest = json.loads((tmp_path / "True" / "manifest.json").read_text())
    assert manifest["interaction_diagnostics"]["sample_count"] == 9
    assert not (tmp_path / "False" / FILENAME).exists()
    with (tmp_path / "True" / FILENAME).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="changed offline interaction"):
        paired.verified_arm(tmp_path / "True")


def test_registration_default_remains_absent_and_true_is_bound(tmp_path):
    for flag in (False, True):
        root = tmp_path / str(flag)
        paired.register(root, jobs=[job()], hypotheses=["observer fixture"], criteria=None,
            stopping_rules=["no simulation"], policy_version="fixture", interaction_diagnostics=flag)
        _, data = paired.read_registration(root / "registration.json")
        assert ("interaction_diagnostics" in data) is flag
    with pytest.raises(ValueError, match="explicit boolean"):
        paired.register(tmp_path / "invalid", jobs=[job()], hypotheses=["fixture"], criteria=None,
            stopping_rules=["fixture"], policy_version="fixture", interaction_diagnostics=1)
