"""PL-04j: the REAL master's RSU merge-region lane map (unit, zero SUMO).

The silent PL-04 defect: ``build_real_components.make_harness`` never passed
``merge_lane_distances``, so the real master's ``SumoLocalSensor`` carried an
EMPTY lane->merge map, no frame could coverage-reach the merge region,
``merge_observed`` resolved False (contract 8.15, fail closed) and the
planner abstained at every decision. The fix resolves the map on the LIVE
connection via ``scenarios.observation_map`` right after connect (master) and
on every branch connection (parity), and REFUSES an empty resolution with
the NAMED :class:`MergeMapUnavailableError`.

These tests run the REAL ``RealMasterHarness.start``/``observe`` and the REAL
``SumoLocalSensor`` over a faked traci module whose edge/lane surfaces carry
the adopted network's real edge ids, lane counts and lengths -- so the
resolved map is bit-identical to the registered one -- and a faked Popen
(records the launch, starts nothing). No SUMO, no signalling.
"""

import subprocess
import sys
from types import SimpleNamespace

import pytest

from traffic_shepherd.config import load_config
from traffic_shepherd.planner.closed_loop import (
    MergeMapUnavailableError,
    RealMasterHarness,
)

#: the adopted adapter's edge ids and their REAL lane counts
EDGE_LANE_COUNTS = {
    "1191554636": 2,  # pre_merge
    "1487606179": 3,  # merge
    "1487606178": 2,  # downstream
    "11553048#1": 1,  # ramp
}
#: the preserved network's real lane lengths (observation_map reads them)
LANE_LENGTHS = {
    "1191554636_0": 2644.22,
    "1191554636_1": 2645.67,
    "1487606179_0": 75.44,
    "1487606179_1": 75.44,
    "1487606179_2": 75.44,
    "1487606178_0": 477.89,
    "1487606178_1": 477.89,
    "11553048#1_0": 378.66,
}
PRE_MERGE_LANE = "1191554636_0"
CAV = "mainline.6"


class _FakePopen:
    """Records the launch; starts nothing (zero SUMO)."""

    def __init__(self):
        self.calls = []

    def __call__(self, cmd, creationflags=None):
        proc = SimpleNamespace(
            pid=4310 + len(self.calls),
            cmd=[str(part) for part in cmd],
            wait=lambda timeout=None: 0,
        )
        self.calls.append(proc)
        return proc


class _World:
    def __init__(self, vehicles):
        self.vehicles = dict(vehicles)  # id -> {lane, pos, speed}
        self.time = 240.0
        self.speed_modes = {}


class _VehicleSurface:
    def __init__(self, world):
        self._w = world

    def getIDList(self):
        return tuple(sorted(self._w.vehicles))

    def getLaneID(self, vid):
        return self._w.vehicles[vid]["lane"]

    def getLanePosition(self, vid):
        return self._w.vehicles[vid]["pos"]

    def getLength(self, vid):
        return 5.0

    def getSpeed(self, vid):
        return self._w.vehicles[vid]["speed"]

    def getAcceleration(self, vid):
        return 0.0

    def getLaneIndex(self, vid):
        return 0

    def getAllowedSpeed(self, vid):
        return 30.0

    def getSpeedMode(self, vid):
        return self._w.speed_modes.get(vid, 31)

    def setSpeedMode(self, vid, mode):
        self._w.speed_modes[vid] = int(mode)


class _LaneSurface:
    def __init__(self, world):
        self._w = world

    def getLength(self, lane_id):
        return LANE_LENGTHS.get(lane_id, 500.0)

    def getLastStepVehicleIDs(self, lane_id):
        return tuple(
            sorted(v for v, d in self._w.vehicles.items() if d["lane"] == lane_id)
        )


class _EdgeSurface:
    def __init__(self, counts):
        self._counts = dict(counts)

    def getLaneNumber(self, edge_id):
        return self._counts[edge_id]


class _SimSurface:
    def __init__(self, world):
        self._w = world

    def getTime(self):
        return self._w.time


class _FakeTraciMaster:
    """The traci MODULE surface the master harness uses (init registers the
    master; every surface reads the module's own world, exactly the real
    module's doSwitch semantics for the first connection).

    PL-04l: the real ``_launch`` now inits with ``doSwitch=False`` and uses
    ``getConnection('pl04')``; this fake returns ITSELF for the master label
    (the module IS the master connection here -- its surfaces read the
    module's own world), so the harness drives the same surfaces as before."""

    def __init__(self, vehicles, edge_counts=None, with_edge=True):
        self.world = _World(vehicles)
        self.init_calls = []
        self._master_label = None
        self._edge_counts = edge_counts or EDGE_LANE_COUNTS
        if with_edge:
            self.edge = _EdgeSurface(self._edge_counts)
        self.lane = _LaneSurface(self.world)
        self.vehicle = _VehicleSurface(self.world)
        self.simulation = _SimSurface(self.world)

    def init(self, port=None, host=None, numRetries=None, proc=None, label=None,
             doSwitch=True):
        self.init_calls.append(
            {"label": label, "port": int(port), "pid": getattr(proc, "pid", None)}
        )
        if self._master_label is None:
            self._master_label = label
        return (27, " FakeSUMO 1.27.1")

    def getConnection(self, label):
        # the master label resolves to the module itself (the master
        # connection in this fake); there is never a second label here
        return self

    def close(self, wait=True):
        pass


def make_harness(monkeypatch, tmp_path, traci, roster=(CAV,)):
    """A REAL RealMasterHarness over the faked Popen/traci (zero SUMO): the
    REAL SumoLocalSensor, CirclesController, MixedAssignmentSession and
    CommandOwnership all construct and run against the fake surfaces."""
    monkeypatch.setattr(subprocess, "Popen", _FakePopen())
    monkeypatch.setitem(sys.modules, "traci", traci)
    return RealMasterHarness(
        sumo_binary="fake-sumo-binary",
        scenario_cfg=tmp_path / "scenario.sumocfg",
        scenario_routes=[tmp_path / "scenario.sumocfg"],
        roster=roster,
        adapter_config=load_config(),
        seed=1729,
    )


class TestPL04jMergeMapResolved:
    """(a) start() resolves a non-empty merge map from the LIVE connection
    and observe() reads merge_observed off the REAL sensor frames."""

    def test_start_builds_the_registered_merge_map_from_the_connection(
        self, monkeypatch, tmp_path
    ):
        traci = _FakeTraciMaster({CAV: {"lane": PRE_MERGE_LANE, "pos": 2600.0, "speed": 20.0}})
        harness = make_harness(monkeypatch, tmp_path, traci)
        harness.start()

        # the registered 12-lane map: 2 pre_merge + 3 merge + 2 downstream
        # + 1 ramp + 4 internal junction lanes, pre_merge at 0.0
        merge_map = harness._merge_lane_distances
        assert len(merge_map) == 12
        assert merge_map[PRE_MERGE_LANE] == 0.0
        assert merge_map["1487606179_0"] == -75.44
        assert merge_map["1487606178_0"] == -(75.44 + 477.89)
        assert merge_map[":1780739017_1_0"] == 0.0  # internal -> merge
        assert merge_map[":13638027693_0_0"] == -75.44  # internal -> downstream
        # the receipt evidence travels on the master launch record
        assert harness.launch_record["merge_lane_map_source"] == "observation_map"
        assert harness.launch_record["merge_lane_map_size"] == 12

    def test_observe_yields_merge_observed_true_within_front_coverage(
        self, monkeypatch, tmp_path
    ):
        # 2600.0 on the 2644.22 m pre_merge lane: 44.22 m to the merge,
        # inside the 100 m sensor range -> the frame REACHES the merge
        traci = _FakeTraciMaster({CAV: {"lane": PRE_MERGE_LANE, "pos": 2600.0, "speed": 20.0}})
        harness = make_harness(monkeypatch, tmp_path, traci)
        harness.start()

        samples, _ = harness.observe()
        (meta,) = [sample.meta for sample in samples]
        assert meta["merge_observed"] is True
        assert meta["merge_observed_reason"] is None
        assert meta["frame_missing"] == []
        # the frame's own pair (contract 8.15 inputs): the 44.22 m remaining
        # distance is within the frame's 44.22 m front coverage
        frame = harness._control_write.sensor.capture(CAV, now_s=harness.now_s())
        assert frame.distance_to_merge_m == pytest.approx(44.22)
        assert frame.front_coverage_m == pytest.approx(44.22)
        assert frame.distance_to_merge_m <= frame.front_coverage_m

    def test_observe_false_with_reason_when_no_frame_reaches_the_merge(
        self, monkeypatch, tmp_path
    ):
        # 2000.0 on the 2644.22 m pre_merge lane: 644.22 m to the merge, far
        # beyond the 100 m sensor range -> no frame reaches it (fail closed)
        traci = _FakeTraciMaster({CAV: {"lane": PRE_MERGE_LANE, "pos": 2000.0, "speed": 20.0}})
        harness = make_harness(monkeypatch, tmp_path, traci)
        harness.start()

        samples, _ = harness.observe()
        (meta,) = [sample.meta for sample in samples]
        assert meta["merge_observed"] is False
        assert meta["merge_observed_reason"] == (
            "no live CAV sensor frame reaches the merge region"
        )

    def test_observe_false_with_reason_when_the_lane_is_unmapped(
        self, monkeypatch, tmp_path
    ):
        # a lane observation_map does not map: no merge-region input at all
        traci = _FakeTraciMaster({CAV: {"lane": "L0", "pos": 400.0, "speed": 20.0}})
        harness = make_harness(monkeypatch, tmp_path, traci)
        harness.start()

        samples, _ = harness.observe()
        (meta,) = [sample.meta for sample in samples]
        assert meta["merge_observed"] is False
        assert meta["merge_observed_reason"] == (
            "no RSU merge-region inputs in any sensor frame"
        )

    def test_injected_map_wins_over_the_resolution(self, monkeypatch, tmp_path):
        traci = _FakeTraciMaster({CAV: {"lane": PRE_MERGE_LANE, "pos": 2600.0, "speed": 20.0}})
        monkeypatch.setattr(subprocess, "Popen", _FakePopen())
        monkeypatch.setitem(sys.modules, "traci", traci)
        harness = RealMasterHarness(
            sumo_binary="fake-sumo-binary",
            scenario_cfg=tmp_path / "scenario.sumocfg",
            scenario_routes=[tmp_path / "scenario.sumocfg"],
            roster=(CAV,),
            adapter_config=load_config(),
            seed=1729,
            merge_lane_distances={"custom_lane": 7.5},
        )
        harness.start()
        assert harness._merge_lane_distances == {"custom_lane": 7.5}
        assert harness.launch_record["merge_lane_map_source"] == "injected"
        assert harness.launch_record["merge_lane_map_size"] == 1


class TestPL04jEmptyMapFailsLoudly:
    """(b) an empty or unresolvable merge map refuses the run with the
    NAMED error -- never a silent empty-map start."""

    def test_empty_resolved_map_raises_the_named_error(self, monkeypatch, tmp_path):
        import dataclasses

        # a network with NO junction-internal lanes: with zero lanes per
        # adapter edge, observation_map resolves a truly EMPTY map
        net = tmp_path / "plain.net.xml"
        net.write_text(
            "<net>"
            + "".join(
                f'<edge id="{edge}"><lane id="{edge}_0" length="100.0"/></edge>'
                for edge in EDGE_LANE_COUNTS
            )
            + "</net>",
            encoding="utf-8",
        )
        adapter = dataclasses.replace(load_config(), network_path=net)
        traci = _FakeTraciMaster(
            {CAV: {"lane": PRE_MERGE_LANE, "pos": 2600.0, "speed": 20.0}},
            edge_counts={edge: 0 for edge in EDGE_LANE_COUNTS},
        )
        monkeypatch.setattr(subprocess, "Popen", _FakePopen())
        monkeypatch.setitem(sys.modules, "traci", traci)
        harness = RealMasterHarness(
            sumo_binary="fake-sumo-binary",
            scenario_cfg=tmp_path / "scenario.sumocfg",
            scenario_routes=[tmp_path / "scenario.sumocfg"],
            roster=(CAV,),
            adapter_config=adapter,
            seed=1729,
        )
        with pytest.raises(MergeMapUnavailableError, match="EMPTY.*merge-region"):
            harness.start()

    def test_unresolvable_map_raises_the_named_error(self, monkeypatch, tmp_path):
        # no edge surface at all: observation_map cannot run on this api
        traci = _FakeTraciMaster(
            {CAV: {"lane": PRE_MERGE_LANE, "pos": 2600.0, "speed": 20.0}},
            with_edge=False,
        )
        harness = make_harness(monkeypatch, tmp_path, traci)
        with pytest.raises(MergeMapUnavailableError, match="could not be resolved"):
            harness.start()

    def test_empty_injection_falls_back_to_the_live_resolution(
        self, monkeypatch, tmp_path
    ):
        # an empty injection normalizes to "no injection": start() resolves
        # the map on the live connection instead of silently running empty
        traci = _FakeTraciMaster({CAV: {"lane": PRE_MERGE_LANE, "pos": 2600.0, "speed": 20.0}})
        monkeypatch.setattr(subprocess, "Popen", _FakePopen())
        monkeypatch.setitem(sys.modules, "traci", traci)
        harness = RealMasterHarness(
            sumo_binary="fake-sumo-binary",
            scenario_cfg=tmp_path / "scenario.sumocfg",
            scenario_routes=[tmp_path / "scenario.sumocfg"],
            roster=(CAV,),
            adapter_config=load_config(),
            seed=1729,
            merge_lane_distances={},
        )
        harness.start()
        assert len(harness._merge_lane_distances) == 12
        assert harness.launch_record["merge_lane_map_source"] == "observation_map"
