from types import SimpleNamespace

import pytest

from traffic_shepherd.config import load_config
from traffic_shepherd.controller_factory import create_controller, create_pi_session
from traffic_shepherd.damping import CirclesController
from traffic_shepherd.ownership import CommandOwnership
from traffic_shepherd.sensing import NeighborMeasurement, SensorFrame


class Vehicle:
    def __init__(self):
        self.writes = []
        self.fail = None
    def getSpeedMode(self, vehicle): return 31
    def slowDown(self, vehicle, target, duration):
        if vehicle == self.fail:
            raise RuntimeError("writer failure")
        self.writes.append(("slowDown", vehicle, target, duration))
    def setSpeed(self, vehicle, target): self.writes.append(("setSpeed", vehicle, target))


def frames(now):
    return {vehicle: SensorFrame(vehicle, now, 10., 0., 30., 0,
        front=NeighborMeasurement(20., 0.), front_coverage_m=100., rear_coverage_m=100.)
        for vehicle in ("cav-a", "cav-b")}


def owner_session():
    config = load_config()
    session = create_pi_session(config, {"cav-a", "cav-b"}, desired_speed_mps=31.29)
    vehicle = Vehicle()
    api = SimpleNamespace(vehicle=vehicle)
    owner = CommandOwnership(api, {"cav-a", "cav-b"}, session.adapter)
    return session, owner, vehicle


def test_factory_defaults_and_fixed_u_contract():
    config = load_config()
    assert isinstance(create_controller("native", config)[0], CirclesController)
    assert isinstance(create_controller("damping", config)[0], CirclesController)
    session = create_pi_session(config, {"cav-a"}, desired_speed_mps=31.29)
    assert session.identity["u_provenance"] == "fixed_registered_setpoint"
    with pytest.raises(ValueError):
        create_pi_session(config, {"cav-a"}, desired_speed_mps=31.28)


def test_real_owner_batch_and_bounded_evidence_drain():
    session, owner, vehicle = owner_session()
    records = session.step(owner, frames(1.), now_s=1., live_ids={"cav-a", "cav-b"})
    assert [record.vehicle_id for record in records] == ["cav-a", "cav-b"]
    evidence = session.pending_evidence()
    assert len(evidence) == 6
    assert {row["kind"] for row in evidence} == {"pi_source_evidence", "pi_owner_receipt"}
    persisted = []
    session.flush_evidence(persisted.append)
    assert session.pending_evidence() == []


def test_partial_owner_failure_keeps_first_ack_and_stops_session():
    session, owner, vehicle = owner_session()
    vehicle.fail = "cav-b"
    with pytest.raises(Exception):
        session.step(owner, frames(1.), now_s=1., live_ids={"cav-a", "cav-b"})
    rows = session.pending_evidence()
    assert any(row["kind"] == "pi_owner_receipt" and row["status"] == "committed_raw_proposal"
               and row["vehicle_id"] == "cav-a" for row in rows)
    with pytest.raises(Exception):
        session.step(owner, frames(2.), now_s=2., live_ids={"cav-a", "cav-b"})


def test_evidence_flush_failure_does_not_advance_or_drop_rows():
    session, owner, _ = owner_session()
    session.step(owner, frames(1.), now_s=1., live_ids={"cav-a", "cav-b"})
    before = session.pending_evidence()
    def fail(_row):
        raise RuntimeError("sink failure")
    with pytest.raises(RuntimeError, match="sink failure"):
        session.flush_evidence(fail)
    assert session.pending_evidence() == before
