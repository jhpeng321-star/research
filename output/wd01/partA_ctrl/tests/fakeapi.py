"""A minimal fake SUMO api sufficient for the frozen sensor + ownership.

Reads/writes plain dicts; NO simulation, NO traci import.  Per control
tick the test scripts the whole traffic state (positions/speeds on
lanes); the real SumoLocalSensor / CommandOwnership run against this
view exactly as they would against traci.
"""
from __future__ import annotations


class FakeApi:
    def __init__(self):
        self.vehicles: dict[str, dict] = {}   # vid -> state dict
        self.lanes: dict[str, dict] = {}      # lane_id -> {length, speed}
        self.issued: list[tuple] = []         # slowDown calls
        self.released: list[str] = []
        self.vehicle = self._VehicleView(self)
        self.lane = self._LaneView(self)

    def set_lane(self, lane_id, *, length, speed):
        self.lanes[lane_id] = dict(length=length, speed=speed)

    def set_vehicle(self, vid, *, lane_id, position, speed, accel=0.0,
                    length=5.0, allowed=None, lane_index=0):
        lane = self.lanes[lane_id]
        self.vehicles[vid] = dict(lane_id=lane_id, position=position,
                                  speed=speed, accel=accel, length=length,
                                  allowed=lane["speed"] if allowed is None else allowed,
                                  lane_index=lane_index)

    def drop_vehicle(self, vid):
        self.vehicles.pop(vid, None)

    class _VehicleView:
        def __init__(self, api):
            self._api = api

        def getLaneID(self, vid):
            return self._api.vehicles[vid]["lane_id"]

        def getLanePosition(self, vid):
            return self._api.vehicles[vid]["position"]

        def getLength(self, vid):
            return self._api.vehicles[vid]["length"]

        def getSpeed(self, vid):
            return self._api.vehicles[vid]["speed"]

        def getAcceleration(self, vid):
            return self._api.vehicles[vid]["accel"]

        def getAllowedSpeed(self, vid):
            return self._api.vehicles[vid]["allowed"]

        def getLaneIndex(self, vid):
            return self._api.vehicles[vid]["lane_index"]

        def getSpeedMode(self, vid):
            return 31

        def setSpeedMode(self, vid, mode):
            raise AssertionError("speedMode 31 must never be reconfigured")

        def slowDown(self, vid, target, dt):
            self._api.issued.append((vid, float(target), float(dt)))

        def setSpeed(self, vid, speed):
            if float(speed) == -1.0:
                self._api.released.append(vid)

    class _LaneView:
        def __init__(self, api):
            self._api = api

        def getLength(self, lane_id):
            return self._api.lanes[lane_id]["length"]

        def getLastStepVehicleIDs(self, lane_id):
            on_lane = [v for v, s in self._api.vehicles.items()
                       if s["lane_id"] == lane_id]
            return sorted(on_lane)
