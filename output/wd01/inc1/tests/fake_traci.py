"""A minimal fake traci/SUMO api for the INC1 unit tests (ZERO SUMO).

Reads/writes plain dicts; every method the runner, the obstacle placer and
the frozen ``CommandOwnership`` call is implemented with real names.  No
traci import and no subprocess anywhere.
"""
from __future__ import annotations


class FakeTraci:
    def __init__(self, *, time_s: float = 300.0, physics_dt_s: float = 0.5):
        self._time = float(time_s)
        self.physics_dt_s = float(physics_dt_s)
        self.vehicles: dict = {}
        self.lanes: dict = {}
        self.speed_modes: dict = {}
        self.issued: list = []
        self.released: list = []
        self.added: list = []
        self.stops: list = []
        self.moves: list = []
        self.saved_states: list = []
        self.load_calls: list = []
        self.departed_ids: tuple = ()
        self._collisions: list = []
        self.vehicle = self._Vehicle(self)
        self.lane = self._Lane(self)
        self.simulation = self._Simulation(self)

    # -- state helpers used by the tests --------------------------------
    def set_lane(self, lane_id, *, length=3000.0, speed=30.0):
        self.lanes[lane_id] = {"length": float(length), "speed": float(speed)}

    def set_vehicle(self, vid, *, lane_id, position, speed=0.0, accel=0.0,
                    length=5.0, type_id="normal_car", allowed=None):
        self.vehicles[vid] = {
            "lane_id": lane_id, "position": float(position),
            "speed": float(speed), "accel": float(accel),
            "length": float(length), "type_id": type_id,
            "allowed": (self.lanes[lane_id]["speed"] if allowed is None
                        else float(allowed)),
        }
        self.speed_modes.setdefault(vid, 31)

    def drop_vehicle(self, vid):
        self.vehicles.pop(vid, None)

    class _Vehicle:
        def __init__(self, api):
            self._api = api

        def getIDList(self):
            return sorted(self._api.vehicles)

        def getLaneID(self, v):
            return self._api.vehicles[v]["lane_id"]

        def getLanePosition(self, v):
            return self._api.vehicles[v]["position"]

        def getSpeed(self, v):
            return self._api.vehicles[v]["speed"]

        def getAcceleration(self, v):
            return self._api.vehicles[v]["accel"]

        def getRoadID(self, v):
            return self._api.vehicles[v]["lane_id"].rsplit("_", 1)[0]

        def getLength(self, v):
            return self._api.vehicles[v]["length"]

        def getAllowedSpeed(self, v):
            return self._api.vehicles[v]["allowed"]

        def getTypeID(self, v):
            return self._api.vehicles[v]["type_id"]

        def getLastStepVehicleIDs(self, lane_id):
            return sorted(v for v, s in self._api.vehicles.items()
                          if s["lane_id"] == lane_id)

        def getSpeedMode(self, v):
            return self._api.speed_modes.get(v, 31)

        def setSpeedMode(self, v, mode):
            self._api.speed_modes[v] = int(mode)

        def add(self, vid, route_id, type_id=None, departPos="0",
                departSpeed="0", **kwargs):
            self._api.added.append((vid, route_id, type_id, departPos))
            lane_id = kwargs.get("lane_id")
            self._api.vehicles[vid] = {
                "lane_id": lane_id, "position": float(departPos),
                "speed": float(departSpeed or 0.0), "accel": 0.0,
                "length": 5.0, "type_id": type_id or "normal_car",
                "allowed": 30.0,
            }
            self._api.speed_modes.setdefault(vid, 31)

        def setStop(self, vid, lane_id=None, pos=None, duration=None):
            self._api.stops.append((vid, lane_id, pos, duration))

        def moveTo(self, vid, lane_id, pos):
            self._api.moves.append((vid, lane_id, pos))
            if vid in self._api.vehicles:
                self._api.vehicles[vid]["lane_id"] = lane_id
                self._api.vehicles[vid]["position"] = float(pos)

        def setSpeed(self, vid, speed):
            if float(speed) == -1.0:
                self._api.released.append(vid)
                return
            self._api.vehicles[vid]["speed"] = float(speed)

        def slowDown(self, vid, target, dt):
            self._api.issued.append((vid, float(target), float(dt)))
            self._api.vehicles[vid]["speed"] = float(target)

    class _Lane:
        def __init__(self, api):
            self._api = api

        def getLength(self, lane_id):
            return self._api.lanes[lane_id]["length"]

        def getMaxSpeed(self, lane_id):
            return self._api.lanes[lane_id]["speed"]

    class _Simulation:
        def __init__(self, api):
            self._api = api

        def getTime(self):
            return float(self._api._time)

        def simulationStep(self):
            self._api._time += self._api.physics_dt_s

        def saveState(self, path):
            self._api.saved_states.append(str(path))
            from pathlib import Path
            Path(path).write_text("<state t=\"%s\"/>\n" % self._api._time,
                                  encoding="utf-8")

        def loadState(self, path):
            self._api.load_calls.append(str(path))
            return True

        def getCollisions(self):
            return list(self._api._collisions)

        def getStartingTeleportNumber(self):
            return 0

        def getEndingTeleportNumber(self):
            return 0

        def getDepartedIDList(self):
            return list(self._api.departed_ids)
