"""Memoize identical TraCI sensor reads within one immutable control tick.

The existing sensor determines visibility independently for every CAV. Cached
simulator fields never cross the SensorFrame boundary. There is no reuse across
physics ticks, no extrapolation and no change to the adopted sensor semantics.
"""
from types import SimpleNamespace

from .sensing import SumoLocalSensor


class _ReadCache:
    def __init__(self, domain):
        self._domain, self._values = domain, {}
        self.requests = self.queries = 0

    def __getattr__(self, name):
        if not name.startswith("get"):
            raise AttributeError("sensor cache exposes read queries only")
        function = getattr(self._domain, name)

        def query(*args):
            self.requests += 1
            key = name, args
            if key not in self._values:
                self._values[key] = function(*args)
                self.queries += 1
            return self._values[key]
        return query


def capture_batch(api, cav_ids, *, now_s, config=None, lane_end_distance_to_merge=None):
    """Exactly the old per-CAV perception with fewer redundant transport calls."""
    vehicle, lane = _ReadCache(api.vehicle), _ReadCache(api.lane)
    sensor = SumoLocalSensor(SimpleNamespace(vehicle=vehicle, lane=lane), config,
                             lane_end_distance_to_merge)
    frames, errors = {}, {}
    for cav in sorted(cav_ids):
        try:
            frames[cav] = sensor.capture(cav, now_s=now_s)
        except ValueError as exc:
            errors[cav] = str(exc)
    return frames, errors, dict(sensor_read_requests=vehicle.requests+lane.requests,
                               sensor_transport_queries=vehicle.queries+lane.queries)
