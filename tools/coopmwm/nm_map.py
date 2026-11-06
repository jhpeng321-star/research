"""N9 lane-drop map descriptor: STUB of the N1 map-descriptor API (c4/nm-abstract).

N9's move families read lane-drop geometry ONLY through :func:`lane_drop`. Until N1's
map abstraction lands, the values below are a stub; the integrator binds N1 with
:func:`bind` (one call, no family code changes). A bound provider must return a
mapping with the :class:`LaneDrop` fields; it is validated exactly like the stub.

Lane indices are WM lanes (the runtime ``rows['lane']`` convention): on the merge
map WM lane 0 is the leftmost mainline lane, 1 the right mainline lane and 2 the
ramp (traffic_shepherd.c4.geometry.LANE_TABLE). The G5 lane path (group_lane
envelope + env actuator) only issues changes to WM ``lane - 1`` and never into
lane 2, so a lane drop is G5-executable iff ``adjacent = closing - 1`` and
``far = adjacent - 1`` (``step = -1``).

Stubs:
- ``wz3to2`` (assets/research/newmap_candidates/wz3to2/MAP_SPEC.json sha256 386400b7...):
  SUMO lane 0 (rightmost) of edge ``up`` ends at x = 2700 m; SUMO lanes 1, 2 continue.
  Assumed WM mapping (same orientation as the merge map): SUMO 0 -> WM 2 (closing),
  SUMO 1 -> WM 1 (adjacent), SUMO 2 -> WM 0 (far). ASSUMED s coordinate: distance
  along the route from the start of edge ``up`` (newmap_probe EDGE_OFFSET up = 0), so
  the drop point is s = 2700 m. N1 decides the real origin; bind it.
- ``merge`` (the original merge map viewed as a lane drop): the ramp lane (WM 2)
  ends at the merge-zone end MERGE_S1_M = 75.44 m; WM 1 is adjacent, WM 0 far.
"""
from dataclasses import dataclass, fields
import math

STUB_SOURCE = 'N9 stub (N1 c4/nm-abstract not landed at 2026-11-06 build time)'
WZ3TO2_MAP_SPEC_SHA256 = '386400b78d27e50a376ee28a5d18d25a5baeb27fd4bb0436fb171e5ccafd4e0d'


@dataclass(frozen=True)
class LaneDrop:
    name: str
    n_lanes: int
    closing_lane: int        # WM lane that ends at drop_s_m
    adjacent_lane: int       # through lane next to the closing lane
    far_lane: object         # next through lane, or None (2 -> 1 drops)
    drop_s_m: float          # WM s coordinate where the closing lane ends
    source: str

    def __post_init__(self):
        lanes = [self.closing_lane, self.adjacent_lane] + ([] if self.far_lane is None else [self.far_lane])
        if (type(self.n_lanes) is not int or not 2 <= self.n_lanes <= 3
                or any(type(x) is not int or not 0 <= x < self.n_lanes for x in lanes)
                or len(set(lanes)) != len(lanes)):
            raise ValueError('lane drop needs distinct integer WM lanes inside n_lanes')
        if isinstance(self.drop_s_m, bool) or not math.isfinite(float(self.drop_s_m)):
            raise ValueError('lane drop point must be a finite WM s coordinate')
        if not isinstance(self.name, str) or not self.name or not isinstance(self.source, str):
            raise ValueError('lane drop name/source required')

    @property
    def through_lanes(self):
        return (self.adjacent_lane,) + (() if self.far_lane is None else (self.far_lane,))

    @property
    def g5_executable(self):
        """Every family lane change (closing->adjacent, adjacent->far) is a G5 ``lane-1`` change."""
        return (self.adjacent_lane == self.closing_lane-1
                and (self.far_lane is None or self.far_lane == self.adjacent_lane-1))

    def lane_role(self, lane):
        lane = int(lane)
        if lane == self.closing_lane:
            return 'closing'
        if lane == self.adjacent_lane:
            return 'adjacent'
        if self.far_lane is not None and lane == self.far_lane:
            return 'far'
        return None

    def to_drop(self, s_m):
        """Distance (m) still upstream of the drop point (negative = past it)."""
        return float(self.drop_s_m)-float(s_m)

    def to_dict(self):
        return {f.name: getattr(self, f.name) for f in fields(self)}


def _merge_stub():
    from traffic_shepherd.c4.geometry import MERGE_S1_M
    return LaneDrop('merge', 3, 2, 1, 0, float(MERGE_S1_M),
                    STUB_SOURCE+'; ramp lane as closing lane, traffic_shepherd.c4.geometry.MERGE_S1_M')


STUBS = {
    'wz3to2': lambda: LaneDrop('wz3to2', 3, 2, 1, 0, 2700.0,
                               STUB_SOURCE+'; MAP_SPEC sha256 '+WZ3TO2_MAP_SPEC_SHA256
                               + '; ASSUMED s = distance along route from edge up start'),
    'merge': _merge_stub,
}

_PROVIDER = None


def bind(provider):
    """Bind N1's descriptor provider: ``provider(name) -> mapping of LaneDrop fields``.

    Returns the previous provider (tests restore it). ``None`` restores the stub.
    """
    global _PROVIDER
    previous, _PROVIDER = _PROVIDER, provider
    return previous


def lane_drop(name):
    """The registered lane-drop descriptor for ``name`` (N1 provider if bound, else stub)."""
    if _PROVIDER is not None:
        value = _PROVIDER(name)
        if isinstance(value, LaneDrop):
            return value
        if not isinstance(value, dict):
            raise ValueError('bound map provider must return a LaneDrop or its field mapping')
        return LaneDrop(**{f.name: value[f.name] for f in fields(LaneDrop)})
    if name not in STUBS:
        raise ValueError(f'unregistered lane-drop map {name!r}')
    return STUBS[name]()
