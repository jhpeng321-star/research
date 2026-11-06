"""N9 lane-drop geometry, read from the N1 map descriptor (tools.coopmwm.map_descriptor).

N9's move families read lane-drop geometry ONLY through :func:`lane_drop`. Source order:

1. a test provider bound with :func:`bind` (tests only);
2. the N1 process map ``map_descriptor.current()`` when its name matches: a lane-drop
   descriptor gives ``closing_lane``/``adjacent_lane``/``drop_s_m`` from
   ``desc.bottleneck`` (WM frame: closing lane 2, adjacent 1, far 0, drop at s = 75.44 m);
   the original map (``merge_original``, alias ``merge``) is the ramp lane viewed as a
   closing lane ending at the frame merge-zone end;
3. otherwise fail closed (a family registered for a map the process is not bound to).

``STUBS`` are test fixtures only (same WM frame as N1: drop at s = 75.44 m).

Lane indices are WM lanes (0 = main left, 1 = main right, 2 = merge/closing lane). A
lane drop is G5-executable iff every family lane change is a WM ``lane - 1`` change
(``adjacent = closing - 1``, ``far = adjacent - 1``). ``change_zone_s_m`` is where a
closing-lane vehicle is physically next to the adjacent lane (merge map: the merge zone;
lane drop: the whole approach edge up to the drop). ``guard_points_s_m`` are the edge
boundaries near which discretionary mainline changes are not issued (G5 LaneRules guard).
"""
from dataclasses import dataclass, fields
import math

STUB_SOURCE = 'N9 test stub (WM frame as N1 map_descriptor)'
MERGE_ALIASES = ('merge', 'merge_original')


@dataclass(frozen=True)
class LaneDrop:
    name: str
    n_lanes: int
    closing_lane: int        # WM lane that ends at drop_s_m
    adjacent_lane: int       # through lane next to the closing lane
    far_lane: object         # next through lane, or None (2 -> 1 drops)
    drop_s_m: float          # WM s coordinate where the closing lane ends
    change_zone_s_m: tuple   # (s0, s1): closing lane physically adjacent to the adjacent lane
    guard_points_s_m: tuple  # edge boundaries guarding discretionary mainline changes
    source: str

    def __post_init__(self):
        lanes = [self.closing_lane, self.adjacent_lane] + ([] if self.far_lane is None else [self.far_lane])
        if (type(self.n_lanes) is not int or not 2 <= self.n_lanes <= 3
                or any(type(x) is not int or not 0 <= x < self.n_lanes for x in lanes)
                or len(set(lanes)) != len(lanes)):
            raise ValueError('lane drop needs distinct integer WM lanes inside n_lanes')
        if isinstance(self.drop_s_m, bool) or not math.isfinite(float(self.drop_s_m)):
            raise ValueError('lane drop point must be a finite WM s coordinate')
        zone = tuple(float(x) for x in self.change_zone_s_m)
        if len(zone) != 2 or not zone[0] < zone[1] or zone[1] != float(self.drop_s_m) \
                or not math.isfinite(zone[1]):
            raise ValueError('change zone must be an ordered (s0, drop_s_m) pair')
        if any(not math.isfinite(float(x)) for x in self.guard_points_s_m):
            raise ValueError('guard points must be finite')
        if not isinstance(self.name, str) or not self.name or not isinstance(self.source, str):
            raise ValueError('lane drop name/source required')

    @property
    def through_lanes(self):
        return (self.adjacent_lane,) + (() if self.far_lane is None else (self.far_lane,))

    @property
    def g5_executable(self):
        """Every family lane change (closing->adjacent, adjacent->far) is a WM ``lane-1`` change."""
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


def from_descriptor(desc):
    """LaneDrop of an N1 ``MapDescriptor`` (lane drop or the original merge map)."""
    frame = desc.frame
    main_left, main_right = int(frame.main_lanes[0]), int(frame.main_lanes[1])
    if desc.kind == 'lane_drop':
        neck = desc.bottleneck or {}
        closing, adjacent = int(neck['closing_lane']), int(neck['adjacent_lane'])
        through = [int(x) for x in neck['through_lanes']]
        far = next((x for x in through if x != adjacent), None)
        drop = float(neck['drop_s_m'])
        # The approach edge starts at x = 0: s = x - anchor_x_m (N1 projection).
        start = -float(neck['anchor_x_m'])
        return LaneDrop(desc.name, int(frame.n_lanes), closing, adjacent, far, drop, (start, drop), (drop,),
                        f'N1 map_descriptor {desc.name} digest {desc.digest}')
    s0, s1 = (float(x) for x in frame.merge_zone_s_m)
    return LaneDrop(desc.name, int(frame.n_lanes), int(frame.merge_lane), main_right, main_left, s1,
                    (s0, s1), (s0, s1), f'N1 map_descriptor {desc.name} (ramp lane as the closing lane)')


def _stub_merge():
    from .map_descriptor import ORIGINAL
    out = from_descriptor(ORIGINAL)
    return LaneDrop(**dict(out.to_dict(), name='merge', source=STUB_SOURCE+'; '+out.source))


STUBS = {
    # Test fixtures only. Same WM frame as N1's wz3to2 projection (up_0 end -> s = 75.44 m;
    # up_0 lane end x = 2696 m, anchor 2620.56 m).
    'wz3to2': lambda: LaneDrop('wz3to2', 3, 2, 1, 0, 75.44, (-2620.56, 75.44), (75.44,),
                               STUB_SOURCE+'; wz3to2 projection'),
    'merge': _stub_merge,
}

_PROVIDER = None


def bind(provider):
    """TEST hook: ``provider(name) -> LaneDrop | mapping``; ``None`` restores N1 resolution.

    Returns the previous provider."""
    global _PROVIDER
    previous, _PROVIDER = _PROVIDER, provider
    return previous


def stub(name):
    """Test fixture geometry (never used by production resolution)."""
    if name not in STUBS:
        raise ValueError(f'no lane-drop test stub for map {name!r}')
    return STUBS[name]()


def lane_drop(name):
    """Lane-drop geometry for ``name`` from the bound N1 process map (fail closed otherwise)."""
    if _PROVIDER is not None:
        value = _PROVIDER(name)
        if isinstance(value, LaneDrop):
            return value
        if not isinstance(value, dict):
            raise ValueError('bound map provider must return a LaneDrop or its field mapping')
        return LaneDrop(**{f.name: value[f.name] for f in fields(LaneDrop)})
    from . import map_descriptor
    desc = map_descriptor.current()
    if desc.name == name or (desc.is_original and name in MERGE_ALIASES):
        out = from_descriptor(desc)
        return out if desc.name == name else LaneDrop(**dict(out.to_dict(), name=name))
    raise ValueError(f'lane-drop map {name!r} is not the process map {desc.name!r} '
                     f'(bind it via {map_descriptor.ENV})')
