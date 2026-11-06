"""N2 (c4/nm-moves): lane-drop observations, ``closing_pair`` / ``abreast_wall`` moves and their
V2V initiate/join protocol for the wz3to2 work-zone map. Observable-only and DEFAULT OFF.

Evidence (runs/active/C4-V7FT-20261017/NEWMAP_RESEARCH.md, 2026-11-06): on wz3to2 (3 lanes,
the closing lane ends at the drop) abreast 3-CAV walls gave 8/34 admitted cooperative
situations; in both mechanism checks ONE 2-CAV pair that includes the closing-lane CAV
reproduced the whole group effect (a moving bottleneck across the closing lane plus a through
lane that lets the drop queue drain, then platoon release). Gap-maker + merger pairs: 0/12.

Map API (N1, c4/nm-abstract e512b059, map_descriptor.MapDescriptor): ONE road frame, the WM
frame of the runtime observation rows (``s_m``, ``lane``). The view needs only
    distance_to_drop(s_m) -> float   lane-drop point minus s (negative past the drop)
    lane_ids              -> ints    upstream lanes (frame lane ids)
    dropped_lane          -> int     the closing lane
``MapView.from_descriptor`` reads N1's descriptor (``bottleneck['drop_s_m']``,
``bottleneck['closing_lane']``, ``range(frame.n_lanes)``; wz3to2: drop at s = 75.44, closing
lane = WM lane 2), a stub exposing the three items, or the plain mapping ``{name,
lane_drop_s_m, lane_ids, dropped_lane}`` that the registered config stores; ``from_map_spec``
goes through ``map_descriptor.load`` only. ``adjacent_lane`` = the through lane bordering the
dropped lane (the merge target). Nothing reads MAP_SPEC/net files at runtime.

(1) ``features``: one block per CAV from its OWN sensing packet at t and the selected V2V
    partners' direct-sensing packets published at t-1 (exactly the merge_features packet
    authorization: valid own packet required, nearest V2V_K selected partners within
    V2V_RANGE_M, no forwarding / RSU field / hidden or driver parameters). Distance to the
    drop, own lane role, closing-lane occupancy/queue over the registered ROI before the drop
    (coverage fraction, report age, missing bits) and the adjacent-lane front/rear gaps at the
    ego position (own packet only; open-lane flags). It enters the trigger context and the
    cooperation-layer residual as ZERO-initialised additive columns; the frozen MAPPO
    observation (x / v2v / rsu) is untouched.
(2) ``moves``: joint move-library entries executed through the existing MoveExecutor
    commitment and the layer's committed-offset column (the single writer):
    * ``closing_pair`` = (closing-lane CAV, adjacent-lane CAV), |dx| <= 100 m (variants pair_g,
      pair_h, pair_t; pair_t releases both when the closing-lane anchor is 150 m before the drop);
    * ``abreast_wall`` = one CAV per lane, spread <= 200 m (wall_g, wall_h);
    * ``group_hold`` = 2-3 CAVs in any lanes within 300 m (the R_random hold procedure).
    All values are the NEWMAP2 registration (VARIANTS / R_SETS, zone 400..1800 m before the
    drop). Law per member and second (NEWMAP2 rule): a = clip((v_w - v)/1 s, -dmax, amax); a
    member releases at its own drop crossing, at the end of the hold, or by the pair_t anchor.
    Runtime offset = a - native - MAPPO mean, so the executed request is a (+ exploration noise
    + residual); deterministic + zero residual = the law exactly. Optional zip/gap_dv (owner
    example, never drawn): the closing-lane CAV attempts ONE lane change at the last hold second
    through the G5 env path (CAVs only, own-sensing gap recheck, never forced).
    Executable representations (G5 convention): ``plan()`` with recorded member speeds and
    positions (profile == the law exactly) and ``plan_from_mined_row`` for the verified NEWMAP2
    / PLANGEN / PLANGEN2 rows (the row's executed profile, per-member release el, SUMO lanes
    mapped to WM lanes by the N1 descriptor); ``from_plan`` binds both exactly.
(3) ``v2v``: initiate/join protocol for these moves (reuses the W1 v2v_wall.Protocol ledger):
    a free CAV initiates a closing_pair or abreast_wall session; the initiator broadcasts the
    schedule and the lanes already covered at t, visible at t+1 within ``range_m``
    (registered 100..300 m, default DEFAULT_V2V_RANGE_M); a free CAV in an uncovered role lane
    receives a ``lane_drop_join`` option (event decision on first receipt). Ownership: one
    move per CAV; session cap = min(member_cap, 2 for closing_pair / number of lanes).
    The optional RSU initiator (``lane_drop_rsu``) opens sessions on the same ledger.

Config (absent/False = OFF, byte-identical):
    lane_drop = dict(enabled=True, rule=RULE, provenance=str, map={name, lane_drop_s_m,
                     lane_ids, dropped_lane}, features=bool, moves=bool,
                     [v2v=dict(range_m=100..300, member_cap=2..4)], [rsu=dict(...)])
B and R read the same keys (common track2 config); A never builds EpisodePolicy.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from itertools import combinations, product
import math

import numpy as np
import torch

from . import moves, v2v_wall
from .merge_features import (CONTROL_DT_S, N_LANES, QUEUE_SPEED_MPS, SENSOR_RANGE_M,
                             _local_rows, _valid_report)
from .v2v_obs import V2V_K, V2V_RANGE_M

RULE = 'n2-lane-drop-moves-v1'
PROVENANCE = ('N2 build (c4/nm-moves; MAIN 2026-11-06 newmap_spec N2; evidence '
              'runs/active/C4-V7FT-20261017/NEWMAP_RESEARCH.md)')
JOINT_FAMILIES = ('closing_pair', 'abreast_wall', 'group_hold')
PROTOCOL_FAMILIES = ('lane_drop_initiate', 'lane_drop_join')
KINDS = ('closing_pair', 'abreast_wall')   # protocol / RSU kinds (group_hold is joint only)
INITIATE_SLOTS = tuple(f'lane_drop_initiate:{kind}' for kind in KINDS)   # F2 slate slots per kind
RSU_ID = 'RSU'

V2V_RANGE_WINDOW_M = (100., 300.)
DEFAULT_V2V_RANGE_M = 200.     # W1 caveat: 100 m reaches 2 of 3 partners at 60 m spacing
# Move values are the NEWMAP2 registration (runs/active/C4-V7FT-20261017/NEWMAP2_REPORT.md
# section 2 'Moments'/'Moves', NEWMAP2-20261106; reused unchanged by PLANGEN and PLANGEN2).
ACCEL_MAX_MPS2 = 1.0                  # amax 1.0 (every variant)
VARIANTS = dict(
    pair_g=dict(family='closing_pair', target_mps=22., hold=120, depth=1.0, anchor=False),
    pair_h=dict(family='closing_pair', target_mps=15., hold=90, depth=1.5, anchor=False),
    pair_t=dict(family='closing_pair', target_mps=22., hold=150, depth=1.0, anchor=True),
    wall_g=dict(family='abreast_wall', target_mps=22., hold=120, depth=1.0, anchor=False),
    wall_h=dict(family='abreast_wall', target_mps=15., hold=90, depth=1.5, anchor=False))
R_SETS = dict(target_mps=(15., 18., 22.), hold=(90, 120, 180), depth=(1.0, 1.5))   # R0/R1 random sets
TARGET_WINDOW_MPS = (15., 22.)        # v_w over every registered variant / R set
BRAKE_WINDOW_MPS2 = (1.0, 1.5)        # dmax over every registered variant / R set
HOLD_WINDOW_S = (int(moves._HOLD_LO), 180)   # longest registered hold 180 s; lower bound = the
#                                       registered minimum REMAINING hold of a join / relabel
ONSET_S = 0                           # members act from el = 0 (start_s 0)
ONSET_WINDOW_S = (int(moves._ONSET_LO), int(moves._ONSET_HI))   # remaining onset of a join / relabel
ANCHOR_RELEASE_M = 150.               # pair_t: until the closing-lane CAV is 150 m before the drop
ZONE_WINDOW_M = (400., 1800.)         # moment zone x in [X_B - 1800, X_B - 400]
PAIR_DX_M = 100.                      # pair: lane-0 + lane-1 CAV, |dx| <= 100 m
WALL_SPREAD_M = 200.                  # wall: one CAV per lane, spread <= 200 m
GROUP_SPREAD_M = 300.                 # R0/R1: 2-3 zone CAVs within 300 m of each other
GROUP_SIZE = (2, 3)
GAP_DV_WINDOW_MPS = (0., 4.5)         # owner-example zip-gap extension: never drawn (needs MAIN registration)
CLOSING_ROI_M = 500.                  # observation ROI (probe gap-pair merger window, NEWMAP_RESEARCH)
LANE_DROP_REGISTERED = dict(
    variants=VARIANTS, r_sets=R_SETS, target_window_mps=TARGET_WINDOW_MPS,
    brake_window_mps2=BRAKE_WINDOW_MPS2, accel_max_mps2=ACCEL_MAX_MPS2, hold_window_s=HOLD_WINDOW_S,
    onset_s=ONSET_S, onset_window_s=ONSET_WINDOW_S, release='own drop crossing (x >= X_B) or hold end',
    anchor_release_m=ANCHOR_RELEASE_M, zone_window_m=ZONE_WINDOW_M, pair_dx_m=PAIR_DX_M,
    wall_spread_m=WALL_SPREAD_M, group_spread_m=GROUP_SPREAD_M, group_size=GROUP_SIZE,
    closing_roi_m=CLOSING_ROI_M, queue_speed_mps=QUEUE_SPEED_MPS,
    v2v_range_window_m=V2V_RANGE_WINDOW_M, default_v2v_range_m=DEFAULT_V2V_RANGE_M,
    provenance='NEWMAP2_REPORT.md (NEWMAP2-20261106) moments/moves/variants; PLANGEN/PLANGEN2 reuse')


def _finite(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float, np.floating, np.integer))
            and math.isfinite(float(value)))


# ---------------------------------------------------------------- map view

@dataclass(frozen=True)
class MapView:
    """Lane-drop geometry in the runtime road frame (see the module docstring)."""
    name: str
    lane_drop_s_m: float
    lane_ids: tuple
    dropped_lane: int

    def __post_init__(self):
        lanes = tuple(self.lane_ids)
        object.__setattr__(self, 'lane_ids', lanes)
        if not isinstance(self.name, str) or not self.name:
            raise ValueError('map view needs a name')
        if not _finite(self.lane_drop_s_m):
            raise ValueError('lane_drop_s_m must be finite')
        object.__setattr__(self, 'lane_drop_s_m', float(self.lane_drop_s_m))
        if (len(lanes) < 2 or len(set(lanes)) != len(lanes)
                or any(type(lane) is not int or lane not in range(N_LANES) for lane in lanes)):
            raise ValueError(f'lane_ids must be 2..{N_LANES} distinct frame lane ids')
        if type(self.dropped_lane) is not int or self.dropped_lane not in lanes:
            raise ValueError('dropped_lane must be one of lane_ids')

    @property
    def through_lanes(self):
        return tuple(lane for lane in self.lane_ids if lane != self.dropped_lane)

    @property
    def adjacent_lane(self):
        """Through lane bordering the dropped lane (the merge target)."""
        return min(self.through_lanes, key=lambda lane: (abs(lane-self.dropped_lane), lane))

    def distance_to_drop(self, s_m):
        return self.lane_drop_s_m-float(s_m)

    def to_dict(self):
        return dict(name=self.name, lane_drop_s_m=self.lane_drop_s_m, lane_ids=list(self.lane_ids),
                    dropped_lane=self.dropped_lane)

    @classmethod
    def from_descriptor(cls, desc):
        """N1 descriptor / mapping / stub -> MapView in the runtime (WM) frame; fail closed.

        N1 ``map_descriptor.MapDescriptor`` (c4/nm-abstract e512b059): drop point
        ``bottleneck['drop_s_m']`` (wz3to2: 75.44 = the frame's merge-zone end), closing lane
        ``bottleneck['closing_lane']`` (WM lane 2), lanes ``range(frame.n_lanes)``; its
        ``bottleneck['adjacent_lane']`` must agree with the derived merge target. A descriptor
        without a lane-drop bottleneck (the original merge map) is refused."""
        if isinstance(desc, MapView):
            return desc
        if isinstance(desc, dict):
            if set(desc)-{'name', 'lane_drop_s_m', 'lane_ids', 'dropped_lane'}:
                raise ValueError('unregistered map descriptor keys')
            return cls(desc.get('name', ''), desc.get('lane_drop_s_m'), tuple(desc.get('lane_ids') or ()),
                       desc.get('dropped_lane'))
        name = getattr(desc, 'name', None)
        name = name if isinstance(name, str) else ''
        if hasattr(desc, 'bottleneck') and hasattr(desc, 'frame'):
            neck = desc.bottleneck
            if not isinstance(neck, dict) or 'drop_s_m' not in neck or 'closing_lane' not in neck:
                raise ValueError(f'map {name!r} has no lane-drop bottleneck in the model frame')
            view = cls(name, neck['drop_s_m'], tuple(range(int(desc.frame.n_lanes))), neck['closing_lane'])
            if neck.get('adjacent_lane', view.adjacent_lane) != view.adjacent_lane:
                raise ValueError('descriptor adjacent_lane disagrees with the derived merge target')
            if 'through_lanes' in neck and sorted(neck['through_lanes']) != list(view.through_lanes):
                raise ValueError('descriptor through_lanes disagree with the frame lanes')
            return view
        drop = getattr(desc, 'lane_drop_s_m', None)
        if drop is None and callable(getattr(desc, 'distance_to_drop', None)):
            drop = float(desc.distance_to_drop(0.))
            if not math.isclose(desc.distance_to_drop(1000.), drop-1000., abs_tol=1e-9):
                raise ValueError('distance_to_drop must be linear in the road frame s')
        lanes = getattr(desc, 'lane_ids', None)
        return cls(name, drop, tuple(lanes() if callable(lanes) else (lanes or ())),
                   getattr(desc, 'dropped_lane', None))


def from_map_spec(path):
    """MAP_SPEC.json (or its directory) -> registered config mapping, ONLY through N1's
    ``map_descriptor.load`` so the lane-drop view lives in exactly one (the WM) frame."""
    from .map_descriptor import load
    return MapView.from_descriptor(load(path)).to_dict()


# ---------------------------------------------------------------- settings

def settings(cfg):
    """Resolved options, or None when ``lane_drop`` is absent/False (default OFF)."""
    raw = (cfg or {}).get('lane_drop', False)
    if raw is False or raw is None:
        return None
    keys, optional = ('enabled', 'rule', 'provenance', 'map'), ('features', 'moves', 'v2v', 'rsu')
    if (not isinstance(raw, dict) or not set(keys) <= set(raw) or set(raw)-set(keys)-set(optional)
            or raw['enabled'] is not True or raw['rule'] != RULE
            or not isinstance(raw['provenance'], str) or not raw['provenance'].strip()):
        raise ValueError('lane_drop requires enabled=True, the registered rule, provenance and map')
    view = MapView.from_descriptor(raw['map'])
    out = dict(rule=RULE, provenance=raw['provenance'], view=view, v2v=None, rsu=None)
    for key in ('features', 'moves'):
        value = raw.get(key, False)
        if type(value) is not bool:
            raise ValueError(f'lane_drop {key} must be bool')
        out[key] = value
    if not (out['features'] or out['moves']):
        raise ValueError('lane_drop enabled with neither features nor moves')
    v2v = raw.get('v2v')
    if v2v not in (None, False):
        if not out['moves']:
            raise ValueError('lane_drop v2v initiate/join requires moves=True')
        if not isinstance(v2v, dict) or set(v2v)-{'range_m', 'member_cap'}:
            raise ValueError('lane_drop v2v keys are range_m/member_cap')
        range_m = v2v.get('range_m', DEFAULT_V2V_RANGE_M)
        if not _finite(range_m) or not V2V_RANGE_WINDOW_M[0] <= float(range_m) <= V2V_RANGE_WINDOW_M[1]:
            raise ValueError(f'lane_drop v2v range_m must be in {V2V_RANGE_WINDOW_M} m')
        cap = v2v.get('member_cap', min(len(view.lane_ids), moves.MOVE_MAX_MEMBERS))
        if type(cap) is not int or not 2 <= cap <= moves.MOVE_MAX_MEMBERS:
            raise ValueError(f'lane_drop v2v member_cap must be an integer in 2..{moves.MOVE_MAX_MEMBERS}')
        out['v2v'] = dict(range_m=float(range_m), member_cap=cap)
    if raw.get('rsu') not in (None, False):
        if out['v2v'] is None:
            raise ValueError('lane_drop rsu suggestions require the v2v join protocol')
        from .lane_drop_rsu import rsu_settings
        out['rsu'] = rsu_settings(raw['rsu'], view)
    return out


def config_keys(*, provenance, descriptor, features=True, moves_on=True, v2v_range_m=None,
                member_cap=None, v2v=False, rsu=None):
    """Exact binding keys for the common B/R config."""
    out = dict(enabled=True, rule=RULE, provenance=provenance,
               map=MapView.from_descriptor(descriptor).to_dict(), features=features, moves=moves_on)
    if v2v or v2v_range_m is not None or member_cap is not None:
        out['v2v'] = {k: v for k, v in (('range_m', v2v_range_m), ('member_cap', member_cap)) if v is not None}
    if rsu is not None:
        out['rsu'] = rsu
    return dict(lane_drop=out)


# ---------------------------------------------------------------- observation block

LANE_DROP_FEATURES = (
    'own_missing', 'dist_to_drop_m', 'past_drop', 'in_dropped_lane', 'in_adjacent_lane',
    'lane_offset_from_dropped',
    'closing_count_veh', 'closing_queue_veh', 'closing_mean_speed_mps', 'closing_queue_tail_m',
    'closing_coverage_frac', 'closing_report_age_s', 'closing_missing', 'closing_speed_missing',
    'adjacent_front_gap_m', 'adjacent_front_dv_mps', 'adjacent_front_none',
    'adjacent_rear_gap_m', 'adjacent_rear_dv_mps', 'adjacent_rear_none',
    'adjacent_count_veh', 'adjacent_missing',
    'partner_report_count', 'partner_report_age_s', 'partner_reports_missing',
)
LANE_DROP_DIM = len(LANE_DROP_FEATURES)
FEATURE_SCALE = dict(dist_to_drop_m=1000., closing_count_veh=10., closing_queue_veh=10.,
                     closing_mean_speed_mps=30., closing_queue_tail_m=1000., adjacent_front_gap_m=100.,
                     adjacent_front_dv_mps=30., adjacent_rear_gap_m=100., adjacent_rear_dv_mps=30.,
                     adjacent_count_veh=10., partner_report_count=float(V2V_K))
SCALE = np.array([FEATURE_SCALE.get(name, 1.) for name in LANE_DROP_FEATURES], dtype=np.float32)
_MISSING_BITS = ('own_missing', 'closing_missing', 'closing_speed_missing', 'adjacent_missing',
                 'partner_reports_missing')
_F = {name: i for i, name in enumerate(LANE_DROP_FEATURES)}


def missing_block():
    """Explicitly missing block (no own packet / bank record without sensing)."""
    out = np.zeros(LANE_DROP_DIM, dtype=np.float32)
    for name in _MISSING_BITS:
        out[_F[name]] = 1.
    return out


def _union_length(intervals):
    total, end = 0., -math.inf
    for lo, hi in sorted(intervals):
        if hi <= end:
            continue
        total += hi-max(lo, end)
        end = hi
    return total


def lane_drop_features(own, cav_id, partner_reports=None, *, now_s, partner_ids=(), view,
                       k=V2V_K, range_m=V2V_RANGE_M):
    """``[LANE_DROP_DIM]`` float32 physical block from local packets only (module doc (1))."""
    if not np.isfinite(now_s):
        raise ValueError('now_s must be finite')
    view = MapView.from_descriptor(view)
    out = missing_block()
    if not _valid_report(own, cav_id, now_s):
        return out
    ego = own.source
    out[_F['own_missing']] = 0.
    dist = view.distance_to_drop(ego.s_m)
    out[_F['dist_to_drop_m']] = dist
    out[_F['past_drop']] = float(dist < 0)
    out[_F['in_dropped_lane']] = float(ego.lane == view.dropped_lane)
    out[_F['in_adjacent_lane']] = float(ego.lane == view.adjacent_lane)
    spread = max(view.lane_ids)-min(view.lane_ids)
    out[_F['lane_offset_from_dropped']] = (ego.lane-view.dropped_lane)/spread
    partners = []
    for vid in sorted(set(partner_ids)):
        if vid == cav_id:
            continue
        report = (partner_reports or {}).get(vid)
        if not _valid_report(report, vid, now_s-CONTROL_DT_S):
            continue
        distance = abs(report.source.s_m-ego.s_m)
        if distance <= range_m:
            partners.append((distance, vid, report))
    partners.sort(key=lambda item: (item[0], item[1]))
    accepted = [(0., own)]+[(CONTROL_DT_S, item[2]) for item in partners[:k]]
    if len(accepted) > 1:
        out[_F['partner_report_count']] = len(accepted)-1
        out[_F['partner_report_age_s']] = CONTROL_DT_S
        out[_F['partner_reports_missing']] = 0.
    # Own sensing wins duplicate ids, then the nearest publisher (accepted order).
    visible = {}
    for _, report in accepted:
        for row in _local_rows(report):
            visible.setdefault(row.vehicle_id, row)
    lo_roi, hi_roi = view.lane_drop_s_m-CLOSING_ROI_M, view.lane_drop_s_m
    spans, ages = [], []
    for age, report in accepted:
        if view.dropped_lane not in report.observed_lanes:
            continue
        lo = max(lo_roi, report.source.s_m-SENSOR_RANGE_M)
        hi = min(hi_roi, report.source.s_m+SENSOR_RANGE_M)
        if hi > lo:
            spans.append((lo, hi))
            ages.append(age)
    if spans:
        closing = [visible[v] for v in sorted(visible)
                   if visible[v].lane == view.dropped_lane
                   and any(lo <= visible[v].s_m <= hi for lo, hi in spans)]
        queued = [r for r in closing if r.v_mps < QUEUE_SPEED_MPS]
        out[_F['closing_count_veh']] = len(closing)
        out[_F['closing_queue_veh']] = len(queued)
        out[_F['closing_queue_tail_m']] = max((view.distance_to_drop(r.s_m) for r in queued), default=0.)
        out[_F['closing_coverage_frac']] = _union_length(spans)/CLOSING_ROI_M
        out[_F['closing_report_age_s']] = max(ages)
        out[_F['closing_missing']] = 0.
        if closing:
            out[_F['closing_mean_speed_mps']] = float(np.mean([r.v_mps for r in closing]))
            out[_F['closing_speed_missing']] = 0.
    if view.adjacent_lane in own.observed_lanes:
        out[_F['adjacent_missing']] = 0.
        rows = [r for r in _local_rows(own) if r.vehicle_id != cav_id and r.lane == view.adjacent_lane]
        out[_F['adjacent_count_veh']] = len(rows)
        front = [r for r in rows if r.s_m >= ego.s_m]
        rear = [r for r in rows if r.s_m < ego.s_m]
        if front:
            f = min(front, key=lambda r: (r.s_m-ego.s_m, r.vehicle_id))
            out[_F['adjacent_front_gap_m']] = max(0., f.s_m-ego.s_m-f.length_m)
            out[_F['adjacent_front_dv_mps']] = f.v_mps-ego.v_mps
        else:
            out[_F['adjacent_front_gap_m']], out[_F['adjacent_front_none']] = SENSOR_RANGE_M, 1.
        if rear:
            r = min(rear, key=lambda r: (ego.s_m-r.s_m, r.vehicle_id))
            out[_F['adjacent_rear_gap_m']] = max(0., ego.s_m-r.s_m-ego.length_m)
            out[_F['adjacent_rear_dv_mps']] = r.v_mps-ego.v_mps
        else:
            out[_F['adjacent_rear_gap_m']], out[_F['adjacent_rear_none']] = SENSOR_RANGE_M, 1.
    return out


def scaled(block):
    value = np.asarray(block, dtype=np.float32)
    if value.shape != (LANE_DROP_DIM,) or not np.isfinite(value).all():
        raise ValueError('lane-drop block must be finite [LANE_DROP_DIM]')
    return value/SCALE


# ---------------------------------------------------------------- joint moves

def _check_schedule(onset, hold, depth):
    if type(onset) is not int or type(hold) is not int:
        raise ValueError('lane-drop onset/hold must be integer seconds')
    if (not ONSET_WINDOW_S[0] <= onset <= ONSET_WINDOW_S[1]
            or not HOLD_WINDOW_S[0] <= hold <= HOLD_WINDOW_S[1]
            or not _finite(depth) or not BRAKE_WINDOW_MPS2[0] <= float(depth) <= BRAKE_WINDOW_MPS2[1]):
        raise ValueError(f'lane-drop schedule outside the registered windows onset {ONSET_WINDOW_S} s, '
                         f'hold {HOLD_WINDOW_S} s, braking cap {BRAKE_WINDOW_MPS2} m/s^2')


def _check_speeds(target, gap_dv):
    if not _finite(target) or not TARGET_WINDOW_MPS[0] <= float(target) <= TARGET_WINDOW_MPS[1]:
        raise ValueError(f'target_mps outside the registered window {TARGET_WINDOW_MPS}')
    if not _finite(gap_dv) or not GAP_DV_WINDOW_MPS[0] <= float(gap_dv) <= GAP_DV_WINDOW_MPS[1]:
        raise ValueError(f'gap_dv_mps outside the window {GAP_DV_WINDOW_MPS}')


@dataclass(frozen=True)
class LaneDropMove:
    """Joint lane-drop hold move (NEWMAP2 rule); members in role order with their lanes.

    closing_pair: (closing-lane CAV, adjacent-lane CAV); ``anchor`` 0 = pair_t (all release when
    the closing-lane CAV is ANCHOR_RELEASE_M before the drop). abreast_wall: one member per
    lane, ordered by lane id. group_hold: 2..GROUP_SIZE[1] CAVs in any lanes (the R_random
    hold procedure). Every member holds its target with the speed law and releases at its own
    drop crossing or at the end of the hold."""
    family: str
    members: tuple
    onset: int
    hold: int
    depth: float
    target_mps: float
    gap_dv_mps: float = 0.
    zip: bool = False
    lanes: tuple = ()
    anchor: int = -1
    self_validated_windows = True   # trigger_policy: registered lane-drop windows, not variety

    def __post_init__(self):
        object.__setattr__(self, 'members', tuple(self.members))
        object.__setattr__(self, 'lanes', tuple(self.lanes))
        if self.family not in JOINT_FAMILIES:
            raise ValueError(f'unknown lane-drop family {self.family!r}')
        members, lanes = self.members, self.lanes
        if (not 2 <= len(members) <= moves.MOVE_MAX_MEMBERS or len(set(members)) != len(members)
                or any(not isinstance(v, str) or not v for v in members)):
            raise ValueError(f'lane-drop moves need 2..{moves.MOVE_MAX_MEMBERS} distinct CAV ids')
        if len(lanes) != len(members) or any(type(lane) is not int or lane not in range(N_LANES)
                                              for lane in lanes):
            raise ValueError('lane-drop members need one observed lane each')
        if self.family != 'group_hold' and len(set(lanes)) != len(lanes):
            raise ValueError(f'{self.family} members are in distinct lanes')
        _check_schedule(self.onset, self.hold, self.depth)
        _check_speeds(self.target_mps, self.gap_dv_mps)
        if type(self.zip) is not bool or type(self.anchor) is not int:
            raise ValueError('zip must be bool and anchor an integer index')
        if self.family == 'closing_pair' and len(members) != 2:
            raise ValueError('closing_pair is exactly (closing-lane CAV, adjacent-lane CAV)')
        if self.family != 'closing_pair' and (self.zip or float(self.gap_dv_mps) != 0. or self.anchor != -1):
            raise ValueError(f'{self.family} holds ONE common speed, no zip/gap/anchor')
        if self.family == 'closing_pair' and self.anchor not in (-1, 0):
            raise ValueError('the pair_t anchor is the closing-lane CAV (members[0])')
        if self.family == 'abreast_wall' and list(lanes) != sorted(lanes):
            raise ValueError('abreast_wall members are ordered by lane id')
        if self.family == 'group_hold' and len(members) > GROUP_SIZE[1]:
            raise ValueError(f'group_hold has {GROUP_SIZE} members')

    @property
    def onset_s(self):
        return self.onset

    @property
    def hold_s(self):
        return self.hold

    @property
    def actors(self):
        return self.members

    def role_targets(self):
        if self.family == 'closing_pair':
            return {self.members[0]: float(self.target_mps),
                    self.members[1]: float(self.target_mps)-float(self.gap_dv_mps)}
        return dict.fromkeys(self.members, float(self.target_mps))

    def check_roles(self, view):
        view = MapView.from_descriptor(view)
        if self.family == 'closing_pair':
            if self.lanes != (view.dropped_lane, view.adjacent_lane):
                raise ValueError('closing_pair roles must be (dropped lane, adjacent lane)')
        elif self.family == 'abreast_wall':
            if sorted(self.lanes) != sorted(view.lane_ids):
                raise ValueError('abreast_wall needs exactly one member per lane')
        elif not set(self.lanes) <= set(view.lane_ids):
            raise ValueError('group_hold members must be in map lanes')
        return self

    def to_dict(self):
        out = dict(family=self.family, members=list(self.members), onset=int(self.onset),
                   hold=int(self.hold), depth=float(self.depth), target_mps=float(self.target_mps),
                   gap_dv_mps=float(self.gap_dv_mps), zip=bool(self.zip), lanes=list(self.lanes))
        if self.anchor != -1:
            out['anchor'] = int(self.anchor)
        return out

    @classmethod
    def from_dict(cls, data):
        return cls(data['family'], tuple(data['members']), int(data['onset']), int(data['hold']),
                   float(data['depth']), float(data['target_mps']), float(data.get('gap_dv_mps', 0.)),
                   bool(data.get('zip', False)), tuple(data.get('lanes') or ()), int(data.get('anchor', -1)))


def resolve_family(lanes, view):
    """closing_pair / abreast_wall when the member lanes fill those roles, else group_hold."""
    if len(lanes) == 2 and sorted(lanes) == sorted((view.dropped_lane, view.adjacent_lane)):
        return 'closing_pair'
    if sorted(lanes) == sorted(view.lane_ids):
        return 'abreast_wall'
    return 'group_hold'


def law(target_mps, v_mps, depth):
    """NEWMAP2 hold law: a = clip((v_w - v)/1 s, -dmax, amax)."""
    return min(ACCEL_MAX_MPS2, max(-float(depth), (float(target_mps)-float(v_mps))/CONTROL_DT_S))


def at_drop(view, positions, vid):
    """The member's own KNOWN position has reached the drop (NEWMAP2 x >= X_B)."""
    return vid in positions and view.distance_to_drop(positions[vid]) <= 0.


def anchor_passed(view, positions, vid):
    """pair_t: the KNOWN anchor position is within ANCHOR_RELEASE_M of the drop."""
    return vid in positions and view.distance_to_drop(positions[vid]) <= ANCHOR_RELEASE_M


def _params(rng, kind, *, allow_anchor=True):
    """Registered draws -> (onset, hold, depth, target, gap_dv, zip, anchor).

    closing_pair / abreast_wall: one NEWMAP2 B variant of that family (uniform); group_hold:
    the R_random sets. gap_dv/zip are never drawn (unregistered extensions)."""
    if kind == 'group_hold':
        target = float(R_SETS['target_mps'][int(rng.integers(len(R_SETS['target_mps'])))])
        hold = int(R_SETS['hold'][int(rng.integers(len(R_SETS['hold'])))])
        depth = float(R_SETS['depth'][int(rng.integers(len(R_SETS['depth'])))])
        return ONSET_S, hold, depth, target, 0., False, False
    names = [n for n, v in VARIANTS.items() if v['family'] == kind and (allow_anchor or not v['anchor'])]
    v = VARIANTS[names[int(rng.integers(len(names)))]]
    return ONSET_S, int(v['hold']), float(v['depth']), float(v['target_mps']), 0., False, bool(v['anchor'])


def _in_window(view, row, window):
    return window[0] <= view.distance_to_drop(row['s_m']) <= window[1]


def _spread(rows):
    s = [float(r['s_m']) for r in rows]
    return max(s)-min(s)


def joint_groups(family, view, own, partners):
    """Role-ordered member row tuples for one joint family (NEWMAP2 moment geometry); own
    always included. Partners are the ego's acquirable observable CAVs only."""
    if not _in_window(view, own, ZONE_WINDOW_M):
        return []
    pool = [p for p in partners if _in_window(view, p, ZONE_WINDOW_M)]
    if family == 'closing_pair':
        near = [p for p in pool if abs(float(p['s_m'])-float(own['s_m'])) <= PAIR_DX_M]
        if own['lane'] == view.dropped_lane:
            return [(own, p) for p in near if p['lane'] == view.adjacent_lane]
        if own['lane'] == view.adjacent_lane:
            return [(p, own) for p in near if p['lane'] == view.dropped_lane]
        return []
    if family == 'abreast_wall':
        if own['lane'] not in view.lane_ids or len(view.lane_ids) > moves.MOVE_MAX_MEMBERS:
            return []
        per_lane = [[own] if lane == own['lane'] else [p for p in pool if p['lane'] == lane]
                    for lane in sorted(view.lane_ids)]
        return [tuple(group) for group in product(*per_lane) if _spread(group) <= WALL_SPREAD_M]
    if family == 'group_hold':
        out = []
        for size in range(GROUP_SIZE[0]-1, GROUP_SIZE[1]):
            for group in combinations(pool, size):
                rows = (own,)+tuple(group)
                if _spread(rows) <= GROUP_SPREAD_M:
                    out.append(rows)
        return out
    raise ValueError(f'unregistered lane-drop family {family!r}')


def _joint(family, view, group, rng):
    onset, hold, depth, target, gap_dv, zipped, anchor = _params(rng, family)
    return LaneDropMove(family, tuple(r['vid'] for r in group), onset, hold, depth, target, gap_dv,
                        zipped, tuple(int(r['lane']) for r in group), 0 if anchor else -1)


def pool(family, state, own, partners, rng):
    """Feasible options of one lane-drop family (joint or protocol) for the acting CAV."""
    view = state['view']
    if family in JOINT_FAMILIES:
        return [_joint(family, view, group, rng) for group in joint_groups(family, view, own, partners)]
    if family in INITIATE_SLOTS:
        if state.get('only_join'):
            return []
        option = _initiate(family.split(':', 1)[1], own, state, rng)
        return [] if option is None else [option]
    if family == 'lane_drop_join':
        return [o for o in state.get('offers', ()) if o.members == (own['vid'],)]
    raise ValueError(f'unregistered lane-drop family {family!r}')


def redraw(template, state, own, by_vid, rng):
    """Slate entry from a chosen pool template: re-sample parameters, joins are exact."""
    view = state['view']
    if isinstance(template, LaneDropMove):
        return _joint(template.family, view, [by_vid[v] for v in template.members], rng)
    if template.family == 'lane_drop_initiate':
        return _initiate(template.kind, own, state, rng)
    return template


def families(structured, state, event):
    """Slate pool keys with the lane-drop entries appended (join-only on event ticks).

    Each initiate KIND is its own F2 slot ('lane_drop_initiate:<kind>'), so the balanced
    slate offers closing-pair and abreast-wall initiations equally; the option family
    recorded in the table stays 'lane_drop_initiate'."""
    if event:
        out = tuple(f for f in structured if f in ('wall_join',))
        return out+(('lane_drop_join',) if state.get('v2v') else ())
    out = tuple(structured)
    if state.get('moves'):
        out += JOINT_FAMILIES
    if state.get('v2v'):
        out += INITIATE_SLOTS+('lane_drop_join',)
    return out


def owns(family):
    """Slate pool key handled by this module."""
    return family in JOINT_FAMILIES or family in INITIATE_SLOTS or family == 'lane_drop_join'


# ---------------------------------------------------------------- executable representation

def member_command(move, vid, t, view, *, speed, positions, protocol=None):
    """Absolute law acceleration for ONE member at second t of its own commitment, or None.

    None = no command (released / outside the hold: base policy). Release (NEWMAP2): own drop
    crossing, hold end, pair_t anchor 150 m before the drop, or a received release message.
    ``positions`` are the member's own observable positions (own sensing + t-1 partner
    packets); an unknown anchor never triggers an early release (W1 convention)."""
    option = getattr(move, 'move', move)
    if isinstance(option, LaneDropMove):
        if vid not in option.members or not option.onset <= t < option.onset+option.hold:
            return None
        if at_drop(view, positions, vid) or (
                option.anchor >= 0 and anchor_passed(view, positions, option.members[option.anchor])):
            return None
        return law(option.role_targets()[vid], speed, option.depth)
    if isinstance(option, LaneDropOption):
        if option.members != (vid,) or not option.onset <= t < option.onset+option.hold:
            return None
        if protocol is not None and (vid, option.wall_id) in protocol.release_latch:
            return None
        if at_drop(view, positions, vid) or (option.anchor and anchor_passed(view, positions, option.initiator)):
            return None
        return law(option.own_target(view), speed, option.depth)
    return None


def zip_target(move, vid, t, view, position=None):
    """Lane request (adjacent lane id) for the closing-lane member at its zip second, else None."""
    option = getattr(move, 'move', move)
    if isinstance(option, LaneDropMove):
        ok = option.family == 'closing_pair' and option.zip and vid == option.members[0]
    elif isinstance(option, LaneDropOption):
        ok = (option.kind == 'closing_pair' and option.zip and option.members == (vid,)
              and option.role_lane == view.dropped_lane)
    else:
        return None
    if not ok or t != option.onset+option.hold-1:
        return None
    if position is not None and view.distance_to_drop(position) <= 0:
        return None
    return view.adjacent_lane


def _matrix(value, shape, name):
    from .group_lane import _matrix as strict
    return strict(value, shape, name)


def executed_profile(move, view, policy_profile, speeds, positions):
    """members x duration absolute commands: law while held, the policy elsewhere."""
    out = np.array(policy_profile, float)
    for t in range(out.shape[1]):
        known = {v: positions[i, t] for i, v in enumerate(move.members)}
        for i, vid in enumerate(move.members):
            command = member_command(move, vid, t, view, speed=speeds[i, t], positions=known)
            if command is not None:
                out[i, t] = command
    return out


def lane_actions(move, view, positions):
    return [[int(zip_target(move, vid, t, view, positions[i, t]) is not None)
             for t in range(positions.shape[1])] for i, vid in enumerate(move.members)]


_FORBIDDEN = ('knots', 'knot_s', 'perturbation_knots', 'perturbation_canonical', 'g5_move',
              'lane_actions', 'uses_lane_actions', 'wall_macro')


def plan(move, view, *, policy_profile, member_speeds, member_positions, plan_id=None):
    """Executable plan (G5 convention) with recorded member speeds/positions (law exact)."""
    if not isinstance(move, LaneDropMove):
        raise ValueError('lane_drop.plan needs a LaneDropMove')
    view = MapView.from_descriptor(view)
    move.check_roles(view)
    shape = (len(move.members), move.onset+move.hold)
    base = _matrix(policy_profile, shape, 'policy_profile')
    speeds = _matrix(member_speeds, shape, 'member_speeds_mps')
    positions = _matrix(member_positions, shape, 'member_positions_m')
    profile = executed_profile(move, view, base, speeds, positions)
    pid = plan_id or (f'ld_{move.family}_{"_".join(move.members)}_o{move.onset}_h{move.hold}'
                      f'_d{move.depth:g}_v{move.target_mps:g}')
    return dict(id=pid, family=move.family, members=list(move.members), duration_s=shape[1],
                lane_drop_macro=move.to_dict(), lane_drop_map=view.to_dict(),
                member_speeds_mps=speeds.tolist(), member_positions_m=positions.tolist(),
                policy_profile=base.tolist(), profile=profile.tolist(),
                perturbation_profile=(profile-base).tolist(),
                lane_drop_lane_actions=lane_actions(move, view, positions))


VERIFIED_TOL = 1e-6   # mined rows store issued commands rounded to 1e-4 and perturbation = profile - policy


def _bind_verified(plan, move, view, dur):
    """Mined/verified representation: the row's own executed profile + per-member release el."""
    shape = (len(move.members), dur)
    profile = _matrix(plan['profile'], shape, 'profile')
    base = _matrix(plan['policy_profile'], shape, 'policy_profile')
    stored = _matrix(plan['perturbation_profile'], shape, 'perturbation_profile')
    release = plan['member_release_el']
    if (not isinstance(release, list) or len(release) != len(move.members)
            or any(type(el) is not int or not 0 <= el <= move.onset+move.hold for el in release)
            or dur != max(release) or dur < 1):
        raise ValueError('verified lane-drop plan: release el per member within the hold; duration = last release')
    if not np.allclose(stored, profile-base, atol=VERIFIED_TOL, rtol=0):
        raise ValueError('verified lane-drop plan: perturbation_profile != profile - policy_profile')
    lo, hi = -float(move.depth)-VERIFIED_TOL, ACCEL_MAX_MPS2+VERIFIED_TOL
    for i, el in enumerate(release):
        held = profile[i, move.onset:el]
        if held.size and (held.min() < lo or held.max() > hi):
            raise ValueError('verified lane-drop plan: a held command exceeds the registered law bounds')
    if plan['lane_drop_lane_actions'] != [[0]*dur for _ in move.members]:
        raise ValueError('verified hold plans carry no lane actions')
    return profile


def from_plan(plan):
    """Exact executable binding of a ``lane_drop_macro`` plan (never a lossy projection).

    Two representations: recorded law (member_speeds_mps/member_positions_m: the profile must
    equal the law on them) and verified mined rows (member_release_el: the row's executed
    profile within the registered law bounds, see plan_from_mined_row)."""
    macro = plan.get('lane_drop_macro')
    if not isinstance(macro, dict):
        raise ValueError('lane-drop plan requires an explicit lane_drop_macro')
    move = LaneDropMove.from_dict(macro)
    view = MapView.from_descriptor(plan.get('lane_drop_map'))
    move.check_roles(view)
    if list(move.members) != plan.get('members') or plan.get('family') != move.family:
        raise ValueError('lane-drop plan member/family binding mismatch')
    dur = plan.get('duration_s')
    if type(dur) is not int:
        raise ValueError('lane-drop plan needs an integer duration_s')
    if any(key in plan for key in _FORBIDDEN):
        raise ValueError('lane-drop plan carries a conflicting representation')
    verified = 'member_release_el' in plan
    needed = ('profile', 'policy_profile', 'perturbation_profile', 'lane_drop_lane_actions')+(
        ('member_release_el',) if verified else ('member_speeds_mps', 'member_positions_m'))
    for key in needed:
        if key not in plan:
            raise ValueError(f'lane-drop plan requires {key} (incomplete executable representation)')
    if verified:
        if 'member_speeds_mps' in plan or 'member_positions_m' in plan:
            raise ValueError('lane-drop plan mixes the verified and recorded-law representations')
        profile = _bind_verified(plan, move, view, dur)
    else:
        if dur != move.onset+move.hold:
            raise ValueError('lane-drop plan duration_s must equal onset + hold')
        shape = (len(move.members), dur)
        profile = _matrix(plan['profile'], shape, 'profile')
        base = _matrix(plan['policy_profile'], shape, 'policy_profile')
        stored = _matrix(plan['perturbation_profile'], shape, 'perturbation_profile')
        speeds = _matrix(plan['member_speeds_mps'], shape, 'member_speeds_mps')
        positions = _matrix(plan['member_positions_m'], shape, 'member_positions_m')
        wanted = executed_profile(move, view, base, speeds, positions)
        if not np.allclose(profile, wanted, atol=1e-8, rtol=0) \
                or not np.allclose(stored, wanted-base, atol=1e-8, rtol=0):
            raise ValueError('lane-drop executable profile does not equal the law on the recorded speeds')
        if plan['lane_drop_lane_actions'] != lane_actions(move, view, positions):
            raise ValueError('lane-drop plan lane actions do not match the executable macro')
    from .plans import acceleration_at
    for t in range(dur):
        executed = acceleration_at(plan, t)
        if set(executed) != set(move.members) or any(
                abs(executed[v]-profile[i, t]) > 1e-8 for i, v in enumerate(move.members)):
            raise ValueError('lane-drop executor does not run the bound executable profile')
    return move


class NotRepresentable(ValueError):
    """A mined row whose family needs lane-change moves (N9), not a hold macro."""


_MINED_PARAMS = {'release_x', 'amax', 'v_w', 'hold_s', 'dmax', 'anchor', 'anchor_release_x', 'start_s'}


def plan_from_mined_row(row, descriptor):
    """Exact converter: coopmwm-mined-move-v1 NEWMAP2 / PLANGEN / PLANGEN2 hold row -> plan.

    SUMO lanes are mapped through the N1 descriptor's lane table (wz3to2: SUMO 0 = closing ->
    WM 2); release_x must be the registered drop x and anchor_release_x the registered anchor
    distance. closing_pair / abreast_wall rows must fill their roles (else refused); R_random
    hold rows become closing_pair / abreast_wall when their lanes fill those roles, otherwise
    group_hold. Rows with lane-change roles or commands raise NotRepresentable (N9). The
    returned plan is bound exactly by from_plan (moves.from_plan never projects it)."""
    if row.get('schema') != 'coopmwm-mined-move-v1':
        raise ValueError('mined row schema must be coopmwm-mined-move-v1')
    view = MapView.from_descriptor(descriptor)
    neck = descriptor.bottleneck
    if (row.get('world') or {}).get('map') != descriptor.name or row['move'].get('map') != descriptor.name:
        raise ValueError('mined row map differs from the descriptor map')
    entries = row['move']['members']
    ids = [m['id'] for m in entries]
    if ids != list(row['members']) or len(set(ids)) != len(ids):
        raise ValueError('mined row move members differ from its member list')
    roles = {m.get('role') for m in entries}
    if not roles <= {None, 'hold'} or row.get('commands'):
        raise NotRepresentable(f"{row['family']}/{row['variant']}: roles {sorted(map(str, roles))} or "
                               'lane commands need N9 lane-change moves')
    family = row['family']
    if family not in ('closing_pair', 'abreast_wall', 'R_random'):
        raise NotRepresentable(f'{family}: not a lane-drop hold family')
    params = [m['params'] for m in entries]
    if any(set(p)-_MINED_PARAMS for p in params):
        raise NotRepresentable(f"{family}/{row['variant']}: unregistered member parameters")
    common = {k: params[0].get(k) for k in ('v_w', 'hold_s', 'dmax', 'amax', 'release_x', 'start_s',
                                             'anchor', 'anchor_release_x')}
    if any({k: p.get(k) for k in common} != common for p in params):
        raise ValueError('mined row members do not share one hold macro')
    if float(common['release_x']) != float(neck['drop_x_m']):
        raise ValueError('mined row release_x is not the registered drop')
    if float(common['amax']) != ACCEL_MAX_MPS2:
        raise ValueError('mined row amax is not the registered 1.0 m/s^2')
    lanes = [descriptor.lane_table[(neck['edge'], int(m['lane']))][0] for m in entries]
    resolved = resolve_family(lanes, view)
    if family != 'R_random' and resolved != family:
        raise ValueError(f'{family} row lanes {lanes} do not fill the {family} roles')
    if resolved == 'closing_pair':
        order = sorted(range(2), key=lambda i: lanes[i] != view.dropped_lane)
    elif resolved == 'abreast_wall':
        order = sorted(range(len(lanes)), key=lambda i: lanes[i])
    else:
        order = list(range(len(lanes)))
    anchor = -1
    if common['anchor'] is not None:
        if float(neck['drop_x_m'])-float(common['anchor_release_x']) != ANCHOR_RELEASE_M:
            raise ValueError('mined row anchor release is not the registered distance')
        anchor = [ids[i] for i in order].index(common['anchor'])
    move = LaneDropMove(resolved, tuple(ids[i] for i in order), int(common['start_s'] or 0),
                        int(common['hold_s']), float(common['dmax']), float(common['v_w']), 0., False,
                        tuple(lanes[i] for i in order), anchor)
    by_id = {v: i for i, v in enumerate(ids)}
    return verified_plan(move, view, duration_s=row['duration_s'],
        release_el={v: row['release_el'][v] for v in ids},
        profile={v: row['profile'][by_id[v]] for v in ids},
        policy_profile={v: row['policy_profile'][by_id[v]] for v in ids},
        perturbation_profile={v: row['perturbation_profile'][by_id[v]] for v in ids},
        plan_id=f"ld_{row['verification_id']}",
        source=dict(schema=row['schema'], arm=row.get('arm'), verification_id=row['verification_id'],
                    plan_sha256=row.get('plan_sha256'), family=family, variant=row.get('variant'),
                    r_kind=row.get('r_kind'), situation_id=row.get('situation_id'), t0=row.get('t0'),
                    seed=row.get('seed'), sumo_lanes=[int(entries[i]['lane']) for i in order]))


def move_from_mined_row(row, descriptor):
    """The LaneDropMove (WM frame) that a mined hold row executed (see plan_from_mined_row)."""
    return LaneDropMove.from_dict(plan_from_mined_row(row, descriptor)['lane_drop_macro'])


def verified_plan(move, view, *, duration_s, release_el, profile, policy_profile, perturbation_profile=None,
                  plan_id=None, source=None):
    """Verified-run representation for a SUMO-executed lane-drop move (mined rows and N4's
    provisional builder): per-member executed profile (issued while held, else measured),
    policy profile, release el. Mappings are keyed by member id; bound by from_plan."""
    view = MapView.from_descriptor(view)
    members = list(move.members)
    if set(release_el) != set(members) or set(profile) != set(members) or set(policy_profile) != set(members):
        raise ValueError('verified plan needs every member release el / profile / policy profile')
    prof = [list(profile[v]) for v in members]
    base = [list(policy_profile[v]) for v in members]
    pert = ([list(perturbation_profile[v]) for v in members] if perturbation_profile is not None
            else (np.array(prof, float)-np.array(base, float)).tolist())
    out = dict(id=plan_id or f'ld_verified_{move.family}_{"_".join(members)}', family=move.family,
               members=members, duration_s=int(duration_s), lane_drop_macro=move.to_dict(),
               lane_drop_map=view.to_dict(), member_release_el=[release_el[v] for v in members],
               profile=prof, policy_profile=base, perturbation_profile=pert,
               lane_drop_lane_actions=[[0]*int(duration_s) for _ in members], source=dict(source or {}))
    from_plan(out)
    return out


def newmap_members(move, descriptor):
    """SUMO execution spec of a WM-frame LaneDropMove in the NEWMAP2 executor format
    (nm2_core Arm members: id, SUMO lane on the approach edge, role 'hold', params). This is the
    API for a SUMO verification run of a proposed move (N4 provisional builder, origin
    clean4_provisional_speed_target); the run's result returns through verified_plan."""
    view = MapView.from_descriptor(descriptor)
    move.check_roles(view)
    if move.zip or float(move.gap_dv_mps) != 0. or move.onset != ONSET_S:
        raise NotRepresentable('zip / gap_dv / delayed onset have no NEWMAP2 executor form')
    neck = descriptor.bottleneck
    sumo_of = {wm: idx for (edge, idx), (wm, _) in descriptor.lane_table.items() if edge == neck['edge']}
    params = dict(release_x=float(neck['drop_x_m']), amax=ACCEL_MAX_MPS2, v_w=float(move.target_mps),
                  hold_s=int(move.hold), dmax=float(move.depth))
    if move.anchor >= 0:
        params.update(anchor=move.members[move.anchor],
                      anchor_release_x=float(neck['drop_x_m'])-ANCHOR_RELEASE_M)
    return [dict(id=v, lane=int(sumo_of[lane]), role='hold', params=dict(params))
            for v, lane in zip(move.members, move.lanes)]


# ---------------------------------------------------------------- lane changes (G5 env path)

def safe_targets(requests, local_states, eligible_cavs, view):
    """Execution-time recheck of zip requests with each commanded CAV's OWN sensing only."""
    from wms.control.lanes import LaneRules, _target_gap_ok
    view = MapView.from_descriptor(view)
    out, blocked = {}, {}
    for vid, target in sorted(requests.items()):
        state = local_states.get(vid) if vid in eligible_cavs else None
        rows = (state or {}).get('rows') or []
        by = {r['vid']: r for r in rows}
        ego = by.get(vid)
        if state is None or state.get('self') != vid or ego is None or not ego.get('is_cav') \
                or len(by) != len(rows):
            blocked[vid] = 'missing_observable_gap_or_cav'
            continue
        if any(type(r.get('is_cav')) is not bool or r.get('lane') not in range(N_LANES)
               or any(not _finite(r.get(k)) for k in ('s_m', 'v_mps', 'length_m'))
               or r['v_mps'] < 0 or r['length_m'] <= 0
               or abs(r['s_m']-ego['s_m']) > SENSOR_RANGE_M for r in rows):
            blocked[vid] = 'missing_observable_gap_or_cav'
        elif type(target) is not int or target != view.adjacent_lane or ego['lane'] != view.dropped_lane:
            blocked[vid] = 'source_lane_changed'
        elif view.distance_to_drop(ego['s_m']) <= 0:
            blocked[vid] = 'past_drop'
        else:
            s = torch.tensor([[float(r['s_m']) for r in rows]])
            v = torch.tensor([[float(r['v_mps']) for r in rows]])
            length = torch.tensor([[float(r['length_m']) for r in rows]])
            lane = torch.tensor([[int(r['lane']) for r in rows]])
            valid = torch.ones_like(lane, dtype=torch.bool)
            index = rows.index(ego)
            goal = lane.clone()
            goal[0, index] = target
            ok = _target_gap_ok(s, v, length, valid, lane, goal, LaneRules())[0, index]
            if bool(ok):
                out[vid] = target
            else:
                blocked[vid] = 'unsafe_gap_or_edge'
    return out, blocked


def controller_lane_targets(controller, policy, rows, now):
    """Lane-drop zip requests (live samples + committed verified plans) after the own-sensing
    recheck; mirrors group_lane.controller_targets (branch release, CAV-only, no extra writer)."""
    if not controller.events:
        return {}
    sample = controller.events[-1]['sample']
    requests = dict(sample.get('lane_drop_lane_requests') or {})
    ld = getattr(policy, 'lane_drop', None)
    view = ld['view'] if isinstance(ld, dict) else None
    for entry in controller.active_plans:
        plan = entry['plan']
        if plan.get('lane_drop_macro') is None:
            continue
        move = from_plan(plan)
        view = MapView.from_descriptor(plan['lane_drop_map']) if view is None else view
        for vid in move.members:
            requests.pop(vid, None)
        elapsed = now-entry['start_s']
        for i, vid in enumerate(move.members):
            if 0 <= elapsed < plan['duration_s'] and plan['lane_drop_lane_actions'][i][int(elapsed)]:
                requests[vid] = view.adjacent_lane
    branch = controller.branch
    if branch and now >= branch['time_s']:
        release = branch.get('group', ()) if now >= branch.get('until', math.inf) else branch['revert']
        for vid in release:
            requests.pop(vid, None)
        if branch.get('mappo_background'):
            allowed = set(branch['group'])-set(branch['revert']) if now < branch['until'] else set()
            requests = {v: t for v, t in requests.items() if v in allowed}
    if view is None:
        if requests:
            raise ValueError('lane-drop lane requests need the registered map view')
        return {}
    eligible = {r['vid'] for r in rows if r.get('is_cav')}
    # Each commanded CAV's OWN sensing packet (same whitelist/radius as EpisodePolicy's
    # sensed rows), never the slot-limited candidate rows or the full scene.
    local = {}
    for actor in rows:
        if actor['vid'] in requests and actor.get('is_cav'):
            sensed = [{k: r[k] for k in ('vid', 's_m', 'v_mps', 'lane', 'is_cav', 'length_m')}
                      for r in rows if abs(r['s_m']-actor['s_m']) <= SENSOR_RANGE_M]
            local[actor['vid']] = dict(self=actor['vid'], rows=sensed)
    safe, blocked = safe_targets(requests, local, eligible, view)
    if requests or 'lane_drop_lane_requests' in sample:
        sample['lane_drop_lane_requests'] = requests
        sample['lane_drop_lane_gate'] = dict(safe=safe, blocked=blocked)
    return safe


# ---------------------------------------------------------------- V2V initiate/join protocol

@dataclass(frozen=True)
class LaneDropOption:
    """Per-CAV protocol option (members = (ego,)); onset/hold relative to the decision second.

    ``target_mps`` / ``gap_dv_mps`` / ``zip`` are the session's; the CAV's own law target
    follows from its role lane (closing_pair partner = target - gap_dv)."""
    family: str
    members: tuple
    kind: str
    onset: int
    hold: int
    depth: float
    target_mps: float
    gap_dv_mps: float
    zip: bool
    role_lane: int
    wall_id: str
    initiator: str
    gap_m: float = 0.
    member_count: int = 1
    cap: int = 2
    range_m: float = DEFAULT_V2V_RANGE_M
    covered: tuple = ()
    anchor: bool = False      # pair_t session: release when the initiator (closing-lane anchor) nears
    self_validated_windows = True   # trigger_policy: registered lane-drop windows, not variety

    def __post_init__(self):
        object.__setattr__(self, 'members', tuple(self.members))
        object.__setattr__(self, 'covered', tuple(self.covered))
        if self.family not in PROTOCOL_FAMILIES or len(self.members) != 1 or self.kind not in KINDS:
            raise ValueError('lane-drop option: registered family/kind and exactly the acting CAV')
        if (not isinstance(self.wall_id, str) or not self.wall_id or not isinstance(self.initiator, str)
                or (self.family == 'lane_drop_initiate' and self.initiator != self.members[0])
                or (self.family == 'lane_drop_join' and self.initiator == self.members[0])):
            raise ValueError('lane-drop option initiator/session id mismatch')
        for key in ('member_count', 'cap', 'role_lane'):
            if type(getattr(self, key)) is not int:
                raise ValueError(f'lane-drop option {key} must be an integer')
        _check_schedule(self.onset, self.hold, self.depth)
        _check_speeds(self.target_mps, self.gap_dv_mps)
        if type(self.zip) is not bool or (self.kind == 'abreast_wall' and (self.zip or self.gap_dv_mps)):
            raise ValueError('abreast_wall sessions have no zip / gap offset')
        if type(self.anchor) is not bool or (self.anchor and self.kind != 'closing_pair'):
            raise ValueError('only closing_pair sessions carry the pair_t anchor release')
        least = 0 if self.initiator == RSU_ID else 1   # an RSU session may have no member yet
        if not 2 <= self.cap <= moves.MOVE_MAX_MEMBERS or not least <= self.member_count < self.cap:
            raise ValueError('lane-drop option membership outside the registered cap')
        if self.role_lane in self.covered or any(type(lane) is not int for lane in self.covered):
            raise ValueError('lane-drop option role lane already covered')
        if not (math.isfinite(self.gap_m) and math.isfinite(self.range_m) and self.range_m > 0):
            raise ValueError('lane-drop option geometry must be finite')

    @property
    def onset_s(self):
        return self.onset

    @property
    def hold_s(self):
        return self.hold

    @property
    def actors(self):
        return self.members

    def own_target(self, view):
        if self.kind == 'closing_pair' and self.role_lane != view.dropped_lane:
            return float(self.target_mps)-float(self.gap_dv_mps)
        return float(self.target_mps)

    def to_dict(self):
        return dict(family=self.family, members=list(self.members), kind=self.kind, onset=int(self.onset),
                    hold=int(self.hold), depth=float(self.depth), target_mps=float(self.target_mps),
                    gap_dv_mps=float(self.gap_dv_mps), zip=bool(self.zip), role_lane=int(self.role_lane),
                    wall_id=self.wall_id, initiator=self.initiator, gap_m=float(self.gap_m),
                    member_count=int(self.member_count), cap=int(self.cap), range_m=float(self.range_m),
                    covered=list(self.covered), anchor=bool(self.anchor))

    @classmethod
    def from_dict(cls, data):
        return cls(data['family'], tuple(data['members']), data['kind'], int(data['onset']),
                   int(data['hold']), float(data['depth']), float(data['target_mps']),
                   float(data['gap_dv_mps']), bool(data['zip']), int(data['role_lane']), data['wall_id'],
                   data['initiator'], float(data['gap_m']), int(data['member_count']), int(data['cap']),
                   float(data['range_m']), tuple(data.get('covered') or ()), bool(data.get('anchor', False)))


def option_from_dict(data):
    """Recorded table entry -> option (lane-drop, then W1 protocol / shared vocabulary)."""
    family = data['family']
    if family in JOINT_FAMILIES:
        return LaneDropMove.from_dict(data)
    if family in PROTOCOL_FAMILIES:
        return LaneDropOption.from_dict(data)
    return v2v_wall.option_from_dict(data)


def is_lane_drop(move):
    return isinstance(getattr(move, 'move', move), (LaneDropMove, LaneDropOption))


def kind_cap(kind, view, member_cap):
    return min(member_cap, 2 if kind == 'closing_pair' else len(view.lane_ids))


def role_lanes(kind, view):
    return (view.dropped_lane, view.adjacent_lane) if kind == 'closing_pair' else tuple(view.lane_ids)


def _initiate(kind, own, state, rng):
    """Initiate option of one kind for the acting CAV, or None (own state only)."""
    view = state['view']
    if own['lane'] not in role_lanes(kind, view) or not _in_window(view, own, ZONE_WINDOW_M):
        return None
    # pair_t anchors on the closing-lane CAV, so only a closing-lane initiator may draw it.
    onset, hold, depth, target, gap_dv, zipped, anchor = _params(
        rng, kind, allow_anchor=kind == 'closing_pair' and own['lane'] == view.dropped_lane)
    ego = own['vid']
    return LaneDropOption('lane_drop_initiate', (ego,), kind, onset, hold, depth, target, gap_dv, zipped,
                          int(own['lane']), wall_id=f'{ego}@{state["now"]}', initiator=ego,
                          cap=kind_cap(kind, view, state['cap']), range_m=state['range_m'], anchor=anchor)


@dataclass(frozen=True)
class LaneDropIntent:
    """Session message published at ``timestamp_s`` (visible at t+1) by a CAV or the RSU."""
    wall_id: str
    initiator: str
    kind: str
    timestamp_s: int
    source_s_m: float
    source_lane: int
    start_s: int
    onset: int
    hold: int
    depth: float
    target_mps: float
    gap_dv_mps: float
    zip: bool
    covered: tuple
    member_count: int
    cap: int
    released: bool
    radius_m: float = 0.     # RSU suggestions: join radius around the anchor (source_s_m)
    anchor: bool = False     # pair_t session (released when the initiator nears the drop)

    @property
    def end_s(self):
        return self.start_s+self.onset+self.hold

    def to_dict(self):
        return dict(wall_id=self.wall_id, initiator=self.initiator, kind=self.kind,
                    timestamp_s=self.timestamp_s, source_s_m=float(self.source_s_m),
                    source_lane=int(self.source_lane), start_s=self.start_s, onset=self.onset,
                    hold=self.hold, depth=float(self.depth), target_mps=float(self.target_mps),
                    gap_dv_mps=float(self.gap_dv_mps), zip=bool(self.zip), covered=list(self.covered),
                    member_count=self.member_count, cap=self.cap, released=bool(self.released),
                    radius_m=float(self.radius_m), anchor=bool(self.anchor))


class Protocol(v2v_wall.Protocol):
    """Lane-drop session ledger; reuses the W1 Protocol (expiry, events, offers bookkeeping).

    Policy inputs are only t-1 messages (offers) and each CAV's own state; sessions are the
    actuation arbitration ledger (ownership + cap), never a policy input."""
    FAMILIES = PROTOCOL_FAMILIES

    def __init__(self, ld):
        self.view = ld['view']
        self.member_cap = ld['v2v']['member_cap']
        super().__init__(dict(member_cap=self.member_cap, range_m=ld['v2v']['range_m'], rule=RULE,
                              provenance=ld['provenance']))

    def reset(self):
        super().reset()
        self.rsu_reception = None   # (rsu_position_m, rsu_range_m) when an RSU is registered

    def receive(self, now, own_rows):
        """Offers from messages published at now-1; own_rows = {cav: own row at now}."""
        now = v2v_wall._second(now)
        self._expire(now)
        offers, self.received_at = {}, now
        view = self.view
        for vid, own in own_rows.items():
            s, lane = float(own['s_m']), int(own['lane'])
            out = []
            for msg in sorted(self.board.values(), key=lambda m: m.wall_id):
                if msg.timestamp_s != now-CONTROL_DT_S or msg.initiator == vid:
                    continue
                if msg.initiator == RSU_ID:
                    position, reach = self.rsu_reception or (None, None)
                    if position is None or abs(s-position) > reach or abs(s-msg.source_s_m) > msg.radius_m:
                        continue   # outside the RSU broadcast or the suggested formation
                elif abs(msg.source_s_m-s) > self.range_m:
                    continue       # radio range: the message is not received
                if msg.released:
                    if self.membership.get(vid) == msg.wall_id:
                        self.release_latch.add((vid, msg.wall_id))
                    continue
                if vid in self.membership or msg.member_count >= msg.cap:
                    continue
                if (lane not in role_lanes(msg.kind, view) or lane in msg.covered
                        or not ZONE_WINDOW_M[0] <= view.distance_to_drop(s) <= ZONE_WINDOW_M[1]):
                    continue
                begin = msg.start_s+msg.onset
                onset, hold = max(0, begin-now), msg.end_s-max(now, begin)
                if (not HOLD_WINDOW_S[0] <= hold <= HOLD_WINDOW_S[1]
                        or not ONSET_WINDOW_S[0] <= onset <= ONSET_WINDOW_S[1]):
                    continue
                out.append(LaneDropOption('lane_drop_join', (vid,), msg.kind, int(onset), int(hold),
                    float(msg.depth), float(msg.target_mps), float(msg.gap_dv_mps), bool(msg.zip), lane,
                    wall_id=msg.wall_id, initiator=msg.initiator, gap_m=float(msg.source_s_m-s),
                    member_count=int(msg.member_count), cap=int(msg.cap),
                    range_m=float(msg.radius_m if msg.initiator == RSU_ID else self.range_m),
                    covered=tuple(msg.covered), anchor=bool(msg.anchor)))
            if out:
                offers[vid] = tuple(sorted(out, key=lambda o: (abs(o.gap_m), o.wall_id)))
        self.offers = offers
        return offers

    def state(self, vid, now):
        return dict(now=v2v_wall._second(now), offers=self.offers.get(vid, ()), cap=self.member_cap,
                    range_m=self.range_m)

    def mark_offered(self, ego, table):
        for move in table:
            if move is not None and getattr(move, 'family', None) == 'lane_drop_join':
                self.offered.setdefault(ego, set()).add(move.wall_id)

    def _session(self, option, now, lane):
        return dict(initiator=option.initiator, kind=option.kind, start_s=now, onset=option.onset,
                    hold=option.hold, depth=float(option.depth), target_mps=float(option.target_mps),
                    gap_dv_mps=float(option.gap_dv_mps), zip=bool(option.zip), cap=int(option.cap),
                    anchor=bool(option.anchor), members={option.initiator: now},
                    lanes={option.initiator: lane}, released=False, closed=False)

    def start(self, move, *, ego, executor, observable_cav_ids, now):
        """Atomic protocol arbitration + executor commitment. Returns accepted."""
        option = getattr(move, 'move', move)
        now = v2v_wall._second(now)
        if not isinstance(option, LaneDropOption) or option.members != (ego,):
            raise ValueError('lane-drop protocol option must act through the deciding CAV only')
        if ego in self.membership:
            return False
        if option.family == 'lane_drop_initiate':
            if (option.wall_id in self.sessions or option.wall_id != f'{ego}@{now}'
                    or option.cap != kind_cap(option.kind, self.view, self.member_cap)):
                return False
            if not executor.start(move, ego=ego, observable_cav_ids=observable_cav_ids, now=now,
                                  until=now+option.onset+option.hold):
                return False
            self.sessions[option.wall_id] = self._session(option, now, option.role_lane)
            self.membership[ego] = option.wall_id
            return True
        session = self.sessions.get(option.wall_id)
        if (session is None or session['released'] or session['closed']
                or len(session['members']) >= session['cap']
                or option.role_lane in session['lanes'].values()
                or option.role_lane not in role_lanes(session['kind'], self.view)):
            return False
        end = session['start_s']+session['onset']+session['hold']
        if (now+option.onset+option.hold != end or float(option.depth) != session['depth']
                or option.kind != session['kind'] or float(option.target_mps) != session['target_mps']
                or float(option.gap_dv_mps) != session['gap_dv_mps'] or option.zip != session['zip']
                or option.anchor != session['anchor']):
            raise ValueError('join option schedule differs from the session it names')
        if not executor.start(move, ego=ego, observable_cav_ids=observable_cav_ids, now=now, until=end):
            return False
        session['members'][ego] = now
        session['lanes'][ego] = option.role_lane
        self.membership[ego] = option.wall_id
        self.last_joins.append(dict(wall_id=option.wall_id, vid=ego, time_s=now))
        return True

    def open_offline(self, option, now):
        self.sessions[option.wall_id] = self._session(option, v2v_wall._second(now), option.role_lane)
        self.membership[option.initiator] = option.wall_id

    def admit_offline(self, option, now):
        session = self.sessions.get(option.wall_id)
        vid = option.members[0]
        if (session is None or session['released'] or len(session['members']) >= session['cap']
                or vid in self.membership or option.role_lane in session['lanes'].values()):
            return False
        session['members'][vid] = v2v_wall._second(now)
        session['lanes'][vid] = option.role_lane
        self.membership[vid] = option.wall_id
        return True

    def open_rsu(self, suggestion, now):
        """RSU-initiated session (no CAV initiator, no executor commitment of its own)."""
        now = v2v_wall._second(now)
        wall_id = f'{RSU_ID}@{now}'
        if wall_id in self.sessions:
            return False
        self.sessions[wall_id] = dict(initiator=RSU_ID, kind=suggestion['kind'], start_s=now,
            onset=int(suggestion['onset']), hold=int(suggestion['hold']), depth=float(suggestion['depth']),
            target_mps=float(suggestion['target_mps']), gap_dv_mps=float(suggestion['gap_dv_mps']),
            zip=bool(suggestion['zip']), cap=kind_cap(suggestion['kind'], self.view, self.member_cap),
            anchor=False, members={}, lanes={}, released=False, closed=False,
            anchor_s_m=float(suggestion['anchor_s_m']),
            anchor_v_mps=float(suggestion['anchor_v_mps']), radius_m=float(suggestion['radius_m']))
        return True

    def publish(self, now, sources, rsu_reports=None):
        """Initiators (and the RSU) broadcast at ``now``; sources = {cav: own ObservedVehicle}.

        RSU sessions use only the members' t-1 uplink reports (``rsu_reports``) for the
        anchor/release, never same-tick state."""
        now = v2v_wall._second(now)
        self._expire(now)
        board = {}
        for wall_id, s in sorted(self.sessions.items()):
            if s['released'] or s['closed']:
                continue
            known = {v: lane for v, lane in s['lanes'].items() if s['members'][v] <= now-1}
            if s['initiator'] == RSU_ID:
                reports = {v: r for v, r in (rsu_reports or {}).items() if v in s['members']}
                if reports:
                    anchor = float(np.mean([r.source.s_m for r in reports.values()]))
                else:
                    anchor = s['anchor_s_m']+s['anchor_v_mps']*(now-s['start_s'])
                if self.view.distance_to_drop(anchor) <= 0.:
                    s['closed'] = True    # formation passed the drop: no new joins
                    continue
                board[wall_id] = LaneDropIntent(wall_id, RSU_ID, s['kind'], now, anchor, -1, s['start_s'],
                    s['onset'], s['hold'], s['depth'], s['target_mps'], s['gap_dv_mps'], s['zip'],
                    tuple(sorted(known.values())), len(known), s['cap'], False, s['radius_m'])
            else:
                source = sources.get(s['initiator'])
                if source is None:
                    continue   # departed/missing initiator: no broadcast this second
                distance = self.view.distance_to_drop(source.s_m)
                if distance <= 0. and not s['anchor']:
                    s['closed'] = True    # initiator passed the drop: no new joins (members keep own release)
                    continue
                # pair_t: the closing-lane anchor 150 m before the drop releases every member.
                released = bool(s['anchor']) and distance <= ANCHOR_RELEASE_M
                covered = dict(known, **{s['initiator']: s['lanes'][s['initiator']]})
                board[wall_id] = LaneDropIntent(wall_id, s['initiator'], s['kind'], now, float(source.s_m),
                    int(source.lane), s['start_s'], s['onset'], s['hold'], s['depth'], s['target_mps'],
                    s['gap_dv_mps'], s['zip'], tuple(sorted(covered.values())), len(covered), s['cap'],
                    released, 0., bool(s['anchor']))
            if board[wall_id].released:
                s['released'] = True
        self.board = board
        self.last_published = [m.to_dict() for m in board.values()]
        return board

    def receipt(self):
        out = dict(rule=RULE, published=self.last_published,
                   offers={v: [o.wall_id for o in opts] for v, opts in sorted(self.offers.items())},
                   joins=list(self.last_joins),
                   sessions=[dict(wall_id=w, initiator=s['initiator'], kind=s['kind'],
                                  members=len(s['members']), released=s['released'], closed=s['closed'])
                             for w, s in sorted(self.sessions.items())])
        self.last_joins = []
        return out


# ---------------------------------------------------------------- trigger extension

LD_DESCRIPTOR = ('ld_is_initiate', 'ld_is_join', 'ld_kind_closing_pair', 'ld_kind_abreast_wall',
                 'ld_target_frac', 'ld_gap_dv_frac', 'ld_zip', 'ld_role_dropped_lane',
                 'ld_initiator_gap_over_range', 'ld_member_count_over_cap', 'ld_rsu_initiated',
                 'ld_kind_group_hold', 'ld_anchor', 'ld_depth_frac', 'ld_hold_frac')


def ld_descriptor(move, view):
    """Per-candidate descriptor (zeros for none / non-lane-drop candidates)."""
    out = [0.]*len(LD_DESCRIPTOR)
    if move is None:
        return out
    get = (lambda k, d=None: move.get(k, d)) if isinstance(move, dict) else (lambda k, d=None: getattr(move, k, d))
    family = get('family')
    if family not in JOINT_FAMILIES+PROTOCOL_FAMILIES:
        return out
    kind = family if family in JOINT_FAMILIES else get('kind')
    lo, hi = TARGET_WINDOW_MPS
    out[0], out[1] = float(family == 'lane_drop_initiate'), float(family == 'lane_drop_join')
    out[2], out[3] = float(kind == 'closing_pair'), float(kind == 'abreast_wall')
    out[4] = (float(get('target_mps'))-lo)/(hi-lo)
    out[5] = float(get('gap_dv_mps'))/GAP_DV_WINDOW_MPS[1]
    out[6] = float(bool(get('zip')))
    if family in PROTOCOL_FAMILIES:
        out[7] = float(int(get('role_lane')) == view.dropped_lane)
        out[8] = max(-1., min(1., float(get('gap_m'))/float(get('range_m'))))
        out[9] = int(get('member_count'))/int(get('cap'))
        out[10] = float(get('initiator') == RSU_ID)
        out[12] = float(bool(get('anchor', False)))
    else:
        out[7] = float(view.dropped_lane in tuple(get('lanes') or ()))
        out[9] = 1.
        out[12] = float(int(get('anchor', -1)) >= 0)
    out[11] = float(kind == 'group_hold')
    out[13] = (float(get('depth'))-BRAKE_WINDOW_MPS2[0])/(BRAKE_WINDOW_MPS2[1]-BRAKE_WINDOW_MPS2[0])
    out[14] = float(get('hold'))/HOLD_WINDOW_S[1]
    return out


class TriggerExtension:
    """W1 extension (wall / awareness) + lane-drop columns; every new column zero-initialised.

    descriptor: [W1 descriptor | LD_DESCRIPTOR (moves on)];
    context:    [W1 context    | scaled lane-drop block (features on)]."""

    def __init__(self, base, ld):
        self.base, self.ld = base, ld
        self.wall = getattr(base, 'wall', None)
        self.aware = getattr(base, 'aware', None)
        self.lane_drop = ld
        self.families = (tuple(base.families) if base is not None else ())+extension_families(ld)
        self.ld_descriptor_width = len(LD_DESCRIPTOR) if ld['moves'] else 0
        self.ld_context_width = LANE_DROP_DIM if ld['features'] else 0
        self.descriptor_width = (base.descriptor_width if base is not None else 0)+self.ld_descriptor_width
        self.context_width = (base.context_width if base is not None else 0)+self.ld_context_width

    def descriptor(self, move):
        out = list(self.base.descriptor(move)) if self.base is not None else []
        if self.ld_descriptor_width:
            out += ld_descriptor(move, self.ld['view'])
        return out

    def context_features(self, context):
        parts = []
        if self.base is not None and self.base.context_width:
            parts.append(self.base.context_features(context))
        if self.ld_context_width:
            parts.append(context_block(context))
        out = torch.cat(parts, -1) if parts else torch.zeros((len(context['members']), 0))
        return out.reshape(len(context['members']), self.context_width)


def extension_families(ld):
    """Executable lane-drop option families (joint moves when ``moves``, protocol when ``v2v``)."""
    return (JOINT_FAMILIES if ld['moves'] else ())+(PROTOCOL_FAMILIES if ld['v2v'] is not None else ())


def context_block(context):
    """[members, LANE_DROP_DIM] scaled blocks from the recorded context (missing if absent)."""
    blocks = context.get('lane_drop') or {}
    rows = [scaled(blocks[v] if blocks.get(v) is not None else missing_block()) for v in context['members']]
    out = torch.as_tensor(np.asarray(rows, dtype=np.float32).reshape(len(rows), LANE_DROP_DIM))
    if not bool(torch.isfinite(out).all()):
        raise ValueError('nonfinite lane-drop features')
    return out


def extension(cfg):
    """Trigger extension: exactly v2v_wall.extension(cfg) when lane_drop is OFF."""
    base, ld = v2v_wall.extension(cfg), settings(cfg)
    if ld is None:
        return base
    return TriggerExtension(base, ld)


def widen_layer(layer):
    """Zero lane-drop columns before the committed-offset column of the residual (after W1)."""
    first = layer.residual[0]
    layer.residual[0] = v2v_wall.widen_columns(first, first.in_features-1, LANE_DROP_DIM)
    layer.config = replace(layer.config, feature_dim=layer.config.feature_dim+LANE_DROP_DIM)
    layer.lane_drop_width = LANE_DROP_DIM
    return layer


def attach_context(context, reader, members):
    """Bank contexts: the reader's own causal lane-drop blocks (no-op when features off)."""
    ld = getattr(reader, 'lane_drop', None)
    if ld is not None and ld['features']:
        context['lane_drop'] = {v: list(reader.lane_drop_blocks[v]) for v in members
                                if v in reader.lane_drop_blocks}
    return context


# ---------------------------------------------------------------- bank ingestion (protocol on)

def protocol_contexts(policy, plan, label, pairs, frames, *, t0):
    """Initiate/join imitation contexts for one verified lane-drop example (W1 rule).

    The initiator is the member reaching the most other members within the message range
    (ties: plan role order); it imitates ``lane_drop_initiate`` from its own observation; each
    other member imitates ``lane_drop_join`` at the earliest second its OWN receive step
    produced the offer for the reconstructed t-1 message. Yields (context, option, reader,
    weight, time) like bank_breadth.observable_contexts."""
    view = policy.lane_drop['view']
    cap = policy.ld_protocol.member_cap
    if label.family not in KINDS:
        return   # group_hold (R procedure) has no protocol form: taught as the joint label only
    members, onset, hold = label.members, label.onset, label.hold
    by_t = {}
    for sample, weight in sorted(pairs, key=lambda pair: int(pair[0]['intent_link']['t'])):
        by_t.setdefault(int(sample['intent_link']['t']), (sample, weight))
    reader = deepcopy(policy)
    reader.ld_protocol.reset()
    t0 = int(t0)
    initiator, wall_id, remaining = None, None, set()

    def build(source, keep):
        context = dict(source, members=list(keep),
                       x=[source['x'][source['members'].index(v)] for v in keep],
                       trigger_v2v={v: reader.local_intents[v] for v in keep})
        return v2v_wall.attach_context(context, reader, keep)   # W1 + N2 blocks (opt-in)

    lanes = dict(zip(members, label.lanes))
    for t in range(t0-1, t0+onset+hold):
        if t not in frames:
            continue
        reader.observe(frames[t]['vehicles'], t)
        if t < t0:
            continue
        elapsed = t-t0
        left_onset, left_hold = max(0, onset-elapsed), hold-max(0, elapsed-onset)
        if left_hold < moves._HOLD_LO:
            break
        pair = by_t.get(t)
        if initiator is None:
            if pair is None:
                continue
            source = pair[0]['context']
            rows = {r['vid']: r for r in frames[t]['vehicles']}
            candidates = [v for v in members if v in source['members'] and v in reader.local_intents]
            if not candidates:
                continue

            def reach(v):
                return sum(1 for m in members if m != v and m in rows
                           and abs(float(rows[m]['s_m'])-float(rows[v]['s_m'])) <= reader.ld_protocol.range_m)
            if label.anchor >= 0:
                # pair_t: the anchor (closing-lane CAV) must initiate (its position releases all).
                if members[label.anchor] not in candidates:
                    continue
                initiator = members[label.anchor]
            else:
                initiator = max(candidates, key=lambda v: (reach(v), -members.index(v)))
            wall_id = f'{initiator}@{t}'
            option = LaneDropOption('lane_drop_initiate', (initiator,), label.family, left_onset, left_hold,
                                    label.depth, label.target_mps, label.gap_dv_mps, label.zip,
                                    lanes[initiator], wall_id=wall_id, initiator=initiator,
                                    cap=kind_cap(label.family, view, cap), range_m=reader.ld_protocol.range_m,
                                    anchor=label.anchor >= 0)
            remaining = set(members)-{initiator}
            yield build(source, [initiator]), option, reader, float(pair[1]), t
            reader.ld_protocol.open_offline(option, t)
        elif pair is not None and remaining:
            source = pair[0]['context']
            for vid in sorted(remaining):
                if vid not in source['members'] or vid not in reader.local_intents:
                    continue
                offer = next((o for o in reader.ld_protocol.offers.get(vid, ()) if o.wall_id == wall_id), None)
                if offer is None or not reader.ld_protocol.admit_offline(offer, t):
                    continue
                remaining.discard(vid)
                yield build(source, [vid]), offer, reader, float(pair[1]), t
        if initiator is not None:
            reader.ld_protocol.publish(t, {v: r.source for v, r in reader.reports.items()})
            if not remaining:
                break
