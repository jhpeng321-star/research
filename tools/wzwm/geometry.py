"""wz3to2 geometry (values = N3/PLANGEN pg_plan.py and N1 map_descriptor), WM-frame projection
and the seed/path guard of the WM-C data path.

Map: edge 'up' 0..2696 m with 3 lanes, SUMO lane 0 (rightmost) ends at X_B = 2696 m; edge 'wz'
has 2 lanes. N1 projection into the WM-S model frame: SUMO up lane 2 -> WM 0, up 1 -> WM 1,
up 0 (closing) -> WM 2 (the 'merge' lane); s = x - (X_B - 75.44), i.e. the lane end is s = 75.44.

Seeds: wz3to2 TRAIN worlds 989200-989599 only. HELD-OUT 989600-989699 and the evaluation ranges
962xxx / 964000-964199 / 977xxx are refused. Paths containing holdout/heldout/cal_split/chunk5/
j4b/inc1/sealed or a driver-truth token are refused before they are opened.
"""
from __future__ import annotations

from pathlib import Path

MAP = 'wz3to2'
X_B = 2696.0                                  # end of up lane 0 (pg_plan.X_B, nm2_core.X_B)
ZONE = (X_B - 1800.0, X_B - 400.0)            # pg_plan.ZONE (geometry-only member zone)
PAIR_DX = 100.0                               # pg_plan.PAIR_DX
WALL_SPREAD = 200.0                           # pg_plan.WALL_SPREAD
R_SPREAD = 300.0
H_MAIN = 300
SPEED_LIMIT = 31.29
DROP_S = 75.44                                # FrameContract.merge_zone_s_m[1]
ANCHOR_X = X_B - DROP_S
EDGE_OFFSET = {'up': 0.0, ':n1_0': 2696.0, 'wz': 2704.0}
# (edge kind, SUMO lane) -> WM lane (N1 map_descriptor.load: wm_of = {2: 0, 1: 1, 0: 2};
# downstream lanes inherit the WM lane of their upstream connection: up_1 -> wz_0, up_2 -> wz_1).
_WM_LANE = {('up', 2): 0, ('up', 1): 1, ('up', 0): 2, ('down', 1): 0, ('down', 0): 1}

TRAIN_SEEDS = (989200, 989599)
HELDOUT_SEEDS = (989600, 989699)
FORBIDDEN_SEED_RANGES = ((962000, 962999), (964000, 964199), (977000, 977999), HELDOUT_SEEDS)
FORBIDDEN_PATH_TOKENS = ('holdout', 'heldout', 'cal_split', 'chunk5', 'j4b', 'inc1', 'sealed',
                         'hetdata', 'truth', 'vehicle_params', 'driver_params', 'vtype')


class ForbiddenSeedError(ValueError):
    """A record references a held-out / evaluation / non-TRAIN world seed."""


class ForbiddenPathError(ValueError):
    """A path contains a forbidden token (never opened)."""


def check_seed(seed, where=''):
    try:
        s = int(seed)
    except (TypeError, ValueError):
        raise ForbiddenSeedError(f'unparseable world seed {seed!r} {where}'.strip())
    if any(lo <= s <= hi for lo, hi in FORBIDDEN_SEED_RANGES):
        raise ForbiddenSeedError(f'held-out/evaluation world seed {s} refused {where}'.strip())
    if not TRAIN_SEEDS[0] <= s <= TRAIN_SEEDS[1]:
        raise ForbiddenSeedError(f'world seed {s} is not a wz3to2 TRAIN world (989200-989599) {where}'.strip())
    return s


def path_allowed(path):
    low = str(path).replace('\\', '/').lower()
    return not any(tok in low for tok in FORBIDDEN_PATH_TOKENS)


def check_path(path):
    if not path_allowed(path):
        raise ForbiddenPathError(f'forbidden path refused (never opened): {path}')
    return Path(path)


def wm_lane(edge, lane):
    """WM lane of a SUMO (edge, lane); None for an unknown edge."""
    if edge == 'up':
        return _WM_LANE.get(('up', int(lane)))
    if edge == 'wz' or str(edge).startswith(':'):
        return _WM_LANE.get(('down', int(lane)))
    return None


def wm_row(cav, is_cav=True, length_m=5.0):
    """WM-frame observable row (vid, s_m, v_mps, lane, is_cav, length_m) of a CAV self-state."""
    lane = wm_lane(cav['edge'], cav['lane'])
    if lane is None:
        return None
    return dict(vid=str(cav['id']), s_m=round(float(cav['x']) - ANCHOR_X, 3), v_mps=float(cav['v']), lane=lane,
                is_cav=bool(is_cav), length_m=float(length_m))


def zone_cavs(cavs, zone=ZONE):
    """CAVs on 'up' inside the member zone (pg_plan.zone_cavs)."""
    return [c for c in cavs if c['edge'] == 'up' and zone[0] <= float(c['x']) <= zone[1]]
