"""FULLPOL trigger extension: a zero-initialised free_program descriptor block (DEFAULT OFF).

Architecture decision (FULLPOL_DESIGN.md section 3): keep the existing slot-choice trigger.
It already scores every slate entry with one shared MLP over [frozen-MAPPO hidden | candidate
descriptor | per-ego context], i.e. it is a pointer over a sampled action set and can express
ANY program once the descriptor encodes the program. FULLPOL therefore only appends a
descriptor block to the W1/N2 extension (same ``widen_columns`` zero-init mechanism): every
pre-existing logit, sample and log-probability is unchanged until the block is trained, and
with ``fullpol`` OFF ``extension(cfg)`` IS ``lane_drop.extension(cfg)``.

Descriptor (egocentric, relational, no vehicle ids, no provenance / source class, no regime
or configuration label): per partner slot in the acting CAV's own V2V slot order (ego first)
SLOT_FEATURES columns, then GROUP_FEATURES. Normalisers are the registered range maxima.
"""
import torch

from . import fullpol_programs as fp

SLOT_FEATURES = (
    'in_program', 'lon_base', 'lon_native', 'lon_speed', 'lon_track', 'lon_match', 'lon_accel',
    'v_w_frac', 'dmax_frac', 'amax_frac', 'accel_frac', 'track_off_frac', 'hold_frac', 'start_frac',
    'tau_set', 'tau_frac', 'lc_mode_512', 'lane_request', 'lane_left', 'lane_right', 'into_closing',
    'out_of_closing', 'lane_offset_frac', 'lane_arm_to_drop_frac', 'lane_giveup_to_drop_frac',
    'member_to_drop_frac', 'member_rel_s_frac', 'rule_until_reached', 'rule_right_of_target',
    'rule_from_lane', 'release_relative', 'release_position', 'join_frac')
GROUP_FEATURES = ('member_count_frac', 'duration_frac', 'lane_change_share')
V_W = (15., 22.)            # registered v_w window (NEWMAP2/LCMOVES grids)
DMAX = (1.0, 1.5)           # registered braking caps
CAV_CAP_MPS2 = 1.2          # registered shared CAV acceleration cap (trigger_policy.MoveExecutor)
HOLD_MAX_S = 300.           # longest registered hold (COOPKINDS lane_balance / zip_merge)
OFFSET_MAX_S = 60.          # longest registered start / LC offset (LCMOVES)
TAU_MAX_S = 2.4             # largest registered own tau (COOPKINDS/PLANGEN2 R grid)
DIST_MAX_M = 2400.          # farthest registered window to the drop (COOPKINDS early_merge M)
SENSING_M = 100.            # own sensing radius
TRACK_OFF_MAX_M = 100.


def width(slots):
    return slots*len(SLOT_FEATURES)+len(GROUP_FEATURES)


def _frac(x, lo, hi):
    return 0. if x is None else (float(x)-lo)/(hi-lo)


def _dist(view, s):
    return 0. if s is None else max(-1., min(2., (fp._drop(view)-float(s))/DIST_MAX_M))


def _slot(view, prog, i, lane_s, ego_s, join_el):
    p = prog['programs'][i]
    lon = p['lon']
    lane, s = lane_s
    out = [1.]+[float(lon['kind'] == k) for k in fp.LON_KINDS]
    out += [_frac(lon.get('v_w'), *V_W) if lon.get('v_w') is not None else 0.,
            _frac(lon['dmax'], *DMAX) if lon['kind'] in ('speed', 'track', 'match') else 0.,
            float(lon['amax'])/CAV_CAP_MPS2 if lon['kind'] in ('speed', 'track', 'match') else 0.,
            float(lon['a'])/CAV_CAP_MPS2 if lon['kind'] == 'accel' else 0.,
            float(lon['off'])/TRACK_OFF_MAX_M if lon['kind'] == 'track' else 0.,
            float(p['hold_s'] or 0)/HOLD_MAX_S, float(p['start_el'])/OFFSET_MAX_S,
            float(p['tau'] is not None), 0. if p['tau'] is None else float(p['tau'])/TAU_MAX_S,
            float(int(p['lc_mode']) == fp.LC_MODE_TRACI_ONLY)]
    q = p.get('lane')
    if q is None:
        out += [0.]*7
    else:
        target = int(q['target'])
        out += [1., float(target < lane), float(target > lane), float(target == view.dropped_lane),
                float(lane == view.dropped_lane and target != view.dropped_lane),
                float(q['from_el'])/OFFSET_MAX_S, _dist(view, q.get('from_s'))]
    out += [_dist(view, None if q is None else q.get('giveup_s')), _dist(view, s),
            max(-1., min(1., (float(s)-float(ego_s))/SENSING_M))]
    out += [float(q is not None and q['rule'] == r) for r in fp.LANE_RULES]
    rel = set(p['releases'])
    out += [float(bool(rel & {'anchor', 'partner_left', 'merged_zip', 'merger'})),
            float(bool(rel & {'release_s', 'zone', 'fallback'})), float(join_el)/OFFSET_MAX_S]
    return out


def descriptor(move, visible, view, slots):
    """[slots*SLOT + GROUP] floats; zeros for none / non-free_program candidates."""
    out = [0.]*width(slots)
    inner = getattr(move, 'move', move)
    if move is None or not fp.is_free_program(inner):
        return out
    option = inner if isinstance(inner, fp.FreeProgramOption) else fp.option_from_dict(inner)
    prog = option.program
    visible = list(visible)
    ego = visible[0]
    if ego not in option.observed:
        raise ValueError('free_program descriptor needs the acting CAV observed geometry')
    ego_s = option.observed[ego][1]
    n = len(SLOT_FEATURES)
    lane_changes = 0
    for i, vid in enumerate(prog['members']):
        if vid not in visible or visible.index(vid) >= slots:
            raise ValueError('free_program members must occupy the acting CAV observable slots')
        j = visible.index(vid)
        out[j*n:(j+1)*n] = _slot(view, prog, i, option.observed[vid], ego_s, prog['join_el'][i])
        lane_changes += prog['programs'][i].get('lane') is not None
    k = len(prog['members'])
    out[slots*n:] = [k/float(slots), float(prog['duration_s'])/HOLD_MAX_S, lane_changes/float(k)]
    return out


class Extension:
    """Base W1/N2 extension (or None) + the FULLPOL descriptor block (zero-initialised)."""
    relative = True   # trigger_policy passes the acting CAV's visible slot order

    def __init__(self, base, settings, slots):
        self.base, self.settings, self.slots = base, settings, slots
        self.view = fp.view_of(settings)
        self.families = (tuple(base.families) if base is not None else ())+fp.extension_families(settings)
        self.fp_width = width(slots)
        self.descriptor_width = (base.descriptor_width if base is not None else 0)+self.fp_width
        self.context_width = base.context_width if base is not None else 0
        # bank readers (v2v_wall.attach_context) look up the W1/N2 switches on the extension
        self.wall = getattr(base, 'wall', None)
        self.aware = getattr(base, 'aware', None)
        self.lane_drop = getattr(base, 'lane_drop', None)

    def descriptor(self, move, visible=None):
        out = list(self.base.descriptor(move)) if self.base is not None else []
        if visible is None:
            raise ValueError('FULLPOL descriptors are relative to the acting CAV slots')
        return out+descriptor(move, visible, self.view, self.slots)

    def context_features(self, context):
        if self.base is not None and self.base.context_width:
            return self.base.context_features(context)
        return torch.zeros((len(context['members']), 0))


def extension(cfg):
    """Exactly ``lane_drop.extension(cfg)`` when fullpol is OFF (byte-identical)."""
    from . import lane_drop
    base = lane_drop.extension(cfg)
    settings = fp.settings(cfg)
    if settings is None:
        return base
    return Extension(base, settings, int(cfg['trigger_partner_slots']))


__all__ = ['SLOT_FEATURES', 'GROUP_FEATURES', 'width', 'descriptor', 'Extension', 'extension']
