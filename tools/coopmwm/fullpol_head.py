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

    def __getattr__(self, name):
        # Other switches of the wrapped (W1/N2/N9) extension; deepcopy probes uninitialised objects.
        if 'base' not in self.__dict__ or self.__dict__['base'] is None:
            raise AttributeError(name)
        return getattr(self.__dict__['base'], name)

    def descriptor(self, move, visible=None):
        out = list(self.base.descriptor(move)) if self.base is not None else []
        if visible is None:
            raise ValueError('FULLPOL descriptors are relative to the acting CAV slots')
        return out+descriptor(move, visible, self.view, self.slots)

    def context_features(self, context):
        if self.base is not None and self.base.context_width:
            return self.base.context_features(context)
        return torch.zeros((len(context['members']), 0))


def extension(cfg, base=None):
    """Exactly ``base`` (default ``lane_drop.extension(cfg)``) when fullpol is OFF (byte-identical)."""
    if base is None:
        from . import lane_drop
        base = lane_drop.extension(cfg)
    settings = fp.settings(cfg)
    if settings is None:
        return base
    return Extension(base, settings, int(cfg['trigger_partner_slots']))


__all__ = ['SLOT_FEATURES', 'GROUP_FEATURES', 'width', 'descriptor', 'Extension', 'extension']


# ---------------------------------------------------------------- clean5b: LCEXEC template options

TEMPLATE_CAP = 6            # lc_program.REGISTRY['cap'] max (S-LCMOVES <=4; COOPKINDS early_merge 5; 6-cap)
TEMPLATE_SLOT_FEATURES = (
    'present', 'lon_base', 'lon_native', 'lon_speed', 'lon_track', 'lon_match', 'lon_accel', 'v_w_frac', 'dmax_frac',
    'amax_frac', 'accel_frac', 'track_off_frac', 'hold_frac', 'start_frac', 'tau_set', 'tau_frac', 'lc_mode_512',
    'lane_request', 'target_lane_0', 'target_lane_1', 'target_lane_2', 'into_closing', 'lane_offset_frac',
    'lane_giveup_to_drop_frac', 'rule_hold', 'rule_until_reached', 'rule_right_of_target', 'rule_from_lane',
    'release_relative', 'release_position', 'join_any_lane', 'join_lane_0', 'join_lane_1', 'join_lane_2',
    'join_ds_lo_frac', 'join_ds_hi_frac', 'is_own_slot')
TEMPLATE_GROUP_FEATURES = ('is_initiate', 'is_join', 'slots_frac', 'duration_frac', 'member_count_frac',
                           'gap_over_range', 'own_lane_0', 'own_lane_1', 'own_lane_2')
TEMPLATE_RULES = ('hold', 'until_reached', 'right_of_target', 'from_lane')


def template_width():
    return TEMPLATE_CAP*len(TEMPLATE_SLOT_FEATURES)+len(TEMPLATE_GROUP_FEATURES)


def _template_dict(option):
    t = getattr(option, 'template', None)
    if t is None and isinstance(option, dict):
        t = option.get('template')
    return t.to_dict() if hasattr(t, 'to_dict') else dict(t)


def _template_slot(view, p, rule, own):
    lon = p['lon']
    out = [1.]+[float(lon['kind'] == k) for k in fp.LON_KINDS]
    out += [_frac(lon.get('v_w'), *V_W) if lon.get('v_w') is not None else 0.,
            _frac(lon['dmax'], *DMAX) if lon['kind'] in ('speed', 'track', 'match') else 0.,
            float(lon['amax'])/CAV_CAP_MPS2 if lon['kind'] in ('speed', 'track', 'match') else 0.,
            float(lon['a'])/CAV_CAP_MPS2 if lon['kind'] == 'accel' and lon.get('a') is not None else 0.,
            float(lon['off'])/TRACK_OFF_MAX_M if lon['kind'] == 'track' else 0.,
            float(p.get('hold_s') or 0)/HOLD_MAX_S, float(p['start_el'])/OFFSET_MAX_S,
            float(p.get('tau') is not None), 0. if p.get('tau') is None else float(p['tau'])/TAU_MAX_S,
            float(int(p['lc_mode']) == fp.LC_MODE_TRACI_ONLY)]
    q = p.get('lane')
    if q is None:
        out += [0.]*6
    else:
        t = int(q['target'])
        out += [1.]+[float(t == ln) for ln in (0, 1, 2)]+[float(t == view.dropped_lane), float(q['from_el'])/OFFSET_MAX_S]
    out += [_dist(view, None if q is None else q.get('giveup_s'))]
    out += [float(q is not None and q['rule'] == r) for r in TEMPLATE_RULES]
    rel = set(p['releases'])
    out += [float(bool(rel & {'anchor', 'partner_left', 'merged_zip', 'merger'})),
            float(bool(rel & {'release_s', 'zone', 'fallback'}))]
    if rule is None:
        out += [0.]*6
    else:
        lane = rule.get('lane')
        out += [float(lane is None)]+[float(lane == ln) for ln in (0, 1, 2)]
        out += [max(-1., min(1., float(rule['ds'][0])/SENSING_M)), max(-1., min(1., float(rule['ds'][1])/SENSING_M))]
    out += [float(own)]
    return out


def template_descriptor(option, view):
    """[TEMPLATE_CAP * slot | group] floats for an LCEXEC free_program initiate/join option (zeros
    for any other candidate). Absolute WM lane one-hots (single registered map frame); the
    acting CAV's own slot is flagged (slot 0 for an initiate, ``option.slot`` for a join)."""
    out = [0.]*template_width()
    family = None if option is None else (option.get('family') if isinstance(option, dict)
                                          else getattr(getattr(option, 'move', option), 'family', None))
    if family not in ('free_program_initiate', 'free_program_join'):
        return out
    inner = getattr(option, 'move', option)
    get = (lambda k, d=None: inner.get(k, d)) if isinstance(inner, dict) else (lambda k, d=None: getattr(inner, k, d))
    t = _template_dict(inner)
    rules = list(t.get('join_rules') or [None]*len(t['programs']))
    own_slot = 0 if family == 'free_program_initiate' else int(get('slot'))
    n = len(TEMPLATE_SLOT_FEATURES)
    for k, p in enumerate(t['programs'][:TEMPLATE_CAP]):
        out[k*n:(k+1)*n] = _template_slot(view, p, rules[k] if k < len(rules) else None, k == own_slot)
    cap = float(get('cap', len(t['programs'])) or len(t['programs']))
    rng_m = float(get('range_m', SENSING_M) or SENSING_M)
    out[TEMPLATE_CAP*n:-3] = [float(family == 'free_program_initiate'), float(family == 'free_program_join'),
                            len(t['programs'])/float(TEMPLATE_CAP), float(t['duration_s'])/HOLD_MAX_S,
                            float(get('member_count', 1) or 1)/cap, max(-1., min(1., float(get('gap_m', 0.) or 0.)/rng_m))]
    own_lane = get('lane')
    out[-3:] = [float(own_lane == ln) for ln in (0, 1, 2)]
    return out


PROGRAM_FAMILIES = ('free_program_initiate', 'free_program_join')   # registered by LCEXEC lc_families only
_VIEW = None


def _program_view(option=None):
    """The process map's lane-drop view (production binds wz3to2; fails closed on the merge map).

    Read through the option's own lc_families module when available (the same import context as
    the protocol that bound the map), else this package's map_descriptor."""
    global _VIEW
    import sys
    from .lane_drop import MapView
    owner = sys.modules.get(type(option).__module__) if option is not None else None
    if owner is not None and callable(getattr(owner, '_descriptor', None)):
        desc = owner._descriptor()
    else:
        from .map_descriptor import current
        desc = current()
    key = getattr(desc, 'digest', None) or getattr(desc, 'name', None)
    if _VIEW is None or (isinstance(_VIEW, tuple) and _VIEW[0] != key):
        _VIEW = (key, MapView.from_descriptor(desc))
    return _VIEW[1] if isinstance(_VIEW, tuple) else _VIEW


def program_descriptor(option):
    """LCEXEC lc_families descriptor hook (cfg['lc_families']['descriptor'] =
    'tools.coopmwm.fullpol_head:program_descriptor', descriptor_width = PROGRAM_DESCRIPTOR_WIDTH):
    FULLPOL's template descriptor of one ProgramOption (zeros for any other candidate)."""
    return template_descriptor(option, _program_view(option))


PROGRAM_DESCRIPTOR_WIDTH = TEMPLATE_CAP*len(TEMPLATE_SLOT_FEATURES)+len(TEMPLATE_GROUP_FEATURES)
