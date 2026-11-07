"""clean5a: speed-only expected-value mode on the N2 lane-drop families (DEFAULT OFF).

MAIN 2026-11-06 ~19:3x (owner framing "learn WHEN to cooperate - conditions, not moments"):
the clean5a candidate slate is N2 families only - ``group_hold`` with ONE member (the solo
hold; S-REGIME solo strategies wall_{g,h}_solo{L}), ``group_hold`` with two members (the N2
R_random hold procedure), ``abreast_wall`` (NEWMAP2 wall_g / wall_h, incl. the N2 V2V
initiate/join protocol when lane_drop v2v is on) - plus ``none``. No LCEXEC dependency.

``group_hold`` becomes a runtime proposal family ONLY behind the ``clean5_ev`` key; with the key
absent/False every output is byte-identical (clean4 is untouched). Execution is N2's own
closed-loop speed law (lane_drop.member_command) on the single-writer committed-offset column;
a solo hold is a 1-member ``LaneDropMove`` subclass (``SoloHold``) so it runs through exactly
the same executor, release rules (own drop crossing / hold end) and trigger descriptor.

Registered values only: solo parameters = the NEWMAP2 wall_g / wall_h variants (the S-REGIME
solo strategies hold exactly those), pair parameters = N2 R_SETS, zone [X_B-1800, X_B-400],
pair spread 300 m (NEWMAP2 R spread). Learning signals live in fullpol_learn (EV-BC on recorded
slates, report-only receipts) - the regime label is an offline training label only.
"""
from copy import deepcopy
from itertools import combinations

from . import lane_drop as ld
from .lane_drop import LaneDropMove

KEY = 'clean5_ev'
RULE = 'fullpol-clean5a-ev-n2-v1'
PROVENANCE = ('MAIN 2026-11-06 ~19:3x clean5a (speed-only EV mode on N2 families, no LCEXEC dependency); '
              'owner framing "learn WHEN to cooperate"; S-REGIME solo strategies (REGISTRATION.md 18:45)')
FAMILIES = ('abreast_wall', 'group_hold', 'closing_pair', 'free_program')   # free_program = clean5b (LCEXEC)
PROGRAM_KEYS = ('free_program_initiate', 'free_program_join')
SLOTS = ('group_hold:solo', 'group_hold:pair')       # one F2 slot per group size (like N2 initiate kinds)
DUO_SLOT = 'group_hold:duo'                            # clean5b-F (fair.duo12): wall_g 2-member hold, SUMO lanes 1+2
DUO_LANES_SUMO = (1, 2)
GATE_BEHIND_M = 100.                                   # gate0: no vehicle within 100 m behind in own lane (sensing)
SOLO_VARIANTS = ('wall_g', 'wall_h')                  # S-REGIME solo strategies hold these variants
SOLO_GEOMETRY = ('zone', 'wall')


def settings(cfg):
    """None when OFF; else the validated common B = R clean5a slot."""
    raw = (cfg or {}).get(KEY, False)
    if raw is False or raw is None:
        return None
    keys = {'enabled', 'rule', 'provenance', 'families', 'group_sizes', 'solo_geometry', 'learn',
            'regime_guidance', 'parity_families', 'sumo_lanes', 'ev_bc_steps', 'ppo_kl', 'hold_lane_pin', 'fair'}
    if (not isinstance(raw, dict) or raw.get('enabled') is not True or raw.get('rule') != RULE
            or set(raw)-keys or not isinstance(raw.get('provenance'), str) or not raw['provenance'].strip()):
        raise ValueError('clean5_ev requires enabled=True, the registered rule, provenance and known keys')
    families = tuple(raw.get('families', ('abreast_wall', 'group_hold')))
    sizes = tuple(raw.get('group_sizes', (1, 2)))
    if not families or any(f not in FAMILIES for f in families) or len(set(families)) != len(families):
        raise ValueError(f'clean5_ev families must be distinct entries of {FAMILIES}')
    if not sizes or any(k not in (1, 2) for k in sizes):
        raise ValueError('clean5_ev group sizes are 1 (solo hold) and/or 2 (pair hold)')
    if raw.get('solo_geometry', 'zone') not in SOLO_GEOMETRY:
        raise ValueError(f'clean5_ev solo_geometry in {SOLO_GEOMETRY}')
    if ld.settings(cfg) is None or not ld.settings(cfg)['moves']:
        raise ValueError('clean5_ev requires the N2 lane_drop moves')
    if 'ev_bc_steps' in raw and (type(raw['ev_bc_steps']) is not int or raw['ev_bc_steps'] < 1):
        raise ValueError('clean5_ev ev_bc_steps must be a positive integer (registered)')
    if raw.get('ppo_kl', 'row_sum') not in PPO_KL:
        raise ValueError(f'clean5_ev ppo_kl in {PPO_KL}')
    if type(raw.get('hold_lane_pin', False)) is not bool:
        raise ValueError('clean5_ev hold_lane_pin must be bool')
    fair = raw.get('fair')
    if fair:
        from .fullpol_fair import RULE as FAIR_RULE, THRESHOLD_S
        if (not isinstance(fair, dict) or fair.get('rule') != FAIR_RULE
                or float(fair.get('threshold_s', -1)) != THRESHOLD_S
                or set(fair)-{'rule', 'threshold_s', 'solo_variants', 'duo12', 'gate0', 'ev_filter', 'reward'}
                or any(v not in SOLO_VARIANTS for v in fair.get('solo_variants', SOLO_VARIANTS))
                or any(type(fair.get(k, False)) is not bool for k in ('duo12', 'gate0', 'ev_filter', 'reward'))):
            raise ValueError('clean5_ev.fair: registered rule, threshold_s 10, solo_variants and boolean switches')
    out = deepcopy(raw)
    out.update(families=families, group_sizes=sizes, solo_geometry=raw.get('solo_geometry', 'zone'))
    return out


PPO_KL = ('row_sum', 'row_mean')   # trigger-PPO KL guard: inherited row-sum (OFF) / per-CAV-row mean
EV_BC_STEPS = 60                    # MAIN ~21:1x registered: EV-BC optimiser steps per update (lr, cap unchanged)
PACE_RULE = ('MAIN 2026-11-06 ~21:1x (pre-launch, registered): EV-BC imitation steps 60 at the registered lr 3e-4 '
             'and KL cap 0.045 (per-row-mean semantics, 24080ea2); trigger-PPO KL guard = per-CAV-row MEAN capped '
             'at the registered 0.045 instead of the row SUM (B and R identically; FULLPOL pace fixture: 20 steps '
             'reached P(target) 0.8 only at u24, 60 steps by u9)')


def config_keys(*, provenance, families=('abreast_wall', 'group_hold'), group_sizes=(1, 2), solo_geometry='zone',
                learn=None, regime_guidance=None, parity_families=None, sumo_lanes=None, ev_bc_steps=None,
                ppo_kl=None, hold_lane_pin=None):
    out = dict(enabled=True, rule=RULE, provenance=provenance, families=list(families),
               group_sizes=list(group_sizes), solo_geometry=solo_geometry)
    if ev_bc_steps is not None:
        out['ev_bc_steps'] = int(ev_bc_steps)
    if ppo_kl is not None:
        out['ppo_kl'] = ppo_kl
    if hold_lane_pin is not None:
        out['hold_lane_pin'] = bool(hold_lane_pin)
    for key, value in (('learn', learn), ('regime_guidance', regime_guidance),
                       ('parity_families', None if parity_families is None else sorted(parity_families)),
                       ('sumo_lanes', sumo_lanes)):
        if value is not None:
            out[key] = deepcopy(value)
    return {KEY: out}


# ---------------------------------------------------------------- the solo hold

class SoloHold(LaneDropMove):
    """``group_hold`` with ONE member: N2's registered hold law, release and descriptor.

    Same fields, same executor (lane_drop.member_command dispatches on LaneDropMove), same
    trigger descriptor (ld_descriptor kind group_hold). Only the arity check differs."""

    def __post_init__(self):
        object.__setattr__(self, 'members', tuple(self.members))
        object.__setattr__(self, 'lanes', tuple(self.lanes))
        if self.family != 'group_hold' or len(self.members) != 1 or len(self.lanes) != 1:
            raise ValueError('a solo hold is group_hold with exactly one member and its lane')
        if not isinstance(self.members[0], str) or not self.members[0]:
            raise ValueError('solo hold member must be a vehicle id')
        if type(self.lanes[0]) is not int or self.lanes[0] not in range(ld.N_LANES):
            raise ValueError('solo hold needs its observed lane')
        ld._check_schedule(self.onset, self.hold, self.depth)
        ld._check_speeds(self.target_mps, self.gap_dv_mps)
        if self.zip or float(self.gap_dv_mps) != 0. or self.anchor != -1:
            raise ValueError('a solo hold holds ONE common speed, no zip/gap/anchor')

    @classmethod
    def from_dict(cls, data):
        out = cls(data['family'], tuple(data['members']), int(data['onset']), int(data['hold']),
                  float(data['depth']), float(data['target_mps']), float(data.get('gap_dv_mps', 0.)),
                  bool(data.get('zip', False)), tuple(data.get('lanes') or ()), int(data.get('anchor', -1)))
        if data.get('gate0'):
            object.__setattr__(out, 'gate0', True)
        return out

    def to_dict(self):
        out = super().to_dict()
        if getattr(self, 'gate0', False):
            out['gate0'] = True           # clean5b-F: offered while the observable gate0 condition held
        return out


def is_solo(data):
    return isinstance(data, dict) and data.get('family') == 'group_hold' and len(data.get('members') or ()) == 1


def option_from_dict(data):
    """Recorded table entry -> option (1-member group_hold -> SoloHold, else N2/W1/vocabulary)."""
    return SoloHold.from_dict(data) if is_solo(data) else ld.option_from_dict(data)


# ---------------------------------------------------------------- labels (offline EV matching only)

def sumo_lane(view, wm_lane, ev=None):
    """SUMO index on the approach edge of a WM lane (closing lane = SUMO 0, left = SUMO 2)."""
    table = (ev or {}).get('sumo_lanes')
    if table is not None:
        return int(table[int(wm_lane)])
    return max(view.lane_ids)-int(wm_lane)


VARIANT_FAMILY = dict(abreast_wall='wall_', closing_pair='pair_', group_hold='wall_')   # solo = wall variants


def variant_of(entry, family=None):
    """NEWMAP2 variant of the entry's family whose registered (target, hold, depth, anchor) it holds.

    pair_g/wall_g and pair_h/wall_h share parameters, so the variant name is family-scoped
    (a solo hold is a single wall member: S-REGIME wall_{g,h}_solo{L})."""
    family = entry.get('family') if family is None else family
    if family in ld.PROTOCOL_FAMILIES:
        family = entry.get('kind')
    prefix = VARIANT_FAMILY.get(family)
    if prefix is None:
        return None
    a = entry.get('anchor', -1)
    anchored = a is True or (type(a) is int and a >= 0)   # joint: role index; protocol: bool
    for name, v in sorted(ld.VARIANTS.items()):
        if (name.startswith(prefix) and float(entry.get('target_mps', -1)) == float(v['target_mps'])
                and int(entry.get('hold', -1)) == int(v['hold']) and float(entry.get('depth', -1)) == float(v['depth'])
                and anchored == bool(v['anchor'])):
            return name
    return None


def labels(entry, view, ev=None):
    """Strategy labels of one recorded N2 entry (S-REGIME names), most specific first."""
    family = entry.get('family')
    if family == 'group_hold':
        if len(entry['members']) == 1:
            v = variant_of(entry)
            lane = sumo_lane(view, entry['lanes'][0], ev)
            # Lane-specific ONLY (MAIN 22:1x): a solo carries the EV of the solo strategy of its own
            # current SUMO lane, never a generic solo/group row.
            if v not in SOLO_VARIANTS:
                return ('solo_unregistered_params',)
            if entry.get('gate0'):
                return (f'{v}_solo{lane}_gate0', f'{v}_solo{lane}')
            return (f'{v}_solo{lane}',)
        if (ev or {}).get('fair') and len(entry['members']) == 2 and variant_of(entry, 'abreast_wall') == 'wall_g' \
                and sorted(sumo_lane(view, x, ev) for x in entry['lanes']) == list(DUO_LANES_SUMO):
            return ('wall_g_duo12', 'group_hold_pair', 'group_hold')
        return ('group_hold_pair', 'group_hold')
    if family in ('abreast_wall', 'closing_pair'):
        v = variant_of(entry)
        return ((v,) if v else ())+(family,)
    if family in ld.PROTOCOL_FAMILIES:
        kind = entry.get('kind')
        v = variant_of(entry)
        return ((v,) if v else ())+((kind,) if kind else ())+(family,)
    return (family,)


# ---------------------------------------------------------------- slate hooks (feasible_proposals)

def families(structured, ev, event):
    """Only the clean5a N2 keys: abreast_wall (+ its V2V initiate slot / joins), closing_pair if
    listed, and one F2 slot per enabled group size of group_hold. Event ticks keep joins only."""
    allowed = set(ev['families'])
    keep = []
    for key in structured:
        if key in ('abreast_wall', 'closing_pair') and key in allowed:
            keep.append(key)
        elif key.startswith('lane_drop_initiate:') and key.split(':', 1)[1] in allowed:
            keep.append(key)
        elif key == 'lane_drop_join' and allowed & {'abreast_wall', 'closing_pair'}:
            keep.append(key)
        elif key in PROGRAM_KEYS and 'free_program' in allowed:
            keep.append(key)               # clean5b: LCEXEC per-CAV program initiate / join slots
    if event:
        return tuple(k for k in keep if k in ('lane_drop_join', 'free_program_join'))
    if 'group_hold' in allowed:
        keep += [s for s, k in zip(SLOTS, (1, 2)) if k in ev['group_sizes']]
        if (ev.get('fair') or {}).get('duo12'):
            keep.append(DUO_SLOT)
    return tuple(keep)


def owns(key):
    return key in SLOTS or key == DUO_SLOT


def _in_zone(view, row):
    return ld._in_window(view, row, ld.ZONE_WINDOW_M)


def _solo_ok(ev, view, own, partners):
    if not _in_zone(view, own):
        return False
    if ev['solo_geometry'] == 'zone':
        return True
    return bool(ld.joint_groups('abreast_wall', view, own, partners))


def _gate0(own, rows):
    """Observable gate0: no vehicle (any type) within GATE_BEHIND_M behind in the own lane (own sensing)."""
    s, lane = float(own['s_m']), int(own['lane'])
    return not any(r['vid'] != own['vid'] and int(r['lane']) == lane and 0. < s-float(r['s_m']) <= GATE_BEHIND_M
                   for r in rows)


def _solo(view, own, rng, ev=None, rows=None):
    fair = (ev or {}).get('fair') or {}
    variants = tuple(fair.get('solo_variants', SOLO_VARIANTS))
    v = ld.VARIANTS[variants[int(rng.integers(len(variants)))]]
    out = SoloHold('group_hold', (own['vid'],), ld.ONSET_S, int(v['hold']), float(v['depth']),
                   float(v['target_mps']), 0., False, (int(own['lane']),))
    if fair.get('gate0') and rows is not None and _gate0(own, rows):
        object.__setattr__(out, 'gate0', True)
    return out


def _duos(view, own, partners, ev):
    lanes = {max(view.lane_ids)-k for k in DUO_LANES_SUMO} if (ev or {}).get('sumo_lanes') is None else {
        ln for ln in view.lane_ids if sumo_lane(view, ln, ev) in DUO_LANES_SUMO}
    if not _in_zone(view, own) or int(own['lane']) not in lanes:
        return []
    other = (lanes-{int(own['lane'])})
    return [(own, p) for p in partners if _in_zone(view, p) and int(p['lane']) in other
            and ld._spread((own, p)) <= ld.GROUP_SPREAD_M]


def _duo(group):
    v = ld.VARIANTS['wall_g']
    return LaneDropMove('group_hold', tuple(r['vid'] for r in group), ld.ONSET_S, int(v['hold']), float(v['depth']),
                        float(v['target_mps']), 0., False, tuple(int(r['lane']) for r in group))


def _pairs(view, own, partners):
    if not _in_zone(view, own):
        return []
    zone = [p for p in partners if _in_zone(view, p)]
    return [(own, p) for (p,) in combinations(zone, 1) if ld._spread((own, p)) <= ld.GROUP_SPREAD_M]


def _pair(view, group, rng):
    onset, hold, depth, target, _, _, _ = ld._params(rng, 'group_hold')
    return LaneDropMove('group_hold', tuple(r['vid'] for r in group), onset, hold, depth, target, 0., False,
                        tuple(int(r['lane']) for r in group))


def pool(key, ev, view, own, partners, rng, by_vid=None):
    rows = None if by_vid is None else list(by_vid.values())
    if key == 'group_hold:solo':
        return [_solo(view, own, rng, ev, rows)] if _solo_ok(ev, view, own, partners) else []
    if key == 'group_hold:pair':
        return [_pair(view, g, rng) for g in _pairs(view, own, partners)]
    if key == DUO_SLOT:
        return [_duo(g) for g in _duos(view, own, partners, ev)]
    raise ValueError(f'unregistered clean5a slot {key!r}')


def redraw(template, key, ev, view, own, by_vid, rng):
    """Fresh registered parameters for the same member set (never None for a pooled set)."""
    if key == 'group_hold:solo':
        return _solo(view, own, rng, ev, list(by_vid.values()))
    if key == DUO_SLOT:
        return _duo([by_vid[v] for v in template.members])
    return _pair(view, [by_vid[v] for v in template.members], rng)


__all__ = ['KEY', 'RULE', 'settings', 'config_keys', 'SoloHold', 'is_solo', 'option_from_dict', 'labels',
           'variant_of', 'sumo_lane', 'families', 'owns', 'pool', 'redraw']


SOLO_LABELS = tuple(f'wall_{v}_solo{k}' for v in 'gh' for k in (0, 1, 2))
PARITY_FAMILIES = SOLO_LABELS
EV_TARGET_RULE = ('MAIN 2026-11-06 ~20:4x: EV-BC TARGETS are the solo strategies only (solo2 dominates the '
                  'abreast wall in every u0 cell: group - best solo > 0 in 5/6 cells); walls and 2-member holds '
                  'stay OFFERED in the slate so PPO and G4 can still discover group moves where they help')
PARITY_NOTE = ('N2 exact executor (N5 PASS 59d8bace: 81/81 mined hold rows bind exactly); a solo hold runs the '
               'identical N2 hold law for one member (SoloHold), so wall_{g,h}_solo{L} share that executor')


def recommended(*, provenance, regime_guidance, learn=None):
    """MAIN clean5a recipe (common B = R key; R simply has no R rows in the guidance):
    families abreast_wall + group_hold (solo, pair) OFFERED; EV-BC targets = the lane-specific solo
    strategies only (EV_TARGET_RULE); EV-BC ON, EV credit OFF (PPO on team reward + G4LD credit
    carry the RL part, incl. group moves); report-only receipts ON."""
    return config_keys(provenance=provenance, families=('abreast_wall', 'group_hold'), group_sizes=(1, 2),
                       solo_geometry='zone', regime_guidance=regime_guidance, parity_families=PARITY_FAMILIES,
                       ev_bc_steps=EV_BC_STEPS, ppo_kl='row_mean',
                       learn=learn or dict(ev_bc=True, ev_credit=False, report_kl_unlabelled=True,
                                           report_valsplit_auc=True))


GENERATOR = 'tools.coopmwm.fullpol_programs:propose'
DESCRIPTOR_HOOK = 'tools.coopmwm.fullpol_head:program_descriptor'


def recommended_5b(*, provenance, regime_guidance, cap, range_m=None, templates=(), classes=None, structured=None,
                   learn=None):
    """clean5b recipe = clean5a recipe + LCEXEC per-CAV free_program slots with FULLPOL's template
    generator and descriptor. ``templates`` = THIS arm's own verified slot templates (B: converted
    verified rows; R: none - the only B/R difference). EV-BC targets stay the registered solo labels
    unless MAIN registers program families with LCEXEC parity (parity_families)."""
    from . import fullpol_head, fullpol_templates, lc_families
    out = recommended(provenance=provenance, regime_guidance=regime_guidance, learn=learn)
    out[KEY]['families'] = ['abreast_wall', 'group_hold', 'free_program']
    args = dict(classes=list(classes or fullpol_templates.CLASSES),
                structured=list(structured or fullpol_templates.STRUCTURED),
                templates=({k: list(v) for k, v in templates.items()} if isinstance(templates, dict)
                           else list(templates)))
    out.update(lc_families.config_keys(provenance=provenance, cap=cap, range_m=range_m, generator=GENERATOR,
                                       generator_args=args, descriptor=DESCRIPTOR_HOOK,
                                       descriptor_width=fullpol_head.PROGRAM_DESCRIPTOR_WIDTH))
    return out


HOLD_LANE_PIN_RULE = ('MAIN 2026-11-07 ~01:0x (registered): during an ACTIVE N2 hold (the N2 law of the member is '
                      'commanding this second) each member gets its own current lane as a G5 lane target every '
                      'tick through the existing single-writer G5 path - the same mechanism as the verified plant '
                      'path (S-REGIMEU0 RuleHook: laneChangeMode 513 on hold members, restored at release). No '
                      'lane change is requested (target = own lane); zip members keep their N2 zip request.')


def hold_pins(active, own_lanes, zip_members=()):
    """{vid: own current WM lane} for the active N2 hold members (``active``: vids whose N2 law is
    commanding this second, incl. unresolved-native ones); zip-request members and vehicles without
    an own lane observation are excluded. Pure function of the member's OWN state."""
    zips = set(zip_members)
    return {v: int(own_lanes[v]) for v in sorted(set(active)) if v not in zips and v in own_lanes}


FAIR_PARITY = ('wall_g_solo2', 'wall_g_solo2_gate0', 'wall_g_duo12')


def fair_slot(*, solo_variants=('wall_g',), duo12=True, gate0=True, ev_filter=True, reward=True):
    """clean5b-F registered slot (owner 04:0x): wall_h (firm 15 m/s) solos never offered."""
    from .fullpol_fair import RULE as FAIR_RULE, THRESHOLD_S
    return dict(rule=FAIR_RULE, threshold_s=THRESHOLD_S, solo_variants=list(solo_variants), duo12=duo12, gate0=gate0,
                ev_filter=ev_filter, reward=reward)


def with_fairness(cfg_part, *, regime_guidance):
    """Add clean5b-F to a clean5_ev config value (e.g. recommended_5b(...)): fair slot, fairness-flagged
    guidance (fullpol_fair.build_guidance), parity = the fairness-eligible strategy labels, pin ON."""
    out = deepcopy(cfg_part)
    out[KEY].update(fair=fair_slot(), regime_guidance=deepcopy(regime_guidance),
                    parity_families=sorted(FAIR_PARITY), hold_lane_pin=True)
    return out
