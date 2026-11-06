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
FAMILIES = ('abreast_wall', 'group_hold', 'closing_pair')
SLOTS = ('group_hold:solo', 'group_hold:pair')       # one F2 slot per group size (like N2 initiate kinds)
SOLO_VARIANTS = ('wall_g', 'wall_h')                  # S-REGIME solo strategies hold these variants
SOLO_GEOMETRY = ('zone', 'wall')


def settings(cfg):
    """None when OFF; else the validated common B = R clean5a slot."""
    raw = (cfg or {}).get(KEY, False)
    if raw is False or raw is None:
        return None
    keys = {'enabled', 'rule', 'provenance', 'families', 'group_sizes', 'solo_geometry', 'learn',
            'regime_guidance', 'parity_families', 'sumo_lanes'}
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
    out = deepcopy(raw)
    out.update(families=families, group_sizes=sizes, solo_geometry=raw.get('solo_geometry', 'zone'))
    return out


def config_keys(*, provenance, families=('abreast_wall', 'group_hold'), group_sizes=(1, 2), solo_geometry='zone',
                learn=None, regime_guidance=None, parity_families=None, sumo_lanes=None):
    out = dict(enabled=True, rule=RULE, provenance=provenance, families=list(families),
               group_sizes=list(group_sizes), solo_geometry=solo_geometry)
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
        return cls(data['family'], tuple(data['members']), int(data['onset']), int(data['hold']),
                   float(data['depth']), float(data['target_mps']), float(data.get('gap_dv_mps', 0.)),
                   bool(data.get('zip', False)), tuple(data.get('lanes') or ()), int(data.get('anchor', -1)))


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
            return (f'{v}_solo{lane}',) if v in SOLO_VARIANTS else ('solo_unregistered_params',)
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
    if event:
        return tuple(k for k in keep if k == 'lane_drop_join')
    if 'group_hold' in allowed:
        keep += [s for s, k in zip(SLOTS, (1, 2)) if k in ev['group_sizes']]
    return tuple(keep)


def owns(key):
    return key in SLOTS


def _in_zone(view, row):
    return ld._in_window(view, row, ld.ZONE_WINDOW_M)


def _solo_ok(ev, view, own, partners):
    if not _in_zone(view, own):
        return False
    if ev['solo_geometry'] == 'zone':
        return True
    return bool(ld.joint_groups('abreast_wall', view, own, partners))


def _solo(view, own, rng):
    v = ld.VARIANTS[SOLO_VARIANTS[int(rng.integers(len(SOLO_VARIANTS)))]]
    return SoloHold('group_hold', (own['vid'],), ld.ONSET_S, int(v['hold']), float(v['depth']),
                    float(v['target_mps']), 0., False, (int(own['lane']),))


def _pairs(view, own, partners):
    if not _in_zone(view, own):
        return []
    zone = [p for p in partners if _in_zone(view, p)]
    return [(own, p) for (p,) in combinations(zone, 1) if ld._spread((own, p)) <= ld.GROUP_SPREAD_M]


def _pair(view, group, rng):
    onset, hold, depth, target, _, _, _ = ld._params(rng, 'group_hold')
    return LaneDropMove('group_hold', tuple(r['vid'] for r in group), onset, hold, depth, target, 0., False,
                        tuple(int(r['lane']) for r in group))


def pool(key, ev, view, own, partners, rng):
    if key == 'group_hold:solo':
        return [_solo(view, own, rng)] if _solo_ok(ev, view, own, partners) else []
    if key == 'group_hold:pair':
        return [_pair(view, g, rng) for g in _pairs(view, own, partners)]
    raise ValueError(f'unregistered clean5a slot {key!r}')


def redraw(template, key, ev, view, own, by_vid, rng):
    """Fresh registered parameters for the same member set (never None for a pooled set)."""
    if key == 'group_hold:solo':
        return _solo(view, own, rng)
    return _pair(view, [by_vid[v] for v in template.members], rng)


__all__ = ['KEY', 'RULE', 'settings', 'config_keys', 'SoloHold', 'is_solo', 'option_from_dict', 'labels',
           'variant_of', 'sumo_lane', 'families', 'owns', 'pool', 'redraw']


PARITY_FAMILIES = ('wall_g', 'wall_h')+tuple(f'wall_{v}_solo{k}' for v in 'gh' for k in (0, 1, 2))
PARITY_NOTE = ('N2 exact executor (N5 PASS 59d8bace: 81/81 mined hold rows bind exactly); a solo hold runs the '
               'identical N2 hold law for one member (SoloHold), so wall_{g,h}_solo{L} share that executor')


def recommended(*, provenance, regime_guidance, learn=None):
    """MAIN clean5a recipe (common B = R key; R simply has no R rows in the guidance):
    families abreast_wall + group_hold (solo, pair); EV-BC ON, EV credit OFF (PPO on team
    reward + G4LD credit carry the RL part); report-only receipts ON."""
    return config_keys(provenance=provenance, families=('abreast_wall', 'group_hold'), group_sizes=(1, 2),
                       solo_geometry='zone', regime_guidance=regime_guidance, parity_families=PARITY_FAMILIES,
                       learn=learn or dict(ev_bc=True, ev_credit=False, report_kl_unlabelled=True,
                                           report_valsplit_auc=True))
