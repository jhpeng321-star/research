"""A75 hold-forms search extension (S-HOLDFORMS search side, 2026-11-13). Zero SUMO.

Wraps the frozen search code ``tools.wzprop.strategy_search`` (code_v12r7b, sha256 02e119c1...; NOT edited) so that
the evolutionary WM search can propose more hold forms. ``install(cfg)`` patches the module's generator / validation /
key / descriptor / selection / member-spec functions; every patch delegates to the original code object when the
cfg has none of the A75 keys, so an old-grid cfg generates byte-identical candidates (tests/test_a75ss.py).

A75 cfg keys (all optional; absent = old behaviour):
  search_grid         may now hold values outside the registered grids (validation accepts them)
  zones               list of named zones the generator samples (full/near/far/mid/merge; executor A75 zones)
  zone_d_grid         {"p", "d_up", "d_down", "min_width"}: with probability p a plan gets zone 'custom' + zone_d
  cooldown_grid       list of cooldowns (s) sampled (only values >= max(onset+hold)//2, the registered rule)
  max_roles_mutation  mutants with more roles are rejected
  holdforms           per-role new fields (SPEC_HOLDFORMS.md): {"p_gap", "gap": {...grids}, "p_dv_rel", "dv_rel",
                      "dv_floor", "p_taper", "taper_in_s", "taper_out_s", "p_pulse", "pulse_on_s", "pulse_off_s"}
  descriptor_v2       MAP-Elites descriptor + (hold-length bin, low-speed flag, form letters)
  rank_activated_first plans that never activate on any scored state rank after every activated plan (parents,
                      archives, finalists, final order); select_for_sumo already drops them
Field names / defaults / refusal rules mirror the executor ``tools/coopmwm/a75_holdforms.py`` (SPEC v1); the key is
its strategy_key (oracle_plan key + zone_d when set; roles carry non-default new fields only).

Driver (GPU box, from the code tree code_split/code_v12r7b):
  python -m a75ss --states data/regime_STATES.json --cell D2800_p60 --config CFG.json --out OUT --ckpt wm_s.pt
WM evaluation goes through ``a75_wmforms:evaluate`` (hold-form controller + E1-FIX common random numbers).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np

from tools.wzprop import families as F
from tools.wzprop import geometry as G
from tools.wzprop import strategy_search as SS

SCHEMA = 'a75.holdforms.search.v1'
X_B = G.X_B
# executor a75_holdforms.ZONES (oracle_plan.ZONES + mid / merge)
A75_ZONES = {'full': (X_B - 1800.0, X_B - 400.0), 'near': (X_B - 1200.0, X_B - 400.0),
             'far': (X_B - 1800.0, X_B - 1000.0), 'mid': (X_B - 1400.0, X_B - 600.0),
             'merge': (X_B - 600.0, X_B - 100.0)}
ZONE_ORDER = ('far', 'full', 'mid', 'near', 'merge', 'custom')
ZONE_D_MAX = 2400.0
ROLE_DEFAULTS = dict(start='onset', gap_dx_lo=None, gap_dx_hi=None, gap_drop_m=None, gap_arm_s=None,
                     gap_release='timer', gap_events=1, dv_rel=None, taper_in_s=0, taper_out_s=0,
                     pulse_on_s=None, pulse_off_s=None)
ROLE_TYPES = dict(start=str, gap_dx_lo=float, gap_dx_hi=float, gap_drop_m=float, gap_arm_s=int, gap_release=str,
                  gap_events=int, dv_rel=float, taper_in_s=int, taper_out_s=int, pulse_on_s=int, pulse_off_s=int)
OLD_ROLE_KEYS = ('lane', 'rank', 'v_w', 'dmax', 'hold_s', 'onset_s', 'lc_target')
HOLD_S_RANGE = (1, 900)
V_W_MAX = 33.0
GAP_SUMO_LANE = 1
GAP_DX_MAX = 100.0
FORM_LETTERS = (('start', 'g'), ('dv_rel', 'd'), ('taper_in_s', 't'), ('taper_out_s', 't'), ('pulse_on_s', 'p'))

ORIG = {n: getattr(SS, n) for n in ('grids', 'validate', 'strategy_key', 'random_strategy', 'mutate', 'descriptor',
                                     'struct_vec', 'select_members', 'member_specs', 'rank_key')}
ORIG_ZONES = dict(SS.ZONES)
_A = {}


# ------------------------------------------------------------------------------------------ executor mirrors (pure)
def _cast(k, v):
    return None if v is None else ROLE_TYPES[k](v)


def canonical_role(r):
    """Role with new fields cast to their types and default-valued ones dropped (executor canonical())."""
    out = {k: v for k, v in r.items() if k not in ROLE_DEFAULTS}
    for k in ROLE_DEFAULTS:
        if k in r:
            v = _cast(k, r[k])
            if v != ROLE_DEFAULTS[k]:
                out[k] = v
    return out


def role_field(r, k):
    return r.get(k, ROLE_DEFAULTS[k])


def is_extended(s):
    return (s.get('zone_d') is not None or s.get('zone') not in ORIG_ZONES
            or any(k in r for r in s['roles'] for k in ROLE_DEFAULTS))


def zone_bounds(s):
    if s.get('zone_d') is not None:
        d_up, d_down = s['zone_d']
        return (X_B - float(d_up), X_B - float(d_down))
    return A75_ZONES[s['zone']]


def _int_ok(v, lo, hi=None):
    return isinstance(v, int) and not isinstance(v, bool) and v >= lo and (hi is None or v <= hi)


def holdform_refusal(s):
    """executor a75_holdforms.holdform_refusal (None = executable)."""
    zd = s.get('zone_d')
    if zd is not None:
        if s.get('zone') != 'custom' or len(zd) != 2 or not 0.0 <= float(zd[1]) < float(zd[0]) <= ZONE_D_MAX:
            return 'bad_zone'
    elif s.get('zone') not in A75_ZONES:
        return 'bad_zone'
    ext = is_extended(s)
    for r in s['roles']:
        if set(r) - set(OLD_ROLE_KEYS) - set(ROLE_DEFAULTS):
            return 'bad_holdform'
        if not _int_ok(r['hold_s'], *HOLD_S_RANGE) or not _int_ok(r['onset_s'], 0):
            return 'bad_holdform'
        if not 0.0 <= float(r['v_w']) <= V_W_MAX or not float(r['dmax']) > 0.0:
            return 'bad_holdform'
        f = dict(ROLE_DEFAULTS, **{k: r[k] for k in ROLE_DEFAULTS if k in r})
        if ext and any(x.get('lc_target') is not None for x in s['roles']):
            return 'lc_with_holdform'
        if f['start'] not in ('onset', 'gap') or f['gap_release'] not in ('timer', 'clear'):
            return 'bad_holdform'
        gap_keys = ('gap_dx_lo', 'gap_dx_hi', 'gap_drop_m', 'gap_arm_s', 'gap_release', 'gap_events')
        if f['start'] == 'gap':
            if int(r['lane']) != GAP_SUMO_LANE:
                return 'gap_role_lane'
            if f['gap_dx_lo'] is None or f['gap_dx_hi'] is None or f['gap_arm_s'] is None:
                return 'bad_holdform'
            if not -GAP_DX_MAX <= float(f['gap_dx_lo']) < float(f['gap_dx_hi']) <= GAP_DX_MAX:
                return 'bad_holdform'
            if not _int_ok(f['gap_arm_s'], 1) or not _int_ok(f['gap_events'], 1):
                return 'bad_holdform'
            if f['gap_drop_m'] is not None and not float(f['gap_drop_m']) > 0.0:
                return 'bad_holdform'
        elif any(f[k] != ROLE_DEFAULTS[k] for k in gap_keys):
            return 'bad_holdform'
        if f['dv_rel'] is not None and not float(f['dv_rel']) > 0.0:
            return 'bad_holdform'
        if not _int_ok(f['taper_in_s'], 0) or not _int_ok(f['taper_out_s'], 0) \
                or f['taper_in_s'] + f['taper_out_s'] > r['hold_s']:
            return 'bad_holdform'
        if (f['pulse_on_s'] is None) != (f['pulse_off_s'] is None):
            return 'bad_holdform'
        if f['pulse_on_s'] is not None and not (_int_ok(f['pulse_on_s'], 1) and _int_ok(f['pulse_off_s'], 1)):
            return 'bad_holdform'
    return None


def offer_condition(s):
    """Executor bank offer_condition of a search spec (a75_holdforms.bank_doc role mapping)."""
    roles = []
    for r in s['roles']:
        d = dict(sumo_lane=int(r['lane']), rank=int(r['rank']), v_w=float(r['v_w']), dmax=float(r['dmax']),
                 hold_s=int(r['hold_s']), onset_s=int(r['onset_s']), lc_target_sumo=r['lc_target'])
        d.update({k: r[k] for k in ROLE_DEFAULTS if k in r})
        roles.append(d)
    cond = dict(trigger=s['trigger'], zone=s['zone'], spread_m=float(s['spread_m']), cooldown_s=int(s['cooldown_s']),
                slots=int(s['slots']), gate=s.get('gate'), start_gate=s.get('start_gate'), phases=s.get('phases'),
                situation=dict(unconditional=True), roles=roles)
    if s.get('zone_d') is not None:
        cond['zone_d'] = list(s['zone_d'])
    return cond


# ------------------------------------------------------------------------------------------ patched functions
def grids(search=False, dense=False):
    g = ORIG['grids'](search, dense)
    extra = _A.get('valid_extra')
    if search or not extra:
        return g
    return {k: sorted(set(v) | set(extra.get(k, ()))) for k, v in g.items()}


def validate(s, allow_banned=False):
    t = s if s['zone'] in SS.ZONES else dict(s, zone=next(iter(SS.ZONES)))   # zone checked below
    ORIG['validate'](t, allow_banned)
    if _A.get('defer'):                 # inside the original mutate: the A75 step repairs, then validates fully
        return s
    why = holdform_refusal(s)
    if why is not None:
        raise ValueError(f'A75 executor refusal: {why}')
    return s


def strategy_key(s):
    """Executor a75_holdforms.strategy_key: oracle_plan key (= strategy_search key) + zone_d when set."""
    body = {k: s[k] for k in ('trigger', 'zone', 'spread_m', 'cooldown_s', 'slots', 'roles')}
    for k in ('gate', 'start_gate', 'phases', 'zone_d'):
        if s.get(k) is not None:
            body[k] = s[k]
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]


def _need(s):
    return max(int(r['onset_s']) + int(r['hold_s']) for r in s['roles'])


def _valid_cooldowns(s, grid):
    return [int(c) for c in grid if int(c) >= _need(s) // 2]


def draw_forms(rng, r, hf):
    """Fresh draw of the new role fields (clears the old ones). Pure given rng; canonical output."""
    for k in ROLE_DEFAULTS:
        r.pop(k, None)
    if int(r['lane']) == GAP_SUMO_LANE and hf.get('p_gap') and rng.random() < float(hf['p_gap']):
        g = hf['gap']
        lo = float(rng.choice(g['gap_dx_lo']))
        his = [float(x) for x in g['gap_dx_hi'] if float(x) > lo]
        r.update(start='gap', gap_dx_lo=lo, gap_dx_hi=float(rng.choice(his)), gap_arm_s=int(rng.choice(g['gap_arm_s'])),
                 hold_s=int(rng.choice(g['hold_s'])))
        dm = rng.choice(g['gap_drop_m'])
        if dm is not None:
            r['gap_drop_m'] = float(dm)
        rel = rng.choice(g['gap_release'])
        if rel != 'timer':
            r['gap_release'] = str(rel)
        ne = int(rng.choice(g['gap_events']))
        if ne != 1:
            r['gap_events'] = ne
    if hf.get('p_dv_rel') and rng.random() < float(hf['p_dv_rel']):
        r['dv_rel'] = float(rng.choice(hf['dv_rel']))
        r['v_w'] = float(rng.choice(hf['dv_floor']))
    if hf.get('p_taper') and rng.random() < float(hf['p_taper']):
        tin, tout = int(rng.choice(hf['taper_in_s'])), int(rng.choice(hf['taper_out_s']))
        if 0 < tin + tout <= int(r['hold_s']):
            if tin:
                r['taper_in_s'] = tin
            if tout:
                r['taper_out_s'] = tout
    if hf.get('p_pulse') and rng.random() < float(hf['p_pulse']):
        on, off = int(rng.choice(hf['pulse_on_s'])), int(rng.choice(hf['pulse_off_s']))
        if on + off <= int(r['hold_s']):
            r.update(pulse_on_s=on, pulse_off_s=off)
    return r


def repair_forms(r, rng=None, hf=None):
    """After a hold_s mutation: a gap role's event length goes back to the gap grid; taper / pulse fields that
    became inconsistent are dropped (executor refusal otherwise)."""
    if r.get('start') == 'gap' and hf and int(r['hold_s']) not in [int(x) for x in hf['gap']['hold_s']]:
        r['hold_s'] = int(rng.choice(hf['gap']['hold_s']))
    if role_field(r, 'taper_in_s') + role_field(r, 'taper_out_s') > int(r['hold_s']):
        r.pop('taper_in_s', None)
        r.pop('taper_out_s', None)
    on, off = role_field(r, 'pulse_on_s'), role_field(r, 'pulse_off_s')
    if on is not None and on + off > int(r['hold_s']):
        r.pop('pulse_on_s', None)
        r.pop('pulse_off_s', None)
    return r


def draw_zone_d(rng, zd):
    pairs = [(float(u), float(d)) for u in zd['d_up'] for d in zd['d_down']
             if float(u) - float(d) >= float(zd.get('min_width', 200.0)) and 0.0 <= float(d) < float(u) <= ZONE_D_MAX]
    return list(rng.choice(pairs))


def _extend(rng, s, prefix, p=None, parent=None):
    """A75 extra draws after the original generator: forms per role, custom zone, cooldown; renamed + validated.
    p None = fresh strategy (random_strategy), else mutation probability."""
    t = dict(s, roles=[dict(r) for r in s['roles']])
    hf = _A.get('holdforms')
    if hf:
        for r in t['roles']:
            if p is None or rng.random() < p / 2:
                draw_forms(rng, r, hf)
            else:
                repair_forms(r, rng, hf)
    zd = _A.get('zone_d_grid')
    if t['zone'] != 'custom':
        t.pop('zone_d', None)
    if zd and rng.random() < (float(zd['p']) if p is None else p / 2):
        t['zone'], t['zone_d'] = 'custom', draw_zone_d(rng, zd)
    cg = _A.get('cooldown_grid')
    if cg:
        ok = _valid_cooldowns(t, cg)
        keep = parent is not None and int(parent['cooldown_s']) in ok and rng.random() >= p / 2
        if keep:
            t['cooldown_s'] = int(parent['cooldown_s'])
        elif ok:
            t['cooldown_s'] = int(rng.choice(ok))
        else:
            t['cooldown_s'] = int(_need(t))
    t['roles'] = [canonical_role(r) for r in t['roles']]
    t['name'] = prefix + strategy_key(t)
    return t


def random_strategy(rng, max_roles=6, dense=False, hold_only=False, role_counts=None):
    s = ORIG['random_strategy'](rng, max_roles, dense, hold_only, role_counts)
    if not _A.get('ext_gen'):
        return s
    return validate(_extend(rng, s, 'rand_'))


def mutate(rng, s, p, dense=False, hold_only=False):
    _A['defer'] = bool(_A.get('ext_gen'))
    try:
        t = ORIG['mutate'](rng, s, p, dense, hold_only)
    finally:
        _A['defer'] = False
    if t is None:
        return None
    cap = _A.get('max_roles_mutation')
    if cap is not None and len(t['roles']) > int(cap):
        return None
    if not _A.get('ext_gen'):
        return t
    t = _extend(rng, t, 'mut_', p=p, parent=s)
    try:
        return validate(t)
    except ValueError:
        return None


def form_letters(s):
    return ''.join(sorted({ch for r in s['roles'] for k, ch in FORM_LETTERS if k in r}))


def descriptor(s):
    d = ORIG['descriptor'](s)
    if not _A.get('descriptor_v2'):
        return d
    hmax = max(int(r['hold_s']) for r in s['roles'])
    v = float(np.mean([r['v_w'] for r in s['roles']]))
    return d + (0 if hmax <= 30 else 1 if hmax <= 90 else 2, int(v < 14.0), form_letters(s))


def struct_vec(s):
    if not _A.get('nov_v2'):
        return ORIG['struct_vec'](s)
    t = dict(s, zone='far')                       # zone feature replaced below (A75 zone order)
    old = ORIG['struct_vec'](t) if 'far' in SS.ZONES else None
    if old is None:
        z = SS.ZONES
        SS.ZONES = ORIG_ZONES
        try:
            old = ORIG['struct_vec'](t)
        finally:
            SS.ZONES = z
    R = s['roles']
    frac = [float(np.mean([k in r for r in R])) for k in ('start', 'dv_rel', 'taper_in_s', 'pulse_on_s')]
    zi = ZONE_ORDER.index(s['zone']) / float(len(ZONE_ORDER) - 1)
    return np.concatenate([old[:-1], [zi], frac])


def select_members(s, cavs, observed=None, phase=0):
    """executor a75_holdforms.select_members (zone window from zone_bounds); same as the original otherwise."""
    if not _A.get('zones_on'):
        return ORIG['select_members'](s, cavs, observed, phase)
    seen_rows = list(cavs) + list(observed or [])
    if not SS.gate_open(s, seen_rows):
        return None
    roles = SS.active_roles(s, phase)
    lo, hi = zone_bounds(s)
    z = [c for c in cavs if c['edge'] == 'up' and lo <= float(c['x']) <= hi]
    if s['trigger'] == 'wall':
        wall = SS._pick_wall(z)
        if wall is None:
            return None
        out = [wall[r['lane']] for r in roles]
    else:
        out = []
        for r in roles:
            lane = sorted((c for c in z if int(c['lane']) == r['lane']), key=lambda c: (-float(c['x']), c['id']))
            if len(lane) <= r['rank']:
                return None
            out.append(lane[r['rank']])
        if len({c['id'] for c in out}) != len(out):
            return None
        xs = [float(c['x']) for c in out]
        if max(xs) - min(xs) > float(s['spread_m']):
            return None
    sg = SS.START_GATES[s.get('start_gate')]
    if sg is not None and any(SS.behind_count(seen_rows, c, sg['behind_m']) > sg['max_behind'] for c in out):
        return None
    return out


def member_specs(s, members, phase=0):
    """Original member specs; roles with new fields add params['form'] (read by a75_wmforms in the WM rollout)."""
    roles = SS.active_roles(s, phase)
    if not any(k in r for r in roles for k in ROLE_DEFAULTS):
        return ORIG['member_specs'](s, members, phase)
    out = []
    for r, c in zip(roles, members):
        prm = dict(v_w=float(r['v_w']), hold_s=int(r['hold_s']), dmax=float(r['dmax']))
        if r['onset_s']:
            prm['start_s'] = int(r['onset_s'])
        form = {k: r[k] for k in ROLE_DEFAULTS if k in r}
        if form:
            prm['form'] = form
        out.append(F.member_spec(c, prm))
    return out


def activated(r):
    return float(r.get('feasible') or 0.0) > 0.0


def rank_key(archive):
    k = ORIG['rank_key'](archive)
    if not _A.get('rank_activated_first'):
        return k
    return lambda r: (0 if activated(r) else 1, k(r))


# ------------------------------------------------------------------------------------------ install
A75_KEYS = ('zones', 'zone_d_grid', 'cooldown_grid', 'max_roles_mutation', 'holdforms', 'descriptor_v2',
            'rank_activated_first')


def install(cfg):
    """Patch strategy_search for this cfg (idempotent; uninstall() restores the original code objects)."""
    uninstall()
    cfg = dict(cfg or {})
    hf = cfg.get('holdforms') or None
    extra = {}
    for k, v in (cfg.get('search_grid') or {}).items():
        extra.setdefault(k, set()).update(v)
    if hf:
        extra.setdefault('hold_s', set()).update(int(x) for x in hf.get('gap', {}).get('hold_s', ()))
        extra.setdefault('v_w', set()).update(float(x) for x in hf.get('dv_floor', ()))
    zones = cfg.get('zones')
    if zones:
        bad = [z for z in zones if z not in A75_ZONES]
        if bad:
            raise ValueError(f'unknown zones {bad}')
    _A.update(valid_extra=extra, holdforms=hf, zone_d_grid=cfg.get('zone_d_grid') or None,
              cooldown_grid=cfg.get('cooldown_grid') or None, max_roles_mutation=cfg.get('max_roles_mutation'),
              descriptor_v2=bool(cfg.get('descriptor_v2')), rank_activated_first=bool(cfg.get('rank_activated_first')),
              zones_on=bool(zones or cfg.get('zone_d_grid')),
              nov_v2=bool(zones or cfg.get('zone_d_grid') or hf))
    _A['ext_gen'] = bool(hf or _A['zone_d_grid'] or _A['cooldown_grid'])
    if zones:
        SS.ZONES = {z: A75_ZONES[z] for z in zones}
    for n in ('grids', 'validate', 'strategy_key', 'random_strategy', 'mutate', 'descriptor', 'struct_vec',
              'select_members', 'member_specs', 'rank_key'):
        setattr(SS, n, globals()[n])
    SS._SEED_VECS = None
    return dict(_A)


def uninstall():
    for n, f in ORIG.items():
        setattr(SS, n, f)
    SS.ZONES = dict(ORIG_ZONES)
    SS._SEED_VECS = None
    _A.clear()


# ------------------------------------------------------------------------------------------ search
def search_cell(wm, states, cfg=None, log=None):
    """strategy_search.search_cell (same algorithm, same RNG use) + rank_activated_first when set."""
    cfg = dict(SS.DEFAULTS, **(cfg or {}))
    SS.UNCOND, SS.F_VS_BASE = bool(cfg.get('unconditional')), bool(cfg.get('f_vs_base'))
    SS.NOV_W = float(cfg.get('novelty_weight', 0.0))
    SS.MIN_ROLES = int(cfg.get('min_roles', 0))
    SS.LANE2_EXTRA_S = float(cfg.get('lane2_extra_margin_s', 0.0))
    SS.REC_W = float(cfg.get('rec_weight', 0.0))
    extra = [dict(s) for s in (cfg.get('extra_seeds') or [])]
    for s in extra:
        SS.validate(s)
    SS.ONSET_MAX = cfg.get('onset_max')
    SS.SEARCH_GRID = cfg.get('search_grid')
    act_first = bool(_A.get('rank_activated_first'))

    def act(r):
        return int(activated(r)) if act_first else 0
    rng = random.Random(cfg['seed'])
    lam, groups = float(cfg['lam']), bool(cfg['fairness'])
    sk = dict(fair_tol=float(cfg['fair_tol']), bound_s=float(cfg['bound_s']), bound_stat=cfg['bound_stat'],
              fairer_total=cfg['fairer_total'], harm_cal=cfg['harm_cal'])
    dense, hold_only = bool(cfg['dense']), bool(cfg['hold_only'])
    screen = rng.sample(states, min(len(states), int(cfg['n_screen_states'])))
    pop = [s for s in SS.seeds() if not (hold_only and any(r['lc_target'] is not None for r in s['roles']))]
    if dense:
        pop += SS.dense_seeds()
    if SS.ONSET_MAX is not None:
        pop = [s for s in pop if all(int(r['onset_s']) <= int(SS.ONSET_MAX) for r in s['roles'])]
    if SS.SEARCH_GRID:
        pop = [s for s in pop if all(r[k] in SS.SEARCH_GRID[k] for r in s['roles'] for k in SS.SEARCH_GRID)]
    rc = tuple(cfg['role_counts']) if cfg.get('role_counts') else None
    pop += extra
    pop += [SS.random_strategy(rng, cfg['max_roles'], dense, hold_only, rc) for _ in range(int(cfg['n_init']))]
    evaluated, total = {}, 0
    archives = {a: {} for a in SS.ARCHIVES}
    comp_keys = {SS.strategy_key(c) for c in SS.COMPARATORS}

    def run(cands, sts, K):
        nonlocal total
        cands = [SS.uncond(c) for c in cands]
        cands = [c for c in cands if c is not None and len(c['roles']) >= SS.MIN_ROLES]
        cands = [c for c in cands if c is not None and SS.strategy_key(c) not in evaluated and not SS.banned(c)]
        if not cands:
            return
        per, n = SS.evaluate(wm, sts, cands, K, cfg['H'], groups=groups)
        total += n
        for c in cands:
            k = SS.strategy_key(c)
            res = dict(strategy=c, key=k, descriptor=list(SS.descriptor(c)), novelty=SS.novelty(c),
                       **SS.summarize(per[k], lam, **sk))
            evaluated[k] = res
            d = SS.descriptor(c)
            for a, arch in archives.items():
                if SS.admits(a, res) and (d not in arch or SS.rank_key(a)(res) < SS.rank_key(a)(arch[d])):
                    arch[d] = res
        if log:
            log(dict(evaluated=len(evaluated), rollouts=total, **{f'archive_{a}': len(v) for a, v in archives.items()}))

    def top(arch, a, n):
        return sorted(arch.values(), key=SS.rank_key(a))[:n]

    run(pop, screen, cfg['K_screen'])
    for _ in range(int(cfg['generations'])):
        ranked = sorted(evaluated.values(),
                        key=lambda r: (act(r), r['score'] + SS.NOV_W * min(SS.NOV_CAP, r.get('novelty', 0.0))),
                        reverse=True)
        parents = [r['strategy'] for r in ranked[:int(cfg['n_elite'])]]
        for a in SS.ARCHIVES:
            parents += [r['strategy'] for r in top(archives[a], a, int(cfg['n_elite']) if a != 'fair'
                                                   else int(cfg['n_elite']) // 2)]
        parents += extra
        kids = [SS.mutate(rng, rng.choice(parents), cfg['p_mutate'], dense, hold_only)
                for _j in range(int(cfg['n_children']))]
        kids += [SS.random_strategy(rng, cfg['max_roles'], dense, hold_only, rc)
                 for _j in range(int(cfg['n_immigrants']))]
        run(kids, screen, cfg['K_screen'])
    nf = int(cfg['finalists'])
    fin = {}
    for r in sorted(evaluated.values(), key=lambda r: (act(r), r['score']), reverse=True)[:nf]:
        fin[r['key']] = r
    for a in ('F', 'D', 'bounded', 'fairer', 'fair', 'delay'):
        for r in top(archives[a], a, nf):
            if len(fin) < 4 * nf:
                fin.setdefault(r['key'], r)
    finals = [dict(f['strategy']) for f in fin.values()]
    for c in SS.COMPARATORS:
        if SS.strategy_key(c) not in {SS.strategy_key(f) for f in finals}:
            finals.append(dict(c))
    fstates = states
    if int(cfg['n_final_states']) and len(states) > int(cfg['n_final_states']):
        fstates = random.Random(cfg['seed'] + 1).sample(states, int(cfg['n_final_states']))
    per, n = SS.evaluate(wm, fstates, finals, cfg['K_final'], cfg['H'], groups=groups)
    total += n
    final = []
    for f in finals:
        k = SS.strategy_key(f)
        final.append(dict(strategy=f, key=k, descriptor=list(SS.descriptor(f)), comparator=k in comp_keys,
                          novelty=SS.novelty(f), banned=SS.banned(f), **SS.summarize(per[k], lam, **sk)))
    final.sort(key=lambda r: (act(r), r['score']), reverse=True)
    cands = [r for r in final if not r['comparator'] and not r['banned']]

    def best(a):
        xs = sorted((r for r in cands if SS.admits(a, r)), key=SS.rank_key(a))
        return xs[0] if xs else None
    beats = [r for r in cands if r['gain_vs_inc_lcb'] > 0 and r.get('bounded') is not False]
    fairness = ('per-vehicle' if final and final[0].get('bounded') is not None else
                'groups only' if final and final[0].get('fair') is not None else
                'unavailable (WM has no per-group delay)')

    def listing(a):
        return [dict(descriptor=list(d), key=r['key'], name=r['strategy']['name'], score=r['score'],
                     worst_group_inc=r.get('worst_group_inc'), cvar90_inc=r.get('cvar90_inc'))
                for d, r in sorted(archives[a].items(), key=lambda kv: SS.rank_key(a)(kv[1]))]
    return dict(schema='wmc-strategy-search-v3', config=cfg, incumbent=SS.INCUMBENT['name'],
                banned_v_w=list(SS.BANNED_V_W),
                n_states=len(states), n_screen_states=len(screen), n_final_states=len(fstates),
                evaluated=len(evaluated), wm_rollouts=total, fairness=fairness,
                archive_cells={a: len(v) for a, v in archives.items()}, final=final,
                best={a: best(a) for a in SS.ARCHIVES}, best_beating_incumbent=beats[0] if beats else None,
                comparators={r['strategy']['name']: {k: r.get(k) for k in (
                    'gain_vs_base', 'gain_vs_base_lcb', 'group_inc', 'worst_group_inc', 'cvar90_inc', 'bounded')}
                    for r in final if r['comparator']},
                archives={a: listing(a) for a in SS.ARCHIVES},
                screened=[dict(key=r['key'], name=r['strategy']['name'], n_roles=len(r['strategy']['roles']),
                               trigger=r['strategy']['trigger'], feasible=r['feasible'],
                               gain_vs_base=r['gain_vs_base'], gain_vs_inc=r['gain_vs_inc'],
                               fair_cal=r.get('fair_cal')) for r in evaluated.values()])


# ------------------------------------------------------------------------------------------ driver
WM_CKPT_SHA256 = '6da9d3337aad78799c2e6ac97c87e79b927346011f55861e98887bb89728c063'
HERE = Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m a75ss')
    ap.add_argument('--states', required=True)
    ap.add_argument('--cell', required=True)
    ap.add_argument('--config', required=True, help='cfg JSON file (dict, or {"config": dict})')
    ap.add_argument('--out', required=True, help='NEW directory')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--map-spec', default='maps/wz3to2/MAP_SPEC.json')
    ap.add_argument('--workers', type=int, default=10)
    ap.add_argument('--max-rows', type=int, default=64)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--k', type=int, default=4)
    ap.add_argument('--top-bank', type=int, default=12, help='final plans written as executor offer_conditions')
    a = ap.parse_args(argv)
    raw = json.loads(Path(a.config).read_text(encoding='utf-8'))
    cfg = raw.get('config', raw)
    if sha(a.ckpt) != WM_CKPT_SHA256:
        raise SystemExit('REFUSE: checkpoint is not the frozen WM-S wm_s.pt 6da9d333')
    out = G.check_path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    states = [m for m in json.loads(G.check_path(a.states).read_text()) if m.get('valid') and m['cell'] == a.cell]
    if not states:
        raise SystemExit(f'REFUSE: no valid states for {a.cell}')
    info = install(cfg)
    os.environ['PYTHONPATH'] = os.pathsep.join([str(HERE)] + [p for p in [os.environ.get('PYTHONPATH')] if p])
    log_dir = out / 'crn_log'
    log_dir.mkdir()
    os.environ['WM_FIX_LOG_DIR'] = str(log_dir)
    from tools.wzprop import wm_backend as W
    kw = dict(checkpoint=a.ckpt, map_spec=a.map_spec, device=a.device, max_rows=int(a.max_rows))
    wm = W.ParallelWM('wma:WMA_READY.json', a.workers, api=dict(evaluate='a75_wmforms:evaluate'),
                      local_kw=dict(kw, device='cpu'), mem_fraction=None, **kw)
    t0 = time.time()
    log = (out / f'LOG_{a.cell}.jsonl').open('x')
    res = search_cell(wm, states, cfg, log=lambda d: (log.write(json.dumps(d) + '\n'), log.flush()))
    log.close()
    import a75_wmforms as WF
    import wm_fix_crn as CRN
    prov = dict(schema=SCHEMA, cell=a.cell, cfg_sha256=sha(a.config), ckpt_sha256=WM_CKPT_SHA256,
                states_sha256=sha(a.states), n_states=len(states),
                code=dict(a75ss=sha(__file__), a75_wmforms=sha(WF.__file__), wm_fix_crn=sha(CRN.__file__),
                          strategy_search=sha(SS.__file__), planapi=sha('src/wms/planapi.py'),
                          wm_backend=sha(W.__file__)),
                installed={k: v for k, v in info.items() if k != 'valid_extra'},
                crn=dict(module='wm_fix_crn (E1-FIX, Amendment 74)', entrant_cav_share=CRN.ENTRANT_CAV_SHARE),
                env=dict(WZPROP_SPLIT_ACTIONS=os.environ.get('WZPROP_SPLIT_ACTIONS')), max_rows=int(a.max_rows),
                workers=int(a.workers), new_sumo_sim_s=0)
    res.update(cell=a.cell, backend=wm.meta, wall_s=round(time.time() - t0, 1), new_sumo_sim_s=0, a75=prov)
    (out / f'STRATEGY_SEARCH_{a.cell}.json').write_text(json.dumps(res, indent=1, default=str))
    arms = {a.cell: dict(SS.wm_arms(res, a.k), incumbent=SS.INCUMBENT, best={k2: (v or {}).get('strategy')
                                                                           for k2, v in res['best'].items()},
                         comparators=res['comparators'], fairness=res['fairness'])}
    (out / 'ARMS.json').write_text(json.dumps(arms, indent=1))
    top = [r for r in res['final'] if not r['comparator'] and not r['banned'] and activated(r)][:int(a.top_bank)]
    (out / f'TOP_OFFER_CONDITIONS_{a.cell}.json').write_text(json.dumps(
        [dict(family=r['strategy']['name'], strategy_key=r['key'], score=r['score'], gain_vs_base=r['gain_vs_base'],
              gain_vs_base_lcb=r['gain_vs_base_lcb'], feasible=r['feasible'], extended=is_extended(r['strategy']),
              offer_condition=offer_condition(r['strategy'])) for r in top], indent=1))
    t = res['final'][0]
    print(a.cell, 'evaluated', res['evaluated'], 'rollouts', res['wm_rollouts'], 'best',
          (t['strategy']['name'], t['gain_vs_base'], t['gain_vs_base_lcb']), flush=True)
    wm.close()


if __name__ == '__main__':
    main()
