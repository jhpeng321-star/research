"""Strategy-level, condition-aware WM search on the MAPPO u0 base (owner directions 2026-11-06 ~20:45 and
2026-11-07 04:03/04:3x, REGISTRATION.md).

Owner rules (04:03): the firm 15 m/s hold (wall_h_solo2, "solo2") is BANNED as a guidance target - it is kept only as
a labelled comparator; a strategy is ALLOWED only if every group (SUMO origin lane 0/1/2; HDV / CAV) loses at most
10 s per vehicle vs u0 (mean per-vehicle delay increase, point estimate; the CI upper bound is reported too).
The reference incumbent is the gentle hold wall_g_solo2 (S-LOWSHARE-U0 / S-MULTICRIT: wall trigger, only the SUMO
lane-2 member holds 22 m/s for 120 s, dmax 1.0, cooldown 120 s).

A STRATEGY is an episode rule: whenever its trigger is feasible (and the cooldown has passed), the CAVs it selects
run their programs; the plant base (MAPPO u0 in SUMO) drives everything else. Schema (JSON):
  name, trigger 'wall' (pg_plan.pick_wall geometry; roles must be rank 0 in distinct lanes) | 'lanes' (per role the
  rank-th most downstream zone CAV of its SUMO lane; picked CAVs distinct and within spread_m),
  zone (registered zones), spread_m, cooldown_s, slots (concurrent activations),
  roles [{lane, rank, v_w, dmax, hold_s, onset_s, lc_target}] (1-6 roles; lc_target = adjacent SUMO lane or None;
      any lc_target makes every role an LCMOVES 'prog' program, executed in SUMO by the LCEXEC lane path),
  gate None | 5 | 10 | 15 (observable closing-lane field speed gate, m/s),
  start_gate None | 'empty100' (S-MULTICRIT wall_g_solo2_gate0: start only if no observed vehicle is within 100 m
      behind any member in its lane),
  phases None | [[role indices], ...] (alternating holders: activation n runs only the roles of phase n mod P).
Every value is drawn from the registered NEWMAP2 / COOPKINDS / PLANGEN2 / WMDATA ranges (lcm_plan constants);
v_w = 15 m/s is excluded from the search (banned).

WM evaluation (imagination): at each TRAIN start state of a demand cell the strategy's activation at t0 (if feasible;
phase = state index mod P) runs for H = 300 s next to the base (u0) and the incumbent in ONE WM call;
gain_vs_base = J(base) - J(cand) (veh-s, positive = better). Per-vehicle fairness from the planapi trace
(WM-A 7138ba9a: masks, is_cav, origin_lane, lanes): per group g the mean per-vehicle delay increase
inc_g = (J_g(cand) - J_g(base)) / N_g(base) (s/veh, positive = worse) and the CVaR90 change
CVaR90(cand) - CVaR90(base) of the per-vehicle window delays (mean of the worst ceil(0.1 n) vehicles, S-MULTICRIT P3).
Per cell: means over states, one-sided 95% world-cluster bootstrap bounds (one shared resampling plan per strategy).

Three archives (MAP-Elites over descriptors), all feeding the finals:
  (a) 'delay'   : score = LCB(gain_vs_base) - lambda * disagreement / sqrt(n)
  (b) 'bounded' : the same score among strategies with every predicted group increase <= 10 s/veh
  (c) 'fairer'  : among strategies with predicted total delay not worse than u0 (gain_vs_base >= 0), minimise the
                  predicted worst-group mean increase (0.5 s/veh bins), then the CVaR90 change.
(The strict no-harm 'fair' flag of 2026-11-06 is still reported.) One activation per state approximates the episode
rule (many states per cell cover the episode); this is a ranking device, SUMO paired seeds decide.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
from collections import defaultdict

import numpy as np

from . import families as F
from . import families_lc as FL
from . import geometry as G
from . import wm_backend as W

ZONES = {'full': (G.X_B - 1800.0, G.X_B - 400.0), 'near': (G.X_B - 1200.0, G.X_B - 400.0),
         'far': (G.X_B - 1800.0, G.X_B - 1000.0)}
# Observable condition gate (condition-aware within a cell): activate only if the mean speed of OBSERVED vehicles
# (CAV self + CAV-detected) in the N1 RSU closing-lane field x in [X_B-454.10, X_B] is <= gate (m/s); no observed
# vehicle there = closed; None = always (S-REGIME amendment 4 'fld_v' proxy, instantaneous at the decision second).
GATES = (None, 5.0, 10.0, 15.0)
FIELD_X = (G.X_B - 454.10, G.X_B)
# Start gates (S-MULTICRIT regimeu0.GATES, c4/multicrit 3fe79c61): a new activation starts only if, for every member,
# at most max_behind observed vehicles are in its lane within behind_m behind it.
START_GATES = {None: None, 'empty100': dict(behind_m=100.0, max_behind=0)}
BANNED_V_W = (15.0,)          # owner 2026-11-07 (REGISTRATION 04:03): the firm 15 m/s hold is banned
GROUP_BOUND_S = 10.0          # allowed: every group's mean per-vehicle delay increase <= 10 s vs u0
FAIRER_BIN_S = 0.5            # archive (c): worst-group increase compared in 0.5 s/veh bins, then CVaR90
WALL_G = dict(v_w=22.0, dmax=1.0, hold_s=120)     # pg2_plan VARIANTS['wall_g'] (S-LOWSHARE-U0 / S-MULTICRIT)
WALL_M = dict(v_w=18.0, dmax=1.0, hold_s=90)      # S-MULTICRIT wall_m (NEWMAP2 R grid point)
WALL_H = dict(v_w=15.0, dmax=1.5, hold_s=90)      # pg2_plan VARIANTS['wall_h'] (banned)


def _role(lane, prog, rank=0, onset=0, lc=None):
    return dict(lane=lane, rank=rank, v_w=float(prog['v_w']), dmax=float(prog['dmax']), hold_s=int(prog['hold_s']),
                onset_s=int(onset), lc_target=lc)


def _spec(name, trigger, roles, spread=None, cooldown=None, **kw):
    s = dict(name=name, trigger=trigger, zone='full', spread_m=float(spread or (G.WALL_SPREAD if trigger == 'wall'
                                                                                 else G.R_SPREAD)),
             cooldown_s=int(cooldown or max(r['onset_s'] + r['hold_s'] for r in roles)), slots=1, gate=None,
             start_gate=None, phases=None, roles=roles)
    s.update(kw)
    return s


INCUMBENT = _spec('wall_g_solo2', 'wall', [_role(2, WALL_G)])
SOLO2 = _spec('solo2', 'wall', [_role(2, WALL_H)])            # wall_h_solo2: BANNED comparator only
COMPARATORS = (INCUMBENT, SOLO2)
DEFAULTS = dict(H=300, K_screen=4, K_final=8, n_init=40, generations=3, n_elite=6, n_children=24, p_mutate=0.35,
                n_screen_states=12, finalists=12, lam=0.5, max_roles=6, seed=20261107, fairness=True, fair_tol=0.0,
                n_final_states=0, bound_s=GROUP_BOUND_S, bound_stat='mean', fairer_total='mean')
ARCHIVES = ('delay', 'bounded', 'fairer', 'fair')


def grids(search=False):
    """Registered value grids; search=True drops the banned hold speed."""
    P = FL.lcm()
    v = [x for x in P.R_GRID['v_w'] if not (search and float(x) in BANNED_V_W)]
    return dict(v_w=list(v), dmax=list(P.R_GRID['dmax']), hold_s=list(P.R_GRID['hold_s']), onset_s=list(P.OFFSETS))


def strategy_key(s):
    body = {k: s[k] for k in ('trigger', 'zone', 'spread_m', 'cooldown_s', 'slots', 'roles')}
    for k in ('gate', 'start_gate', 'phases'):           # only when set: older keys stay stable
        if s.get(k) is not None:
            body[k] = s[k]
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]


def banned(s):
    return any(float(r['v_w']) in BANNED_V_W for r in s['roles'])


def validate(s, allow_banned=False):
    """Registered ranges + well-formed roles (raises ValueError); banned speeds only with allow_banned."""
    g = grids()
    if s['trigger'] not in ('wall', 'lanes') or s['zone'] not in ZONES:
        raise ValueError('trigger/zone')
    if s.get('gate') not in GATES:
        raise ValueError('gate not registered')
    if s.get('start_gate') not in START_GATES:
        raise ValueError('start_gate not registered')
    if not 1 <= len(s['roles']) <= 6:
        raise ValueError('1-6 roles')
    if banned(s) and not allow_banned:
        raise ValueError('banned: the firm 15 m/s hold (owner 2026-11-07)')
    seen = set()
    for r in s['roles']:
        if r['lane'] not in (0, 1, 2) or r['rank'] not in (0, 1):
            raise ValueError('lane/rank')
        for k in ('v_w', 'dmax', 'hold_s', 'onset_s'):
            if r[k] not in g[k]:
                raise ValueError(f'{k}={r[k]} not registered')
        if r['lc_target'] is not None and abs(int(r['lc_target']) - int(r['lane'])) != 1:
            raise ValueError('lane change must target an adjacent lane')
        if (r['lane'], r['rank']) in seen:
            raise ValueError('duplicate role')
        seen.add((r['lane'], r['rank']))
    ph = s.get('phases')
    if ph is not None:
        flat = [i for p in ph for i in p]
        if not 2 <= len(ph) <= 3 or any(not p for p in ph) or sorted(set(flat)) != list(range(len(s['roles']))) \
                or len(flat) != len(set(flat)):
            raise ValueError('phases must partition the roles into 2-3 non-empty groups')
    if s['trigger'] == 'wall' and any(r['rank'] for r in s['roles']):
        raise ValueError('wall trigger roles are rank 0')
    if not 0 < float(s['spread_m']) <= 300.0 or int(s['cooldown_s']) < max(r['onset_s'] + r['hold_s'] for r in
                                                                            s['roles']) // 2:
        raise ValueError('spread/cooldown')
    return s


# ------------------------------------------------------------------------------------------ activation
def _pick_wall(z):
    lanes = [[c for c in z if int(c['lane']) == ln] for ln in (0, 1, 2)]
    if any(not x for x in lanes):
        return None
    best = None
    for combo in itertools.product(*lanes):
        xs = [float(c['x']) for c in combo]
        sp = max(xs) - min(xs)
        if sp <= G.WALL_SPREAD:
            key = (sp, -max(xs))
            if best is None or key < best[0]:
                best = (key, list(combo))
    return None if best is None else {int(c['lane']): c for c in best[1]}


def field_speed(rows):
    sp = [float(r['v']) for r in rows if r['edge'] == 'up' and int(r['lane']) == 0
          and FIELD_X[0] <= float(r['x']) <= FIELD_X[1]]
    return sum(sp) / len(sp) if sp else None


def gate_open(s, observed):
    if s.get('gate') is None:
        return True
    fv = field_speed(observed)
    return fv is not None and fv <= float(s['gate'])


def behind_count(rows, member, behind_m):
    """Observed vehicles in the member's lane (same edge) within behind_m behind it (excluding itself)."""
    x, lane, edge = float(member['x']), int(member['lane']), member.get('edge', 'up')
    seen = set()
    for r in rows:
        if r['id'] == member['id'] or r['id'] in seen:
            continue
        if r.get('edge', 'up') == edge and int(r['lane']) == lane and x - behind_m <= float(r['x']) < x:
            seen.add(r['id'])
    return len(seen)


def active_roles(s, phase=0):
    ph = s.get('phases')
    return list(s['roles']) if not ph else [s['roles'][i] for i in ph[int(phase) % len(ph)]]


def select_members(s, cavs, observed=None, phase=0):
    """CAV rows (id, edge, lane, x, v) chosen by the strategy's trigger at this snapshot, in active-role order, else
    None. observed = every observable row (CAV self + CAV-detected) for the gates (default: the CAVs only)."""
    seen_rows = list(cavs) + list(observed or [])
    if not gate_open(s, seen_rows):
        return None
    roles = active_roles(s, phase)
    lo, hi = ZONES[s['zone']]
    z = [c for c in cavs if c['edge'] == 'up' and lo <= float(c['x']) <= hi]
    if s['trigger'] == 'wall':
        wall = _pick_wall(z)
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
    sg = START_GATES[s.get('start_gate')]
    if sg is not None and any(behind_count(seen_rows, c, sg['behind_m']) > sg['max_behind'] for c in out):
        return None
    return out


def member_specs(s, members, phase=0):
    """WM / SUMO member specs: all nm2 'hold' when no active role changes lanes, else all LCMOVES 'prog'."""
    roles = active_roles(s, phase)
    if all(r['lc_target'] is None for r in roles):
        out = []
        for r, c in zip(roles, members):
            prm = dict(v_w=float(r['v_w']), hold_s=int(r['hold_s']), dmax=float(r['dmax']))
            if r['onset_s']:
                prm['start_s'] = int(r['onset_s'])
            out.append(F.member_spec(c, prm))
        return out
    P = FL.lcm()
    out = []
    for r, c in zip(roles, members):
        out.append(P.prog(c, lc_target=r['lc_target'], lc_from=int(r['onset_s']), lc_giveup_x=P.X_B - P.GIVEUP_OFF,
                          speed='hold', v_w=float(r['v_w']), dmax=float(r['dmax']), sp_from=int(r['onset_s']),
                          hold_s=int(r['onset_s']) + int(r['hold_s'])))
    return out


def activation(s, cavs, observed=None, phase=0):
    mem = select_members(s, cavs, observed, phase)
    return None if mem is None else member_specs(s, mem, phase)


def descriptor(s):
    v = float(np.mean([r['v_w'] for r in s['roles']]))
    on = [r['onset_s'] for r in s['roles']]
    return (len(s['roles']), len({r['lane'] if r['lc_target'] is None else r['lc_target'] for r in s['roles']}),
            sum(r['lc_target'] is not None for r in s['roles']), int(v >= 17.0) + int(v >= 20.0),
            int(max(on) > min(on)), s['zone'], s['trigger'], s.get('gate') is not None,
            s.get('start_gate') is not None, s.get('phases') is not None)


# ------------------------------------------------------------------------------------------ sampling
def random_strategy(rng, max_roles=6):
    g = grids(search=True)
    n = rng.choice([x for x in (1, 1, 2, 2, 3, 3, 4, 5, 6) if x <= max_roles])
    trigger = 'wall' if n <= 3 and rng.random() < 0.4 else 'lanes'
    slots = [(ln, 0) for ln in (0, 1, 2)] if trigger == 'wall' else [(ln, rk) for ln in (0, 1, 2) for rk in (0, 1)]
    picks = rng.sample(slots, min(n, len(slots)))
    roles = []
    for ln, rk in picks:
        tl = rng.choice([None, None, None] + [x for x in (ln - 1, ln + 1) if 0 <= x <= 2])
        roles.append(dict(lane=ln, rank=rk, v_w=rng.choice(g['v_w']), dmax=rng.choice(g['dmax']),
                          hold_s=rng.choice(g['hold_s']), onset_s=rng.choice(g['onset_s']), lc_target=tl))
    s = dict(name='', trigger=trigger, zone=rng.choice(list(ZONES)), spread_m=float(rng.choice([200.0, 300.0])),
             slots=1, roles=roles, gate=rng.choice(GATES) if rng.random() < 0.3 else None,
             start_gate='empty100' if rng.random() < 0.2 else None,
             phases=[[i] for i in range(len(roles))] if 2 <= len(roles) <= 3 and rng.random() < 0.2 else None)
    s['cooldown_s'] = int(max(r['onset_s'] + r['hold_s'] for r in roles))
    s['name'] = 'rand_' + strategy_key(s)
    return validate(s)


def seeds():
    """Known allowed strategies: the gentle-hold incumbent and the S-MULTICRIT rules, then the owner's new
    combinations (alternating / relay holders across lanes 1/2, staggered gentle holds, holds plus middle-lane
    gap-giving, lane-change programs)."""
    G22, M18 = WALL_G, WALL_M
    out = [dict(INCUMBENT),
           _spec('wall_g_duo12', 'wall', [_role(1, G22), _role(2, G22)]),
           _spec('wall_g_solo2_gate0', 'wall', [_role(2, G22)], start_gate='empty100'),
           _spec('wall_m_solo2', 'wall', [_role(2, M18)]),
           _spec('wall_g_solo1', 'wall', [_role(1, G22)]),
           _spec('wall_g_duo12_gate0', 'wall', [_role(1, G22), _role(2, G22)], start_gate='empty100'),
           # alternating holders: activation n holds in lane 2, activation n+1 in lane 1, ...
           _spec('wall_g_alt21', 'wall', [_role(2, G22), _role(1, G22)], phases=[[0], [1]]),
           _spec('wall_g_alt21_gate0', 'wall', [_role(2, G22), _role(1, G22)], phases=[[0], [1]],
                 start_gate='empty100'),
           # relay holders: the lane-1 member takes over while the lane-2 member finishes
           _spec('wall_g_relay21', 'wall', [_role(2, dict(G22, hold_s=90)), _role(1, G22, onset=60)]),
           _spec('wall_g_relay12', 'wall', [_role(1, dict(G22, hold_s=90)), _role(2, G22, onset=60)]),
           # staggered gentle holds
           _spec('g2_stagger40', 'lanes', [_role(2, G22), _role(2, G22, rank=1, onset=40)]),
           _spec('wall_g_duo12_stagger40', 'wall', [_role(2, G22), _role(1, G22, onset=40)]),
           # gentle lane-2 hold plus middle-lane gap-giving for closing-lane mergers
           _spec('g2_gap1', 'lanes', [_role(2, G22), _role(1, M18, onset=20)]),
           _spec('g2_gap1_gate0', 'lanes', [_role(2, G22), _role(1, M18, onset=20)], start_gate='empty100'),
           # lane-change programs (LCEXEC lane path): early merge of a closing-lane CAV / middle-lane gap opening
           _spec('g2_lc01', 'lanes', [_role(2, G22), _role(0, G22, lc=1)]),
           _spec('lc01_g22', 'lanes', [_role(0, G22, lc=1)]),
           _spec('g2_gapopen12', 'lanes', [_role(2, G22), _role(1, M18, rank=1, onset=20, lc=2)])]
    return [validate(s) for s in out]


def mutate(rng, s, p):
    g = grids(search=True)
    roles = []
    for r in s['roles']:
        r = dict(r)
        for k in ('v_w', 'dmax', 'hold_s', 'onset_s'):
            if rng.random() < p or (k == 'v_w' and float(r[k]) in BANNED_V_W):
                r[k] = rng.choice(g[k])
        if rng.random() < p / 2:
            r['lc_target'] = rng.choice([None] + [x for x in (r['lane'] - 1, r['lane'] + 1) if 0 <= x <= 2])
        roles.append(r)
    t = dict(s, roles=roles)
    if rng.random() < p / 2:
        t['zone'] = rng.choice(list(ZONES))
    if rng.random() < p / 2:
        t['gate'] = rng.choice(GATES)
    if rng.random() < p / 3:
        t['start_gate'] = rng.choice(list(START_GATES))
    if rng.random() < p / 3 and t['trigger'] == 'lanes' and len(roles) < 6:
        free = [(ln, rk) for ln in (0, 1, 2) for rk in (0, 1) if (ln, rk) not in {(x['lane'], x['rank']) for x in roles}]
        if free:
            ln, rk = rng.choice(free)
            roles.append(dict(lane=ln, rank=rk, v_w=rng.choice(g['v_w']), dmax=rng.choice(g['dmax']),
                              hold_s=rng.choice(g['hold_s']), onset_s=rng.choice(g['onset_s']), lc_target=None))
            t['phases'] = None
    if rng.random() < p / 4 and 2 <= len(roles) <= 3:
        t['phases'] = None if t.get('phases') else [[i] for i in range(len(roles))]
    t['cooldown_s'] = int(max(max(r['onset_s'] + r['hold_s'] for r in roles), s['cooldown_s'] // 2))
    t['name'] = 'mut_' + strategy_key(t)
    try:
        return validate(t)
    except ValueError:
        return None


# ------------------------------------------------------------------------------------------ WM evaluation
def _spec_key(spec):
    return json.dumps([[x['id'], x['role'], x['params']] for x in spec], sort_keys=True)


def evaluate(wm, states, strategies, K, H, z=1.645, groups=False):
    """Per strategy per state the WM draws -> gains vs base (u0) and vs the incumbent (see module docstring).
    groups=True (and a backend with evaluate_groups) adds per-group gains vs base and, when the backend reports
    vehicle counts and tails, the per-vehicle group increases and the CVaR90 / p90 changes."""
    per = defaultdict(list)
    rollouts = 0
    prepared = []
    for i, m in enumerate(states):
        mi = W.moment_input(m)
        cavs = m['all_cavs_t0']
        observed = list((m.get('observed') or {}).get('vehicles') or [])
        state = wm.observe(mi)
        acts = [W.action('base', 'base', None, 'base', [], H)]
        inc = activation(INCUMBENT, cavs, observed)
        if inc:
            acts.append(W.action('inc', 'group', INCUMBENT['name'], INCUMBENT['name'], inc, H))
        aid_of, specs_seen = {}, {}
        if inc:                                  # a candidate whose activation IS the incumbent's reuses its row
            specs_seen[_spec_key(inc)] = 'inc'
        for s in strategies:
            spec = activation(s, cavs, observed, phase=i)
            if spec is None:
                continue
            sk = _spec_key(spec)
            if sk not in specs_seen:
                specs_seen[sk] = f'c{len(specs_seen)}'
                acts.append(W.action(specs_seen[sk], 'group', 'strategy', sk, spec, H))
            aid_of[strategy_key(s)] = specs_seen[sk]
        prepared.append((m, state, acts, aid_of, inc, mi['crn_seed']))
    calls = [(state, acts, K, H, seed) for _m, state, acts, _a, _i, seed in prepared]
    use_groups = bool(groups) and hasattr(wm, 'evaluate_groups')
    graws = None
    if use_groups:
        try:
            graws = wm.map_groups(calls) if hasattr(wm, 'map_groups') else [wm.evaluate_groups(*c) for c in calls]
            raws = [{aid: g['all'] for aid, g in r.items()} for r in graws]
        except W.FairnessUnavailable:
            graws, use_groups = None, False
    if graws is None:
        raws = wm.map(calls) if hasattr(wm, 'map') else [wm.evaluate(*c) for c in calls]
    for i, ((m, state, acts, aid_of, inc, _seed), raw) in enumerate(zip(prepared, raws)):
        out = W.normalize(raw, acts, K, H)
        rollouts += len(acts) * K
        base = np.asarray(out['base'])
        ref = np.asarray(out['inc']) if inc else base
        for s in strategies:
            key = strategy_key(s)
            aid = aid_of.get(key)
            j = np.asarray(out[aid]) if aid else base
            gb_, gi = base - j, ref - j
            row = dict(world=int(m['seed']), gain_base=float(gb_.mean()), gain_inc=float(gi.mean()),
                       sd=float(gb_.std(ddof=1)) if K > 1 else 0.0, sd_inc=float(gi.std(ddof=1)) if K > 1 else 0.0,
                       feasible=aid is not None)
            if use_groups:
                gb = graws[i]['base']
                gc = graws[i][aid] if aid else gb
                row['groups'] = {g: float(np.mean(np.asarray(gb[g]) - np.asarray(gc[g]))) for g in W.FAIR_GROUPS}
                if all(f'n_{g}' in gb for g in W.FAIR_GROUPS) and 'cvar90' in gb:
                    row['inc'] = {g: float(np.mean([(gc[g][k] - gb[g][k]) / max(1, gb[f'n_{g}'][k])
                                                    for k in range(int(K))])) for g in W.FAIR_GROUPS}
                    row['cvar90_inc'] = float(np.mean(np.asarray(gc['cvar90']) - np.asarray(gb['cvar90'])))
                    row['p90_inc'] = float(np.mean(np.asarray(gc['p90']) - np.asarray(gb['p90'])))
            per[key].append(row)
    return per, rollouts


class _Boot:
    """World-cluster bootstrap with ONE resampling plan shared by every metric of a strategy (pooled-row means)."""

    def __init__(self, rows, b, seed):
        worlds = sorted({r['world'] for r in rows})
        idx = {w: i for i, w in enumerate(worlds)}
        self.n = len(worlds)
        self.wi = np.array([idx[r['world']] for r in rows], int)
        self.pick = np.random.default_rng(seed).integers(0, self.n, (int(b), self.n))
        self.den = np.bincount(self.wi, minlength=self.n)[self.pick].sum(1)

    def q(self, vals, q):
        s = np.bincount(self.wi, weights=np.asarray(vals, float), minlength=self.n)
        return float(np.quantile(s[self.pick].sum(1) / self.den, q))


def summarize(rows, lam, b=1000, seed=20261107, fair_tol=0.0, bound_s=GROUP_BOUND_S, bound_stat='mean',
              fairer_total='mean'):
    """Per-cell expected gains (vs base, vs incumbent) with one-sided 95% world-cluster bounds; per-group gains and
    the strict no-harm 'fair' flag; per-vehicle group increases (s/veh, positive = worse), CVaR90 / p90 changes, the
    'bounded' flag (every group increase <= bound_s; bound_stat 'mean' = registered point estimate, 'ucb' = 95%
    upper bound) and the 'fairer_ok' flag (total gain_vs_base >= 0; fairer_total 'mean' or 'lcb')."""
    bt = _Boot(rows, b, seed)
    n = max(1, len(rows))
    mean_b = float(np.mean([r['gain_base'] for r in rows]))
    mean_i = float(np.mean([r['gain_inc'] for r in rows]))
    dis = float(np.mean([r['sd'] for r in rows]))
    lcb_b = bt.q([r['gain_base'] for r in rows], 0.05)
    lcb_i = bt.q([r['gain_inc'] for r in rows], 0.05)
    out = dict(n_states=len(rows), feasible=float(np.mean([r['feasible'] for r in rows])), gain_vs_base=round(mean_b, 2),
               gain_vs_base_lcb=round(lcb_b, 2), gain_vs_inc=round(mean_i, 2), gain_vs_inc_lcb=round(lcb_i, 2),
               disagreement=round(dis, 2), score=round(lcb_b - lam * dis / math.sqrt(n), 2),
               score_vs_inc=round(lcb_i - lam * float(np.mean([r['sd_inc'] for r in rows])) / math.sqrt(n), 2))
    if rows and all('groups' in r for r in rows):
        gm = {g: round(float(np.mean([r['groups'][g] for r in rows])), 2) for g in W.FAIR_GROUPS}
        gl = {g: round(bt.q([r['groups'][g] for r in rows], 0.05), 2) for g in W.FAIR_GROUPS}
        worst = min(gl, key=gl.get)
        out.update(group_gain_vs_base=gm, group_gain_vs_base_lcb=gl, worst_group=worst,
                   worst_group_lcb=gl[worst], fair=bool(lcb_b > 0 and all(v >= -fair_tol for v in gl.values())))
    else:
        out.update(fair=None)
    if rows and all('inc' in r for r in rows):
        im = {g: round(float(np.mean([r['inc'][g] for r in rows])), 3) for g in W.FAIR_GROUPS}
        iu = {g: round(bt.q([r['inc'][g] for r in rows], 0.95), 3) for g in W.FAIR_GROUPS}
        wg = max(im, key=im.get)
        cv = float(np.mean([r['cvar90_inc'] for r in rows]))
        stat = iu if bound_stat == 'ucb' else im
        total = lcb_b if fairer_total == 'lcb' else mean_b
        out.update(group_inc=im, group_inc_ucb=iu, worst_group_inc=im[wg], worst_group_inc_group=wg,
                   worst_group_inc_ucb=max(iu.values()), cvar90_inc=round(cv, 3),
                   cvar90_inc_ucb=round(bt.q([r['cvar90_inc'] for r in rows], 0.95), 3),
                   p90_inc=round(float(np.mean([r['p90_inc'] for r in rows])), 3),
                   bounded=bool(all(v <= bound_s for v in stat.values())), bound_s=bound_s, bound_stat=bound_stat,
                   fairer_ok=bool(total >= 0.0),
                   fairer_key=[math.floor(im[wg] / FAIRER_BIN_S) * FAIRER_BIN_S, round(cv, 3)])
    else:
        out.update(bounded=None, fairer_ok=None)
    return out


def admits(archive, res):
    if archive == 'delay':
        return True
    return bool(res.get({'bounded': 'bounded', 'fairer': 'fairer_ok', 'fair': 'fair'}[archive]))


def rank_key(archive):
    """Sort key (ascending = better first)."""
    if archive == 'fairer':
        return lambda r: (tuple(r['fairer_key']), -r['score'])
    return lambda r: -r['score']


def search_cell(wm, states, cfg=None, log=None):
    """MAP-Elites + evolution over allowed (non-banned) strategies for ONE demand cell's start states. Archives
    'delay', 'bounded', 'fairer' (and the strict 'fair' flag) all feed parents and finals; the incumbent and the
    banned solo2 comparator are always evaluated in the finals (labelled, never selected)."""
    cfg = dict(DEFAULTS, **(cfg or {}))
    rng = random.Random(cfg['seed'])
    lam, groups = float(cfg['lam']), bool(cfg['fairness'])
    sk = dict(fair_tol=float(cfg['fair_tol']), bound_s=float(cfg['bound_s']), bound_stat=cfg['bound_stat'],
              fairer_total=cfg['fairer_total'])
    screen = rng.sample(states, min(len(states), int(cfg['n_screen_states'])))
    pop = seeds() + [random_strategy(rng, cfg['max_roles']) for _ in range(int(cfg['n_init']))]
    evaluated, total = {}, 0
    archives = {a: {} for a in ARCHIVES}
    comp_keys = {strategy_key(c) for c in COMPARATORS}

    def run(cands, sts, K):
        nonlocal total
        cands = [c for c in cands if c is not None and strategy_key(c) not in evaluated and not banned(c)]
        if not cands:
            return
        per, n = evaluate(wm, sts, cands, K, cfg['H'], groups=groups)
        total += n
        for c in cands:
            k = strategy_key(c)
            res = dict(strategy=c, key=k, descriptor=list(descriptor(c)), **summarize(per[k], lam, **sk))
            evaluated[k] = res
            d = descriptor(c)
            for a, arch in archives.items():
                if admits(a, res) and (d not in arch or rank_key(a)(res) < rank_key(a)(arch[d])):
                    arch[d] = res
        if log:
            log(dict(evaluated=len(evaluated), rollouts=total, **{f'archive_{a}': len(v) for a, v in archives.items()}))

    def top(arch, a, n):
        return sorted(arch.values(), key=rank_key(a))[:n]

    run(pop, screen, cfg['K_screen'])
    for _ in range(int(cfg['generations'])):
        ranked = sorted(evaluated.values(), key=lambda r: r['score'], reverse=True)
        parents = [r['strategy'] for r in ranked[:int(cfg['n_elite'])]]
        for a in ARCHIVES:
            parents += [r['strategy'] for r in top(archives[a], a, int(cfg['n_elite']) if a != 'fair'
                                                   else int(cfg['n_elite']) // 2)]
        kids = [mutate(rng, rng.choice(parents), cfg['p_mutate']) for _j in range(int(cfg['n_children']))]
        run(kids, screen, cfg['K_screen'])
    nf = int(cfg['finalists'])
    fin = {}
    for r in sorted(evaluated.values(), key=lambda r: r['score'], reverse=True)[:nf]:
        fin[r['key']] = r
    for a in ('bounded', 'fairer', 'fair', 'delay'):
        for r in top(archives[a], a, nf):
            if len(fin) < 3 * nf:
                fin.setdefault(r['key'], r)
    finals = [dict(f['strategy']) for f in fin.values()]
    for c in COMPARATORS:
        if strategy_key(c) not in {strategy_key(f) for f in finals}:
            finals.append(dict(c))
    fstates = states
    if int(cfg['n_final_states']) and len(states) > int(cfg['n_final_states']):
        fstates = random.Random(cfg['seed'] + 1).sample(states, int(cfg['n_final_states']))
    per, n = evaluate(wm, fstates, finals, cfg['K_final'], cfg['H'], groups=groups)
    total += n
    final = []
    for f in finals:
        k = strategy_key(f)
        final.append(dict(strategy=f, key=k, descriptor=list(descriptor(f)), comparator=k in comp_keys,
                          banned=banned(f), **summarize(per[k], lam, **sk)))
    final.sort(key=lambda r: r['score'], reverse=True)
    cands = [r for r in final if not r['comparator'] and not r['banned']]

    def best(a):
        xs = sorted((r for r in cands if admits(a, r)), key=rank_key(a))
        return xs[0] if xs else None
    beats = [r for r in cands if r['gain_vs_inc_lcb'] > 0 and r.get('bounded') is not False]
    fairness = ('per-vehicle' if final and final[0].get('bounded') is not None else
                'groups only' if final and final[0].get('fair') is not None else
                'unavailable (WM has no per-group delay)')

    def listing(a):
        return [dict(descriptor=list(d), key=r['key'], name=r['strategy']['name'], score=r['score'],
                     worst_group_inc=r.get('worst_group_inc'), cvar90_inc=r.get('cvar90_inc'))
                for d, r in sorted(archives[a].items(), key=lambda kv: rank_key(a)(kv[1]))]
    return dict(schema='wmc-strategy-search-v3', config=cfg, incumbent=INCUMBENT['name'], banned_v_w=list(BANNED_V_W),
                n_states=len(states), n_screen_states=len(screen), n_final_states=len(fstates),
                evaluated=len(evaluated), wm_rollouts=total, fairness=fairness,
                archive_cells={a: len(v) for a, v in archives.items()}, final=final,
                best={a: best(a) for a in ARCHIVES}, best_beating_incumbent=beats[0] if beats else None,
                comparators={r['strategy']['name']: {k: r.get(k) for k in (
                    'gain_vs_base', 'gain_vs_base_lcb', 'group_inc', 'worst_group_inc', 'cvar90_inc', 'bounded')}
                    for r in final if r['comparator']},
                archives={a: listing(a) for a in ARCHIVES})


def select_for_sumo(result, k=4, archive='delay', fair=False, exclude=()):
    """Top strategies of one archive, distinct descriptor cells first (then the next best to fill k); comparators,
    banned strategies and strategies that never activated on the final states are excluded; 'delay' / 'bounded'
    picks also need a positive predicted total gain vs u0."""
    archive = 'fair' if fair else archive
    ok = []
    for r in sorted(result['final'], key=rank_key(archive)):
        if r.get('comparator') or r.get('banned') or banned(r['strategy']) or not admits(archive, r):
            continue
        if r['key'] in set(exclude) or not r.get('feasible') or (archive in ('delay', 'bounded')
                                                                   and r['gain_vs_base'] <= 0):
            continue
        ok.append(r)
    out, cells = [], set()
    for r in ok:
        if tuple(r['descriptor']) not in cells and len(out) < k:
            out.append(r)
            cells.add(tuple(r['descriptor']))
    for r in ok:
        if len(out) >= k:
            break
        if r not in out:
            out.append(r)
    return [r['strategy'] for r in out]


def wm_arms(result, k=4):
    """ARMS.json WM groups: archive (a), then (b) and (c) excluding strategies already picked by earlier groups."""
    out, seen = {}, set()
    for group, arch in (('wm', 'delay'), ('wm_bounded', 'bounded'), ('wm_fairer', 'fairer'), ('wm_fair', 'fair')):
        out[group] = select_for_sumo(result, k, arch, exclude=seen)
        seen |= {strategy_key(s) for s in out[group]}
    return out


REFERENCE_ARM = ('wall_g_duo12', 'wall_g_solo2_gate0')                  # S-MULTICRIT measured gentle rules
STRUCTURED_ARM = ('wall_g_alt21', 'wall_g_relay21', 'g2_stagger40', 'g2_gap1')


def reference_arm():
    by = {s['name']: s for s in seeds()}
    return [by[n] for n in REFERENCE_ARM]


def structured_arm(k=4):
    """Structured arm (no WM): a fixed, pre-registered set of the owner's gentle combinations (alternating holders,
    relay holders, a staggered second lane-2 hold, a middle-lane gap-giver)."""
    by = {s['name']: s for s in seeds()}
    return [by[n] for n in STRUCTURED_ARM[:k]]


def random_arm(k=4, seed=20261107):
    rng = random.Random(seed)
    return [random_strategy(rng) for _ in range(k)]
