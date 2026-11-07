"""Creative / variety-seeking strategy exploration with the WM as a PROPOSAL GENERATOR only (C4 WM-C, 2026-11-07).

The WM failed qualification as a predictor (within-moment Spearman 0.14, prec@10 0.60, regret 0.84, group harm
underestimated ~2x), so nothing here trusts its ranking beyond a coarse screen: SUMO verifies. This module wraps the
registered strategy space and the WM evaluation of strategy_search (imported as SS, UNCHANGED; its rollouts,
summaries, archives and scores are reused verbatim) and adds

  * fine_descriptor(s)        which lanes/ranks/lane targets, v_w mix, hold bucket, onset pattern, gate, start gate,
                              phase shape, zone, trigger, spread (the SS descriptor ignores most of these)
  * genome_distance(a, b)     exact optimal role matching (bitmask DP assignment, <= 6 roles) over per-field costs
                              + spec-level differences; d(a, a) = 0, symmetric
  * behaviour vectors         activation mask over the screen states, members activated per state, the 5-group
                              per-vehicle increase vector and the gain; novelty = mean combined distance to the k
                              nearest of everything evaluated + SS.seeds() (structured / reference families) + comparators
  * archive 'novel'           keyed on fine_descriptor; admission: feasible >= min_feasible, predicted worst-group
                              increase <= novel_harm_s (5 s/veh = half the owner's 10 s bound, because the WM
                              underestimates group harm ~2x), gain_vs_base >= -novel_eps; elite per cell = most novel
  * min_feasible              (default 0.25) before ANY archive admission (never-activating strategies used to get
                              gain 0 and occupy cells)
  * operators                 mutate_struct (remove / add role, move or swap lane/rank, flip wall<->lanes, spread
                              200<->300, cooldown stepped (onset+hold)..240, mixed-group phases, toggle a lane change),
                              SS.mutate, crossover(a, b), ~15% fresh random immigrants per generation; every child passes
                              SS.validate AND the WMVERIFY executor rules (wmverify_check, reimplemented here)
  * select_diverse_for_sumo   allowed + feasible + WM worst-group <= 5 s, not within delta of the incumbent /
                              structured / reference families; farthest-point clustering on genome_distance, one pick
                              per cluster (best predicted inside the cluster); in- and out-of-WM-data-support picks are
                              clustered and ranked SEPARATELY (never against each other) -> ARMS group 'wm_novel'.
  * lane gate awareness       the CAVs' own-sensing LCEXEC lane gate withholds ~99% of program lane-change requests in
                              heavy traffic; gate_pass(s, states) applies the exact LaneRules gate to the observable
                              start state (constant-velocity extrapolation to t0 + onset; a PROXY: the WM exposes no
                              imagined per-vehicle states); lane-change plans enter 'novel' / 'wm_novel' only with
                              gate_pass_rate >= min_gate_pass (0.3); gate-pass bucket is a fine_descriptor dimension;
                              operators gap_open (hold in target lane + delayed changer into it) and early_change
                              (changers in zone far/full with onset 0)
'disagreement' (mostly missing-CRN noise) is reported by SS.summarize but never used for ranking (lam 0.0);
'bonus' (default 0) is a hook: parent ranking key = score + bonus * novelty.

    python -m tools.wzprop.strategy_explore --states S.json --cells D3200_p60 --out NEW_DIR --wm wma:WMA_READY.json \
        --wm-arg checkpoint=... --workers 8 --mem-fraction 0.12 [--config '{...}'] [--k 4]
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import re
import sys
import time

import numpy as np

from . import geometry as G
from . import strategy_search as SS
from . import wm_backend as W

EXPLORE_SCHEMA = 'wmc-strategy-explore-v1'
DEFAULTS = dict(SS.DEFAULTS, n_init=150, generations=6, n_children=48, n_elite=4, p_mutate=0.6, lam=0.0,
                n_screen_states=16, K_screen=3, finalists=12, n_final_states=30, K_final=6, immigrants=0.15,
                min_feasible=0.25, novelty_k=10, seed=20261108, p_cross=0.3, p_struct=0.5, novel_harm_s=5.0,
                novel_eps=10.0, bonus=0.0, select_n=6, select_delta=1.0, max_tries=30, min_gate_pass=0.3,
                p_team=0.1, p_early=0.08)
ARCHIVES = tuple(SS.ARCHIVES) + ('novel',)
GENOME_SCALE = 3.0           # genome distance units per behaviour distance unit in the combined novelty distance
ROLE_GAP = 2.5               # cost of an unmatched role (present in one strategy only)

# ------------------------------------------------------------------------------------------ WMVERIFY rules (mirror)
# Reimplementation of c4-wmverify tools/wzprop/wmverify.py check_spec (~:78-88, :159-210); NOT imported (no vendored
# copy). Cross-checked by tests against known-valid / known-invalid specs.
SPEC_KEYS = frozenset({'name', 'trigger', 'zone', 'spread_m', 'cooldown_s', 'slots', 'gate', 'start_gate', 'phases',
                       'roles'})
REQUIRED_KEYS = SPEC_KEYS - {'gate', 'start_gate', 'phases'}
ROLE_KEYS = frozenset({'lane', 'rank', 'v_w', 'dmax', 'hold_s', 'onset_s', 'lc_target'})
NAME_RE = re.compile(r'^[A-Za-z][A-Za-z0-9_.-]{0,63}$')
RESERVED = ('u0', 'base', 'wall_g_solo2', 'wall_h_solo2', 'inc_core', 'solo2_core')
SPREADS = (G.WALL_SPREAD, G.R_SPREAD)
SLOTS = (1,)
CLOSING_SUMO_LANE = 0
LANE_EXECUTORS = ('plant', 'n2', 'lcexec')
DEFAULT_LANE_EXECUTOR = 'lcexec'


def wmverify_check(spec, lane_executor=DEFAULT_LANE_EXECUTOR):
    """Strict WMVERIFY executor validation (raises ValueError): known keys only, exact role keys (CAV programs only:
    no vehicle ids), int/number types, SS.validate (registered grids, banned 15 m/s rejected), spread in {200, 300},
    slots == 1, cooldown <= max hold + max onset (240 s), executable lane targets."""
    if not isinstance(spec, dict):
        raise ValueError('spec must be a dict')
    unknown = set(spec) - SPEC_KEYS
    if unknown:
        raise ValueError(f'unknown spec keys {sorted(unknown)}')
    missing = REQUIRED_KEYS - set(spec)
    if missing:
        raise ValueError(f'missing spec keys {sorted(missing)}')
    name = spec['name']
    if not isinstance(name, str) or not NAME_RE.match(name) or name in RESERVED or 'minus' in name:
        raise ValueError(f'strategy name {name!r} not allowed')
    if not isinstance(spec['roles'], list) or not spec['roles']:
        raise ValueError('roles must be a non-empty list')
    for r in spec['roles']:
        if not isinstance(r, dict) or set(r) != ROLE_KEYS:
            raise ValueError(f'role keys must be exactly {sorted(ROLE_KEYS)}')
        for k in ('lane', 'rank', 'hold_s', 'onset_s'):
            if type(r[k]) is not int:
                raise ValueError(f'role {k} must be an int')
        if r['lc_target'] is not None and type(r['lc_target']) is not int:
            raise ValueError('lc_target must be an int or None')
        for k in ('v_w', 'dmax'):
            if isinstance(r[k], bool) or not isinstance(r[k], (int, float)):
                raise ValueError(f'role {k} must be a number')
    for k in ('cooldown_s', 'slots'):
        if type(spec[k]) is not int:
            raise ValueError(f'{k} must be an int')
    ph = spec.get('phases')
    if ph is not None and (not isinstance(ph, list) or any(not isinstance(p, list) or any(type(i) is not int for i in p)
                                                           for p in ph)):
        raise ValueError('phases must be None or a list of lists of role indices')
    if spec.get('start_gate') is not None and not isinstance(spec['start_gate'], str):
        raise ValueError('start_gate must be None or a registered name')
    s = dict(spec, gate=spec.get('gate'), start_gate=spec.get('start_gate'),
             phases=None if ph is None else [list(p) for p in ph], roles=[dict(r) for r in spec['roles']])
    SS.validate(s)
    if float(s['spread_m']) not in SPREADS:
        raise ValueError(f"spread_m {s['spread_m']} not registered {SPREADS}")
    if s['slots'] not in SLOTS:
        raise ValueError(f"slots {s['slots']} not registered {SLOTS}")
    g = SS.grids()
    if int(s['cooldown_s']) > max(g['hold_s']) + max(g['onset_s']):
        raise ValueError('cooldown above the registered hold + onset range')
    if any(r['lc_target'] is not None for r in s['roles']):
        if lane_executor not in LANE_EXECUTORS:
            raise ValueError(f'unknown lane executor {lane_executor!r}')
        if lane_executor == 'plant' and any(r['lc_target'] == CLOSING_SUMO_LANE for r in s['roles']):
            raise ValueError('closing-lane entry is not executable on the frozen G5 lane path (use lcexec)')
    return s


def allowed(s):
    """Child admission: SS.validate (no banned speed) AND the WMVERIFY executor rules."""
    try:
        SS.validate(s)
        wmverify_check(s)
    except (ValueError, KeyError, TypeError):
        return False
    return not SS.banned(s)


def cooldown_max():
    g = SS.grids()
    return int(max(g['hold_s']) + max(g['onset_s']))


# ------------------------------------------------------------------------------------------ descriptors / distance
def _lc_dir(r):
    return None if r['lc_target'] is None else int(r['lc_target']) - int(r['lane'])


def _base_cooldown(roles):
    return int(max(int(r['onset_s']) + int(r['hold_s']) for r in roles))


def phase_shape(s):
    ph = s.get('phases')
    return None if not ph else tuple(sorted(len(p) for p in ph))


def fine_descriptor(s, gate_rate=None):
    """(roles (lane, rank, lc_target|-1) sorted, v_w mix, hold bucket, onset pattern, gate, start_gate, phase shape,
    zone, trigger, spread, gate-pass bucket ('hold' for hold-only plans))."""
    roles = s['roles']
    on = [int(r['onset_s']) for r in roles]
    onset = 'sync0' if max(on) == 0 else ('delayed' if min(on) == max(on) else 'stagger')
    hb = max(int(r['hold_s']) for r in roles)
    return (tuple(sorted((int(r['lane']), int(r['rank']), -1 if r['lc_target'] is None else int(r['lc_target']))
                         for r in roles)),
            tuple(sorted({float(r['v_w']) for r in roles})),
            0 if hb <= 90 else (1 if hb <= 120 else 2),
            onset, s.get('gate'), s.get('start_gate'), phase_shape(s), s['zone'], s['trigger'],
            float(s['spread_m']), gate_bucket(s, gate_rate))


def _role_cost(a, b):
    c = 0.0
    if int(a['lane']) != int(b['lane']):
        c += 1.0
    if int(a['rank']) != int(b['rank']):
        c += 0.5
    da, db = _lc_dir(a), _lc_dir(b)
    if (da is None) != (db is None):
        c += 1.0
    elif da != db:
        c += 0.5
    c += abs(float(a['v_w']) - float(b['v_w'])) / 4.0
    c += abs(float(a['dmax']) - float(b['dmax']))
    c += abs(int(a['hold_s']) - int(b['hold_s'])) / 180.0
    c += abs(int(a['onset_s']) - int(b['onset_s'])) / 120.0
    return c


def assignment_cost(C):
    """Exact minimum-cost perfect matching of a square cost matrix (bitmask DP; n <= ~10)."""
    n = len(C)
    inf = float('inf')
    dp = [inf] * (1 << n)
    dp[0] = 0.0
    for mask in range(1 << n):
        v = dp[mask]
        if v == inf:
            continue
        i = bin(mask).count('1')
        if i >= n:
            continue
        row = C[i]
        for j in range(n):
            bit = 1 << j
            if not mask & bit:
                w = v + row[j]
                if w < dp[mask | bit]:
                    dp[mask | bit] = w
    return dp[(1 << n) - 1]


def _spec_cost(a, b):
    c = 0.0
    if a['trigger'] != b['trigger']:
        c += 1.0
    if a['zone'] != b['zone']:
        c += 0.5
    if float(a['spread_m']) != float(b['spread_m']):
        c += 0.25
    c += 0.5 * abs(int(a['cooldown_s']) - int(b['cooldown_s'])) / float(cooldown_max())
    ga, gb = a.get('gate'), b.get('gate')
    if (ga is None) != (gb is None):
        c += 0.75
    elif ga is not None:
        c += 0.5 * abs(float(ga) - float(gb)) / 10.0
    if a.get('start_gate') != b.get('start_gate'):
        c += 0.5
    pa, pb = phase_shape(a), phase_shape(b)
    if (pa is None) != (pb is None):
        c += 1.0
    elif pa != pb:
        c += 0.5
    return c


def genome_distance(a, b):
    """Optimal role matching (per-field role costs, ROLE_GAP per unmatched role) + spec-level field differences."""
    ra, rb = list(a['roles']), list(b['roles'])
    n = max(len(ra), len(rb))
    C = [[(_role_cost(ra[i], rb[j]) if i < len(ra) and j < len(rb) else
           (0.0 if i >= len(ra) and j >= len(rb) else ROLE_GAP)) for j in range(n)] for i in range(n)]
    return round(assignment_cost(C) + _spec_cost(a, b), 6)


# ------------------------------------------------------------------------------------------ behaviour / novelty
def members_per_state(s, states):
    """Members the strategy's trigger selects at each state (phase = state index, as SS.evaluate)."""
    out = []
    for i, m in enumerate(states):
        observed = list((m.get('observed') or {}).get('vehicles') or [])
        mem = SS.select_members(s, m['all_cavs_t0'], observed, phase=i)
        out.append(0 if mem is None else len(mem))
    return out


def behaviour_vector(rows, members, res, gscale):
    """[activation mask over states | members / 6 | 5-group increase / 5 s | gain / gscale] (weights folded in so that
    the euclidean norm averages over the mask/member blocks)."""
    n = max(1, len(rows))
    mask = np.array([1.0 if r['feasible'] else 0.0 for r in rows]) / math.sqrt(n)
    mem = np.array([float(x) / 6.0 for x in members]) / math.sqrt(max(1, len(members)))
    gi = res.get('group_inc') or {}
    inc = np.array([float(gi.get(g, 0.0)) / 5.0 for g in W.FAIR_GROUPS]) / math.sqrt(len(W.FAIR_GROUPS))
    gain = np.array([float(res.get('gain_vs_base') or 0.0) / float(gscale)])
    return np.concatenate([mask, mem, inc, gain])


def combined_distance(dg, ba=None, bb=None):
    gpart = dg / GENOME_SCALE
    if ba is None or bb is None or len(ba) != len(bb):
        return 2.0 * gpart
    return gpart + float(np.linalg.norm(np.asarray(ba) - np.asarray(bb)))


def novelty(x, ref, k=10, dist=None):
    """Mean distance of item x to its k nearest items of ref (items: {'strategy', optional 'behaviour'}; an item with
    the same strategy key as x is skipped). dist(x, y) defaults to the combined genome + behaviour distance."""
    dist = dist or (lambda p, q: combined_distance(genome_distance(p['strategy'], q['strategy']), p.get('behaviour'),
                                                   q.get('behaviour')))
    kx = SS.strategy_key(x['strategy'])
    ds = sorted(dist(x, y) for y in ref if SS.strategy_key(y['strategy']) != kx)
    if not ds:
        return 0.0
    return round(float(np.mean(ds[:max(1, int(k))])), 6)


class Space:
    """Genome-distance cache keyed by strategy-key pairs."""

    def __init__(self):
        self.cache = {}

    def d(self, a, b, ka=None, kb=None):
        ka, kb = ka or SS.strategy_key(a), kb or SS.strategy_key(b)
        if ka == kb:
            return 0.0
        key = (ka, kb) if ka < kb else (kb, ka)
        v = self.cache.get(key)
        if v is None:
            v = self.cache[key] = genome_distance(a, b)
        return v


# ------------------------------------------------------------------------------------------ WM data support
def support_tag(s):
    """Heuristic in/out of WM data support (no OOD signal is exposed by the WM backends): the WM sees one activation
    = the roles of one phase; WMDATA / NM2 / LCMOVES training families have <= 3 cooperating CAVs with <= 1 lane
    changer. 'in' iff every phase activates <= 3 roles with <= 1 lane change."""
    groups = [list(range(len(s['roles'])))] if not s.get('phases') else s['phases']
    n_act = max(len(p) for p in groups)
    n_lc = max(sum(s['roles'][i]['lc_target'] is not None for i in p) for p in groups)
    reasons = []
    if n_act > 3:
        reasons.append(f'{n_act} roles per activation > 3')
    if n_lc > 1:
        reasons.append(f'{n_lc} lane changers per activation > 1')
    return ('out' if reasons else 'in'), reasons


SUPPORT_BASIS = 'heuristic: per activation <= 3 roles and <= 1 lane changer (WMDATA/NM2/LCMOVES family shapes)'


# ------------------------------------------------------------------------------------------ own-sensing lane gate
# Reimplementation (pure Python, no torch) of the LCEXEC program lane gate: tools/coopmwm/lc_families.py lane_gate ->
# wms.control.lanes.LaneRules._target_gap_ok with the default (LANECAL) rules, sensing range contract.SENSOR_RANGE_M,
# edge guard v * CONTROL_DT_S + edge_guard_m upstream of the drop. Cross-checked against _target_gap_ok in the tests.
LANE_RULES = dict(gap_base_m=1.0, gap_time_s=1.2, closing_time_s=4.5, rear_gap_base_m=3.0, rear_gap_time_s=1.25,
                  rear_closing_time_s=6.0, edge_guard_m=10.0)
SENSOR_RANGE_M = 100.0
CONTROL_DT_S = 1.0
VEH_LENGTH_M = 5.0           # wms.core.constants.DEFAULT_LENGTH_M (rows carry no length)
GATE_BASIS = ('observable start state (CAV self + CAV-detected rows on edge up, within 100 m of the member): exact '
              'LaneRules gate at t0 for onset 0, constant-velocity extrapolation to t0 + onset otherwise; the WM '
              'backends expose no imagined per-vehicle states, so this is a snapshot proxy, not an imagined-state gate')


def target_gap_ok(ego, others, target, rules=None):
    """ego {s, v, length}; others [{s, v, length, lane}] (ego excluded): LaneRules._target_gap_ok for one ego.
    Front bumper gap >= base + time * v_ego + closing * max(0, v_ego - v_leader); rear gap >= rear_base + rear_time *
    v_follower + rear_closing * max(0, v_follower - v_ego); an overlapping vehicle gives a negative gap (blocks);
    nothing in the target lane = free."""
    R = dict(LANE_RULES, **(rules or {}))
    s, v, ln = float(ego['s']), max(0.0, float(ego['v'])), float(ego.get('length', VEH_LENGTH_M))
    fg, vf, rg, vr = math.inf, 0.0, math.inf, 0.0
    for o in others:
        if int(o['lane']) != int(target):
            continue
        ds = float(o['s']) - s
        if ds >= 0:
            g = ds - float(o.get('length', VEH_LENGTH_M))
            if g < fg:
                fg, vf = g, max(0.0, float(o['v']))
        else:
            g = -ds - ln
            if g < rg:
                rg, vr = g, max(0.0, float(o['v']))
    need_f = R['gap_base_m'] + R['gap_time_s'] * v + R['closing_time_s'] * max(0.0, v - vf)
    need_r = R['rear_gap_base_m'] + R['rear_gap_time_s'] * vr + R['rear_closing_time_s'] * max(0.0, vr - v)
    return (math.isinf(fg) or fg >= need_f) and (math.isinf(rg) or rg >= need_r)


def lane_gate_ok(member, target, rows, dt=0.0):
    """One program lane request of `member` (SUMO frame row id/edge/lane/x/v) into SUMO lane `target`, dt seconds
    after the snapshot (constant-velocity extrapolation of every row). Mirrors lc_families.lane_gate: same lane =
    pass; adjacent only; closing lane 0 only before the drop X_B; edge guard near the drop unless leaving lane 0;
    target-lane gaps over the rows sensed within SENSOR_RANGE_M."""
    src, target = int(member['lane']), int(target)
    s = float(member['x']) + float(member['v']) * dt
    v = float(member['v'])
    if target == src:
        return True
    if abs(target - src) != 1 or not 0 <= target <= 2:
        return False
    if target == CLOSING_SUMO_LANE and s >= G.X_B:
        return False
    if src != CLOSING_SUMO_LANE and 0 <= G.X_B - s < v * CONTROL_DT_S + LANE_RULES['edge_guard_m']:
        return False
    others, seen = [], {member['id']}
    for r in rows:
        if r['id'] in seen or r.get('edge', 'up') != 'up':
            continue
        seen.add(r['id'])
        so = float(r['x']) + float(r['v']) * dt
        if abs(so - s) <= SENSOR_RANGE_M:
            others.append(dict(s=so, v=float(r['v']), lane=int(r['lane']), length=VEH_LENGTH_M))
    return target_gap_ok(dict(s=s, v=v, length=VEH_LENGTH_M), others, target)


def has_lane_roles(s):
    return any(r['lc_target'] is not None for r in s['roles'])


def gate_pass(s, states):
    """(pass rate or None, n_requests): every activated lane-change role at every state is one request."""
    n = ok = 0
    if not has_lane_roles(s):
        return None, 0
    for i, m in enumerate(states):
        observed = list((m.get('observed') or {}).get('vehicles') or [])
        mem = SS.select_members(s, m['all_cavs_t0'], observed, phase=i)
        if mem is None:
            continue
        rows = list(m['all_cavs_t0']) + observed
        for r, c in zip(SS.active_roles(s, i), mem):
            if r['lc_target'] is None:
                continue
            n += 1
            ok += bool(lane_gate_ok(c, r['lc_target'], rows, dt=float(r['onset_s'])))
    return (round(ok / n, 4) if n else None), n


def gate_bucket(s, rate):
    if not has_lane_roles(s):
        return 'hold'
    if rate is None:
        return 'na'
    return 'g0' if rate < 0.1 else ('g1' if rate < 0.3 else ('g2' if rate < 0.6 else 'g3'))


def gate_ok(res, cfg):
    """Lane-change plans need gate_pass_rate >= min_gate_pass; hold-only plans are unaffected."""
    if not has_lane_roles(res['strategy']):
        return True
    gp = res.get('gate_pass_rate')
    return gp is not None and float(gp) >= float(cfg['min_gate_pass'])


# ------------------------------------------------------------------------------------------ operators
def _name(prefix, s):
    s['name'] = prefix + SS.strategy_key(s)
    return s


def _finish(s, prefix):
    roles = s['roles']
    if not roles:
        return None
    base = _base_cooldown(roles)
    s['cooldown_s'] = int(min(cooldown_max(), max(int(s.get('cooldown_s') or base), base)))
    s['slots'] = 1
    s['spread_m'] = float(s['spread_m'])
    _name(prefix, s)
    return s if allowed(s) else None


def _remap_phases(phases, removed):
    if not phases:
        return None
    out = []
    for p in phases:
        q = [i - (1 if i > removed else 0) for i in p if i != removed]
        if q:
            out.append(q)
    return out if 2 <= len(out) <= 3 else None


def _fix_lc(r):
    d = _lc_dir(r)
    if d is not None:
        t = int(r['lane']) + d
        r['lc_target'] = t if 0 <= t <= 2 else None


def _free_slots(s, roles):
    used = {(int(r['lane']), int(r['rank'])) for r in roles}
    ranks = (0,) if s['trigger'] == 'wall' else (0, 1)
    return [(ln, rk) for ln in (0, 1, 2) for rk in ranks if (ln, rk) not in used]


def random_partition(rng, n, mixed=True):
    """2-3 non-empty groups of role indices; mixed=True needs a group of >= 2 roles (n >= 3)."""
    if n < 2:
        return None
    if n == 2 or not mixed:
        k = rng.choice([2, 3]) if n >= 3 else 2
    else:
        k = rng.choice([2, 3]) if n >= 4 else 2
    idx = list(range(n))
    rng.shuffle(idx)
    cuts = sorted(rng.sample(range(1, n), k - 1))
    groups = [sorted(idx[a:b]) for a, b in zip([0] + cuts, cuts + [n])]
    return groups


STRUCT_OPS = ('remove_role', 'add_role', 'move_slot', 'swap_slots', 'flip_trigger', 'spread', 'cooldown',
              'mixed_phases', 'toggle_lc')


def _apply_op(rng, s, op):
    g = SS.grids(search=True)
    roles = s['roles']
    if op == 'remove_role':
        if len(roles) < 2:
            return False
        i = rng.randrange(len(roles))
        roles.pop(i)
        s['phases'] = _remap_phases(s.get('phases'), i)
    elif op == 'add_role':
        free = _free_slots(s, roles)
        if len(roles) >= 6 or not free:
            return False
        ln, rk = rng.choice(free)
        roles.append(dict(lane=ln, rank=rk, v_w=rng.choice(g['v_w']), dmax=rng.choice(g['dmax']),
                          hold_s=rng.choice(g['hold_s']), onset_s=rng.choice(g['onset_s']),
                          lc_target=rng.choice([None, None] + [x for x in (ln - 1, ln + 1) if 0 <= x <= 2])))
        if s.get('phases'):
            ph = [list(p) for p in s['phases']]
            ph[rng.randrange(len(ph))].append(len(roles) - 1)
            s['phases'] = ph
    elif op == 'move_slot':
        free = _free_slots(s, roles)
        if not free:
            return False
        r = roles[rng.randrange(len(roles))]
        r['lane'], r['rank'] = rng.choice(free)
        _fix_lc(r)
    elif op == 'swap_slots':
        if len(roles) < 2:
            return False
        i, j = rng.sample(range(len(roles)), 2)
        a, b = roles[i], roles[j]
        if (a['lane'], a['rank']) == (b['lane'], b['rank']):
            return False
        a['lane'], a['rank'], b['lane'], b['rank'] = b['lane'], b['rank'], a['lane'], a['rank']
        _fix_lc(a)
        _fix_lc(b)
    elif op == 'flip_trigger':
        if s['trigger'] == 'wall':
            s['trigger'] = 'lanes'
        else:
            keep, used = [], set()
            for i, r in enumerate(roles):
                if int(r['lane']) not in used:
                    used.add(int(r['lane']))
                    keep.append(i)
            if not keep:
                return False
            for i in sorted(set(range(len(roles))) - set(keep), reverse=True):
                roles.pop(i)
                s['phases'] = _remap_phases(s.get('phases'), i)
            for r in roles:
                r['rank'] = 0
            s['trigger'] = 'wall'
    elif op == 'spread':
        s['spread_m'] = 300.0 if float(s['spread_m']) == 200.0 else 200.0
    elif op == 'cooldown':
        base = _base_cooldown(roles)
        top = cooldown_max()
        choices = sorted({c for c in list(range(base, top + 1, 30)) + [top] if base <= c <= top})
        choices = [c for c in choices if c != int(s['cooldown_s'])] or choices
        s['cooldown_s'] = int(rng.choice(choices))
    elif op == 'mixed_phases':
        if len(roles) < 2:
            return False
        if len(roles) == 2:
            s['phases'] = None if s.get('phases') else [[0], [1]]
        else:
            s['phases'] = random_partition(rng, len(roles), mixed=True)
    elif op == 'toggle_lc':
        r = roles[rng.randrange(len(roles))]
        if r['lc_target'] is None:
            r['lc_target'] = rng.choice([x for x in (r['lane'] - 1, r['lane'] + 1) if 0 <= x <= 2])
        else:
            r['lc_target'] = None
    else:
        raise ValueError(op)
    return True


def _copy(s):
    t = dict(s)
    t['roles'] = [dict(r) for r in s['roles']]
    t['phases'] = None if not s.get('phases') else [list(p) for p in s['phases']]
    return t


def mutate_struct(rng, s, n_ops=None):
    """1-2 structural operators (STRUCT_OPS); the cooldown is kept within [onset+hold, 240]. None if invalid."""
    t = _copy(s)
    n_ops = n_ops or (2 if rng.random() < 0.3 else 1)
    done = 0
    for _ in range(4 * n_ops):
        if done >= n_ops:
            break
        if _apply_op(rng, t, rng.choice(STRUCT_OPS)):
            done += 1
    if not done:
        return None
    return _finish(t, 'xs_')


def crossover(rng, a, b):
    """Union of the parents' roles deduplicated on (lane, rank) (random parent first), capped at 6; trigger, zone,
    spread, gate, start gate and cooldown each from either parent; phases None or (25%) a fresh partition."""
    first, second = (a, b) if rng.random() < 0.5 else (b, a)
    roles, seen = [], set()
    for r in list(first['roles']) + list(second['roles']):
        k = (int(r['lane']), int(r['rank']))
        if k not in seen:
            seen.add(k)
            roles.append(dict(r))
    if len(roles) > 6:
        keep = sorted(rng.sample(range(len(roles)), 6))
        roles = [roles[i] for i in keep]
    t = dict(name='', roles=roles, slots=1)
    for k in ('trigger', 'zone', 'spread_m', 'gate', 'start_gate', 'cooldown_s'):
        t[k] = rng.choice([a, b]).get(k)
    if t['trigger'] == 'wall':
        kept, lanes = [], set()
        for r in roles:
            if int(r['lane']) not in lanes:
                lanes.add(int(r['lane']))
                kept.append(dict(r, rank=0))
        t['roles'] = roles = kept
    t['phases'] = random_partition(rng, len(roles), mixed=rng.random() < 0.5) \
        if len(roles) >= 2 and rng.random() < 0.25 else None
    return _finish(t, 'xc_')


def gap_open(rng, parent=None):
    """GAP-OPENING team move in the current schema: a hold role in target lane L (rank 1 = upstream of the lane's
    leading zone CAV, i.e. behind the intended merge point, with prob 0.7) starting at onset t0, plus a lane-change
    role from L +- 1 (rank 0) into L starting at t0 + delta (both on the onset grid, delta > 0); the holder's v_w is
    lowered to the grid minimum with prob 0.5. Added to a 'lanes' parent when its slots are free, else a fresh plan."""
    g = SS.grids(search=True)
    L = rng.choice([1, 1, 1, 2, 2, 0])
    src = rng.choice([x for x in (L - 1, L + 1) if 0 <= x <= 2])
    on = sorted(g['onset_s'])
    t0 = rng.choice(on[:-1])
    t1 = rng.choice([x for x in on if x > t0])
    hrank = 1 if rng.random() < 0.7 else 0
    holder = dict(lane=L, rank=hrank, v_w=min(g['v_w']) if rng.random() < 0.5 else rng.choice(g['v_w']),
                  dmax=rng.choice(g['dmax']), hold_s=rng.choice(g['hold_s']), onset_s=int(t0), lc_target=None)
    merger = dict(lane=src, rank=0, v_w=rng.choice(g['v_w']), dmax=rng.choice(g['dmax']), hold_s=rng.choice(g['hold_s']),
                  onset_s=int(t1), lc_target=L)
    if parent is not None and parent['trigger'] == 'lanes':
        t = _copy(parent)
        used = {(int(r['lane']), int(r['rank'])) for r in t['roles']}
        if (L, hrank) not in used and (src, 0) not in used and len(t['roles']) <= 4:
            t['roles'] += [holder, merger]
            t['phases'] = None
            t['cooldown_s'] = 0
            return _finish(t, 'xg_')
    t = dict(name='', trigger='lanes', zone=rng.choice(list(SS.ZONES)), spread_m=300.0, cooldown_s=0, slots=1,
             gate=rng.choice(SS.GATES) if rng.random() < 0.3 else None, start_gate=None, phases=None,
             roles=[holder, merger])
    return _finish(t, 'xg_')


def early_change(rng, parent=None):
    """EARLY CHANGE: lane-change roles in zone far / full with onset 0 (upstream, before the queue forms). On a
    parent: zone -> far/full and every lane-change role's onset -> 0 (one hold role becomes a changer if none);
    fresh: 1-2 changers (closing lane 0 -> 1 preferred)."""
    g = SS.grids(search=True)
    zone = rng.choice(['far', 'full'])
    if parent is not None:
        t = _copy(parent)
        t['zone'] = zone
        if not has_lane_roles(t):
            r = t['roles'][rng.randrange(len(t['roles']))]
            r['lc_target'] = rng.choice([x for x in (r['lane'] - 1, r['lane'] + 1) if 0 <= x <= 2])
        for r in t['roles']:
            if r['lc_target'] is not None:
                r['onset_s'] = 0
        t['cooldown_s'] = 0
        return _finish(t, 'xe_')
    roles = [dict(lane=0, rank=0, v_w=rng.choice(g['v_w']), dmax=rng.choice(g['dmax']), hold_s=rng.choice(g['hold_s']),
                  onset_s=0, lc_target=1)]
    if rng.random() < 0.5:
        ln = rng.choice([0, 1, 2])
        rk = 1 if ln == 0 else rng.choice([0, 1])
        roles.append(dict(lane=ln, rank=rk, v_w=rng.choice(g['v_w']), dmax=rng.choice(g['dmax']),
                          hold_s=rng.choice(g['hold_s']), onset_s=0,
                          lc_target=rng.choice([x for x in (ln - 1, ln + 1) if 0 <= x <= 2])))
    t = dict(name='', trigger='lanes', zone=zone, spread_m=rng.choice([200.0, 300.0]), cooldown_s=0, slots=1,
             gate=None, start_gate=None, phases=None, roles=roles)
    return _finish(t, 'xe_')


def immigrant(rng, max_roles=6):
    """Fresh random strategy (SS.random_strategy), sometimes with a mixed-group phase partition."""
    s = SS.random_strategy(rng, max_roles)
    if len(s['roles']) >= 3 and rng.random() < 0.2:
        t = _copy(s)
        t['phases'] = random_partition(rng, len(t['roles']), mixed=True)
        t = _finish(t, 'xi_')
        if t is not None:
            return t
    return s if allowed(s) else None


def make_children(rng, parents, n, cfg, seen):
    """n children: round(immigrants * n) fresh immigrants, the rest crossover / mutate_struct / SS.mutate. Every child
    passes SS.validate + wmverify_check, is not banned and is new (key not in seen)."""
    kids, keys = [], set()
    stats = dict(immigrants=0, crossover=0, team=0, early=0, struct=0, mutate=0, rejected=0)
    n_imm = int(round(float(cfg['immigrants']) * n))
    p = float(cfg['p_mutate'])
    pc, pt, pe = float(cfg['p_cross']), float(cfg['p_team']), float(cfg['p_early'])
    tries = 0
    while len(kids) < n and tries < n * int(cfg['max_tries']):
        tries += 1
        if stats['immigrants'] < n_imm:
            kind, c = 'immigrants', immigrant(rng, cfg['max_roles'])
        else:
            u = rng.random()
            if u < pc and len(parents) >= 2:
                pa, pb = rng.sample(parents, 2)
                kind, c = 'crossover', crossover(rng, pa, pb)
            elif u < pc + pt:
                kind, c = 'team', gap_open(rng, rng.choice(parents) if rng.random() < 0.5 else None)
            elif u < pc + pt + pe:
                kind, c = 'early', early_change(rng, rng.choice(parents) if rng.random() < 0.5 else None)
            elif u < pc + pt + pe + (1.0 - pc - pt - pe) * float(cfg['p_struct']):
                kind, c = 'struct', mutate_struct(rng, rng.choice(parents))
            else:
                kind, c = 'mutate', SS.mutate(rng, rng.choice(parents), p)
        if c is None or not allowed(c):
            stats['rejected'] += 1
            continue
        k = SS.strategy_key(c)
        if k in seen or k in keys:
            stats['rejected'] += 1
            continue
        keys.add(k)
        kids.append(c)
        stats[kind] += 1
    return kids, stats


# ------------------------------------------------------------------------------------------ archives
def feasible_ok(res, cfg):
    return float(res.get('feasible') or 0.0) >= float(cfg['min_feasible'])


def novel_admits(res, cfg):
    wg = res.get('worst_group_inc')
    return bool(feasible_ok(res, cfg) and wg is not None and float(wg) <= float(cfg['novel_harm_s'])
                and float(res['gain_vs_base']) >= -float(cfg['novel_eps']) and gate_ok(res, cfg))


def comparator_refs():
    """Strategies the novel selection must move away from: incumbent, banned comparator, reference + structured."""
    out, seen = [], set()
    for s in list(SS.COMPARATORS) + SS.reference_arm() + SS.structured_arm(len(SS.STRUCTURED_ARM)):
        k = SS.strategy_key(s)
        if k not in seen:
            seen.add(k)
            out.append(s)
    return out


def search_cell(wm, states, cfg=None, log=None):
    """Exploration search for ONE demand cell (SS.evaluate / SS.summarize reused; SS archives + 'novel')."""
    cfg = dict(DEFAULTS, **(cfg or {}))
    rng = random.Random(cfg['seed'])
    lam, groups = float(cfg['lam']), bool(cfg['fairness'])
    sk = dict(fair_tol=float(cfg['fair_tol']), bound_s=float(cfg['bound_s']), bound_stat=cfg['bound_stat'],
              fairer_total=cfg['fairer_total'])
    screen = rng.sample(states, min(len(states), int(cfg['n_screen_states'])))
    pop = list(SS.seeds())          # SS-valid seeds incl. the incumbent (WMVERIFY reserves its name; it is a
    #                                 labelled comparator there); every generated child must pass allowed()
    while len(pop) < len(SS.seeds()) + int(cfg['n_init']):
        s = immigrant(rng, cfg['max_roles'])
        if s is not None and SS.strategy_key(s) not in {SS.strategy_key(p) for p in pop}:
            pop.append(s)
    evaluated, behaviour, total = {}, {}, 0
    archives = {a: {} for a in SS.ARCHIVES}
    novel = {}
    comp_keys = {SS.strategy_key(c) for c in SS.COMPARATORS}
    space = Space()
    extra_refs = [s for s in comparator_refs()]
    state = dict(gscale=None)
    k_nn = int(cfg['novelty_k'])
    gen_stats = []

    def refresh_novelty():
        keys = list(evaluated)
        refs = [(k, evaluated[k]['strategy'], behaviour.get(k)) for k in keys]
        refs += [(SS.strategy_key(s), s, None) for s in extra_refs if SS.strategy_key(s) not in evaluated]
        for k in keys:
            s, b = evaluated[k]['strategy'], behaviour.get(k)
            ds = sorted(combined_distance(space.d(s, rs, k, rk), b, rb) for rk, rs, rb in refs if rk != k)
            evaluated[k]['novelty'] = round(float(np.mean(ds[:k_nn])), 6) if ds else 0.0
            ds_g = sorted(space.d(s, rs, k, rk) for rk, rs, _rb in refs if rk != k)
            evaluated[k]['novelty_genome'] = round(float(np.mean(ds_g[:k_nn])), 6) if ds_g else 0.0
        novel.clear()
        for k in keys:
            r = evaluated[k]
            if not novel_admits(r, cfg):
                continue
            d = tuple(r['descriptor_fine_t'])
            if d not in novel or r['novelty'] > novel[d]['novelty']:
                novel[d] = r

    def run(cands, sts, K):
        nonlocal total
        cands = [c for c in cands if c is not None and SS.strategy_key(c) not in evaluated and not SS.banned(c)]
        if not cands:
            return
        per, n = SS.evaluate(wm, sts, cands, K, cfg['H'], groups=groups)
        total += n
        new = []
        for c in cands:
            k = SS.strategy_key(c)
            gp, nreq = gate_pass(c, sts)
            fd = fine_descriptor(c, gp)
            res = dict(strategy=c, key=k, descriptor=list(SS.descriptor(c)), descriptor_fine=_jsonable(fd),
                       descriptor_fine_t=fd, support=support_tag(c)[0], gate_pass_rate=gp, gate_requests=nreq,
                       **SS.summarize(per[k], lam, **sk))
            evaluated[k] = res
            new.append((c, k, res))
        if state['gscale'] is None:
            gs = [r['gain_vs_base'] for _c, _k, r in new if feasible_ok(r, cfg)]
            state['gscale'] = round(max(1.0, float(np.std(gs))) if len(gs) > 1 else 1.0, 6)
        for c, k, res in new:
            behaviour[k] = behaviour_vector(per[k], members_per_state(c, sts), res, state['gscale'])
            if not feasible_ok(res, cfg):           # min_feasible before ANY archive admission
                continue
            d = SS.descriptor(c)
            for a, arch in archives.items():
                if SS.admits(a, res) and (d not in arch or SS.rank_key(a)(res) < SS.rank_key(a)(arch[d])):
                    arch[d] = res
        refresh_novelty()
        if log:
            log(dict(evaluated=len(evaluated), rollouts=total, archive_novel=len(novel),
                     **{f'archive_{a}': len(v) for a, v in archives.items()}))

    def top(arch, a, n):
        return sorted(arch.values(), key=SS.rank_key(a))[:n]

    def explore_key(r):
        return -(r['score'] + float(cfg['bonus']) * float(r.get('novelty') or 0.0))

    run(pop, screen, cfg['K_screen'])
    for _g in range(int(cfg['generations'])):
        ranked = sorted((r for r in evaluated.values() if feasible_ok(r, cfg)), key=explore_key)
        parents = [r['strategy'] for r in ranked[:int(cfg['n_elite'])]]
        for a in SS.ARCHIVES:
            parents += [r['strategy'] for r in top(archives[a], a, int(cfg['n_elite']) if a != 'fair'
                                                   else int(cfg['n_elite']) // 2)]
        parents += [r['strategy'] for r in sorted(novel.values(), key=lambda r: (-r['novelty'], r['key']))
                    [:2 * int(cfg['n_elite'])]]
        if not parents:
            parents = [r['strategy'] for r in evaluated.values()][:max(2, int(cfg['n_elite']))]
        kids, st = make_children(rng, parents, int(cfg['n_children']), cfg, set(evaluated))
        run(kids, screen, cfg['K_screen'])
        gen_stats.append(dict(st, children=len(kids), parents=len(parents), archive_novel=len(novel)))
    nf = int(cfg['finalists'])
    fin = {}
    for r in sorted((r for r in evaluated.values() if feasible_ok(r, cfg)), key=explore_key)[:nf]:
        fin[r['key']] = r
    for a in ('bounded', 'fairer', 'fair', 'delay'):
        for r in top(archives[a], a, nf):
            if len(fin) < 3 * nf:
                fin.setdefault(r['key'], r)
    for r in sorted(novel.values(), key=lambda r: (-r['novelty'], r['key'])):
        if len(fin) >= 4 * nf:
            break
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
        ev = evaluated.get(k) or {}
        sup, why = support_tag(f)
        gp, nreq = gate_pass(f, fstates)
        final.append(dict(strategy=f, key=k, descriptor=list(SS.descriptor(f)),
                          descriptor_fine=_jsonable(fine_descriptor(f, gp)), comparator=k in comp_keys,
                          banned=SS.banned(f), novelty=ev.get('novelty'), novelty_genome=ev.get('novelty_genome'),
                          screen_feasible=ev.get('feasible'), support=sup, support_reasons=why,
                          gate_pass_rate=gp, gate_requests=nreq, screen_gate_pass_rate=ev.get('gate_pass_rate'),
                          in_novel_archive=any(r['key'] == k for r in novel.values()),
                          **SS.summarize(per[k], lam, **sk)))
    final.sort(key=lambda r: r['score'], reverse=True)
    cands = [r for r in final if not r['comparator'] and not r['banned'] and feasible_ok(r, cfg)]

    def best(a):
        if a == 'novel':
            xs = sorted((r for r in cands if novel_admits(r, cfg) and r['novelty'] is not None),
                        key=lambda r: (-r['novelty'], r['key']))
        else:
            xs = sorted((r for r in cands if SS.admits(a, r)), key=SS.rank_key(a))
        return xs[0] if xs else None
    beats = [r for r in cands if r['gain_vs_inc_lcb'] > 0 and r.get('bounded') is not False]
    fairness = ('per-vehicle' if final and final[0].get('bounded') is not None else
                'groups only' if final and final[0].get('fair') is not None else
                'unavailable (WM has no per-group delay)')

    def listing(a):
        return [dict(descriptor=list(d), key=r['key'], name=r['strategy']['name'], score=r['score'],
                     worst_group_inc=r.get('worst_group_inc'), cvar90_inc=r.get('cvar90_inc'),
                     feasible=r.get('feasible'))
                for d, r in sorted(archives[a].items(), key=lambda kv: SS.rank_key(a)(kv[1]))]

    def listing_novel():
        return [dict(descriptor_fine=_jsonable(d), key=r['key'], name=r['strategy']['name'], novelty=r['novelty'],
                     novelty_genome=r['novelty_genome'], score=r['score'], gain_vs_base=r['gain_vs_base'],
                     worst_group_inc=r.get('worst_group_inc'), feasible=r.get('feasible'), support=r['support'],
                     gate_pass_rate=r.get('gate_pass_rate'), gate_requests=r.get('gate_requests'),
                     strategy=r['strategy'])
                for d, r in sorted(novel.items(), key=lambda kv: (-kv[1]['novelty'], kv[1]['key']))]
    cells = {a: len(v) for a, v in archives.items()}
    cells['novel'] = len(novel)
    arch_out = {a: listing(a) for a in SS.ARCHIVES}
    arch_out['novel'] = listing_novel()
    out = dict(schema='wmc-strategy-search-v3', explore_schema=EXPLORE_SCHEMA, config=cfg, incumbent=SS.INCUMBENT['name'],
               banned_v_w=list(SS.BANNED_V_W), n_states=len(states), n_screen_states=len(screen),
               n_final_states=len(fstates), evaluated=len(evaluated), wm_rollouts=total, fairness=fairness,
               archive_cells=cells, final=final, best={a: best(a) for a in ARCHIVES},
               best_beating_incumbent=beats[0] if beats else None,
               comparators={r['strategy']['name']: {k: r.get(k) for k in (
                   'gain_vs_base', 'gain_vs_base_lcb', 'group_inc', 'worst_group_inc', 'cvar90_inc', 'bounded')}
                   for r in final if r['comparator']},
               archives=arch_out,
               explore=dict(gscale=state['gscale'], generations=gen_stats, support_basis=SUPPORT_BASIS, gate_basis=GATE_BASIS,
                            gate_mode='proxy (observable start-state snapshot)',
                            screened_infeasible=sum(1 for r in evaluated.values() if not feasible_ok(r, cfg)),
                            note='WM = proposal generator only (failed qualification); disagreement not used'))
    out['selection'] = select_diverse_for_sumo(out, int(cfg['select_n']))
    return out


def _jsonable(x):
    if isinstance(x, tuple):
        return [_jsonable(v) for v in x]
    return x


# ------------------------------------------------------------------------------------------ SUMO selection
def select_diverse_for_sumo(result, n=6, delta=None, harm_s=None):
    """Diversity-aware picks from result['final']: allowed (SS.validate + WMVERIFY rules, not banned, not a
    comparator), feasible >= min_feasible, WM worst-group increase <= harm_s (5 s/veh), gain_vs_base >= -novel_eps,
    lane-change plans gate_pass_rate >= min_gate_pass (hold-only plans unaffected),
    genome distance > delta from the incumbent / banned comparator / reference / structured families. In- and
    out-of-support candidates are clustered separately (farthest-point on genome_distance, seeded at the best
    predicted) and one pick (best predicted score) is taken per cluster -> {'wm_novel': [spec], 'tags': [...]}."""
    cfg = dict(DEFAULTS, **(result.get('config') or {}))
    delta = float(cfg['select_delta'] if delta is None else delta)
    harm_s = float(cfg['novel_harm_s'] if harm_s is None else harm_s)
    refs = comparator_refs()
    ref_keys = {SS.strategy_key(s) for s in refs}
    space = Space()
    excluded = dict(comparator=0, banned=0, not_allowed=0, infeasible=0, harm=0, gain=0, lane_gate=0, near_comparator=0)
    tiers = {'in': [], 'out': []}
    seen = set()
    for r in result['final']:
        s = r['strategy']
        k = SS.strategy_key(s)
        if k in seen:
            continue
        seen.add(k)
        if r.get('comparator') or k in ref_keys:
            excluded['comparator'] += 1
            continue
        if r.get('banned') or SS.banned(s):
            excluded['banned'] += 1
            continue
        if not allowed(s):
            excluded['not_allowed'] += 1
            continue
        if not feasible_ok(r, cfg):
            excluded['infeasible'] += 1
            continue
        wg = r.get('worst_group_inc')
        if wg is None or float(wg) > harm_s:
            excluded['harm'] += 1
            continue
        if float(r['gain_vs_base']) < -float(cfg['novel_eps']):
            excluded['gain'] += 1
            continue
        if not gate_ok(r, cfg):
            excluded['lane_gate'] += 1
            continue
        dmin, near = min((space.d(s, c), c['name']) for c in refs)
        if dmin <= delta:
            excluded['near_comparator'] += 1
            continue
        tiers[support_tag(s)[0]].append(dict(r=r, k=k, s=s, dmin=dmin, near=near))
    n = int(n)
    want = {'in': (n + 1) // 2, 'out': n // 2}
    plan = {}
    for t in ('in', 'out'):
        plan[t] = _clusters(tiers[t], want[t], space)
    spare = sum(want[t] - len(plan[t]) for t in plan)
    for t in ('in', 'out'):                 # unused slots go to the other tier (no cross-tier ranking)
        if spare > 0 and len(plan[t]) == want[t]:
            more = _clusters(tiers[t], want[t] + spare, space)
            spare -= len(more) - len(plan[t])
            plan[t] = more
    picks, tags = [], []
    for t in ('in', 'out'):
        for ci, cl in enumerate(plan[t]):
            best = sorted(cl, key=lambda x: (-x['r']['score'], -x['r']['gain_vs_base'], x['k']))[0]
            r = best['r']
            picks.append(best['s'])
            tags.append(dict(key=best['k'], name=best['s']['name'], support=t, support_reasons=support_tag(best['s'])[1],
                             cluster=f'{t}{ci}', cluster_size=len(cl), score=r['score'],
                             gain_vs_base=r['gain_vs_base'], gain_vs_base_lcb=r['gain_vs_base_lcb'],
                             worst_group_inc=r.get('worst_group_inc'), feasible=r.get('feasible'),
                             novelty=r.get('novelty'), gate_pass_rate=r.get('gate_pass_rate'),
                             gate_requests=r.get('gate_requests'), min_dist_to_comparators=best['dmin'],
                             nearest_comparator=best['near']))
    return dict(group='wm_novel', wm_novel=picks, tags=tags, n_requested=n, delta=delta, harm_s=harm_s,
                min_feasible=float(cfg['min_feasible']), min_gate_pass=float(cfg['min_gate_pass']), gate_basis=GATE_BASIS, eligible={t: len(v) for t, v in tiers.items()},
                excluded=excluded, support_basis=SUPPORT_BASIS,
                note='in- and out-of-support picks are clustered and ranked separately; never compare across tiers')


def _clusters(items, k, space):
    if not items or k <= 0:
        return []
    items = sorted(items, key=lambda x: (-x['r']['score'], x['k']))
    centers = [0]
    mind = [space.d(items[0]['s'], x['s'], items[0]['k'], x['k']) for x in items]
    while len(centers) < min(k, len(items)):
        j = max(range(len(items)), key=lambda i: (mind[i], -i))
        if mind[j] <= 0.0:
            break
        centers.append(j)
        mind = [min(mind[i], space.d(items[j]['s'], x['s'], items[j]['k'], x['k'])) for i, x in enumerate(items)]
    groups = [[] for _ in centers]
    for i, x in enumerate(items):
        dists = [(space.d(items[c]['s'], x['s'], items[c]['k'], x['k']), ci) for ci, c in enumerate(centers)]
        groups[min(dists)[1]].append(x)
    return groups


# ------------------------------------------------------------------------------------------ CLI
def main(argv=None):
    """Mirror of `python -m tools.wzprop strategy` (same arguments / backend construction) -> STRATEGY_SEARCH_<cell>.json
    (SS schema + 'novel' archive + 'selection') and ARMS.json (+ 'wm_novel' group and its tags)."""
    from . import __main__ as WZ
    ap = argparse.ArgumentParser(prog='python -m tools.wzprop.strategy_explore')
    ap.add_argument('--states', required=True, help='observable start states (regime states / enriched moments)')
    ap.add_argument('--cells', required=True)
    ap.add_argument('--out', required=True, help='NEW directory')
    ap.add_argument('--wm', default='auto')
    ap.add_argument('--wm-api', action='append')
    ap.add_argument('--wm-arg', action='append')
    ap.add_argument('--config', help='JSON dict of strategy_explore.DEFAULTS overrides')
    ap.add_argument('--k', type=int, default=4)
    ap.add_argument('--mem-fraction', type=float, help='per worker process (with --workers) or this process')
    ap.add_argument('--workers', type=int, default=0, help='parallel WM worker processes (CPU-bound rollouts)')
    a = ap.parse_args(argv)
    out = G.check_path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    if a.workers:
        kw = {}
        for k2, v in WZ._kv(a.wm_arg).items():
            try:
                kw[k2] = json.loads(v)
            except ValueError:
                kw[k2] = v
        local = dict(kw, device='cpu') if 'device' in kw else kw
        wm = W.ParallelWM(a.wm, a.workers, api=WZ._kv(a.wm_api), local_kw=local, mem_fraction=a.mem_fraction, **kw)
    else:
        if a.mem_fraction:
            import torch
            torch.cuda.set_per_process_memory_fraction(float(a.mem_fraction))
        wm = WZ._backend(a)
    cfg = json.loads(a.config) if a.config else None
    states = [m for m in json.loads(G.check_path(a.states).read_text()) if m.get('valid')]
    arms = {}
    for cell in a.cells.split(','):
        st = [m for m in states if m['cell'] == cell]
        t0 = time.time()
        log = (out / f'LOG_{cell}.jsonl').open('x')
        res = search_cell(wm, st, cfg, log=lambda d: (log.write(json.dumps(d) + '\n'), log.flush()))
        log.close()
        res.update(cell=cell, backend=wm.meta, wall_s=round(time.time() - t0, 1), new_sumo_sim_s=0)
        (out / f'STRATEGY_SEARCH_{cell}.json').write_text(json.dumps(res, indent=1, default=str))
        sel = res['selection']
        arms[cell] = dict(SS.wm_arms(res, a.k), wm_novel=sel['wm_novel'], wm_novel_tags=sel['tags'],
                          structured=SS.structured_arm(a.k), random=SS.random_arm(a.k), reference=SS.reference_arm(),
                          incumbent=SS.INCUMBENT, banned_comparator=SS.SOLO2, banned_v_w=list(SS.BANNED_V_W),
                          bound_s=SS.GROUP_BOUND_S,
                          best={k2: (v or {}).get('strategy') for k2, v in res['best'].items()},
                          best_beating_incumbent=(res['best_beating_incumbent'] or {}).get('strategy'),
                          comparators=res['comparators'], fairness=res['fairness'])
        top = res['final'][0]
        print(cell, 'evaluated', res['evaluated'], 'archives', res['archive_cells'], 'wm_novel',
              [(t['name'], t['support'], t['score']) for t in sel['tags']], 'top',
              (top['strategy']['name'], top['gain_vs_base'], top['gain_vs_base_lcb'], top.get('worst_group_inc')))
    (out / 'ARMS.json').write_text(json.dumps(arms, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
