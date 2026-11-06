"""Strategy-level, condition-aware WM search with solo2 as the incumbent to beat (owner direction 2026-11-06 ~20:45).

A STRATEGY is an episode rule: whenever its trigger is feasible (and the cooldown has passed), the CAVs it selects run
their programs; the plant base (MAPPO u0 in SUMO) drives everything else. The incumbent is S-REGIME / S-REGIMEU0
wall_h_solo2 ("solo2"): wall trigger geometry, only the SUMO-lane-2 (left) member holds 15 m/s for 90 s, dmax 1.5,
cooldown 90 s.

Strategy schema (JSON):
  name, trigger 'wall' (pg_plan.pick_wall geometry; roles must be rank 0 in distinct lanes) | 'lanes' (per role the
  rank-th most downstream zone CAV of its SUMO lane; picked CAVs distinct and within spread_m),
  zone [x_lo, x_hi] on 'up' (registered zones), spread_m, cooldown_s, slots (concurrent activations),
  roles [{lane, rank, v_w, dmax, hold_s, onset_s, lc_target}]   (1-6 roles; lc_target = adjacent SUMO lane or None)
Every value is drawn from the registered NEWMAP2 / COOPKINDS / PLANGEN2 / WMDATA ranges (lcm_plan constants).

WM evaluation (imagination): at each TRAIN start state of a demand cell the strategy's activation at t0 (if feasible)
runs for H = 300 s next to the base and the incumbent in ONE CRN-paired WM call; gain_vs_base = J(base) - J(cand),
gain_vs_solo2 = J(solo2 activation or base) - J(cand activation or base). Per cell: mean over states of the WM-draw
mean, a one-sided 95% LCB from a world-cluster bootstrap over states, and the mean draw sd (disagreement).
score = LCB(gain_vs_solo2) - lambda * disagreement. MAP-Elites archive over descriptors (n CAVs, lanes, lane changes,
target-speed bin, stagger, zone, trigger). One activation per state approximates the episode rule (many states per
cell cover the episode); this is a ranking device, SUMO paired seeds decide.
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
SOLO2 = dict(name='solo2', trigger='wall', zone='full', spread_m=200.0, cooldown_s=90, slots=1, gate=None,
             roles=[dict(lane=2, rank=0, v_w=15.0, dmax=1.5, hold_s=90, onset_s=0, lc_target=None)])
DEFAULTS = dict(H=300, K_screen=4, K_final=8, n_init=40, generations=3, n_elite=6, n_children=24, p_mutate=0.35,
                n_screen_states=12, finalists=12, lam=0.5, max_roles=6, seed=20261107, fairness=True, fair_tol=0.0,
                n_final_states=0)


def grids():
    P = FL.lcm()
    return dict(v_w=list(P.R_GRID['v_w']), dmax=list(P.R_GRID['dmax']), hold_s=list(P.R_GRID['hold_s']),
                onset_s=list(P.OFFSETS))


def strategy_key(s):
    body = {k: s[k] for k in ('trigger', 'zone', 'spread_m', 'cooldown_s', 'slots', 'roles')}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]


def validate(s):
    """Registered ranges + well-formed roles (raises ValueError)."""
    g = grids()
    if s['trigger'] not in ('wall', 'lanes') or s['zone'] not in ZONES:
        raise ValueError('trigger/zone')
    if s.get('gate') not in GATES:
        raise ValueError('gate not registered')
    if not 1 <= len(s['roles']) <= 6:
        raise ValueError('1-6 roles')
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


def select_members(s, cavs, observed=None):
    """CAV rows (id, edge, lane, x, v) chosen by the strategy's trigger at this snapshot, in role order, else None.
    observed = every observable row (CAV self + CAV-detected) for the condition gate (default: the CAVs only)."""
    if not gate_open(s, list(cavs) + list(observed or [])):
        return None
    lo, hi = ZONES[s['zone']]
    z = [c for c in cavs if c['edge'] == 'up' and lo <= float(c['x']) <= hi]
    if s['trigger'] == 'wall':
        wall = _pick_wall(z)
        if wall is None:
            return None
        return [wall[r['lane']] for r in s['roles']]
    out = []
    for r in s['roles']:
        lane = sorted((c for c in z if int(c['lane']) == r['lane']), key=lambda c: (-float(c['x']), c['id']))
        if len(lane) <= r['rank']:
            return None
        out.append(lane[r['rank']])
    if len({c['id'] for c in out}) != len(out):
        return None
    xs = [float(c['x']) for c in out]
    return out if max(xs) - min(xs) <= float(s['spread_m']) else None


def member_specs(s, members):
    """WM / SUMO member specs: all nm2 'hold' when no role changes lanes, else all LCMOVES 'prog'."""
    if all(r['lc_target'] is None for r in s['roles']):
        out = []
        for r, c in zip(s['roles'], members):
            prm = dict(v_w=float(r['v_w']), hold_s=int(r['hold_s']), dmax=float(r['dmax']))
            if r['onset_s']:
                prm['start_s'] = int(r['onset_s'])
            out.append(F.member_spec(c, prm))
        return out
    P = FL.lcm()
    out = []
    for r, c in zip(s['roles'], members):
        out.append(P.prog(c, lc_target=r['lc_target'], lc_from=int(r['onset_s']), lc_giveup_x=P.X_B - P.GIVEUP_OFF,
                          speed='hold', v_w=float(r['v_w']), dmax=float(r['dmax']), sp_from=int(r['onset_s']),
                          hold_s=int(r['onset_s']) + int(r['hold_s'])))
    return out


def activation(s, cavs, observed=None):
    mem = select_members(s, cavs, observed)
    return None if mem is None else member_specs(s, mem)


def descriptor(s):
    v = float(np.mean([r['v_w'] for r in s['roles']]))
    on = [r['onset_s'] for r in s['roles']]
    return (len(s['roles']), len({r['lane'] if r['lc_target'] is None else r['lc_target'] for r in s['roles']}),
            sum(r['lc_target'] is not None for r in s['roles']), int(v >= 17.0) + int(v >= 20.0),
            int(max(on) > min(on)), s['zone'], s['trigger'], s.get('gate') is not None)


# ------------------------------------------------------------------------------------------ sampling
def random_strategy(rng, max_roles=6):
    g = grids()
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
             slots=1, roles=roles, gate=rng.choice(GATES) if rng.random() < 0.3 else None)
    s['cooldown_s'] = int(max(r['onset_s'] + r['hold_s'] for r in roles))
    s['name'] = 'rand_' + strategy_key(s)
    return validate(s)


def seeds():
    """Known strategies: the incumbent, S-REGIME wall/solo rules, solo2 variants and lane-2-hold combinations."""
    out = [dict(SOLO2)]
    for lane in (0, 1):
        out.append(dict(SOLO2, name=f'solo{lane}', roles=[dict(SOLO2['roles'][0], lane=lane)]))
    out.append(dict(SOLO2, name='wall_h', roles=[dict(SOLO2['roles'][0], lane=ln) for ln in (0, 1, 2)]))
    for v, h in ((18.0, 90), (15.0, 120), (22.0, 120)):
        out.append(dict(SOLO2, name=f'solo2_v{v:g}_h{h}', roles=[dict(SOLO2['roles'][0], v_w=v, hold_s=h)],
                        cooldown_s=h))
    l2 = dict(lane=2, rank=0, v_w=15.0, dmax=1.5, hold_s=90, onset_s=0, lc_target=None)
    out.append(dict(name='solo2_lanes', trigger='lanes', zone='full', spread_m=300.0, cooldown_s=90, slots=1,
                    roles=[dict(l2)]))
    out.append(dict(name='l2hold_l1gap', trigger='lanes', zone='full', spread_m=300.0, cooldown_s=90, slots=1,
                    roles=[dict(l2), dict(lane=1, rank=0, v_w=18.0, dmax=1.0, hold_s=90, onset_s=0, lc_target=None)]))
    out.append(dict(name='l2hold_fill2', trigger='lanes', zone='full', spread_m=300.0, cooldown_s=90, slots=1,
                    roles=[dict(l2), dict(lane=1, rank=1, v_w=15.0, dmax=1.5, hold_s=90, onset_s=20, lc_target=2)]))
    out.append(dict(name='l2hold_stagger', trigger='lanes', zone='full', spread_m=300.0, cooldown_s=120, slots=1,
                    roles=[dict(l2), dict(l2, rank=1, onset_s=40)]))
    for theta in (10.0, 15.0):
        out.append(dict(SOLO2, name=f'solo2_gate{theta:g}', gate=theta))
    out = [dict(s, gate=s.get('gate')) for s in out]
    return [validate(s) for s in out]


def mutate(rng, s, p):
    g = grids()
    roles = []
    for r in s['roles']:
        r = dict(r)
        for k in ('v_w', 'dmax', 'hold_s', 'onset_s'):
            if rng.random() < p:
                r[k] = rng.choice(g[k])
        if rng.random() < p / 2:
            r['lc_target'] = rng.choice([None] + [x for x in (r['lane'] - 1, r['lane'] + 1) if 0 <= x <= 2])
        roles.append(r)
    t = dict(s, roles=roles)
    if rng.random() < p / 2:
        t['zone'] = rng.choice(list(ZONES))
    if rng.random() < p / 2:
        t['gate'] = rng.choice(GATES)
    if rng.random() < p / 3 and t['trigger'] == 'lanes' and len(roles) < 6:
        free = [(ln, rk) for ln in (0, 1, 2) for rk in (0, 1) if (ln, rk) not in {(x['lane'], x['rank']) for x in roles}]
        if free:
            ln, rk = rng.choice(free)
            roles.append(dict(lane=ln, rank=rk, v_w=rng.choice(g['v_w']), dmax=rng.choice(g['dmax']),
                              hold_s=rng.choice(g['hold_s']), onset_s=rng.choice(g['onset_s']), lc_target=None))
    t['cooldown_s'] = int(max(max(r['onset_s'] + r['hold_s'] for r in roles), s['cooldown_s'] // 2))
    t['name'] = 'mut_' + strategy_key(t)
    try:
        return validate(t)
    except ValueError:
        return None


# ------------------------------------------------------------------------------------------ WM evaluation
def evaluate(wm, states, strategies, K, H, z=1.645, groups=False):
    """Per strategy per state the CRN-paired WM draws -> gains vs base and vs solo2 (see module docstring).
    groups=True (and a backend with evaluate_groups) adds per-group gains vs base for the registered fairness
    groups (SUMO origin lane 0/1/2, HDV / CAV); otherwise fairness is reported as unavailable."""
    per = defaultdict(list)
    rollouts = 0
    prepared = []
    for m in states:
        mi = W.moment_input(m)
        cavs = m['all_cavs_t0']
        observed = list((m.get('observed') or {}).get('vehicles') or [])
        state = wm.observe(mi)
        acts = [W.action('base', 'base', None, 'base', [], H)]
        inc = activation(SOLO2, cavs, observed)
        if inc:
            acts.append(W.action('inc', 'group', 'solo2', 'solo2', inc, H))
        aid_of, specs_seen = {}, {}
        if inc:                                  # a candidate whose activation IS the incumbent's reuses its row
            specs_seen[json.dumps([[x['id'], x['role'], x['params']] for x in inc], sort_keys=True)] = 'inc'
        for s in strategies:
            spec = activation(s, cavs, observed)
            if spec is None:
                continue
            sk = json.dumps([[x['id'], x['role'], x['params']] for x in spec], sort_keys=True)
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
            gi = ref - j
            row = dict(world=int(m['seed']), gain_base=float((base - j).mean()), gain_inc=float(gi.mean()),
                       sd=float(gi.std(ddof=1)) if K > 1 else 0.0, feasible=aid is not None)
            if use_groups:
                gb = graws[i]['base']
                gc = graws[i][aid] if aid else gb
                row['groups'] = {g: float(np.mean(np.asarray(gb[g]) - np.asarray(gc[g]))) for g in W.FAIR_GROUPS}
            per[key].append(row)
    return per, rollouts


def _boot_lcb(by_world, field, b, rng, group=None):
    keys = sorted(by_world)
    vals = []
    for _ in range(b):
        pick = rng.integers(0, len(keys), len(keys))
        xs = [(r['groups'][group] if group else r[field]) for j in pick for r in by_world[keys[j]]]
        vals.append(float(np.mean(xs)))
    return float(np.quantile(vals, 0.05))


def summarize(rows, lam, b=1000, seed=20261107, fair_tol=0.0):
    """Per-cell expected gains (vs base, vs solo2) with world-cluster LCBs; per-group gains vs base and the FAIR flag
    (registered: lower total delay vs the base AND no origin-lane group / class predicted worse: every group's
    gain LCB >= -fair_tol)."""
    worlds = defaultdict(list)
    for r in rows:
        worlds[r['world']].append(r)
    rng = np.random.default_rng(seed)
    mean_b = float(np.mean([r['gain_base'] for r in rows]))
    mean_i = float(np.mean([r['gain_inc'] for r in rows]))
    dis = float(np.mean([r['sd'] for r in rows]))
    lcb_b = _boot_lcb(worlds, 'gain_base', b, rng)
    lcb_i = _boot_lcb(worlds, 'gain_inc', b, rng)
    out = dict(n_states=len(rows), feasible=float(np.mean([r['feasible'] for r in rows])), gain_vs_base=round(mean_b, 2),
               gain_vs_base_lcb=round(lcb_b, 2), gain_vs_solo2=round(mean_i, 2), gain_vs_solo2_lcb=round(lcb_i, 2),
               disagreement=round(dis, 2), score=round(lcb_i - lam * dis / math.sqrt(max(1, len(rows))), 2))
    if rows and all('groups' in r for r in rows):
        gm = {g: round(float(np.mean([r['groups'][g] for r in rows])), 2) for g in W.FAIR_GROUPS}
        gl = {g: round(_boot_lcb(worlds, None, b, rng, group=g), 2) for g in W.FAIR_GROUPS}
        worst = min(gl, key=gl.get)
        out.update(group_gain_vs_base=gm, group_gain_vs_base_lcb=gl, worst_group=worst,
                   worst_group_lcb=gl[worst], fair=bool(lcb_b > 0 and all(v >= -fair_tol for v in gl.values())))
    else:
        out.update(fair=None)
    return out


def search_cell(wm, states, cfg=None, log=None):
    """MAP-Elites + evolution over strategies for ONE demand cell's start states. Two archives: delay (score) and
    FAIR (score among fair candidates); both feed the finals."""
    cfg = dict(DEFAULTS, **(cfg or {}))
    rng = random.Random(cfg['seed'])
    lam, tol, groups = float(cfg['lam']), float(cfg['fair_tol']), bool(cfg['fairness'])
    screen = rng.sample(states, min(len(states), int(cfg['n_screen_states'])))
    pop = seeds() + [random_strategy(rng, cfg['max_roles']) for _ in range(int(cfg['n_init']))]
    evaluated, archive, fair_archive, total = {}, {}, {}, 0

    def run(cands, sts, K):
        nonlocal total
        cands = [c for c in cands if c is not None and strategy_key(c) not in evaluated]
        if not cands:
            return
        per, n = evaluate(wm, sts, cands, K, cfg['H'], groups=groups)
        total += n
        for c in cands:
            k = strategy_key(c)
            res = dict(strategy=c, key=k, descriptor=list(descriptor(c)), **summarize(per[k], lam, fair_tol=tol))
            evaluated[k] = res
            d = descriptor(c)
            if d not in archive or res['score'] > archive[d]['score']:
                archive[d] = res
            if res.get('fair') and (d not in fair_archive or res['score'] > fair_archive[d]['score']):
                fair_archive[d] = res
        if log:
            log(dict(evaluated=len(evaluated), archive=len(archive), fair_archive=len(fair_archive), rollouts=total))

    def top(arch, n):
        return sorted(arch.values(), key=lambda r: r['score'], reverse=True)[:n]

    run(pop, screen, cfg['K_screen'])
    for _ in range(int(cfg['generations'])):
        ranked = sorted(evaluated.values(), key=lambda r: r['score'], reverse=True)
        parents = [r['strategy'] for r in ranked[:int(cfg['n_elite'])] + top(archive, int(cfg['n_elite']))
                   + top(fair_archive, int(cfg['n_elite']) // 2)]
        kids = [mutate(rng, rng.choice(parents), cfg['p_mutate']) for _j in range(int(cfg['n_children']))]
        run(kids, screen, cfg['K_screen'])
    fin = sorted(evaluated.values(), key=lambda r: r['score'], reverse=True)[:int(cfg['finalists'])]
    for r in top(archive, 10 ** 6) + top(fair_archive, 10 ** 6):
        if r['key'] not in {f['key'] for f in fin} and len(fin) < 2 * int(cfg['finalists']):
            fin.append(r)
    finals = [dict(f['strategy']) for f in fin]
    if not any(strategy_key(f) == strategy_key(SOLO2) for f in finals):
        finals.append(dict(SOLO2))
    fstates = states
    if int(cfg['n_final_states']) and len(states) > int(cfg['n_final_states']):
        fstates = random.Random(cfg['seed'] + 1).sample(states, int(cfg['n_final_states']))
    per, n = evaluate(wm, fstates, finals, cfg['K_final'], cfg['H'], groups=groups)
    total += n
    final = []
    for f in finals:
        k = strategy_key(f)
        final.append(dict(strategy=f, key=k, descriptor=list(descriptor(f)), **summarize(per[k], lam, fair_tol=tol)))
    final.sort(key=lambda r: r['score'], reverse=True)
    inc = strategy_key(SOLO2)
    beats = [r for r in final if r['gain_vs_solo2_lcb'] > 0 and r['key'] != inc]
    fair = [r for r in final if r.get('fair') and r['key'] != inc]
    fairness = 'available' if final and final[0].get('fair') is not None else 'unavailable (WM has no per-group delay)'
    return dict(schema='wmc-strategy-search-v2', config=cfg, n_states=len(states), n_screen_states=len(screen),
                n_final_states=len(fstates),
                evaluated=len(evaluated), archive_cells=len(archive), fair_archive_cells=len(fair_archive),
                wm_rollouts=total, fairness=fairness, final=final,
                best_beating_solo2=beats[0] if beats else None, best_fair=fair[0] if fair else None,
                best_fair_beating_solo2=next((r for r in fair if r['gain_vs_solo2_lcb'] > 0), None),
                archive=[dict(descriptor=list(d), key=r['key'], score=r['score'], name=r['strategy']['name'])
                         for d, r in sorted(archive.items(), key=lambda kv: -kv[1]['score'])],
                fair_archive=[dict(descriptor=list(d), key=r['key'], score=r['score'], name=r['strategy']['name'],
                                   worst_group=r.get('worst_group'), worst_group_lcb=r.get('worst_group_lcb'))
                              for d, r in sorted(fair_archive.items(), key=lambda kv: -kv[1]['score'])])


def select_for_sumo(result, k=4, fair=False):
    """WM arm: top LCB strategies from distinct descriptor cells, the incumbent excluded (fair=True: fair only)."""
    out, cells = [], set()
    for r in result['final']:
        if fair and not r.get('fair'):
            continue
        if r['key'] == strategy_key(SOLO2) or tuple(r['descriptor']) in cells:
            continue
        out.append(r['strategy'])
        cells.add(tuple(r['descriptor']))
        if len(out) == k:
            break
    return out


STRUCTURED_ARM = ('solo2_v18_h90', 'solo2_v15_h120', 'l2hold_l1gap', 'l2hold_stagger')


def structured_arm(k=4):
    """Structured arm (no WM): a fixed, pre-registered local neighbourhood of the incumbent (speed, duration,
    a middle-lane gap maker, a staggered second lane-2 hold)."""
    by = {s['name']: s for s in seeds()}
    return [by[n] for n in STRUCTURED_ARM[:k]]


def random_arm(k=4, seed=20261107):
    rng = random.Random(seed)
    return [random_strategy(rng) for _ in range(k)]
