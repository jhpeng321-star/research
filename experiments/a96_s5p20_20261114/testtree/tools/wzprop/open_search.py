"""Open (roaming) WM search mode: broad free-program initialisation, evolutionary / cross-entropy refinement in
imagination, a MAP-Elites diversity archive, a disagreement-penalised lower-confidence score, and a verification
slot split (top / diverse / uncertain). Design note: OPEN_SEARCH.md. Zero SUMO; the WM is any wm_backend backend.

Program space (every range = an lcm_plan constant from the NEWMAP2 / COOPKINDS / PLANGEN2 / WMDATA registrations):
  group      2..6 CAVs of one V2V-local group (anchor + CAVs chained within 100 m), zone [X_B-1800, X_B-400] on 'up'
  per member optional lane change to an adjacent lane (closing lane included) at offset lc_from in OFFSETS
             speed program: hold toward v_w in R_GRID v_w (accelerate or slow), or accelerate at amax ('accel')
             dmax in R_GRID dmax, duration in R_GRID hold_s, start sp_from in OFFSETS, own time gap tau in TAU_SET
  encoding   a member WITHOUT lane change / tau / accel becomes an nm2 'hold' member (start_s, hold_s, v_w, dmax),
             otherwise an LCMOVES 'prog' member; a variant is all-'hold' or all-'prog' (single SUMO executor);
             programs needing 'prog' are only searched with a WM whose supported_roles include 'prog'.
Seeds: the known families (structured walls/pairs, LCMOVES builders) are injected into the initial population.
Descriptors (MAP-Elites cells): n_cavs, lanes used after the programs, n lane changes, mean speed change (v_w - v at
t0, binned), onset relative to the drop (group centroid distance to X_B, binned), spread (binned). The archive keeps
the best-scoring plan per cell; elites are drawn from both the global ranking and the archive.
Score: LCB(M) - lambda * sd(M) over CRN-paired WM draws (M = min(base, best solo) - group); the sd term penalises
WM disagreement so the optimiser does not exploit WM errors.
"""
from __future__ import annotations

import hashlib
import json
import math
import random

import numpy as np

from . import families as F
from . import families_lc as FL
from . import geometry as G
from . import proposer as PR
from . import wm_backend as W

DEFAULTS = dict(k=5, H=300, K_screen=4, K_final=8, n_init=48, generations=3, n_elite=8, n_children=24,
                p_mutate=0.35, cem_smoothing=0.5, finalists=16, z=1.645, lam=0.5, share_top=0.5, share_diverse=0.3,
                max_members=6, chain_m=100.0, seed_families=True)
DV_BINS = (-6.0, -3.0, 0.0, 3.0)
DROP_BINS = (600.0, 1200.0)
SPREAD_BINS = (100.0, 200.0)


def config(overrides=None):
    cfg = dict(DEFAULTS)
    for k, v in (overrides or {}).items():
        if k not in DEFAULTS:
            raise KeyError(f'unknown open-search config key {k!r}')
        cfg[k] = v
    return cfg


def _bin(x, edges):
    return int(sum(x >= e for e in edges))


# ------------------------------------------------------------------------------------------ programs
def local_groups(cavs, chain_m=100.0, max_members=6):
    """V2V-local candidate groups: each zone CAV with its nearest zone CAVs chained within chain_m (x and lane)."""
    z = sorted(G.zone_cavs(cavs), key=lambda c: c['id'])
    out = []
    for a in z:
        near = sorted((c for c in z if c['id'] != a['id']), key=lambda c: (abs(c['x'] - a['x']), c['id']))
        grp = [a]
        for c in near:
            if len(grp) >= max_members:
                break
            if min(abs(c['x'] - g['x']) for g in grp) <= chain_m:
                grp.append(c)
        if len(grp) >= 2:
            out.append(grp)
    return out


def member_program(rng, c, P):
    tl = rng.choice([None] + [ln for ln in (int(c['lane']) - 1, int(c['lane']) + 1) if 0 <= ln <= 2])
    speed = rng.choice(['hold', 'hold', 'hold', 'accel'])
    return dict(lc_target=tl, lc_from=rng.choice(P.OFFSETS), speed=speed, v_w=rng.choice(P.R_GRID['v_w']),
                dmax=rng.choice(P.R_GRID['dmax']), dur=rng.choice(P.R_GRID['hold_s']), sp_from=rng.choice(P.OFFSETS),
                tau=rng.choice(P.TAU_SET))


def is_hold_program(g):
    return g['lc_target'] is None and g['tau'] is None and g['speed'] == 'hold'


def encode(group, genes, P):
    """Member specs of one program (all 'hold' when every gene is hold-expressible, else all 'prog')."""
    if all(is_hold_program(g) for g in genes):
        specs = []
        for c, g in zip(group, genes):
            prm = dict(v_w=float(g['v_w']), hold_s=int(g['dur']), dmax=float(g['dmax']))
            if int(g['sp_from']):
                prm['start_s'] = int(g['sp_from'])
            specs.append(F.member_spec(c, prm))
        return specs, 'hold'
    specs = []
    for c, g in zip(group, genes):
        kw = dict(lc_target=g['lc_target'], lc_from=int(g['lc_from']), lc_giveup_x=P.X_B - P.GIVEUP_OFF,
                  speed=g['speed'], dmax=float(g['dmax']), sp_from=int(g['sp_from']),
                  hold_s=int(g['sp_from']) + int(g['dur']), tau=g['tau'], tau_from=int(g['sp_from']))
        if g['speed'] == 'hold':
            kw['v_w'] = float(g['v_w'])
        specs.append(P.prog(c, **kw))
    return specs, 'prog'


def program_move(group, genes, P, origin):
    specs, role = encode(group, genes, P)
    blob = json.dumps([[s['id'], s['params']] for s in specs], sort_keys=True)
    ids = [c['id'] for c in group]
    return dict(key=f"free_program|{','.join(sorted(ids))}|{hashlib.sha256(blob.encode()).hexdigest()[:16]}",
                set_key=f"free_program|{','.join(sorted(ids))}", family='free_program', role=role,
                knob=dict(origin=origin), params={}, member_ids=ids, members=specs, anchor=None, origin=origin,
                genes=[dict(g) for g in genes], group=[dict(c) for c in group])


def descriptor(move, cavs_by_id):
    """MAP-Elites cell of a move (hold or prog members)."""
    lanes, n_lc, dv = set(), 0, []
    for mm in move['members']:
        l0, l1 = FL.descriptor_lanes(mm)
        lanes.add(l1)
        n_lc += int(l0 != l1)
        p = mm['params']
        v0 = float(cavs_by_id.get(mm['id'], mm).get('v', mm.get('v', 0.0)))
        target = G.SPEED_LIMIT if p.get('speed') == 'accel' else float(p.get('v_w', v0))
        dv.append(target - v0)
    xs = [float(mm['x']) for mm in move['members']]
    return (len(move['members']), len(lanes), min(n_lc, 3), _bin(float(np.mean(dv)), DV_BINS),
            _bin(G.X_B - float(np.mean(xs)), DROP_BINS), _bin(max(xs) - min(xs), SPREAD_BINS))


def mutate(rng, move, P, p_mut):
    genes = []
    for g, c in zip(move['genes'], move['group']):
        g = dict(g)
        if rng.random() < p_mut:
            fresh = member_program(rng, c, P)
            key = rng.choice(list(fresh))
            g[key] = fresh[key]
        genes.append(g)
    return program_move(move['group'], genes, P, 'mutation')


def crossover(rng, a, b, P):
    if [c['id'] for c in a['group']] != [c['id'] for c in b['group']]:
        return mutate(rng, a, P, 1.0)
    genes = [dict(rng.choice([ga, gb])) for ga, gb in zip(a['genes'], b['genes'])]
    return program_move(a['group'], genes, P, 'crossover')


def cem_sample(rng, elites, group, P, smoothing):
    """Per-gene categorical distribution fitted on the elites of this group (smoothed toward uniform)."""
    genes = []
    for i, c in enumerate(group):
        fresh = member_program(rng, c, P)
        g = {}
        for key in fresh:
            vals = [e['genes'][i][key] for e in elites if len(e['genes']) > i]
            g[key] = rng.choice(vals) if vals and rng.random() > smoothing else fresh[key]
        genes.append(g)
    return program_move(group, genes, P, 'cem')


def score(rec, z, lam):
    st = rec.get('M') or rec['T']
    k = max(1, int(rec['K']))
    return float(st['mean']) - z * float(st['sd']) / math.sqrt(k) - lam * float(st['sd'])


# ------------------------------------------------------------------------------------------ search
def search(wm, m, cfg, *, state_file=None):
    """One moment: returns (row, archive, evaluated records)."""
    P = FL.lcm()
    G.check_seed(m['seed'], where=f"(moment {m.get('sid')})")
    mi = W.moment_input(m, state_file)
    state = wm.observe(mi)
    cavs = mi['cavs']
    by = {c['id']: c for c in m['all_cavs_t0']}
    roles = frozenset(getattr(wm, 'supported_roles', {'hold'}))
    seed = mi['crn_seed']
    rng = random.Random(seed + 7)
    z, lam, H = float(cfg['z']), float(cfg['lam']), int(cfg['H'])
    groups = local_groups(m['all_cavs_t0'], cfg['chain_m'], cfg['max_members'])
    ledger = dict(calls=0, rollouts=0)

    def allowed(mv):
        return mv.get('role', 'hold') in roles

    pop = []
    for i in range(int(cfg['n_init'])):
        if not groups:
            break
        grp = rng.choice(groups)
        k = rng.randint(2, len(grp))
        sub = grp[:k]
        mv = program_move(sub, [member_program(rng, c, P) for c in sub], P, 'init')
        if allowed(mv):
            pop.append(mv)
    if cfg['seed_families']:
        hold = F.generate(m['all_cavs_t0'], None, F.BUILTIN_FAMILIES, 2)
        pop += rng.sample(hold, min(len(hold), 12))
        if 'prog' in roles:
            lc = FL.lc_moves(m, max_per_family=1, variations=False, n_free=0)
            pop += lc
    seen, evaluated, archive = set(), {}, {}

    def run(moves, K):
        moves = [mv for mv in moves if mv['key'] not in seen and allowed(mv)]
        for mv in moves:
            seen.add(mv['key'])
        if not moves:
            return []
        recs = PR.evaluate_batch(wm, state, moves, K, H, seed, cavs, z, ledger)
        for r in recs:
            r['score'] = score(r, z, lam)
            r['cell'] = descriptor(r['move'], by)
            evaluated[r['move']['key']] = r
            cur = archive.get(r['cell'])
            if cur is None or r['score'] > cur['score']:
                archive[r['cell']] = r
        return recs

    run(pop, cfg['K_screen'])
    for _ in range(int(cfg['generations'])):
        ranked = sorted(evaluated.values(), key=lambda r: r['score'], reverse=True)
        elites = ranked[:int(cfg['n_elite'])]
        arch = sorted(archive.values(), key=lambda r: r['score'], reverse=True)[:int(cfg['n_elite'])]
        parents = [r['move'] for r in elites + arch if r['move'].get('genes')]
        if not parents:
            break
        children = []
        for j in range(int(cfg['n_children'])):
            a = rng.choice(parents)
            mode = j % 3
            if mode == 0:
                children.append(mutate(rng, a, P, cfg['p_mutate']))
            elif mode == 1:
                children.append(crossover(rng, a, rng.choice(parents), P))
            else:
                same = [r['move'] for r in elites if r['move'].get('group') and
                        [c['id'] for c in r['move']['group']] == [c['id'] for c in a['group']]]
                children.append(cem_sample(rng, same or [a], a['group'], P, cfg['cem_smoothing']))
        run(children, cfg['K_screen'])
    fin_keys = []
    for r in sorted(evaluated.values(), key=lambda r: r['score'], reverse=True)[:int(cfg['finalists'])]:
        fin_keys.append(r['move']['key'])
    for r in sorted(archive.values(), key=lambda r: r['score'], reverse=True):
        if r['move']['key'] not in fin_keys and len(fin_keys) < 2 * int(cfg['finalists']):
            fin_keys.append(r['move']['key'])
    final = PR.evaluate_batch(wm, state, [evaluated[k]['move'] for k in fin_keys], cfg['K_final'], H, seed, cavs, z,
                              ledger)
    for r in final:
        r['score'] = score(r, z, lam)
        r['cell'] = descriptor(r['move'], by)
    picks = select(final, int(cfg['k']), cfg['share_top'], cfg['share_diverse'])
    variants = {}
    for i, (slot, r) in enumerate(picks):
        mv = r['move']
        variants[f'os{i}'] = dict(family=mv['family'], params=dict(mv.get('params') or {}),
                                  members=[dict(mm) for mm in mv['members']],
                                  wm=dict(PR.pred(r, i, dict(score='lcb', H=H), wm.name), open_search=dict(
                                      slot=slot, score=round(r['score'], 3), cell=list(r['cell']),
                                      origin=mv.get('origin'), role=mv.get('role', 'hold'))))
    row = dict(schema='wmc-open-search-v1', sid=m['sid'], t0=int(m['t0']), backend=wm.meta, config=cfg,
               supported_roles=sorted(roles), n_groups=len(groups), n_evaluated=len(evaluated),
               archive_cells=len(archive), ledger=ledger, variants=variants,
               archive=[dict(cell=list(c), score=round(r['score'], 3), key=r['move']['key'],
                             origin=r['move'].get('origin')) for c, r in sorted(archive.items())])
    return row, archive, evaluated


def select(final, k, share_top, share_diverse):
    """Verification slots: top score, then best of not-yet-covered descriptor cells, then the most uncertain
    (largest WM sd among candidates with positive mean M) - active learning."""
    ranked = sorted(final, key=lambda r: r['score'], reverse=True)
    n_top = max(1, int(round(k * share_top))) if k else 0
    n_div = int(round(k * share_diverse))
    picks, used, cells = [], set(), set()
    for r in ranked[:n_top]:
        picks.append(('top', r))
        used.add(r['move']['key'])
        cells.add(r['cell'])
    for r in ranked:
        if len(picks) >= n_top + n_div:
            break
        if r['move']['key'] not in used and r['cell'] not in cells:
            picks.append(('diverse', r))
            used.add(r['move']['key'])
            cells.add(r['cell'])
    unc = sorted((r for r in final if r['move']['key'] not in used and (r.get('M') or r['T'])['mean'] > 0),
                 key=lambda r: -(r.get('M') or r['T'])['sd'])
    for r in unc:
        if len(picks) >= k:
            break
        picks.append(('uncertain', r))
        used.add(r['move']['key'])
    for r in ranked:                                     # fill (never fewer than k when candidates exist)
        if len(picks) >= k:
            break
        if r['move']['key'] not in used:
            picks.append(('top', r))
            used.add(r['move']['key'])
    return picks


def feedback_rows(summary, moments):
    """SUMO verdicts -> WM training/feedback rows (state reference + action + measured per-replicate outcomes)."""
    ms = {m['sid']: m for m in moments}
    out = []
    for vd in summary['verdicts']:
        m = ms.get(vd['sid'])
        if m is None or vd.get('T_r0') is None:
            continue
        G.check_seed(vd['seed'])
        v = m['variants'].get(vd['variant']) or {}
        out.append(dict(schema='wmc-wm-feedback-v1', sid=vd['sid'], world_seed=int(vd['seed']), t0=int(vd['t0']),
                        state_ref=(m.get('observed') or {}).get('tape'), action=dict(family=vd['family'],
                                                                                      members=v.get('members')),
                        measured=dict(T_r0=vd.get('T_r0'), S_r0=vd.get('S_r0'), reps=vd.get('reps'),
                                      admitted=bool(vd['admitted']), stage=vd['stage']),
                        wm_prediction=vd.get('wm_prediction'),
                        sign_convention='T = J(base) - J(group), S = J(best solo) - J(group); positive = better'))
    return out
