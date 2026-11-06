"""Upper-bound study (owner phase ``upper_bound``, own ledger, hard cap 150,000 sim-s), 2026-11-06.

Question: how much GROUP cooperation is possible on the current map if a CENTRALIZED planner sees
everything? Situations are registered merge-pressure moments in registered TRAIN worlds (986000-986071,
arm B = frozen MAPPO u0 base, clean3 plant settings), chosen by a config/time rule from observable
state only (never by outcome). For each situation a centralized full-information search runs entirely
in SUMO over moving walls of 2-6 mainline CAVs inside a registered radius (any CAV, ignoring each CAV's
sensing slots) with onset / hold / depth / gradual-release options: coarse screen, then refinement of
the best few, on the SEARCH seed (seed 1 = collection CRN). The chosen move is then evaluated on
held-out CRN replicate seeds (rb.replicate salts, as coop_probe). Synergy = J(best single-member solo)
- J(group) on the common window; per-seed 30-s circular-block paired bootstrap (bank_statistics._stats,
same statistic as coop_probe / mining per_seed_v1); admitted iff one-sided 95% lower bound > 0.

Reuses the coop_probe (c4/coop-probe 5e6eb9e4) machinery unchanged: collection capture, sv.capture
snapshots, sv.build_branches / sv.run_branch / SubprocessBranchRunner (prefix replay + CRN), mine_verify._pol
policy reference, rb.solo_candidate / rb.replicate, bank_statistics._stats. Only the orchestration differs:
one fixed window per situation [t0, t0+180] for every branch (so the policy branch of a seed is run once
and shared by every stage), and a precharged own ledger. Never replays a charged claim. HDVs are never
commanded (plans address CAV members only).
"""
import json
import os
from pathlib import Path
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(os.environ.get('UB_ROOT', '/home/ubuntu/upperbound20261106'))
CAP = 150_000
PHASE = 'upper_bound'
EXPERIMENT = 'S-UPPERBOUND-20261106'
EVAL = ((962000, 962999), (964000, 964199), (977000, 977999))
D_MAX = 60                      # longest plan (onset+hold+release)
TAIL = 120                      # coop_probe tail
SPAN = D_MAX + TAIL             # every branch of a situation: window [t0, t0+SPAN]
T_LO, T_HI, T_STEP = 300, 420, 5
PROFILE_S = D_MAX + 5
COLLECT_STOP = T_HI + PROFILE_S + 5      # 490: collection needs t0 + profile only
S_LO, S_HI = -400.0, 75.44000244140625   # registered radius: mainline CAVs from 400 m upstream to merge end
SENSOR_RANGE_M = 100.0
WORKERS = int(os.environ.get('UB_WORKERS', '12'))
HELDOUT = (2, 3)
EXTRA = (4, 5)
RESERVE = 3000
MAX_COLLECT = 20
NOISE_OFFSET = 9000              # collection noise seed = 20261106 + world seed + 9000 (arm B)
T0_LAW = (0, 30, 1.0, 0)         # (onset, hold, depth m/s^2, release s): the coop_probe lead-wall law
TIMINGS = [(0, 30, 0.5, 0), (0, 20, 2.0, 0), (10, 30, 1.0, 0), (0, 30, 1.0, 20)]
FILL = [(0, 45, 1.0, 0), (0, 30, 1.5, 0), (5, 30, 1.0, 10), (0, 15, 1.0, 0)]
N_SCREEN, N_REFINE_EACH, N_REFINE_TOP = 8, 2, 2
SELECTION = ('decision time t0 = the 5-s tick in [300,420] maximizing the full-information state score '
             '(min(#eligible mainline CAVs,6), #vehicles on the ramp merge lane in s[-100,75.44], '
             'sum 1/(1+v) over them, #mainline vehicles in s[-100,75.44]) among ticks with >=1 ramp vehicle '
             'in the merge zone (merge pressure) and >=2 eligible mainline CAVs (lane 0/1, s in [-400,75.44], '
             'policy-controlled at t0); ties -> earliest. State at t0 only, never an outcome.')

_lock = threading.Lock()


def _write(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, sort_keys=True, default=str))
    os.replace(tmp, p)


def _append(p, obj):
    with _lock:
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        with open(p, 'a') as f:
            f.write(json.dumps(obj, sort_keys=True, default=str)+'\n')
            f.flush(); os.fsync(f.fileno())


def log(msg, **kw):
    _append(ROOT/'RUN_LOG.jsonl', dict(unix=time.time(), msg=msg, **kw))


# ------------------------------------------------------------------ registration / ledger

def registered_worlds():
    from tools.coopmwm.protocol import worlds
    ws = worlds('train')
    by = {(w['cell'], w['repeat']): w for w in ws}
    primary = [by[(c, 0)] for c in range(12)]
    fallback = [by[(c, 1)] for c in range(12)]
    # pass 3 order: descending (penetration, demand); repeat 1, then repeat 2
    order = sorted(range(12), key=lambda c: (-by[(c, 0)]['penetration'], -by[(c, 0)]['demand']))
    fill = [by[(c, 1)] for c in order]+[by[(c, 2)] for c in order]
    return primary, fallback, fill


def check_world(world):
    from tools.coopmwm.protocol import worlds
    s = int(world['seed'])
    assert not any(lo <= s <= hi for lo, hi in EVAL), 'evaluation seed'
    assert any(w == world for w in worlds('train')), 'not a registered TRAIN world'
    assert 986000 <= s <= 986071


def charge(ident, sim_s, kind, branches):
    import fcntl
    from tools.coopmwm.exact_grade import BudgetBlocked
    with (ROOT/'LEDGER.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = ROOT/'CHARGES.jsonl'
        rows = [json.loads(s) for s in ledger.read_text().splitlines() if s.strip()] if ledger.exists() else []
        if any(r['id'] == ident for r in rows):
            raise BudgetBlocked('already charged; never replay '+ident)
        if sum(r['sim_s'] for r in rows)+int(sim_s) > CAP:
            raise BudgetBlocked(f'upper_bound cap {CAP} reached')
        with ledger.open('a') as f:
            f.write(json.dumps(dict(id=ident, phase=PHASE, kind=kind, sim_s=int(sim_s), branches=branches,
                                    unix=time.time()))+'\n')
            f.flush(); os.fsync(f.fileno())


def spent():
    ledger = ROOT/'CHARGES.jsonl'
    rows = [json.loads(s) for s in ledger.read_text().splitlines() if s.strip()] if ledger.exists() else []
    return sum(r['sim_s'] for r in rows)


# ------------------------------------------------------------------ collection

def eligible(rows, members, commands):
    out = [r for r in rows if r['is_cav'] and r['lane'] in (0, 1) and S_LO <= r['s_m'] <= S_HI
           and r['vid'] in members and r['vid'] in commands]
    return sorted(out, key=lambda r: (-r['s_m'], r['vid']))


def collect(world):
    """One plain frozen-u0 B episode to COLLECT_STOP; the registered moment is SELECTION. No outcome input."""
    import torch
    from tools.coopmwm import accel_plant, sumo_verify as sv
    from tools.mining.collect_contexts import FIELDS
    check_world(world)
    seed = int(world['seed'])
    ident = f'ub_B_train_{seed}'
    work = ROOT/'collect'/ident
    work.mkdir(parents=True)          # exclusive: never replay a collection
    weights = torch.load(ROOT/'frozen_u0.pt', weights_only=False, map_location='cpu')['policy']
    job = dict(id=ident, path=str(work), world=world, role='train', arm='B', update=0, training_seed=20261101,
               weights=weights, runtime='/home/ubuntu/e3final',
               checkpoint='/home/ubuntu/mappospeed/production2/policy_00169.pt',
               noise_seed=20261106+seed+NOISE_OFFSET, deterministic=False, guided=False, b_replay=True, v2v=True,
               health_guards=False, sim_s=COLLECT_STOP, stop=COLLECT_STOP, warm_history=True,
               accel_bounds=[-3., 2.6])
    best = {}

    class Capture:
        def __call__(self, controller, job_, prefixes):
            self.c, self.job, self.prefixes = controller, job_, prefixes

        def after_step(self, t, after):
            e = self.c.events[-1]
            t = int(e['time_s'])
            members = e['sample']['context']['members']
            action = e['sample']['action']
            if best:
                k = t-best['t0']
                if 0 < k < PROFILE_S:
                    for vid, row in best['profiles'].items():
                        row.append(float(e['commands'][vid])+float(action[members.index(vid)][0])
                                   if vid in members and vid in e['commands'] else None)
            if not T_LO <= t <= T_HI or t % T_STEP:
                return
            rows = e['rows']
            elig = eligible(rows, members, e['commands'])
            if len(elig) < 2:
                return
            ramp = [r for r in rows if r['lane'] == 2 and -100 <= r['s_m'] <= S_HI]
            if not ramp:
                return
            main = [r for r in rows if r['lane'] in (0, 1) and -100 <= r['s_m'] <= S_HI]
            pressure = (len(ramp), round(sum(1/(1+r['v_mps']) for r in ramp), 6), len(main))
            score = (min(len(elig), 6),)+pressure
            if best and best['score'] >= score:
                return
            snap = sv.capture(self.c, self.prefixes, self.job, t, e['sample'], self.c.generator)
            snap['end_s'] = 1200      # registered episode end; collection physically stops at COLLECT_STOP
            band = [r for r in rows if r['is_cav'] and -400 <= r['s_m'] <= 100]
            sensed = {r['vid'] for c in band for r in rows if abs(r['s_m']-c['s_m']) <= SENSOR_RANGE_M}
            ctx_rows = [{k: r[k] for k in FIELDS} for r in rows if r['vid'] in sensed]
            cavs = [r['vid'] for r in band]
            best.clear()
            best.update(t0=t, score=score, snap=snap, profiles={
                v: [float(e['commands'][v])+float(action[members.index(v)][0])
                    if v in members and v in e['commands'] else None] for v in cavs},
                context=dict(state_id=f'{ident}_t{t}', world=world, arm='B', t0=t, rows=ctx_rows, cavs=cavs,
                             eligible=[r['vid'] for r in elig], pressure=list(pressure),
                             score=list(score), snapshot_sha256=sv.snapshot_sha(snap),
                             selection=SELECTION))

    charge('collect_'+ident, COLLECT_STOP, 'collect', 1)
    result = accel_plant.episode(job, controller_hook=Capture())
    receipt = dict(world=world, arm='B', actual_sim_s=result.get('actual_sim_s'), unix=time.time(),
                   noise_seed=job['noise_seed'], stop=COLLECT_STOP)
    if not best:
        receipt['moment'] = None
        _write(work/'COLLECTION_RECEIPT.json', receipt)
        return None
    d = work/f'ctx_t{best["t0"]}'
    d.mkdir()
    torch.save(best['snap'], d/'SNAPSHOT.pt')
    ctx = dict(best['context'], policy_profile=best['profiles'])
    _write(d/'CONTEXT.json', ctx)
    receipt['moment'] = dict(t0=best['t0'], dir=str(d), score=list(best['score']))
    _write(work/'COLLECTION_RECEIPT.json', receipt)
    return str(d)


# ------------------------------------------------------------------ candidates

def wall_plan(ctx, members, o, h, d, r):
    """Members ease by `d` m/s^2 on top of their seed-1 policy reference during [o, o+h); with r > 0 the
    easing is released gradually (linear taper to 0 over r s); then the live MAPPO policy resumes."""
    import numpy as np
    from tools.mining.mine_verify import _pol
    duration = int(o)+int(h)+int(r)
    assert 0 < duration <= D_MAX and 2 <= len(members) <= 6
    pol = _pol(ctx, list(members), duration)
    if pol is None:
        return None
    off = np.zeros((len(members), duration))
    off[:, o:o+h] = -float(d)
    for j in range(int(r)):
        off[:, o+h+j] = -float(d)*(1-(j+1)/(r+1))
    base = np.asarray(pol, float)
    pid = f'ub_wall_{"_".join(members)}_o{o}_h{h}_d{d:g}_r{r}'
    return dict(id=pid, family='wall', members=list(members), duration_s=duration, origin='upper_bound',
                policy_profile=base.tolist(), profile=(base+off).tolist(), perturbation_profile=off.tolist(),
                ub=dict(onset=int(o), hold=int(h), depth=float(d), release=int(r)))


def solo_plans(plan):
    from tools.coopmwm import replay_bank as rb
    q = plan['ub']
    out = []
    for m in plan['members']:
        s = rb.solo_candidate(plan, m)
        s['id'] = f"ub_solo_{m}_o{q['onset']}_h{q['hold']}_d{q['depth']:g}_r{q['release']}"
        out.append(s)
    return out


def member_sets(ctx):
    elig = ctx['eligible']               # sorted downstream -> upstream
    n = len(elig)
    sets = []
    for k in (min(4, n), 2, min(6, n)):
        s = tuple(elig[:k])
        if len(s) >= 2 and s not in sets:
            sets.append(s)
    up = tuple(elig[-min(4, n):])
    if len(up) >= 2 and up not in sets:
        sets.append(up)
    return sets


def screen_slate(ctx):
    """Registered coarse screen (<= 8): primary set x lead law; other sets x lead law; primary set x
    timing variants; then fill timings on the primary set."""
    sets = member_sets(ctx)
    keys = [(sets[0], T0_LAW)]+[(s, T0_LAW) for s in sets[1:]]+[(sets[0], t) for t in TIMINGS]
    keys += [(sets[0], t) for t in FILL]
    out, seen = [], set()
    for s, t in keys:
        if len(out) >= N_SCREEN:
            break
        if (s, t) in seen:
            continue
        seen.add((s, t))
        p = wall_plan(ctx, s, *t)
        if p is not None:
            out.append(p)
    return out


def _key(p):
    q = p['ub']
    return tuple(p['members']), (q['onset'], q['hold'], q['depth'], q['release'])


def refine_slate(ctx, screened, J, J_policy):
    """Registered refinement: for each of the top-2 screened candidates (lowest seed-1 J, safety-clean),
    the first 2 untried valid neighbours in this order: depth step (direction from the screen's
    d1.0 vs d0.5 comparison), member-count step (direction from the set-size comparison), hold +15,
    release toggle (0 <-> 15), onset +5, opposite depth step."""
    elig = ctx['eligible']
    tried = {_key(p) for p in screened}
    by_key = {_key(p): J[p['id']] for p in screened if J.get(p['id']) is not None}
    sets = member_sets(ctx)
    s0 = sets[0]
    j_lead, j_soft = by_key.get((s0, T0_LAW)), by_key.get((s0, TIMINGS[0]))
    down = j_lead is not None and j_soft is not None and j_soft < j_lead
    sizes = {len(s): by_key.get((s, T0_LAW)) for s in sets if s == tuple(elig[:len(s)])}
    sizes = {k: v for k, v in sizes.items() if v is not None}
    grow = bool(sizes) and max(sizes, key=lambda k: (-sizes[k], k)) >= len(s0)
    ranked = sorted([p for p in screened if J.get(p['id']) is not None], key=lambda p: (J[p['id']], p['id']))
    out = []
    for p in ranked[:N_REFINE_TOP]:
        members, (o, h, d, r) = _key(p)
        front = members == tuple(elig[:len(members)])
        upstream = members == tuple(elig[-len(members):])

        def resize(k):
            if not 2 <= k <= min(6, len(elig)):
                return None
            if front:
                return tuple(elig[:k])
            if upstream:
                return tuple(elig[-k:])
            return None
        dstep = (round(d*2/3, 2) if down else min(2.5, round(d*1.5, 2)))
        dstep2 = (min(2.5, round(d*1.5, 2)) if down else round(d*2/3, 2))
        k = len(members)
        nbrs = [(members, (o, h, dstep, r)),
                (resize(k+1 if grow else k-1), (o, h, d, r)),
                (members, (o, h+15, d, r)),
                (members, (o, h, d, 15 if r == 0 else 0)),
                (members, (o+5, h, d, r)),
                (members, (o, h, dstep2, r)),
                (resize(k-1 if grow else k+1), (o, h, d, r))]
        added = 0
        for m, t in nbrs:
            if added >= N_REFINE_EACH:
                break
            if m is None or len(m) < 2 or sum(t[:2])+t[3] > D_MAX or t[2] <= 0 or (m, t) in tried:
                continue
            q = wall_plan(ctx, m, *t)
            if q is None:
                continue
            tried.add((m, t))
            out.append(q)
            added += 1
    return out


# ------------------------------------------------------------------ observability

def observability(ctx, members):
    """Clean3 rule at t0 (track2_backend.observe): a member's K slots = itself + its K-1 nearest CAVs
    (|ds| <= 100 m, any lane, ties by vid). Observable_K iff some member's slots contain every member."""
    rows = {r['vid']: r for r in ctx['rows']}
    cavs = [v for v, r in rows.items() if r['is_cav']]
    out = {}
    for K in (4, 6):
        cover = {}
        for v in members:
            if v not in rows:
                cover[v] = 0
                continue
            s = rows[v]['s_m']
            near = sorted((u for u in cavs if u != v and abs(rows[u]['s_m']-s) <= SENSOR_RANGE_M),
                          key=lambda u: (abs(rows[u]['s_m']-s), u))[:K-1]
            vis = {v}|set(near)
            cover[v] = sum(m in vis for m in members)
        out[f'observable_{K}'] = any(c == len(members) for c in cover.values())
        out[f'max_cover_{K}'] = max(cover.values()) if cover else 0
    out['span_m'] = (max(rows[m]['s_m'] for m in members)-min(rows[m]['s_m'] for m in members)
                     if all(m in rows for m in members) else None)
    return out


# ------------------------------------------------------------------ execution

class Situation:
    def __init__(self, ctx_dir):
        import torch
        from tools.coopmwm import sumo_verify as sv
        self.dir = Path(ctx_dir)
        self.ctx = json.loads((self.dir/'CONTEXT.json').read_text())
        check_world(self.ctx['world'])
        self.snap = torch.load(self.dir/'SNAPSHOT.pt', weights_only=False)
        assert sv.snapshot_sha(self.snap) == self.ctx['snapshot_sha256']
        self.sid = self.ctx['state_id']
        self.t0 = int(self.ctx['t0'])
        self.window = dict(start_s=self.t0, end_s=self.t0+SPAN)
        self.res = {}           # seed -> {cand_id: branch result row}
        self.plans = {}

    def seed_snapshot(self, r):
        from tools.coopmwm import replay_bank as rb
        from tools.coopmwm.protocol import digest
        if r == 1:
            return self.snap, None
        salt = f'upperbound#{self.sid}#r{r}'
        for n in range(100):
            s = salt if not n else f'{salt}#x{n}'
            g = int(digest(dict(replication=s))[:15], 16)
            if not any(lo <= g <= hi for lo, hi in EVAL):
                return rb.replicate(self.snap, s), s
        raise ValueError('salt')


class Engine:
    def __init__(self, workers):
        self.pool = ThreadPoolExecutor(max_workers=workers)

    def batch(self, sit, r, stage, cands, need_policy):
        """Run the given candidates (and the policy branch if needed) on seed r with the fixed
        situation window. Returns {cand_id or 'policy': row}."""
        from tools.coopmwm import sumo_verify as sv
        from tools.coopmwm.exact_grade import _write_new
        if (ROOT/'STOP').exists():
            raise RuntimeError('STOP file present')
        snap_r, salt = sit.seed_snapshot(r)
        cands = [c for c in cands if c['id'] not in sit.res.get(r, {})]
        if not cands and not need_policy:
            return {}
        if not cands:   # policy-only batch: a zero-perturbation dummy is never run; build with a template
            raise ValueError('policy-only batch not used')
        maxd = max(int(c['duration_s']) for c in cands)
        specs = sv.build_branches(snap_r, cands, 'wm', tail_s=SPAN-maxd)
        assert all(s['window']['end_s'] == sit.t0+SPAN and s['window']['start_s'] == sit.t0 for s in specs)
        if not need_policy:
            specs = specs[1:]
        vid = specs[0]['verification_id']
        key = f'{vid}_r{r}'
        work = ROOT/'verify'/sit.sid/f'r{r}'
        runner = sv.SubprocessBranchRunner(work/'branches', workers=len(specs), timeout_s=3600)
        blocked = runner.admit(specs)
        if blocked:
            raise RuntimeError('resource guard: '+blocked)
        claim = ROOT/'claims'/f'{key}.json'
        _write_new(claim, dict(verification_id=vid, seed=r, stage=stage, sid=sit.sid, pid=os.getpid(),
                               unix=time.time(), cands=[c['id'] for c in cands]))
        charge(key, sum(int(s['sim_s']) for s in specs), stage, len(specs))
        for c in cands:
            if c['id'] not in sit.plans:
                _append(ROOT/'results'/'plans.jsonl', dict(id=c['id'], sid=sit.sid, members=c['members'],
                                                          ub=c.get('ub'), duration_s=c['duration_s']))
        runner.prepare(specs, snap_r)
        futs = [self.pool.submit(sv.run_branch, s, snap_r, runner) for s in specs]
        out = {}
        idx = {i+1: c for i, c in enumerate(cands)}
        for s, f in zip(specs, futs):
            b = f.result()
            cid = 'policy' if s['index'] == 0 else idx[s['index']]['id']
            row = dict(sid=sit.sid, seed=r, salt=salt, stage=stage, cand=cid, verification_id=vid,
                       branches_dir=str(work/'branches'), branch=b, J=b.get('J_veh_s'),
                       window=b.get('window'), status=b['status'])
            out[cid] = row
            _append(ROOT/'results'/'branches.jsonl', row)
        sit.res.setdefault(r, {}).update(out)
        for c in cands:
            sit.plans[c['id']] = c
        _write(ROOT/'receipts'/f'{key}.json', dict(key=key, stage=stage, sid=sit.sid, seed=r, salt=salt,
                                                    rows={k: {kk: v[kk] for kk in ('J', 'status')} for k, v in out.items()},
                                                    plans=[c['id'] for c in cands], unix=time.time()))
        return out


def tape(row):
    from tools.coopmwm import bank_statistics as bs
    return bs._read_result(row['branches_dir'], row['verification_id'], row['branch'])


def stats(ref_row, alt_row, window):
    """One-sided 95% LB (per-seed paired bootstrap) of J(ref) - J(alt) on the situation window."""
    from tools.coopmwm import bank_statistics as bs
    st = bs._stats([(tape(ref_row), tape(alt_row), window)])
    return dict(total=st['pooled_total_veh_s'], lb95=st['lower95_one_sided_veh_s'], unit=st['unit_source'])


def safety(pol, row):
    from tools.coopmwm import replay_bank as rb
    keys = tuple(rb.SAFETY_KEYS)+('ttc_below_3_vs',)
    return {k: float(row['window'].get(k, 0))-float(pol['window'].get(k, 0)) for k in keys
            if row.get('window') and pol.get('window')}


def clean(pol, row):
    s = safety(pol, row)
    return row['status'] == 'graded' and s.get('collisions', 0) <= 0 and s.get('teleports', 0) <= 0


def best_of(sit, r):
    res = sit.res.get(r, {})
    pol = res.get('policy')
    if pol is None or pol['status'] != 'graded':
        return None
    ok = [c for c, row in res.items() if c != 'policy' and c in sit.plans and len(sit.plans[c]['members']) >= 2
          and clean(pol, row)]
    if not ok:
        return None
    return min(ok, key=lambda c: (res[c]['J'], c))


def evaluate_seed(eng, sit, r, move, stage, solos_if_admitted=True):
    """Policy (if not yet run) + move on seed r; member solos iff the move is admitted vs base there."""
    res = sit.res.get(r, {})
    plan = sit.plans[move]
    if move not in res or 'policy' not in res:
        eng.batch(sit, r, stage, [plan], need_policy='policy' not in res)
    res = sit.res[r]
    pol, g = res['policy'], res[move]
    row = dict(sid=sit.sid, seed=r, move=move, stage=stage)
    if pol['status'] != 'graded' or g['status'] != 'graded':
        row['error'] = 'ungraded'
        return row
    a = stats(pol, g, sit.window)
    row.update(adv=pol['J']-g['J'], adv_lb95=a['lb95'], safety=safety(pol, g), J_policy=pol['J'], J_group=g['J'])
    if not solos_if_admitted or a['lb95'] <= 0:
        row['synergy_status'] = 'not_measured_group_not_admitted_vs_base'
        return row
    solos = solo_plans(plan)
    cost = (sit.t0+SPAN)*sum(s['id'] not in sit.res.get(r, {}) for s in solos)
    if CAP-spent()-RESERVE < cost:
        row['synergy_status'] = 'skipped_budget'
        return row
    eng.batch(sit, r, stage+'_solo', solos, need_policy=False)
    res = sit.res[r]
    sj = {s['id']: res[s['id']]['J'] for s in solos}
    if any(v is None for v in sj.values()):
        row['synergy_status'] = 'incomplete'
        return row
    b = min(sj, key=lambda k: (sj[k], k))
    sy = stats(res[b], g, sit.window)
    row.update(solos={k: pol['J']-v for k, v in sj.items()}, best_solo=b, best_solo_adv=pol['J']-sj[b],
               synergy=sj[b]-g['J'], synergy_lb95=sy['lb95'],
               synergy_status='admitted' if sy['lb95'] > 0 else 'not_admitted')
    return row


def run_situation_search(eng, sit):
    ctx = sit.ctx
    scr = screen_slate(ctx)
    log('screen', sid=sit.sid, n=len(scr), eligible=len(ctx['eligible']))
    if not scr:
        return dict(sid=sit.sid, error='no executable candidate')
    eng.batch(sit, 1, 'screen', scr, need_policy=True)
    res = sit.res[1]
    J = {c['id']: res[c['id']]['J'] for c in scr if res[c['id']]['status'] == 'graded'
         and clean(res['policy'], res[c['id']])}
    if res['policy']['status'] != 'graded':
        return dict(sid=sit.sid, error='policy branch ungraded')
    ref = refine_slate(ctx, scr, J, res['policy']['J'])
    log('refine', sid=sit.sid, n=len(ref))
    if ref:
        eng.batch(sit, 1, 'refine', ref, need_policy=False)
    return dict(sid=sit.sid, screened=len(scr), refined=len(ref))


def _collect_subprocess(worlds_):
    """Each collection is its own process (SUMO state is per process), nice 19, own PIDs logged."""
    import subprocess
    out, procs, pending = {}, {}, []
    for w in worlds_:
        rec = ROOT/'collect'/f'ub_B_train_{w["seed"]}'/'COLLECTION_RECEIPT.json'
        if (ROOT/'collect'/f'ub_B_train_{w["seed"]}').exists():   # adopt; never replay
            moment = json.loads(rec.read_text()).get('moment') if rec.exists() else None
            out[w['seed']] = dict(world=w, returncode=None, moment=moment, adopted=True)
        else:
            pending.append(w)
    while pending or procs:
        while pending and len(procs) < WORKERS:
            w = pending.pop(0)
            logf = open(ROOT/'logs'/f'collect_{w["seed"]}.log', 'w')
            pr = subprocess.Popen([sys.executable, '-B', '-u', '-m', 'tools.mining.upperbound', 'collect',
                                   str(w['seed'])], stdout=logf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                  preexec_fn=lambda: os.nice(max(0, 19-os.nice(0))))
            _append(ROOT/'PIDS.jsonl', dict(kind='collect', seed=w['seed'], pid=pr.pid, unix=time.time()))
            procs[pr] = (w, logf)
        time.sleep(2)
        for pr in [x for x in procs if x.poll() is not None]:
            w, logf = procs.pop(pr)
            logf.close()
            rec = ROOT/'collect'/f'ub_B_train_{w["seed"]}'/'COLLECTION_RECEIPT.json'
            moment = json.loads(rec.read_text()).get('moment') if rec.exists() else None
            out[w['seed']] = dict(world=w, returncode=pr.returncode, moment=moment)
    return out


def main_run():
    """Phase-ordered run (barriers between phases keep the essential evidence inside the cap)."""
    assert (ROOT/'REGISTRATION.json').exists(), 'register before any physics'
    primary, fallback, fill = registered_worlds()
    (ROOT/'logs').mkdir(exist_ok=True)
    _append(ROOT/'PIDS.jsonl', dict(kind='orchestrator', pid=os.getpid(), unix=time.time()))
    log('start', pid=os.getpid(), spent=spent())
    if (ROOT/'results'/'SITUATIONS.json').exists():
        raise RuntimeError('situations already fixed; resume of later phases is not implemented (never replay)')
    # smoke gate: one collection (highest-pressure cell) must complete with a receipt before the rest
    smoke = _collect_subprocess([primary[11]])
    if not all(v.get('adopted') or v['returncode'] == 0 for v in smoke.values()):
        log('STOP collection smoke failed', smoke=smoke)
        raise RuntimeError('collection smoke failed')
    # Phase 1: collections. Pass 1: repeat-0 world of every cell. Pass 2: repeat-1 world of each cell
    # without a qualifying moment. Pass 3: fill to 12 situations with unsampled worlds in descending
    # (penetration, demand) order, repeat 1 then repeat 2.
    col = _collect_subprocess(primary)
    miss = [fallback[c] for c in range(12)
            if not any(v['moment'] for v in col.values() if v['world']['cell'] == c)][:MAX_COLLECT-len(col)]
    if miss:
        col.update(_collect_subprocess(miss))
    have = [s for s, v in col.items() if v['moment']]
    if len(have) < 12:
        extra = [w for w in fill if w['seed'] not in col][:min(12-len(have), MAX_COLLECT-len(col))]
        if extra:
            col.update(_collect_subprocess(extra))
    sit_dirs = {s: v['moment']['dir'] for s, v in col.items() if v['moment']}
    _write(ROOT/'results'/'SITUATIONS.json', dict(collections=col, situations=sit_dirs, spent=spent()))
    sits = [Situation(sit_dirs[s]) for s in sorted(sit_dirs)]
    log('collections_done', n=len(sits), spent=spent())
    evaluate_phases(sits)


def evaluate_phases(sits):
    from tools.coopmwm.exact_grade import BudgetBlocked
    eng = Engine(WORKERS)

    def par(fn, items):
        if not items:
            return []
        with ThreadPoolExecutor(max_workers=len(items)) as tp:
            out = []
            for f in [tp.submit(fn, x) for x in items]:
                try:
                    out.append(f.result())
                except BudgetBlocked as e:
                    out.append(dict(error='BudgetBlocked: '+str(e)))
                except Exception as e:
                    import traceback
                    out.append(dict(error=repr(e), tb=traceback.format_exc()[-1500:]))
            return out
    # Phase 2: centralized search on seed 1 (screen + refine); the first situation is a smoke gate
    first = run_situation_search(eng, sits[0])
    _append(ROOT/'results'/'search.jsonl', first)
    graded = [c for c, v in sits[0].res.get(1, {}).items() if v['status'] == 'graded']
    if 'policy' not in graded or len(graded) < 2:
        log('STOP branch smoke failed', first=first, statuses={c: v['status'] for c, v in sits[0].res.get(1, {}).items()})
        raise RuntimeError('branch smoke failed')
    for row in par(lambda s: run_situation_search(eng, s), sits[1:]):
        _append(ROOT/'results'/'search.jsonl', row)
    log('search_done', spent=spent())
    # Phase 3: seed-1 best move (clairvoyant search seed) + its seed-1 synergy (solos iff admitted vs base)
    best = {s.sid: best_of(s, 1) for s in sits}
    _write(ROOT/'results'/'BEST.json', best)

    def seed1(s):
        if best[s.sid] is None:
            return dict(sid=s.sid, seed=1, move=None)
        return evaluate_seed(eng, s, 1, best[s.sid], 'search_eval')
    s1 = {}
    for r in par(seed1, sits):
        _append(ROOT/'results'/'perseed.jsonl', r)
        if 'sid' in r:
            s1[r['sid']] = r
    log('seed1_done', spent=spent())
    # Phase 4: held-out seeds 2,3 where the full-information planner ACTS (seed-1 best admitted vs base,
    # adv LB95 > 0); otherwise its choice is 'no move' (benefit 0, no branches).
    acting = [s for s in sits if (s1.get(s.sid) or {}).get('adv_lb95', -1) > 0]
    fit, room = [], CAP-spent()-RESERVE
    for s in acting:                     # fixed order (world seed); never start what cannot finish
        need = 2*len(HELDOUT)*(s.t0+SPAN)
        if need <= room:
            fit.append(s)
            room -= need
    if len(fit) < len(acting):
        log('heldout_budget_limited', acting=[s.sid for s in acting], kept=[s.sid for s in fit])
    acting = fit
    _write(ROOT/'results'/'ACTING.json', [s.sid for s in acting])

    def held(args):
        s, r, solos = args
        return evaluate_seed(eng, s, r, best[s.sid], 'heldout', solos_if_admitted=solos)
    for r in par(held, [(s, r, False) for s in acting for r in HELDOUT]):
        _append(ROOT/'results'/'perseed_nosolo.jsonl', r)
    log('heldout_base_done', spent=spent())
    # Phase 5: held-out member solos where the move is admitted vs base on that seed
    for r in par(held, [(s, r, True) for s in acting for r in HELDOUT]):
        _append(ROOT/'results'/'perseed.jsonl', r)
    log('heldout_done', spent=spent())
    # Phase 6: optional extra held-out seeds 4,5 (policy + move; solos iff admitted) if a whole seed fits
    for r_extra in EXTRA:
        need = sum((s.t0+SPAN)*(2+len(s.plans[best[s.sid]]['members'])) for s in acting)
        if not acting or CAP-spent()-RESERVE < need:
            log('extra_seed_skipped', seed=r_extra, need=need, remaining=CAP-spent())
            break
        for r in par(held, [(s, r_extra, True) for s in acting]):
            _append(ROOT/'results'/'perseed.jsonl', r)
        log('extra_seed_done', seed=r_extra, spent=spent())
    _write(ROOT/'DONE.json', dict(unix=time.time(), spent=spent(), situations=len(sits), acting=len(acting)))
    log('done', spent=spent())


# ------------------------------------------------------------------ summary

def _rows(name):
    p = ROOT/'results'/name
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def summary():
    import statistics as st
    plans = {x['id']: x for x in _rows('plans.jsonl')}
    branches = _rows('branches.jsonl')
    per = {}
    for r in _rows('perseed.jsonl'):
        if 'sid' in r and 'seed' in r:
            per.setdefault(r['sid'], {})[r['seed']] = r
    sits = json.loads((ROOT/'results'/'SITUATIONS.json').read_text())
    best = json.loads((ROOT/'results'/'BEST.json').read_text()) if (ROOT/'results'/'BEST.json').exists() else {}
    acting = set(json.loads((ROOT/'results'/'ACTING.json').read_text())) if (ROOT/'results'/'ACTING.json').exists() else set()
    out = []
    for seed_ in sorted(sits['situations'], key=int):
        ctx = json.loads((Path(sits['situations'][seed_])/'CONTEXT.json').read_text())
        sid = ctx['state_id']
        b1 = {b['cand']: b for b in branches if b['sid'] == sid and b['seed'] == 1}
        pol = b1.get('policy')
        searched = {c: b for c, b in b1.items() if c != 'policy' and b['stage'] in ('screen', 'refine')}
        adv = {c: (pol['J']-b['J'] if pol and b['J'] is not None and pol['J'] is not None else None)
               for c, b in searched.items()}
        move = best.get(sid)
        members = plans[move]['members'] if move in plans else None
        obs = observability(ctx, members) if members else None
        seeds = {}
        for k, v in sorted(per.get(sid, {}).items()):
            seeds[str(k)] = {kk: v.get(kk) for kk in ('adv', 'adv_lb95', 'synergy', 'synergy_lb95', 'synergy_status',
                                                      'best_solo', 'best_solo_adv', 'safety', 'solos', 'error')}
        worst = {}
        for c, b in searched.items():
            if pol and b.get('window') and pol.get('window'):
                for k in ('collisions', 'teleports', 'hard_brake', 'ttc_below_3_vs'):
                    d = float(b['window'].get(k, 0))-float(pol['window'].get(k, 0))
                    worst[k] = max(worst.get(k, d), d)
        out.append(dict(sid=sid, world=ctx['world'], t0=ctx['t0'], eligible=len(ctx['eligible']),
                        score=ctx.get('score'), n_searched=len(searched), searched_adv=adv,
                        move=move, move_params=(plans.get(move) or {}).get('ub'), n_members=len(members or []),
                        acting=sid in acting, observability=obs, seeds=seeds, max_safety_delta_over_search=worst))

    def q(xs):
        xs = [x for x in xs if x is not None]
        return dict(n=len(xs), mean=(sum(xs)/len(xs) if xs else None), median=(st.median(xs) if xs else None))
    agg = dict(situations=len(out), moves=sum(bool(s['move']) for s in out), acting=len(acting))
    agg['search_seed_ceiling_veh_s'] = q([max([0.]+[v for v in s['searched_adv'].values() if v is not None])
                                          for s in out if s['n_searched']])
    per_sit = []
    for s in out:
        vals = [v.get('adv') for k, v in s['seeds'].items() if k != '1' and v.get('adv') is not None]
        per_sit.append((sum(vals)/len(vals) if vals else None) if s['acting'] else 0.)
    agg['heldout_planner_gain_per_situation_veh_s'] = q(per_sit)
    agg['heldout_seed_adv_acting_veh_s'] = q([v.get('adv') for s in out for k, v in s['seeds'].items() if k != '1'])
    agg['synergy_admitted_search_seed'] = sum((s['seeds'].get('1', {}).get('synergy_lb95') or -1) > 0 for s in out)
    agg['synergy_admitted_any_heldout_seed'] = sum(any((v.get('synergy_lb95') or -1) > 0 for k, v in s['seeds'].items()
                                                       if k != '1') for s in out)
    agg['synergy_admitted_any_seed'] = sum(any((v.get('synergy_lb95') or -1) > 0 for v in s['seeds'].values())
                                           for s in out)
    agg['observable_4'] = sum(bool(s['observability'] and s['observability']['observable_4']) for s in out)
    agg['observable_6'] = sum(bool(s['observability'] and s['observability']['observable_6']) for s in out)
    res = dict(situations=out, aggregate=agg, spent=spent(), collections=sits['collections'])
    _write(ROOT/'results'/'SUMMARY.json', res)
    return res


def register():
    """Write the registration (rules, worlds, vocabulary, budget) before any physics; never overwrite."""
    import hashlib
    from tools.coopmwm.exact_grade import _write_new
    primary, fallback, fill = registered_worlds()
    src = Path(__file__).read_bytes()
    reg = dict(experiment=EXPERIMENT, phase=PHASE, cap_sim_s=CAP, ledger=str(ROOT/'CHARGES.jsonl'),
               owner_authority='owner-approved upper_bound phase (MAIN brief 2026-11-06), cap 150,000 sim-s, own ledger',
               base='frozen MAPPO u0 (frozen_u0.pt, clean2 A_checkpoint_000.pt), arm B plant settings as coop_probe',
               worlds=dict(primary=[w['seed'] for w in primary], fallback=[w['seed'] for w in fallback],
                           fill_order=[w['seed'] for w in fill],
                           rule='pass1 repeat-0 world per cell; pass2 repeat-1 for cells w/o moment; pass3 fill to 12'),
               excluded_seed_ranges=EVAL, selection=SELECTION, collection_stop_s=COLLECT_STOP,
               noise_seed='20261106 + world seed + %d' % NOISE_OFFSET,
               radius='mainline (lane 0/1) CAVs with s in [-400, 75.44] m, policy-controlled at t0; any CAV '
                      'regardless of its sensing slots',
               vocabulary=dict(lead_law=T0_LAW, timings=TIMINGS, fill=FILL, d_max_s=D_MAX,
                               plan='members follow seed-1 policy reference + offset; offset -depth in [o,o+h); '
                                    'release r>0 tapers linearly to 0 over r s; then live MAPPO resumes',
                               member_sets='front-4, front-2, front-6, upstream-4 (eligible sorted downstream first)'),
               search=dict(screen=N_SCREEN, refine=f'{N_REFINE_EACH} neighbours x top-{N_REFINE_TOP}',
                           seed=1, window=f'[t0, t0+{SPAN}] for every branch (tail >= {TAIL})',
                           best='lowest J among searched multi-CAV candidates with no collision/teleport increase'),
               evaluation=dict(acting_rule='planner acts iff seed-1 best adv LB95 > 0 (else no move, gain 0)',
                               heldout_seeds=HELDOUT, extra_seeds=EXTRA, reserve=RESERVE,
                               solos='all member solos on a seed iff the move is admitted vs base (adv LB95 > 0) there',
                               statistic='bank_statistics._stats per seed (30-s circular block / per-vehicle '
                                         'paired bootstrap B=2000), one-sided 95% lower bound',
                               synergy='J(best solo) - J(group); admitted iff LB95 > 0',
                               observability='clean3 track2_backend rule at t0: member slots = self + K-1 nearest '
                                             'CAVs within 100 m; K=4 and K=6'),
               source_sha256=hashlib.sha256(src).hexdigest(), unix=time.time())
    _write_new(ROOT/'REGISTRATION.json', reg)
    print(json.dumps(dict(registered=True, source_sha256=reg['source_sha256'])))


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'run':
        main_run()
    elif cmd == 'register':
        register()
    elif cmd == 'collect':
        from tools.coopmwm.protocol import worlds
        w = next(x for x in worlds('train') if int(x['seed']) == int(sys.argv[2]))
        print(json.dumps(dict(seed=w['seed'], ctx=collect(w))), flush=True)
    elif cmd == 'summary':
        print(json.dumps(summary()['aggregate'], indent=1, default=str))
