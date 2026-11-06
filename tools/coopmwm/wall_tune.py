"""Per-situation wall tuning (W2): a SUMO JOB BUILDER plus its cpuV executor. Nothing runs here.

COOP_PROBE_REPORT (2026-11-06): one verified 4-CAV wall (R 986061 t455, o0 h30 d1) has
per-seed synergy +271..+487 veh-s on some CRN seeds and loses up to -505 on others; the
weaker d0.6/h20 variant gave ~0. The probe never searched the wall's parameters. This
tool searches, for ONE verified wall situation, onset / hold / depth / release shape
(``wall_release``) / member subset, with the probe's machinery: one ``sumo_verify.verify``
per (stage, CRN seed) holding policy-continues (the base), every group candidate and --
in synergy mode -- every member solo, on one shared CRN window; seed r >= 2 is
``replay_bank.replicate(snapshot, salt)`` with evaluation seed ranges excluded; per-seed
circular-block paired bootstrap ``bank_statistics._stats``. Native confirmation is
``verify_perseed.confirm_all8`` (seed 2 + later harm + per-seed solos + extra seeds), and
``emit`` writes MINED_MOVES rows (the clean3 ``coopmwm-mined-move-v1`` binding format) and
WALL_NEGATIVES rows (``wall_negatives``).

Ported from c4/coop-probe 5e6eb9e4 (tools/mining/probe.py verify, probe_native.py run /
extract) onto tools.coopmwm only, with three registered differences: (1) the coarse
screen is group-vs-policy only (no solos), synergy is measured in refinement and native
confirmation; (2) solo plans are content-addressed so identical member profiles are run
once and never collide across release shapes; (3) fresh tuning salts
``walltune#<state_id>#r<k>`` (the probe's selection seeds are not reused for tuning).

Per-seed rule (never pooled): every candidate of a stage is measured on the SAME
registered seed set; a candidate with any missing/failed seed is ineligible (never
imputed); ranking keys are counts of per-seed positives and the WORST seed, never a
mean; emitted rows bind their own seed; negatives bind their own seed.

Budget: ``init`` registers caps (situation, per stage, root total) BEFORE any job is
built; builders quote every job exactly (branches x branch sim-s from t=0, as sv.quote)
and drop the lowest-priority candidates until the stage fits min(stage cap, situation
cap, root cap) minus everything already reserved by built plans; the executor charges a
durable ledger and refuses (BudgetBlocked) any charge that differs from the quote, is a
replay, or would exceed a cap. Confirmation is quoted at its upper bound (every
candidate admitted and cooperative on every seed).

CLI (cpuV for run/confirm; everything else is CPU-only file work):
  python -m tools.coopmwm.wall_tune quote [t0] [members]
  python -m tools.coopmwm.wall_tune init <root> [root_cap_sim_s]
  python -m tools.coopmwm.wall_tune screen <root> <situation.json>
  python -m tools.coopmwm.wall_tune refine <root> <situation.json> [round]
  python -m tools.coopmwm.wall_tune confirm-job <root> <situation.json>
  python -m tools.coopmwm.wall_tune run <root> <plan.json> [workers]          (SUMO)
  python -m tools.coopmwm.wall_tune confirm <root> <confirm.json> [workers]   (SUMO)
  python -m tools.coopmwm.wall_tune emit <root> <bank_root> <arm> [--stage-negatives]
situation.json: {situation_id, ctx_dir (probe collection dir with CONTEXT.json +
SNAPSHOT.pt), arm, members (the verified wall, leader first)}.
"""
from itertools import combinations
import json
import os
from pathlib import Path
import sys
import time

from . import exact_grade as _xg
from .protocol import digest

SCHEMA_REGISTRATION = 'coopmwm-wall-tune-registration-v1'
SCHEMA_PLAN = 'coopmwm-wall-tune-plan-v1'
SCHEMA_JOB = 'coopmwm-wall-tune-job-v1'
SCHEMA_CONFIRM = 'coopmwm-wall-tune-confirm-v1'
SCHEMA_RESULT = 'coopmwm-wall-tune-result-v1'
PHASE = 'wall_tune'
EVAL_RANGES = ((962000, 962999), (964000, 964199), (977000, 977999))
STAGES = ('screen', 'refine', 'confirm')
PROFILE_S = 130          # probe collection: per-second policy references for 130 s from t0
TAIL_S = 120             # sumo_verify.TAIL_S (the probe's screening tail)

REGISTERED = dict(
    screen=dict(seeds=(1, 2, 3), mode='group', onset_s=(0,), hold_s=(20, 30), depth=(.6, 1.),
                release=(('step', 0), ('linear', 20), ('cosine', 20))),
    refine=dict(seeds=(1, 2, 3), mode='synergy', top=2, max_candidates=4,
                steps=dict(onset_s=5, hold_s=10, depth=.2, release_s=10), default_release_s=20),
    confirm=dict(extra_seeds=4, top=1),
    caps=dict(situation_sim_s=120000, screen_sim_s=30000, refine_sim_s=50000, confirm_sim_s=45000),
    salt='walltune#{state_id}#r{seed}', tail_s=TAIL_S, profile_s=PROFILE_S,
    rule=('per-seed: same registered seeds for every candidate of a stage; missing seed => '
          'ineligible; rank by per-seed positive counts then the worst seed; never pooled'),
    provenance='W2 owner 2026-11-06 (COOP_PROBE_REPORT); c4/coop-probe 5e6eb9e4 machinery')


class BudgetBlocked(_xg.BudgetBlocked):
    """A tuning job would exceed a registered cap (or replay a charge); nothing runs.

    Subclasses exact_grade.BudgetBlocked so sumo_verify.verify records 'budget_blocked'
    (no physics) instead of crashing the whole plan."""


# ------------------------------------------------------------------ registration / situation

def _write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, sort_keys=True, indent=1), encoding='utf-8')
    os.replace(tmp, path)


def _guard(path):
    from .track2_backend import _path
    return _path(path)


def init(root, root_cap_sim_s=250000, caps=None):
    """Register caps BEFORE any job is built; an existing registration is never rewritten."""
    root = _guard(root)
    path = root/'REGISTRATION.json'
    if path.exists():
        return registration(root)
    caps = dict(REGISTERED['caps'], **(caps or {}))
    if any(type(v) is not int or v <= 0 for v in caps.values()) or type(root_cap_sim_s) is not int \
            or root_cap_sim_s <= 0:
        raise ValueError('caps must be positive integer sim-s')
    if caps['situation_sim_s'] > root_cap_sim_s or any(
            caps[f'{s}_sim_s'] > caps['situation_sim_s'] for s in STAGES):
        raise ValueError('stage caps <= situation cap <= root cap')
    body = dict(schema=SCHEMA_REGISTRATION, phase=PHASE, caps=caps, root_cap_sim_s=root_cap_sim_s,
                registered=REGISTERED, unix=time.time())
    body['registration_sha256'] = digest({k: v for k, v in body.items() if k != 'unix'})
    _write(path, body)
    return registration(root)


def registration(root):
    path = _guard(root)/'REGISTRATION.json'
    if not path.exists():
        raise ValueError('wall_tune root has no registered budget (run init first)')
    body = json.loads(path.read_text(encoding='utf-8'))
    if body.get('schema') != SCHEMA_REGISTRATION or body['registration_sha256'] != digest(
            {k: v for k, v in body.items() if k not in ('unix', 'registration_sha256')}):
        raise ValueError('wall_tune registration changed after it was written')
    return body


def check_world(world):
    s = int(world['seed'])
    if any(lo <= s <= hi for lo, hi in EVAL_RANGES) or not 986000 <= s <= 986071:
        raise ValueError('wall tuning uses registered TRAIN worlds 986000..986071 only')


def load_situation(spec):
    """Validated situation: the verified wall + its observable collection context."""
    if isinstance(spec, (str, Path)):
        spec = json.loads(_guard(spec).read_text(encoding='utf-8'))
    ctx_dir = _guard(spec['ctx_dir'])
    ctx = json.loads((ctx_dir/'CONTEXT.json').read_text(encoding='utf-8'))
    check_world(ctx['world'])
    if spec['arm'] not in ('B', 'R') or ctx['arm'] != spec['arm']:
        raise ValueError('situation arm must equal its collection/verification arm (B or R)')
    members = list(spec['members'])
    from . import moves
    if not 2 <= len(members) <= moves.MOVE_MAX_MEMBERS or len(set(members)) != len(members):
        raise ValueError(f'verified wall must have 2..{moves.MOVE_MAX_MEMBERS} distinct CAVs')
    rows = {r['vid']: r for r in ctx['rows']}
    if any(v not in rows or not rows[v]['is_cav'] or v not in ctx['policy_profile'] for v in members):
        raise ValueError('wall members must be observed CAVs with policy references')
    t0 = int(ctx['t0'])
    if not 300 < t0 < 1200:
        raise ValueError('decision time outside the controlled episode')
    return dict(situation_id=str(spec['situation_id']), state_id=ctx['state_id'], ctx_dir=str(ctx_dir),
                arm=spec['arm'], t0=t0, world=ctx['world'], snapshot_sha256=ctx['snapshot_sha256'],
                members=members, lanes={v: int(rows[v]['lane']) for v in members},
                policy_profile={v: ctx['policy_profile'][v] for v in members})


# ------------------------------------------------------------------ candidates

def _policy_rows(situation, members, duration):
    rows = []
    for v in members:
        ref = situation['policy_profile'][v]
        if len(ref) < duration or any(x is None for x in ref[:duration]):
            return None
        rows.append([float(x) for x in ref[:duration]])
    return rows


def candidate(situation, members, onset, hold, depth, shape, release_s):
    """Executable wall plan (wall_release.plan) or None (invalid / no policy reference)."""
    from . import moves, wall_release
    try:
        move = moves.Move('wall', tuple(members), int(onset), int(hold), float(depth),
                          int(release_s) if shape != 'step' else 0, shape)
    except ValueError:
        return None
    if len({situation['lanes'][v] for v in members}) < int(moves._WALL_MIN_LANES):
        return None
    dur = wall_release.duration(move)
    if dur > REGISTERED['profile_s']:
        return None
    pol = _policy_rows(situation, members, dur)
    return None if pol is None else wall_release.plan(move, pol)


def solo(plan, member):
    """Member-only plan (replay_bank.solo_candidate semantics), content-addressed id."""
    i = plan['members'].index(member)
    body = dict(family='wall_solo', members=[member], duration_s=plan['duration_s'],
                profile=[plan['profile'][i]], policy_profile=[plan['policy_profile'][i]],
                perturbation_profile=[plan['perturbation_profile'][i]])
    return dict(body, id=f'wt_solo_{member}_{digest(body)[:16]}')


def screen_candidates(situation):
    reg = REGISTERED['screen']
    out = []
    for onset in reg['onset_s']:
        for hold in reg['hold_s']:
            for depth in reg['depth']:
                for shape, release in reg['release']:
                    plan = candidate(situation, situation['members'], onset, hold, depth, shape, release)
                    if plan is not None:
                        out.append(plan)
    return out


def neighbours(situation, plan):
    """Registered one-coordinate refinement neighbourhood of a wall plan (priority order)."""
    from . import moves, wall_release
    move = moves.from_plan(plan)
    reg = REGISTERED['refine']
    st = reg['steps']
    shape = move.release_shape if move.release_s else 'step'
    R = move.release_s
    lo_r, hi_r = wall_release.RELEASE_WINDOW_S
    options = []
    if shape == 'step':
        options += [(move.members, move.onset, move.hold, move.depth, s, reg['default_release_s'])
                    for s in wall_release.RELEASE_SHAPES]
    else:
        options += [(move.members, move.onset, move.hold, move.depth, s, R)
                    for s in wall_release.RELEASE_SHAPES if s != shape]
        options += [(move.members, move.onset, move.hold, move.depth, shape, r)
                    for r in (R+st['release_s'], R-st['release_s']) if lo_r <= r <= hi_r]
        options.append((move.members, move.onset, move.hold, move.depth, 'step', 0))
    options += [(move.members, move.onset, move.hold, round(move.depth+d, 10), shape, R)
                for d in (st['depth'], -st['depth'])]
    options += [(move.members, move.onset, move.hold+d, move.depth, shape, R)
                for d in (st['hold_s'], -st['hold_s'])]
    options += [(move.members, move.onset+d, move.hold, move.depth, shape, R)
                for d in (st['onset_s'], -st['onset_s'])]
    if len(move.members) >= 3:
        options += [(sub, move.onset, move.hold, move.depth, shape, R)
                    for sub in combinations(move.members, len(move.members)-1)]
    out = []
    for members, onset, hold, depth, s, r in options:
        p = candidate(situation, members, onset, hold, depth, s, r)
        if p is not None and digest(p) != digest(plan):
            out.append(p)
    return out


# ------------------------------------------------------------------ quotes / jobs

def salt_for(state_id, seed):
    """None for the collection CRN (seed 1); otherwise a tuning salt outside eval seeds."""
    if seed == 1:
        return None
    base = REGISTERED['salt'].format(state_id=state_id, seed=seed)
    for n in range(100):
        s = base if not n else f'{base}#x{n}'
        g = int(digest(dict(replication=s))[:15], 16)
        if not any(lo <= g <= hi for lo, hi in EVAL_RANGES):
            return s
    raise ValueError('no tuning salt outside the evaluation seed ranges')


def branch_sim_s(t0, plans, tail_s=TAIL_S):
    from . import sumo_verify as sv
    return sv.branch_sim_s(sv.window_end(t0, plans, tail_s=tail_s))


def build_jobs(situation, stage, plans, *, round_=0):
    """One job per registered seed: policy + groups (+ content-addressed solos in synergy mode)."""
    reg = REGISTERED[stage]
    solos, owner = [], {}
    if reg['mode'] == 'synergy':
        seen = {}
        for p in plans:
            for m in p['members']:
                s = solo(p, m)
                seen.setdefault(s['id'], s)
                owner.setdefault(p['id'], []).append(s['id'])
        solos = list(seen.values())
    sim = branch_sim_s(situation['t0'], plans+solos)
    jobs = []
    for seed in reg['seeds']:
        job = dict(schema=SCHEMA_JOB, phase=PHASE, stage=stage, round=int(round_), mode=reg['mode'],
                   situation={k: situation[k] for k in ('situation_id', 'state_id', 'ctx_dir', 'arm', 't0',
                                                         'world', 'snapshot_sha256')},
                   seed=int(seed), salt=salt_for(situation['state_id'], seed), groups=plans, solos=solos,
                   owner=owner, tail_s=REGISTERED['tail_s'], branch_sim_s=sim,
                   n_branches=1+len(plans)+len(solos), quote_sim_s=(1+len(plans)+len(solos))*sim)
        job['tag'] = f"{situation['situation_id']}_{stage}{round_}_r{seed}_{digest(job)[:10]}"
        job['job_sha256'] = digest(job)
        jobs.append(job)
    return jobs


def confirm_quote(situation, plans, extra_seeds):
    """Upper bound of one native all8 confirmation (all admitted + cooperative everywhere)."""
    from . import verify_perseed as vp
    idx = list(range(len(plans)))
    q = vp.quote_decision(situation['t0'], [p['duration_s'] for p in plans], [p['members'] for p in plans],
                          admitted={1: idx, 2: idx}, extra_seeds=extra_seeds, cooperative=idx)
    return int(q.get('all8_total_with_extra_sim_s', q['all8_total_sim_s']))


# ------------------------------------------------------------------ ledger / reservations

def _ledger(root):
    path = Path(root)/'CHARGES.jsonl'
    if not path.exists():
        return []
    return [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]


def _plans(root):
    out = []
    for p in sorted((Path(root)/'plans').glob('*.json')):
        out.append(json.loads(p.read_text(encoding='utf-8')))
    return out


def reserved(root, situation_id=None, stage=None):
    """Quoted upper bound of every BUILT plan (charges never exceed their quote)."""
    total = 0
    for plan in _plans(root):
        if situation_id is not None and plan['situation']['situation_id'] != situation_id:
            continue
        if stage is not None and plan['stage'] != stage:
            continue
        total += plan['quote_sim_s']
    return total


def available(root, situation_id, stage):
    reg = registration(root)
    caps = reg['caps']
    return min(caps[f'{stage}_sim_s']-reserved(root, situation_id, stage),
               caps['situation_sim_s']-reserved(root, situation_id),
               reg['root_cap_sim_s']-reserved(root))


def _fit(root, situation, stage, ordered, build):
    """Drop lowest-priority candidates until the stage quote fits every remaining cap."""
    cap = available(root, situation['situation_id'], stage)
    plans = list(ordered)
    while plans:
        jobs = build(plans)
        quote = sum(j['quote_sim_s'] for j in jobs) if isinstance(jobs, list) else jobs['quote_sim_s']
        if quote <= cap:
            return plans, jobs, quote, cap
        plans = plans[:-1]
    raise BudgetBlocked(f'{stage}: no candidate fits the remaining cap {cap} sim-s')


def _save_plan(root, situation, stage, round_, jobs, quote, cap, extra=None):
    body = dict(schema=SCHEMA_PLAN, phase=PHASE, stage=stage, round=int(round_),
                situation={k: situation[k] for k in ('situation_id', 'state_id', 'ctx_dir', 'arm', 't0',
                                                      'world', 'snapshot_sha256', 'members')},
                jobs=jobs, quote_sim_s=int(quote), cap_at_build_sim_s=int(cap),
                registration_sha256=registration(root)['registration_sha256'], rule=REGISTERED['rule'],
                **(extra or {}))
    body['plan_sha256'] = digest(body)
    path = Path(root)/'plans'/f"{situation['situation_id']}_{stage}{round_}.json"
    if path.exists():
        raise ValueError(f'stage plan already built (never rebuilt): {path.name}')
    _write(path, body)
    return path, body


def _situation(value):
    return value if isinstance(value, dict) and 'lanes' in value else load_situation(value)


def build_screen(root, situation):
    situation = _situation(situation)
    plans, jobs, quote, cap = _fit(root, situation, 'screen', screen_candidates(situation),
                                   lambda ps: build_jobs(situation, 'screen', ps))
    return _save_plan(root, situation, 'screen', 0, jobs, quote, cap)


# ------------------------------------------------------------------ results / per-seed ranking

def results(root, stage, situation_id):
    path = Path(root)/'results'/f'{stage}.jsonl'
    if not path.exists():
        return []
    return [r for r in (json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip())
            if r['situation_id'] == situation_id]


def per_candidate(rows, seeds, mode):
    """{plan_sha256: {seed: row}} for candidates COMPLETE on every registered seed (per-seed rule)."""
    by = {}
    for r in rows:
        by.setdefault(r['plan_sha256'], {})[int(r['seed'])] = r
    out = {}
    for key, seeds_rows in by.items():
        if set(seeds_rows) != set(seeds):
            continue
        ok = True
        for r in seeds_rows.values():
            need = ('adv', 'adv_lb95') + (('synergy', 'synergy_lb95') if mode == 'synergy' else ())
            if r.get('status') != 'graded' or any(r.get(k) is None for k in need):
                ok = False
            elif any(float((r.get('safety') or {}).get(k) or 0.) > 0 for k in ('collisions', 'teleports')):
                ok = False
        if ok:
            out[key] = seeds_rows
    return out


def rank_key(seeds_rows, mode):
    rows = list(seeds_rows.values())
    if mode == 'group':
        return (sum(r['adv_lb95'] > 0 for r in rows), sum(r['adv'] > 0 for r in rows),
                min(r['adv'] for r in rows))
    return (sum(r['synergy_lb95'] > 0 and r['adv_lb95'] > 0 for r in rows),
            sum(r['synergy_lb95'] > 0 for r in rows), sum(r['synergy'] > 0 for r in rows),
            min(r['synergy'] for r in rows))


def ranked(rows, stage):
    reg = REGISTERED[stage]
    complete = per_candidate(rows, reg['seeds'], reg['mode'])
    order = sorted(complete, key=lambda k: (tuple(-x for x in rank_key(complete[k], reg['mode'])), k))
    return [(k, complete[k]) for k in order]


def build_refine(root, situation, round_=1):
    situation = _situation(situation)
    sid = situation['situation_id']
    screen = ranked(results(root, 'screen', sid), 'screen')
    prior = ranked([r for rd in range(1, int(round_)) for r in results(root, 'refine', sid)
                    if r['round'] == rd], 'refine') if round_ > 1 else []
    tops = [rows[min(rows)]['plan'] for _, rows in (prior or screen)[:REGISTERED['refine']['top']]]
    if not tops:
        raise ValueError('refinement needs complete per-seed screen results (none eligible)')
    measured = {r['plan_sha256'] for r in results(root, 'refine', sid)}
    lists = [[t]+neighbours(situation, t) for t in tops]
    ordered, seen = [], set(measured)
    for i in range(max(len(x) for x in lists)):
        for lst in lists:
            if i < len(lst) and digest(lst[i]) not in seen:
                seen.add(digest(lst[i]))
                ordered.append(lst[i])
    ordered = ordered[:REGISTERED['refine']['max_candidates']]
    if not ordered:
        raise ValueError('refinement neighbourhood exhausted')
    plans, jobs, quote, cap = _fit(root, situation, 'refine', ordered,
                                   lambda ps: build_jobs(situation, 'refine', ps, round_=round_))
    return _save_plan(root, situation, 'refine', round_, jobs, quote, cap,
                      dict(seeded_from=[digest(t) for t in tops]))


def build_confirm(root, situation):
    situation = _situation(situation)
    sid = situation['situation_id']
    refine = ranked(results(root, 'refine', sid), 'refine')
    reg = REGISTERED['confirm']
    best = [rows for _, rows in refine if rank_key(rows, 'synergy')[0] > 0][:reg['top']]
    if not best:
        raise ValueError('no refined wall is admitted (adv LB95 > 0 and synergy LB95 > 0) on any seed')
    plans = [rows[min(rows)]['plan'] for rows in best]

    def build(ps):
        body = dict(schema=SCHEMA_CONFIRM, phase=PHASE, stage='confirm',
                    situation={k: situation[k] for k in ('situation_id', 'state_id', 'ctx_dir', 'arm', 't0',
                                                          'world', 'snapshot_sha256')},
                    plans=ps, extra_seeds=reg['extra_seeds'],
                    quote_sim_s=confirm_quote(situation, ps, reg['extra_seeds']))
        body['tag'] = f"{sid}_confirm_{digest(body)[:10]}"
        body['job_sha256'] = digest(body)
        return body
    plans, job, quote, cap = _fit(root, situation, 'confirm', plans, build)
    return _save_plan(root, situation, 'confirm', 0, [job], quote, cap)


def quote_situation(t0=455, n_members=4, lanes=(0, 1, 1, 0)):
    """Expected (= capped upper-bound) sim-s of the registered pipeline for one situation."""
    members = [f'c{i}' for i in range(n_members)]
    situation = dict(situation_id='quote', state_id='quote', t0=int(t0), members=members, ctx_dir='quote',
                     arm='B', world=dict(seed=986061), snapshot_sha256='0'*64,
                     lanes={v: lanes[i % len(lanes)] for i, v in enumerate(members)},
                     policy_profile={v: [0.]*PROFILE_S for v in members})
    screen = screen_candidates(situation)
    s_jobs = build_jobs(situation, 'screen', screen)
    tops = screen[:REGISTERED['refine']['top']]
    lists = [[t]+neighbours(situation, t) for t in tops]
    ordered = []
    for i in range(max(len(x) for x in lists)):
        for lst in lists:
            if i < len(lst) and lst[i] not in ordered:
                ordered.append(lst[i])
    worst = sorted(ordered, key=lambda p: -p['duration_s']-100*len(p['members']))
    r_jobs = build_jobs(situation, 'refine', worst[:REGISTERED['refine']['max_candidates']])
    caps = REGISTERED['caps']
    out = dict(t0=int(t0), members=n_members,
               screen_sim_s=sum(j['quote_sim_s'] for j in s_jobs),
               refine_sim_s=sum(j['quote_sim_s'] for j in r_jobs),
               confirm_sim_s=confirm_quote(situation, [worst[0]], REGISTERED['confirm']['extra_seeds']),
               screen_candidates=len(screen), refine_candidates=len(r_jobs[0]['groups']),
               refine_solos=len(r_jobs[0]['solos']), seeds=list(REGISTERED['screen']['seeds']))
    for s in STAGES:
        out[f'{s}_capped_sim_s'] = min(out[f'{s}_sim_s'], caps[f'{s}_sim_s'])
    out['total_sim_s'] = sum(out[f'{s}_sim_s'] for s in STAGES)
    out['total_capped_sim_s'] = min(sum(out[f'{s}_capped_sim_s'] for s in STAGES), caps['situation_sim_s'])
    return out


# ------------------------------------------------------------------ executor (cpuV; SUMO)

def charge(root, job, ident, sim_s, kind='verify'):
    """Durable precharge under the registered caps; never replays an id (BudgetBlocked)."""
    import fcntl
    reg = registration(root)
    sid = job['situation']['situation_id']
    with (Path(root)/'LEDGER.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = _ledger(root)
        if any(r['id'] == ident for r in rows):
            raise BudgetBlocked('already charged; never replay '+ident)
        stage_rows = [r for r in rows if r['situation_id'] == sid and r['stage'] == job['stage']]
        job_rows = [r for r in rows if r['job_sha256'] == job['job_sha256']]
        caps = reg['caps']
        if sum(r['sim_s'] for r in job_rows)+int(sim_s) > job['quote_sim_s']:
            raise BudgetBlocked('charge exceeds the builder quote of this job')
        if sum(r['sim_s'] for r in stage_rows)+int(sim_s) > caps[f"{job['stage']}_sim_s"]:
            raise BudgetBlocked(f"{job['stage']} cap reached")
        if sum(r['sim_s'] for r in rows if r['situation_id'] == sid)+int(sim_s) > caps['situation_sim_s']:
            raise BudgetBlocked('situation cap reached')
        if sum(r['sim_s'] for r in rows)+int(sim_s) > reg['root_cap_sim_s']:
            raise BudgetBlocked('root cap reached')
        with (Path(root)/'CHARGES.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(dict(id=ident, phase=PHASE, stage=job['stage'], situation_id=sid,
                                    job_sha256=job['job_sha256'], kind=kind, sim_s=int(sim_s),
                                    unix=time.time()))+'\n')
            f.flush()
            os.fsync(f.fileno())


def _snapshot(situation, seed=1, salt=None):
    import torch
    from . import replay_bank as rb, sumo_verify as sv
    check_world(situation['world'])
    snap = torch.load(Path(situation['ctx_dir'])/'SNAPSHOT.pt', weights_only=False)
    if sv.snapshot_sha(snap) != situation['snapshot_sha256']:
        raise ValueError('situation snapshot changed')
    if (snap.get('episode') or {}).get('arm') != situation['arm']:
        # The same rule bank_breadth.verified_restart enforces downstream; refuse before spending.
        raise ValueError('snapshot episode arm differs from the situation arm (use the arm-labelled context)')
    return snap if salt is None else rb.replicate(snap, salt)


def _check_job(job):
    if job.get('schema') != SCHEMA_JOB or job['job_sha256'] != digest(
            {k: v for k, v in job.items() if k != 'job_sha256'}):
        raise ValueError('tuning job changed after it was built')
    if job['salt'] != salt_for(job['situation']['state_id'], job['seed']):
        raise ValueError('tuning job salt is not the registered CRN salt for its seed')


def run_job(root, job, *, workers=8, verify=None, stats=None, read_result=None, snapshot=None, runner=None):
    """One (stage, seed) verification -> per-group result rows (also appended to results/)."""
    from . import bank_statistics as bs, replay_bank as rb, sumo_verify as sv
    _check_job(job)
    verify = verify or sv.verify
    stats = stats or bs._stats
    read_result = read_result or bs._read_result
    situation = job['situation']
    snap = snapshot if snapshot is not None else _snapshot(situation, job['seed'], job['salt'])
    cands = job['groups']+job['solos']
    work = Path(root)/'verify'/job['stage']/job['tag']
    runner = runner or sv.SubprocessBranchRunner(work/'branches', workers=workers, timeout_s=3600)

    def precharge(vid, specs):
        total = sum(int(s['sim_s']) for s in specs)
        if total != job['quote_sim_s']:
            raise BudgetBlocked(f'verification charge {total} differs from the job quote {job["quote_sim_s"]}')
        charge(root, job, f"{job['tag']}_{vid}", total)
    rec = verify(snap, cands, runner, precharge, 'wm', workers=workers, root=work, tail_s=job['tail_s'])
    base = dict(schema=SCHEMA_RESULT, stage=job['stage'], round=job['round'], mode=job['mode'],
                situation_id=situation['situation_id'], state_id=situation['state_id'], t0=situation['t0'],
                world=situation['world'], arm=situation['arm'], seed=job['seed'], salt=job['salt'],
                tag=job['tag'], job_sha256=job['job_sha256'], verification_id=rec['verification_id'],
                rec_status=rec['status'], branches_dir=str(work/'branches'))
    rows = []
    if rec['status'] not in ('accepted', 'rejected'):
        for g in job['groups']:
            rows.append(dict(base, plan_sha256=digest(g), plan=g, status=rec['status'],
                             reason=rec.get('reason')))
    else:
        br = rec['branches']
        idx = {c['id']: i+1 for i, c in enumerate(cands)}
        tapes = {}

        def tape(i):
            if i not in tapes:
                tapes[i] = read_result(work/'branches', rec['verification_id'], br[i])
            return tapes[i]
        Jp = rec['J_policy_veh_s']
        for g in job['groups']:
            gi = idx[g['id']]
            Jg = br[gi]['J_veh_s']
            row = dict(base, plan_sha256=digest(g), plan=g, group_id=g['id'], members=g['members'],
                       J_policy=Jp, J_group=Jg, adv=None if Jg is None else Jp-Jg, adv_lb95=None,
                       safety={k: (None if Jg is None else float(br[gi]['window'][k])-float(br[0]['window'][k]))
                               for k in rb.SAFETY_KEYS}, status='graded')
            if Jg is None:
                row['status'] = 'failed'
                rows.append(row)
                continue
            try:
                row['adv_lb95'] = stats([(tape(0), tape(gi), rec['window'])])['lower95_one_sided_veh_s']
                if job['mode'] == 'synergy':
                    sj = {s: br[idx[s]]['J_veh_s'] for s in job['owner'][g['id']]}
                    row['J_solos'] = sj
                    if any(v is None for v in sj.values()):
                        row.update(status='incomplete', synergy=None, synergy_lb95=None)
                    else:
                        best = min(sj, key=lambda k: (sj[k], k))
                        st = stats([(tape(idx[best]), tape(gi), rec['window'])])
                        row.update(best_solo=best, synergy=sj[best]-Jg,
                                   synergy_lb95=st['lower95_one_sided_veh_s'])
            except Exception as error:     # tape missing: never imputed
                row.update(status='data_missing', error=repr(error))
            rows.append(row)
    path = Path(root)/'results'/f"{job['stage']}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True)+'\n')
    return rows


def run_plan(root, plan_path, workers=8, **kwargs):
    plan = json.loads(_guard(plan_path).read_text(encoding='utf-8'))
    if plan['plan_sha256'] != digest({k: v for k, v in plan.items() if k != 'plan_sha256'}):
        raise ValueError('stage plan changed after it was built')
    out = []
    for job in plan['jobs']:
        if (Path(root)/'STOP').exists():
            break
        claim = Path(root)/'claims'/(job['tag']+'.claim')
        claim.parent.mkdir(parents=True, exist_ok=True)
        try:
            with claim.open('x') as f:
                f.write(str(os.getpid()))
        except FileExistsError:
            continue
        out += run_job(root, job, workers=workers, **kwargs)
    return out


def run_confirm(root, confirm_path, workers=8):
    """Native all8 confirmation of the refined wall (probe_native.run semantics, own ledger)."""
    import torch
    from . import replay_bank as rb, sumo_verify as sv, verify_perseed as vp
    from .storage import write as swrite
    plan = json.loads(_guard(confirm_path).read_text(encoding='utf-8'))
    job = plan['jobs'][0]
    if job.get('schema') != SCHEMA_CONFIRM or job['job_sha256'] != digest(
            {k: v for k, v in job.items() if k != 'job_sha256'}):
        raise ValueError('confirmation job changed after it was built')
    situation = job['situation']
    arm, sid = situation['arm'], situation['state_id']
    snap = _snapshot(situation)
    bank_dir = Path(root)/'minebank'/'bank'
    bank = Path(rb.bank_root(bank_dir, arm))
    (bank/'snapshots').mkdir(parents=True, exist_ok=True)
    if not (bank/'snapshots'/(sid+'.pt')).exists():
        torch.save(snap, bank/'snapshots'/(sid+'.pt'))
        swrite(bank/'snapshots'/(sid+'.json'), dict(state_id=sid, arm=arm, job=sid.rsplit('_t', 1)[0],
               world=situation['world'], t0=int(situation['t0']), group=None, regime=None, policy_update=0,
               record_sha256=situation['snapshot_sha256'], unix=time.time(),
               wall_tune=dict(situation_id=situation['situation_id'], job_sha256=job['job_sha256'])))

    def charger(kind):
        return lambda vid, specs: charge(root, job, f"{job['tag']}_{kind}_{vid}",
                                         sum(int(s['sim_s']) for s in specs), kind)
    runner = sv.SubprocessBranchRunner(bank/'sumo_verification'/'branches', workers=workers, timeout_s=3600)
    r = sv.verify(snap, job['plans'], runner, charger('first'), 'wm', workers=workers,
                  root=bank/'sumo_verification', tail_s=sv.TAIL_S)
    (bank/'decisions').mkdir(exist_ok=True)
    _write(bank/'decisions'/(sid+'.json'), dict(state_id=sid, verification_id=r['verification_id'],
           status=r['status'], policy_update=0, arm=arm, available_k=len(job['plans']),
           issued_unix=time.time(), matched_B=None))
    out = dict(state_id=sid, tag=job['tag'], verification_id=r['verification_id'], first_status=r['status'])
    if r['status'] not in ('accepted', 'rejected'):
        return dict(out, status='first_'+r['status'], reason=r.get('reason'))
    rb._confirm_charge = lambda run_root, phase, kind: charger(kind)   # this process only (as the probe)
    receipt = dict(r, _branches=str(bank/'sumo_verification'/'branches'), _policy_version='u0')
    conf = vp.confirm_all8(bank_dir, arm, receipt, snap,
                           lambda work: sv.SubprocessBranchRunner(Path(work)/'branches', workers=workers,
                                                                  timeout_s=3600),
                           str(bank_dir), PHASE, workers=workers, version='u0', candidates=job['plans'],
                           slim_rejected_tapes=False, extra_seeds=int(job['extra_seeds']))
    return dict(out, status='confirmed', admitted=conf['admitted'],
                confirmation_sha256=conf['confirmation_sha256'])


# ------------------------------------------------------------------ emit (MINED_MOVES + WALL_NEGATIVES)

def _example_plan(e, plans):
    return e.get('plan') or plans.get(e.get('plan_id'))


def emit(bank_dir, arm, out_dir, *, source, stage_rows=(), frames_reader=None):
    """Native per-seed examples -> MINED_MOVES rows (positives) + WALL_NEGATIVES rows.

    Positive (probe extract rule): admitted (LB95 vs policy > 0, own-seed safety, later harm)
    AND cooperative (own-seed synergy LB95 > 0) AND no group-window collision/teleport, wall
    family, >= 2 members. Negative: wall_negatives.RULE on the example's own seed. Optional
    ``stage_rows``: tuning refinement rows (synergy measured per seed) also yield negatives.
    """
    from . import moves, replay_bank as rb, verify_perseed as vp, wall_negatives as wn
    from .exact_grade import verify_sealed
    frames_reader = frames_reader or wn.frames_from_trace
    bank = Path(rb.bank_root(bank_dir, arm))
    mined, negatives, failures, examples = [], [], [], 0
    for f in sorted((bank/vp.RECORDS).glob('*.json')):
        c = json.loads(f.read_text(encoding='utf-8'))
        verify_sealed(c, 'confirmation_sha256')
        vid = c['verification_id']
        plans = {}
        for s in (bank/'sumo_verification'/'branches'/vid).glob('cand*.spec.json'):
            p = json.loads(s.read_text(encoding='utf-8'))['plan']
            plans[p['id']] = p
        first = json.loads((bank/'sumo_verification'/'verifications'/(vid+'.json')).read_text(encoding='utf-8'))
        ep = first.get('episode') or {}
        situation = f"{ep.get('id')}_t{c['t0']}"
        exs = c['per_seed_v1']['examples']
        n_reps = 2+max([0]+[int(e['seed_index'])-2 for e in exs])
        for e in exs:
            examples += 1
            plan = _example_plan(e, plans)
            try:
                move = moves.from_plan(plan) if plan else None
            except ValueError as error:
                failures.append(dict(example_id=e.get('example_id'), reason=str(error)[:200]))
                continue
            if move is None or move.family != 'wall' or plan.get('family', 'wall') != 'wall' \
                    or len(plan['members']) < 2:
                continue
            seed = int(e['seed_index'])
            syn = e.get('synergy') or {}
            win = (e.get('tape') or {}).get('branch', {}).get('window') or {}
            ok = (e.get('admitted') and e.get('cooperative') is True and e.get('safety_ok') is not False
                  and not win.get('collisions') and not win.get('teleports'))
            if ok:
                s_ = str(seed)
                row = dict(schema='coopmwm-mined-move-v1', arm=arm, run_root=str(bank_dir),
                           verification_id=vid, plan_sha256=digest(plan), replay_sources={}, seed=seed,
                           situation_id=situation, world=ep.get('world'), t0=c['t0'], family='wall',
                           members=plan['members'], duration_s=plan['duration_s'], profile=plan['profile'],
                           policy_profile=plan['policy_profile'],
                           perturbation_profile=plan['perturbation_profile'],
                           synergy_veh_s={s_: syn.get('pooled_total_veh_s')},
                           synergy_lower95={s_: syn.get('lower95_one_sided_veh_s')},
                           advantage_veh_s={s_: e.get('advantage_veh_s')},
                           adv_lower95={s_: e.get('lower95_one_sided_veh_s')},
                           best_solo=syn.get('best_solo_member'), n_reps=n_reps,
                           statistic=('per-seed one-sided 95% circular-block (30 s) paired bootstrap B=2000, '
                                      'best solo minus group on the common window (verify_perseed._cooperation)'),
                           base='frozen MAPPO u0', source=source)
                if plan.get('wall_macro') is not None:
                    row['wall_macro'] = plan['wall_macro']
                mined.append(row)
                continue
            synergy = syn.get('pooled_total_veh_s') if syn else None
            rule_value = synergy if synergy is not None else e.get('advantage_veh_s')
            if rule_value is None or not rule_value < 0:
                continue
            tape = e.get('tape') or {}
            try:
                if not tape:
                    raise ValueError('example has no own-seed tape')
                trace = Path(tape['branches_dir'])/tape['verification_id']/'policy'/'observable.jsonl.gz'
                frames = frames_reader(trace, int(c['t0']))
                neg = wn.example(arm=arm, situation_id=situation, verification_id=vid, plan=plan, seed=seed,
                                 t0=int(c['t0']), world=ep.get('world') or {}, frames=frames, source=source,
                                 synergy=synergy,
                                 synergy_lower95=syn.get('lower95_one_sided_veh_s') if syn else None,
                                 advantage=e.get('advantage_veh_s'))
                if neg is not None:
                    negatives.append(neg)
            except (ValueError, KeyError, OSError, TypeError) as error:
                failures.append(dict(example_id=e.get('example_id'), reason=str(error)[:200]))
    for r in stage_rows:
        if r.get('arm') != arm or r.get('mode') != 'synergy' or r.get('status') != 'graded' \
                or r.get('synergy') is None or not r['synergy'] < 0:
            continue
        try:
            trace = Path(r['branches_dir'])/r['verification_id']/'policy'/'observable.jsonl.gz'
            neg = wn.example(arm=arm, situation_id=r['situation_id'], verification_id=r['verification_id'],
                             plan=r['plan'], seed=int(r['seed']), t0=int(r['t0']), world=r['world'],
                             frames=frames_reader(trace, int(r['t0'])), source=source+f":{r['stage']}",
                             synergy=r['synergy'], synergy_lower95=r.get('synergy_lb95'), advantage=r.get('adv'))
            if neg is not None:
                negatives.append(neg)
        except (ValueError, KeyError, OSError, TypeError) as error:
            failures.append(dict(tag=r.get('tag'), reason=str(error)[:200]))
    out_dir = _guard(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir/f'MINED_MOVES_{arm}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in mined), encoding='utf-8')
    (out_dir/f'WALL_NEGATIVES_{arm}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in negatives),
                                                       encoding='utf-8')
    receipt = dict(arm=arm, bank=str(bank_dir), examples=examples, mined=len(mined), negatives=len(negatives),
                   negative_rule=wn.RULE, failures=failures, source=source, unix=time.time())
    _write(out_dir/f'EMIT_RECEIPT_{arm}.json', receipt)
    return mined, negatives, receipt


# ------------------------------------------------------------------ CLI

def main(argv):
    cmd = argv[0]
    if cmd == 'quote':
        t0 = int(argv[1]) if len(argv) > 1 else 455
        n = int(argv[2]) if len(argv) > 2 else 4
        print(json.dumps(quote_situation(t0, n), indent=1))
    elif cmd == 'init':
        print(json.dumps(init(argv[1], int(argv[2]) if len(argv) > 2 else 250000), indent=1))
    elif cmd == 'screen':
        path, body = build_screen(argv[1], load_situation(argv[2]))
        print(json.dumps(dict(plan=str(path), jobs=len(body['jobs']), quote_sim_s=body['quote_sim_s'])))
    elif cmd == 'refine':
        path, body = build_refine(argv[1], load_situation(argv[2]), int(argv[3]) if len(argv) > 3 else 1)
        print(json.dumps(dict(plan=str(path), jobs=len(body['jobs']), quote_sim_s=body['quote_sim_s'])))
    elif cmd == 'confirm-job':
        path, body = build_confirm(argv[1], load_situation(argv[2]))
        print(json.dumps(dict(plan=str(path), quote_sim_s=body['quote_sim_s'])))
    elif cmd == 'run':
        rows = run_plan(argv[1], argv[2], int(argv[3]) if len(argv) > 3 else 8)
        print(json.dumps(dict(rows=len(rows))))
    elif cmd == 'confirm':
        print(json.dumps(run_confirm(argv[1], argv[2], int(argv[3]) if len(argv) > 3 else 8)))
    elif cmd == 'emit':
        root, bank, arm = argv[1], argv[2], argv[3]
        stage = []
        if '--stage-negatives' in argv:
            path = Path(root)/'results'/'refine.jsonl'
            if path.exists():
                stage = [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]
        _, _, receipt = emit(bank, arm, Path(root)/'emit', source=f'wall_tune emit {bank}', stage_rows=stage)
        print(json.dumps({k: receipt[k] for k in ('arm', 'examples', 'mined', 'negatives')}))
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main(sys.argv[1:])
