"""W3 label builder: SUMO-measured (situation, joint move) outcomes from the mining / coop_probe
ledgers, joined to the observable situation, the WM query and the WM's own prediction.

Sources discovered under each --root (forbidden directories are pruned before listing):
  results/*.jsonl                       coop_probe screen rows (tools/mining/probe.verify): J_policy,
                                        J_group, J_solos, synergy, synergy_lb95, synergy_status
  **/confirmations_all8/*.json          native per_seed_v1 records (verify_perseed.confirm_all8)
  **/MINED_EXAMPLES*.jsonl              flat per-seed examples (probe_native.extract / mining)
  **/CONTEXT.json                       collected observable situations (CAV-sensed rows only)
  **/*.query.json, **/*.rank.json       WM queries (record + 60 s history) and WM rankings
  **/jobs/**.json                       probe jobs/slates (ctx_dir, plans, WM top-pick predictions)

Every world seed is checked (guard.check_seed): a record from 962xxx / 964000-964199 / 977xxx or
any non-TRAIN world REFUSES the whole build (ForbiddenSeedError; nothing is written).

Units: one SEED ROW per (situation, move, CRN seed identity). Seed 1 is the collection CRN shared
by every source ('crn1'); replicate seeds are identified by their salt (probe 'coopprobe#..#rN',
all8 'all8:<vid>:<k>'), so different replicate streams are never merged by index. Pairs aggregate
their seed rows; the split is by WORLD seed (all situations of a world share a fold).
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import time

from . import guard
from .features import move_from_action, move_from_plan, move_key, normalize_move, parse_plan_id

SCHEMA = 'wallfit-labels-v1'
FIELDS = ('vid', 's_m', 'v_mps', 'lane', 'is_cav', 'length_m')
DEFAULT_TAIL_S = 120
DEFAULT_EXCLUDE_SETTINGS = ('lowcoop',)     # owner-approved probe-only HDV variant: a hidden world label
SPLIT_SALT = 'wallfit-w3-split-v1'
SOURCE_PRIORITY = {'all8': 0, 'flat': 1, 'probe': 2}
_VID_RE = re.compile(r'^sv_(.+)_t(\d+)_([0-9a-f]{8})$')
_SID_SEED_RE = re.compile(r'_(\d{6})_t(\d+)$')


def state_id_from_vid(vid):
    """'sv_cp_hp_R_train_986067_t0340_9b768543' -> ('cp_hp_R_train_986067_t340', 340)."""
    m = _VID_RE.match(str(vid))
    if not m:
        raise ValueError(f'unparseable verification id {vid!r}')
    return f'{m.group(1)}_t{int(m.group(2))}', int(m.group(2))


def world_seed_from_state_id(sid):
    m = _SID_SEED_RE.search(str(sid))
    return int(m.group(1)) if m else None


def _read_json(p):
    guard.check_path(p)
    return json.loads(Path(p).read_text())


def _aux_json(p, bad):
    """Auxiliary inputs (contexts / queries / ranks / jobs): unreadable files are counted, not fatal.
    Label files never use this (a corrupt label file fails the build)."""
    try:
        return _read_json(p)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        if bad is None:
            raise
        bad.append(str(p))
        return None


def _read_jsonl(p):
    guard.check_path(p)
    out = []
    for line in Path(p).read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _f(x):
    return None if x is None else float(x)


# ------------------------------------------------------------------ seed-row adapters

def _seed_row(**kw):
    row = dict(source=None, source_path=None, tag=None, state_id=None, world_seed=None, t0=None, setting=None, arm=None,
               plan_id=None, move=None, seed_id=None, seed_index=None, tail_s=None, J_policy=None, J_group=None,
               J_solos=None, group_adv=None, solo_advs=None, best_solo_adv=None, synergy=None, synergy_lb95=None,
               adv_lb95=None, admitted=None, status=None)
    unknown = set(kw) - set(row)
    if unknown:
        raise KeyError(f'unknown seed-row keys {sorted(unknown)}')
    row.update(kw)
    return row


def _solo_member(key, members):
    hits = [m for m in members if f'solo_{m}' in key or key.endswith(m)]
    return hits[0] if len(hits) == 1 else None


def from_probe_row(row, path, plans=None):
    """tools/mining/probe.verify result row -> seed row (or (None, skip_reason))."""
    if row.get('error') is not None or row.get('status') not in ('accepted', 'rejected') or 'group_id' not in row:
        return None, 'probe_row_without_outcome'
    seed = guard.check_seed(row['world']['seed'], where=f'({path})')
    plan = (plans or {}).get((row.get('tag'), row['group_id']))
    move = move_from_plan(plan) if plan else normalize_move(parse_plan_id(row['group_id']))
    if row.get('family'):
        move['family'] = row['family']
    if row.get('members'):
        move['members'] = [str(v) for v in row['members']]
    if len(move['members']) < 2:
        return None, 'single_member_move'
    Jp, Jg, Js = _f(row.get('J_policy')), _f(row.get('J_group')), row.get('J_solos') or {}
    solo_advs = {}
    for i, (key, J) in enumerate(Js.items()):
        m = _solo_member(key, move['members']) or (move['members'][i] if len(Js) == len(move['members']) else None)
        if m is not None and J is not None and Jp is not None:
            solo_advs[m] = Jp - float(J)
    status = row.get('synergy_status')
    admitted = True if status == 'admitted' else False if status == 'not_admitted' else None
    synergy = _f(row.get('synergy')) if admitted is not None else None
    k, salt = int(row.get('seed_index') or 1), row.get('salt')
    seed_id = 'crn1' if k == 1 and not salt else f'salt:{salt}' if salt else f'probe:{row.get("tag")}:{k}'
    group_adv = _f(row.get('group_adv')) if row.get('group_adv') is not None else (
        Jp - Jg if Jp is not None and Jg is not None else None)
    best = _f(row.get('best_solo_adv')) if row.get('best_solo_adv') is not None else (
        max(solo_advs.values()) if solo_advs else None)
    return _seed_row(source='probe', source_path=str(path), tag=row.get('tag'), state_id=row['state_id'],
                     world_seed=seed,
                     t0=int(row['t0']), setting=row.get('setting'), arm=row.get('arm'), plan_id=row['group_id'],
                     move=move, seed_id=seed_id, seed_index=k, tail_s=(row.get('window') or {}).get('tail_s'),
                     J_policy=Jp, J_group=Jg, J_solos={k_: _f(v) for k_, v in Js.items()}, group_adv=group_adv,
                     solo_advs=solo_advs or None, best_solo_adv=best, synergy=synergy,
                     synergy_lb95=_f(row.get('synergy_lb95')) if synergy is not None else None,
                     adv_lb95=_f(row.get('adv_lb95')), admitted=admitted, status=status), None


def _all8_admission(ex):
    """(admitted, synergy, lb95, status) under the native per_seed_v1 cooperative rule."""
    cstat = ex.get('cooperative_status')
    syn = ex.get('synergy') if isinstance(ex.get('synergy'), dict) else None
    if ex.get('status') == 'data_missing':
        return None, None, None, 'data_missing'
    if not ex.get('admitted'):
        return False, None, None, 'not_admitted_vs_policy'
    if cstat in ('cooperative', 'not_cooperative') and syn is not None:
        return cstat == 'cooperative', _f(syn.get('pooled_total_veh_s')), _f(syn.get('lower95_one_sided_veh_s')), cstat
    return None, None, None, cstat or 'cooperation_unmeasured'


def from_all8_record(rec, path):
    """confirmations_all8/<vid>.json -> [seed rows], skip reasons."""
    vid = rec['verification_id']
    sid, t0 = state_id_from_vid(vid)
    seed = guard.check_seed(world_seed_from_state_id(sid), where=f'({path})')
    rows, skips = [], []
    for ex in (rec.get('per_seed_v1') or {}).get('examples') or []:
        plan = ex.get('plan') or {}
        move = move_from_plan(plan) if plan else normalize_move(parse_plan_id(ex.get('plan_id')))
        if len(move['members']) < 2:
            skips.append('single_member_move')
            continue
        k = int(ex['seed_index'])
        admitted, synergy, lb, status = _all8_admission(ex)
        advs = ex.get('solo_advantages_veh_s') or []
        solo = {m: float(a) for m, a in zip(move['members'], advs) if a is not None} if len(advs) == len(move['members']) else {}
        window = ((ex.get('tape') or {}).get('window') or {})
        rows.append(_seed_row(source='all8', source_path=str(path), state_id=sid, world_seed=seed,
                              t0=int(rec.get('t0', t0)), setting='native', arm=rec.get('arm'),
                              plan_id=ex.get('plan_id') or plan.get('id'), move=move,
                              seed_id='crn1' if k == 1 else f'all8:{vid}:{k}', seed_index=k,
                              tail_s=window.get('tail_s'), group_adv=_f(ex.get('advantage_veh_s')),
                              solo_advs=solo or None, best_solo_adv=max(solo.values()) if solo else None,
                              synergy=synergy, synergy_lb95=lb, adv_lb95=_f(ex.get('lower95_one_sided_veh_s')),
                              admitted=admitted, status=status))
    return rows, skips


def from_flat_example(row, path):
    """MINED_EXAMPLES-style flat example (probe_native.extract) -> seed row."""
    if 'verification_id' not in row or 'seed' not in row:
        return None, 'flat_row_without_verification'
    vid = row['verification_id']
    sid = row.get('situation_id') or row.get('state_id') or state_id_from_vid(vid)[0]
    wseed = (row.get('world') or {}).get('seed', world_seed_from_state_id(sid))
    seed = guard.check_seed(wseed, where=f'({path})')
    move = normalize_move(dict(parse_plan_id(row.get('plan_id')), **{k: row[k] for k in ('family', 'members') if row.get(k)}))
    if len(move['members']) < 2:
        return None, 'single_member_move'
    k = int(row['seed'])
    if row.get('status') == 'data_missing':
        admitted = None
    else:
        admitted = bool(row.get('cooperative')) if row.get('admitted') else False
    synergy = _f(row.get('synergy_veh_s')) if row.get('admitted') else None
    return _seed_row(source='flat', source_path=str(path), state_id=sid, world_seed=seed, t0=int(row.get('t0') or 0),
                     setting='native', arm=row.get('arm'), plan_id=row.get('plan_id'), move=move,
                     seed_id='crn1' if k == 1 else f'all8:{vid}:{k}', seed_index=k,
                     group_adv=_f(row.get('advantage_veh_s')), synergy=synergy,
                     synergy_lb95=_f(row.get('synergy_lower95')) if synergy is not None else None,
                     adv_lb95=_f(row.get('adv_lower95')), admitted=admitted, status=row.get('status')), None


# ------------------------------------------------------------------ discovery

def _kind(path):
    name = path.name
    parts = [p.lower() for p in path.parts]
    if name == 'CONTEXT.json':
        return 'context'
    if name.endswith('.query.json'):
        return 'query'
    if name.endswith('.rank.json'):
        return 'rank'
    if name.startswith('MINED_EXAMPLES') and name.endswith('.jsonl'):
        return 'flat'
    if name.endswith('.json') and len(parts) > 1 and parts[-2] == 'confirmations_all8':
        return 'all8'
    if name.endswith('.jsonl') and len(parts) > 1 and parts[-2] == 'results':
        return 'results'
    if name.endswith('.json') and 'jobs' in parts[:-1]:
        return 'job'
    return None


def discover(roots, skipped=None):
    found = {k: [] for k in ('context', 'query', 'rank', 'flat', 'all8', 'results', 'job')}
    for root in roots:
        for p in guard.walk_files(root, lambda n: n.endswith('.json') or n.endswith('.jsonl'), skipped):
            k = _kind(p)
            if k:
                found[k].append(p)
    return found


def _rows(rows):
    return [{k: r.get(k) for k in FIELDS} for r in rows]


def index_contexts(paths, bad=None):
    out = {}
    for p in paths:
        c = _aux_json(p, bad)
        if not isinstance(c, dict) or 'state_id' not in c or 'rows' not in c:
            continue
        sid = c['state_id']
        seed = guard.check_seed((c.get('world') or {}).get('seed', world_seed_from_state_id(sid)), where=f'({p})')
        out.setdefault(sid, dict(path=str(p), world_seed=seed, t0=int(c['t0']), rows=_rows(c['rows'])))
    return out


def index_queries(paths, bad=None):
    out = {}
    for p in paths:
        d = _aux_json(p, bad)
        if not isinstance(d, dict) or 'record' not in d.get('query', d):
            continue
        q = d.get('query', d)
        rec = q['record']
        seed = guard.check_seed(rec.get('world_id', world_seed_from_state_id(rec['state_id'])), where=f'({p})')
        qid = p.name[:-len('.query.json')]
        out[qid] = dict(path=str(p), query_id=qid, state_id=rec['state_id'], world_seed=seed, query=q,
                        vids={str(r['vid']) for r in rec['rows']}, actions=d.get('actions') or [],
                        groups=d.get('groups') or {}, H=d.get('H'))
    return out


def index_ranks(paths, queries, bad=None):
    """{(state_id, move_key): [wm prediction dicts]} from probe (group/plan_id) or mining lane ranks."""
    out = {}
    for p in paths:
        d = _aux_json(p, bad)
        if not isinstance(d, dict):
            continue
        stem = p.name[:-len('.rank.json')]
        q = queries.get(stem)
        sid = q['state_id'] if q else d.get('state_id') or stem.split('__')[-2 if '__' in stem else 0]
        if world_seed_from_state_id(sid) is not None:
            guard.check_seed(world_seed_from_state_id(sid), where=f'({p})')
        horizon = d.get('synergy_horizon_s') or d.get('H') or (q or {}).get('H') or 120
        acts = {a['id']: a for a in (q or {}).get('actions', [])}
        for r in d.get('ranked') or []:
            if r.get('wm_mean_synergy') is None:
                continue
            pid = r.get('plan_id') or ((q or {}).get('groups', {}).get(r.get('group'), {}) or {}).get('plan_id')
            if r.get('group') in acts:
                move = move_from_action(acts[r['group']], pid)
            elif r.get('members'):
                move = normalize_move(dict(family=r.get('family'), members=r['members'], onset=r.get('onset'),
                                           hold=r.get('hold'), depth=r.get('depth')))
            elif pid:
                move = normalize_move(parse_plan_id(pid))
            else:
                continue
            out.setdefault((_alias(sid), move_key(move)), []).append(dict(
                mean_synergy=float(r['wm_mean_synergy']), saving_vs_policy=_f(r.get('wm_saving_vs_policy')),
                best_solo_saving=_f(r.get('wm_best_solo_saving')), draws=r.get('wm_synergy_draws'),
                horizon_s=int(horizon), source='rank', rank_path=str(p), query_id=stem if q else None))
    return out


def _alias(sid):
    """Mining rank files may drop the 'op2_collect01_' collection prefix from the state id."""
    return str(sid).replace('op2_collect01_', '')


def index_jobs(paths, root_of, bad=None):
    """plans by (tag, plan_id); ctx_dir by tag; WM top-pick predictions by (tag, plan_id)."""
    plans, ctx, wm, slates = {}, {}, {}, {}
    for p in paths:
        d = _aux_json(p, bad)
        items = d if isinstance(d, list) else [d]
        for j in items:
            if not isinstance(j, dict):
                continue
            if 'ctx_dir' in j and 'plans' in j and 'tag' in j:
                for pl in j['plans']:
                    if isinstance(pl, dict) and 'id' in pl:
                        plans[(j['tag'], pl['id'])] = pl
                cd = Path(j['ctx_dir'])
                ctx[j['tag']] = cd if cd.is_absolute() else root_of(p) / cd
                w = j.get('wm') or {}
                if w.get('mean_synergy') is not None:
                    pid = w.get('plan_id') or (j['plans'][0]['id'] if len(j['plans']) == 1 else None)
                    if pid:
                        wm[(j['tag'], pid)] = dict(mean_synergy=float(w['mean_synergy']),
                                                   saving_vs_policy=_f(w.get('saving')),
                                                   best_solo_saving=_f(w.get('best_solo_saving')),
                                                   draws=w.get('draws'), horizon_s=None, source='job')
            elif 'ctx_dir' in j and 'plans' in j and p.name.endswith('.slate.json'):
                slates[p.name[:-len('.slate.json')]] = j
    return plans, ctx, wm, slates


# ------------------------------------------------------------------ aggregation / split

def _mean(xs):
    xs = [float(x) for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def aggregate(seed_rows):
    measured = [r for r in seed_rows if r['synergy'] is not None]
    determined = [r for r in seed_rows if r['admitted'] is not None]
    crn1 = [r for r in seed_rows if r['seed_id'] == 'crn1']
    return dict(n_seeds=len(seed_rows), n_measured=len(measured), n_determined=len(determined),
                synergy=_mean(r['synergy'] for r in measured),
                synergy_crn1=crn1[0]['synergy'] if crn1 else None,
                synergy_lb95_max=max((r['synergy_lb95'] for r in measured if r['synergy_lb95'] is not None), default=None),
                admitted_any=any(r['admitted'] for r in determined) if determined else None,
                admitted_rate=(sum(bool(r['admitted']) for r in determined) / len(determined)) if determined else None,
                group_adv=_mean(r['group_adv'] for r in seed_rows),
                best_solo_adv=_mean(r['best_solo_adv'] for r in seed_rows),
                n_group=sum(r['group_adv'] is not None for r in seed_rows),
                n_solo=sum(r['best_solo_adv'] is not None for r in seed_rows))


def split_by_seed(world_seeds, folds=5, salt=SPLIT_SALT, test_seeds=None):
    """Outcome-blind, deterministic split by WORLD seed: seeds ordered by sha256(salt:seed) and dealt
    round-robin into folds; the test split is fold 0 (or the explicit test_seeds)."""
    seeds = sorted({guard.check_seed(s) for s in world_seeds})
    order = sorted(seeds, key=lambda s: hashlib.sha256(f'{salt}:{s}'.encode()).hexdigest())
    nf = max(1, min(int(folds), len(order)))
    fold = {s: i % nf for i, s in enumerate(order)}
    test = sorted({guard.check_seed(s) for s in test_seeds}) if test_seeds else sorted(s for s in seeds if fold[s] == 0)
    return dict(folds=nf, fold_of={str(s): fold[s] for s in seeds}, test_seeds=test,
                train_seeds=[s for s in seeds if s not in test], salt=salt)


FILL_FIELDS = ('J_policy', 'J_group', 'J_solos', 'group_adv', 'solo_advs', 'best_solo_adv', 'adv_lb95')
SYNERGY_FIELDS = ('synergy', 'synergy_lb95')


def _dedup(seed_rows):
    """One row per (situation, move, CRN seed identity). The highest-priority source (native all8 >
    flat examples > probe screens) keeps its admission label; measured outcomes it lacks (e.g. a
    probe-screen synergy for a group the native rule did not admit vs policy) are filled from the
    duplicates, recorded in `filled_from`."""
    best, conflicts = {}, 0
    for r in sorted(seed_rows, key=lambda r: SOURCE_PRIORITY.get(r['source'], 9)):
        key = (r['state_id'], move_key(r['move']), r['seed_id'])
        if key not in best:
            best[key] = dict(r)
            continue
        cur = best[key]
        a, b = cur['synergy'], r['synergy']
        if a is not None and b is not None and abs(a - b) > 1e-3:
            conflicts += 1
        filled = [f for f in FILL_FIELDS if cur.get(f) is None and r.get(f) is not None]
        if cur['synergy'] is None and r['synergy'] is not None:
            filled += list(SYNERGY_FIELDS)
        for f in filled:
            cur[f] = r[f]
        if filled:
            cur['filled_from'] = sorted(set(cur.get('filled_from') or []) | {r['source']})
    return list(best.values()), conflicts


# ------------------------------------------------------------------ build

def owners_for(members, rows):
    """WM packet owners for a rebuilt query: the member pair whose 100 m packets cover the most
    members (probe_wmq.write_wmq rule restricted to this group; ties -> first pair in member order)."""
    s = {str(r['vid']): float(r['s_m']) for r in rows}
    ms = [m for m in members if m in s]
    pairs = [(a, b) for i, a in enumerate(ms) for b in ms[i + 1:]]
    if not pairs:
        return list(members[:2])
    return list(max(pairs, key=lambda p: sum(any(abs(s[v] - s[o]) <= 100. for o in p) for v in ms)))


def load_builder(spec):
    """'module:function' -> callable(ctx_dir, owners) -> WM query (e.g. tools.mining.probe_wmq:build_query)."""
    if not spec:
        return None
    import importlib
    mod, _, fn = spec.partition(':')
    return getattr(importlib.import_module(mod), fn or 'build_query')


def build(roots, out_dir, *, examples=(), tail_s=DEFAULT_TAIL_S, exclude_settings=DEFAULT_EXCLUDE_SETTINGS,
          folds=5, salt=SPLIT_SALT, query_builder=None):
    """Collect, check, join and write LABELS.jsonl + QUERIES.jsonl.gz + BUILD_RECEIPT.json.
    Raises ForbiddenSeedError (and writes nothing) if any record references a refused seed.
    query_builder (optional, callable(ctx_dir, owners)): rebuilds the causal WM query for pairs that
    have a collection context but no saved WM query, so they get the same situation rows and WM latent
    as at ranking time; failures are counted and fall back to the context rows."""
    begin = time.time()
    out_dir = guard.check_path(out_dir)
    roots = [guard.check_path(r) for r in roots]
    skipped_paths, skips = [], {}

    def skip(reason, n=1):
        skips[reason] = skips.get(reason, 0) + n
    found = discover(roots, skipped_paths)
    found['flat'] += [guard.check_path(e) for e in examples]

    def root_of(p):
        for r in roots:
            try:
                Path(p).relative_to(r)
                return r
            except ValueError:
                pass
        return Path(p).parent
    bad = []
    contexts = index_contexts(found['context'], bad)
    queries = index_queries(found['query'], bad)
    ranks = index_ranks(found['rank'], queries, bad)
    plans, ctx_by_tag, job_wm, slates = index_jobs(found['job'], root_of, bad)
    for tag, cd in ctx_by_tag.items():          # job ctx_dirs outside the scanned roots
        p = Path(cd) / 'CONTEXT.json'
        if guard.path_allowed(p) and p.exists():
            for sid, c in index_contexts([p], bad).items():
                contexts.setdefault(sid, c)

    seed_rows = []
    for p in found['results']:
        for row in _read_jsonl(p):
            r, why = from_probe_row(row, p, plans)
            (seed_rows.append(r) if r else skip(why))
    for p in found['all8']:
        rows, why = from_all8_record(_read_json(p), p)
        seed_rows += rows
        for w in why:
            skip(w)
    for p in found['flat']:
        for row in _read_jsonl(p):
            r, why = from_flat_example(row, p)
            (seed_rows.append(r) if r else skip(why))
    n_raw = len(seed_rows)
    kept = []
    for r in seed_rows:
        if r['setting'] in exclude_settings:
            skip(f'excluded_setting_{r["setting"]}')
        elif tail_s is not None and r['tail_s'] is not None and int(r['tail_s']) != int(tail_s):
            skip(f'tail_s_{r["tail_s"]}')
        else:
            kept.append(r)
    n_filtered = len(kept)
    kept, conflicts = _dedup(kept)
    if n_filtered > len(kept):
        skip('duplicate_seed_rows', n_filtered - len(kept))

    pairs = {}
    for r in kept:
        pairs.setdefault((r['state_id'], move_key(r['move'])), []).append(r)
    out_rows, used_queries = [], {}
    for (sid, mk), rows in sorted(pairs.items()):
        seeds = {r['world_seed'] for r in rows}
        if len(seeds) != 1:
            raise ValueError(f'pair {sid} spans world seeds {sorted(seeds)}')
        move = rows[0]['move']
        cands = [q for q in queries.values() if q['state_id'] == sid and set(move['members']) <= q['vids']]
        wm_rank = ranks.get((_alias(sid), mk), [])
        ranked_q = {w['query_id'] for w in wm_rank}
        cands.sort(key=lambda q: (q['query_id'] not in ranked_q, -len(q['vids']), q['query_id']))
        if not cands and query_builder is not None and sid in contexts:
            own = owners_for(move['members'], contexts[sid]['rows'])
            try:
                built = query_builder(Path(contexts[sid]['path']).parent, own)
                rec = built['record']
                bseed = guard.check_seed(rec.get('world_id', rows[0]['world_seed']), where=f'(rebuilt {sid})')
                if rec['state_id'] != sid or bseed != rows[0]['world_seed']:
                    raise ValueError('rebuilt query belongs to another situation')
                qid = f'built__{sid}__{"+".join(own)}'
                vids = {str(r['vid']) for r in rec['rows']}
                queries[qid] = dict(path=None, query_id=qid, state_id=sid, world_seed=bseed, query=built, vids=vids,
                                    actions=[], groups={}, H=None)
                if set(move['members']) <= vids:
                    cands = [queries[qid]]
                else:
                    skip('rebuilt_query_missing_members')
            except guard.ForbiddenSeedError:
                raise
            except Exception:
                skip('query_build_error')
        if cands:
            q = cands[0]
            situation = dict(source='query', query_id=q['query_id'], rows=_rows(q['query']['record']['rows']))
            used_queries[q['query_id']] = q
        elif sid in contexts:
            situation = dict(source='context', query_id=None, rows=contexts[sid]['rows'])
        else:
            skip('pair_without_situation', len(rows))
            continue
        if sid in contexts and contexts[sid]['world_seed'] != rows[0]['world_seed']:
            raise ValueError(f'context world seed mismatch for {sid}')
        wm = None
        if wm_rank:
            same = [w for w in wm_rank if w['query_id'] == situation['query_id']]
            wm = dict((same or wm_rank)[0])
        else:   # screening jobs carry the WM's top-pick prediction for the verified plan
            hits = [job_wm[(r['tag'], r['plan_id'])] for r in rows if (r['tag'], r['plan_id']) in job_wm]
            wm = dict(hits[0], horizon_s=hits[0]['horizon_s'] or 120) if hits else None
        labels = aggregate(rows)
        if labels['n_measured'] == 0 and labels['n_determined'] == 0:
            skip('pair_without_outcome', len(rows))
            continue
        out_rows.append(dict(schema=SCHEMA, pair_id=f'{sid}#{mk}', state_id=sid, world_seed=rows[0]['world_seed'],
                             t0=rows[0]['t0'], settings=sorted({str(r['setting']) for r in rows}), move=move,
                             plan_ids=sorted({str(r['plan_id']) for r in rows}), situation=situation, wm=wm,
                             labels=labels, seeds=rows))
    split = split_by_seed([r['world_seed'] for r in out_rows], folds, salt) if out_rows else None
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_path = out_dir / 'LABELS.jsonl'
    _write_text(labels_path, ''.join(json.dumps(r, sort_keys=True) + '\n' for r in out_rows))
    qpath = out_dir / 'QUERIES.jsonl.gz'
    payload = ''.join(json.dumps(dict(query_id=k, state_id=q['state_id'], world_seed=q['world_seed'],
                                      query=q['query']), sort_keys=True) + '\n' for k, q in sorted(used_queries.items()))
    tmp = qpath.with_suffix('.gz.tmp')
    with gzip.open(tmp, 'wt') as f:
        f.write(payload)
    os.replace(tmp, qpath)
    receipt = dict(schema=SCHEMA, roots=[str(r) for r in roots], examples=[str(e) for e in examples],
                   tail_s=tail_s, exclude_settings=list(exclude_settings),
                   files={k: len(v) for k, v in found.items()}, skipped_forbidden_paths=len(skipped_paths),
                   unreadable_aux_files=bad,
                   seed_rows_raw=n_raw, seed_rows_kept=len(kept), duplicate_conflicts=conflicts, skips=skips,
                   pairs=len(out_rows), pairs_with_query=sum(r['situation']['source'] == 'query' for r in out_rows),
                   pairs_with_wm=sum(r['wm'] is not None for r in out_rows),
                   pairs_measured=sum(r['labels']['n_measured'] > 0 for r in out_rows),
                   pairs_admitted=sum(bool(r['labels']['admitted_any']) for r in out_rows),
                   world_seeds=sorted({r['world_seed'] for r in out_rows}), split_preview=split,
                   queries=len(used_queries), query_builder=getattr(query_builder, '__module__', None),
                   sha256=dict(labels=_sha(labels_path), queries=_sha(qpath)),
                   wall_s=time.time() - begin, new_sumo_sim_s=0,
                   sign_convention='advantage = J(policy) - J(move); synergy = J(best solo) - J(group); '
                                   'positive = group better (veh-s)')
    _write_text(out_dir / 'BUILD_RECEIPT.json', json.dumps(receipt, indent=1, sort_keys=True))
    return receipt


def _sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _write_text(p, text):
    p = guard.check_path(p)
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(text)
    os.replace(tmp, p)


# ------------------------------------------------------------------ loading (training / evaluation)

def load_pairs(paths):
    """Merge one or more LABELS.jsonl files at SEED-ROW level (cpuV + gpuA builds may overlap).
    Every world seed is re-checked. Returns pairs with a usable outcome."""
    merged = {}
    for p in paths:
        for r in _read_jsonl(p):
            if r.get('schema') != SCHEMA:
                raise ValueError(f'unknown label schema {r.get("schema")!r} in {p}')
            guard.check_seed(r['world_seed'], where=f'({p})')
            for s in r['seeds']:
                guard.check_seed(s['world_seed'], where=f'({p})')
            cur = merged.get(r['pair_id'])
            if cur is None:
                merged[r['pair_id']] = r
                continue
            cur['seeds'], _ = _dedup(cur['seeds'] + r['seeds'])
            if cur['situation']['source'] != 'query' and r['situation']['source'] == 'query':
                cur['situation'] = r['situation']
            if cur.get('wm') is None and r.get('wm') is not None:
                cur['wm'] = r['wm']
    out = []
    for r in merged.values():
        r['labels'] = aggregate(r['seeds'])
        if r['labels']['n_measured'] or r['labels']['n_determined']:
            out.append(r)
    return sorted(out, key=lambda r: r['pair_id'])


def load_queries(paths):
    out = {}
    for p in paths:
        guard.check_path(p)
        with gzip.open(p, 'rt') as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    guard.check_seed(d['world_seed'], where=f'({p})')
                    out.setdefault(d['query_id'], d)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m tools.wallfit build', description=__doc__.split('\n')[0])
    ap.add_argument('--root', action='append', default=[], help='mining / coop_probe root (repeatable)')
    ap.add_argument('--examples', action='append', default=[], help='extra flat MINED_EXAMPLES jsonl')
    ap.add_argument('--out', required=True)
    ap.add_argument('--tail-s', type=int, default=DEFAULT_TAIL_S)
    ap.add_argument('--include-setting', action='append', default=[],
                    help='re-include an excluded setting (default excluded: lowcoop)')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--query-builder', default=None,
                    help="optional 'module:function' rebuilding WM queries for context-only pairs, e.g. "
                         "tools.mining.probe_wmq:build_query (needs the mining source + torch)")
    a = ap.parse_args(argv)
    excl = tuple(s for s in DEFAULT_EXCLUDE_SETTINGS if s not in a.include_setting)
    r = build(a.root, a.out, examples=a.examples, tail_s=a.tail_s, exclude_settings=excl, folds=a.folds,
              query_builder=load_builder(a.query_builder))
    print(json.dumps({k: r[k] for k in ('pairs', 'pairs_with_query', 'pairs_with_wm', 'pairs_measured',
                                        'pairs_admitted', 'world_seeds', 'skips', 'files')}, sort_keys=True))
    return r
