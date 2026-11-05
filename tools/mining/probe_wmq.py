"""Cooperation probe: WM query/slate builder (runs on cpuV; the GPU ranks it on gpuA via probe_wm)."""
import json
import os
from pathlib import Path

GPUA_U0 = '/root/mining20261106/frozen_u0.pt'


def build_query(ctx_dir, members):
    """Causal observable query exactly as mining wm_lane_worker2.build_query, group = `members`."""
    import gzip, hashlib, torch
    from tools.mining.collect_contexts import FIELDS
    from tools.mining.wm_contract import digest, validate_query, U0_SHA
    ctx_dir = Path(ctx_dir)
    ctx = json.loads((ctx_dir/'CONTEXT.json').read_text())
    episode = ctx_dir.parent
    marker = json.loads((episode/'RESULT.json').read_text())
    raw = gzip.decompress((episode/marker['file']).read_bytes())
    assert hashlib.sha256(raw).hexdigest() == marker['sha256']
    tape = json.loads(raw)
    snap = torch.load(ctx_dir/'SNAPSHOT.pt', weights_only=False, map_location='cpu')
    t0 = int(ctx['t0'])
    history, query = [], None
    for event in tape['steps']:
        if not t0-60 <= event['time_s'] <= t0:
            continue
        by = {r['vid']: r for r in event['rows']}
        if members[0] not in by:
            continue
        packets = {v: dict(time_s=event['time_s'], observed_lanes=[0, 1, 2],
                          rows=[{k: row[k] for k in FIELDS} for row in event['rows']
                                if abs(row['s_m']-by[v]['s_m']) <= 100])
                   for v in members if v in by}
        roster = [{k: row[k] for k in FIELDS} for row in event['rows']
                  if any(abs(row['s_m']-by[v]['s_m']) <= 100 for v in packets)]
        history.append(dict(time_s=event['time_s'], rows=roster, entrance_counts=None))
        if event['time_s'] == t0:
            sample = event['sample']
            own = dict(zip(sample['context']['members'], [float(x[0]) for x in sample['own_action']]))
            cavs = [r['vid'] for r in roster if r['is_cav']]
            record = dict(state_id=ctx['state_id'], world_id=int(ctx['world']['seed']), time_s=t0,
                          rows=roster, group=list(members),
                          commands={v: float(event['commands'][v]) for v in cavs},
                          policy_residuals={v: own[v] for v in cavs}, history=history)
            query = dict(record=record, ego=members[0], sensing_packets=packets,
                         source_sha256=digest(dict(snapshot=ctx['snapshot_sha256'], tape_sha256=marker['sha256'],
                                                   time_s=t0, actors=list(members))),
                         base_continuation=dict(checkpoint=GPUA_U0, checkpoint_sha256=U0_SHA,
                             deterministic=False, generator_state_after_t0=snap['generator_state'].tolist(),
                             memory_after_t0={v: own[v] for v in cavs}))
    if query is None:
        raise ValueError('t0 frame missing from tape')
    return validate_query(query)


def wm_actions(plans):
    """Ease plans -> WM library moves (label 'ramp_assist' = every listed member eases by depth during
    [onset, onset+hold), no zone release: the same offsets SUMO executes); G5 plans -> role schedules."""
    acts, groups, seen = [dict(id='policy'), dict(id='MAPPO')], {}, set()
    for p in plans:
        if p.get('probe'):
            q = p['probe']
            mv = dict(family='ramp_assist', members=list(p['members']), onset=q['onset'], hold=q['hold'],
                      depth=q['depth'])
            gid = 'W'+str(len(groups))
            acts.append(dict(id=gid, wm_move=mv, wm_active_members=list(p['members'])))
            sids = []
            for m in p['members']:
                sid = f"S_{m}_o{q['onset']}_h{q['hold']}_d{q['depth']:g}"
                if sid not in seen:
                    seen.add(sid)
                    acts.append(dict(id=sid, wm_move=mv, wm_active_members=[m]))
                sids.append(sid)
            groups[gid] = dict(plan_id=p['id'], solos=sids)
        elif p.get('g5_move'):
            from tools.mining.wm_lane_schedule import make_schedule, action_set
            g = p['g5_move']
            fam = 'g5_yield_lane' if g['family'] == 'yield_lane' else g['family']
            sch = action_set(make_schedule(fam, g['members'][0], g['members'][1], g['onset'], g['hold'],
                                           g['depth']))
            if sch[0]['id'] in seen:
                continue
            seen.add(sch[0]['id'])
            acts += sch
            groups[sch[0]['id']] = dict(plan_id=p['id'], solos=[a['id'] for a in sch[1:]])
    return acts, groups


def write_wmq(ctx_dir, plans, out):
    acts, groups = wm_actions(plans)
    members = []
    for a in acts:
        for v in (a.get('wm_move') or {}).get('members', [])+(a.get('wm_role_schedule') or {}).get('members', []):
            if v not in members:
                members.append(v)
    ctx = json.loads((Path(ctx_dir)/'CONTEXT.json').read_text())
    s = {r['vid']: r['s_m'] for r in ctx['rows']}
    members = sorted([v for v in members if v in s], key=lambda v: -s[v])
    # WM contract: explicit two-CAV group (packet owners); all other slate members must be in the
    # union of their 100 m packets, otherwise the action is dropped (recorded). Owners = the member
    # pair whose packets fully cover the most groups (then most members), by t0 positions.
    gmembers = [set((a.get('wm_move') or {}).get('members', [])+(a.get('wm_role_schedule') or {}).get('members', []))
                for a in acts if a['id'] in groups]

    def cover(pair):
        ok = {v for v in members if any(abs(s[v]-s[o]) <= 100 for o in pair)}
        return (sum(g <= ok for g in gmembers), len(ok))
    pairs = [(a, b) for i, a in enumerate(members) for b in members[i+1:]
             if s.get(a) is not None and s.get(b) is not None]
    owner = max(pairs, key=cover) if pairs else (members[0], members[-1])
    query = build_query(ctx_dir, list(owner))
    ids = {r['vid'] for r in query['record']['rows']}
    keep, dropped = [], []
    for a in acts:
        mem = (a.get('wm_move') or {}).get('members', [])+(a.get('wm_role_schedule') or {}).get('members', [])
        (keep if set(mem) <= ids else dropped).append(a)
    acts = keep
    kept_ids = {a['id'] for a in acts}
    groups = {g: v for g, v in groups.items() if g in kept_ids and all(x in kept_ids for x in v['solos'])}
    out = Path(out)
    tmp = out.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(query=query, actions=acts, groups=groups,
                                   H=int(os.environ.get('PROBE_WM_H', '120')), ctx_dir=str(ctx_dir),
                                   plans=[p['id'] for p in plans], dropped=[a['id'] for a in dropped])))
    os.replace(tmp, out)
    return out
