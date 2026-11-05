"""Real-GPU WM lane ranking of operator-2 collected contexts (rank only, zero SUMO).

Successor of wm_lane_worker (wm_lane03 FAULT: tools.coopmwm resolved to the
qualified wm_worker01 tree, which lacks moves.py). The import closure is fixed
in the deployed tools/mining/__init__.py by APPENDING the accepted-G5 executor
coopmwm directory after the qualified tree, so every already-qualified module
still resolves first and only missing modules (moves, ...) come from b903c6c2.

Queries are built exactly like collect_contexts + wm_lane_worker.enriched_query:
causal six-field sensing from the episode tape at t <= t0 (60 s history), each
actor's own 100 m packet, live commands/own residuals at t0, and the decision
snapshot's generator state. WM scores are for ranking only.
"""
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

from .collect_contexts import FIELDS, write
from .wm_contract import digest, read_json, validate_query, U0_SHA
from .wm_prescreen import load_base, score_grid, validate_scores
from .wm_lane_adapter import evaluate_lane_grid
from .wm_lane_schedule import make_schedule, action_set

PARENT = Path('/root/mining20261106')
SOURCE = PARENT/'op2_collect01'
DEADLINE_LOCAL = '11:30'


def build_query(ctx_dir):
    import torch
    ctx = read_json(ctx_dir/'CONTEXT.json')
    episode = ctx_dir.parent
    marker = read_json(episode/'RESULT.json')
    raw = gzip.decompress((episode/marker['file']).read_bytes())
    assert hashlib.sha256(raw).hexdigest() == marker['sha256']
    tape = json.loads(raw)
    snap = torch.load(ctx_dir/'SNAPSHOT.pt', weights_only=False, map_location='cpu')
    t0 = int(ctx['t0'])
    assert int(snap['t0']) == t0
    members = list(ctx['group'])
    history, record = [], None
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
                          rows=roster, group=members, commands={v: float(event['commands'][v]) for v in cavs},
                          policy_residuals={v: own[v] for v in cavs}, history=history)
            query = dict(record=record, ego=members[0], sensing_packets=packets,
                         source_sha256=digest(dict(snapshot=ctx['snapshot_sha256'], tape_sha256=marker['sha256'],
                                                   time_s=t0, actors=members)),
                         base_continuation=dict(checkpoint=str(PARENT/'frozen_u0.pt'), checkpoint_sha256=U0_SHA,
                             deterministic=False, generator_state_after_t0=snap['generator_state'].tolist(),
                             memory_after_t0={v: own[v] for v in cavs}))
    if record is None:
        raise ValueError('t0 frame missing from tape')
    return validate_query(query), ctx


RANK_H = 300
RANK_HORIZONS = (60, 120, 300)


def slate(main, ramp):
    """Reduced registered slate (wall-time bound: ~1.7 min per action at K8 on gpuA)."""
    actions = [dict(id='policy'), dict(id='MAPPO')]
    for onset in (0, 10):
        actions += action_set(make_schedule('g5_yield_lane', main, ramp, onset, 20, .3))
    for onset in (0, 5):
        for hold in (20, 40):
            for depth in (.6, 1.):
                actions += action_set(make_schedule('open_gap_merge', main, ramp, onset, hold, depth))
    return actions


def check_scores(scores, actions):
    import math
    ids = [a['id'] for a in actions]
    for h in RANK_HORIZONS:
        values = scores[str(h)]
        assert set(values) == set(ids), 'incomplete candidate scores'
        draws = [v['draw_id'] for v in values['policy']]
        assert len(draws) == 8 and len(set(draws)) == 8
        for key in ids:
            assert [v['draw_id'] for v in values[key]] == draws, 'CRN draw mismatch'
            assert all(math.isfinite(v['team_delay']) for v in values[key])


def rank(scores, actions, main, ramp):
    def mean(values):
        return sum(v['team_delay'] for v in values)/len(values)
    out = []
    for a in actions:
        if 'wm_role_schedule' not in a or 'solo_of' in a:
            continue
        s = a['wm_role_schedule']
        solos = [x['id'] for x in actions if x.get('solo_of') == a['id']]
        row = dict(family='yield_lane' if s['family'] == 'g5_yield_lane' else s['family'],
                   members=[main, ramp], onset=s['onset_s'], hold=s['hold_s'], depth=s['depth'],
                   wm_action_id=a['id'])
        for h in ('120', '300'):
            c = scores[h]
            paired = [min(c[x][k]['team_delay'] for x in solos)-c[a['id']][k]['team_delay'] for k in range(8)]
            row[f'wm_mean_synergy_{h}'] = sum(paired)/8
            row[f'wm_saving_vs_policy_{h}'] = mean(c['policy'])-mean(c[a['id']])
        row['wm_mean_synergy'] = row['wm_mean_synergy_300']
        row['wm_saving_vs_policy'] = row['wm_saving_vs_policy_300']
        out.append(row)
    return sorted(out, key=lambda r: (-r['wm_mean_synergy'], -r['wm_saving_vs_policy'], r['wm_action_id']))


def past_deadline():
    import datetime
    tz = datetime.timezone(datetime.timedelta(hours=8))
    hh, mm = (int(x) for x in DEADLINE_LOCAL.split(':'))
    return datetime.datetime.now(tz) >= datetime.datetime(2026, 11, 6, hh, mm, tzinfo=tz)


def main(root):
    import torch
    from tools.wmteach.runtime import bootstrap
    from tools.coopmwm.wm import cost_adapter
    import tools.coopmwm.moves as moves  # closure check: must be the accepted G5 copy
    assert hasattr(moves, 'LANE_FAMILIES'), moves.__file__
    prior = PARENT/'wm_worker01'
    write(root/f'OWNER_{os.getpid()}.json', dict(pid=os.getpid(), start_ticks=Path('/proc/self/stat').read_text().split()[21],
                                  boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                                  moves_file=moves.__file__, unix=time.time()))
    rt = bootstrap()
    torch.cuda.set_per_process_memory_fraction(.8)
    rt.frozen.costs = cost_adapter(rt.frozen.costs, read_json(prior/'FROZEN_COSTS.json')['code'],
                                   horizons=(60, 120, 300, 360, 420, 600))
    write(root/f'MODEL_READY_{os.getpid()}.json', dict(mode='FULL-v4 unoptimized reference, K8 H600', unix=time.time()))
    qualified = (root/'LANE_NOOP_QUALIFICATION.json').exists()
    while not (root/'STOP').exists() and not past_deadline():
        work = []
        for receipt in sorted(SOURCE.glob('*/COLLECTION_RECEIPT.json')):
            for c in read_json(receipt)['contexts']:
                d = Path(c['dir'])
                sid = read_json(d/'CONTEXT.json')['state_id']
                if not any((root/f'{sid}.{x}').exists() for x in ('rank.json', 'rank_error.json', 'claim')):
                    work.append((float(c['pressure'][0]), d, sid))
        if not work:
            time.sleep(30)
            continue
        work.sort(key=lambda x: (-x[0], x[2]))
        claimed = None
        for _, d, sid in work:
            try:
                fd = os.open(str(root/f'{sid}.claim'), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                continue
            os.write(fd, str(os.getpid()).encode()); os.close(fd)
            claimed = (d, sid)
            break
        if claimed is None:
            time.sleep(30)
            continue
        d, sid = claimed
        started = time.time()
        try:
            query, ctx = build_query(d)
            weights = load_base(query)
            coverage = {v: tuple(p['observed_lanes']) for v, p in query['sensing_packets'].items()}
            if not qualified:
                bases = [dict(id='policy'), dict(id='MAPPO')]
                expected, _ = score_grid(rt, query, bases, weights)
                actual = evaluate_lane_grid(rt, query, weights, bases, coverage_by_member={})
                ok = expected == actual['scores']
                write(root/'LANE_NOOP_QUALIFICATION.json', dict(pass_=ok,
                      state_id=sid, K=8, H=600, new_sumo_sim_s=0, unix=time.time()))
                if not ok:
                    raise SystemExit('lane-OFF identity qualification FAILED; no ranking')
                qualified = True
            main_, ramp = query['record']['group']
            actions = slate(main_, ramp)
            result = evaluate_lane_grid(rt, query, weights, actions, coverage_by_member=coverage,
                                        H=RANK_H, horizons=RANK_HORIZONS)
            check_scores(result['scores'], actions)
            ranked = rank(result['scores'], actions, main_, ramp)
            accepted = {}
            for e in result['lane_events']:
                accepted.setdefault(e['action'], []).append(bool(e['accepted_in_imagination']))
            for r in ranked:
                r['wm_lane_accepted_in_imagination'] = accepted.get(r['wm_action_id'])
            write(root/f'{sid}.rank.json', dict(state_id=sid, arm=ctx['arm'], world=ctx['world'], t0=ctx['t0'],
                  context_dir=str(d), query_sha256=query['source_sha256'], ranked=ranked,
                  policy_mean_team_delay_300=sum(v['team_delay'] for v in result['scores']['300']['policy'])/8,
                  K=8, H=RANK_H, synergy_horizon_s=300, new_sumo_sim_s=0, wall_s=time.time()-started, unix=time.time(),
                  interpretation='WM predictions for ranking only; SUMO verification decides'))
        except Exception as e:
            write(root/f'{sid}.rank_error.json', dict(state_id=sid, error=repr(e),
                  traceback=traceback.format_exc(), unix=time.time()))
    write(root/f'DONE_{os.getpid()}.json', dict(unix=time.time()))


if __name__ == '__main__':
    root = Path(sys.argv[1])
    try:
        main(root)
    except BaseException as e:
        write(root/'FAULT.json', dict(error=repr(e), traceback=traceback.format_exc(), unix=time.time()))
        raise
