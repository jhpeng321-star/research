"""Real-GPU FULL-v4 reference ranking; fresh lane no-op identity qualification.

The completed worker01 eight-gate evidence is retained, never claimed to belong
to this PID. This worker uses the unoptimized reference evaluator throughout.
"""
import copy
import json
import os
from pathlib import Path
import sys
import time
import traceback
import gzip
import hashlib

from .collect_contexts import write
from .wm_contract import check_sources, grid, read_json, validate_query, HORIZONS
from .wm_prescreen import ASSET_PINS, load_base, score_grid, validate_scores
from .wm_lane_adapter import evaluate_lane_grid
from .wm_lane_schedule import make_schedule, action_set


def enriched_query(path):
    """Join only causal, six-field actor sensing from the hash-bound native tape."""
    from .collect_contexts import FIELDS
    from .wm_contract import digest
    query = read_json(path)
    marker = read_json(path.parent / 'RESULT.json')
    raw = gzip.decompress((path.parent / marker['file']).read_bytes())
    assert hashlib.sha256(raw).hexdigest() == marker['sha256']
    tape = json.loads(raw)
    t0 = query['record']['time_s']
    members = query['record']['group']
    history = []
    for event in tape['steps']:
        if not t0 - 60 <= event['time_s'] <= t0:
            continue
        by = {r['vid']: r for r in event['rows']}
        if query['ego'] not in by:
            continue
        packets = {v: dict(time_s=event['time_s'], observed_lanes=[0, 1, 2],
                          rows=[{k: row[k] for k in FIELDS} for row in event['rows']
                                if abs(row['s_m'] - by[v]['s_m']) <= 100])
                   for v in members if v in by}
        roster = [{k: row[k] for k in FIELDS} for row in event['rows']
                  if any(abs(row['s_m'] - by[v]['s_m']) <= 100 for v in packets)]
        history.append(dict(time_s=event['time_s'], rows=roster, entrance_counts=None))
        if event['time_s'] == t0:
            query['sensing_packets'] = packets
            record = query['record']
            record.update(rows=roster, history=history)
            own = dict(zip(event['sample']['context']['members'],
                           [float(x[0]) for x in event['sample']['own_action']]))
            cavs = [r['vid'] for r in roster if r['is_cav']]
            record['commands'] = {v: event['commands'][v] for v in cavs}
            record['policy_residuals'] = {v: own[v] for v in cavs}
            query['base_continuation']['memory_after_t0'] = {v: own[v] for v in cavs}
    query['source_sha256'] = digest(dict(original_snapshot=query['source_sha256'],
                                       tape_sha256=marker['sha256'], time_s=t0, actors=members))
    return validate_query(query)


def slate(query):
    main, ramp = query['record']['group']
    actions, rejected = grid(query, families=('gap_open', 'ramp_assist', 'ease_off'),
                             onsets=(0,), holds=(20,), depths=(.3,))
    for onset in (0, 10):
        actions += action_set(make_schedule('g5_yield_lane', main, ramp, onset))
        for hold in (20, 60):
            for depth in (.3, 1.):
                actions += action_set(make_schedule('open_gap_merge', main, ramp, onset, hold, depth))
    return actions, rejected


def rank(scores, actions):
    rows = []
    for action in actions:
        if 'solo_of' in action or not ('wm_move' in action or 'wm_role_schedule' in action):
            continue
        ids = [a['id'] for a in actions if a.get('solo_of') == action['id']]
        joint = scores['600'][action['id']]
        mean = lambda values: sum(v['team_delay'] for v in values) / len(values)
        synergy = ([min(scores['600'][s][k]['team_delay'] for s in ids) - joint[k]['team_delay']
                    for k in range(8)] if len(ids) == 2 else [])
        rows.append(dict(action=action, mean_delay=mean(joint),
                         saving_vs_policy=mean(scores['600']['policy']) - mean(joint),
                         paired_synergy=synergy, mean_synergy=sum(synergy) / 8 if synergy else None))
    return sorted(rows, key=lambda r: (-(r['mean_synergy'] if r['mean_synergy'] is not None else -1e30),
                                      r['mean_delay'], r['action']['id']))


def main(root):
    import torch
    from tools.wmteach.runtime import bootstrap
    from tools.coopmwm.wm import cost_adapter
    parent = root.parent
    prior = parent / 'wm_worker01'
    qualification = read_json(prior / 'FRESH_QUALIFICATION.json')
    assert qualification['pass_'] and qualification['pid'] == 3230
    assert (prior / 'MINING_WM_EXIT.json').is_file()
    check_sources(read_json(prior / 'MINING_SOURCE_MANIFEST.json'))
    check_sources(read_json(root / 'LANE_SOURCE_MANIFEST.json'))
    check_sources(ASSET_PINS)
    write(root / 'OWNER.json', dict(pid=os.getpid(), start_ticks=Path('/proc/self/stat').read_text().split()[21],
                                  boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), unix=time.time()))
    rt = bootstrap()
    torch.cuda.set_per_process_memory_fraction(.8)
    rt.frozen.costs = cost_adapter(rt.frozen.costs, read_json(prior / 'FROZEN_COSTS.json')['code'],
                                  horizons=(60, 120, 300, 360, 420, 600))
    write(root / 'MODEL_READY.json', dict(device=str(rt.device) if hasattr(rt, 'device') else 'cuda',
         original_gate_pid=3230, mode='FULL-v4 unoptimized reference; new lane identity pending', unix=time.time()))
    deadline = time.time() + 12000
    source = parent / 'executor_b903c6c2_v2'
    seen = set()
    qualified = False
    while time.time() < deadline:
        for path in sorted(source.glob('*train_*/MINING_QUERY.json')):
            if str(path) in seen:
                continue
            if not (path.parent / 'RESULT.json').is_file():
                continue
            query = enriched_query(path)
            write(root / (path.parent.name + '.query.json'), query)
            weights = load_base(query)
            coverage = {v: tuple(p['observed_lanes']) for v, p in query['sensing_packets'].items()}
            if not qualified:
                bases = [dict(id='policy'), dict(id='MAPPO')]
                expected, _ = score_grid(rt, query, bases, weights)
                actual = evaluate_lane_grid(rt, query, weights, bases, coverage_by_member={})
                assert expected == actual['scores'], 'lane decoder OFF differs from original reference'
                write(root / 'LANE_NOOP_QUALIFICATION.json', dict(pass_=True, exact_scores=True,
                    K=8, H=600, source_query=str(path), source_sha256=query['source_sha256'],
                    original_gates_pid=3230, current_pid=os.getpid(), new_sumo_sim_s=0, unix=time.time()))
                qualified = True
            actions, rejected = slate(query)
            result = evaluate_lane_grid(rt, query, weights, actions, coverage_by_member=coverage)
            validate_scores(result['scores'], actions)
            result.update(query_path=str(path), query_sha256=query['source_sha256'], actions=actions,
                          geometry_rejections=rejected, ranked=rank(result['scores'], actions),
                          actor_coverage=coverage, unix=time.time())
            write(root / (path.parent.name + '.result.json'), result)
            seen.add(str(path))
        if (source / 'COLLECTION_DONE.json').is_file() and len(seen) == 4:
            break
        time.sleep(10)
    write(root / 'DONE.json', dict(completed=sorted(seen), unix=time.time()))


if __name__ == '__main__':
    root = Path(sys.argv[1])
    try:
        main(root)
    except BaseException as e:
        write(root / 'FAULT.json', dict(error=repr(e), traceback=traceback.format_exc(), unix=time.time()))
        raise
