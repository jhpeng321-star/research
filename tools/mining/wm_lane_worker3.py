"""G5-geometry WM proposer (retarget 04:50 +08, MAIN): rank only, zero SUMO.

Pairs are NOT CONTEXT.group: every (lane-1 CAV main BEHIND lane-2 CAV ramp), both
with s in the observable merge zone [-100, 75.44] and within 100 m of each other,
from the context's observable rows. Per pair: open_gap_merge hold 20, onsets
0..10 step 2, depths {.3,.6,1.0}, each with both member-release solos, plus
policy/MAPPO; K8/H300 (lane-OFF identity qualified at K8/H600 in wm_lane04).
Ranked by predicted group-over-best-solo synergy. Outputs under wm_lane04/g5pairs/.
"""
import json
import os
from pathlib import Path
import sys
import time
import traceback

from .collect_contexts import write
from .wm_contract import read_json
from .wm_prescreen import load_base
from .wm_lane_adapter import evaluate_lane_grid
from .wm_lane_schedule import make_schedule, action_set
from . import wm_lane_worker2 as w2

PARENT = Path('/root/mining20261106')
ZONE = (-100., 75.44)
ONSETS = (0, 2, 4, 6, 8, 10)
DEPTHS = (.3, .6, 1.)
HOLD = 20


def pairs(rows):
    out = []
    for m in rows:
        if not (m['is_cav'] and m['lane'] == 1 and ZONE[0] <= m['s_m'] <= ZONE[1]):
            continue
        for r in rows:
            if (r['is_cav'] and r['lane'] == 2 and ZONE[0] <= r['s_m'] <= ZONE[1]
                    and m['s_m'] < r['s_m'] and r['s_m']-m['s_m'] <= 100):
                out.append((m['vid'], r['vid'], r['s_m']-m['s_m']))
    return sorted(out, key=lambda x: (x[2], x[0], x[1]))


def slate(main, ramp):
    actions = [dict(id='policy'), dict(id='MAPPO')]
    for onset in ONSETS:
        for depth in DEPTHS:
            actions += action_set(make_schedule('open_gap_merge', main, ramp, onset, HOLD, depth))
    return actions


def sources():
    """Priority 0: explicit ctx4 lists; priority 1: op2_collect03 by pressure."""
    work = []
    for name in ('ctx4_B.txt', 'ctx4_R.txt'):
        p = PARENT/'op2_verify01'/name
        if p.exists():
            for line in p.read_text().split():
                d = Path(line.strip())
                if (d/'CONTEXT.json').exists():
                    work.append((0, 0., d))
    for receipt in sorted((PARENT/'op2_collect03').glob('*/COLLECTION_RECEIPT.json')):
        for c in read_json(receipt)['contexts']:
            work.append((1, -float(c['pressure'][0]), Path(c['dir'])))
    return sorted(work, key=lambda x: (x[0], x[1], str(x[2])))


def merge(out, ctx, sid):
    done = sorted(out.glob(f'{sid}__*.pair.json'))
    errors = sorted(out.glob(f'{sid}__*.pair_error.json'))
    ranked, meta = [], []
    for p in done:
        d = read_json(p)
        ranked += d['ranked']
        meta.append(dict(members=d['members'], policy_mean_team_delay_300=d['policy_mean_team_delay_300'],
                         wall_s=d['wall_s']))
    ranked.sort(key=lambda r: (-r['wm_mean_synergy'], -r['wm_saving_vs_policy'], r['wm_action_id']))
    total = len(pairs(ctx['rows']))
    tmp = out/f'{sid}.rank.json.tmp'
    tmp.write_text(json.dumps(dict(state_id=sid, arm=ctx['arm'], world=ctx['world'], t0=ctx['t0'],
        context_dir=str(out), ranked=ranked, pairs=meta, pairs_done=len(done), pair_errors=len(errors),
        pairs_total=total, complete=len(done)+len(errors) == total, K=8, H=w2.RANK_H, synergy_horizon_s=300,
        pair_rule='lane1 CAV behind lane2 CAV, both s in [-100,75.44], within 100 m',
        new_sumo_sim_s=0, unix=time.time(),
        interpretation='WM predictions for ranking only; SUMO verification decides'), sort_keys=True))
    os.replace(tmp, out/f'{sid}.rank.json')


def main(root):
    import torch
    from tools.wmteach.runtime import bootstrap
    from tools.coopmwm.wm import cost_adapter
    import tools.coopmwm.moves as moves
    assert hasattr(moves, 'LANE_FAMILIES'), moves.__file__
    assert read_json(root/'LANE_NOOP_QUALIFICATION.json')['pass_']
    out = root/'g5pairs'
    out.mkdir(exist_ok=True)
    write(out/f'OWNER_{os.getpid()}.json', dict(pid=os.getpid(),
          start_ticks=Path('/proc/self/stat').read_text().split()[21], unix=time.time()))
    rt = bootstrap()
    torch.cuda.set_per_process_memory_fraction(.45)
    rt.frozen.costs = cost_adapter(rt.frozen.costs, read_json(PARENT/'wm_worker01'/'FROZEN_COSTS.json')['code'],
                                   horizons=(60, 120, 300, 360, 420, 600))
    while not (out/'STOP').exists() and not w2.past_deadline():
        claimed = None
        for _, _, d in sources():
            ctx = read_json(d/'CONTEXT.json')
            sid = ctx['state_id']
            for main_, ramp, _ in pairs(ctx['rows']):
                key = f'{sid}__{main_}__{ramp}'
                try:
                    fd = os.open(str(out/f'{key}.claim'), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                except FileExistsError:
                    continue
                os.write(fd, str(os.getpid()).encode()); os.close(fd)
                claimed = (d, ctx, sid, main_, ramp, key)
                break
            if claimed:
                break
        if claimed is None:
            time.sleep(30)
            continue
        d, ctx, sid, main_, ramp, key = claimed
        started = time.time()
        try:
            ctx_pair = dict(ctx, group=[main_, ramp])
            orig = w2.read_json
            w2.read_json = lambda p, _c=ctx_pair, _o=orig: _c if Path(p).name == 'CONTEXT.json' else _o(p)
            try:
                query, _ = w2.build_query(d)
            finally:
                w2.read_json = orig
            weights = load_base(query)
            coverage = {v: tuple(p['observed_lanes']) for v, p in query['sensing_packets'].items()}
            actions = slate(main_, ramp)
            result = evaluate_lane_grid(rt, query, weights, actions, coverage_by_member=coverage,
                                        H=w2.RANK_H, horizons=w2.RANK_HORIZONS)
            w2.check_scores(result['scores'], actions)
            ranked = w2.rank(result['scores'], actions, main_, ramp)
            accepted = {}
            for e in result['lane_events']:
                accepted.setdefault(e['action'], []).append(bool(e['accepted_in_imagination']))
            for r in ranked:
                a = accepted.get(r['wm_action_id']) or []
                r['wm_lane_accepted_any'] = any(a)
                r['wm_lane_attempts'] = len(a)
            write(out/f'{key}.pair.json', dict(state_id=sid, members=[main_, ramp], ranked=ranked,
                  query_sha256=query['source_sha256'],
                  policy_mean_team_delay_300=sum(v['team_delay'] for v in result['scores']['300']['policy'])/8,
                  wall_s=time.time()-started, unix=time.time()))
        except Exception as e:
            write(out/f'{key}.pair_error.json', dict(state_id=sid, members=[main_, ramp], error=repr(e),
                  traceback=traceback.format_exc(), unix=time.time()))
        merge(out, ctx, sid)
    write(out/f'DONE_{os.getpid()}.json', dict(unix=time.time()))


if __name__ == '__main__':
    root = Path(sys.argv[1])
    try:
        main(root)
    except BaseException as e:
        write(root/'g5pairs'/f'FAULT_{os.getpid()}.json', dict(error=repr(e),
              traceback=traceback.format_exc(), unix=time.time()))
        raise
