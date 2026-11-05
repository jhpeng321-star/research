"""Cooperation probe WM proposer (gpuA GPU, rank only, zero SUMO). Reads <Q>/*.query.json written by
probe_wmq.write_wmq on cpuV, evaluates the slate K8 / H (default 120 s) with the mining lane-capable
evaluate_lane_grid (frozen FULL-v4, original CRN), ranks groups by predicted synergy
(min solo team delay - group team delay, paired per draw) and by saving vs policy."""
import json
import os
from pathlib import Path
import sys
import time
import traceback


def main(q):
    import torch
    import tools.mining  # noqa: F401  (mining __init__ appends the accepted-G5 coopmwm dir: moves.py)
    from tools.wmteach.runtime import bootstrap
    from tools.coopmwm.wm import cost_adapter
    import tools.coopmwm.moves as moves
    moves.MOVE_MAX_MEMBERS = 6       # owner 04:20: group moves up to six CAVs (probe WM only)
    from tools.mining.wm_contract import read_json
    from tools.mining.wm_prescreen import load_base
    from tools.mining.wm_lane_adapter import evaluate_lane_grid
    (q/f'OWNER_{os.getpid()}.json').write_text(json.dumps(dict(pid=os.getpid(), unix=time.time())))
    rt = bootstrap()
    torch.cuda.set_per_process_memory_fraction(.4)
    rt.frozen.costs = cost_adapter(rt.frozen.costs,
                                   read_json(Path('/root/mining20261106/wm_worker01/FROZEN_COSTS.json'))['code'],
                                   horizons=(60, 120, 300, 360, 420, 600))
    (q/f'READY_{os.getpid()}.json').write_text(json.dumps(dict(unix=time.time())))
    while not (q/'STOP').exists():
        job = None
        for p in sorted(q.glob('*.query.json'), key=lambda p: p.stat().st_mtime):
            out = p.with_name(p.name.replace('.query.json', '.rank.json'))
            if out.exists():
                continue
            try:
                fd = os.open(str(p)+'.claim', os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                continue
            os.close(fd)
            job = (p, out)
            break
        if job is None:
            time.sleep(15)
            continue
        p, out = job
        started = time.time()
        try:
            d = json.loads(p.read_text())
            query, actions, H = d['query'], d['actions'], int(d.get('H', 120))
            weights = load_base(query)
            coverage = {v: tuple(x['observed_lanes']) for v, x in query['sensing_packets'].items()}
            hs = tuple(sorted({60, H}))
            K = int(d.get('K', 8))
            if K == 8:
                res = evaluate_lane_grid(rt, query, weights, actions, coverage_by_member=coverage, H=H, horizons=hs)
            else:   # same body as evaluate_lane_grid with K draws (K4/H120 is a qualified startup shape)
                import copy as _copy
                from tools.coopmwm import plan_wm
                from tools.coopmwm.plan_service import request_memory
                from tools.mining.wm_contract import validate_query
                from tools.mining.wm_lane_adapter import check_actions, lane_executor
                validate_query(query)
                record = _copy.deepcopy(query['record'])
                check_actions(actions, [r['vid'] for r in record['rows']])
                memory = request_memory(rt, record, {'memory_mode': 'warm'})
                with lane_executor(rt, query, weights, coverage_by_member=coverage):
                    costs = plan_wm.evaluate(rt, record, actions, K, H, chunk_actions=1, trim=False,
                                             pad_actions=False, horizons=hs, memory=memory, speed2=False,
                                             transfers=False)
                res = dict(scores=costs)
            sc = res['scores'][str(H)]

            def mean(xs):
                return sum(x['team_delay'] for x in xs)/len(xs)
            ranked = []
            for gid, g in d['groups'].items():
                joint = sc[gid]
                paired = [min(sc[s][k]['team_delay'] for s in g['solos'])-joint[k]['team_delay']
                          for k in range(len(joint))]
                ranked.append(dict(group=gid, plan_id=g['plan_id'], wm_mean_synergy=sum(paired)/len(paired),
                                   wm_synergy_draws=paired, wm_saving_vs_policy=mean(sc['policy'])-mean(joint),
                                   wm_best_solo_saving=max(mean(sc['policy'])-mean(sc[s]) for s in g['solos'])))
            ranked.sort(key=lambda r: (-r['wm_mean_synergy'], -r['wm_saving_vs_policy']))
            tmp = out.with_suffix('.tmp')
            tmp.write_text(json.dumps(dict(ranked=ranked, H=H, K=int(d.get('K', 8)), n_actions=len(actions),
                                           wall_s=time.time()-started, new_sumo_sim_s=0, unix=time.time(),
                                           interpretation='WM predictions for ranking only; SUMO decides')))
            os.replace(tmp, out)
        except Exception as e:
            p.with_name(p.name.replace('.query.json', '.rank_error.json')).write_text(
                json.dumps(dict(error=repr(e), tb=traceback.format_exc()[-4000:], unix=time.time())))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
