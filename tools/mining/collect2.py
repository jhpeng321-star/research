"""Multi-context ordinary-TRAIN collection (operator 2, 2026-11-06 03:00 +08).

Per training world: ONE plain policy episode (no guide, no intervention). At
merge-pressure moments (observable local sensing only: ramp queue/arrivals,
mainline occupancy near the merge) up to three decision snapshots are captured
(distinct (main, ramp) pairs, >= 20 s apart). For each captured t0 the live
policy's own per-second raw command (base command + sampled residual) of every
locally sensed CAV is recorded for the next PROFILE_S seconds: that is the
immutable ``policy_profile`` reference of the executable G5 plan (seed-1 CRN).
No outcome, hidden type or truth field is read.
"""
import copy
import json
import os
from pathlib import Path
import sys
import time

from .collect_contexts import FIELDS, opportunities, write, charge

PROFILE_S = 130
KEEP = 3
T_MIN, T_MAX, EVERY = 300, 640, 5
SELECT = os.environ.get('MINE_SELECT', 'pressure')
KEEP = int(os.environ.get('MINE_KEEP', '3'))
SPACING = int(os.environ.get('MINE_SPACING', '20'))


def main(root, arm, first, count):
    import torch
    from tools.coopmwm import accel_plant, sumo_verify as sv
    from tools.coopmwm.protocol import worlds
    parent = root.parent
    assert arm in ('B', 'R')
    pool = [w for w in worlds('train') if 986000 <= w['seed'] <= 986071
            and w['demand'] >= 2400 and w['penetration'] >= .2]
    chosen = pool[first:first+count]
    assert chosen and all(986000 <= w['seed'] <= 986071 for w in chosen)
    weights = torch.load(parent/'frozen_u0.pt', weights_only=False, map_location='cpu')['policy']
    os.nice(19)
    write(root/f'OWNER_{arm}_{first}.json', dict(pid=os.getpid(), arm=arm, first=first, count=count,
          start_ticks=Path('/proc/self/stat').read_text().split()[21], unix=time.time()))
    for world in chosen:
        ident = f'{root.name}_{arm}_train_{world["seed"]}'
        work = root/ident
        work.mkdir()
        job = dict(id=ident, path=str(work), world=world, role='train', arm=arm, update=0,
                   training_seed=20261101, weights=weights,
                   runtime='/home/ubuntu/e3final',
                   checkpoint='/home/ubuntu/mappospeed/production2/policy_00169.pt',
                   noise_seed=20261106+world['seed']+(0 if arm == 'B' else 500000)+int(os.environ.get('MINE_NOISE_OFFSET', '0')),
                   deterministic=False, guided=False, b_replay=True, v2v=True,
                   health_guards=False, sim_s=1200, stop=1200, warm_history=True,
                   accel_bounds=[-3., 2.6])
        kept = []        # [pressure, t0, pair, snap, context]
        profiles = {}    # t0 -> {vid: [raw ...]}

        class Capture:
            def __call__(self, controller, job_, prefixes):
                self.c, self.job, self.prefixes = controller, job_, prefixes

            def after_step(self, t, after):
                e = self.c.events[-1]
                t = int(e['time_s'])
                members = e['sample']['context']['members']
                action = e['sample']['action']
                for t0, prof in profiles.items():
                    k = t-t0
                    if 0 <= k < PROFILE_S:
                        for vid, row in prof.items():
                            if vid in members and vid in e['commands']:
                                row.append(float(e['commands'][vid])+float(action[members.index(vid)][0]))
                            else:
                                row.append(None)
                if not T_MIN <= t <= T_MAX or t % EVERY:
                    return
                options = opportunities(e['rows'])
                if not options:
                    return
                rows = {r['vid']: r for r in e['rows']}
                behind = [o for o in options if rows[o[1]]['s_m'] < rows[o[2]]['s_m']]
                if SELECT == 'g5':
                    # accepted-G5 open_gap_merge geometry: lane-1 CAV behind a lane-2 CAV and the
                    # ramp CAV's own 100 m packet passes the accepted lane envelope NOW (observable)
                    from tools.mining.mine_verify import feasible
                    behind = [o for o in behind if feasible(e['rows'], o[2])]
                    if not behind:
                        return
                pressure, main, ramp, local = (behind or options)[0]
                pair = (main, ramp)
                conflicts = [x for x in kept if x[2] == pair or abs(x[1]-t) < SPACING]
                if any(x[0] >= pressure for x in conflicts):
                    return
                rest = [x for x in kept if x not in conflicts]
                if len(rest) >= KEEP and min(x[0] for x in rest) >= pressure:
                    return
                snap = sv.capture(self.c, self.prefixes, self.job, t, e['sample'], self.c.generator)
                cavs = [r['vid'] for r in local if r['is_cav']]
                context = dict(state_id=f'{ident}_t{t}', world=world, arm=arm, t0=t, group=list(pair),
                               pressure=list(pressure), rows=[{k: r[k] for k in FIELDS} for r in local],
                               cavs=cavs, snapshot_sha256=sv.snapshot_sha(snap),
                               selection='max (local ramp count, local mainline count, sum 1/(1+v_ramp), '
                                         '-|ds|); main behind ramp preferred; observable sensing only')
                profiles[t] = {v: [float(e['commands'][v])+float(action[members.index(v)][0])
                                   if v in members and v in e['commands'] else None] for v in cavs}
                rest.append([pressure, t, pair, snap, context])
                rest.sort(key=lambda x: x[0], reverse=True)
                kept[:] = rest[:KEEP]

        from tools.mining.mine_verify import _charge_factory
        _charge_factory(parent)(ident, [dict(sim_s=1200)])   # mining ledger, total cap 490k
        result = accel_plant.episode(job, controller_hook=Capture())
        saved = []
        for pressure, t0, pair, snap, context in kept:
            d = work/f'ctx_t{t0}'
            d.mkdir()
            torch.save(snap, d/'SNAPSHOT.pt')
            context['policy_profile'] = profiles[t0]
            write(d/'CONTEXT.json', context)
            saved.append(dict(t0=t0, group=list(pair), pressure=list(pressure), dir=str(d)))
        write(work/'COLLECTION_RECEIPT.json', dict(world=world, arm=arm, actual_sim_s=result['actual_sim_s'],
                                                     contexts=saved, unix=time.time()))
    write(root/f'DONE_{arm}_{first}.json', dict(unix=time.time()))


if __name__ == '__main__':
    main(Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
