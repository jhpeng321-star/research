"""Bounded ordinary-TRAIN collection; pressure selection uses local sensing only."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

FIELDS = ('vid', 's_m', 'v_mps', 'lane', 'is_cav', 'length_m')


def opportunities(rows):
    """No outcome or hidden field is accepted by this selector."""
    rows = [{k: r[k] for k in FIELDS} for r in rows]
    out = []
    for main in rows:
        if not main['is_cav'] or main['lane'] != 1 or not -100 <= main['s_m'] <= 75.44000244140625:
            continue
        local = [r for r in rows if abs(r['s_m'] - main['s_m']) <= 100]
        ramps = [r for r in local if r['lane'] == 2]
        mains = [r for r in local if r['lane'] in (0, 1)]
        for ramp in ramps:
            if not ramp['is_cav'] or not -100 <= ramp['s_m'] <= 75.44000244140625:
                continue
            # Queue/arrivals and occupancy are continuous/rank features, not
            # a new behavioural safety threshold. Largest pressure wins.
            pressure = (len(ramps), len(mains), sum(1 / (1 + r['v_mps']) for r in ramps),
                        -abs(main['s_m'] - ramp['s_m']))
            out.append((pressure, main['vid'], ramp['vid'], local))
    return sorted(out, key=lambda x: (x[0], x[1], x[2]), reverse=True)


def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, sort_keys=True, allow_nan=False)
        f.flush(); os.fsync(f.fileno())


def charge(root, ident, amount):
    import fcntl
    with (root / 'MINING_BUDGET.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = root / 'CHARGES.jsonl'
        rows = [json.loads(s) for s in ledger.read_text().splitlines() if s.strip()]
        assert not any(r['id'] == ident for r in rows), 'attempt already charged; never replay'
        # gpuA slice includes parity; cpuV's disjoint slice is 200,000.
        assert sum(r['sim_s'] for r in rows) + amount <= 300000
        with ledger.open('a') as f:
            f.write(json.dumps(dict(id=ident, phase='mining', sim_s=amount, unix=time.time())) + '\n')
            f.flush(); os.fsync(f.fileno())


def main(root):
    import torch
    from tools.coopmwm import accel_plant, sumo_verify as sv
    from tools.coopmwm.protocol import worlds
    from .wm_contract import U0_SHA, validate_query
    parent = root.parent
    assert hashlib.sha256((parent / 'frozen_u0.pt').read_bytes()).hexdigest() == U0_SHA
    assert json.loads((parent / 'PARITY_PASS.json').read_bytes())['pass_']
    authority = json.loads((root / 'COLLECTION_AUTHORITY.json').read_bytes())
    assert authority['g5_commit'] == 'b903c6c2' and authority['gpu_cap_sim_s'] == 300000
    worlds_ = [w for w in worlds('train') if 986000 <= w['seed'] <= 986071
               and w['demand'] >= 2400 and w['penetration'] >= .2][:4]
    assert len(worlds_) == 4
    weights = torch.load(parent / 'frozen_u0.pt', weights_only=False, map_location='cpu')['policy']
    os.nice(19); os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    write(root / 'COLLECTION_OWNER.json', dict(pid=os.getpid(),
          start_ticks=Path('/proc/self/stat').read_text().split()[21],
          boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), unix=time.time()))
    summary = []
    for world in worlds_:
        ident = 'mining_train_' + str(world['seed'])
        work = root / ident; work.mkdir()
        job = dict(id=ident, path=str(work), world=world, role='train', arm='B', update=0,
                   training_seed=20261101, weights=weights,
                   runtime='/home/ubuntu/e3final',
                   checkpoint='/home/ubuntu/mappospeed/production2/policy_00169.pt',
                   noise_seed=20261106 + world['seed'], deterministic=False, guided=False,
                   health_guards=False, sim_s=1200, stop=1200, warm_history=True,
                   accel_bounds=[-3., 2.6])
        selected = []

        class Capture:
            def __call__(self, controller, job_, prefixes):
                self.c, self.job, self.prefixes = controller, job_, prefixes

            def after_step(self, t, after):
                if t > 600:
                    return
                e = self.c.events[-1]
                options = opportunities(e['rows'])
                if not options or selected and options[0][0] <= selected[0][0]:
                    return
                pressure, ego, ramp, local = options[0]
                snap = sv.capture(self.c, self.prefixes, self.job, t, e['sample'], self.c.generator)
                hist = []
                for event in self.c.events:
                    if not t - 60 <= event['time_s'] <= t:
                        continue
                    table = {r['vid']: r for r in event['rows']}
                    if ego not in table:
                        continue
                    sensed = [{k: r[k] for k in FIELDS} for r in event['rows']
                              if abs(r['s_m'] - table[ego]['s_m']) <= 100]
                    hist.append(dict(time_s=event['time_s'], rows=sensed, entrance_counts=None))
                own = dict(zip(e['sample']['context']['members'],
                               [float(x[0]) for x in e['sample']['own_action']]))
                cavs = [r['vid'] for r in local if r['is_cav']]
                record = dict(state_id=ident + '_t' + str(t), world_id=world['seed'], time_s=t,
                              rows=local, group=[ego, ramp], commands={v: e['commands'][v] for v in cavs},
                              policy_residuals={v: own[v] for v in cavs}, history=hist)
                query = dict(record=record, ego=ego, source_sha256=sv.snapshot_sha(snap),
                    base_continuation=dict(checkpoint=str(parent / 'frozen_u0.pt'), checkpoint_sha256=U0_SHA,
                        deterministic=False, generator_state_after_t0=snap['generator_state'].tolist(),
                        memory_after_t0={v: snap['rng']['memory_after_t0'][v] for v in cavs}))
                validate_query(query)
                selected[:] = [(pressure, copy.deepcopy(snap), query)]

        charge(parent, ident, 1200)
        result = accel_plant.episode(job, controller_hook=Capture())
        item = dict(world=world, actual_sim_s=result['actual_sim_s'], selected=bool(selected))
        if selected:
            pressure, snap, query = selected[0]
            torch.save(snap, work / 'MINING_SNAPSHOT.pt')
            write(work / 'MINING_QUERY.json', query)
            item.update(pressure=pressure, time_s=snap['t0'], group=query['record']['group'],
                        snapshot=str(work / 'MINING_SNAPSHOT.pt'), query=str(work / 'MINING_QUERY.json'))
        write(work / 'COLLECTION_RECEIPT.json', item)
        summary.append(item)
    write(root / 'COLLECTION_DONE.json', dict(worlds=summary, charged_sim_s=4800, unix=time.time()))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
