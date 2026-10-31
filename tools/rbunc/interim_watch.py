"""Read-only peer inputs; bounded, native FIT-only uncertainty diagnostics."""
from datetime import datetime
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import random
import statistics
import time

WORLDS = tuple(range(969420, 969432))
METRICS = ('team_delay_vs', 'road_delay_vs', 'source_wait_vs')
DEADLINE = datetime.fromisoformat('2026-11-01T09:00:00+08:00').timestamp()
PANEL = Path('/root/rbeval20261031/int/WM-PHYS-FULL-v1')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=1, allow_nan=False)


def inspect_world(panel, world, target):
    if world not in WORLDS:
        raise ValueError('explicit FIT world paths only')
    paths = []
    for future in range(8):
        matches = [panel/f'shard{i}'/f'w{world}_k{future}.json' for i in range(3)]
        matches = [p for p in matches if p.is_file()]
        if len(matches) > 1:
            raise ValueError('duplicate candidate future')
        if not matches:
            return None
        paths.append(matches[0])
    identities, forecasts, hashes, source_receipts = set(), [], {}, []
    for future, path in enumerate(paths):
        raw = path.read_bytes()
        task = json.loads(raw)
        if (task['schema'] != 'rbeval/1' or task['seed'] != world or task['future'] != future
                or task['arms'] != ['native', 'ctrl0', 'mappo', 'pi', 'circles']):
            raise ValueError('candidate task identity differs')
        identities.add(task['identity_sha256'])
        source = task['future_sampling']
        if (source['variant'] != 'FULL' or source['future_sampling'] != 'hidden_state'
                or source['future'] != future or source['physics_noise'] is not False
                or source['embedding_noise'] is not False
                or source['population_audit']['finite'] is not True
                or source['population_audit']['conserved'] is not True):
            raise ValueError('FULL deterministic source audit differs')
        rates = source['demand_rate_draws']
        if len(rates) != 5 or any(row != rates[0] for row in rates):
            raise ValueError('candidate actions do not share demand-rate draws')
        episodes = task['episodes']
        if (len(episodes) != 5 or [e['arm'] for e in episodes] != task['arms']
                or any(e['seed'] != world or e['future'] != future
                       or e['initial_sha256'] != target['initial_sha256'] for e in episodes)):
            raise ValueError('native target/start identity differs')
        forecast = episodes[0]['clocks']['120']
        if any(not math.isfinite(forecast[n]) for n in METRICS):
            raise ValueError('nonfinite native forecast')
        if not math.isclose(forecast['team_delay_vs'], forecast['road_delay_vs']
                            + forecast['source_wait_vs'], abs_tol=.001, rel_tol=1e-6):
            raise ValueError('native source/road ledger differs')
        forecasts.append({n:float(forecast[n]) for n in METRICS})
        hashes[path.relative_to(panel).as_posix()] = hashlib.sha256(raw).hexdigest()
        source_receipts.append(dict(fitted_artifacts=source['fitted_artifacts'],
            learned_arrival_sha256=source['learned_arrival_sha256'],
            learned_arrival_observations=source.get('learned_arrival_observations'),
            demand_rates_paired=True))
    if len(identities) != 1 or any(s['fitted_artifacts'] != source_receipts[0]['fitted_artifacts']
                                  or s['learned_arrival_sha256'] != source_receipts[0]['learned_arrival_sha256']
                                  for s in source_receipts):
        raise ValueError('mixed candidate/posterior/arrival sources')
    return dict(world=world, identity_sha256=identities.pop(), hashes=hashes,
                target=target, forecasts=forecasts, source_receipts=source_receipts)


def interval(values):
    if len(values) < 2:
        return None
    rng = random.Random(20261101)
    draws = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(10000))
    return [draws[249], draws[9749]]


def summarize(records):
    if not records:
        return None
    if len({r['identity_sha256'] for r in records}) != 1:
        raise ValueError('mixed candidate identities across worlds')
    result = {}
    for metric in METRICS:
        rows = []
        for record in records:
            x = [f[metric] for f in record['forecasts']]
            y = record['target']['real'][metric]
            absolute = statistics.mean(abs(v-y) for v in x)
            pairs = sum(abs(a-b) for a in x for b in x)
            rows.append(dict(world=record['world'], real=y, mean=statistics.mean(x),
                bias=statistics.mean(x)-y, within_K_sd=statistics.stdev(x),
                empirical_crps=absolute-pairs/(2*8*8), fair_crps=absolute-pairs/(2*8*7),
                range_covered=min(x) <= y <= max(x)))
        fields = ('bias', 'within_K_sd', 'empirical_crps', 'fair_crps', 'range_covered')
        result[metric] = dict(worlds=rows, **{
            n:dict(mean=statistics.mean(float(r[n]) for r in rows),
                   world_bootstrap_ci95=interval([float(r[n]) for r in rows])) for n in fields},
            real_across_subset_world_sd=statistics.stdev(r['real'] for r in rows) if len(rows)>1 else None)
    return dict(metrics=result, complete_fit_worlds=[r['world'] for r in records], K=8,
                horizon_s=120, arm='native', range_exchangeability_reference=7/9,
                qualification='incomplete FIT-only source/spread diagnostic, not distributional qualification',
                physical_sampling_scale_changed=False,
                sources=[dict(world=r['world'], hashes=r['hashes'], target=r['target'],
                              source_receipts=r['source_receipts']) for r in records])


def run(panel, out, targets, *, poll_s=1200):
    if panel.resolve() != PANEL or not out.is_dir():
        raise ValueError('explicit INT panel and existing UNC output root required')
    if poll_s < 60:
        raise ValueError('bounded polling only')
    targets_raw = targets.read_bytes()
    target_data = json.loads(targets_raw)
    if (target_data['schema'] != 'rbunc_native_interim_targets/1'
            or set(target_data['worlds']) != {str(w) for w in WORLDS}):
        raise ValueError('explicit FIT target snapshot required')
    save(out/'STARTED.json', dict(pid=os.getpid(), started_unix=time.time(), deadline=DEADLINE,
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],
        source_sha256=digest(Path(__file__)), target_sha256=hashlib.sha256(targets_raw).hexdigest(),
        panel=str(panel), cpu_only=True, poll_s=poll_s))
    signature = None
    previous_hashes = {}
    try:
        while True:
            admitted = os.getloadavg()[0] < 16
            complete = []
            if admitted:
                for world in WORLDS:
                    record = inspect_world(panel, world, target_data['worlds'][str(world)])
                    if record is not None:
                        complete.append(record)
                current_hashes = {name:sha for r in complete for name,sha in r['hashes'].items()}
                if any(current_hashes.get(name) != sha for name,sha in previous_hashes.items()):
                    raise ValueError('previously completed FIT input changed or disappeared')
                if complete:
                    current = hashlib.sha256(json.dumps(current_hashes, sort_keys=True).encode()).hexdigest()
                    if current != signature:
                        report = summarize(complete)
                        report.update(source_signature=current, unix=time.time(), target_source_sha256=target_data['source_sha256'])
                        save(out/f'DIAGNOSTIC_{current[:16]}.json', report)
                        signature, previous_hashes = current, current_hashes
            save(out/f'STATUS_{time.time_ns()}.json', dict(unix=time.time(), load=os.getloadavg(),
                admitted=admitted, complete_fit_worlds=[r['world'] for r in complete] if admitted else None,
                latest_source_signature=signature, deadline=DEADLINE))
            print(json.dumps(dict(status='FIT_UNCERTAINTY_OBSERVER', admitted=admitted,
                complete_fit_worlds=[r['world'] for r in complete] if admitted else None,
                source_signature=signature)), flush=True)
            if time.time() >= DEADLINE:
                save(out/'TERMINAL.json', dict(reason='owner09:00 deadline', unix=time.time(),
                    latest_source_signature=signature, peer_process_actions=0, inference_dispatches=0))
                break
            time.sleep(min(poll_s, max(0., DEADLINE-time.time())))
    except BaseException as error:
        save(out/'FAILED.json', dict(error=repr(error), unix=time.time(), peer_process_actions=0))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--panel', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--targets', type=Path, required=True)
    args = parser.parse_args()
    run(args.panel, args.out, args.targets)
