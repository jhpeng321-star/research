"""MAIN01:28 receipt-only PARTIAL analysis. No simulation/inference imports."""
import csv
import hashlib
import json
import math
from pathlib import Path

from .analyze import analyze_rows
from .design import CONTROLLERS, measurement_indices, worlds
from .episode import ARMS
from .storage import canonical, write

EVIDENCE = Path(__file__).parent/'evidence'
MIRROR = EVIDENCE/'shared_v10_panel'
OUTPUT = Path('docs/paper_results/c1switch')


def paired_rows(root):
    """Select the intersection of final results, never PHYSICS-only or partial picks."""
    rows = {a:[] for a in ARMS}
    panel = []
    availability = {}
    inputs = {}

    def load(path):
        raw = path.read_bytes()
        inputs[str(path.relative_to(root))] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    for world in worlds('main'):
        seed = world['seed']
        paths = {a:root/'jobs'/f'{a}_s{seed}_episode'/'RESULT.json' for a in ARMS}
        available = [a for a,p in paths.items() if p.exists()]
        availability[str(seed)] = available
        if len(available) != len(ARMS):
            continue
        paired = {}
        for arm,path in paths.items():
            r = load(path)
            if r['arm'] != arm or r['world'] != world or len(r['decisions']) != 10:
                raise ValueError('episode pairing mismatch')
            close = load(path.with_name('CLOSE.json'))
            if close != dict(actual_sim_s=900.,charged_sim_s=900.):
                raise ValueError('incomplete episode accounting')
            if [d['pick'] for d in r['decisions']] != r['schedule']:
                raise ValueError('schedule mismatch')
            if not all(math.isfinite(x) for x in r['J600'].values()):
                raise ValueError('nonfinite episode metrics')
            measured = [d['j'] for d in r['decisions'] if 'branches' in d]
            if measured != (measurement_indices(seed) if arm in ('wm','bb') else []):
                raise ValueError('registered measurement subset mismatch')
            for j,d in enumerate(r['decisions']):
                if d['j'] != j or d['t_s'] != 300+60*j or d['pick'] not in CONTROLLERS:
                    raise ValueError('decision clock/controller mismatch')
                if 'branches' not in d:
                    continue
                if set(d['branches']) != set(CONTROLLERS):
                    raise ValueError('five-way branch set required')
                if set(d['predicted_costs']) != set(CONTROLLERS) or not all(
                    math.isfinite(x) for x in d['predicted_costs'].values()
                ):
                    raise ValueError('finite five-way predictions required')
                for c,b in d['branches'].items():
                    job = root/'jobs'/f'{arm}_s{seed}_j{j}_{c}'
                    standalone = load(job/'RESULT.json')
                    duration = 390+60*j
                    if (canonical(b) != canonical(standalone) or b['prefix'] != d['prefix']
                        or b['actual_sim_s'] != duration or load(job/'CLOSE.json') !=
                        dict(actual_sim_s=duration,charged_sim_s=duration)):
                        raise ValueError('branch/live/standalone/accounting mismatch')
                    if not all(math.isfinite(x) for clock in ('60','90') for x in b[clock].values()):
                        raise ValueError('nonfinite branch metrics')
            paired[arm] = r
        if len({canonical(r['decisions'][0]['prefix']) for r in paired.values()}) != 1:
            raise ValueError('paired warmup mismatch')
        panel.append(world)
        for a in ARMS:
            rows[a].append(paired[a])
    if not panel:
        raise ValueError('no all-arm-complete worlds')
    return panel,rows,availability,inputs


def summarize_partial(root=MIRROR, evidence=EVIDENCE):
    summary = json.loads((evidence/'V10_COLLECTION.json').read_text())
    if not summary['all_cpu_owned_absent'] or summary['gpu']['owned_pid_alive']:
        raise ValueError('terminal owned jobs required')
    binding = json.loads((root/'BINDING.json').read_text())
    ledger = [json.loads(x) for x in (root/'accounting.jsonl').read_text().splitlines()]
    closes = [json.loads(p.read_text()) for p in root.glob('jobs/*/CLOSE.json')]
    actual = (47010+binding['retained_superseded_sim_s']+binding['retained_abandoned_sim_s']
              +sum(r['actual_sim_s'] for r in closes))
    if (sum(r['sim_s'] for r in ledger) != summary['charged_sim_s']
        or actual != summary['actual_closed_sim_s'] or actual != summary['charged_sim_s']
        or actual > 500000 or any(r['actual_sim_s'] != r['charged_sim_s'] for r in closes)):
        raise ValueError('terminal accounting mismatch')
    panel,rows,availability,inputs = paired_rows(root)
    result = analyze_rows(panel,rows)
    result.update(status='PARTIAL',authorization='MAIN01:28',worlds=panel,arms=list(ARMS),
        selection_rule='intersection of final RESULT.json for all eight episode arms; all WM/BB branches required',
        availability=availability,source_sha256=inputs,new_sim_s=0,new_gpu_s=0,
        charged_sim_s=actual,actual_gpu_s=summary['actual_gpu_accounting']['total_s'])
    result['caveats'].insert(0,
        'PARTIAL scheduler/budget-selected low-penetration cohort; descriptive conditional intervals, '
        'not the registered 16-world E3 population or an efficacy verdict')
    return result


def main():
    result = summarize_partial()
    write(OUTPUT/'partial_results.json',result)
    for name,rows in [('partial_decisions',result['measured_decisions']),
                      ('partial_worlds',[
                          dict(seed=r['world']['seed'],penetration=r['world']['penetration'],
                               demand=r['world']['demand'],
                               **{f'J600_{a}':v for a,v in r['J600'].items()},
                               **{f'regret60_{a}':v for a,v in r['regret60'].items()},
                               **{f'wm_minus_{c}_regret60':v for c,v in r['wm_minus_each_fixed_regret60'].items()})
                          for r in result['world_rows']])]:
        with (OUTPUT/(name+'.csv')).open('w',newline='',encoding='utf-8') as f:
            writer = csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({k:result[k] for k in ('status','worlds','arms','primary','episode_mean',
        'regret60_mean','secondary_episode_paired','safety','switching_mean','ranking_direction')},indent=2))


if __name__ == '__main__':
    main()
