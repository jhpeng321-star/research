"""Receipt-only fresh-state headroom analysis; no simulator/model imports."""
import csv
import json
from pathlib import Path
import statistics

from .core import EPS
from .design import CONTROLLERS, STEP0_SEEDS, STEP0_TIMES, sha
from .storage import canonical, write


def analyze(receipt):
    states = receipt['states']
    expected = {(s,t) for s in STEP0_SEEDS for t in STEP0_TIMES}
    got = {(r['world']['seed'],r['t_s']) for r in states}
    if len(got) != len(states) or got != expected or 'STOP' in receipt:
        raise ValueError('incomplete/stopped Step0; no headroom verdict')
    if receipt['charged_sim_s'] != 47010 or receipt['closed_actual_sim_s_incl_legacy'] != 47010:
        raise ValueError('Step0 accounting mismatch')
    # Independently recompute the execution gate from all exact increment receipts.
    noise = max(abs(r['branches']['native']['90']['team_delay_vs']-
                    r['branches']['native_repeat']['90']['team_delay_vs']) for r in states)
    threshold = max(EPS,2*noise)
    rows, wins, non_dominant, margin = [], {}, {}, {}
    for clock in ('60','90'):
        wins[clock] = dict.fromkeys(CONTROLLERS, 0)
        clock_rows = []
        for state in states:
            branch = state['branches']
            native = branch['native']
            prefix = canonical(native['prefix'])
            for c in CONTROLLERS+('native_repeat',):
                if canonical(branch[c]['prefix']) != prefix:
                    raise ValueError('unmatched full prefix in completed receipt')
            for key in ('60','90','end'):
                if canonical(native[key]) != canonical(branch['native_repeat'][key]):
                    raise ValueError('native repeat mismatch in completed receipt')
            costs = {c:branch[c][clock]['team_delay_vs'] for c in CONTROLLERS}
            ordered = sorted((v,c) for c,v in costs.items())
            tied = [c for c in CONTROLLERS if costs[c] <= ordered[0][0]+EPS]
            unique = tied[0] if len(tied) == 1 else None
            if unique:
                wins[clock][unique] += 1
            row = dict(**state['world'], decision_s=state['t_s'], horizon_s=int(clock),
                       winners='|'.join(tied), unique_winner=unique,
                       margin_vs=ordered[1][0]-ordered[0][0],
                       noise_resolved=ordered[1][0]-ordered[0][0] > threshold,
                       **{c+'_cost_vs':costs[c] for c in CONTROLLERS},
                       **{k:v for k,v in state['observables'].items() if k != 'coverage'})
            rows.append(row);clock_rows.append(row)
        dominant = max(CONTROLLERS, key=wins[clock].get)
        non_dominant[clock] = sum(r['unique_winner'] is not None and r['unique_winner'] != dominant and
                                   r['noise_resolved'] for r in clock_rows)
        margin[clock] = statistics.median(r['margin_vs'] for r in clock_rows)
    everywhere = any(wins['60'][c] >= .95*len(states) and wins['90'][c] >= .95*len(states)
                     for c in CONTROLLERS)
    unresolved = any(n < .05*len(states) for n in non_dominant.values())
    gate = receipt['STEP0_GATE.json']
    passed = not (everywhere or unresolved)
    if gate['pass_'] != passed or gate['unique_wins'] != wins or gate['resolved_non_dominant'] != non_dominant:
        raise ValueError('independent gate reconstruction disagrees with worker')
    conditions = []
    for clock in (60,90):
        for c in CONTROLLERS:
            group = [r for r in rows if r['horizon_s'] == clock and r['unique_winner'] == c]
            value = dict(horizon_s=clock, controller=c, unique_wins=len(group),
                         win_share=len(group)/len(states))
            for k in ('near_count','near_mean_speed_mps','observed_queue_v_lt5',
                      'cav_self_count','observed_cav_fraction','margin_vs'):
                valid = [r[k] for r in group if r[k] is not None]
                value[k+'_median'] = statistics.median(valid) if valid else None
            conditions.append(value)
    return dict(pass_=passed, unique_wins=wins, noise_vs=noise, noise_threshold_vs=threshold,
                resolved_non_dominant=non_dominant, median_margin_vs=margin,
                states=len(states), charged_sim_s=receipt['charged_sim_s'],
                actual_sim_s=receipt['closed_actual_sim_s_incl_legacy'],
                main_authorized=False), rows, conditions


def main():
    source = Path('tools/c1switch/evidence/CPUQ_COLLECT.json')
    summary, rows, conditions = analyze(json.loads(source.read_text()))
    summary['source_sha256'] = sha(source)
    out = Path('docs/paper_results/c1switch')
    for name, values in [('step0b_decisions.csv',rows),('step0b_winner_conditions.csv',conditions)]:
        with (out/name).open('x', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(values[0]))
            writer.writeheader();writer.writerows(values)
    write(out/'step0b_summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
