"""Paper figure and complete per-world/cell exports from recorded fixed-arm receipts."""
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import statistics

from .core import EPS
from .design import CELLS, CONTROLLERS, sha
from .storage import write

OUT = Path('docs/paper_results/c1switch')
LABELS = dict(native='Native', ctrl0='CTRL0', mappo='MAPPO-speed', pi='PI external-U', circles='CIRCLES')


def csv_write(path, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def recorded():
    path = Path('tools/c1switch/evidence/RECORDED.json')
    source = json.loads(path.read_text())
    groups = defaultdict(dict)
    for r in source['rows']:
        key = (r['dataset'], r['seed'], r['penetration'], r['demand'], r['clock_s'])
        if r['arm'] in groups[key] or not math.isfinite(r['team_delay_vs']):
            raise ValueError('duplicate/nonfinite source row')
        groups[key][r['arm']] = r
    if len(groups) != 270:
        raise ValueError('complete90x3 panel required')
    rows, summary, cells = [], [], []
    for (dataset, seed, p, d, clock), arms in sorted(groups.items()):
        if set(arms) != set(CONTROLLERS) or len({r['initial_state_sha256'] for r in arms.values()}) != 1:
            raise ValueError('five-arm/prefix pairing mismatch')
        costs = {c:arms[c]['team_delay_vs'] for c in CONTROLLERS}
        best = min(costs.values())
        tied = [c for c in CONTROLLERS if costs[c] <= best+EPS]
        ordered = sorted(costs.values())
        rows.append(dict(dataset=dataset, seed=seed, penetration=p, demand_vph=d, clock_s=clock,
                         winner='|'.join(tied), tie_count=len(tied), best_team_delay_vs=best,
                         runnerup_margin_vs=ordered[1]-ordered[0],
                         **{c+'_team_delay_vs':costs[c] for c in CONTROLLERS}))
    for dataset in ('WMDATA', 'WMDATA2', 'POOLED'):
        for clock in (120, 300, 600):
            panel = [r for r in rows if r['clock_s'] == clock and (dataset == 'POOLED' or r['dataset'] == dataset)]
            counts = Counter(r['winner'] for r in panel)
            means = {c:statistics.mean(r[c+'_team_delay_vs'] for r in panel) for c in CONTROLLERS}
            summary.append(dict(dataset=dataset, clock_s=clock, n_worlds=len(panel),
                                unique_wins={c:counts[c] for c in CONTROLLERS},
                                ties=sum(r['tie_count'] > 1 for r in panel),
                                fractional_win_shares={c:sum(c in r['winner'].split('|') and 1/r['tie_count'] or 0 for r in panel)/len(panel) for c in CONTROLLERS},
                                best_fixed_mean=min(means,key=means.get), mean_team_delay_vs=means,
                                median_margin_vs=statistics.median(r['runnerup_margin_vs'] for r in panel)))
            if dataset == 'POOLED':
                continue
            for p,d in CELLS:
                group = [r for r in panel if (r['penetration'],r['demand_vph']) == (p,d)]
                for c in CONTROLLERS:
                    cells.append(dict(dataset=dataset, clock_s=clock, penetration=p, demand_vph=d,
                                      controller=c, n_worlds=len(group),
                                      unique_wins=sum(r['winner'] == c for r in group),
                                      fractional_win_share=sum(c in r['winner'].split('|') and 1/r['tie_count'] or 0 for r in group)/len(group)))
    OUT.mkdir(parents=True, exist_ok=True)
    csv_write(OUT/'recorded_world_winners.csv', rows)
    csv_write(OUT/'recorded_cell_winshares.csv', cells)
    write(OUT/'recorded_summary.json', dict(source_export_sha256=sha(path), rows=summary))
    figure(cells)
    print(json.dumps(summary, indent=2))


def figure(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    colors = ['#777777', '#d59a34', '#537ca5', '#36876b', '#a77395']
    fig, axes = plt.subplots(2, 3, figsize=(14.8, 9.7), sharex=True)
    y = np.arange(12)
    for i, dataset in enumerate(('WMDATA', 'WMDATA2')):
        for j, clock in enumerate((120, 300, 600)):
            ax = axes[i,j]
            left = np.zeros(12)
            for c, color in zip(CONTROLLERS, colors):
                group = [next(r for r in rows if (r['dataset'],r['clock_s'],r['penetration'],r['demand_vph'],r['controller']) == (dataset,clock,p,d,c)) for p,d in CELLS]
                values = np.asarray([r['fractional_win_share'] for r in group])*100
                ax.barh(y, values, left=left, color=color, height=.76, label=LABELS[c])
                for k, v in enumerate(values):
                    if v >= 22:
                        ax.text(left[k]+v/2, k, f'{v:.0f}', ha='center', va='center', color='white', fontsize=8)
                left += values
            counts = [next(r['n_worlds'] for r in rows if (r['dataset'],r['clock_s'],r['penetration'],r['demand_vph']) == (dataset,clock,p,d)) for p,d in CELLS]
            ax.set_yticks(y, [f'{p*100:g}% / {d}  (n={n})' for (p,d), n in zip(CELLS, counts)], fontsize=8)
            ax.invert_yaxis()
            ax.set_xlim(0,100)
            ax.set_xticks([0,25,50,75,100])
            ax.set_title(f'{dataset} · cumulative {clock} s', fontsize=11, loc='left')
            ax.set_xlabel('Fractional share of minimum team-delay worlds (%)', fontsize=9)
            if j == 0:
                ax.set_ylabel('Scenario CAV share (%) / demand (vehicles/hour)', fontsize=9)
            ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Which fixed controller wins depends on traffic conditions and horizon', x=.07, ha='left', fontsize=17)
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=5, loc='upper center', bbox_to_anchor=(.55,.935), frameon=False, fontsize=10)
    fig.text(.07,.022, 'Source: recorded S-WMDATA (40 worlds) and S-WMDATA2 (50 worlds), 31 Oct 2026. Native warmup 300 s; team delay = fleet + source (veh-s).\nEach world is counted once per clock; exact ties split credit. Unequal cell sample sizes shown. TRAIN/descriptive fixed episodes; not per-decision switching evidence.', fontsize=9)
    fig.subplots_adjust(left=.18,right=.985,top=.88,bottom=.11,wspace=.57,hspace=.38)
    fig.savefig(OUT/'recorded_winner_by_condition.png', dpi=190)
    fig.savefig(OUT/'recorded_winner_by_condition.pdf')
    plt.close(fig)


if __name__ == '__main__':
    recorded()
