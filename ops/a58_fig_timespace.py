"""AMENDMENT 58 illustration (not evidence): time-space diagram, MAPPO (U0) | searched plan (A42 BIG_BANK), one world.

Inputs: the two rerun episode directories (ground-truth per-vehicle log vehicles.jsonl.gz at 1 s: s, v, a, lane per
vehicle; observable.jsonl.gz: the per-step controller of every CAV, 'setpoint' = commanded by an active plan).
Lane view: the closing lane (WM lane 2, ends at the drop, s = 75.44 m) and the adjacent through lane (WM lane 1);
a vehicle's line is drawn while it is in one of these two lanes. Lines coloured by speed (one sequential colormap,
shared colour bar); plan-member CAVs drawn thicker in one contrasting colour while their controller is 'setpoint'.

python ops/a58_fig_timespace.py --u0 <dir> --plan <dir> --out <dir> [--world <id>]
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

DROP_S_M = 75.44          # wz3to2: the closing lane ends here (MAP_SPEC bottleneck; lane_drop.MapView)
LANES = (1, 2)            # WM lane indices: 2 = closing lane, 1 = adjacent through lane
Y_RANGE = (-2650.0, 450.0)
V_MAX = 31.29             # speed limit (m/s)
MEMBER_COLOUR = '#d62728'


def read_truth(d):
    ids, rows = {}, []
    with gzip.open(Path(d)/'vehicles.jsonl.gz', 'rt', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            if 'veh' in r:
                ids[r['veh']] = r['id']
            elif 't' in r:
                rows.append(r)
    return ids, rows


def read_setpoint(d):
    on = set()
    with gzip.open(Path(d)/'observable.jsonl.gz', 'rt', encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            t = int(round(r['time_s']))
            for v in r['vehicles']:
                if v.get('controller') == 'setpoint':
                    on.add((t, v['vid']))
    return on


def trajectories(d):
    """{vid: arrays t, s, v, lane, member} at 1 s."""
    ids, rows = read_truth(d)
    on = read_setpoint(d)
    tr = {}
    for r in rows:
        t = int(r['t'])
        for k, s, v, ln in zip(r['k'], r['s'], r['v'], r['lane']):
            vid = ids[k]
            tr.setdefault(vid, []).append((t, s, v, ln, (t, vid) in on))
    return {vid: np.array(x, dtype=float) for vid, x in tr.items()}, len(on)


def segments(tr):
    """Speed-coloured segments between consecutive 1 s samples both in the displayed lanes; member segments apart."""
    seg, col, mseg = [], [], []
    for x in tr.values():
        for i in range(len(x)-1):
            a, b = x[i], x[i+1]
            if b[0]-a[0] != 1 or a[3] not in LANES or b[3] not in LANES:
                continue
            p = [(a[0], a[1]), (b[0], b[1])]
            seg.append(p)
            col.append(0.5*(a[2]+b[2]))
            if a[4] and b[4]:
                mseg.append(p)
    return seg, np.array(col), mseg


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--u0', required=True)
    p.add_argument('--plan', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--world', default='')
    a = p.parse_args(argv)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8,
                         'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8, 'pdf.fonttype': 42,
                         'ps.fonttype': 42, 'axes.linewidth': 0.6})
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9), sharey=True, constrained_layout=True)
    cmap = plt.get_cmap('viridis')
    stats = {}
    lc = None
    for ax, (name, d) in zip(axes, (('MAPPO (U0)', a.u0), ('Searched plan (bank of 5-6-CAV holds)', a.plan))):
        tr, n_on = trajectories(d)
        seg, col, mseg = segments(tr)
        lc = LineCollection(seg, cmap=cmap, norm=plt.Normalize(0, V_MAX), linewidths=0.25, capstyle='butt')
        lc.set_array(col)
        ax.add_collection(lc)
        if mseg:
            ax.add_collection(LineCollection(mseg, colors=MEMBER_COLOUR, linewidths=1.0, capstyle='butt', zorder=3))
        ax.axhline(DROP_S_M, color='k', lw=0.8, ls='--', zorder=4)
        ax.set_xlim(300, 1200)
        ax.set_ylim(*Y_RANGE)
        ax.set_xlabel('Time (s)')
        ax.set_title(name, pad=3)
        ax.tick_params(length=2.5, width=0.6)
        stats[name] = dict(vehicles=len(tr), segments=len(seg), member_segments=len(mseg), setpoint_rows=n_on)
    axes[0].set_ylabel('Position toward lane drop (m)')
    axes[0].text(310, DROP_S_M+60, 'lane drop', fontsize=8, va='bottom', ha='left',
                 bbox=dict(boxstyle='square,pad=0.15', fc='white', ec='none'), zorder=5)
    axes[1].legend(handles=[Line2D([], [], color=MEMBER_COLOUR, lw=1.0, label='plan-member CAV (active hold)')],
                   loc='lower right', frameon=True, framealpha=0.9, edgecolor='none', borderpad=0.3,
                   handlelength=1.6)
    cb = fig.colorbar(lc, ax=axes, pad=0.01, aspect=30, fraction=0.03)
    cb.set_label('Speed (m/s)')
    cb.ax.tick_params(length=2.5, width=0.6)
    cb.outline.set_linewidth(0.6)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out/'fig_timespace.pdf')
    fig.savefig(out/'fig_timespace.png', dpi=300)
    print(json.dumps(dict(world=a.world, lanes=LANES, drop_s_m=DROP_S_M, stats=stats)))


if __name__ == '__main__':
    main()
