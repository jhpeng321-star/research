"""Figures for the A72 claim-validation package. Values are read from recorded JSON only (no new statistics).
Run: C:/Users/jpeng/miniconda3/envs/rl_driving/python.exe -I make_figures.py
Greyscale-safe: identity is carried by marker shape / hatch and labels, not colour.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401

CV = Path('E:/research-dense/experiments/claim_validation_20261113_0231')
OUT = CV / 'deliverables' / 'figures'
FB = CV / 'deliverables' / 'RAW_RESULTS' / 'e1_bank_box4_extracted' / 'bank' / 'FROZEN_BANKS.json'
E1A = CV / 'e1_conf' / 'E1_ANALYSIS.json'
E2A = CV / 'box4_out' / 'e2' / 'E2_ANALYSIS.json'
SEEDS = [11, 23, 42, 67, 89, 103]
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})


def fig_a():
    p = json.loads(FB.read_text())['primary_admitted_count']
    x = range(len(SEEDS))
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    w = 0.38
    ax.bar([i - w / 2 for i in x], p['WM'], w, color='0.25', edgecolor='black', label=f"WM order (total {sum(p['WM'])})")
    ax.bar([i + w / 2 for i in x], p['RANDOM'], w, color='white', edgecolor='black', hatch='//',
           label=f"Random order (total {sum(p['RANDOM'])})")
    ax.set_xticks(list(x), [str(s) for s in SEEDS])
    ax.set_xlabel('Search seed')
    ax.set_ylabel('Plans admitted (of first 6)')
    ax.set_ylim(0, 6.5)
    ax.set_yticks(range(7))
    ax.grid(axis='y', color='0.85', lw=0.6)
    ax.set_axisbelow(True)
    perm = p['permutation']
    ax.set_title(f"E1 primary: admitted count, WM - Random mean {perm['mean']:.1f}, "
                 f"exact sign-flip p = {perm['p_two_sided']:.2f}", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8.5, loc='upper left')
    fig.tight_layout()
    fig.savefig(OUT / 'fig_e1_admitted_counts.png', dpi=200)
    plt.close(fig)


def fig_b():
    b6 = json.loads(E1A.read_text())['by_budget']['6']
    per = b6['wm_minus_random_per_seed']
    ov = b6['wm_minus_random']
    labels = [f'seed {s}' + (' (WM bank = U0)' if s == 42 else '') for s in SEEDS] + ['overall (6 seeds x 96 worlds)']
    y = list(range(len(labels)))[::-1]
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.axvline(0, color='black', lw=0.8)
    ax.plot(per, y[:-1], 'o', color='0.3', ms=7, label='per seed (point only; no per-seed CI recorded)')
    lo, hi = ov['ci95']
    ax.errorbar([ov['diff']], [y[-1]], xerr=[[ov['diff'] - lo], [hi - ov['diff']]], fmt='D', color='black',
                ms=8, capsize=4, lw=1.5, label='overall, 95% CI (seed x world bootstrap)')
    ax.set_yticks(y, labels)
    ax.set_xlabel('WM bank - Random bank, total delay (veh-s per world)\nnegative = WM bank less delay')
    ax.grid(axis='x', color='0.85', lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title(f"E1 secondary, budget-6 banks, 96 worlds per seed: not separated\n"
                 f"overall +{ov['diff']:,.0f} [{lo:,.0f}, +{hi:,.0f}] veh-s, "
                 f"sign-flip p = {b6['permutation']['p_two_sided']:.3f}", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8, loc='upper center', bbox_to_anchor=(0.4, -0.25), ncol=1)
    fig.tight_layout()
    fig.savefig(OUT / 'fig_e1_bank_contrast.png', dpi=200)
    plt.close(fig)


def fig_c():
    a = json.loads(E2A.read_text())
    left = [('F - H20 (PRIMARY)', a['primary']), ('F - S (secondary)', a['F_minus_S'])]
    right = [(f'{k} - U0 (secondary)', a['vs_u0'][k]) for k in ('F', 'H20', 'S')]
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.4))
    for ax, rows, title in ((axes[0], left, 'Between bank arms'), (axes[1], right, 'Each arm vs reference U0')):
        y = list(range(len(rows)))[::-1]
        for yi, (lab, c) in zip(y, rows):
            lo, hi = c['ci95']
            mk = 'D' if 'PRIMARY' in lab else 'o'
            ax.errorbar([c['diff']], [yi], xerr=[[c['diff'] - lo], [hi - c['diff']]], fmt=mk, color='black',
                        ms=7, capsize=4, lw=1.5)
            ax.annotate(f"{c['diff']:+,.0f} [{lo:+,.0f}, {hi:+,.0f}]  p={c['p_two_sided']:.3g}",
                        (c['diff'], yi), xytext=(0, 9), textcoords='offset points', ha='center', fontsize=7.5)
        ax.axvline(0, color='black', lw=0.8)
        ax.set_yticks(y, [r[0] for r in rows])
        ax.set_ylim(min(y) - 0.6, max(y) + 0.7)
        ax.set_xlabel('Difference in total delay (veh-s per world)\nnegative = treatment less delay')
        ax.grid(axis='x', color='0.85', lw=0.6)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
        ax.set_title(title, fontsize=9.5)
    fig.suptitle('E2 (96 confirmation worlds, paired percentile bootstrap 10,000, 95% CI; separate panels = separate x scales)',
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / 'fig_e2_arm_differences.png', dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    fig_a()
    fig_b()
    fig_c()
    print('ok')
