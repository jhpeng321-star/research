"""MAIN21:12 necessary-condition gate. Same draws/statistics, no fabricated A."""
from itertools import combinations

import numpy as np

from .audit import ARMS, allowed_seed
from .core import TIMES, finite, metrics

METHODS = ('B', 'C', 'D', 'E')


def evaluate_short(rows):
    cells = {}
    for row in rows:
        key = allowed_seed(row['seed'], 'test'), row['t0'], row['arm']
        if key in cells:
            raise ValueError('duplicate cell')
        finite([row[m] for m in ('real', *METHODS)])
        cells[key] = row
    expected = {(s, t, a) for s in range(969432, 969440) for t in TIMES for a in ARMS}
    if set(cells) != expected:
        raise ValueError('require all 200 cells')
    pairs = []
    for s in range(969432, 969440):
        for t in TIMES:
            for a, b in combinations(ARMS, 2):
                pairs.append({m: cells[s, t, b][m]-cells[s, t, a][m]
                              for m in ('real', *METHODS)})
    values = {m: np.array([p[m] for p in pairs]).reshape(8, 50) for m in ('real', *METHODS)}
    real = values['real']
    point = {m: metrics(real.ravel(), values[m].ravel()) for m in METHODS}
    boot = {m: np.empty((10000, 3)) for m in METHODS}
    rng = np.random.default_rng(20261031)
    for i in range(10000):
        worlds = rng.integers(0, 8, 8)
        for m in METHODS:
            boot[m][i] = metrics(real[worlds].ravel(), values[m][worlds].ravel())
    summary = {}
    for m in METHODS:
        ci = np.percentile(boot[m], [2.5, 97.5], axis=0)
        summary[m] = {name: dict(value=float(point[m][j]), ci95=ci[:, j].tolist())
                      for j, name in enumerate(('sign_accuracy', 'spearman', 'mae'))}
        summary[m]['predicted_zero_pairs'] = int((values[m] == 0).sum())
    contrasts, survivors = {}, []
    for m in ('B', 'C'):
        passes = []
        for ref in ('D', 'E'):
            sci = np.percentile(boot[m][:, 0]-boot[ref][:, 0], [2.5, 97.5])
            mci = np.percentile(boot[ref][:, 2]-boot[m][:, 2], [2.5, 97.5])
            passes.extend([bool(sci[0] > 0), bool(mci[0] > 0)])
            contrasts[f'{m}-{ref}'] = dict(sign_gain=float(point[m][0]-point[ref][0]),
                sign_ci95=sci.tolist(), mae_reduction=float(point[ref][2]-point[m][2]),
                mae_ci95=mci.tolist())
        if all(passes):
            survivors.append(m)
    return dict(schema='wmvalue_short_results/1', n_worlds=8, n_pairs=400,
        real_zero_pairs=int((real == 0).sum()), metrics=summary, contrasts=contrasts,
        survivors=survivors, verdict='NEEDS_A' if survivors else 'FAIL',
        bootstrap=dict(unit='whole world', draws=10000, seed=20261031),
        policy_training_authorized=False, A_evaluated=False)
