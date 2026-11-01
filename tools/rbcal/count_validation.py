"""Exact count-distribution CRPS/coverage for world-excluded arrival scalars."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import nbinom, poisson

from tools.rbunc.arrival_prefix import discrete_crps
from tools.wmdecide.report import write_json


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--fit', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    records = json.loads((args.fit/'arrival_records.json').read_bytes())
    validation = json.loads((args.fit/'validation.json').read_bytes())
    if {r['world'] for r in records} != set(range(969400, 969432)) or len(records) != 64:
        raise ValueError('frozen FIT32 entrance observations required')
    dispersion = {r['world']:r['parameters']['arrival_variance_multiplier'] for r in validation['crossfits']}
    scores = []
    for row in records:
        count, target = row['prefix'], row['target']
        for name, multiplier in [('baseline', 1.), ('scalar_world_excluded', dispersion[row['world']])]:
            distribution = nbinom((count+.5)/multiplier, 1/(1+multiplier)) if multiplier else poisson(count+.5)
            intervals = []
            for level in (.5, .8, .9):
                lo, hi = distribution.ppf([(1-level)/2, (1+level)/2])
                intervals.append(dict(level=level, low=int(lo), high=int(hi),
                    covered=bool(lo <= target <= hi),
                    discrete_nominal_mass=float(distribution.cdf(hi)-distribution.cdf(lo-1))))
            scores.append(dict(row, method=name, multiplier=multiplier,
                crps=discrete_crps(distribution, target), nll=float(-distribution.logpmf(target)),
                intervals=intervals))
    summaries = []
    for entrance in (None, 0, 1):
        for name in ('baseline', 'scalar_world_excluded'):
            selected = [r for r in scores if r['method'] == name and (entrance is None or r['entrance'] == entrance)]
            summaries.append(dict(method=name, entrance=entrance, worlds=32, outcomes=len(selected),
                crps=float(np.mean([r['crps'] for r in selected])), nll=float(np.mean([r['nll'] for r in selected])),
                coverage={str(level):float(np.mean([r['intervals'][i]['covered'] for r in selected]))
                          for i, level in enumerate((.5, .8, .9))},
                discrete_nominal_mass={str(level):float(np.mean([r['intervals'][i]['discrete_nominal_mass'] for r in selected]))
                          for i, level in enumerate((.5, .8, .9))}))
    result = dict(schema='rb2cal.count_validation/1', summary=summaries, records=scores,
        qualification='FIT scalar crossfits only; prefix-count outcomes, NOT traffic-delay coverage',
        interval_reference='inclusive discrete quantiles; exact nominal mass reported; no full-world CIs claimed')
    write_json(args.out, result)
    print(json.dumps(summaries, indent=1))


if __name__ == '__main__':
    main()
