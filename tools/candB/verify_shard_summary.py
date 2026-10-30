"""Independent raw-row WORLD bootstrap check, without importing the summarizer."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.distill.runtime import safe_path

METRICS = ('J90', 'J300', 'objective90_loss', 'objective300_loss')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(summary_path):
    summary = json.loads(summary_path.read_text())
    arms = {}
    verified_inputs = {}
    for directory, item in summary['inputs'].items():
        path = safe_path(item['path'])
        assert digest(path) == item['sha256']
        verified_inputs[directory] = item['sha256']
        raw = json.loads(path.read_text())
        for i, row in enumerate(raw['rows']):
            index = row.get('plan_index', i + raw.get('start_index', 0))
            for arm, value in row['arms'].items():
                by_index = arms.setdefault(arm, {})
                by_index.setdefault(index, (row['metadata'][0]['world'], value,
                                             row['paired_CTRL0'], row['paired_native_physical']))

    def check(rows, expected, metrics=METRICS):
        worlds = sorted(set(r[0] for r in rows))
        assert expected['n_starts'] == len(rows) and expected['n_worlds'] == len(worlds)
        draw = np.random.default_rng(20261030).integers(0, len(worlds), (5000, len(worlds)))
        for metric in metrics:
            # Average futures, then repeated starts inside a world, then worlds.
            by_world = np.array([[np.mean([np.mean(row[which][metric]) for row in rows
                                          if row[0] == world]) for which in (1, 2)]
                                 for world in worlds])
            for key, values in (('value', by_world[:, 0]),
                                ('paired_delta', by_world[:, 0] - by_world[:, 1])):
                actual = np.r_[values.mean(), np.percentile(values[draw].mean(1), [2.5, 97.5])]
                saved = expected[metric][key]
                np.testing.assert_allclose(actual, [saved['mean'], saved['lo'], saved['hi']],
                                           atol=1e-9, rtol=1e-10)

    canonical = list(arms[summary['canonical_reference_arm']].values())
    for arm, expected in summary['per_arm'].items():
        if arm == 'CTRL0':
            check([(w, ref, ref, native) for w, _, ref, native in canonical], expected)
        elif arm == 'native':
            check([(w, native, ref, native) for w, _, ref, native in canonical], expected,
                  ('J90', 'J300'))
        else:
            check(list(arms[arm].values()), expected)
    common = set(summary['common_start_indices'])
    for arm, expected in summary['common'].items():
        if arm in ('native', 'CTRL0'):
            chosen = [r for i, r in arms[summary['canonical_reference_arm']].items() if i in common]
            if arm == 'native':
                check([(w, native, ref, native) for w, _, ref, native in chosen], expected,
                      ('J90', 'J300'))
            else:
                check([(w, ref, ref, native) for w, _, ref, native in chosen], expected)
        else:
            check([r for i, r in arms[arm].items() if i in common], expected)
    if summary.get('native_actor_K4'):
        check(list(arms['native'].values()), summary['native_actor_K4'])
    return dict(independent_world_bootstrap='PASS', draws=5000, seed=20261030,
                summary_sha256=digest(summary_path), verified_inputs=verified_inputs,
                per_arm_starts={arm: x['n_starts'] for arm, x in summary['per_arm'].items()},
                common_starts=len(common), no_rollouts=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = verify(safe_path(args.summary))
    safe_path(args.out).write_text(json.dumps(result, indent=1) + '\n', encoding='utf-8')
    print(result)
