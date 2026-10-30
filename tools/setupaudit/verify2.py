"""Offline receipt checks; file inputs must pass the same prohibited-path guard."""
import argparse
import json
from pathlib import Path

import numpy as np

from inspect_inputs import allowed


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--probe', required=True)
    p.add_argument('--original', required=True)
    a = p.parse_args()
    assert allowed(a.probe) and allowed(a.original)
    probe = json.loads(Path(a.probe).read_text())
    original = json.loads(Path(a.original).read_text())
    for row in probe['rows']:
        fi = row['future']
        for actor in ('CTRL0', 'MAPPO'):
            reference = next(r for r in original['rows'] if r['actor'] == actor)['dense']['all_J_futures'][fi]
            measured = sum(row['costs']['time_bins_30s']['J'][probe['actors'].index(actor)])
            np.testing.assert_allclose(measured, reference, rtol=0, atol=5e-5)
            print(fi, actor, 'new=', measured, 'original=', reference, 'delta=', measured-reference)
        native_refs = [r['dense']['native_J_futures'][fi] for r in original['rows']]
        assert len(set(native_refs)) == 1
        native = sum(row['costs']['time_bins_30s']['J'][probe['actors'].index('native')])
        np.testing.assert_allclose(native, native_refs[0], rtol=0, atol=5e-5)
        source = row['costs']['time_bins_30s']['source']
        assert source[0] == source[-1]
        assert row['costs']['births'][0] == row['costs']['births'][-1]
    print('PASS: real-checkpoint CPU adapter match and frozen source/birth accounting')


if __name__ == '__main__':
    main()
