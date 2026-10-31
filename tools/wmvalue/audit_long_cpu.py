"""Actual frozen WM90s on a small synthetic prefix, no real outcome/seed access."""
import argparse
import json
import time

import numpy as np

from .audit import ARMS, digest, safe
from .restart import prefix


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--check-prefix', action='store_true')
    a = p.parse_args()
    from . import runtime as r
    begun = time.monotonic()
    r.torch.set_num_threads(1)
    times = np.arange(0., 300.5, .5)
    shape = (len(times), 2)
    tape = dict(time_s=times, s_m=np.zeros(shape), lane=np.zeros(shape, np.int64),
        v_mps=np.full(shape, 8.5), a_mps2=np.zeros(shape), present=np.zeros(shape, bool),
        is_cav=np.zeros(shape, bool), vid_index=np.full(shape, -1, np.int64),
        vehicle_ids=np.array(['synthetic.mainline.0', 'synthetic.mainline.1']))
    for j, birth in enumerate((0., 5.)):
        use = times >= birth
        tape['present'][use, j] = True
        tape['vid_index'][use, j] = j
        tape['is_cav'][use, j] = j == 1
        tape['s_m'][use, j] = -2645.67+8.5*(times[use]-birth)
    pref = prefix(tape, 300)
    rt = r.load('cpu')
    batches, events = [], []
    for arm in ARMS:
        b, ev = r.make_batch(pref, tape['vehicle_ids'].tolist(), rt['model'], [0., 0.], {})
        batches.append(b)
        events.append(ev)
    result = r.rollout(rt, batches, events, 300, 0, 'cpu', seconds=90)
    assert result['steps'] == 180 and set(result['windows']) == {'60', '90'}
    for h in (60, 90):
        w = result['windows'][str(h)]
        x = np.asarray(w['x'])
        assert x.shape == (5, 111) and np.isfinite(x).all()
        assert np.all(x[:, -1] == 600-h)
        assert np.all(np.asarray(w['J']) <= np.asarray(result['J'])+1e-9)
    assert result['windows']['90']['J'] == result['J']
    np.testing.assert_allclose(result['J'], np.array(result['road'])+np.array(result['source']), atol=1e-9)
    if a.check_prefix:
        for fi in (0, 3):
            short = r.rollout(rt, batches, events, 300, fi, 'cpu', seconds=90)
            longer = r.rollout(rt, batches, events, 300, fi, 'cpu', seconds=120)
            assert short['windows'] == longer['windows'], 'truncation changed a bootstrap checkpoint'
    doc = dict(schema='wmvalue_full90_cpu_audit/1', verdict='PASS', synthetic_only=True,
        real_outcomes_opened=False, frozen_WM=True, device='cpu', steps=180,
        terminal_windows=[60, 90], all_five_arms=True, features_per_terminal=111,
        exact_90_endpoint=True, seconds=time.monotonic()-begun,
        exact_prefix_parity_futures=[0, 3] if a.check_prefix else [],
        runtime_sha256=digest(safe(__file__).parent/'runtime.py'), results=result)
    with safe(a.out).open('x') as stream:
        json.dump(doc, stream, indent=1, allow_nan=False)
    print(json.dumps({k: v for k, v in doc.items() if k != 'results'}), flush=True)


if __name__ == '__main__':
    main()
