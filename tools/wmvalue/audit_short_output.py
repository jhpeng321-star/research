"""Check first completed short-task bindings without loading real labels or V."""
import argparse
import json

import numpy as np

from .audit import ARMS, digest, safe
from .evaluate_frozen import WM_SHA
from .finalize_short import OLD_WORKER
from .orchestrate import save_new


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--short', required=True)
    p.add_argument('--original', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    root, original = safe(a.short), safe(a.original)
    ack = json.loads((root/'ACK.json').read_text())
    reference = sorted((original/'tasks').glob('*.json'))[0]
    ref = json.loads(reference.read_text())
    sources = dict(ref['source_sha256'])
    assert sources.pop('worker') == OLD_WORKER
    paths = sorted((root/'tasks').glob('*.json'))
    assert paths, 'no completed short task yet'
    for path in paths:
        d = json.loads(path.read_text())
        source = dict(d['source_sha256'])
        assert source.pop('worker') == ack['worker_sha256']
        diff = [k for k in set(source) | set(sources) if source.get(k) != sources.get(k)]
        assert not diff, repr(diff)
        assert d['end_time'] == d['t0']+90 and d['results']['steps'] == 180
        assert d['arms'] == list(ARMS) and d['K'] == 8 and d['mix']
        assert not d['real_outcomes_opened'] and d['wm'][0]['checkpoint_sha256'] == WM_SHA
        assert d['gate_sha256'] == ack['gate_sha256'] == ref['gate_sha256']
        for k in ('prefixes_sha256', 'candidate_sha256', 'export_sha256'):
            assert d[k] == ref[k]
        assert d['wm'][0]['config_sha256'] == ref['wm'][0]['config_sha256']
        for h in (60, 90):
            w = d['results']['windows'][str(h)]
            x = np.asarray(w['x'])
            assert x.shape == (5, 111) and np.isfinite(x).all()
            assert np.all(x[:, -1] == 900-d['t0']-h)
        assert d['results']['windows']['90']['J'] == d['results']['J']
    result = dict(verdict='PASS', schema='wmvalue_short_binding/1',
        checked=len(paths), real_outcomes_opened=False, frozen_models_opened=False,
        identical_runtime_source_count=len(sources), only_registered_worker_driver_changed=True,
        reference_sha256=digest(reference), task_sha256={p.name: digest(p) for p in paths})
    save_new(safe(a.out), result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
