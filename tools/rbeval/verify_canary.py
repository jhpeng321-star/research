"""Verify canary schema/detector and historical four-arm cost parity."""
import json
import numpy as np
from core import ARMS, HERE, HS, REGIONS, breakdown, digest, load, save_new


def main():
    root = HERE/'evidence/wm2_shard0'
    task = load(root/'w969420_k0.json')
    assert task['identity_sha256'] == digest(root/'IDENTITY.json')
    old = load('E:/agent-work/c4-wmcalib/tools/wmcalib/data/wm40/wm_gpu_shard2/w20_k0.json')
    diffs = {}
    for h in HS:
        for key in ('real_J', 'real_road', 'real_source'):
            x = np.array(task['costs'][str(h)][key][1:])
            y = np.array(old['costs'][str(h)][key])
            diffs[f'{h}/{key}'] = float(np.max(np.abs(x-y)))
    for e in task['episodes']:
        assert e['arm'] in ARMS
        for reg in REGIONS:
            b = e['bins'][reg]
            assert len(b['speed']) == 66 and len(b['post_counts']) == 60
            for h in HS:
                got, expected = breakdown(b['speed'][:6+h//10]), b['events'][str(h)]
                for key in ('event', 'onset_s', 'preexisting'):
                    assert got.get(key) == expected.get(key)
                for key in ('drop_mps', 'bic_improvement'):
                    if key in got:
                        assert np.isclose(got[key], expected[key], atol=1e-8, rtol=0)
    result = dict(seed=969420, future=0, task_sha256=digest(root/'w969420_k0.json'),
        historical_task_sha256=digest('E:/agent-work/c4-wmcalib/tools/wmcalib/data/wm40/wm_gpu_shard2/w20_k0.json'),
        five_arm_schema=True, detector_recheck=True, max_absolute_cost_difference=diffs,
        seconds=task['seconds'], peak_reserved_mib=task['peak_reserved_mib'])
    save_new(HERE/'CANARY_VERIFIED.json', result)
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
