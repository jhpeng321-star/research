"""Once-only reserved FIT outcomes export, gated by the completed CPU adapter audit.

No fitting or selection; this is prepare_causal's fixed real clock/projection on
the registered test cells. Future request schedules remain label-only.
"""
import argparse
import json

import numpy as np

from .audit import ARMS, allowed_seed, digest, safe
from .core import TIMES, features
from .reconstruct import OBS_KEYS, command_summaries, label_clock, observations


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--gate', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    out, root = safe(a.out), safe(a.root)
    gate = json.loads(safe(a.gate).read_text())
    assert gate['verdict'] == 'PASS' and gate['terminal_60_90_path']
    assert digest(safe(__file__).parent/'reconstruct.py') == gate['code_sha256']['reconstruct.py']
    if out.exists() or not out.parent.is_dir():
        raise ValueError('new output and existing parent required')
    x, y, act, seeds, times, arms, records = [], [], [], [], [], [], []
    for seed in range(969432, 969440):
        allowed_seed(seed, 'test')
        for arm in ARMS:
            job = safe(root/'jobs'/f's{seed}_{arm}')
            ep = safe(job/'deploy/episode/episode.npz')
            manifest = json.loads(safe(job/'deploy/episode/manifest.json').read_text())
            assert manifest['seed'] == seed and manifest['status'] == 'complete'
            with np.load(ep, allow_pickle=False) as z:
                tape = {k: z[k] for k in (*OBS_KEYS, 'vehicle_ids')}
            objectives = {str(h): json.loads(safe(job/f'objective_{h}.json').read_text()) for h in (120, 300, 600)}
            clock = label_clock(tape, seed, float(manifest['demand_vph']), objectives)
            known = {t: {str(tape['vehicle_ids'][i]) for i in tape['vid_index'][t*2][
                           tape['present'][t*2] & tape['is_cav'][t*2]]} for t in TIMES}
            log = safe(job/'deploy/episode/run/action_timeline.jsonl.gz')
            summary = command_summaries(log, TIMES, known)
            for state in observations(tape, TIMES):
                t = int(state['time_s'])
                x.append(features(state))
                y.append(float(clock['remaining'][(t-300)*2]))
                act.append(summary[t])
                seeds.append(seed)
                times.append(t)
                arms.append(arm)
            records.append(dict(seed=seed, arm=arm, episode_sha256=digest(ep), command_sha256=digest(log),
                objectives_sha256={str(h): digest(job/f'objective_{h}.json') for h in (120, 300, 600)},
                clock_differences=clock['reconciled']))
    assert len(x) == 200
    with out.open('xb') as stream:
        np.savez_compressed(stream, x=np.stack(x), y=np.array(y), actions=np.stack(act),
                            seed=np.array(seeds), time_s=np.array(times), arm=np.array(arms))
    receipt = dict(schema='wmvalue_reserved_real/1', states=200, worlds=8, records=records,
                   arrays_sha256=digest(out), gate_sha256=digest(a.gate), fit_or_selection=False)
    with out.with_suffix('.json').open('x') as stream:
        json.dump(receipt, stream, indent=1, allow_nan=False)
    print(json.dumps({k: v for k, v in receipt.items() if k != 'records'}), flush=True)


if __name__ == '__main__':
    main()
