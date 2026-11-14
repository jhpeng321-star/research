"""MAIN13:35 B/native secondary contrast, immutable fixed-u8 supplement only."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from tools.coopmwm.branch_eval import cluster_bootstrap, in_focus
from tools.coopmwm.protocol import checked, digest, sha
from tools.coopmwm.storage import read, write


def analyze(records, panel):
    table = {(e['arm'], e['world']['seed']): e for e in records}
    expected = {(a, w['seed']) for a in ('A', 'B', 'R', 'mappo', 'native') for w in panel}
    if len(table) != len(records) or set(table) != expected:
        raise ValueError('complete unique original five-arm grid required')
    if any(e['world'] != next(w for w in panel if w['seed'] == e['world']['seed'])
           for e in records):
        raise ValueError('registered cell mismatch')
    result = {}
    for h in ('600', '900'):
        for scope, ws in dict(all=panel, focus=[w for w in panel if in_focus(w)]).items():
            if not ws:
                raise ValueError('focus scope cannot be empty')
            for metric in ('team_delay_vs', 'collisions', 'teleports', 'ttc_below_3_vs', 'hard_brake'):
                arrays = [np.array([[table[a, w['seed']]['metrics'][h][metric]] for w in ws])
                          for a in ('B', 'native')]
                row = cluster_bootstrap(*arrays)
                if not np.isfinite(row['pct_delay_reduction']):
                    row['pct_delay_reduction'] = None
                row['pct_ci95'] = [float(z) if np.isfinite(z) else None for z in row['pct_ci95']]
                result[f'{h}:{scope}:B-native:{metric}'] = row
    return dict(results=result, worlds=len(panel), focus_worlds=sum(in_focus(w) for w in panel),
        contrast='B u8 minus native; negative delay is better',
        role='secondary descriptive fixed-u8 interim supplement; u24 remains primary',
        multiplicity='no adjusted confirmatory claim for native; original Holm families unchanged',
        bootstrap='10000 whole-world paired resamples; RNG20261101; percentile 95% CI',
        no_training_selection=True, new_sim_s=0)


def run(root, control, registration):
    import fcntl
    import os
    from interim_u8_analysis import metadata, load_metrics
    from tools.coopmwm.accel_protocol import worlds
    root, control = checked(root), checked(control)
    lock = (control/'RUN.claim').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (control/'COMPLETE.json').exists():
        return read(control/'COMPLETE.json')
    if (control/'STARTED.json').exists():
        raise ValueError('adopt existing interrupted supplement; never duplicate')
    reg = read(registration)
    if reg['amendment_sha256'] not in (root/'AUTHORITY.md').read_text():
        raise ValueError('supplement authority missing')
    if sha(Path(__file__)) != reg['driver_sha256']:
        raise ValueError('supplement driver changed')
    if sha(Path(reg['original_driver'])) != reg['original_driver_sha256']:
        raise ValueError('qualified original loader changed')
    original = read(reg['original_registration'])
    for name, h in original['frozen_dependencies'].items():
        if sha(Path(original['source_root'])/name) != h:
            raise ValueError('qualified dependency changed')
    panel = worlds('final_test')
    manifest = metadata(root, panel)
    if digest(panel) != reg['panel_sha256'] or digest(manifest) != reg['public_manifest_sha256']:
        raise ValueError('original registered interim grid changed')
    write(control/'STARTED.json', dict(unix=time.time(), pid=os.getpid(), new_sim_s=0))
    result = dict(analyze(load_metrics(root, manifest), panel), unix=time.time(),
                  amendment_sha256=reg['amendment_sha256'], manifest_sha256=digest(manifest))
    write(control/'RESULT.json', result)
    complete = dict(pass_=True, unix=time.time(), new_sim_s=0, training_unchanged=True,
                    result_sha256=sha(control/'RESULT.json'), report_to='MAIN only')
    write(control/'COMPLETE.json', complete)
    return complete


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('root', 'control', 'registration'):
        p.add_argument(name)
    a = p.parse_args()
    try:
        print(json.dumps(run(Path(a.root), Path(a.control), Path(a.registration))))
    except BaseException as e:
        write(Path(a.control)/'FAILED.json', dict(error=repr(e), new_sim_s=0, unix=time.time()))
        raise
