"""MAIN12:35's fixed-u8 analysis only; never opens final-endpoint outcomes."""
import argparse
from pathlib import Path
import time

import numpy as np
from tools.coopmwm.accel_storage import read
from tools.coopmwm.storage import write
from tools.coopmwm.protocol import checked, digest, sha
from tools.coopmwm.accel_protocol import worlds
from tools.coopmwm.branch_eval import cluster_bootstrap, holm, in_focus

ARMS = ('A', 'B', 'R', 'mappo', 'native')
CONTRASTS = (('B', 'A'), ('B', 'mappo'), ('B', 'R'), ('B', 'native'), ('A', 'mappo'), ('R', 'mappo'))
METRICS = ('team_delay_vs', 'collisions', 'teleports', 'ttc_below_3_vs', 'hard_brake')


def metadata(root, panel):
    """Validate the entire public completion grid before any outcome read."""
    root = checked(root)
    for complete, failed in (
        ('replay_composite_0957/interim_COMPLETE.json', 'replay_composite_0957/interim_FAILED.json'),
        ('interim_r_1235_v2/COMPLETE.json', 'interim_r_1235_v2/FAILED.json'),
    ):
        if not checked(root/complete).is_file() or checked(root/failed).exists():
            raise ValueError('interim not completely certified; outcomes remain closed')
    result = []
    checkpoints = {a: sha(root/f'{a}_checkpoint_008.pt') for a in ARMS[:3]}
    for a in ARMS:
        for w in panel:
            name = f'interim008_{a}_{w["seed"]}' if a in checkpoints else f'fixedfinal_{a}_{w["seed"]}'
            p = checked(root/'final_receipts'/(name+'.json'))
            v = read(p)
            if (v['id'] != name or v['world'] != w or v['arm'] != a
                    or v['kind'] != 'episode' or v['sim_s'] != 1200 or not v['outcomes_closed']):
                raise ValueError('public completion identity mismatch')
            if a in checkpoints and (v['checkpoint_update'] != 8 or v['checkpoint_sha256'] != checkpoints[a]):
                raise ValueError('fixed-u8 checkpoint mismatch')
            if 'RESULT.json' not in v['files']:
                raise ValueError('no completed result fingerprint')
            result.append(dict(arm=a, world=w, id=name, public_sha256=sha(p), files=v['files']))
    return result


def load_metrics(root, manifest, reader=read):
    """Read only allowlisted completed u8/MAPPO/native episodes, never u24/A48."""
    out = []
    for item in manifest:
        expected = ('interim008_' if item['arm'] in ARMS[:3] else 'fixedfinal_')+item['arm']+'_'+str(item['world']['seed'])
        if item['id'] != expected:
            raise ValueError('episode outside fixed interim allowlist')
        base = checked(Path(root)/'jobs'/item['id'])
        for name in ('RESULT.json', 'RESULT.json.gz'):
            if name in item['files'] and sha(base/name) != item['files'][name]['sha256']:
                raise ValueError('completed outcome bytes changed')
        ep = reader(checked(base/'RESULT.json'))
        if ep['id'] != item['id'] or ep['world'] != item['world'] or ep['arm'] != item['arm']:
            raise ValueError('episode identity differs from public completion')
        out.append(dict(id=ep['id'], arm=ep['arm'], world=ep['world'], metrics=ep['metrics']))
    return out


def analyze(records, panel):
    table = {(e['arm'], e['world']['seed']): e for e in records}
    if len(table) != len(records) or set(table) != {(a, w['seed']) for a in ARMS for w in panel}:
        raise ValueError('complete unique five-arm panel required')
    if any(table[a, w['seed']]['world'] != w for a in ARMS for w in panel):
        raise ValueError('registered world/cell mismatch')
    scopes = dict(all=panel, focus=[w for w in panel if in_focus(w)])
    results = {}
    for h in ('600', '900'):
        for scope, ws in scopes.items():
            for a, b in CONTRASTS:
                for metric in METRICS:
                    x = np.array([[table[a, w['seed']]['metrics'][h][metric]] for w in ws])
                    y = np.array([[table[b, w['seed']]['metrics'][h][metric]] for w in ws])
                    row = cluster_bootstrap(x, y)
                    # Percentage change is undefined for zero-count safety metrics.
                    # Keep their absolute differences/CIs and serialize null percent.
                    if not np.isfinite(row['pct_delay_reduction']):
                        row['pct_delay_reduction'] = None
                    row['pct_ci95'] = [float(z) if np.isfinite(z) else None for z in row['pct_ci95']]
                    results[f'{h}:{scope}:{a}-{b}:{metric}'] = row
        # Interim episode analogue: two scopes per comparator, separately at each
        # endpoint. No claim of multiplicity control across endpoints/interim/final.
        for b in ('A', 'mappo', 'R', 'native'):
            keys = [f'{h}:{scope}:B-{b}:team_delay_vs' for scope in scopes]
            for k, adjusted in holm({k: results[k]['p_two_sided'] for k in keys}).items():
                results[k]['holm_within_endpoint_comparator'] = adjusted
    cells = {str(c): {a: {h: {m: float(np.mean([e['metrics'][h][m] for e in records
                    if e['arm'] == a and e['world']['cell'] == c])) for m in METRICS}
                    for h in ('600', '900')} for a in ARMS}
             for c in sorted({w['cell'] for w in panel})}
    return dict(results=results, cells=cells, worlds=len(panel), focus_worlds=len(scopes['focus']),
        checkpoints=dict(A=8, B=8, R=8), role='fixed interim; u24 remains primary',
        sign='treatment minus comparator; negative delay is better; positive percent reduction is better',
        bootstrap='10000 whole-world paired resamples; RNG20261101; 95% percentile intervals',
        holm='alpha .05; {all,focus} within each B-A/B-MAPPO/B-R/B-native comparator and endpoint separately; no cross-endpoint or interim/final correction; added secondary families do not change the original families',
        safety='existing episode metrics, reported individually; no new thresholds or selection',
        no_training_selection=True)


def run(root, control, registration):
    import fcntl
    import os
    root, control = checked(root), checked(control)
    lock = (control/'RUN.claim').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (control/'COMPLETE.json').exists():
        return read(control/'COMPLETE.json')
    if (control/'STARTED.json').exists():
        raise ValueError('adopt interrupted analysis; never duplicate')
    reg = read(registration)
    if reg['amendment_sha256'] not in (root/'AUTHORITY.md').read_text():
        raise ValueError('analysis authority missing')
    if sha(Path(__file__)) != reg['driver_sha256']:
        raise ValueError('analysis driver changed')
    for name, h in reg['frozen_dependencies'].items():
        if sha(Path(reg['source_root'])/name) != h:
            raise ValueError('qualified analysis dependency changed')
    panel = worlds('final_test')
    if digest(panel) != reg['panel_sha256']:
        raise ValueError('registered panel changed')
    manifest = metadata(root, panel)
    write(control/'MANIFEST.json', dict(entries=manifest, sha256=digest(manifest), outcomes_opened=False))
    write(control/'STARTED.json', dict(unix=time.time(), pid=os.getpid(), authority=reg['authority'],
        outcomes='only completed fixed-u8 and reused MAPPO/native episodes; no final outcomes'))
    result = dict(analyze(load_metrics(root, manifest), panel), unix=time.time(),
        manifest_sha256=digest(manifest), amendment_sha256=reg['amendment_sha256'], new_sim_s=0)
    write(control/'RESULT.json', result)
    complete = dict(unix=time.time(), pass_=True, worlds=96, checkpoints=dict(A=8, B=8, R=8),
        result_sha256=sha(control/'RESULT.json'), manifest_sha256=digest(manifest),
        new_sim_s=0, report_to='MAIN only', training_unchanged=True)
    write(control/'COMPLETE.json', complete)
    return complete


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('root');p.add_argument('control');p.add_argument('registration')
    a = p.parse_args()
    import os
    os.nice(max(0, 19-os.getpriority(os.PRIO_PROCESS, 0)))
    try:
        result = run(Path(a.root), Path(a.control), Path(a.registration))
    except BaseException as e:
        write(Path(a.control)/'FAILED.json', dict(unix=time.time(), error_type=type(e).__name__,
            new_sim_s=0, analysis_only=True))
        raise
    print(__import__('json').dumps(result))
