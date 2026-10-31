"""Normalize S-SUMOCORE/other simulator observables without running a simulator.

Manifest: {model, model_sha256, config_sha256, sources: {path: sha256},
tasks: [{seed,future,episodes:[{arm,initial_sha256,trajectory_npz OR bins,
clocks:{'120':{team_delay_vs,road_delay_vs,source_wait_vs},'300':...,'600':...}}]}]}.
NPZ allowlist: t,s,v,lane,present (present optional for flat observed samples).
bins contains three regions, each with exactly 60 post-t0 speed/post_counts bins.
"""
import argparse
import json
from pathlib import Path

import numpy as np

if __package__:
    from .core import (ARMS, HERE, HS, REGIONS, SEEDS, attach_events, digest, load,
                       reduce_trajectory, safe, save_new)
else:
    from core import (ARMS, HERE, HS, REGIONS, SEEDS, attach_events, digest, load,
                      reduce_trajectory, safe, save_new)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    manifest_path = safe(args.manifest)
    manifest = load(manifest_path)
    out = safe(args.out)
    assert out.parent.is_dir()
    out.mkdir(exist_ok=False)
    for path, sha in manifest['sources'].items():
        assert digest(safe(manifest_path.parent/path)) == sha
    identity = dict(schema='rbeval/1', K=8, arms=ARMS, model=manifest['model'],
        wm_sha256=manifest['model_sha256'], config_sha256=manifest['config_sha256'],
        sources=manifest['sources'], switches=manifest.get('switches', {}),
        candidate_sha256=manifest.get('candidate_sha256'), export_sha256=manifest.get('export_sha256'),
        real_receipt_sha256=digest(HERE/'real_observables.json'), seed=manifest.get('rng_seed'),
        update=None, t0=300, horizon=600, mix=manifest.get('mix', False),
        native_lanes=manifest.get('native_lanes', True),
        external_manifest_sha256=digest(manifest_path), adapter_sha256=digest(__file__))
    save_new(out/'IDENTITY.json', identity)
    identity_sha = digest(out/'IDENTITY.json')
    real = {(e['seed'], e['arm']): e for e in load(HERE/'real_observables.json')['episodes']}
    seen = set()
    for task in manifest['tasks']:
        s, k = task['seed'], task['future']
        assert s in SEEDS and k in range(8) and (s, k) not in seen
        seen.add((s, k))
        assert {e['arm'] for e in task['episodes']} == set(ARMS) and len(task['episodes']) == 5
        episodes, checks = [], []
        for e in task['episodes']:
            arm = e['arm']
            assert e['initial_sha256'] == real[s, arm]['initial_sha256']
            if 'trajectory_npz' in e:
                source = safe(manifest_path.parent/e['trajectory_npz'])
                assert digest(source) == e['trajectory_sha256']
                with np.load(source, allow_pickle=False) as f:
                    t, pos, v, lane = [f[n] for n in ('t', 's', 'v', 'lane')]
                    present = f['present'] if 'present' in f.files else np.ones_like(v, bool)
                bins = reduce_trajectory(t, pos, v, lane, present)
                checks.append(dict(arm=arm, source=str(source), sha256=digest(source)))
            else:
                bins = e['bins']
                for reg in REGIONS:
                    assert len(bins[reg]['speed']) == len(bins[reg]['post_counts']) == 60
                    assert all(isinstance(n, int) and n >= 0 and
                        ((v is None) == (n == 0)) and (v is None or np.isfinite(v))
                        for v, n in zip(bins[reg]['speed'], bins[reg]['post_counts']))
            for h in HS:
                c = e['clocks'][str(h)]
                assert set(c) == {'team_delay_vs', 'road_delay_vs', 'source_wait_vs'}
                assert np.isfinite(list(c.values())).all()
                assert abs(c['team_delay_vs']-c['road_delay_vs']-c['source_wait_vs']) < 1e-4
            episodes.append(dict(seed=s, future=k, arm=arm, initial_sha256=e['initial_sha256'],
                bins=attach_events(bins, real[s, arm]), clocks=e['clocks']))
        save_new(out/f'w{s}_k{k}.json', dict(schema='rbeval/1', seed=s, future=k, arms=ARMS,
            identity_sha256=identity_sha, episodes=episodes, observable_sources=checks))
    save_new(out/'IMPORTED.json', dict(tasks=len(seen), manifest_sha256=digest(manifest_path)))
    print(json.dumps(dict(tasks=len(seen), out=str(out))))


if __name__ == '__main__':
    main()
