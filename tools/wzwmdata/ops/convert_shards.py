"""WM-B: convert published shards into WM-A c4_episodes sources (zero physics), registered split.

Run on gpuA with WM-A's converter (c4/wz-wm-port) first on PYTHONPATH:
    python convert_shards.py <shards dir> <packs dir> <MAP_SPEC.json> <shard k> [<shard k> ...]
Per shard: <packs>/shardKK_train and <packs>/shardKK_val (--val-frac 0: the registered world split
of the shard decides the role). Canary tapes (map only in manifest.json) are skipped.
"""
import json
import subprocess
import sys
from pathlib import Path

shards, packs, spec = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
packs.mkdir(parents=True, exist_ok=True)
procs = []
for k in sys.argv[4:]:
    body = json.loads((shards / f'SHARD_{int(k):02d}.json').read_text())
    for split in ('train', 'val'):
        dirs = [e['episode_dir'] for e in body['episodes'] if e['split'] == split and not e['canary']]
        if not dirs:
            continue
        out = packs / f'shard{int(k):02d}_{split}'
        log = (packs / f'shard{int(k):02d}_{split}.log').open('x')
        cmd = [sys.executable, '-m', 'wms.data.c4_episodes', '--episodes', *dirs, '--out', str(out),
               '--map-spec', spec, '--val-frac', '0']
        procs.append((out, len(dirs), subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)))
for out, n, p in procs:
    rc = p.wait()
    print(json.dumps(dict(out=str(out), episodes=n, returncode=rc)), flush=True)
