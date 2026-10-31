"""Export approved raw prefixes only, without reading any real outcome file."""
import argparse
import gzip
import io
import json
import zipfile

import numpy as np

from .audit import ARMS, allowed_seed, digest, safe
from .core import TIMES
from .reconstruct import OBS_KEYS
from .restart import prefix


def export(root, out, seeds):
    import torch
    root, out = safe(root), safe(out)
    if out.exists() or not out.parent.is_dir():
        raise ValueError('new archive with existing parent required')
    receipts = []
    with zipfile.ZipFile(out, 'x', compression=zipfile.ZIP_DEFLATED) as bundle:
        for seed in seeds:
            allowed_seed(seed, 'fit' if seed <= 969431 else 'test')
            for arm in ARMS:
                job = safe(root/'jobs'/f's{seed}_{arm}')
                ep = safe(job/'deploy/episode/episode.npz')
                with np.load(ep, allow_pickle=False) as z:
                    # Immediately truncate; load only whitelisted observables.
                    data = {k: z[k][:1081] for k in OBS_KEYS}
                    seen = np.unique(data['vid_index'][data['present'].astype(bool)])
                    registry = z['vehicle_ids']
                    names = np.array([str(registry[int(i)]) for i in seen])
                    ids = np.full(data['vid_index'].shape, -1, dtype=np.int64)
                    use = data['present'].astype(bool)
                    ids[use] = np.searchsorted(seen, data['vid_index'][use])
                    data['vid_index'] = ids
                    data['vehicle_ids'] = names
                # Validate every decision prefix, without inspecting its future.
                for t in TIMES:
                    prefix(data, t)
                stpath = safe(job/'start.pt')
                st = torch.load(stpath, map_location='cpu', weights_only=False)
                q = list(map(float, st['observed_queue_history'][-1][1:]))
                assert sum(q) == st['queue'] and st['t0'] == 300
                commands = {str(t): {} for t in TIMES}
                log = safe(job/'deploy/episode/run/action_timeline.jsonl.gz')
                with gzip.open(log, 'rt', encoding='utf-8') as stream:
                    for line in stream:
                        d = json.loads(line)
                        t = float(d['timestamp_s'])
                        if t > 539:
                            break
                        for start in TIMES:
                            if t == start-1 and d['issued_command'] is not None:
                                commands[str(start)][d['vehicle_id']] = float(d['issued_command'])
                record = dict(seed=seed, arm=arm, queue300=q, previous_commands=commands,
                              episode_sha256=digest(ep), start_sha256=digest(stpath),
                              commands_sha256=digest(log), prefix_endpoint=540,
                              outcomes_opened=False, source_prior='recorded warmup only')
                stream = io.BytesIO()
                np.savez_compressed(stream, **data)
                name = f's{seed}_{arm}'
                bundle.writestr(name+'.npz', stream.getvalue())
                bundle.writestr(name+'.json', json.dumps(record, allow_nan=False))
                receipts.append(record)
                print(name, 'prefix exported', flush=True)
        bundle.writestr('RECEIPT.json', json.dumps(dict(schema='wmvalue_prefix/1', records=receipts,
            reserved_outcomes_opened=False, source_features=False), allow_nan=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--fit-canary', action='store_true')
    a = p.parse_args()
    export(a.root, a.out, [969400] if a.fit_canary else range(969432, 969440))
