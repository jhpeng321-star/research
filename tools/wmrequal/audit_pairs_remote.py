"""CPU-only identifiability audit of registered pairs300 tapes; no assignments."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np

p = Path('/root/autodl-tmp/c4/wmrequal20261030')
real = json.loads((p/'REAL.json').read_text())['panels']['pairs300']
rows = []
for r in real:
    ep = Path(r['input_dir'])
    # Restrict inspection to the recorded identities/counters and metadata only.
    meta = json.loads((ep/'episode.json').read_text())
    ts = np.load(ep/'time_s.npy', mmap_mode='r', allow_pickle=False)
    names = [n for n in ('vehicle_ids.json', 'vehicle_ids.npy', 'identities.json', 'snapshot.json')
             if (ep/n).is_file()]
    rows.append(dict(pair_key=r['pair_key'], arm=r['arm'], queue0_total=r['queue0_total'],
                     queue0=r['queue0'], tape_start=float(ts[0]), warmup_s=meta['warmup_s'],
                     identity_name_candidates=names,
                     metadata_sha256=hashlib.sha256((ep/'episode.json').read_bytes()).hexdigest()))
# These are exact metadata-derived original recorded directories, not searches.
first = json.loads((Path(real[0]['input_dir'])/'episode.json').read_text())
original = []
for key, value in (('base_episode', first['branch']['base_episode']), ('source', first['source'])):
    for prefix in ('/root/autodl-tmp/c4', '/root/autodl-tmp/c4/v5'):
        q = Path(value.replace('/home/ubuntu/c4', prefix))
        original.append(dict(kind=key, path=str(q), exists=q.is_dir()))
result = dict(schema='wmrequal_pairs_identifiability/1', episodes=len(rows),
              keys=len({r['pair_key'] for r in rows}),
              missing_per_source_Q0=sum(r['queue0'] is None for r in rows),
              identity_name_candidates=sum(bool(r['identity_name_candidates']) for r in rows),
              original_recorded_directories=original, rows=rows,
              conclusion='Exact total backlog is identified; per-source initial backlog is not. '
                         'K8 scorer refuses guessed queue allocation. No pairs300 scoring launched.')
q = p/'PAIRS300_BLOCKER.json'
if q.exists():
    assert json.loads(q.read_text()) == result
else:
    q.write_text(json.dumps(result, indent=1)+'\n')
print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, indent=1))

owned = []
for name in ('LAUNCH.json', 'FAST_LAUNCH.json', 'COMPACT_PROBE_LAUNCH.json',
             'DENSE_RESUME_LAUNCH.json', 'OWNED_CHILDREN.jsonl', 'FAST_CHILDREN.jsonl',
             'DENSE_RESUME_CHILDREN.jsonl'):
    q = p/name
    if not q.is_file():
        continue
    records = [json.loads(s) for s in q.read_text().splitlines()] if name.endswith('.jsonl') else [json.loads(q.read_text())]
    for r in records:
        stat = Path(f'/proc/{r["pid"]}/stat')
        state = 'absent'
        if stat.is_file():
            fields = stat.read_text().rsplit(')', 1)[1].split()
            state = 'different_incarnation' if fields[19] != r['start_ticks'] else fields[0]
        assert state in ('absent', 'different_incarnation', 'Z'), (r['pid'], state)
        owned.append(dict(receipt=name, pid=r['pid'], start_ticks=r['start_ticks'], state=state))
release = dict(time=time.time(), owned_active_jobs=0, jobs=owned,
               scope='Only task receipt PIDs checked; no peer process inspection or signals.')
(p/'GPU_RELEASE.json').write_text(json.dumps(release, indent=1)+'\n')
print('GPU_RELEASE', json.dumps(release))
