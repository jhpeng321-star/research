from pathlib import Path
import json
import numpy as np

base = Path('/root/autodl-tmp/c4')
for p in (base/'e3data', base/'panel120/package', base/'wmdeep20261027/score_pairs'):
    print('PANEL', p, p.is_dir())
    if not p.is_dir():
        continue
    d = json.loads((p/'C4_EPISODES_MANIFEST.json').read_text())
    entries = d['episodes']
    print('N_EP', len(entries), 'FIRST', entries[0])
    e = next((e for e in entries if e.get('e3data_split') == 'validation'), entries[0])
    ep = p/e['dir']
    print('EPFILES', [x.name for x in ep.iterdir()])
    print('EPJSON', (ep/'episode.json').read_text()[:9000])
    for n in ('time_s','vid','entry_backlog','request_counter','counter'):
        q = ep/(n+'.npy')
        if q.is_file():
            a = np.load(q, mmap_mode='r')
            print(n, a.shape, 'T0', a[e.get('t0_index', 0)].tolist()[:12]
                  if a.ndim > 1 else float(a[e.get('t0_index', 0)]))
for q in (base/'cct2inc20261029/score_wm3_panels.py',
          base/'wmcombined20261027/code/score.py',
          base/'wmcombined20261027/runtime/tools/loop14_checks/accept.py'):
    print('FILE', q, q.is_file())
    if q.is_file():
        print(q.read_text()[:25000])
