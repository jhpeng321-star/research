"""Adopt peer TRAIN-only partial findings as smooth state priorities, never moves."""
import csv
import json
import numpy as np
from .protocol import checked,digest,sha


def fit(exam):
    exam = checked(exam)
    states_path,labels_path = exam/'states.jsonl',exam/'labels.csv'
    states = [json.loads(s) for s in states_path.read_text().splitlines()]
    with labels_path.open(newline='') as f:
        labels = list(csv.DictReader(f))
    if not states or any(not 981000 <= int(s['world_id']) <= 981079 for s in states):
        raise ValueError('only reserved COOPBANDIT TRAIN worlds may supply a prior')
    x,y,world_ids = [],[],[]
    for s in states:
        available = [float(r['D']) for r in labels if r['state_id'] == s['state_id']
                     and r['horizon_s'] == '600' and r['available'] == 'True' and r.get('option') != 'O0']
        if not available:
            continue
        f = s['features']
        x.append([sum(f[f'l{i}_cavs'] for i in range(3)),
                  sum(f[f'l{i}_density'] for i in range(3))/3.,float(f['l2_count'] > 0)])
        y.append(max(available))
        world_ids.append(int(s['world_id']))
    if len(set(world_ids)) < 7:
        raise ValueError('need >=7 complete TRAIN worlds for a contextual prior')
    x,y = np.asarray(x),np.asarray(y)
    scale = x.std(0)
    scale[scale == 0] = 1.
    z = (x-x.mean(0))/scale
    # Standardised signed effect; this only biases state sampling, with ordinary support retained.
    target = (y-y.mean())/max(float(y.std()),1.)
    coef = np.linalg.solve(z.T@z+np.eye(3),z.T@target)/scale
    source = dict(states_sha256=sha(states_path),labels_sha256=sha(labels_path))
    return dict(scope='TRAIN',fit_worlds=sorted(set(world_ids)),states=len(x),source=source,
                source_sha256=digest(source),coefficients=dict(zip(('cav_count','density','ramp'),map(float,coef))),
                purpose='state priority only; peer move identities are not inputs or actions')
