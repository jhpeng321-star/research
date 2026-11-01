"""Allowlisted existing labels; TRAIN and never-fit DEV remain distinct."""
import csv
import json
from pathlib import Path

import numpy as np

from tools.coopbandit.protocol import checked, sha, observable, option

OPTIONS = ('O0', 'O1', 'O2', 'O3', 'O4')
KEYS = tuple(sorted(observable([], 300, 0)))
EXAM = Path('E:/agent-work/c4-coopbandit/docs/paper_results/coopbandit/full/exam')
LOCAL = Path(__file__).resolve().parents[2]
PLANNER = LOCAL / 'docs/paper_results/coopplan'


def read(path):
    return json.loads(checked(path).read_text(encoding='utf-8'))


def save(path, value):
    with checked(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, sort_keys=True, allow_nan=False)
        f.write('\n')


def training_role(world):
    if type(world) is not int or not 981000 <= world < 981080:
        raise ValueError('only registered EXAM TRAIN worlds may fit/calibrate/select')


def encode(rows, t, setting):
    features = observable(rows, t, setting)
    if tuple(sorted(features)) != KEYS:
        raise ValueError('causal feature schema changed')
    x = np.array([features[k] for k in KEYS], dtype=float)
    if not np.isfinite(x).all():
        raise ValueError('nonfinite observable input')
    return x


def load_exam(root=EXAM):
    root = checked(root)
    manifest = read(root / 'MANIFEST.json')
    required = {
        'states.jsonl': '03b7eaa91fb137c6cf156d05258558c7a9080581eee493ac66ab73ba5921495a',
        'labels.csv': '7202836d4466acdf5705aff40bd965903c7a042dcb3e0e7f829882b6ea74b59b',
        'worlds.json': '85009c43a73fd369bb6a17cb28b491988d6125a1051f9719f4779055b53d8b36',
    }
    for name, digest in required.items():
        if manifest.get(name) != digest or sha(root / name) != digest:
            raise ValueError('original complete EXAM byte hash mismatch: ' + name)
    states = [json.loads(s) for s in checked(root / 'states.jsonl').read_text().splitlines()]
    with checked(root / 'labels.csv').open(newline='') as f:
        labels = list(csv.DictReader(f))
    index = {(r['state_id'], r['option'], int(r['horizon_s'])): r for r in labels}
    if len(states) != 300 or len(index) != len(labels) or len(labels) != 4500:
        raise ValueError('full unique 300-state/4500-label EXAM required')
    records = []
    for s in states:
        training_role(s['world_id'])
        t, setting = s['t0'], int(s['features']['setting'])
        if s['state_id'] != f's{s["world_id"]}_t{t}' or t not in (300,360,420,480,540):
            raise ValueError('invalid state identity/clock')
        features = observable(s['observed_vehicles'], t, setting)
        if features != s['features']:
            raise ValueError('original causal builder feature parity failed')
        effects, available = {}, []
        for o in OPTIONS:
            active = o == 'O0' or option(o, s['observed_vehicles'], setting)['available']
            available.append(active)
            for h in (120,300,600):
                r = index[s['state_id'], o, h]
                if (int(r['world_id']) != s['world_id'] or int(r['t0']) != t or
                        int(r['setting']) != setting or (r['available'] == 'True') != active):
                    raise ValueError('label context/availability mismatch')
                d = float(r['D'])
                if not np.isfinite(d) or (not active or o == 'O0') and d != 0:
                    raise ValueError('invalid no-op/label')
                effects.setdefault(str(h), []).append(d)
        records.append(dict(state_id=s['state_id'], world_id=s['world_id'], role='TRAIN',
                            time_s=t, setting=setting, features=features,
                            available=available, effects=effects))
    groups = [r['world_id'] for r in records]
    if len(set(groups)) != 60 or any(groups.count(g) != 5 for g in set(groups)):
        raise ValueError('60 complete TRAIN worlds required')
    return records, dict(source=str(root), hashes=required, states=300, worlds=60,
                         feature_builder_sha256=sha(LOCAL / 'tools/coopbandit/protocol.py'))


def arrays(records, horizon=600, require_train=True):
    if require_train:
        for r in records:
            if r['role'] != 'TRAIN':
                raise ValueError('never-fit DEV cannot enter fitting/CV')
            training_role(r['world_id'])
    x = np.array([[r['features'][k] for k in KEYS] for r in records], float)
    y = np.array([r['effects'][str(horizon)] for r in records], float)
    available = np.array([r['available'] for r in records], bool)
    groups = np.array([r['world_id'] for r in records])
    return x, y, available, groups


def planner_records():
    """Extract all720 decisions from local committed receipts, no SUMO or remote IO.

    Quota-pruned eligible options are missing labels, NEVER zero/imputed labels.
    H120 and H300 are retained separately; no D600 label is invented.
    """
    result, sources = [], {}
    panels = (('final/RECEIPTS.json', 120, set(range(983100,983112))),
              ('extension/final/RECEIPTS.json', 120, set(range(983200,983224))),
              ('h300/final/RECEIPTS.json', 300, set(range(983300,983324))))
    for name, horizon, expected in panels:
        path = checked(PLANNER / name)
        snapshots = read(path)
        sources[str(path)] = sha(path)
        seen = set()
        for snap in snapshots:
            if not snap['complete'] or snap['stopped'] or snap['incomplete']:
                raise ValueError('incomplete planner source')
            for r in snap['records']:
                seed = r['world']['seed']
                if seed not in expected or seed in seen:
                    raise ValueError('planner registered DEV world mismatch')
                seen.add(seed)
                if len(r['history']) != 12:
                    raise ValueError('planner decision history incomplete')
                for e in r['history']:
                    t = e['time_s']
                    setting = ((t-300)//60) % 4
                    rows = e['initial_rows']
                    means = e['decision']['expected_delay']
                    # Budget fallback can have no evaluation, even for O0.
                    effects = {o: means['O0']-v for o,v in means.items()} if means else {}
                    candidates = {a['id'] for a in e.get('available_candidates', e['candidates'])}
                    actual = {'O0'} | {o for o in OPTIONS[1:] if option(o,rows,setting)['available']}
                    if candidates != actual or not set(effects) <= actual:
                        raise ValueError('planner causal option parity failed')
                    chosen = e['executed_action']['id']
                    teacher = max(effects, key=lambda o:(effects[o], -OPTIONS.index(o))) if effects else 'O0'
                    if teacher != chosen:
                        raise ValueError('teacher argmax label parity failed')
                    result.append(dict(state_id=f's{seed}_t{t}', world_id=seed, role='DEV_NEVER_FIT',
                                       time_s=t, setting=setting, horizon_s=horizon,
                                       features=observable(rows,t,setting), available=sorted(actual),
                                       evaluated_effects=effects, teacher=teacher,
                                       unmeasured=sorted(actual-set(effects))))
        if seen != expected:
            raise ValueError('incomplete planner world panel')
    if len(result) != 720 or len({r['state_id'] for r in result}) != 720:
        raise ValueError('720 unique DEV decision states required')
    return result, sources
