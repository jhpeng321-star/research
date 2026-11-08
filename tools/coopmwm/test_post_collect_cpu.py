"""SPD2: parallel post-collect TTC work is exactly the serial result (values, order, written JSON bytes)."""
import gzip
import hashlib
import json
from pathlib import Path
import random
import shutil

import pytest

from tools.coopmwm import post_collect
from tools.coopmwm.replay_bank import ttc_events


def trace(path, seed, *, vehicles=60, start=290, end=1205):
    rng = random.Random(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wt', encoding='utf-8') as f:
        for t in range(start, end):
            rows = []
            for i in range(vehicles):
                row = dict(vid=f'v{i}', lane=f'e_{rng.randrange(3)}', s_m=rng.uniform(0, 3000),
                           v_mps=rng.uniform(0, 30), is_cav=i % 5 == 0)
                if rng.random() < .9:
                    row['length_m'] = rng.choice([4.5, 5., 12.])
                if row['is_cav']:
                    row.update(issued_acceleration_mps2=rng.uniform(-3, 1.5), controller='policy_mean')
                rows.append(row)
            f.write(json.dumps(dict(time_s=float(t), vehicles=rows, actions=[]), sort_keys=True)+'\n')
    return path


@pytest.fixture(autouse=True)
def _reset():
    post_collect.clear(); post_collect.configure(1)
    yield
    post_collect.clear(); post_collect.configure(1)


def test_prefetch_equals_serial_in_order(tmp_path):
    paths = [trace(tmp_path/f'e{i}'/'observable.jsonl.gz', i, vehicles=40) for i in range(6)]
    serial = [ttc_events(p, 300, 1200) for p in paths]
    assert any(serial) and len(set(serial)) > 1
    order = paths[::-1]+[paths[2]]          # reversed + duplicate: order follows input exactly
    assert post_collect.prefetch(order, 300, 1200, n=4) == [ttc_events(p, 300, 1200) for p in order]
    assert [post_collect.ttc_events_cached(p, 300, 1200) for p in paths] == serial
    assert all(type(v) is float for v in post_collect.prefetch(paths, 300, 1200, n=4))
    post_collect.clear()
    assert post_collect.prefetch(paths, 300, 1200, n=1) == serial


def test_memo_never_reuses_a_changed_file_or_other_window(tmp_path):
    p = trace(tmp_path/'x'/'observable.jsonl.gz', 1, vehicles=40)
    first = post_collect.prefetch([p], 300, 1200, n=2)[0]
    assert post_collect.ttc_events_cached(p, 300, 900) == ttc_events(p, 300, 900)
    trace(p, 2, vehicles=40)                 # rewritten bytes -> new (size, mtime_ns) key
    assert post_collect.ttc_events_cached(p, 300, 1200) == ttc_events(p, 300, 1200) != first


def test_env_override_and_default(monkeypatch):
    monkeypatch.delenv(post_collect.ENV, raising=False)
    assert post_collect.configure({}.get('post_collect_workers', post_collect.DEFAULT_WORKERS)) == 8
    monkeypatch.setenv(post_collect.ENV, '1')
    assert post_collect.configure(8) == 1
    with pytest.raises(ValueError):
        monkeypatch.setenv(post_collect.ENV, '0'); post_collect.configure(8)


def _root(root, n):
    from tools.coopmwm.composite_health import AUTHORITY
    root.mkdir(parents=True)
    (root/'AUTHORITY.md').write_text('amend-sha-x\n')
    (root/'SAFETY_RULE_0957.json').write_text(json.dumps(dict(authority=AUTHORITY, TTC_threshold=1.5,
        bootstrap=10000, bootstrap_seed=20261101, alpha=.05, amendment_sha256='amend-sha-x')))
    a_records, episodes = [], []
    for i in range(n):
        world = dict(seed=1000+i, demand=2000)
        for arm, seed in (('A', 10*i), ('B', 10*i+5)):
            ident = f'warm_u003_{arm}_{1000+i}'
            job = root/'jobs'/ident
            trace(job/'observable.jsonl.gz', seed, vehicles=40)
            rec = dict(id=ident, world=world, metrics={'900': dict(collisions=0, teleports=0,
                                                                     hard_brake=float(i % 3))})
            (job/'RESULT.json').write_text(json.dumps(rec, sort_keys=True))
            (a_records if arm == 'A' else episodes).append(rec)
    return a_records, episodes


def _files(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob('*')) if p.is_file()}


def test_composite_guard_writes_identical_bytes_serial_vs_parallel(tmp_path):
    from tools.coopmwm.composite_health import braking
    serial_root, parallel_root = tmp_path/'serial', tmp_path/'parallel'
    a, b = _root(serial_root, 5)
    shutil.copytree(serial_root, parallel_root)
    pauses = []
    post_collect.configure(1)
    assert braking(serial_root, 'B', 3, b, a, lambda *x: pauses.append(x))
    post_collect.clear(); post_collect.configure(4)
    a2 = [json.loads((parallel_root/'jobs'/r['id']/'RESULT.json').read_text()) for r in a]
    b2 = [json.loads((parallel_root/'jobs'/r['id']/'RESULT.json').read_text()) for r in b]
    # launch_training order: the arm's own traces first, then the guard reuses them and fetches A refs.
    post_collect.prefetch([parallel_root/'jobs'/e['id']/'observable.jsonl.gz' for e in b2], 300, 1200)
    assert braking(parallel_root, 'B', 3, b2, a2, lambda *x: pauses.append(x))
    assert _files(serial_root) == _files(parallel_root)
    assert (serial_root/'B_SAFETY_COMPOSITE_003.json').read_bytes() == \
           (parallel_root/'B_SAFETY_COMPOSITE_003.json').read_bytes()
    assert len(list((serial_root/'safety_diagnosis_0957'/'ttc_cache').iterdir())) == 10


def _episode(rng, i):
    return dict(id=f'g4_u003_B_{i}', world=dict(seed=i, demand=2000), practice=None if i % 2 else dict(slot=f's_p{i}'),
                ttc_events=i, note='caf\u00e9 \u2192 "q" \\ \n\t\x01', empty=[], nested={},
                steps=[dict(time_s=float(t), reward=rng.uniform(-1, 1), big=1e300*rng.random(), tiny=5e-324,
                            ints=[rng.randrange(-10**20, 10**20) for _ in range(3)], flag=t % 2 == 0, none=None,
                            sample=dict(context=dict(members=[f'v{k}' for k in range(t % 4)]),
                                        probs=[rng.random() for _ in range(20)]))
                       for t in range(300, 340)])


def test_g4_input_publish_bytes_identical_serial_vs_parallel(tmp_path):
    from tools.coopmwm import g4_runtime as g4
    rng = random.Random(7)
    value = dict(binding=dict(settings='x', weight=1.0), update=3, arm='B',
                 episodes=[_episode(rng, i) for i in range(7)])
    g4.publish(tmp_path/'serial'/'B_INPUT_003.json', value)
    post_collect.configure(8)
    for n in (2, 4, 8):
        post_collect.publish_episodes(tmp_path/f'p{n}'/'B_INPUT_003.json', value, n)
        assert (tmp_path/f'p{n}'/'B_INPUT_003.json').read_bytes() == (tmp_path/'serial'/'B_INPUT_003.json').read_bytes()
        assert sorted(p.name for p in (tmp_path/f'p{n}').iterdir()) == ['B_INPUT_003.json']   # parts/temp removed
    assert g4.file_sha(tmp_path/'p4'/'B_INPUT_003.json') == g4.published_sha(value)
    assert g4.read(tmp_path/'p4'/'B_INPUT_003.json') == value
    with pytest.raises(FileExistsError):                       # same no-overwrite contract as _publish
        post_collect.publish_episodes(tmp_path/'p4'/'B_INPUT_003.json', value, 4)
    # like g4_runtime._publish, a refused link leaves only its pid temp; the published bytes are untouched
    assert sorted(p.name for p in (tmp_path/'p4').iterdir())[0] == 'B_INPUT_003.json'
    assert g4.file_sha(tmp_path/'p4'/'B_INPUT_003.json') == g4.published_sha(value)
    bad = dict(value, episodes=value['episodes'][:2]+[dict(x=float('nan'))])
    with pytest.raises(ValueError):
        post_collect.publish_episodes(tmp_path/'bad'/'B_INPUT_003.json', bad, 4)
    assert not any((tmp_path/'bad').iterdir())                # no temp/parts/pickles left after a worker error
    assert post_collect.publish_workers() == 8 and post_collect.publish_workers(3) == 3
    post_collect.configure(1)
    assert post_collect.publish_workers(8) == 1                  # serial switch covers both phases


def test_torch_free_chunker_matches_g4_runtime():
    from tools.coopmwm import g4_runtime as g4
    rng = random.Random(3)
    value = dict(z=[_episode(rng, 1)], a={'k': [1, 2.5, None, 'xé'], 'e': {}}, m=[], n={1: 2})
    for depth in range(6):
        assert ''.join(post_collect._value_chunks(value, depth)) == ''.join(g4._value_chunks(value, depth))             == json.dumps(value, sort_keys=True, allow_nan=False)
