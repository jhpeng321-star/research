"""Parallel per-episode post-collect trace work (SPD2, 2026-11-08); results are exactly the serial ones.

The only per-episode, path-addressed work in the update phase after collection is the TTC<1.5 s
count over each episode's ``observable.jsonl.gz`` (launch_training's per-episode ``ttc_events``
and the composite safety guard's A-reference / arm episodes). Workers receive ONLY a trace path
and the window and return one float computed by the unchanged ``replay_bank.ttc_events``; the
parent keeps every episode/reward/G4 object, its order and every JSON byte it writes.

``ttc_events_cached`` returns a value that ``prefetch`` already computed for the SAME file
(resolved path + size + mtime_ns) and window, else computes serially exactly as before. Workers
default to 1 (serial) unless ``configure`` is called; launch_training configures
``cfg.get('post_collect_workers', 8)``; ``C4_POST_COLLECT_WORKERS`` overrides (1 = old serial path).
"""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from pathlib import Path

ENV = 'C4_POST_COLLECT_WORKERS'
DEFAULT_WORKERS = 8
_state = dict(workers=1)
_memo = {}


def configure(workers):
    value = os.environ.get(ENV)
    workers = int(value) if value not in (None, '') else int(workers)
    if workers < 1:
        raise ValueError('post_collect_workers must be >= 1')
    _state['workers'] = workers
    return workers


def workers():
    return _state['workers']


def _key(path, start, end, threshold):
    p = Path(path).resolve()
    st = p.stat()
    return (str(p), st.st_size, st.st_mtime_ns, start, end, threshold)


def _one(args):
    from .replay_bank import ttc_events
    path, start, end, threshold = args
    return ttc_events(Path(path), start, end, threshold)


def _threshold(threshold):
    if threshold is None:
        from .replay_bank import TTC_EVENT_S
        return TTC_EVENT_S
    return threshold


def prefetch(paths, start, end, threshold=None, *, n=None):
    """Compute ttc_events for every path not yet memoized, in parallel; returns values in input order."""
    threshold = _threshold(threshold)
    n = workers() if n is None else int(n)
    paths = [Path(p) for p in paths]
    keys = [_key(p, start, end, threshold) for p in paths]
    todo = []
    for p, k in zip(paths, keys):
        if k not in _memo and k not in {key for _, key in todo}:
            todo.append((p, k))
    if todo:
        jobs = [(str(p), start, end, threshold) for p, _ in todo]
        if n <= 1 or len(jobs) <= 1:
            values = [_one(job) for job in jobs]
        else:
            with ProcessPoolExecutor(max_workers=min(n, len(jobs)),
                                     mp_context=multiprocessing.get_context('spawn')) as pool:
                values = list(pool.map(_one, jobs))
        for (_, k), v in zip(todo, values):
            _memo[k] = v
    return [_memo[k] for k in keys]


def ttc_events_cached(path, start, end, threshold=None):
    """Drop-in for replay_bank.ttc_events(path, start, end): memoized value of the same file, else serial."""
    threshold = _threshold(threshold)
    k = _key(path, start, end, threshold)
    if k not in _memo:
        _memo[k] = _one((str(path), start, end, threshold))
    return _memo[k]


def clear():
    _memo.clear()


# ---- G4 input publication (B_INPUT/R_INPUT): per-episode JSON encoding in worker processes ----------------
# Bytes are exactly g4_runtime.publish(path, value): json.dumps(value, sort_keys=True, allow_nan=False)+'\n'.
# The parent pickles each episode to a hand-off file next to the target (bounded: no in-memory copy); a worker
# loads ONE episode, deletes the hand-off, streams its JSON to its own part file (no full string kept); the
# parent writes the top-level frame and appends the parts in episode order (deleting each after copying) into the
# same no-overwrite temp -> fsync -> link publication. Memory: <= publish-workers episodes resident in workers.
# Transient disk: pending pickles (~1/3 of their JSON) + not-yet-appended parts.
PUBLISH_ENV = 'C4_POST_COLLECT_PUBLISH_WORKERS'


def publish_workers(cfg_value=None):
    """Encoder count: env override, else cfg ``post_collect_publish_workers``, else the configured TTC count."""
    value = os.environ.get(PUBLISH_ENV)
    n = int(value) if value not in (None, '') else int(workers() if cfg_value is None else cfg_value)
    if n < 1:
        raise ValueError('post_collect_publish_workers must be >= 1')
    return n if workers() > 1 else 1


def _value_chunks(value, depth):
    """Same contract/bytes as g4_runtime._value_chunks (kept torch-free for the spawned encoders; tested equal)."""
    import json
    if depth > 0 and isinstance(value, dict) and value and all(type(k) is str for k in value):
        yield '{'
        for i, key in enumerate(sorted(value)):
            yield (', ' if i else '')+json.dumps(key)+': '
            yield from _value_chunks(value[key], depth-1)
        yield '}'
    elif depth > 0 and isinstance(value, (list, tuple)) and value:
        yield '['
        for i, item in enumerate(value):
            if i:
                yield ', '
            yield from _value_chunks(item, depth-1)
        yield ']'
    else:
        yield json.dumps(value, sort_keys=True, allow_nan=False)


def _encode_part(args):
    episode, part = args
    if isinstance(episode, str):        # pickle hand-off file (avoids the slow single IPC pipe for GB objects)
        import pickle
        with open(episode, 'rb') as stream:
            data = pickle.load(stream)
        os.unlink(episode)
        episode = data
    size = 0
    with open(part, 'xb') as stream:
        for chunk in _value_chunks(episode, 3):
            data = chunk.encode()
            stream.write(data)
            size += len(data)
    return size


def publish_episodes(path, value, n):
    """g4_runtime.publish(path, value) with value['episodes'] encoded by ``n`` workers; identical bytes."""
    import json
    import shutil
    from . import g4_runtime as g4
    episodes = value.get('episodes')
    if n <= 1 or not isinstance(episodes, list) or len(episodes) < 2 or not all(type(k) is str for k in value):
        return g4.publish(path, value)
    path = g4.tb._path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    parts_dir = path.with_name('.'+path.name+f'.{os.getpid()}.parts')
    parts_dir.mkdir()
    try:
        import pickle
        parts = [parts_dir/f'{i:05}.json' for i in range(len(episodes))]
        temp = path.with_name(path.name+f'.{os.getpid()}.tmp')
        keys = sorted(value)
        with temp.open('xb') as stream, ProcessPoolExecutor(max_workers=min(n, len(episodes)),
                mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = []
            for i, (episode, part) in enumerate(zip(episodes, parts)):
                handoff = parts_dir/f'{i:05}.pkl'
                with handoff.open('xb') as dump:
                    pickle.dump(episode, dump, protocol=pickle.HIGHEST_PROTOCOL)
                futures.append(pool.submit(_encode_part, (str(handoff), str(part))))
            stream.write(b'{')
            for i, key in enumerate(keys):
                stream.write(((', ' if i else '')+json.dumps(key)+': ').encode())
                if key != 'episodes':
                    stream.write(json.dumps(value[key], sort_keys=True, allow_nan=False).encode())
                    continue
                stream.write(b'[')
                for j, (future, part) in enumerate(zip(futures, parts)):
                    future.result()                 # in episode order; a worker error propagates here
                    if j:
                        stream.write(b', ')
                    with part.open('rb') as source:
                        shutil.copyfileobj(source, stream, 1 << 24)
                    part.unlink()                   # transient disk: only not-yet-copied parts remain
                stream.write(b']')
            stream.write(b'}\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, path)
        temp.unlink()
        if os.name == 'posix':
            fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    except BaseException:
        temp = path.with_name(path.name+f'.{os.getpid()}.tmp')
        if temp.exists() and not path.exists():
            temp.unlink()                           # never published; nothing partial left behind
        raise
    finally:
        for part in list(parts_dir.glob('*.json'))+list(parts_dir.glob('*.pkl')):
            part.unlink()
        parts_dir.rmdir()
