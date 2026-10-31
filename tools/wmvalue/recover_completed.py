"""Recover complete local transfers using exact worker DONE hashes, no inference."""
import argparse
import json

from .audit import digest, safe
from .orchestrate import fetch, save_new, task_path


def recover(root, key):
    log = root/'logs'/(key+'.log')
    rows = []
    for line in log.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    done = [r for r in rows if r.get('status') == 'DONE']
    assert len(done) == 1, 'exactly one completed remote worker required'
    task = task_path(root, key)
    claim = json.loads((root/'claims'/(key+'.json')).read_text())
    if not task.is_file() or digest(task) != done[0]['sha256']:
        partial_sha = digest(task) if task.is_file() else None
        parent = root/'recovered_tasks'
        parent.mkdir(exist_ok=True)
        for i in range(1, 101):
            target = parent/(key+f'.copy{i}.json')
            if target.exists():
                continue  # Preserve every partial copy, never overwrite/delete.
            fetch(claim['host'], claim['remote'], target)
            assert digest(target) == done[0]['sha256'], 'recovered transfer still partial'
            save_new(parent/(key+'.receipt.json'), dict(path=str(target.relative_to(root)),
                sha256=digest(target), original_partial_sha256=partial_sha,
                worker_done_log_sha256=digest(log), inference_repeated=False))
            task = target
            break
        else:
            raise RuntimeError('too many preserved partial copies; audit manually')
    d = json.loads(task.read_text())
    assert [d['seed'], d['t0'], d['future']] == claim['task']
    assert d['end_time'] in (d['t0']+90, 900)
    assert d['results']['steps'] == 2*(d['end_time']-d['t0'])
    return task


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--key', required=True)
    a = p.parse_args()
    root = safe(a.root)
    task = recover(root, a.key)
    log = root/'logs'/(a.key+'.log')
    claim = json.loads((root/'claims'/(a.key+'.json')).read_text())
    receipt = dict(sha256=digest(task), worker_done_log_sha256=digest(log),
        host=claim['host'], device=claim['device'], completed_transfer_recovered=True,
        inference_repeated=False)
    save_new(root/'claims'/(a.key+'.complete.json'), receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
