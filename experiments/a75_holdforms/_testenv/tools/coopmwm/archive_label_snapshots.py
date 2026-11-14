"""Archive completed label-runner copies; original bank inputs stay on cpuQ.

MAIN14:50 candidate (a), restricted to SNAPSHOT.pt copies for successful final
labels. Bank reconstruction uses collection/snapshots + pending + episode RESULT,
not label_attempts/.../branches/SNAPSHOT.pt (replay_bank._ls_snapshot). Runner
*.result.json return copies are read only by SubprocessBranchRunner.run while a
decision is running. Successful final labels have durable verification receipts;
those decisions are never replayed. All bank inputs, branch RESULT.json(.gz)
and observables, and all label receipts are retained.
"""
from pathlib import Path
import json
from . import archive_completed as core

COLLECTION = Path('/home/ubuntu/coopmwm20261102_lscollect1')


def final_snapshot_paths(collection=COLLECTION):
    collection = Path(collection)
    allowed = set()
    for label in (collection/'labels').glob('*.json'):
        assert not any(x in str(label).lower() for x in core.BLOCKED)
        assert not label.is_symlink()
        value = json.loads(label.read_text())
        if not 986000 <= value.get('world_id', 0) <= 986071: continue
        v = value.get('verification', {})
        if v.get('status') not in ('accepted', 'rejected'): continue
        key = value.get('search_id', '')
        if not key or not label.stem.startswith(key+'_'): continue
        generation = label.stem[len(key)+1:]
        if generation not in ('SUMOENV1', 'SUMOPATH1'): continue
        vid = v.get('verification_id', '')
        if not vid or Path(vid).name != vid or '/' in vid or '\\' in vid: continue
        p = collection/'label_attempts'/generation/'branches'/vid/'SNAPSHOT.pt'
        assert not any(x in str(p).lower() for x in core.BLOCKED)
        allowed.add(p)
        for branch in v.get('branches', []):
            name = branch.get('branch', '')
            if not name or '/' in name or '\\' in name or Path(name).name != name: continue
            copy = p.parent/(name+'.result.json')
            assert not any(x in str(copy).lower() for x in core.BLOCKED)
            allowed.add(copy)
    return allowed


def checked_snapshot(path):
    p = Path(path)
    assert p in final_snapshot_paths(), 'not a successful final label runner copy'
    assert (p.name == 'SNAPSHOT.pt' or p.name.endswith('.result.json'))
    assert not p.is_symlink() and p.resolve() == p
    return p


def configure():
    core.CONTROL = Path('/home/ubuntu/coopmwm_archive_20261103_label_runner_copies')
    core.DEST = '/root/cpuq_archive_20261103/ls_label_runner_copies'
    core.RECEIPT_NAME = 'ARCHIVE_MOVE_ls_label_runner_copies.json'
    core.checked_tape = checked_snapshot
    core.source_paths = lambda: sorted(p for p in final_snapshot_paths() if p.is_file())


if __name__ == '__main__':
    configure()
    core.run()
