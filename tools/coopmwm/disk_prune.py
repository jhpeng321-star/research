"""clean5b-F8P disk pruning of CONSUMED transient G4 data (MAIN 2026-11-07, after cpuZ filled its disk).

After every arm has sealed update u (LEARNING_MONITOR_u<u>.json exists, i.e. A/B/R updates incl. their paired G4
envelopes are complete) the update's transient G4 data is no longer read by the run:
  * jobs/<ident>_u<u>_*/g4_captures/*   (boundary SUMO captures, ~0.3-3 GB per B/R episode) -> sha256 recorded, deleted
  * g4/<ARM>_INPUT_<u>.json              (~10 GB per arm)                                   -> sha256 recorded, gzipped
The record pruned/PRUNE_u<u>.json (path, bytes, sha256 per file; written before any deletion) keeps the hashes.
Episode RESULT/observable/vehicle logs, checkpoints, envelopes and ledgers are never touched. Idempotent.
CLI (report-only process next to a run): python -m tools.coopmwm.disk_prune <run_root> [--loop]
"""
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def consumed(root, u):
    return (Path(root)/f'LEARNING_MONITOR_u{int(u):03}.json').exists()


def targets(root, u):
    root = Path(root)
    caps = sorted(p for d in (root/'jobs').glob(f'*_u{int(u):03}_*') if (d/'g4_captures').is_dir()
                  for p in (d/'g4_captures').rglob('*') if p.is_file())
    inputs = sorted((root/'g4').glob(f'*_INPUT_{int(u):03}.json'))
    return caps, inputs


def prune_update(root, u):
    """Hash, record, then delete captures / gzip inputs of a CONSUMED update. Returns the record (or None)."""
    root = Path(root)
    if not consumed(root, u):
        raise ValueError(f'update {u} not consumed by every arm yet (no LEARNING_MONITOR_u{int(u):03}.json)')
    record_path = root/'pruned'/f'PRUNE_u{int(u):03}.json'
    caps, inputs = targets(root, u)
    if record_path.exists() and not caps and not inputs:
        return None
    record = dict(update=int(u), unix=time.time(), rule='consumed transient G4 data (MAIN 2026-11-07)',
                  captures=[dict(path=str(p.relative_to(root)), bytes=p.stat().st_size, sha256=_sha(p)) for p in caps],
                  inputs=[dict(path=str(p.relative_to(root)), bytes=p.stat().st_size, sha256=_sha(p),
                               gzip=str(p.relative_to(root))+'.gz') for p in inputs])
    record_path.parent.mkdir(parents=True, exist_ok=True)
    name = record_path if not record_path.exists() else record_path.with_name(record_path.stem+f'.{int(time.time())}.json')
    with name.open('x', encoding='utf-8') as f:
        json.dump(record, f, indent=1); f.flush(); os.fsync(f.fileno())
    for p in inputs:
        gz = p.with_name(p.name+'.gz')
        with open(p, 'rb') as src, gzip.open(gz, 'xb', compresslevel=3) as dst:
            shutil.copyfileobj(src, dst, 1 << 20)
        p.unlink()
    for p in caps:
        p.unlink()
    return record


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    root, loop = Path(argv[0]), '--loop' in argv
    done = set()
    while True:
        for u in range(1, 25):
            if u not in done and consumed(root, u):
                rec = prune_update(root, u)
                done.add(u)
                print(json.dumps(dict(update=u, captures=len(rec['captures']) if rec else 0,
                                      inputs=len(rec['inputs']) if rec else 0,
                                      freed_bytes=sum(x['bytes'] for x in (rec['captures'] if rec else [])))), flush=True)
        if not loop or (root/'MONITOR_COMPLETE.json').exists() or (root/'STOP.json').exists():
            return
        time.sleep(120)


if __name__ == '__main__':
    main()
