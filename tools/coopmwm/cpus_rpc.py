"""cpuS plan-checking RPC server - runs on cpuQ, one short process per call (ssh from cpuS).

Single authoritative ledger (MAIN 2026-11-04 cpuS move, requirement 1): every charge of a cpuS
verification is written HERE, on cpuQ, by the UNCHANGED tools.coopmwm.accel_storage.reserve
(ACCOUNTING.claim lock + registered phase caps installed by warm_protocol.install_accounting from
the run root's binding), after the delegate-cap check taken under the delegate lock(s) (own lock
plus the BACKFILL0057 lock, so both backfill generations serialise on the same 1.5M transfer).
Every other op is read-only on the run root:
  reserve  {jobs, phase, host}      -> ledger rows of exactly these ids (after the write)
  ledger   {}                       -> all ledger rows (shared lock)
  health   {arm}                    -> warm_health.assert_running(root, arm) verdict
  budget   {}                       -> delegate/phase/total usage and caps
  disk     {}                       -> free bytes of /home/ubuntu
  absent   {paths}                  -> the listed paths that already EXIST (must be empty to push)
  verify   {files:[{path,sha256,size}]} -> mismatching / missing files
  note     {name, row}              -> append one JSON row to <control>/<name>.jsonl (receipts)
argv: <control> <op>; JSON payload on stdin; one JSON line on stdout; never raises over the wire.
Only paths under <run_root>/replay_bank are accepted by absent/verify; protected names refused."""
import fcntl
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

ALL8 = re.compile(r'^confirm_(seed\d+_all8|extended_all8|extended_seed\d+_all8|solo_all8_s\d+)_')
BAD = re.compile(r'sealed|holdout|heldout|cal_split|chunk5|j4b|inc1|hetdata_truth', re.I)


def _cfg(control):
    return json.loads((Path(control)/'RPC_BINDING.json').read_text())


def _bank_path(cfg, p):
    p = Path(p)
    base = Path(cfg['run_root'])/'replay_bank'
    if BAD.search(str(p)) or '..' in p.parts or base not in p.parents:
        raise ValueError(f'path outside the replay bank or protected: {p}')
    return p


def all8_used(rows):
    return sum(int(r['sim_s']) for r in rows if ALL8.match(str(r['id'])))


def op_reserve(control, cfg, payload):
    from tools.coopmwm import accel_storage
    from tools.coopmwm.storage import ledger
    from tools.coopmwm.warm_protocol import install_accounting
    root = Path(cfg['run_root']);phase = payload['phase'];jobs = payload['jobs']
    if phase != cfg['phase']:
        return dict(ok=False, error=f'phase {phase} is not the registered {cfg["phase"]}')
    if not all(ALL8.match(str(j['id'])) for j in jobs):
        return dict(ok=False, error='only all8-kind confirmation ids are charged through this RPC')
    need = sum(int(j['sim_s']) for j in jobs)
    locks = []
    try:
        for name in cfg['delegate_locks']:
            f = open(name, 'a');fcntl.flock(f, fcntl.LOCK_EX);locks.append(f)
        used = all8_used(ledger(root))
        if used+need > int(cfg['delegate_cap_sim_s']):
            return dict(ok=False, error=f'delegate cap {cfg["delegate_cap_sim_s"]} sim-s (1.5M transfer): '
                                        f'used {used} + {need}')
        install_accounting(root)
        try:
            accel_storage.reserve(root, jobs, phase)
        except ValueError as error:
            return dict(ok=False, error=str(error))
        ids = {j['id'] for j in jobs}
        rows = [r for r in ledger(root) if r['id'] in ids]
        with (Path(control)/'CHARGES_RPC.jsonl').open('a') as f:
            f.write(json.dumps(dict(unix=time.time(), host=payload.get('host'), phase=phase, n=len(jobs),
                                    sim_s=need, all8_used_before=used, ids=sorted(ids)))+'\n')
        return dict(ok=True, rows=rows, all8_used_after=used+need)
    finally:
        for f in locks:
            f.close()


def op_ledger(control, cfg, payload):
    from tools.coopmwm.storage import ledger
    return dict(ok=True, rows=ledger(Path(cfg['run_root'])))


def op_health(control, cfg, payload):
    from tools.coopmwm.warm_health import assert_running
    try:
        assert_running(Path(cfg['run_root']), payload['arm'])
    except Exception as error:      # verdict travels back; the client re-raises the same class name
        return dict(ok=False, type=type(error).__name__, error=str(error))
    return dict(ok=True)


def op_budget(control, cfg, payload):
    from tools.coopmwm import accel_storage
    from tools.coopmwm.storage import ledger
    from tools.coopmwm.warm_protocol import install_accounting
    root = Path(cfg['run_root']);rows = ledger(root)
    install_accounting(root)
    g = accel_storage._reserve.__globals__
    phase = cfg['phase']
    return dict(ok=True, all8_used=all8_used(rows), delegate_cap=int(cfg['delegate_cap_sim_s']),
                phase=phase, phase_used=sum(int(r['sim_s']) for r in rows if r['phase'] == phase),
                phase_cap=int(g['CAPS'][phase]), total_used=sum(int(r['sim_s']) for r in rows),
                total_cap=int(g['CAP']), rows=len(rows))


def op_disk(control, cfg, payload):
    return dict(ok=True, free=shutil.disk_usage('/home/ubuntu').free)


def op_absent(control, cfg, payload):
    present = [str(p) for p in payload['paths'] if os.path.lexists(_bank_path(cfg, p))]
    return dict(ok=True, present=present)


def op_verify(control, cfg, payload):
    bad = []
    for f in payload['files']:
        p = _bank_path(cfg, f['path'])
        if not p.is_file() or p.is_symlink():
            bad.append(dict(path=str(p), problem='missing'));continue
        if p.stat().st_size != int(f['size']):
            bad.append(dict(path=str(p), problem='size'));continue
        h = hashlib.sha256()
        with p.open('rb') as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b''):
                h.update(chunk)
        if h.hexdigest() != f['sha256']:
            bad.append(dict(path=str(p), problem='sha256'))
    return dict(ok=True, bad=bad, n=len(payload['files']))


def op_note(control, cfg, payload):
    name = payload['name']
    if not re.fullmatch(r'[A-Z0-9_]+', name):
        return dict(ok=False, error='bad note name')
    with (Path(control)/f'{name}.jsonl').open('a') as f:
        f.write(json.dumps(dict(payload['row'], unix_q=time.time()))+'\n')
    return dict(ok=True)


OPS = dict(reserve=op_reserve, ledger=op_ledger, health=op_health, budget=op_budget, disk=op_disk,
           absent=op_absent, verify=op_verify, note=op_note)


def main(argv=None):
    control, op = (argv or sys.argv[1:])[:2]
    try:
        payload = json.loads(sys.stdin.read() or '{}')
        out = OPS[op](control, _cfg(control), payload)
    except Exception as error:
        out = dict(ok=False, type=type(error).__name__, error=repr(error), transport_ok=True)
    sys.stdout.write(json.dumps(out)+'\n')


if __name__ == '__main__':
    main()
