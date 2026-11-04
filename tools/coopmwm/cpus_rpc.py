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


def _checking_modules(cfg):
    """Checking-source modules (verify_perseed/slim_tapes/... of c4/checking-cpus) take precedence over
    the src7 package; the reviewed slim_remote_archive adapter comes from slim_migration1/source4."""
    import tools.coopmwm
    for d in reversed(cfg['checking_module_dirs']):
        if d not in tools.coopmwm.__path__:
            tools.coopmwm.__path__.insert(0, d)


def _file_sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def op_view(control, cfg, payload):
    """cpuQ reader view of a STAGED (not yet published) record against the cpuQ bank."""
    _checking_modules(cfg)
    from tools.coopmwm.cpus_checking import reader_view
    arm, vid = payload['arm'], payload['verification_id']
    if arm not in ('B', 'R') or not re.fullmatch(r'[A-Za-z0-9_]+', vid):
        return dict(ok=False, error='bad arm/vid')
    staged = Path(control)/'staged_records'/f'{arm}_{vid}.json'
    if _file_sha(staged) != payload['record_sha256']:
        return dict(ok=False, error='staged record sha differs')
    record = json.loads(staged.read_text())
    view = reader_view(Path(cfg['run_root'])/'replay_bank'/arm, record)
    return dict(ok=True, view=json.loads(json.dumps(view, default=str)))


def _open_paths():
    out = set()
    for fd_dir in Path('/proc').glob('[0-9]*/fd'):
        try:
            for fd in fd_dir.iterdir():
                try:
                    out.add(os.readlink(fd))
                except OSError:
                    pass
        except OSError:
            pass
    return out


def _cpus_proof(cfg, relatives, slim_rels):
    import subprocess
    code = ('import hashlib,json,shutil,sys\nfrom pathlib import Path\nA=Path(%r);B=Path(%r)\nq=json.loads(sys.stdin.read())\n'
            'h=lambda p: hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() and not p.is_symlink() else None\n'
            'print(json.dumps(dict(hashes={r:h(A/r) for r in q["files"]}, slim={r:h(B/r) for r in q["slim"]},'
            'free=shutil.disk_usage(str(A)).free)))\n') % (cfg['cpus_archive'], cfg['cpus_bank_root'])
    ssh = list(cfg['cpus_ssh'])
    subprocess.run(ssh+['cat > /tmp/cpus_first_seed_proof.py'], input=code, text=True, timeout=120, check=True)
    p = subprocess.run(ssh+['python3 /tmp/cpus_first_seed_proof.py'], input=json.dumps(dict(files=relatives, slim=slim_rels)),
                       capture_output=True, text=True, timeout=600)
    if p.returncode:
        raise RuntimeError('cpuS proof transport %s: %s' % (p.returncode, p.stderr[-800:]))
    return json.loads(p.stdout)


def op_slim_first_seed(control, cfg, payload):
    """MAIN 13:05 item 3: after a decision landed, slim its FIRST-SEED rejected candidate tapes on cpuQ with
    the reviewed slim_tapes (slim_plan/build_slim/slim_branch) and the reviewed slim_remote_archive.commit_branch
    boundary. Remote proof = sha256 of the byte-identical copies already in the cpuS archive (moved there by the
    cpuS seal step), re-hashed now; a missing/different copy keeps the cpuQ source untouched."""
    _checking_modules(cfg)
    from tools.coopmwm import slim_tapes as slim, slim_remote_archive as sra
    from tools.coopmwm.accel_storage import read
    from tools.coopmwm.exact_grade import _write_new
    from tools.coopmwm.protocol import digest
    arm, vid = payload['arm'], payload['verification_id']
    bank = Path(cfg['run_root'])/'replay_bank'/arm
    record = slim.records(bank)[vid]
    first = bank/'sumo_verification'/'branches'/vid
    plans, rows = [], []
    for entry in slim.slim_plan(bank, record):
        bdir, branch, _ = entry
        bdir = Path(bdir)
        if bdir.parent != first:
            continue
        if (bdir/slim.SLIM).exists() or not (bdir/slim.MARKER).exists():
            rows.append(dict(dir=str(bdir), status='skip_already_slim_or_no_tape'));continue
        payload_p = bdir/slim.PAYLOAD
        tape = read(bdir/slim.MARKER)
        body = slim.build_slim(branch, tape, _file_sha(payload_p) if payload_p.exists() else None)
        rel_dir = bdir.relative_to(bank)
        files = [dict(source=str(bdir/n), relative=str(rel_dir/n), **sra.file_identity(bdir/n))
                 for n in (slim.PAYLOAD, slim.MARKER) if (bdir/n).exists()]
        plans.append((entry, bdir, body, files, rel_dir))
    if not plans:
        return dict(ok=True, verification_id=vid, slimmed=0, bytes_freed=0, rows=rows)
    proof_all = _cpus_proof(cfg, [f['relative'] for _, _, _, fs, _ in plans for f in fs],
                            [str(Path('replay_bank')/arm/rd/slim.SLIM) for *_, rd in plans])
    freed = 0;moved = 0;slim_equal = 0
    original = slim._move_verified
    try:
        for entry, bdir, body, files, rel_dir in plans:
            proof = dict(free=proof_all['free'], hashes={f['relative']: proof_all['hashes'].get(f['relative']) for f in files})
            prepared = dict(entries=files, body_sha256=digest(body), archive=cfg['cpus_archive']+'/'+str(rel_dir))

            def revalidate(entry=entry):
                cur = slim.records(bank)[vid]
                if cur['confirmation_sha256'] != record['confirmation_sha256'] or entry not in slim.slim_plan(bank, cur):
                    raise ValueError('sealed rejected-candidate eligibility changed')

            def move(b, dest, bd, prepared=prepared, proof=proof, revalidate=revalidate):
                sra.commit_branch(slim, b, bd, prepared, proof, opened=_open_paths, revalidate=revalidate,
                                  write=_write_new)
            slim._move_verified = move
            try:
                row = slim.slim_branch(entry, bank=bank, archive_dir=Path(cfg['cpus_archive']))
            except Exception as error:      # proof missing/different or source changed: source kept
                rows.append(dict(dir=str(bdir), status='kept', error=repr(error)))
                if isinstance(error, slim.SlimPartial):
                    break
                continue
            cpus_slim = proof_all['slim'].get(str(Path('replay_bank')/arm/rel_dir/slim.SLIM))
            same = cpus_slim is not None and cpus_slim == _file_sha(bdir/slim.SLIM)
            slim_equal += bool(same)
            rows.append(dict(row, slim_json_equals_cpus=same))
            moved += row['status'] == 'moved'
            freed += row['bytes_moved']-row['slim_bytes']
    finally:
        slim._move_verified = original
    out = dict(ok=True, verification_id=vid, arm=arm, planned=len(plans), slimmed=moved, bytes_freed=freed,
               slim_json_equal=slim_equal, rows=rows, unix=time.time())
    d = Path(control)/'first_seed_slim';d.mkdir(exist_ok=True)
    (d/f'{arm}_{vid}.json').write_text(json.dumps(out, default=str))
    return out


OPS = dict(reserve=op_reserve, ledger=op_ledger, health=op_health, budget=op_budget, disk=op_disk,
           absent=op_absent, verify=op_verify, note=op_note, view=op_view, slim_first_seed=op_slim_first_seed)


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
