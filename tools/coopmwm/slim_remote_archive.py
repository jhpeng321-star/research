"""MAIN10:12 transport-only adapter for the reviewed slim_tapes selection.

No payload staging on cpuQ. Batches copy directly to gpuA over the registered
rolling mover route; durable remote byte/payload proofs precede every source
unlink. Admitted/reference/solo selection remains entirely in slim_tapes.
"""
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time

HOME = Path('/home/ubuntu')
FLOOR = 8 * 1024**3
BATCH_BYTES = 64 * 1024**2


def file_identity(path):
    p = Path(path)
    if p.is_symlink() or p.resolve() != p or not p.is_file():
        raise ValueError('nonregular/symlink source')
    st = p.stat()
    return dict(bytes=st.st_size, inode=st.st_ino, mtime_ns=st.st_mtime_ns,
                sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def validate_proof(entries, proof):
    if set(proof['hashes']) != {e['relative'] for e in entries}:
        raise ValueError('incomplete remote proof')
    for e in entries:
        if proof['hashes'][e['relative']] != e['sha256']:
            raise ValueError('remote SHA differs')
    if proof['free'] < FLOOR:
        raise ValueError('gpuA archive floor breached; retain sources')


def commit_branch(slim, bdir, body, prepared, proof, *, opened, revalidate, write):
    """The only destructive boundary; never removes any remote archive copy."""
    from .protocol import digest
    validate_proof(prepared['entries'], proof)
    if digest(body) != prepared['body_sha256']:
        raise ValueError('SLIM body changed after planning')
    revalidate()
    active = opened()
    for e in prepared['entries']:
        if e['source'] in active or file_identity(e['source']) != {k:e[k] for k in ('bytes','inode','mtime_ns','sha256')}:
            raise ValueError('source open or changed after remote proof')
    # Preserve the reviewed order and partial-move rule. Proof is already durable.
    unlinked = []
    try:
        write(Path(bdir)/slim.SLIM, body)
        for e in prepared['entries']:
            Path(e['source']).unlink()
            unlinked.append(Path(e['source']).name)
    except Exception as error:
        if unlinked:
            from .exact_grade import _seal
            partial = _seal(dict(schema=slim.SCHEMA, state='partial_move',
                dir=str(bdir), archive=prepared['archive'], unlinked=unlinked,
                kept_source=[Path(e['source']).name for e in prepared['entries'] if Path(e['source']).name not in unlinked],
                error=repr(error), unix=time.time()), 'partial_sha256')
            write(Path(bdir)/slim.PARTIAL, partial)
            raise slim.SlimPartial('partial remote move; all verified archive copies retained') from error
        raise


def run(control, dry_run=False):
    import fcntl
    import signal
    import sys
    from . import slim_tapes as slim
    from . import bank_statistics as bs
    from .accel_storage import read
    from .exact_grade import _write_new
    from .protocol import digest

    c = Path(control)
    authority = json.loads((c/('QUALIFICATION_BINDING.json' if dry_run else 'AUTHORITY.json')).read_text())
    for name, want in authority['source_hashes'].items():
        if hashlib.sha256((c/name).read_bytes()).hexdigest() != want:
            raise ValueError('migration immutable source SHA differs')
    lock = (c/('DRY.claim' if dry_run else 'RUN.claim')).open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not dry_run and (c/'OWNER.json').exists():
        raise ValueError('adopt existing migration owner')
    sys.path.insert(0, authority['archive_source'])
    import rolling_archive as mover
    for name, want in authority['archive_source_hashes'].items():
        if hashlib.sha256((Path(authority['archive_source'])/name).read_bytes()).hexdigest() != want:
            raise ValueError('qualified archive helper SHA differs')
    stop = HOME/'coopmwm20261103_collect2323/STOP.json'
    if not stop.is_file():
        raise ValueError('collector pause receipt required')
    root = Path(authority['run_root'])
    destination = Path(authority['destination'])
    if destination.parent != Path('/root/cpuq_archive_20261103') or mover.base.BAD.search(str(destination)):
        raise ValueError('unregistered remote namespace')

    def put(path, value):
        _write_new(path, value)

    def remote(code):
        transport = json.loads((Path(authority['transport_control'])/'TRANSPORT.json').read_text())
        if not Path(transport['socket']).exists():
            raise RuntimeError('forwarded agent unavailable; retain sources')
        env = dict(os.environ, SSH_AUTH_SOCK=transport['socket'])
        command = list(authority['ssh_command'])
        p = subprocess.run(command+['/root/miniconda3/bin/python -'], input=code,
                           text=True, capture_output=True, env=env, timeout=90)
        if p.returncode:
            raise RuntimeError('remote transport '+str(p.returncode)+': '+p.stderr[-1500:])
        return json.loads(p.stdout)

    def retry(call):
        for attempt in range(6):
            try:
                return call()
            except (RuntimeError, subprocess.TimeoutExpired) as error:
                network = isinstance(error, subprocess.TimeoutExpired) or any(x in str(error) for x in
                    ('Connection closed','Connection reset','timed out','Temporary failure in name resolution','transport 255','forwarded agent unavailable'))
                if not network or attempt == 5:
                    raise
                time.sleep((5,10,20,40,80)[attempt])

    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    os.nice(max(0, 19-os.getpriority(os.PRIO_PROCESS,0)))
    if not dry_run:
        put(c/'OWNER.json', mover.base.owner())

    plans = []
    errors = []
    seen = set()
    opened = mover.base.open_files()
    for arm in 'BR':
        bank = root/'replay_bank'/arm
        for vid, record in sorted(slim.records(bank).items()):
            try:
                entries = slim.slim_plan(bank, record)
            except Exception as error:
                errors.append(dict(arm=arm, verification_id=vid, error=repr(error)))
                continue  # reviewed rule: an unplannable record keeps all tapes
            for entry in entries:
                bdir, branch, _ = entry
                if str(bdir) in seen:
                    continue
                seen.add(str(bdir))
                if (bdir/slim.SLIM).exists() or not (bdir/slim.MARKER).exists():
                    continue
                try:
                    if mover.base.BAD.search(str(bdir)):
                        raise ValueError('restricted branch path')
                    payload = bdir/slim.PAYLOAD
                    tape = read(bdir/slim.MARKER)
                    body = slim.build_slim(branch, tape, hashlib.sha256(payload.read_bytes()).hexdigest() if payload.exists() else None)
                    files = []
                    for name in (slim.PAYLOAD, slim.MARKER):
                        p = bdir/name
                        if not p.exists():
                            continue
                        relative = str(p.relative_to(HOME))
                        if str(p) in opened:
                            raise ValueError('source open; keep tape')
                        files.append(dict(source=str(p), relative=relative, **file_identity(p)))
                    plans.append(dict(arm=arm, bank=str(bank), verification_id=vid,
                        confirmation_sha256=record['confirmation_sha256'], entry=entry,
                        prepared=dict(entries=files, body_sha256=digest(body), archive=str(destination/bdir.relative_to(HOME)))))
                except Exception as error:
                    errors.append(dict(arm=arm, verification_id=vid, branch=branch['branch'], error=repr(error)))
    plan = dict(unix=time.time(),branches=len(plans),files=sum(len(p['prepared']['entries']) for p in plans),
        bytes=sum(e['bytes'] for p in plans for e in p['prepared']['entries']),
        errors=errors, cpu_free=shutil.disk_usage(root).free, new_sim_s=0)
    put(c/('DRY_RUN.json' if dry_run else 'PLAN.json'), plan)
    if dry_run:
        return plan

    total = 0
    batches = 0
    while plans:
        chosen = []
        size = 0
        while plans and size < BATCH_BYTES:
            p = plans.pop(0)
            chosen.append(p)
            size += sum(e['bytes'] for e in p['prepared']['entries'])
        folder = c/f'batch_{batches:06}'
        folder.mkdir()
        files = [e for p in chosen for e in p['prepared']['entries']]
        manifest = dict(entries=files, destination=str(destination), new_sim_s=0)
        put(folder/'MANIFEST.json', manifest)
        free = retry(lambda: remote('import json,shutil; print(json.dumps(dict(free=shutil.disk_usage("/root").free)))'))['free']
        if free-size < FLOOR:
            put(c/'STOP_GPU_SPACE.json', dict(free=free, bytes=size, floor=FLOOR, new_sim_s=0))
            return
        setup = 'from pathlib import Path; import json; p=Path('+repr(str(destination))+'); assert p.parent.is_dir(); p.mkdir(exist_ok=True); print(json.dumps(dict(ready=True)))'
        retry(lambda: remote(setup))
        listing = folder/'FILES.null'
        listing.write_bytes(b''.join(e['relative'].encode()+b'\0' for e in files))

        def transfer():
            transport = json.loads((Path(authority['transport_control'])/'TRANSPORT.json').read_text())
            env = dict(os.environ, SSH_AUTH_SOCK=transport['socket'])
            command = authority['ssh_command']
            args = ['rsync','-az','--relative','--checksum','--no-owner','--no-group','--from0',
                '--files-from='+str(listing),'-e',shlex.join(command[:-1]),str(HOME)+'/',command[-1]+':'+str(destination)+'/']
            p = subprocess.run(args, env=env, capture_output=True, text=True, timeout=1200)
            if p.returncode:
                raise RuntimeError('rsync transport '+str(p.returncode)+': '+p.stderr[-1500:])
        retry(transfer)
        branch_proofs = [dict(path=str(Path(p['entry'][0]).relative_to(HOME)), result_sha256=p['entry'][1]['result_sha256']) for p in chosen]
        code = '''from pathlib import Path
import json,hashlib,gzip,shutil
root=Path(DEST);checks=CHECKS;out={}
for e in checks:
 p=root/e['relative'];assert p.is_file() and not p.is_symlink();assert p.stat().st_size==e['bytes']
 h=hashlib.sha256(p.read_bytes()).hexdigest();assert h==e['sha256'];out[e['relative']]=h
for b in BRANCHES:
 d=root/b['path'];marker=json.loads((d/'RESULT.json').read_text())
 if marker.get('encoding')=='coopmwm-lossless-gzip-v1':
  assert marker['file']=='RESULT.json.gz';data=gzip.decompress((d/'RESULT.json.gz').read_bytes());assert hashlib.sha256(data).hexdigest()==marker['sha256'];tape=json.loads(data)
 else:tape=marker
 canonical=json.dumps(tape,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
 assert hashlib.sha256(canonical).hexdigest()==b['result_sha256']
print(json.dumps(dict(hashes=out,free=shutil.disk_usage(root).free)))
'''.replace('DEST',repr(str(destination))).replace('CHECKS',repr(files)).replace('BRANCHES',repr(branch_proofs))
        proof = retry(lambda: remote(code))
        validate_proof(files, proof)
        put(folder/'REMOTE_VERIFIED.json', dict(manifest_sha256=digest(manifest), **proof))
        original = slim._move_verified
        try:
            for p in chosen:
                branch_prepared = p['prepared']
                subset = dict(proof, hashes={e['relative']:proof['hashes'][e['relative']] for e in branch_prepared['entries']})
                def revalidate():
                    current = slim.records(p['bank'])[p['verification_id']]
                    if current['confirmation_sha256'] != p['confirmation_sha256'] or p['entry'] not in slim.slim_plan(p['bank'],current):
                        raise ValueError('sealed rejected-candidate eligibility changed')
                def move(bdir, dest, body):
                    commit_branch(slim,bdir,body,branch_prepared,subset,opened=mover.base.open_files,revalidate=revalidate,write=put)
                slim._move_verified = move
                row = slim.slim_branch(p['entry'], bank=p['bank'], archive_dir=destination)
                if row['status'] != 'moved':
                    raise ValueError('unexpected branch state at commit')
                put(folder/(digest(str(p['entry'][0]))+'.json'), dict(row, archive=branch_prepared['archive'], new_sim_s=0))
                total += row['bytes_moved']-row['slim_bytes']
        finally:
            slim._move_verified = original
        batches += 1
        put(c/f'PROGRESS_{batches:06}.json', dict(unix=time.time(),batches=batches,net_bytes_recovered=total,cpu_free=shutil.disk_usage(root).free,new_sim_s=0))
    put(c/'COMPLETE.json', dict(unix=time.time(),batches=batches,net_bytes_recovered=total,cpu_free=shutil.disk_usage(root).free,planning_errors=errors,new_sim_s=0))


if __name__ == '__main__':
    import sys
    try:
        value = run(sys.argv[1], dry_run='--dry-run' in sys.argv[2:])
        if value is not None:
            print(json.dumps(value))
    except BaseException as error:
        import traceback
        c = Path(sys.argv[1])
        p = c/('DRY_FAILED.json' if '--dry-run' in sys.argv[2:] else 'FAILED.json')
        if not p.exists():
            with p.open('x') as f:
                json.dump(dict(error=repr(error),traceback=traceback.format_exc(),unix=time.time(),new_sim_s=0),f)
        raise
