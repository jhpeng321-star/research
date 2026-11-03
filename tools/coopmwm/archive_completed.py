"""MAIN14:50 SHA-gated move of completed interim observable tapes only.

Runs on cpuQ. Never opens outcome payloads or SUMO configuration files. All
RESULTs, receipts, checkpoints and live owner files remain in their original place.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

HOME = Path('/home/ubuntu')
WARM = HOME/'coopmwm20261102_warm1'
CONTROL = HOME/'coopmwm_archive_20261103_1450'
DEST = '/root/cpuq_archive_20261103/interim008_observables'
HOST = 'root@connect.westb.seetacloud.com'
HOST_KEY = '[connect.westb.seetacloud.com]:53897 ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIK3kwiFvUVgx1wVKa/LqYm6JNHZqht3d+OSpVJVG/AWa\n'
BLOCKED = ('sealed', 'holdout', 'heldout', 'cal_split', 'chunk5', 'j4b', 'inc1',
           'hetdata_truth', 'vehicle_params', 'driver_params', 'vtype')


def checked_tape(path):
    p = Path(path)
    assert not any(x in str(p).lower() for x in BLOCKED), str(p)
    assert p.name == 'observable.jsonl.gz' and p.parent.parent == WARM/'jobs'
    assert p.parent.name.startswith('interim008_')
    assert not p.is_symlink() and p.resolve() == p
    assert (p.parent/'RESULT.json').is_file(), 'incomplete episode'
    return p


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def immutable(path, value):
    data = json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+'\n'
    p = Path(path)
    if p.exists():
        assert p.read_text() == data, str(p)
    else:
        with p.open('x') as f:
            f.write(data); f.flush(); os.fsync(f.fileno())


def open_references(paths):
    wanted = set(map(str, paths)); hits = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit(): continue
        try:
            for fd in (proc/'fd').iterdir():
                try:
                    target = os.readlink(fd)
                    if target in wanted: hits.append(dict(pid=int(proc.name), path=target))
                except FileNotFoundError: pass
        except FileNotFoundError: pass
        # An unreadable process cannot be certified idle. Fail closed.
    return hits


def binding_references(paths):
    needles = [str(p).encode() for p in paths]; hits = []
    # Read only public top-level binding/config/manifest metadata. Never traverse
    # private evaluation directories or inspect any RESULT/outcome payload.
    for root in HOME.glob('coopmwm*'):
        if not root.is_dir() or root.is_symlink(): continue
        for p in root.glob('*.json'):
            if any(x in str(p).lower() for x in BLOCKED): continue
            if not any(x in p.name.upper() for x in ('BINDING', 'CONFIG', 'MANIFEST')): continue
            if p.is_symlink(): continue
            data = p.read_bytes()
            for needle in needles:
                if needle in data: hits.append(dict(metadata=str(p), path=needle.decode()))
    return hits


def ssh():
    return ['ssh', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
            '-o', 'UserKnownHostsFile='+str(CONTROL/'known_hosts'),
            '-o', 'ConnectTimeout=20', '-o', 'ServerAliveInterval=15',
            '-o', 'ServerAliveCountMax=8', '-i', '/home/ubuntu/.ssh/id_ed25519_boxsync',
            '-p', '53897']


def remove_verified(manifest, verified):
    assert verified['verified_files'] == len(manifest)
    assert verified['verified_bytes'] == sum(x['bytes'] for x in manifest)
    paths = [HOME/x['path'] for x in manifest if (HOME/x['path']).exists()]
    assert not open_references(paths) and not binding_references(paths)
    # Verify the complete remaining source set before removing its first file.
    for x in manifest:
        p = HOME/x['path']
        if not p.exists(): continue  # already removed after durable remote certificate
        checked_tape(p)
        assert p.stat().st_size == x['bytes'] and sha(p) == x['sha256']
    for p in paths: checked_tape(p).unlink()


def run():
    import fcntl
    assert CONTROL.is_dir() and WARM.is_dir()
    lock = (CONTROL/'RUN.claim').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    receipt = CONTROL/'ARCHIVE_MOVE_interim008_observables.json'
    if receipt.exists(): print(receipt.read_text(), flush=True); return
    os.nice(19)
    saved = CONTROL/'MANIFEST.json'
    manifest = json.loads(saved.read_text()) if saved.exists() else None
    paths = ([HOME/x['path'] for x in manifest if (HOME/x['path']).exists()]
             if manifest is not None else sorted((WARM/'jobs').glob('interim008_*/observable.jsonl.gz')))
    assert paths or (CONTROL/'REMOTE_VERIFIED.json').exists(), 'no eligible source traces'
    for p in paths: checked_tape(p)
    refs = open_references(paths) + binding_references(paths)
    assert not refs, refs
    before = shutil.disk_usage(HOME).free
    if manifest is None:
        manifest = [dict(path=str(p.relative_to(HOME)), bytes=p.stat().st_size,
                         sha256=sha(p)) for p in paths]
    immutable(CONTROL/'MANIFEST.json', manifest)
    mh = sha(CONTROL/'MANIFEST.json')
    immutable(CONTROL/'AUDIT.json', dict(authority='MAIN2026-11-03 14:50',
        manifest_sha256=mh, open_references=[], binding_references=[],
        source_files=len(manifest), source_bytes=sum(x['bytes'] for x in manifest),
        result_files_retained=True, sim_s=0))
    if (CONTROL/'REMOTE_VERIFIED.json').exists():
        verified = json.loads((CONTROL/'REMOTE_VERIFIED.json').read_text())
        assert verified['manifest_sha256'] == mh
        remove_verified(manifest, verified)
        immutable(receipt, dict(authority='MAIN2026-11-03 14:50', unix=time.time(),
            destination_host='gpuR', destination=DEST, **verified,
            source_removed=True, results_retained=True, disk_free_before=before,
            disk_free_after=shutil.disk_usage(HOME).free, sim_s=0))
        print(receipt.read_text(), flush=True); return
    key = CONTROL/'known_hosts'
    if key.exists(): assert key.read_text() == HOST_KEY
    else: key.write_text(HOST_KEY)
    setup = "from pathlib import Path; import shutil; p=Path("+repr(DEST)+"); assert p.parent.parent.is_dir(); p.parent.mkdir(exist_ok=True); p.mkdir(exist_ok=True); assert shutil.disk_usage(p).free>"+str(sum(x['bytes'] for x in manifest)+1024**3)
    subprocess.run(ssh()+[HOST, 'python -c '+__import__('shlex').quote(setup)], check=True, timeout=60)
    filelist = CONTROL/'FILES.null'
    data = b''.join(x['path'].encode()+b'\0' for x in manifest)
    if filelist.exists(): assert filelist.read_bytes() == data
    else: filelist.write_bytes(data)
    subprocess.run(['rsync', '-a', '--from0', '--files-from='+str(filelist),
                    '-e', __import__('shlex').join(ssh()), str(HOME)+'/',
                    HOST+':'+DEST+'/'], check=True, timeout=1800)
    code = '''import sys,json,hashlib
from pathlib import Path
m=json.load(sys.stdin);root=Path(DEST)
for x in m:
 p=root/x['path'];assert p.is_file() and not p.is_symlink()
 assert p.stat().st_size==x['bytes'],str(p)
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 assert h.hexdigest()==x['sha256'],str(p)
v=dict(verified_files=len(m),verified_bytes=sum(x['bytes'] for x in m),manifest_sha256=MH)
out=root/'REMOTE_VERIFIED.json'
if out.exists():assert json.loads(out.read_text())==v
else:
 with out.open('x') as f:json.dump(v,f,sort_keys=True)
print(json.dumps(v))
'''.replace('Path(DEST)', 'Path('+repr(DEST)+')').replace('MH)', repr(mh)+')')
    result = subprocess.run(ssh()+[HOST, 'python -c '+__import__('shlex').quote(code)],
        input=json.dumps(manifest), capture_output=True, text=True, check=True, timeout=180)
    verified = json.loads(result.stdout)
    assert verified['verified_files'] == len(manifest) and verified['manifest_sha256'] == mh
    immutable(CONTROL/'REMOTE_VERIFIED.json', verified)
    # Re-audit immediately before the only destructive operation, then rehash
    # every source. Partial cleanup is resumed only from this exact manifest.
    remove_verified(manifest, verified)
    immutable(receipt, dict(authority='MAIN2026-11-03 14:50', unix=time.time(),
        destination_host='gpuR', destination=DEST,
        **verified, source_removed=True, results_retained=True,
        disk_free_before=before, disk_free_after=shutil.disk_usage(HOME).free, sim_s=0))
    print(receipt.read_text(), flush=True)


if __name__ == '__main__': run()
