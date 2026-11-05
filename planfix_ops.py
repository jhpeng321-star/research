"""Resume-only operator transport; numerical collector source remains f631ce90.

CPU uses the accepted F6 shared admission pool. GPU uses a persistent WSL
OpenSSH master. Retries apply only to read-only/idempotent chunk operations;
unknown dispatch outcomes must be inspected, never replayed automatically.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
import zlib

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

HOME = Path('E:/agent-work/_sol')
TEMP = Path('E:/Temp/opencode')
STATUS = Path('E:/research-dense/runs/active/C4-V7FT-20261017/CLEAN1_STATUS.md')
F6 = '/mnt/e/agent-work/c4-fix-ssh'
ARCHIVE = TEMP / 'planfix_f631_gpu_src_361a974fd6b9274a.z'
REMOTE_ARCHIVE = '/root/' + ARCHIVE.name


def note(host, text):
    line = 'PLANFIX ' + ('GPU ' if host == 'gpuA' else '') + time.strftime('%Y-%m-%dT%H:%M:%S%z') + ' ' + text
    print(line, flush=True)
    with STATUS.open('a', encoding='utf-8') as out:
        out.write(line + '\n')


def remote(host, code, *, timeout=40, attempts=1, backoff=60, stdin_data=None):
    assert host in ('cpuV', 'gpuA') and 1 <= attempts <= 6
    # Pass one shell-quoted command to WSL, preserving Python source via stdin.
    interpreter = 'python3' if host == 'cpuV' else '/root/miniconda3/bin/python'
    packed=base64.b64encode(zlib.compress(code.encode(),9)).decode()
    executable_code=('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))' if stdin_data is None else code)
    target_command = interpreter + ' -B -c ' + shlex.quote(executable_code)
    if host == 'cpuV':
        command = ('python3 -B /mnt/e/agent-work/c4-variety-collect/planfix_cpuv_transport.py --timeout '
                   + str(timeout) + ' ' + shlex.quote(target_command))
    else:
        command = shlex.join(['timeout', str(timeout), 'ssh', '-F', '/mnt/c/Users/jpeng/.ssh/config',
            '-i', '/home/jinghan/.ssh/sol_cpuv_rented_box',
            '-o', 'UserKnownHostsFile=/mnt/c/Users/jpeng/.ssh/known_hosts',
            '-o', 'StrictHostKeyChecking=yes', '-o', 'ControlMaster=auto',
            '-o', 'ControlPersist=600', '-o', 'ControlPath=/tmp/planfix-gpua-%C',
            '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ConnectionAttempts=1',
            'gpuA', target_command])
    for attempt in range(1, attempts + 1):
        note(host, 'transport attempt ' + str(attempt) + '/' + str(attempts))
        try:
            result = subprocess.run(['wsl', '-d', 'Ubuntu-22.04', '--', 'bash', '-c', command],
                input=b'' if stdin_data is None else stdin_data,
                capture_output=True, timeout=timeout+65)
            output = result.stdout.decode(errors='replace').replace('\x00', '')
            error = result.stderr.decode(errors='replace').replace('\x00', '')
            print(output, error, flush=True)
            receipt = dict(host=host, unix=time.time(), attempt=attempt, returncode=result.returncode,
                           stdout=output, stderr=error, code_sha256=hashlib.sha256(code.encode()).hexdigest())
        except subprocess.TimeoutExpired:
            receipt = dict(host=host, unix=time.time(), attempt=attempt, returncode=124,
                           outcome='unknown; inspect durable receipts before any dispatch')
        assert TEMP.is_dir()
        with (TEMP / ('PLANFIX_TRANSPORT_' + host + '_' + str(time.time_ns()) + '.json')).open('x') as out:
            json.dump(receipt, out, indent=2)
        note(host, 'transport attempt outcome rc=' + str(receipt['returncode']))
        if receipt['returncode'] == 0:
            return receipt['stdout']
        if receipt['returncode'] not in (124, 255):
            raise RuntimeError(host + ' remote command failed; inspect before retry')
        if attempt < attempts:
            delay = min(300, backoff * 2**(attempt-1))
            note(host, 'backoff ' + str(delay) + 's')
            time.sleep(delay)
    raise RuntimeError(host + ' bounded attempts exhausted; remote outcome not assumed')


def chunks(host):
    """Idempotent content-addressed uploads; every chunk rehashed before assembly."""
    data = ARCHIVE.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    assert sha.startswith('361a974fd6b9274a')
    parent = '/root' if host == 'gpuA' else '/home/ubuntu/coopmwm20261105_clean1/plan_fix'
    root = parent + '/source_chunks_' + sha[:16]
    parts = [data[i:i+16384] for i in range(0, len(data), 16384)]
    manifest = [hashlib.sha256(part).hexdigest() for part in parts]
    note(host, 'resumable chunk stream bytes=' + str(len(data)) + ' chunks=' + str(len(parts)))
    target = REMOTE_ARCHIVE if host == 'gpuA' else parent + '/' + ARCHIVE.name
    reader = '''from pathlib import Path
import base64,hashlib,json,os,sys
parent=Path(__PARENT__); assert parent.is_dir()
root=Path(__ROOT__);root.mkdir(exist_ok=True)
expected=__MANIFEST__
for line in sys.stdin.buffer:
 row=json.loads(line);i=row['index'];sha=expected[i]
 data=base64.b64decode(row['data']);assert hashlib.sha256(data).hexdigest()==sha
 p=root/('%04d_%s.part'%(i,sha))
 if p.exists():assert hashlib.sha256(p.read_bytes()).hexdigest()==sha
 else:
  temporary=p.with_suffix('.partial')
  with temporary.open('wb') as out:out.write(data);out.flush();os.fsync(out.fileno())
  assert hashlib.sha256(temporary.read_bytes()).hexdigest()==sha
  os.rename(temporary,p)
 print('CHUNK_SHA_OK',i,sha,flush=True)
chunks=[]
for i,sha in enumerate(expected):
 data=(root/('%04d_%s.part'%(i,sha))).read_bytes()
 assert hashlib.sha256(data).hexdigest()==sha
 chunks.append(data)
data=b''.join(chunks);assert hashlib.sha256(data).hexdigest()==__SHA__
p=Path(__TARGET__);assert p.parent.is_dir()
if p.exists():assert hashlib.sha256(p.read_bytes()).hexdigest()==__SHA__
else:
 with p.open('xb') as out:out.write(data);out.flush();os.fsync(out.fileno())
receipt=dict(archive=str(p),sha256=__SHA__,chunks=len(chunks),all_chunks_verified=True,bytes=len(data))
q=root/'VERIFIED.json'
if q.exists():assert json.loads(q.read_text())==receipt
else:
 with q.open('x') as out:json.dump(receipt,out,sort_keys=True)
print(json.dumps(receipt))
'''.replace('__PARENT__', repr(parent)).replace('__ROOT__', repr(root)).replace('__MANIFEST__', repr(manifest)).replace(
        '__SHA__', repr(sha)).replace('__TARGET__', repr(target))
    payload = b''.join((json.dumps(dict(index=i,data=base64.b64encode(part).decode()))+'\n').encode()
                       for i,part in enumerate(parts))
    remote(host, reader, attempts=1, timeout=100, stdin_data=payload)


def rsync_gpu():
    """One bounded append-verify attempt; callers back off between attempts."""
    shell = shlex.join(['ssh','-F','/mnt/c/Users/jpeng/.ssh/config',
        '-i','/home/jinghan/.ssh/sol_cpuv_rented_box',
        '-o','UserKnownHostsFile=/mnt/c/Users/jpeng/.ssh/known_hosts',
        '-o','StrictHostKeyChecking=yes','-o','ControlMaster=auto',
        '-o','ControlPersist=600','-o','ControlPath=/tmp/planfix-gpua-%C',
        '-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ConnectionAttempts=1'])
    command = shlex.join(['timeout','35','rsync','--partial','--append-verify','--timeout=25',
        '--info=progress2','-e',shell,'/mnt/e/Temp/opencode/'+ARCHIVE.name,
        'gpuA:'+REMOTE_ARCHIVE])
    note('gpuA','rsync --partial --append-verify bounded35s attempt')
    result = subprocess.run(['wsl','-d','Ubuntu-22.04','--','bash','-c',command],capture_output=True,timeout=45)
    receipt = dict(returncode=result.returncode,stdout=result.stdout.decode(errors='replace'),
                   stderr=result.stderr.decode(errors='replace'),unix=time.time())
    with (TEMP/('PLANFIX_GPU_RSYNC_'+str(time.time_ns())+'.json')).open('x') as out:json.dump(receipt,out)
    print(receipt,flush=True)
    note('gpuA','rsync outcome rc='+str(result.returncode))
    if result.returncode:
        raise RuntimeError('resumable rsync incomplete; retained partial, back off')
    data=ARCHIVE.read_bytes();sha=hashlib.sha256(data).hexdigest()
    hashes=[hashlib.sha256(data[i:i+16384]).hexdigest() for i in range(0,len(data),16384)]
    code='''from pathlib import Path
import hashlib,json,os
p=Path(__PATH__);data=p.read_bytes();hashes=__HASHES__
assert len(data)==__BYTES__ and hashlib.sha256(data).hexdigest()==__SHA__
for i,sha in enumerate(hashes):assert hashlib.sha256(data[i*16384:(i+1)*16384]).hexdigest()==sha
target=Path(__PATH__)
if target.exists():assert hashlib.sha256(target.read_bytes()).hexdigest()==__SHA__
else:
 with target.open('xb') as out:out.write(data);out.flush();os.fsync(out.fileno())
print(json.dumps(dict(all_chunks_verified=True,chunks=len(hashes),sha256=__SHA__,bytes=len(data),archive=str(target))))
'''.replace('__PATH__',repr(REMOTE_ARCHIVE)).replace('__HASHES__',repr(hashes)).replace(
        '__SHA__',repr(sha)).replace('__BYTES__',str(len(data)))
    remote('gpuA',code,timeout=25)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('probe', 'chunks', 'rsync', 'file'))
    parser.add_argument('host', choices=('cpuV', 'gpuA'))
    parser.add_argument('--file')
    parser.add_argument('--attempts', type=int, default=1)
    parser.add_argument('--timeout', type=int, default=40)
    args = parser.parse_args()
    if args.action == 'rsync':
        assert args.host=='gpuA'
        rsync_gpu()
    elif args.action == 'chunks':
        chunks(args.host)
    else:
        filename = args.file or str(HOME / ('planfix_quick_remote.py' if args.host == 'cpuV' else 'planfix_gpua_status_remote.py'))
        remote(args.host, Path(filename).read_text(), attempts=args.attempts, timeout=args.timeout)
