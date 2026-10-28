"""Small allowlisted SSH operations; all task writes remain under ~/entrant."""
import base64
import io
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
BAD = ('sealed', 'holdout', 'heldout', 'cal_split', 'chunk5', 'j4b', 'inc1', 'hetdata_truth')


def ssh(host, source, data=None):
    assert host in ('cpuP', 'cpuQ', 'gpuR')
    b = base64.b64encode(source.encode()).decode()
    command = "python3 -c 'import base64;exec(base64.b64decode(\"" + b + "\"))'"
    return subprocess.run(['ssh', '-o', 'BatchMode=yes', host, command], input=data, check=True)


def detach(host, name, source):
    """Launch one named task-owned operation with immediate PID-incarnation receipt."""
    assert name.replace('_', '').isalnum()
    payload = base64.b64encode(source.encode()).decode()
    ssh(host, '''
import base64,json,subprocess,time
from pathlib import Path
p=Path.home()/'entrant';assert p.is_dir()
receipt=p/(NAME+'_LAUNCH.json')
if receipt.exists():
    old=json.loads(receipt.read_text());stat=Path('/proc')/str(old['pid'])/'stat'
    if stat.exists() and stat.read_text().split()[21]==old['start_ticks']:
        raise RuntimeError('named owned operation already running')
log=(p/(NAME+'.log')).open('ab')
child=subprocess.Popen([str(Path.home()/'miniconda3/bin/python'),'-u','-c',
    base64.b64decode(PAYLOAD).decode()],stdin=subprocess.DEVNULL,stdout=log,
    stderr=subprocess.STDOUT,start_new_session=True)
r={'pid':child.pid,'start_ticks':Path(f'/proc/{child.pid}/stat').read_text().split()[21],
   'started':time.time(),'operation':NAME}
receipt.write_text(json.dumps(r));print(json.dumps(r))
'''.replace('NAME', repr(name)).replace('PAYLOAD', repr(payload)))


def stage(host):
    files = ['src/wms/control/entrant.py', 'src/wms/control/dreamer.py',
             'src/wms/train/inputs.py', 'src/wms/train/settings.py',
             'src/wms/train/scoring_random.py', 'src/wms/config/data.py',
             'src/wms/eval/predictor.py', 'src/wms/eval/entries_input.py', 'src/wms/eval/run.py',
              'src/wms/config/legacy_flat.py', 'src/wms/data/collate.py',
              'src/wms/data/pack_history.py', 'src/wms/data/c4_episodes.py',
              'src/wms/data/dataset.py', 'src/wms/control/start_states.py',
              'src/wms/control/bridge.py', 'traffic_shepherd/c4/genrand.py',
             'tools/wmcombined/e3_score.py', 'tools/loop14_checks/accept.py',
             'tools/lrdiag/replay.py', 'tests/unit/test_entrant.py']
    files += [str(p.relative_to(ROOT)).replace('\\', '/') for p in (ROOT/'tools/entrant').glob('*.py')]
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w:gz') as tar:
        for name in files:
            assert not any(x in name.lower() for x in BAD)
            tar.add(ROOT/name, arcname=name, recursive=False)
    code = '''
import hashlib, io, json, shutil, sys, tarfile
from pathlib import Path
p=Path.home()/'entrant'; assert p.parent.is_dir(); p.mkdir(exist_ok=True)
bad=('sealed','holdout','heldout','cal_split','chunk5','j4b','inc1','hetdata_truth')
runtime=p/'runtime'
for name in ('e3data','pairs300','wm2'):
    q=p/name; original=Path.home()/'lrdiag'/name
    assert original.is_dir()
    if not q.exists(): q.symlink_to(original,target_is_directory=True)
if not runtime.exists():
    src=Path.home()/'lrdiag/runtime'; assert src.is_dir()
    for f in src.rglob('*'):
        if any(x in str(f).lower() for x in bad) or '__pycache__' in str(f): continue
        if f.is_file() and f.suffix in ('.py','.toml','.json'):
            q=runtime/f.relative_to(src); q.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(f,q)
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz') as tar:
    for m in tar:
        q=runtime/m.name
        assert q.resolve().is_relative_to(runtime.resolve())
        assert not any(x in str(q).lower() for x in bad) and not m.issym() and not m.islnk()
        tar.extract(m,runtime,filter='data')
receipt={str(f.relative_to(runtime)):hashlib.sha256(f.read_bytes()).hexdigest()
         for f in runtime.rglob('*') if f.is_file() and f.suffix in ('.py','.toml','.json')
         and not any(x in str(f).lower() for x in bad)}
(p/'SOURCE.json').write_text(json.dumps(receipt,sort_keys=True))
print(json.dumps({'files':len(receipt),'root':str(p)}))
'''
    ssh(host, code, stream.getvalue())


if __name__ == '__main__':
    stage(sys.argv[1])
