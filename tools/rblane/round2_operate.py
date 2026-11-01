"""Identity-receipted CPU-only round-2 diagnostics; no signals or peer jobs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import time
import zipfile

from .operate import remote, PYTHON, ROOT

RECEIPTS = ROOT/'docs/rb2lane'


def launch(args):
    if not RECEIPTS.parent.is_dir():
        raise ValueError('missing existing documentation parent')
    RECEIPTS.mkdir(exist_ok=True)
    record_path = RECEIPTS/(args.kind+'_DISPATCH.json')
    if record_path.exists():
        raise ValueError('immutable dispatch exists; inspect rather than duplicate')
    with tempfile.TemporaryDirectory(prefix='rb2lane_', dir='E:/Temp/opencode') as temporary:
        bundle = Path(temporary)/'code.zip'
        hashes = {}
        with zipfile.ZipFile(bundle, 'w', zipfile.ZIP_DEFLATED) as archive:
            paths = [*sorted((ROOT/'src/wms').rglob('*.py')), *sorted((ROOT/'tools/rblane').glob('*.py')),
                *sorted((ROOT/'artifacts/rbfit/lane_raw_c5602102891e').glob('*.json'))]
            for p in paths:
                rel = p.relative_to(ROOT).as_posix()
                data = p.read_bytes()
                hashes[rel] = hashlib.sha256(data).hexdigest()
                archive.writestr(rel, data)
        digest = hashlib.sha256(bundle.read_bytes()).hexdigest()
        run = '/home/ubuntu/rb2lane20261101_'+args.kind+'_'+digest[:12]
        preflight = json.loads(remote(f'''import pathlib,os,json
p=pathlib.Path({run!r})
assert p.parent.is_dir() and not p.exists()
assert os.getloadavg()[0]<20
p.mkdir()
print(json.dumps(dict(root=str(p),load=os.getloadavg(),cpus=os.cpu_count())))
'''))
        record = dict(kind=args.kind, root=run, bundle_sha256=digest, source_sha256=hashes,
            parent_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            worlds=args.worlds, arms=args.arms, samples=args.samples, horizon=args.horizon,
            preflight=preflight, status='STAGING', started_at=time.time(), cpu_only=True)
        record_path.write_text(json.dumps(record, indent=2)+'\n')
        subprocess.run(['scp','-q',str(bundle),'cpuQ:'+run+'/code.zip'],check=True,timeout=60)
    code = f'''import pathlib,os,hashlib,json,zipfile,subprocess,time
p=pathlib.Path({run!r})
assert hashlib.sha256((p/'code.zip').read_bytes()).hexdigest()=={digest!r}
with zipfile.ZipFile(p/'code.zip') as z:z.extractall(p)
env=dict(os.environ,PYTHONPATH=str(p/'src'),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
cmd=['/usr/bin/nice','-n','19',{PYTHON!r},'-B','-u','-m','tools.rblane.round2_validate',
'--continuation',str(p/'artifacts/rbfit/lane_raw_c5602102891e'),'--out',str(p/'results'),
'--samples',{str(args.samples)!r},'--horizon',{str(args.horizon)!r},'--worlds']+{[str(w) for w in args.worlds]!r}+['--arms']+{args.arms!r}
with (p/'run.log').open('x') as log:
 child=subprocess.Popen(cmd,cwd=p,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
proc=dict(pid=child.pid,start_ticks=pathlib.Path('/proc/'+str(child.pid)+'/stat').read_text().split()[21],command=cmd)
(p/'PID.json').write_text(json.dumps(proc,indent=2)+'\\n')
print(json.dumps(proc))
'''
    record.update(status='DISPATCHED', process=json.loads(remote(code)))
    record_path.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps({key:value for key,value in record.items() if key!='source_sha256'}, indent=2))


def status(args):
    record = json.loads((RECEIPTS/(args.kind+'_DISPATCH.json')).read_text())
    run = record['root']
    report = json.loads(remote(f'''import pathlib,json,os
p=pathlib.Path({run!r}); proc=json.loads((p/'PID.json').read_text())
f=pathlib.Path('/proc/'+str(proc['pid'])+'/stat')
alive=f.exists() and f.read_text().split()[21]==proc['start_ticks'] and f.read_text().split()[2]!='Z'
summary=p/'results/SUMMARY.json'
print(json.dumps(dict(alive=alive,complete=summary.is_file(),load=os.getloadavg(),
progress=(p/'run.log').read_text().splitlines()[-8:],
summary=json.loads(summary.read_text()) if summary.is_file() else None)))
'''))
    brief = {k:v for k,v in report.items() if k not in ('summary',)}
    if report['summary']:
        brief['elapsed_s'] = report['summary']['elapsed_s']
        brief['scores'] = {h:{m:r['exact'] for m,r in reports.items()}
            for h,reports in report['summary']['summary'].items()}
        brief['progress'] = 'complete'
    print(json.dumps(brief, indent=2))
    if args.command == 'collect':
        if report['alive'] or not report['complete']:
            raise ValueError('cannot collect live/incomplete diagnostic')
        out = RECEIPTS/(args.kind+'_results')
        if out.exists():
            raise ValueError('collection exists')
        out.mkdir()
        subprocess.run(['scp','-q','cpuQ:'+run+'/results/*.json',str(out)],check=True,timeout=120)
        (RECEIPTS/(args.kind+'_COMPLETE.json')).write_text(json.dumps(dict(
            root=run, process=record['process'], complete=True, alive=False,
            collected_at=time.time(), bundle_sha256=record['bundle_sha256']),indent=2)+'\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=('launch','status','collect'))
    ap.add_argument('--kind', required=True)
    ap.add_argument('--worlds', nargs='+', type=int, default=[969400])
    ap.add_argument('--arms', nargs='+', default=['native'])
    ap.add_argument('--samples', type=int, default=8)
    ap.add_argument('--horizon', type=int, choices=(60,300), default=60)
    args = ap.parse_args()
    from .validate import ARMS
    import re
    if (not re.fullmatch('[a-z0-9_]+', args.kind) or not set(args.worlds)<=set(range(969400,969432))
            or not set(args.arms)<=set(ARMS) or args.samples<1):
        raise ValueError('invalid immutable FIT experiment')
    launch(args) if args.command == 'launch' else status(args)


if __name__ == '__main__':
    main()
