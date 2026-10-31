"""Singleton observer for corrected HOLD checkpoints, diagnostics and preserving archives."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import time

from tools.coopbandit.ops import remote
from tools.coopfree_publish import (git, immutable, replace_block, DocumentationBusy,
                                    dev_text, discovery_text)

LOCAL = Path(__file__).resolve().parents[1]
ROOT = '/home/ubuntu/coopfree20261101_build5'
PUBLIC = LOCAL/'tools/coopfree/publications/hold'
ARCHIVE = Path('E:/Temp/opencode/coopfree20261101_observable_archive')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def collect(known):
    return json.loads(remote('cpuQ', f'''import pathlib,json,hashlib,base64,os,time
r=pathlib.Path({ROOT!r});read=lambda p:json.loads(p.read_text())
known=set({known!r});files={{}}
binding=read(r/'BINDING.json')
assert hashlib.sha256((r/'BINDING.json').read_bytes()).hexdigest()=='987a7ae0bd0c056bd06f8c6b4a099f00852ca2ba98961659efd6bb5a1481e05f'
for name,expected in binding['sources'].items():
 assert hashlib.sha256((r/'pytools'/name).read_bytes()).hexdigest()==expected
cert=read(r/'HOLD_CPU_APPROVED.json')
assert cert['pass_'] and cert['tests']==22 and cert['sim_s']==0 and not cert['skips']
assert cert['design_sha256']=='8be794771cd449c1341ceafeed30eb046faef53fe45471c0c6ad61fe3b1511f3'
assert cert['sources']['coopfree_hold.py']=='01a46df016e2e58c7097d125d90463274d8952cf2804ffc57577caa697faaadb'
for name,expected in cert['sources'].items():
 assert hashlib.sha256((r/'pytools/tools'/name).read_bytes()).hexdigest()==expected
def add(name,alias=None):
 p=r/name;alias=alias or name
 if p.exists() and alias not in known:
  data=p.read_bytes();files[alias]=dict(sha256=hashlib.sha256(data).hexdigest(),base64=base64.b64encode(data).decode())
for name in ('BINDING.json','HOLD_CPU_APPROVED.json','HOLD_AUTHORITY.json','HOLD_DISPATCH.json','HOLD_GATE_PARKED.json','HOLD_RETIRED_LEGACY.json','HOLD_STOP.json','STOP.json','HOLD_COMPLETE.json','HOLD_ZERO_DEV.json','HOLD_DEV_001.json','HOLD_DISCOVERY_001.json'):
 add(name)
if (r/'HOLD_UPDATE_001.json').exists():
 report=read(r/'HOLD_UPDATE_001.json')
 assert hashlib.sha256((r/'policy_hold_u001.pt').read_bytes()).hexdigest()==report['checkpoint_sha256']
 assert report['windows']>=150 and report['worlds']>=12
 assert report['baseline'].startswith('zero-update policy-mean HOLD')
 add('HOLD_UPDATE_001.json','UPDATE_001.json');add('policy_hold_u001.pt','policy_u001.pt')
dispatch=read(r/'HOLD_DISPATCH.json');p=pathlib.Path('/proc/'+str(dispatch['pid'])+'/stat')
alive=p.exists() and p.read_text().split()[21]==dispatch['start_ticks'] and p.read_text().split()[2]!='Z'
rows=[json.loads(s) for s in (r/'CHARGES.jsonl').read_text().splitlines()]
closed=[read(r/'jobs'/row['id']/'CLOSE.json') for row in rows if (r/'jobs'/row['id']/'CLOSE.json').exists()]
plan=read(r/'HOLD_PLAN_001.json') if (r/'HOLD_PLAN_001.json').exists() else None
progress=None
if plan:
 progress=dict(tasks=len(plan['tasks']),baselines=len(set(plan['baseline_ids'])),exact_credit_windows=len(plan['exact_indices']),baseline_cost=plan['hold_baseline_cost'],
  complete=sum((r/'jobs'/t['id']/'RESULT.json').exists() and (r/'jobs'/t['id']/'CLOSE.json').exists() for t in plan['tasks']))
archive=[];d=pathlib.Path('/dev/shm/coopfree20261101_observable')
if d.is_dir():
 for p in sorted(d.iterdir()):
  if p.is_file() and p.name.endswith('.jsonl.gz'):
   result=read(r/'jobs'/p.name[:-9]/'RESULT.json')
   archive.append(dict(name=p.name,bytes=p.stat().st_size,sha256=result['trace_sha256']))
print(json.dumps(dict(files=files,dispatch=dispatch,owned_alive=alive,charged=sum(row['sim_s'] for row in rows),actual_closed=sum(row['actual_sim_s'] for row in closed),progress=progress,archive=archive,unix=time.time(),load=os.getloadavg())))
''', timeout=25))


def describe(snapshot):
    lines = ['## Corrected HOLD-credit production',
             'Public artifacts: `tools/coopfree/publications/hold/`; verify its `MANIFEST.json`.',
             'Legacy MAPPO-credit checkpoint is retained remotely and excluded from corrected learning.',
             'Owned cpuQ supervisor PID%s/start%s; alive=%s; four nice19 single-thread workers.' %
             (snapshot['dispatch']['pid'], snapshot['dispatch']['start_ticks'], snapshot['owned_alive'])]
    progress = snapshot['progress']
    if progress:
        lines.append('New HOLD physics: %d/%d complete tasks; %d baselines; exact solo diagnostics for %d/156 windows.' %
                     (progress['complete'], progress['tasks'], progress['baselines'], progress['exact_credit_windows']))
    if (PUBLIC/'UPDATE_001.json').exists():
        r = json.loads((PUBLIC/'UPDATE_001.json').read_text())
        lines += ['### Learned proposal checkpoint u001',
                  '- `tools/coopfree/publications/hold/policy_u001.pt`',
                  '- SHA256 `'+r['checkpoint_sha256']+'`; receipt `hold/UPDATE_001.json`.',
                  '- %d windows/%d worlds; D_G is against exact zero-update mean HOLD.' % (r['windows'],r['worlds']),
                  '- Exact HOLD D_i/C_G coverage: %d windows/%d worlds; remaining windows have no exact solo diagnostic.' %
                  (r['exact_credit_windows'],r['exact_credit_worlds']),
                  '- Executed-change %.4f; issued-change %.4f; executed/issued %s; safety override %.4f.' %
                  (r['executed_change_share'],r['issued_change_share'],r['executed_vs_issued_share'],r['safety_override_share']),
                  '- Sigma %s -> %s; objective2D_G; continuous Gaussian score; W exogenous.' % (r['sigma_used'],r['sigma_next'])]
        for key in ('D_G','sum_D_i','C_G'):
            value = r[key]
            lines.append('- %s: n=%d; mean %.3f; sd %.3f; min/Q1/median/Q3/max %s.' %
                         (key,value['n'],value['mean'],value['sd'],value['quantiles']) if value else '- '+key+': unavailable.')
        lines.append('- Exact subset share C_G>0: %s.' % r['positive_C_G_share'])
        for h, value in r['hold_minus_mappo'].items():
            lines.append('- HOLD-minus-MAPPO structural gap at%ss: mean %.3f vehicle-s (TRAIN windows, not learned gain).' % (h,value['mean']))
    else:
        lines.append('Corrected learned checkpoint pending; no zero-update or legacy checkpoint is advertised as corrected.')
    for name, label in (('HOLD_DEV_001.json','Corrected learned policy'),('HOLD_ZERO_DEV.json','Zero-update mean HOLD structural baseline')):
        if (PUBLIC/name).exists():
            lines += ['### '+label] + dev_text(json.loads((PUBLIC/name).read_text()))
    if (PUBLIC/'HOLD_DISCOVERY_001.json').exists():
        lines += discovery_text(json.loads((PUBLIC/'HOLD_DISCOVERY_001.json').read_text()))
    lines += ['Budget: %d precharged / %.0f closed-actual sim-s; hard cap2,000,000; no refunds/retries.' %
              (snapshot['charged'],snapshot['actual_closed']),
              'TRAIN exploration credit is not paired dev-policy performance.']
    if (PUBLIC/'HOLD_STOP.json').exists() or (PUBLIC/'STOP.json').exists():
        lines.append('**Durable STOP observed; inspect receipts. No automatic restart.**')
    elif (PUBLIC/'HOLD_COMPLETE.json').exists():
        lines.append('Corrected training/dev/discovery COMPLETE; exact-credit coverage limitation remains disclosed.')
    return '\n'.join(lines)


def archive_batch(rows):
    """Copy, never remove, relocated tapes to durable approved external storage."""
    assert ARCHIVE.parent.is_dir()
    ARCHIVE.mkdir(exist_ok=True)
    missing = []
    for row in rows:
        name = row['name']
        if not re.fullmatch(r'[A-Za-z0-9_-]+\.jsonl\.gz', name):
            raise ValueError('unsafe archive name')
        p = ARCHIVE/name
        if p.exists():
            if p.stat().st_size != row['bytes'] or sha(p.read_bytes()) != row['sha256']:
                raise ValueError('existing archive hash mismatch')
        else:
            missing.append(row)
    chosen = missing[:64]
    if not chosen:
        return dict(complete=True, verified_files=len(rows))
    assert shutil.disk_usage(ARCHIVE).free > 3*sum(r['bytes'] for r in chosen)+1024**3
    number = len(list(ARCHIVE.glob('batch_*.tar')))+1
    tarpath = ARCHIVE/f'batch_{number:04}.tar'
    with tarpath.open('xb') as f:
        subprocess.run(['ssh','-o','BatchMode=yes','cpuQ','tar','-C',
                        '/dev/shm/coopfree20261101_observable','-cf','-',*[r['name'] for r in chosen]],
                       stdout=f,stderr=subprocess.PIPE,check=True,timeout=90)
    with tarfile.open(tarpath) as tape:
        members = tape.getmembers()
        assert {m.name for m in members} == {r['name'] for r in chosen}
        for row in chosen:
            member = tape.getmember(row['name'])
            assert member.isfile() and member.size == row['bytes']
            data = tape.extractfile(member).read()
            assert sha(data) == row['sha256']
            immutable(ARCHIVE/row['name'], data)
    immutable(ARCHIVE/f'batch_{number:04}.json', (json.dumps(chosen,indent=2)+'\n').encode())
    return dict(complete=len(chosen)==len(missing), verified_files=len(rows)-len(missing)+len(chosen), copied=len(chosen))


def once(commit):
    PUBLIC.mkdir(exist_ok=True)
    known = [p.name for p in PUBLIC.iterdir() if p.is_file()]
    snapshot = collect(known)
    decoded = {}
    for name, item in snapshot.pop('files').items():
        if Path(name).name != name or not (name.endswith('.json') or name=='policy_u001.pt'):
            raise ValueError('unapproved public path')
        data = base64.b64decode(item['base64'],validate=True)
        assert sha(data) == item['sha256']
        decoded[name] = data
    for name in ('HANDOFF.md','docs/paper_results/COOPFREE.md'):
        if git('status','--porcelain','--',name).stdout.strip():
            raise DocumentationBusy(name)
    for name, data in decoded.items():
        immutable(PUBLIC/name,data)
    lastpath = PUBLIC/'LATEST_STATUS.json'
    old = json.loads(lastpath.read_text()) if lastpath.exists() else None
    archive_rows = snapshot.pop('archive')
    if decoded or old is None or snapshot['unix']-old['unix'] >= 600:
        replace_block(LOCAL/'HANDOFF.md',describe(snapshot))
        replace_block(LOCAL/'docs/paper_results/COOPFREE.md',describe(snapshot))
        lastpath.write_text(json.dumps(snapshot,indent=2)+'\n')
        files = {p.name:sha(p.read_bytes()) for p in PUBLIC.iterdir() if p.is_file() and p.name!='MANIFEST.json'}
        (PUBLIC/'MANIFEST.json').write_text(json.dumps(dict(format='coopfree/proposals/1',
           root=ROOT,scope='corrected HOLD-credit learner; exact-CG subset disclosed',files=files),indent=2)+'\n')
        if commit:
            for args in (('status','--short'),('diff','--','HANDOFF.md','docs/paper_results/COOPFREE.md'),('log','--oneline','-10')):
                print(git(*args).stdout,flush=True)
            git('diff','--check')
            paths=['HANDOFF.md','docs/paper_results/COOPFREE.md','tools/coopfree/publications/hold']
            git('add','--',*paths)
            print(git('commit','-m','coopfree: publish corrected HOLD production milestone','--only','--',*paths).stdout,flush=True)
    archived = archive_batch(archive_rows)
    print(json.dumps(dict(progress=snapshot['progress'],owned_alive=snapshot['owned_alive'],archive=archived)),flush=True)
    terminal = (PUBLIC/'HOLD_COMPLETE.json').exists() or (PUBLIC/'HOLD_STOP.json').exists() or (PUBLIC/'STOP.json').exists()
    return terminal and archived['complete']


def main():
    p=argparse.ArgumentParser();p.add_argument('--spawn',action='store_true');p.add_argument('--commit',action='store_true');p.add_argument('--hours',type=float,default=12);a=p.parse_args()
    assert 0 < a.hours <= 24
    gitdir=Path(git('rev-parse','--absolute-git-dir').stdout.strip())
    assert gitdir.is_dir() and ARCHIVE.parent.is_dir()
    if a.spawn:
        logpath=gitdir/f'coopfree-hold-publisher-{time.time_ns()}.log'
        cmd=[sys.executable,'-B','-u','-m','tools.coopfree_hold_publish','--hours',str(a.hours)]
        if a.commit: cmd.append('--commit')
        options=dict(creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS) if os.name=='nt' else dict(start_new_session=True)
        with logpath.open('x') as log:
            child=subprocess.Popen(cmd,cwd=LOCAL,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,**options)
        receipt=dict(pid=child.pid,unix=time.time(),log=str(logpath),command=cmd,physics_dispatched=0)
        immutable(LOCAL/f'tools/coopfree/evidence/HOLD_PUBLISHER_{child.pid}.json',(json.dumps(receipt,indent=2)+'\n').encode())
        print(json.dumps(receipt));return
    with (gitdir/'coopfree-publisher.lock').open('a+b') as lock:
        if os.name=='nt':
            import msvcrt
            lock.seek(0);lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        else:
            import fcntl
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        deadline=time.monotonic()+a.hours*3600
        while True:
            try:
                done=once(a.commit)
            except DocumentationBusy as error:
                print(json.dumps(dict(documentation_busy=str(error))),flush=True);done=False
            except (subprocess.TimeoutExpired, ConnectionError) as error:
                print(json.dumps(dict(transport_error=str(error),no_physics_retry=True)),flush=True);done=False
            if done or time.monotonic()>=deadline: return
            time.sleep(60)


if __name__=='__main__':
    main()
