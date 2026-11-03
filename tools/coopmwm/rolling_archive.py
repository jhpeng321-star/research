"""MAIN16:47 durable rolling mover. No simulation or outcome decoding."""
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import time
import archive_move as base

CONTROL=Path('/home/ubuntu/coopmwm20261103_rolling1647_v3')
DEST='/root/cpuq_archive_20261103/rolling1647_v3'
MIN_AGE=1800
TARGET=8*1024**3
FLOOR=4*1024**3
LEGACY_SAFE=base.safe_file


def atomic(p,v):
    tmp=Path(str(p)+'.tmp')
    with tmp.open('w') as f:
        json.dump(v,f,sort_keys=True);f.flush();os.fsync(f.fileno())
    tmp.replace(p)


def job_kind(p):
    p=Path(p)
    if p.name!='observable.jsonl.gz' or p.parent.parent!=base.RUN/'jobs':return None
    # Final-policy detector inputs remain available locally. Evaluation outcomes
    # are never opened here; the initial release uses completed TRAIN tapes only.
    m=re.fullmatch(r'(?:replay0735_)?warm_u(\d{3})_([ABR])_(986\d{3})',p.parent.name)
    return (int(m[1]),m[2]) if m else None


def safe(p):
    p=Path(p)
    if base.BAD.search(str(p)) or p.is_symlink() or p.resolve()!=p:return False
    if p.name.lower().endswith(('.xml','.xml.gz','.sumocfg','.sumocfg.gz')):return False
    if 'checkpoint' in p.name.lower() or (p.suffix=='.pt' and p.name!='SNAPSHOT.pt'):return False
    return p.is_file() and (job_kind(p) is not None or diagnostic_kind(p) is not None or LEGACY_SAFE(p))


def diagnostic_kind(p):
    p=Path(p)
    try:parts=p.relative_to(base.RUN/'policy_synergy_isolated_1226_v2').parts
    except ValueError:return None
    if (len(parts)>=5 and re.fullmatch(r'u\d{3}_[BR]',parts[0])
            and parts[1]=='branches'):
        return int(parts[0][1:4]),parts[0][-1]
    return None


def aged(p,now):
    return now-p.stat().st_mtime>=MIN_AGE


def metadata_refs():
    """Direct references in public live metadata; no private/outcome traversal."""
    paths=set()
    def visit(v):
        if isinstance(v,str) and v.startswith(str(base.HOME)+'/') and not base.BAD.search(v):paths.add(v)
        elif isinstance(v,dict):
            for x in v.values():visit(x)
        elif isinstance(v,list):
            for x in v:visit(x)
    for root in base.HOME.glob('coopmwm*'):
        if not root.is_dir() or root.is_symlink() or base.BAD.search(str(root)):continue
        if 'archive' in root.name or 'rolling' in root.name:continue
        for p in root.glob('*.json'):
            if base.BAD.search(str(p)) or not any(k in p.name.upper() for k in ('BINDING','CONFIG','MANIFEST','RUNTIME')):continue
            if not p.is_symlink():visit(json.loads(p.read_text()))
    return paths


def candidates():
    found,skipped=base.candidates();now=time.time();refs=metadata_refs()
    # A's tapes are future composite-safety references. Only retire them after
    # BOTH B and R have already published the corresponding safety/update receipt.
    for p in (base.RUN/'jobs').glob('*/observable.jsonl.gz'):
        kind=job_kind(p)
        if kind is None:continue
        u,arm=kind
        if not (p.parent/'RESULT.json').is_file() or not (p.parent/'CLOSE.json').is_file():continue
        if not aged(p.parent/'CLOSE.json',now):continue
        # B/R fixed endpoints are24. A25..48 references cannot be consumed by
        # any future same-update guard; A's own completed update is mandatory.
        arms=(('A',) if u>24 else ('B','R')) if arm=='A' else (arm,)
        if not all((base.RUN/f'{a}_UPDATE_{u:03}.json').is_file()
                   and aged(base.RUN/f'{a}_UPDATE_{u:03}.json',now) for a in arms):continue
        found[str(p)]='finished_train_observable'
    diag=base.RUN/'policy_synergy_isolated_1226_v2'
    for case in diag.glob('u[0-9][0-9][0-9]_[BR]'):
        u=int(case.name[1:4]);arm=case.name[-1]
        report=base.RUN/f'POLICY_SYNERGY_u{u:03}_{arm}_ISOLATED_V2.json'
        if not report.is_file() or not aged(report,now):continue
        # Read only the registered TRAIN diagnostic completion, never eval outcomes.
        if json.loads(report.read_text()).get('status')!='complete':continue
        for directory,dirs,names in os.walk(case/'branches',followlinks=False):
            dirs[:]=[n for n in dirs if not base.BAD.search(str(Path(directory)/n))
                     and not (Path(directory)/n).is_symlink()]
            for n in names:found[str(Path(directory)/n)]='completed_isolated_diagnostic'
    opened=base.open_files()
    files,trees,keys=base.protected(base.final_labels())
    kept={}
    for name,category in found.items():
        p=Path(name)
        if category=='c_completed_policy_synergy':
            part=p.relative_to(base.RUN/'policy_synergy_1205_v1').parts[0]
            u=int(part[1:4]);arm=part[-1]
            iso=base.RUN/f'POLICY_SYNERGY_u{u:03}_{arm}_ISOLATED_V2.json'
            if not iso.is_file() or not aged(iso,now):continue
        if not safe(p) or not aged(p,now) or name in opened:continue
        if name in files or base.under(name,trees):continue
        # A root/jobs directory is a storage location, not a leaf dependency.
        if any(name==r or (r.startswith(str(base.RUN/'jobs')+'/') and name.startswith(r+'/')) for r in refs):continue
        kept[name]=category
    return kept,skipped


def ssh():
    transport=json.loads((CONTROL/'TRANSPORT.json').read_text())
    env=dict(os.environ,SSH_AUTH_SOCK=transport['socket'])
    if not Path(transport['socket']).exists():raise RuntimeError('forwarded agent unavailable; sources retained')
    return base.ssh_command(),env


def remote(code):
    command,env=ssh()
    p=subprocess.run(command+['/root/miniconda3/bin/python -'],input=code,env=env,
                     text=True,capture_output=True,timeout=600)
    if p.returncode:raise RuntimeError('archive remote check failed: '+p.stderr[-1500:])
    return json.loads(p.stdout)


def destination_free():
    return remote(f'''from pathlib import Path
import json,shutil
p=Path({DEST!r});assert p.parent.is_dir()
print(json.dumps(dict(free=shutil.disk_usage(p.parent).free)))''')['free']


def capacity(free,bytes_):
    return free>=FLOOR and free-bytes_>=FLOOR


def remove_one(e,proof,eligible,opened):
    p=Path(e['source'])
    if proof!=e['sha256']:raise ValueError('destination SHA mismatch')
    if str(p) not in eligible or str(p) in opened:return False
    if not safe(p) or not aged(p,time.time()):return False
    st=p.stat()
    if (st.st_ino,st.st_size,st.st_mtime_ns)!=(e['inode'],e['bytes'],e['mtime_ns']):
        raise ValueError('source identity changed')
    if base.digest_file(p)!=e['sha256']:raise ValueError('source SHA changed')
    p.unlink();return True


def batch(paths,index):
    folder=CONTROL/f'batch_{index:06}';folder.mkdir()
    entries=[]
    for name,category in paths:
        p=Path(name);st=p.stat()
        entries.append(dict(source=name,relative=str(p.relative_to(base.HOME)),category=category,
            bytes=st.st_size,sha256=base.digest_file(p),inode=st.st_ino,mtime_ns=st.st_mtime_ns))
    base.write(folder/'MANIFEST.json',dict(entries=entries,authority_sha256=base.digest_file(CONTROL/'AUTHORITY.json')))
    mh=base.digest_file(folder/'MANIFEST.json');needed=sum(e['bytes'] for e in entries)
    if not capacity(destination_free(),needed):
        base.write(folder/'CAPACITY_BLOCKED.json',dict(unix=time.time(),bytes=needed,floor=FLOOR));return 0
    remote(f'''from pathlib import Path
import json
p=Path({DEST!r});assert p.parent.is_dir();p.mkdir(exist_ok=True)
print(json.dumps(dict(ready=True)))''')
    listing=folder/'FILES.null'
    listing.write_bytes(b''.join(e['relative'].encode()+b'\0' for e in entries))
    command,env=ssh()
    args=['rsync','-a','--relative','--checksum','--no-owner','--no-group','--from0',
        '--files-from='+str(listing),'-e',' '.join(shlex.quote(x) for x in command[:-1]),
        str(base.HOME)+'/',command[-1]+':'+DEST+'/']
    p=subprocess.run(args,env=env,text=True,capture_output=True,timeout=1800)
    if p.returncode:raise RuntimeError('rsync failed; sources retained: '+p.stderr[-1500:])
    checks=[dict(relative=e['relative'],sha256=e['sha256'],bytes=e['bytes']) for e in entries]
    proof=remote(f'''from pathlib import Path
import json,hashlib,subprocess,shutil
root=Path({DEST!r});out={{}}
for e in {checks!r}:
 p=root/e['relative'];assert p.is_file() and not p.is_symlink()
 h=subprocess.run(['sha256sum','--',str(p)],capture_output=True,text=True,check=True).stdout.split()[0]
 assert h==e['sha256'] and p.stat().st_size==e['bytes']
 out[e['relative']]=h
print(json.dumps(dict(hashes=out,free=shutil.disk_usage(root).free)))''')
    # Durable proof precedes the first removal; a crash is always auditable.
    base.write(folder/'REMOTE_VERIFIED.json',dict(manifest_sha256=mh,**proof))
    if proof['free']<FLOOR:raise RuntimeError('gpuR below 4GiB; retain sources and stop')
    eligible,_=candidates();opened=base.open_files();deleted=[]
    for e in entries:
        if remove_one(e,proof['hashes'][e['relative']],eligible,opened):deleted.append(e['relative'])
    moved=sum(e['bytes'] for e in entries if e['relative'] in deleted)
    base.write(folder/'ARCHIVE_MOVE.json',dict(manifest_sha256=mh,destination=DEST,
        deleted_sources=deleted,verified_files=len(entries),moved_bytes=moved,
        cpu_free=shutil.disk_usage(base.HOME).free,gpu_free=proof['free'],unix=time.time(),new_sim_s=0))
    return moved


def run():
    import fcntl
    assert CONTROL.is_dir()
    authority=json.loads((CONTROL/'AUTHORITY.json').read_text())
    assert authority['source_hashes']['rolling_archive.py']==base.digest_file(__file__)
    assert authority['source_hashes']['archive_move.py']==base.digest_file(base.__file__)
    claim=(CONTROL/'RUN.claim').open('a');fcntl.flock(claim,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (CONTROL/'OWNER.json').exists():raise ValueError('adopt existing rolling owner')
    base.write(CONTROL/'OWNER.json',base.owner());total=0;index=0
    while True:
        try:
            free=destination_free()
            if free<FLOOR:
                atomic(CONTROL/'STOP_GPU_SPACE.json',dict(unix=time.time(),gpu_free=free,floor=FLOOR));return
            found,skipped=candidates()
            atomic(CONTROL/'STATUS.json',dict(unix=time.time(),cpu_free=shutil.disk_usage(base.HOME).free,
                gpu_free=free,eligible_files=len(found),eligible_bytes=sum(Path(p).stat().st_size for p in found),
                moved_bytes=total,batches=index,skipped=skipped,target_bytes=TARGET,new_sim_s=0))
            # Small batches bound the verification payload and destination margin.
            ordered=sorted(found.items(),key=lambda x:(x[1],-Path(x[0]).stat().st_size,x[0]))
            while ordered and shutil.disk_usage(base.HOME).free<TARGET:
                xs=[];size=0
                while ordered and size<128*1024**2:
                    item=ordered.pop(0);xs.append(item);size+=Path(item[0]).stat().st_size
                amount=batch(xs,index);index+=1;total+=amount
                atomic(CONTROL/'PROGRESS.json',dict(unix=time.time(),cpu_free=shutil.disk_usage(base.HOME).free,
                    moved_bytes=total,batches=index,new_sim_s=0))
                if not amount:break
            time.sleep(900)
        except Exception as e:
            atomic(CONTROL/'ALERT.json',dict(unix=time.time(),error=repr(e),new_sim_s=0,sources_retained=True))
            # Transient transport faults never kill the durable storage owner.
            # Per-batch claims are not replayed; every retry uses a fresh manifest.
            index=max(index,len(list(CONTROL.glob('batch_*'))));time.sleep(60)


if __name__=='__main__':
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    os.nice(max(0,19-os.getpriority(os.PRIO_PROCESS,0)))
    run()
