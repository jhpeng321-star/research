"""MAIN14:50: manifest/rsync/remote-SHA/source-SHA gated move, no simulation.

Standalone stdlib worker on cpuQ. Forbidden/scenario files are never hashed or
transferred. Only explicitly authorised completed artifact classes are eligible.
"""
import argparse
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

HOME=Path('/home/ubuntu')
RUN=HOME/'coopmwm20261102_warm1'
LS=HOME/'coopmwm20261102_lscollect1'
CONTROL=HOME/'coopmwm20261103_archive1450_v2'
DEST='/root/cpuq_archive_20261103/coopmwm_1450_v1'
TARGET=10*1024**3
BAD=re.compile(r'sealed|holdout|heldout|cal_split|chunk5|j4b|inc1|hetdata_truth|vehicle_params|driver_params|vtype|assignment',re.I)


def digest_file(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def write(p,v):
    with Path(p).open('x') as f:
        json.dump(v,f,sort_keys=True);f.flush();os.fsync(f.fileno())


def allowed_source(p):
    p=Path(p)
    try:
        parts=p.relative_to(LS/'label_attempts').parts
        return (len(parts)>=4 and re.fullmatch('[A-Z0-9]+',parts[0]) is not None
                and parts[1]=='branches' and parts[2].startswith('sv_'))
    except ValueError:pass
    try:
        parts=p.relative_to(LS/'pending').parts
        return bool(parts and re.match('[a-f0-9]{64}(?:[.]|$)',parts[0]))
    except ValueError:pass
    try:
        parts=p.relative_to(RUN/'policy_synergy_1205_v1').parts
        return bool(len(parts)>=3 and re.fullmatch('u[0-9]{3}_[BR]',parts[0]) and parts[1]=='branches')
    except ValueError:pass
    if p.name=='observable.jsonl.gz' and p.parent.parent==RUN/'jobs':
        match=re.fullmatch('interim008_[ABR]_([0-9]{6})',p.parent.name)
        return bool(match and (989600<=int(match[1])<=989635 or 990600<=int(match[1])<=990659))
    return False


def safe_file(p):
    p=Path(p)
    if BAD.search(str(p)) or p.is_symlink() or not allowed_source(p):return False
    if BAD.search(str(p.resolve())) or not allowed_source(p.resolve()):return False
    name=p.name.lower()
    if name.endswith(('.xml','.xml.gz','.sumocfg','.sumocfg.gz')):return False
    if 'checkpoint' in name or (name.endswith('.pt') and p.name!='SNAPSHOT.pt' and p.parent!=LS/'pending'):return False
    return p.is_file()


def owner():
    return dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),uid=os.getuid(),unix=time.time())


def open_files():
    paths=set();denied=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            for fd in (proc/'fd').iterdir():
                try:
                    name=os.readlink(fd)
                    if name.startswith(str(HOME)+'/'):paths.add(name.removesuffix(' (deleted)'))
                except FileNotFoundError:pass
                except PermissionError:denied.append(str(fd))
        except FileNotFoundError:pass
        except PermissionError:denied.append(str(proc/'fd'))
    if denied:raise ValueError('cannot prove open-file exclusion for all processes')
    return paths


def final_labels():
    out=[]
    for p in (LS/'labels').glob('*.json'):
        x=json.loads(p.read_text());r=x['verification']
        if r['status'] not in ('accepted','rejected','inactionable'):continue
        out.append((p,x,r))
    return out


def protected(labels):
    """Exact live bank/lesson state inputs remain on cpuQ, including V2V inputs."""
    quality=json.loads((RUN/'replay_bank/B/BANK_QUALITY.json').read_text())
    admitted={x['verification_id'] for x in quality['synergy_twins'] if x.get('admitted')}
    if len(admitted)!=quality['admitted']:raise ValueError('incomplete admitted-bank protection set')
    lst=RUN/'LESSON_LIST_B.json'
    if lst.exists():admitted.update(json.loads(lst.read_text())['verification_ids'])
    files=set();trees=set();keys=set()
    for p,x,r in labels:
        if r['verification_id'] not in admitted:continue
        key=x['search_id'];keys.add(key)
        suffix=p.stem[len(key):]
        branchroot=LS/'label_attempts'/suffix[1:]/'branches' if suffix else LS/'branches'
        if r.get('chosen') is not None:
            branch=r['branches'][r['chosen']+1]['branch']
            trees.add(str(branchroot/r['verification_id']/branch))
    # Direct leaf/case references in live bindings/configs stay put as well.
    def strings(x):
        if isinstance(x,str):yield x
        elif isinstance(x,dict):
            for v in x.values():yield from strings(v)
        elif isinstance(x,list):
            for v in x:yield from strings(v)
    for base in (RUN,LS):
        for p in base.glob('*.json'):
            if BAD.search(str(p)) or not any(s in p.name.upper() for s in ('BINDING','CONFIG')):continue
            for text in strings(json.loads(p.read_text())):
                if not text.startswith(str(HOME)+'/') or BAD.search(text):continue
                q=Path(text)
                if q.is_file():files.add(str(q))
                elif q.is_dir() and (q.name.startswith('sv_') or re.fullmatch('[a-f0-9]{64}',q.name)):
                    trees.add(str(q))
    return files,trees,keys


def under(p,trees):
    return any(str(p)==t or str(p).startswith(t+'/') for t in trees)


def candidates():
    labels=final_labels();files,trees,keys=protected(labels);found={};skipped=dict(forbidden_or_scenario=0,protected=0,open=0)
    opened=open_files()
    def add_tree(root,category):
        if not root.is_dir() or root.is_symlink():return
        for base,dirs,names in os.walk(root,followlinks=False):
            dirs[:]=[n for n in dirs if not BAD.search(str(Path(base)/n)) and not (Path(base)/n).is_symlink()]
            for n in names:add(Path(base)/n,category)
    def add(p,category):
        if not safe_file(p):skipped['forbidden_or_scenario']+=1;return
        if str(p) in files or under(p,trees):skipped['protected']+=1;return
        if str(p) in opened:skipped['open']+=1;return
        found[str(p)]=category
    for p,x,r in labels:
        key=x['search_id'];suffix=p.stem[len(key):]
        if suffix:
            assert re.fullmatch('_[A-Z0-9]+',suffix)
            add_tree(LS/'label_attempts'/suffix[1:]/'branches'/r['verification_id'],'a_label_attempts')
        if key not in keys:
            for item in (LS/'pending').glob(key+'*'):
                if item.is_dir():add_tree(item,'b_pending')
                else:add(item,'b_pending')
    control=RUN/'policy_synergy_1205_v1'
    for p in control.glob('u[0-9][0-9][0-9]_[BR]'):
        u=int(p.name[1:4]);a=p.name[-1];report=RUN/f'POLICY_SYNERGY_u{u:03}_{a}.json'
        if report.exists() and json.loads(report.read_text())['status']=='complete':
            add_tree(p/'branches','c_completed_policy_synergy')
    if ((RUN/'replay_composite_0957/interim_COMPLETE.json').exists()
            and (RUN/'interim_r_1235_v2/COMPLETE.json').exists()):
        for p in (RUN/'jobs').glob('interim008_*'):
            receipt=RUN/'final_receipts'/(p.name+'.json')
            if receipt.exists() and (p/'RESULT.json').exists() and (p/'CLOSE.json').exists():
                trace=p/'observable.jsonl.gz'
                if trace.exists():add(trace,'d_interim_observables')
    return found,skipped


def prepare():
    assert CONTROL.parent.is_dir()
    CONTROL.mkdir(exist_ok=True)
    cpu=json.loads((CONTROL/'CPU_RESULT.json').read_text())
    if not cpu['pass_'] or digest_file(__file__)!=cpu['source_hashes']['archive_move.py']:
        raise ValueError('qualified archive source required')
    if (CONTROL/'MANIFEST.json').exists():return json.loads((CONTROL/'PREPARE_COMPLETE.json').read_text())
    write(CONTROL/'PREPARE_OWNER.json',owner())
    found,skipped=candidates();entries=[]
    for name,category in sorted(found.items(),key=lambda x:(x[1],-Path(x[0]).stat().st_size,x[0])):
        p=Path(name);st=p.stat()
        h=digest_file(p)
        if category=='d_interim_observables':
            receipt=json.loads((RUN/'final_receipts'/(p.parent.name+'.json')).read_text())
            if h!=receipt['files']['observable.jsonl.gz']['sha256']:raise ValueError('interim trace differs from completion fingerprint')
        entries.append(dict(source=name,relative=str(p.relative_to(HOME)),category=category,
            bytes=st.st_size,sha256=h,inode=st.st_ino,mtime_ns=st.st_mtime_ns))
    write(CONTROL/'MANIFEST.json',dict(authority='MAIN2026-11-03 14:50',entries=entries,
        skipped=skipped,source_free=shutil.disk_usage(HOME).free,target_free_bytes=TARGET,destination=DEST,
        contract='selected completed artifacts only; no scenario/XML/forbidden files; admitted bank inputs protected'))
    result=dict(owner=owner(),manifest_sha256=digest_file(CONTROL/'MANIFEST.json'),files=len(entries),
        bytes=sum(e['bytes'] for e in entries),skipped=skipped,source_free=shutil.disk_usage(HOME).free,new_sim_s=0)
    write(CONTROL/'PREPARE_COMPLETE.json',result);return result


def ssh_command():
    return ['ssh','-T','-o','BatchMode=yes','-o','IdentityFile=none','-o','IdentitiesOnly=no',
        '-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile=/home/ubuntu/coopmwm20261102_remote_routes5/known_hosts_gpuR',
        '-o','ConnectTimeout=20','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=8',
        '-p','53897','root@connect.westb.seetacloud.com']


def remote(code):
    import base64
    encoded=base64.b64encode(code.encode()).decode()
    command=ssh_command()+['/root/miniconda3/bin/python -c '+shlex.quote('import base64;exec(base64.b64decode('+repr(encoded)+'))')]
    p=subprocess.run(command,capture_output=True,text=True,timeout=600)
    if p.returncode:raise RuntimeError('archive SSH verification failed: '+p.stderr[-3000:])
    return json.loads(p.stdout)


def delete_verified(entry,remote_sha,opened):
    p=Path(entry['source'])
    if remote_sha!=entry['sha256']:raise ValueError('remote SHA mismatch; source retained')
    if str(p) in opened:return False
    if not safe_file(p) or digest_file(p)!=entry['sha256']:raise ValueError('source changed; retained')
    st=p.stat()
    if st.st_ino!=entry['inode'] or st.st_size!=entry['bytes']:raise ValueError('source identity changed; retained')
    p.unlink();return True


def move(manifest_sha):
    import fcntl
    claim=(CONTROL/'MOVE.claim').open('a');fcntl.flock(claim,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (CONTROL/'MOVE_OWNER.json').exists():raise ValueError('adopt existing move owner; never duplicate')
    if digest_file(CONTROL/'MANIFEST.json')!=manifest_sha:raise ValueError('manifest changed')
    authority=json.loads((CONTROL/'AUTHORITY.json').read_text())
    if authority['manifest_sha256']!=manifest_sha:raise ValueError('manifest authority missing')
    if digest_file(__file__)!=authority['source_sha256']:raise ValueError('archive worker source changed')
    write(CONTROL/'MOVE_OWNER.json',owner());m=json.loads((CONTROL/'MANIFEST.json').read_text())
    # Both parent location and available target space are verified before transfer.
    target=remote(f'''from pathlib import Path
import json,shutil
p=Path({DEST!r});assert p.parent.parent.is_dir();p.parent.mkdir(exist_ok=True)
if p.exists():raise ValueError('archive namespace exists; inspect/adopt before any overwrite')
p.mkdir();print(json.dumps(dict(free=shutil.disk_usage(p).free)))''')
    needed=max(0,TARGET-shutil.disk_usage(HOME).free)
    if target['free']-needed<2*1024**3:raise ValueError('insufficient archive target margin')
    moved=0;batch=[];size=0
    def transfer(xs,index):
        nonlocal moved
        labels=final_labels();protected_files,protected_trees,protected_keys=protected(labels);opened=open_files()
        xs=[e for e in xs if e['source'] not in protected_files and not under(e['source'],protected_trees)
            and e['source'] not in opened and not (e['category']=='b_pending'
                and Path(e['source']).relative_to(LS/'pending').parts[0].split('.')[0] in protected_keys)]
        if not xs:return
        for e in xs:
            if not safe_file(e['source']) or digest_file(e['source'])!=e['sha256']:raise ValueError('pre-copy source SHA mismatch')
        listing=CONTROL/f'FILES_{index:04}.list'
        with listing.open('xb') as f:f.write(b''.join(e['relative'].encode()+b'\0' for e in xs))
        command=['rsync','-a','--relative','--checksum','--no-owner','--no-group','--from0','--files-from='+str(listing),
            '-e',' '.join(shlex.quote(x) for x in ssh_command()[:-1]),str(HOME)+'/',ssh_command()[-1]+':'+DEST+'/']
        p=subprocess.run(command,capture_output=True,text=True,timeout=1800)
        if p.returncode:raise RuntimeError('archive rsync failed; source retained: '+p.stderr[-3000:])
        checks=[dict(relative=e['relative'],sha256=e['sha256'],bytes=e['bytes']) for e in xs]
        verified=remote(f'''from pathlib import Path
import json,hashlib
base=Path({DEST!r});out={{}}
for e in {checks!r}:
 rel=Path(e['relative']);assert not rel.is_absolute() and '..' not in rel.parts
 p=base/rel;assert p.is_file() and not p.is_symlink();h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 assert p.stat().st_size==e['bytes'] and h.hexdigest()==e['sha256'],str(p)
 out[e['relative']]=h.hexdigest()
proof=base/{('VERIFY_'+str(index).zfill(4)+'.json')!r}
with proof.open('x') as f:json.dump(out,f,sort_keys=True)
print(json.dumps(out))''')
        opened=open_files();deleted=[]
        for e in xs:
            if delete_verified(e,verified[e['relative']],opened):deleted.append(e['relative']);moved+=e['bytes']
        write(CONTROL/f'ARCHIVE_MOVE_batch{index:04}.json',dict(unix=time.time(),manifest_sha256=manifest_sha,
            verified=checks,deleted_sources=deleted,destination=DEST,source_free=shutil.disk_usage(HOME).free,new_sim_s=0))
    index=0
    for entry in m['entries']:
        if shutil.disk_usage(HOME).free>=TARGET:break
        batch.append(entry);size+=entry['bytes']
        if size>=128*1024**2:
            transfer(batch,index);index+=1;batch=[];size=0
    if batch and shutil.disk_usage(HOME).free<TARGET:transfer(batch,index)
    result=dict(unix=time.time(),manifest_sha256=manifest_sha,moved_bytes=moved,
        source_free=shutil.disk_usage(HOME).free,target_reached=shutil.disk_usage(HOME).free>=TARGET,
        destination=DEST,new_sim_s=0,all_deleted_files_remote_and_source_SHA_verified=True)
    write(CONTROL/'COMPLETE.json',result);return result


if __name__=='__main__':
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    os.nice(max(0,19-os.getpriority(os.PRIO_PROCESS,0)))
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','move'));p.add_argument('--manifest-sha')
    a=p.parse_args()
    try:print(json.dumps(prepare() if a.mode=='prepare' else move(a.manifest_sha)))
    except BaseException as e:
        if CONTROL.is_dir():write(CONTROL/(a.mode.upper()+'_FAILED.json'),dict(unix=time.time(),error=repr(e),new_sim_s=0))
        raise
