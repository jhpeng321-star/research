"""MAIN07:12 disjoint V2V extension of the qualified SHA-gated rolling mover.

Keeps the original mover/launch3 liveness anchor running. This companion owns
only older completed V2V episode RESULT payloads/observable tapes, never receipts,
checkpoints, bank inputs, live/current-update outputs or pending-consumer inputs.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time

OUTPUTS = ('RESULT.json', 'RESULT.json.gz', 'observable.jsonl.gz')
BAD = re.compile(r'sealed|holdout|heldout|cal_split|chunk5|j4b|inc1|hetdata_truth|vehicle_params|driver_params|vtype',re.I)


def under(path, parent):
    try:Path(path).relative_to(Path(parent));return True
    except ValueError:return False


def job_info(root, path):
    root, path = Path(root), Path(path)
    if BAD.search(str(path)) or '..' in path.parts or path.is_symlink():return None
    if path.parent.parent != root/'jobs' or path.name not in OUTPUTS:return None
    match = re.fullmatch(r'v2v1_warm_u([0-9]{3})_([ABR])_(986[0-9]{3})',path.parent.name)
    if not match or not 986000 <= int(match[3]) <= 986071:return None
    return int(match[1]), match[2]


def completed_updates(root, now, min_age=1800):
    root = Path(root);latest={};complete={}
    for arm in ('A','B','R'):
        found={}
        for p in root.glob(arm+'_UPDATE_*.json'):
            if not p.stem.rsplit('_',1)[-1].isdigit():continue
            try:value=json.loads(p.read_text())
            except json.JSONDecodeError:continue  # immutable receipt still publishing
            u=int(p.stem.rsplit('_',1)[1])
            if value.get('arm')!=arm or value.get('update')!=u:
                raise ValueError('completed V2V update identity differs')
            found[u]=p
        latest[arm]=max(found,default=0);complete[arm]=found
    current=min(latest.values())
    eligible={u for u in set.intersection(*(set(complete[a]) for a in ('A','B','R')))
        if u <= current-1 and all(now-complete[a][u].stat().st_mtime >= min_age for a in ('A','B','R'))}
    return latest,eligible


def pending_refs(root):
    """Public consumer metadata only; no outcome/evaluation/scenario traversal."""
    root=Path(root);refs=set()
    def visit(value):
        if isinstance(value,str) and under(value,root/'jobs') and not BAD.search(value):refs.add(value)
        elif isinstance(value,dict):
            for v in value.values():visit(v)
        elif isinstance(value,list):
            for v in value:visit(v)
    paths=[p for p in root.glob('*.json') if any(s in p.name.upper() for s in ('BINDING','CONFIG','MANIFEST','RUNTIME'))]
    for name in ('requests','centre_requests','snapshots','lessons','isolated','replay_bank'):
        folder=root/name
        if folder.is_symlink():continue  # the shared bank is handled by base.protected
        for parent,dirs,names in os.walk(folder,followlinks=False):
            dirs[:]=[n for n in dirs if not BAD.search(str(Path(parent)/n)) and not (Path(parent)/n).is_symlink()]
            paths.extend(Path(parent)/n for n in names if n.endswith('.json'))
    for p in paths:
        if not p.is_symlink() and not BAD.search(str(p)):visit(json.loads(p.read_text()))
    return refs


def select(root, opened, files, trees, refs, now, min_age=1800):
    root=Path(root);latest,eligible=completed_updates(root,now,min_age);found={}
    for job in (root/'jobs').iterdir():
        if not job.is_dir() or job.is_symlink():continue
        close=job/'CLOSE.json'
        if not close.is_file() or now-close.stat().st_mtime<min_age:continue
        for name in OUTPUTS:
            p=job/name;info=job_info(root,p);key=str(p)
            if info is None or info[0] not in eligible or not p.is_file():continue
            if p.resolve()!=p or p.stat().st_dev!=root.stat().st_dev or now-p.stat().st_mtime<min_age:continue
            if key in opened or key in files or any(under(key,t) for t in trees):continue
            if any(under(key,q) for q in refs):continue
            found[key]='completed_older_v2v_episode_output'
    return found,latest


def run(control):
    import fcntl
    import gzip
    import signal
    import subprocess
    import sys
    c=Path(control);binding=json.loads((c/'AUTHORITY.json').read_text())
    H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if H(__file__)!=binding['worker_sha256']:raise ValueError('archive extension source changed')
    root=Path(binding['v2v_root']);old=Path(binding['regular_mover_control'])
    sys.path.insert(0,binding['qualified_archive_source'])
    import rolling_archive as m
    for name in ('rolling_archive.py','archive_move.py'):
        if m.base.digest_file(Path(binding['qualified_archive_source'])/name)!=binding['source_hashes'][name]:
            raise ValueError('qualified archive helper changed')
    lock=(c/'RUN.claim').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    m.CONTROL=c;m.configure_target(binding)
    def live_ssh():
        transport=json.loads((old/'TRANSPORT.json').read_text());socket=transport['socket']
        if not Path(socket).exists():raise RuntimeError('forwarded agent unavailable; sources retained')
        return list(binding['archive_target']['ssh_command']),dict(os.environ,SSH_AUTH_SOCK=socket)
    m.ssh=live_ssh
    original_kind=m.job_kind
    m.job_kind=lambda p:job_info(root,p) or original_kind(p)
    def candidates():
        files,trees,_=m.base.protected(m.base.final_labels())
        refs=pending_refs(root)|m.metadata_refs()
        paths,_=select(root,m.base.open_files(),files,trees,refs,time.time())
        return paths,dict(scope='V2V outputs only; original mover stays live')
    m.candidates=candidates
    def put(name,value):m.base.write(c/name,value)
    os.nice(max(0,19-os.getpriority(os.PRIO_PROCESS,0)))
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    put('OWNER.json',m.base.owner())
    total=0;index=0;transport_failures=0
    summary=root/'ARCHIVED_COMPLETED_EPISODE_SUMMARIES';summary.mkdir(exist_ok=True)
    while not (c/'STOP.json').exists():
        try:
            gpu_free=m.destination_free()
            if gpu_free<m.FLOOR:
                m.atomic(c/'STOP_GPU_SPACE.json',dict(unix=time.time(),free=gpu_free,floor=m.FLOOR));return
            paths,_=candidates();latest,_=completed_updates(root,time.time())
            m.atomic(c/'STATUS.json',dict(unix=time.time(),cpu_free=shutil.disk_usage(root).free,
                gpu_free=gpu_free,eligible_files=len(paths),eligible_bytes=sum(Path(p).stat().st_size for p in paths),
                latest=latest,moved_bytes=total,target_bytes=9*1024**3,new_sumo_sim_s=0))
            if paths:
                # Keep each job's marker/payload together, and retain its metrics
                # before any source removal. Exact originals remain in the archive.
                jobs={}
                for name,category in paths.items():jobs.setdefault(Path(name).parent,[]).append((name,category))
                items=[];size=0
                for job,entries in sorted(jobs.items()):
                    marker=job/'RESULT.json'
                    if not marker.is_file():raise ValueError('V2V output marker missing before archive')
                    value=json.loads(marker.read_text())
                    if value.get('encoding')=='coopmwm-lossless-gzip-v1':
                        if value['file']!='RESULT.json.gz':raise ValueError('unexpected V2V output payload path')
                        payload=gzip.decompress((job/value['file']).read_bytes())
                        if hashlib.sha256(payload).hexdigest()!=value['sha256']:raise ValueError('V2V output payload digest changed')
                        value=json.loads(payload)
                    if value['id']!=job.name:raise ValueError('V2V output identity differs')
                    target=summary/(job.name+'.json')
                    if not target.exists():
                        m.base.write(target,dict(episode={k:value[k] for k in ('id','world','arm','metrics')},
                            original_files={Path(n).name:H(n) for n,_ in entries},new_sumo_sim_s=0))
                    items.extend(entries);size+=sum(Path(n).stat().st_size for n,_ in entries)
                    if size>=128*1024**2:break
                # Unchanged qualified per-file SHA manifest -> rsync -> remote
                # sha256sum -> fresh FD/reference/SHA checks -> exact source unlink.
                total+=m.batch(items,index);index+=1
                transport_failures=0
                m.atomic(c/'PROGRESS.json',dict(unix=time.time(),moved_bytes=total,batches=index,
                    cpu_free=shutil.disk_usage(root).free,new_sumo_sim_s=0))
                continue
        except BaseException as error:
            import traceback
            m.atomic(c/'ALERT.json',dict(unix=time.time(),error=repr(error),traceback=traceback.format_exc(),
                sources_retained=True,new_sumo_sim_s=0))
            # A failed namespace is never re-used. No physics or inference retry.
            index+=1
            connection=isinstance(error,subprocess.TimeoutExpired) or any(s in str(error) for s in
                ('Connection closed','Connection reset','timed out','Temporary failure in name resolution',
                 'code 255','forwarded agent unavailable','Connection timed out'))
            transport_failures+=1
            if not connection or transport_failures>5:raise
            time.sleep((10,20,40,80,120)[transport_failures-1])
        time.sleep(60)


if __name__=='__main__':
    import sys
    run(sys.argv[1])
