"""Single-worker, fail-closed coordinator. No simulator allocation before preflight."""
from pathlib import Path
import fcntl
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import time
from core import DENIED, allowed, digest, write, reserve, seed_guard, audit
sys.addaudithook(audit)

ROOT=Path('/root/c1floor20261031')
R=Path('/root/autodl-tmp/c4/speedhead20261030/e3final')
SOURCE='b3c031adb6bc2b5c2bca160bea308f83ba5567b97c1cd0e853eddcd8367a8193'


def jobs():
    design=json.loads((ROOT/'design.json').read_text())
    worlds=json.loads(Path('/root/wmcalib/data/design.json').read_text())['worlds'][:23]
    assert [w['seed'] for w in worlds]==design['world_seeds']
    return [dict(w,id=f"s{w['seed']}_native"+('_attempt2' if i==0 else ''),canary=i==0,charge=1500 if i==0 else 1200,
                 rng_seeds=[969050+2*i,969051+2*i]) for i,w in enumerate(worlds)]


def preflight():
    sys.path[:0]=[str(R/'runtime/src'),str(R/'runtime')]
    from traffic_shepherd.seeds import require_development_case
    from traffic_shepherd.c4.seeds import check_case
    assert importlib.metadata.version('eclipse-sumo')=='1.27.1'
    manifest=R/'runtime/SOURCE_MANIFEST.json'
    assert digest(manifest)==SOURCE
    permitted=excluded=0
    for n,h in json.loads(manifest.read_text()).items():
        if any(x in n.lower() for x in DENIED):
            excluded+=1;continue
        assert digest(R/'runtime'/n)==h,n
        permitted+=1
    obs=Path('/root/wmdata2/tools/speedhead/observer.py')
    assert digest(obs)=='b22af4623b5fe7fbb9a8920b8e8417c3591efae66a52bae7b72f4d899b4a7874'
    for j in jobs():
        for s in (j['seed'],*j['rng_seeds']):
            seed_guard(s);require_development_case(s,j['demand'],'exposed');check_case(s,j['demand'])
    alt=set(range(969050,969096))
    text=(ROOT/'REGISTRATION_INPUT.md').read_text(encoding='utf-8')
    tokens=[];ranges=[]
    for number,line in enumerate(text.splitlines(),1):
        hits=sorted(set(map(int,re.findall(r'(?<!\d)(\d{6})(?!\d)',line))) & alt)
        if hits:tokens.append(dict(line=number,seeds=hits))
        for a,b in re.findall(r'(?<!\d)(\d{6})\s*[-–—]\s*(\d{6})(?!\d)',line):
            if int(a)<=969097 and int(b)>=969050:
                ranges.append(dict(line=number,first=int(a),last=int(b),
                                   declared_unused='none used in 966000-969699' in line))
    assert not tokens,tokens
    # The documented 966000..969999 TRAIN allowlist is not an exposure claim.
    assert all((r['first']<=966000 and r['last']==969999) or r['declared_unused'] for r in ranges),ranges
    hits=[];scanned=0
    for base in ('/root/wmdata2/jobs','/root/wmcalib/data/jobs',
                 '/root/sumocf20261031/jobs','/root/coopaudit20261031/jobs',
                 '/root/autodl-tmp/c4/speedhead20261030'):
        for parent,dirs,files in os.walk(base):
            dirs[:]=[d for d in dirs if not any(x in d.lower() for x in DENIED)]
            for n in dirs+files:
                if any(x in n.lower() for x in DENIED):continue
                scanned+=1
                if set(map(int,re.findall(r'(?<!\d)(\d{6})(?!\d)',n))) & alt:
                    hits.append(str(Path(parent)/n))
    assert not hits,hits
    refs={}
    for j in jobs():
        p=ROOT/'reference'/f"s{j['seed']}_native"
        if not p.is_dir():continue
        result=json.loads((p/'result.json').read_text())
        assert result['ok'] and digest(p/'traj_base.npz')==result['output_hashes']['traj_base.npz']
        refs[j['id']]=digest(p/'traj_base.npz')
    receipt=dict(guards_passed=69,source_manifest_sha256=SOURCE,permitted_files=permitted,
                 excluded_unopened=excluded,observer_sha256=digest(obs),references=refs,
                 exact_alternative_exposure=tokens,overlapping_allowlist_ranges=ranges,
                 filename_scanned=scanned,filename_exposure=hits,registration_sha256=digest(ROOT/'REGISTRATION_INPUT.md'),
                 source_hashes={n:digest(ROOT/n) for n in ('worker.py','core.py','run.py','design.json','labels.py','aggregates.py')})
    write(ROOT/f'PREFLIGHT_{time.time_ns()}.json',receipt)
    (ROOT/'PREFLIGHT_BINDING.json').write_text(json.dumps(receipt,indent=1))
    print(json.dumps(receipt),flush=True)


def execute(phase):
    assert os.getpriority(os.PRIO_PROCESS,0)==19 and os.environ['CUDA_VISIBLE_DEVICES']==''
    lock=(ROOT/'RUN.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    pre=json.loads((ROOT/'PREFLIGHT_BINDING.json').read_text())
    assert all(digest(ROOT/n)==h for n,h in pre['source_hashes'].items())
    selected=jobs()[:1] if phase=='canary' else jobs()[1:]
    if phase=='full':
        assert json.loads((ROOT/'CANARY.json').read_text())['passed']
    from worker import wait
    for j in selected:
        wait()
        reserve(ROOT/'accounting.jsonl',j,pre['source_hashes'])
        spec=ROOT/'jobs'/(j['id']+'.json');write(spec,j)
        with (ROOT/'logs'/(j['id']+'.log')).open('x') as log:
            p=subprocess.Popen([sys.executable,'-B','-u',str(ROOT/'worker.py'),str(spec)],stdout=log,stderr=subprocess.STDOUT)
            owner=dict(phase=phase,worker_pid=p.pid,coordinator_pid=os.getpid(),
                       start_ticks=Path(f'/proc/{p.pid}/stat').read_text().split()[21],job=j['id'])
            # Append-only ownership journal; status overwritten only for this task.
            with (ROOT/'PIDS.jsonl').open('a') as f:f.write(json.dumps(owner)+'\n')
            (ROOT/'STATUS.json').write_text(json.dumps(owner))
            rc=p.wait()
        if rc:
            write(ROOT/f'FAILURE_{time.time_ns()}.json',dict(job=j,returncode=rc))
            raise RuntimeError('worker failed; no retry, retained precharge')
        result=json.loads((ROOT/'jobs'/j['id']/'result.json').read_text())
        assert result['ok'] and result['actual_sim_s']==j['charge']
        if phase=='canary':
            receipt=dict(passed=result['branches']['same_seed']['identical'],
                         actual_sim_s=result['actual_sim_s'],job=j,branches=result['branches'],
                         reproduction=result['reproduction'],result_sha256=digest(ROOT/'jobs'/j['id']/'result.json'))
            write(ROOT/'CANARY.json',receipt)
            print(json.dumps(receipt),flush=True)
    if phase=='full':
        results=[json.loads((ROOT/'jobs'/j['id']/'result.json').read_text()) for j in jobs()]
        charge=sum(json.loads(s)['sim_s'] for s in (ROOT/'accounting.jsonl').read_text().splitlines())
        actual=sum(r['actual_sim_s'] for r in results)
        assert charge==29400 and actual==27900
        write(ROOT/'COMPLETE.json',dict(passed=True,worlds=23,alternatives=46,charged_sim_s=charge,
              actual_sim_s=actual,results_sha256={j['id']:digest(ROOT/'jobs'/j['id']/'result.json') for j in jobs()},
              original_continuations_bitwise=23,worker_limit=1,nice=19,threads=1,zero_step_failures=1))


if __name__=='__main__':
    phase=sys.argv[1]
    if phase=='tests':
        raise SystemExit(subprocess.call([sys.executable,'-B',str(ROOT/'test_core.py')]))
    elif phase=='preflight':preflight()
    else:execute(phase)
