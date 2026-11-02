"""MAIN18:38 collection handover. Every operation adopts durable receipts first."""
import argparse
import inspect
import json
import time
from pathlib import Path
from . import ops
from .storage import read, write
from .protocol import REGISTRY, checked, sha

ROOT = '/home/ubuntu/coopmwm20261102_warm1'
SOURCE = '/home/ubuntu/coopmwm20261102_lsreleasecpu2'
COLLECTION = '/home/ubuntu/coopmwm20261102_lscollect1'


def pause_remote(root, amendment):
    import os, signal, hashlib, sys
    from pathlib import Path
    sys.path.insert(0, '/home/ubuntu/coopmwm20261102_verifybeta13/pytools')
    from tools.coopmwm.warm_spool import immutable
    root = Path(root)
    done = root/'LS_COLLECT_PAUSED.json'
    if done.exists():
        return json.loads(done.read_text())
    assert not (root/'B_checkpoint_004.pt').exists()
    assert not (root/'R_checkpoint_004.pt').exists()
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    def identity(pid):
        p = Path('/proc', str(pid))
        try:
            st = (p/'stat').read_text().split()
            return dict(pid=int(pid), start_ticks=st[21], state=st[2],
                        argv=(p/'cmdline').read_bytes().replace(b'\0', b' ').decode(),
                        children=[int(x) for x in (p/'task'/str(pid)/'children').read_text().split()])
        except FileNotFoundError:
            return None
    def send(d, sig):
        cur = identity(d['pid'])
        if cur is None or cur['state'] == 'Z':
            return
        assert cur['start_ticks'] == d['start_ticks']
        assert cur['argv'] == d['argv'], 'PID identity changed'
        os.kill(d['pid'], sig)
    checkpoints = {a: hashlib.sha256((root/f'{a}_checkpoint_003.pt').read_bytes()).hexdigest() for a in ('B','R')}
    owners = []
    for arm in ('B','R'):
        d = json.loads((root/f'DNS_{arm}_DISPATCH_1651.json').read_text())
        assert d['boot_id'] == boot
        cur = identity(d['pid'])
        if cur and cur['state'] != 'Z':
            assert cur['start_ticks'] == d['start_ticks']
            assert cur['argv'].split() == ['/home/ubuntu/miniconda3/bin/python', '-B', '-u', '-m',
                'tools.coopmwm.incident_recovery', str(root), '--component', arm]
            owners.append(cur)
    # Stop parents first so no optimizer/update or additional pool task can start.
    for d in owners:
        send(d, signal.SIGSTOP)
    tree = []
    def descend(d):
        for pid in d['children']:
            child = identity(pid)
            if child is None or child['state'] == 'Z':
                continue
            assert ('multiprocessing' in child['argv'] or
                    ('sumo ' in child['argv'] and str(root/'jobs/warmdns1651_u004_B_') in child['argv']))
            send(child, signal.SIGSTOP)
            tree.append(child)
            descend(child)
    for d in owners:
        descend(d)
    plan = dict(unix=time.time(), boot_id=boot, owners=owners, descendants=tree,
                checkpoints=checkpoints, amendment_sha256=amendment, new_sim_s=0)
    immutable(root/'LS_COLLECT_PAUSE_SIGNAL_PLAN.json', plan)
    # Only this receipt-verified tree is terminated. No GPU/broker/grader signal.
    for d in reversed(tree):
        send(d, signal.SIGTERM)
        send(d, signal.SIGCONT)
    for d in owners:
        send(d, signal.SIGTERM)
        send(d, signal.SIGCONT)
    deadline = time.monotonic()+10
    while time.monotonic() < deadline and any(identity(d['pid']) and identity(d['pid'])['state'] != 'Z' for d in tree+owners):
        time.sleep(.1)
    for d in tree+owners:
        send(d, signal.SIGKILL)
    jobs = []
    for p in sorted((root/'jobs').glob('warmdns1651_u004_B_*')):
        assert not (p/'RESULT.json').exists(), 'inspect newly completed work before closing'
        if (p/'CLAIM.json').exists():
            immutable(p/'LS_COLLECT_CLOSED.json', dict(reason='MAIN18:38 stalled wave closed; never replay',
                unix=time.time(), amendment_sha256=amendment, charges_retained=True))
            if not (p/'CLOSE.json').exists():
                immutable(p/'CLOSE.json', dict(reason='MAIN18:38 collection pause', actual_sim_s=None,
                    charges_retained=True, unix=time.time()))
            jobs.append(p.name)
    assert checkpoints == {a: hashlib.sha256((root/f'{a}_checkpoint_003.pt').read_bytes()).hexdigest() for a in ('B','R')}
    assert not (root/'B_checkpoint_004.pt').exists() and not (root/'R_checkpoint_004.pt').exists()
    result = dict(plan, closed_jobs=jobs, no_u4_update=True, gpu_work_preserved=True,
                  resume_requires_MAIN=True, completed_unix=time.time())
    immutable(done, result)
    return result


def pause_latest_remote(root, amendment):
    """Close the superseded FAST1844 wave, never the original pause receipt."""
    import os, signal, hashlib, sys
    from pathlib import Path
    sys.path.insert(0, '/home/ubuntu/coopmwm20261102_fast1844cpu1/pytools')
    from tools.coopmwm.warm_spool import immutable
    root = Path(root)
    done = root/'LS_COLLECT_PAUSED_1912.json'
    if done.exists():
        return json.loads(done.read_text())
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    def identity(pid):
        p = Path('/proc', str(pid))
        try:
            st = (p/'stat').read_text().split()
            return dict(pid=int(pid), start_ticks=st[21], state=st[2],
                        argv=(p/'cmdline').read_bytes().replace(b'\0', b' ').decode(),
                        children=[int(x) for x in (p/'task'/str(pid)/'children').read_text().split()])
        except FileNotFoundError:
            return None
    def send(d, sig):
        cur = identity(d['pid'])
        if cur is None or cur['state'] == 'Z':
            return
        assert cur['start_ticks'] == d['start_ticks'] and cur['argv'] == d['argv'], 'PID identity changed'
        os.kill(d['pid'], sig)
    for a in ('B', 'R'):
        assert not (root/f'{a}_checkpoint_004.pt').exists(), 'u4 exists: do not overwrite'
    checkpoints = {a: hashlib.sha256((root/f'{a}_checkpoint_003.pt').read_bytes()).hexdigest() for a in ('B','R')}
    expected = json.loads((root/'LS_COLLECT_PAUSED.json').read_text())['checkpoints']
    assert checkpoints == expected
    owners = []
    for a in ('B','R'):
        d = json.loads((root/f'FAST_{a}_DISPATCH_1844.json').read_text())
        assert d['boot_id'] == boot
        cur = identity(d['pid'])
        if cur and cur['state'] != 'Z':
            assert cur['start_ticks'] == d['start_ticks']
            assert cur['argv'].split() == ['/home/ubuntu/miniconda3/bin/python','-B','-u','-m',
                'tools.coopmwm.incident_recovery',str(root),'--component',a]
            owners.append(cur)
    intent=root/'LS_COLLECT_PAUSE_1912_INTENT.json'
    if intent.exists():
        previous=json.loads(intent.read_text())
        assert previous['checkpoints']==checkpoints and previous['amendment_sha256']==amendment
        assert [(d['pid'],d['start_ticks'],d['argv']) for d in previous['owners']]==[(d['pid'],d['start_ticks'],d['argv']) for d in owners]
    else:
        immutable(intent, dict(unix=time.time(), owners=owners,
            boot_id=boot, checkpoints=checkpoints, amendment_sha256=amendment))
    for d in owners:
        send(d, signal.SIGSTOP)
    tree = []
    def descend(d):
        for pid in identity(d['pid'])['children']:
            child = identity(pid)
            if child is None or child['state'] == 'Z':
                continue
            sumo_owned=('sumo ' in child['argv'] and any(prefix in child['argv'] for prefix in (
                str(root/'jobs')+'/', '/dev/shm/coopmwm20261102_warm1/warm_u004_R_',
                '/dev/shm/coopmwm20261102_warm1/warmfast1844_u004_B_')))
            assert 'multiprocessing' in child['argv'] or sumo_owned, child
            send(child, signal.SIGSTOP)
            tree.append(child)
            descend(child)
    for d in owners:
        descend(d)
    immutable(root/'LS_COLLECT_PAUSE_1912_SIGNAL_PLAN.json',dict(unix=time.time(),owners=owners,descendants=tree,
        amendment_sha256=amendment,boot_id=boot))
    for d in list(reversed(tree))+owners:
        send(d, signal.SIGTERM)
        send(d, signal.SIGCONT)
    deadline = time.monotonic()+10
    while time.monotonic()<deadline and any(identity(d['pid']) and identity(d['pid'])['state']!='Z' for d in tree+owners):
        time.sleep(.1)
    for d in tree+owners:
        send(d, signal.SIGKILL)
    jobs=[]
    for pattern in ('warmfast1844_u004_B_*','warm_u004_R_*'):
        for p in sorted((root/'jobs').glob(pattern)):
            if not (p/'CLAIM.json').exists():
                continue
            result_exists=(p/'RESULT.json').exists() or (p/'RESULT.json.gz').exists()
            immutable(p/'LS_COLLECT_CLOSED_1912.json',dict(unix=time.time(),charges_retained=True,
                result_preserved=result_exists,never_replay=True,amendment_sha256=amendment))
            if not (p/'CLOSE.json').exists():
                immutable(p/'CLOSE.json',dict(unix=time.time(),actual_sim_s=None,charges_retained=True,
                    reason='MAIN19:12 forbidden plain-search u4 wave closed; never replay'))
            jobs.append(dict(job=p.name,result_preserved=result_exists))
    assert checkpoints == {a:hashlib.sha256((root/f'{a}_checkpoint_003.pt').read_bytes()).hexdigest() for a in ('B','R')}
    assert not (root/'B_checkpoint_004.pt').exists() and not (root/'R_checkpoint_004.pt').exists()
    v=dict(unix=time.time(),owners=owners,descendants=tree,closed_jobs=jobs,checkpoints=checkpoints,
           no_u4_update=True,charges_retained=True,new_sim_s=0,resume_requires_MAIN=True,
           gpu_work_preserved=True,amendment_sha256=amendment)
    immutable(done,v)
    return v


def pause_latest():
    target=ops.EVIDENCE/'LS_COLLECT_PAUSE_REGISTRATION_1912.json'
    if not target.exists():
        line=('\nS-COOPMWM MAIN19:12 LATEST-ORDER PAUSE: forbidden FAST1844 plain-search u4 '
              'B/R wave closed by receipt/boot/start/argv-verified owned PIDs only. No u4 update; '
              'durable B3/R3 unchanged; all attempted/completed jobs, charges, responses and pools retained, '
              'closed jobs never replayed. LS-COLLECT only, resume requires MAIN addendum. '
              'Latest cap11900000; restricted interim_u8_test900000 not available for collection. No new SUMO.\n')
        write(target,dict(line=line,authority='MAIN19:12',new_sim_s=0,unix=time.time()))
        with checked(REGISTRY).open('a',encoding='utf-8') as f:
            f.write(line+'amendmentSHA'+sha(target)+'\n')
    script=inspect.getsource(pause_latest_remote)
    out=json.loads(ops.ssh.remote('cpuQ','import json,time\n'+script+
        f'\nprint(json.dumps(pause_latest_remote({ROOT!r},{sha(target)!r})))',timeout=45))
    dest=ops.EVIDENCE/'LS_COLLECT_PAUSED_1912.json'
    if not dest.exists():write(dest,out)
    print(json.dumps(dict(paused=True,no_u4_update=out['no_u4_update'],checkpoints=out['checkpoints'],
        closed_jobs=out['closed_jobs'],new_sim_s=0)))


def audit():
    out=ops.ssh.remote('cpuQ',f'''import sys,json,pathlib,time
sys.path.insert(0,'/home/ubuntu/coopmwm20261102_fast1844cpu1/pytools')
from tools.coopmwm.accel_storage import ledger,charged
r=pathlib.Path({ROOT!r});rows=ledger(r);p=json.loads((r/'LS_COLLECT_PAUSED_1912.json').read_text());ids={{v['job'] for v in p['closed_jobs']}};ph={{}}
for c in rows:ph[c['phase']]=ph.get(c['phase'],0)+c['sim_s']
owners=[]
for d in p['owners']+p['descendants']:
 q=pathlib.Path('/proc',str(d['pid']))
 if q.exists():
  s=(q/'stat').read_text().split();owners.append(dict(pid=d['pid'],state=s[2],same_identity=s[21]==d['start_ticks']))
print(json.dumps(dict(unix=time.time(),charged_sim_s=charged(r),phase_charges=ph,closed_wave_charge=sum(c['sim_s'] for c in rows if c['id'] in ids),closed_jobs=len(ids),remaining_processes=owners,u4_absent=all(not (r/(a+'_checkpoint_004.pt')).exists() for a in ('B','R')))))''',timeout=40)
    v=json.loads(out);write(ops.EVIDENCE/f'LS_COLLECT_PAUSE_AUDIT_{int(v["unix"])}.json',v);print(out)


def qualify():
    """Stage a compact overlay on a verified remote snapshot, never ship it twice."""
    import base64,zlib
    base=read(ops.EVIDENCE/'CPU_coopmwm20261102_fast1844cpu1.json')
    names=('parallel_groups','capacity_broker','test_capacity_broker_cpu','test_parallel_groups_cpu',
           'fcntl_compat','conftest','ls_cpu_test','ls_collection','ls_gpu','accel_plant','test_ls_collection_cpu',
           'mppi_runtime','plans','test_incident_cpu','ls_broker')
    texts={n:checked(ops.LOCAL/'tools/coopmwm'/(n+'.py')).read_text(encoding='utf-8') for n in names}
    packed=base64.b64encode(zlib.compress(json.dumps(texts).encode())).decode()
    out=ops.ssh.remote('cpuQ',f'''import pathlib,json,hashlib,shutil,zlib,base64,os,time,sys
r=pathlib.Path({SOURCE!r});base=pathlib.Path({base['root']!r});assert r.parent.is_dir()
if not r.exists():shutil.copytree(base,r)
texts=json.loads(zlib.decompress(base64.b64decode({packed!r})));hashes=json.loads({json.dumps(base['source_hashes'])!r})
for name,h in hashes.items():assert hashlib.sha256((base/name).read_bytes()).hexdigest()==h,name
for name,text in texts.items():
 p=r/'pytools/tools/coopmwm'/(name+'.py')
 marker=r/('LS_OVERLAY_'+name+'.json');h=hashlib.sha256(text.encode()).hexdigest()
 if marker.exists():assert json.loads(marker.read_text())['sha256']==h and p.read_text()==text
 else:
  p.write_text(text)
  with marker.open('x') as f:json.dump(dict(sha256=h),f)
# Give pytest the same canonical tools.coopmwm package identity as production;
# otherwise absolute patch targets and relative test imports are two modules.
(r/'pytools/tools/__init__.py').touch(exist_ok=True)
for p in (r/'pytools').rglob('*.py'):hashes[str(p.relative_to(r))]=hashlib.sha256(p.read_bytes()).hexdigest()
with (r/'LS_SOURCE_HASHES.json').open('w') as f:json.dump(hashes,f)
with (r/'LS_RELEASE_TESTS.json').open('w') as f:json.dump(['test_parallel_groups_cpu','test_capacity_broker_cpu','test_fast_resume_cpu','test_ls_collection_cpu','test_incident_cpu','test_mppi_search_cpu','test_branch_eval_cpu','test_awr_cpu','test_exact_grade_cpu','test_sumo_verify_cpu','test_verify_continuation_cpu','test_committed_cpu','test_overlap_cpu','test_warm_health_cpu','test_warm_transport_cpu','test_resource_config_cpu','test_warm_broker_cpu','test_warm_spool_cpu'],f)
print(json.dumps(dict(root=str(r),source_hashes=hashes)))''',timeout=45)
    staged=json.loads(out)
    out=ops.ssh.remote('cpuQ',f'''import pathlib,sys,os,json,time,subprocess
r=pathlib.Path({SOURCE!r});done=r/'LS_CPU_RESULT.json';dispatch=r/'LS_CPU_DISPATCH.json'
if done.exists():print(done.read_text())
elif dispatch.exists():print(dispatch.read_text())
else:
 env=dict(os.environ,PYTHONPATH=str(r/'pytools')+':/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime')
 with (r/'LS_CPU.log').open('x') as log:
  p=subprocess.Popen(['/home/ubuntu/miniconda3/bin/python','-B','-u','-m','tools.coopmwm.ls_cpu_test',str(r)],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 d=dict(pid=p.pid,start_ticks=pathlib.Path('/proc',str(p.pid),'stat').read_text().split()[21],boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),unix=time.time())
 with dispatch.open('x') as f:json.dump(d,f)
 print(json.dumps(d))''',timeout=30)
    print(out)


def cpu_result():
    out=ops.ssh.remote('cpuQ',f'''import pathlib,json
r=pathlib.Path({SOURCE!r});p=r/'LS_CPU_RESULT.json'
if p.exists():print(p.read_text())
else:print(json.dumps(dict(pending=True,log=(r/'LS_CPU.log').read_text()[-12000:])))''',timeout=30)
    v=json.loads(out)
    if not v.get('pending'):
        dest=ops.EVIDENCE/('LS_CPU_'+SOURCE.rsplit('/',1)[1]+'.json')
        if not dest.exists():write(dest,v)
        print(json.dumps({k:v[k] for k in ('pass_','tests','skips','sim_s','wall_s','root')}))
        if not v['pass_']:print(ops.ssh.remote('cpuQ',f'from pathlib import Path\nprint(Path({SOURCE!r},"LS_CPU.log").read_text()[-20000:])',timeout=30))
    else:print(out)


def register_collection():
    cpu=read(ops.EVIDENCE/('LS_CPU_'+SOURCE.rsplit('/',1)[1]+'.json'))
    assert cpu['pass_'] and cpu['skips']==cpu['sim_s']==0
    dest=ops.EVIDENCE/'LS_COLLECT_REGISTRATION.json'
    if dest.exists():raise ValueError('adopt registration and remote binding; never register twice')
    out=ops.ssh.remote('cpuQ',f'''import sys,pathlib,json,time,hashlib
sys.path.insert(0,{SOURCE!r}+'/pytools')
from tools.coopmwm.accel_storage import ledger,charged
from tools.coopmwm.accel_protocol import worlds
r=pathlib.Path({ROOT!r});b=json.loads((r/'FAST_RESUME_1844.json').read_text());paused=json.loads((r/'LS_COLLECT_PAUSED_1912.json').read_text())
op=json.loads((r/'EXACT_OPERATIONAL_BINDING.json').read_text());op.update(json.loads((r/'WARM_B_RUNTIME_BINDING_V4.json').read_text())['operational'])
for a in ('B','R'):
 assert not (r/(a+'_checkpoint_004.pt')).exists()
 assert hashlib.sha256((r/(a+'_checkpoint_003.pt')).read_bytes()).hexdigest()==paused['checkpoints'][a]
ph={{}}
for c in ledger(r):ph[c['phase']]=ph.get(c['phase'],0)+c['sim_s']
print(json.dumps(dict(unix=time.time(),budget=b['budget'],worlds=worlds('train'),runtime=op['runtime'],checkpoint=op['checkpoint'],accel_bounds=op['accel_bounds'],checkpoints=paused['checkpoints'],phase_charges=ph,charged_sim_s=charged(r))))''',timeout=30)
    v=json.loads(out);b=v['budget'];assert b['total']==11000000
    b['total']=11900000;b['allocated']=11900000;b['phases']['interim_u8_test']=900000
    assert sum(b['phases'].values())==b['total']
    assert [w['seed'] for w in sorted(v['worlds'],key=lambda w:w['seed'])]==list(range(986000,986072))
    # Capacity quote, not an outcome-based stop: MAIN may stop earlier; after
    # the registered envelope is exhausted the supervisor waits without physics.
    quote=dict(episodes=72,episode_sim_s=86400,max_searches=216,branches_per_search=9,
        verification_upper_sim_s=2332800,total_upper_sim_s=2419200,
        phases=dict(train=86400,sumo_verification=2332800),
        all_failed_attempt_charges_retained=True,restricted_phases_used=False)
    for phase,cost in quote['phases'].items():assert v['phase_charges'].get(phase,0)+cost<=b['phases'][phase]
    config=dict(root=COLLECTION,run_root=ROOT,source_root=SOURCE,budget=b,quote=quote,
        worlds=v['worlds'],runtime=v['runtime'],checkpoint=v['checkpoint'],accel_bounds=v['accel_bounds'],
        checkpoints=v['checkpoints'],cpu=cpu,max_searches=216,training_paused=True,
        search='unchanged plain warm MPPI; N=max(32,2d),3 iterations,K4H300,K8Hfinal; eight existing finalists logged only',
        resume_requires_MAIN=True,stop_clock=None,gpu_deadline=1793894340.,
        continuation_requote_required_before_training_resume=True)
    line=('\nS-COOPMWM MAIN18:58/19:12 LS-COLLECT REGISTERED: B/R PAUSED at durableu3/no u4; '
        'frozen B3 TRAIN986000..986071,72 episodes=86400sim-s train. Every eligible10s window '
        'queues all2..6-member observable components as compressed snapshots; episode NEVER waits/executes a plan. '
        'Up to216 completed plain searches, each original8 finalists+policy exact prefix/CRN branches; '
        'conservative9*1200/list <=2332800sim-s sumo_verification, precharged before search; no refund/double charge. '
        'Total envelope2419200 fits existing train/verification phases; no restricted k8/interim funds. '
        'Main18:50 cap11900000 registered: +900000 exclusively interim_u8_test, unused here. '
        'Four original GPU claims drain once before logging-only profile-capture wrappers; original source/binding/gates kept, '
        'four source-identical gates adopted/four fresh exact gates per process; unsupported-family labels skip, no search change. '
        'harden3+MAIN1854 breadth-first scheduling/withdrawal adopted,201CPU PASS0skips42.334s/zeroSUMO. '
        'Snapshot/label queue spreads worlds by time/groupindex/world; full pools and all9 gold outcomes retained. '
        'Two label lists at once (18branches),24 nonblocking episode workers; remote durable supervision. '
        'No automatic learned activation/training resume/deadline stop; wait MAIN. Envelope exhaustion waits without physics. '
        'All previous jobs/charges retained. Command mirrorFAILED/OFF. Remaining training must be re-quoted before learned resume.\n')
    write(dest,dict(unix=time.time(),authority='MAIN18:58/19:12',line=line,config=config,
        current_charges=v['phase_charges'],charged_sim_s=v['charged_sim_s']))
    h=sha(dest);config['amendment_sha256']=h
    with checked(REGISTRY).open('a',encoding='utf-8') as f:f.write(line+'amendmentSHA'+h+'\n')
    out=ops.ssh.remote('cpuQ',f'''import pathlib,json,sys,hashlib
sys.path.insert(0,{SOURCE!r}+'/pytools')
from tools.coopmwm.warm_spool import immutable
r=pathlib.Path({ROOT!r});c=pathlib.Path({COLLECTION!r});assert c.parent.is_dir();c.mkdir(exist_ok=True)
cfg=json.loads({json.dumps(config)!r})
for n,h in cfg['cpu']['source_hashes'].items():assert hashlib.sha256((pathlib.Path({SOURCE!r})/n).read_bytes()).hexdigest()==h,n
with (r/'AUTHORITY.md').open('a') as f:f.write({(line+'amendmentSHA'+h+chr(10))!r})
immutable(c/'LS_COLLECT_BINDING.json',cfg)
immutable(r/'LS_COLLECT_REGISTRATION_1912.json',dict(root=str(c),amendment_sha256=cfg['amendment_sha256'],quote=cfg['quote'],budget=cfg['budget']))
print(json.dumps(dict(bound=True,root=str(c),amendment_sha256=cfg['amendment_sha256'],quote=cfg['quote'])))''',timeout=30)
    write(ops.EVIDENCE/'LS_COLLECT_BOUND.json',json.loads(out));print(out)


def freeze_broker():
    out=ops.ssh.remote('cpuQ',f'''import pathlib,json,os,signal,time
r=pathlib.Path({ROOT!r});c=pathlib.Path({COLLECTION!r});dest=c/'BROKER_FROZEN.json'
if dest.exists():print(dest.read_text())
else:
 assert (r/'LS_COLLECT_PAUSED_1912.json').exists() and (c/'LS_COLLECT_BINDING.json').exists()
 d=json.loads((r/'FAST_BROKER_DISPATCH_1844.json').read_text());p=pathlib.Path('/proc',str(d['pid']))
 assert d['boot_id']==pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
 assert (p/'stat').read_text().split()[21]==d['start_ticks']
 argv=(p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()
 assert argv.split()==['/home/ubuntu/miniconda3/bin/python','-B','-u','-m','tools.coopmwm.capacity_broker',str(r),str(r/'FAST_BROKER_CONFIG_1844.json')]
 os.kill(d['pid'],signal.SIGSTOP)
 v=dict(unix=time.time(),owner=d,argv=argv,gpu_claims_continue=True)
 with dest.open('x') as f:json.dump(v,f)
 print(json.dumps(v))''',timeout=30)
    v=json.loads(out);p=ops.EVIDENCE/'LS_COLLECT_BROKER_FROZEN.json'
    if not p.exists():write(p,v)
    print(out)


def gpu_collect(host):
    import base64,zlib
    reg=read(ops.EVIDENCE/'LS_COLLECT_BOUND.json')
    assert (ops.EVIDENCE/'LS_COLLECT_BROKER_FROZEN.json').exists()
    script=checked(ops.LOCAL/'tools/coopmwm/ls_gpu.py').read_text(encoding='utf-8')
    payload=base64.b64encode(zlib.compress(script.encode())).decode()
    roots=dict(gpuA=['/root/coopmwm20261102_incident1','/root/coopmwm20261102_deadlineA2'],
        gpuR=['/root/coopmwm20261102_concurrent3','/root/coopmwm20261102_deadlineR2'])[host]
    out=ops.ssh.remote(host,f'''import pathlib,json,os,time,subprocess,base64,zlib,hashlib
base=pathlib.Path('/root/coopmwm20261102_lsops1');assert base.parent.is_dir();base.mkdir(exist_ok=True)
script=zlib.decompress(base64.b64decode({payload!r})).decode();source=base/'ls_gpu.py'
if source.exists():assert source.read_text()==script
else:source.write_text(script)
out=[]
for name in {roots!r}:
 r=pathlib.Path(name);dest=r/'LS_GPU_DISPATCH.json'
 if dest.exists():out.append(json.loads(dest.read_text()));continue
 assert not (r/'LS_GPU_FAILED.json').exists()
 old=json.loads((r/'SPOOL_OWNER.json').read_text());p=pathlib.Path('/proc',str(old['pid']))
 assert old['boot_id']==pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
 assert (p/'stat').read_text().split()[21]==old['start_ticks']
 old['argv']=(p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode()
 assert str(r) in old['argv'] and ('tools.coopmwm.warm_spool' in old['argv'] or 'restart.py' in old['argv'])
 env={{x.split('=',1)[0]:x.split('=',1)[1] for x in (p/'environ').read_bytes().decode().split(chr(0)) if '=' in x}}
 cmd=[env.get('CONDA_PREFIX','/root/miniconda3')+'/bin/python','-B','-u',str(source),str(r),json.dumps(old),'1793894340',{reg['amendment_sha256']!r}]
 with (r/'LS_GPU.log').open('x') as log:
  proc=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 st=pathlib.Path('/proc',str(proc.pid),'stat').read_text().split()
 d=dict(pid=proc.pid,start_ticks=st[21],boot_id=old['boot_id'],unix=time.time(),root=str(r),old=old,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
 with dest.open('x') as f:json.dump(d,f)
 out.append(d)
print(json.dumps(dict(host={host!r},owners=out)))''',timeout=35)
    v=json.loads(out);p=ops.EVIDENCE/('LS_COLLECT_GPU_DISPATCH_'+host+'.json')
    if not p.exists():write(p,v)
    print(out)


def collection_gpu_status(host):
    d=read(ops.EVIDENCE/('LS_COLLECT_GPU_DISPATCH_'+host+'.json'))
    out=ops.ssh.remote(host,f'''import pathlib,json,time,subprocess
out=[]
for d in {d['owners']!r}:
 r=pathlib.Path(d['root']);p=pathlib.Path('/proc',str(d['pid']));st=(p/'stat').read_text().split() if p.exists() else None
 assert not st or st[21]==d['start_ticks']
 v=dict(root=str(r),pid=d['pid'],state=st[2] if st else 'exited',gates={{}})
 for f in r.glob('STARTUP_*_LSCOLLECT.json'):v['gates'][f.name]=json.loads(f.read_text())
 for n in ('LS_GPU_FAILED.json','SERVICE_OWNER.json','SPOOL_OWNER.json'):
  if (r/n).exists():v[n]=json.loads((r/n).read_text())
 v['ready']=len(v['gates'])==4 and all(x['pass_'] and x['max_cost_error']==0 for x in v['gates'].values()) and v.get('SERVICE_OWNER.json',{{}}).get('pid')==d['pid']
 v['log_tail']=(r/'LS_GPU.log').read_text()[-2000:];out.append(v)
print(json.dumps(dict(host={host!r},unix=time.time(),workers=out,gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,power.draw','--format=csv,noheader'],text=True))))''',timeout=30)
    v=json.loads(out);write(ops.EVIDENCE/f'LS_COLLECT_GPU_STATUS_{host}_{int(v["unix"])}.json',v);print(out)


def launch_collection():
    out=ops.ssh.remote('cpuQ',f'''import pathlib,json,os,time,subprocess,hashlib,shutil
r=pathlib.Path({ROOT!r});c=pathlib.Path({COLLECTION!r});cfg=json.loads((c/'LS_COLLECT_BINDING.json').read_text());s=pathlib.Path(cfg['source_root']);dest=c/'DISPATCH.json'
if dest.exists():print(dest.read_text())
else:
 assert cfg['amendment_sha256'] in (r/'AUTHORITY.md').read_text()
 for n,h in cfg['cpu']['source_hashes'].items():assert hashlib.sha256((s/n).read_bytes()).hexdigest()==h,n
 for a in ('B','R'):
  assert not (r/(a+'_checkpoint_004.pt')).exists()
  assert hashlib.sha256((r/(a+'_checkpoint_003.pt')).read_bytes()).hexdigest()==cfg['checkpoints'][a]
 assert (c/'BROKER_FROZEN.json').exists()
 mem={{line.split(':')[0]:int(line.split()[1])*1024 for line in pathlib.Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemAvailable:','MemFree:'))}}
 assert os.getloadavg()[0]<28 and mem['MemAvailable']>=8*1024**3
 assert shutil.disk_usage(c).free>=2*1024**3 and shutil.disk_usage('/dev/shm').free>=8*1024**3
 home='/home/ubuntu/miniconda3/lib/python3.10/site-packages/sumo'
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(s/'pytools')+':/home/ubuntu/e3final/runtime/src:/home/ubuntu/e3final/runtime',SUMO_HOME=home,PATH=home+'/bin:'+os.environ['PATH'])
 with (c/'SUPERVISOR.log').open('x') as log:
  p=subprocess.Popen(['nice','-n','19','/home/ubuntu/miniconda3/bin/python','-B','-u','-m','tools.coopmwm.ls_collection',str(c)],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 st=pathlib.Path('/proc',str(p.pid),'stat').read_text().split()
 v=dict(pid=p.pid,start_ticks=st[21],boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),unix=time.time(),source=str(s),root=str(c),mode='frozen B3 snapshots; no plans/no optimizer')
 with dest.open('x') as f:json.dump(v,f)
 print(json.dumps(v))''',timeout=30)
    v=json.loads(out);p=ops.EVIDENCE/'LS_COLLECT_DISPATCH.json'
    if not p.exists():write(p,v)
    print(out)


def collection_status():
    out=ops.ssh.remote('cpuQ',f'''import pathlib,json,time,os,shutil
r=pathlib.Path({ROOT!r});c=pathlib.Path({COLLECTION!r});d=json.loads((c/'DISPATCH.json').read_text());p=pathlib.Path('/proc',str(d['pid']));st=(p/'stat').read_text().split() if p.exists() else None
assert not st or st[21]==d['start_ticks']
v=dict(unix=time.time(),owner=d,state=st[2] if st else 'exited',load=os.getloadavg(),disk_free=shutil.disk_usage(c).free,no_u4_update=all(not (r/(a+'_checkpoint_004.pt')).exists() for a in ('B','R')))
for n in ('LS_COLLECT_STATUS.json','LS_COLLECT_FAILED.json'):
 if (c/n).exists():v[n]=json.loads((c/n).read_text())
v['log_tail']=(c/'SUPERVISOR.log').read_text()[-6000:]
v['episodes_claimed']=len(list((r/'jobs').glob('lscollect1_B3_*/CLAIM.json')))
v['episodes_completed']=len(list((r/'jobs').glob('lscollect1_B3_*/RESULT.json')))
v['snapshot_queries']=len(list((c/'pending').glob('*.json.gz')))
print(json.dumps(v))''',timeout=30)
    v=json.loads(out);write(ops.EVIDENCE/f'LS_COLLECT_STATUS_{int(v["unix"])}.json',v)
    print(out)


def pause():
    target = ops.EVIDENCE/'LS_COLLECT_PAUSE_REGISTRATION.json'
    if target.exists():
        reg = read(target)
    else:
        line = ('\nS-COOPMWM MAIN2026-11-02 18:38 LS-COLLECT PAUSE REGISTERED: '
                'B/R durable u3; no u4 PPO update. Close receipt-verified stalled warmdns1651_u004_B '
                'wave, never replay jobs; retain every charge, accepted response and full pool. '
                'A48 done. GPU claims continue; no duplicate inference. No new SUMO until '
                'LS_COLLECT_REGISTRATION quote/SHA. Resume only on MAIN addendum.\n')
        with checked(REGISTRY).open('a', encoding='utf-8') as f:
            f.write(line)
        write(target, dict(unix=time.time(), authority='MAIN18:38', line=line, new_sim_s=0))
        reg = read(target)
    amendment = sha(target)
    script = inspect.getsource(pause_remote)
    out = json.loads(ops.ssh.remote('cpuQ',
        'import json,time\n'+script+f'\nprint(json.dumps(pause_remote({ROOT!r},{amendment!r})))', timeout=40))
    dest = ops.EVIDENCE/'LS_COLLECT_PAUSED.json'
    if not dest.exists():
        write(dest, out)
    print(json.dumps(dict(paused=True, checkpoints=out['checkpoints'], closed_jobs=out['closed_jobs'],
                         no_u4_update=out['no_u4_update'], new_sim_s=0)))


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('action', choices=['pause','pause_latest','audit','qualify','cpu_result','register_collection','freeze_broker','gpu_collect','collection_gpu_status','launch_collection','collection_status'])
    p.add_argument('--host',choices=('gpuA','gpuR'))
    a = p.parse_args()
    if a.action in ('gpu_collect','collection_gpu_status'):globals()[a.action](a.host)
    else:globals()[a.action]()
