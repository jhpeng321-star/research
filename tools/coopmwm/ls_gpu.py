"""Logging-only candidate-profile capture around the identical plain MPPI search."""
import copy


def live_identity(boot,stat,argv,old):
    """An empty argv with the same boot/start is exit, never a signal target."""
    assert boot==old['boot_id']
    if stat is None:return False
    assert stat[21]==old['start_ticks']
    if stat[2]=='Z' or not argv:return False
    assert argv==old['argv']
    return True


def capture_search(search,label_list8=False):
    def wrapped(*args,**kwargs):
        # positional API: record, group, policy, bounds, evaluate, regime
        values=list(args);evaluate=values[4];last=[]
        def score(record,actions,k,h):
            costs=evaluate(record,actions,k,h)
            if k==8:
                def mean(key):
                    xs=costs[key]
                    return sum(x['team_delay'] if isinstance(x,dict) else x for x in xs)/len(xs)
                plans=[a for a in actions if 'profile' in a]
                last[:]=[dict(copy.deepcopy(p),saving_vs_mappo=mean('MAPPO')-mean(p['id']),
                    saving_vs_policy=mean('policy')-mean(p['id'])) if label_list8 else copy.deepcopy(p)
                    for p in sorted(plans,key=lambda p:(mean(p['id']),p['id']))]
            return costs
        values[4]=score
        if label_list8:
            # Keep exactly the original eight final WM calls: top_k=8, F/k=1.
            # The labelled list is deliberately unfiltered, including losses.
            kwargs['config']=dict(kwargs.get('config') or {},top_k=8,finalists_per_k=1)
        out=search(*values,**kwargs)
        out['collection_top8']=last[:8]
        if label_list8:out['topk']=copy.deepcopy(last[:8])
        return out
    return wrapped


def run(root,old,deadline,amendment,recover=False):
    import os,sys,time,json,signal,hashlib,fcntl
    from pathlib import Path
    root=Path(root);sys.path.insert(0,str(root/'pytools'))
    from tools.coopmwm.warm_spool import immutable,SpoolInput
    def owned():
        boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        p=Path('/proc',str(old['pid']))
        try:
            st=(p/'stat').read_text().split()
            argv=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
        except FileNotFoundError:
            st=None;argv=''
        return live_identity(boot,st,argv,old)
    if not recover:assert owned()
    quiet=0
    while quiet<3:
        if not owned():
            assert recover and (root/'LS_GPU_ORIGINALS.json').exists()
            break
        missing=[p for p in (root/'inbox').glob('*.json') if not (root/'forecasts'/p.name).exists()]
        quiet=0 if missing else quiet+1;time.sleep(2)
    names=['GPU_BINDING.json','SPOOL_OWNER.json','SERVICE_OWNER.json']+[p.name for p in root.glob('STARTUP_*.json')]
    original={n:json.loads((root/n).read_text()) for n in names}
    if (root/'LS_GPU_ORIGINALS.json').exists():
        assert recover
        original=json.loads((root/'LS_GPU_ORIGINALS.json').read_text())
    else:immutable(root/'LS_GPU_ORIGINALS.json',original)
    binding=original['GPU_BINDING.json']
    for n,h in binding['hashes'].items():assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h,n
    if owned():os.kill(old['pid'],signal.SIGTERM)
    for _ in range(150):
        if not owned():break
        time.sleep(.1)
    else:raise ValueError('old idle search worker did not exit')
    assert all((root/'forecasts'/p.name).exists() for p in (root/'inference_claims').glob('*.json')), 'interrupted inference; never replay'
    lock=(root/'SPOOL_WORKER.claim').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def replace(p,v):
        tmp=p.with_name(p.name+'.ls.publishing')
        with tmp.open('x') as f:json.dump(v,f);f.flush();os.fsync(f.fileno())
        os.replace(tmp,p)
    owner=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],boot_id=old['boot_id'],unix=time.time(),new_sumo_sim_s=0)
    replace(root/'SPOOL_OWNER.json',owner)
    suffix='_RECOVERY1' if recover else ''
    immutable(root/('LS_GPU_RESTART'+suffix+'.json'),dict(old=old,new=owner,amendment_sha256=amendment,inference_replayed=False))
    signal.signal(signal.SIGHUP,signal.SIG_IGN);sys.stdin=SpoolInput(root,deadline)
    from tools.coopmwm import warm_service,plan_service,mppi_search,plans
    # Generic-search and WM arithmetic untouched. This guard is post-hoc only.
    families=plans.family_seeds
    plans.family_seeds=lambda rows,group,*a,**kw:[] if len(group) not in (2,3) else families(rows,group,*a,**kw)
    mppi_search.search=capture_search(mppi_search.search,label_list8=True)
    binding=dict(binding,deadline_unix=deadline,gpu_deadline=deadline)
    binding['adopted_startup_gates']=dict(receipts={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in warm_service.ADOPTABLE},
        numerical_hashes={n:binding['hashes'][n] for n in warm_service.NUMERICAL})
    previous=plan_service.read
    warm_service.read=plan_service.read=lambda p:binding if p==root/'GPU_BINDING.json' else previous(p)
    exclusive=plan_service.write
    def write(p,v):
        if p.parent==root and p.name in original:
            if p.name.startswith('STARTUP_'):return exclusive(root/(p.stem+'_LSCOLLECT'+suffix+'.json'),v)
            if p.name=='SERVICE_OWNER.json':return replace(p,v)
        return exclusive(p,v)
    warm_service.write=plan_service.write=write
    try:warm_service.main(root)
    except BaseException as error:
        immutable(root/('LS_GPU_FAILED'+suffix+'.json'),dict(error=repr(error),unix=time.time(),owner=owner));raise


if __name__=='__main__':
    import json,sys
    run(sys.argv[1],json.loads(sys.argv[2]),float(sys.argv[3]),sys.argv[4],recover='--recover' in sys.argv[5:])
