"""One-use inference-only qualification of the integrated warm MPPI source."""
import argparse
from contextlib import nullcontext
import copy
import json
from pathlib import Path
import time
from .storage import read,write
from .protocol import digest,sha


def setpoint_futures(rt,record,memory):
    """Reuse the registered E2 action set and labels; no new physics."""
    from types import SimpleNamespace
    from .wm import candidates,adapted_futures,executor_adapter
    from .fused import FusedModel
    from .speed2_runtime import compact_geometry,ledger
    from .speed2_transfers import batched_births
    proxy=FusedModel(rt.model,1,trim_absent=True);proxy.local.index=0
    proxy.live_slots=len(record['rows'])
    runtime=SimpleNamespace(**vars(rt));runtime.model=proxy
    ns=adapted_futures((60,120,300,600));ns['_actions']=candidates(record)
    with (memory.context(rt,record) if memory is not None else nullcontext()),compact_geometry(proxy),ledger(proxy),batched_births(cpu_pools=True),executor_adapter():
        result=ns['futures'](runtime,dict(record,setting=0),indices=tuple(range(8)))
    assert result['population_audit']['finite'] and result['population_audit']['conserved']
    return result


def run(root,host):
    import torch
    from tools.wmteach.runtime import bootstrap
    from .wm import cost_adapter
    from .warm_runtime import prepare
    from .credit import paired
    from .plan_wm import evaluate
    from .speed2_runtime import qualify
    from .speed2_memory import native_fixture
    from . import mppi_search as ms
    from .mppi_runtime import imagined_centres
    from .policy_wm import actor_state
    from .acceleration import Policy
    manifest=read(root/'QUAL_BINDING.json')
    for name,h in manifest['hashes'].items():assert sha(root/name)==h,name
    claim=root/'QUAL_STARTED.json'
    write(claim,dict(unix=time.time(),host=host,new_sumo_sim_s=0))
    rt=bootstrap();torch.cuda.set_per_process_memory_fraction(.65)
    rt.frozen.costs=cost_adapter(rt.frozen.costs,read(root/'FROZEN_COSTS.json')['code'],
        horizons=(60,120,300,360,420,600))
    try:
        records=read(root/'E2_INPUTS.json')
        ix=0 if host=='gpuA' else 1
        record=records[ix]
        start=time.monotonic();memory=prepare(rt,record)
        assert memory is not None and memory.posteriors
        write(root/'CAUSAL_MEMORY.json',dict(memory.receipt,preparation_wall_s=time.monotonic()-start,
            posterior_drivers=len(memory.posteriors),new_sumo_sim_s=0))
        e2={}
        exact=read(root/'E2_LABELS.json')[ix]['exact']
        for mode,mem in (('cold',None),('warm',memory)):
            start=time.monotonic();result=setpoint_futures(rt,record,mem)
            credits={h:paired([{a:fs[k] for a,fs in values.items()} for k in range(8)])
                for h,values in result['futures'].items()}
            choices={}
            for h,values in result['futures'].items():
                means={a:sum(v['team_delay'] for v in fs)/8 for a,fs in values.items()}
                chosen=min(means,key=lambda a:(means[a],a))
                labels=dict(joint=0.,all=exact[h]['D_G'],**{f'revert:{i}':v for i,v in enumerate(exact[h]['D_i'])})
                choices[h]=dict(chosen=chosen,predicted_saving=means['all']-means[chosen],
                    exact_labelled_saving=labels['all']-labels[chosen])
            e2[mode]=dict(credits=credits,choices=choices,wall_s=time.monotonic()-start)
            write(root/f'E2_{mode.upper()}.json',dict(e2[mode],world_id=record['world_id'],exact=exact,new_sumo_sim_s=0))
        native,native_memory=native_fixture(rt)
        cavs=[r['vid'] for r in native['rows'] if r['is_cav']]
        write(root/'NATIVE_GEOMETRY.json',dict(world_id=native['world_id'],cavs=len(cavs),
            groups=ms.group_components(native['rows']),new_sumo_sim_s=0))
        sizes=(2,6) if host=='gpuA' else (4,)
        for size in sizes:
            assert len(cavs)>=size,'qualified fixture lacks requested CAV count'
            group=ms.group_components([r for r in native['rows'] if r['vid'] in cavs[:size]])
            assert len(group)==1 and group[0]['size']==size
            group=group[0];rec=dict(native,group=group['members'],policy_state=actor_state(Policy()))
            bounds=(float(rt.mp.policy.accel_min),float(rt.mp.policy.accel_max))
            if size==sizes[0]:
                actions=ms.comparators(rec,600)
                actions.append(ms.sample_unscored(group,[[0.]*300]*size,bounds,'ordinary',20261101,duration_s=300))
                for k,h,width in ((4,120,34),(8,420,10)):
                    receipt=qualify(rt,rec,actions,k,h,width,memory=native_memory,transfers=True)
                    write(root/f'PARITY_K{k}_H{h}.json',receipt)
                    assert receipt['pass_'],'integrated warm batch exact parity failed'
                # Empty memory must be exactly cold in the actual GPU evaluator.
                empty=dict(rec,history=[])
                from .plan_service import request_memory
                assert request_memory(rt,empty,dict(memory_mode='warm')) is None
                a=evaluate(rt,rec,actions,1,60,chunk_actions=3,pad_actions=3,trim=False,speed2=True,transfers=True)
                b=evaluate(rt,empty,actions,1,60,chunk_actions=3,pad_actions=3,trim=False,speed2=True,transfers=True)
                assert a==b
                write(root/'EMPTY_COLD_PARITY.json',dict(pass_=True,max_cost_error=0.,new_sumo_sim_s=0))
            begin=time.monotonic()
            policy,mappo,centre=imagined_centres(rt,rec,native_memory,speed2=True,transfers=True)
            def score(r,actions,k,h):
                width=34 if k==4 else 10
                return evaluate(rt,r,actions,k,h,chunk_actions=width,pad_actions=width,trim=False,
                    speed2=True,transfers=True,memory=native_memory)
            result=ms.search(rec,group,policy,bounds,score,'ordinary',mappo_traj=mappo,seed=20261101)
            wall=time.monotonic()-begin
            write(root/f'SEARCH_GROUP{size}.json',dict(result=result,centre_rollout=centre,
                wall_s=wall,plans_per_hour=3600/wall,new_sumo_sim_s=0))
        write(root/'COMPLETE.json',dict(pass_=True,host=host,new_sumo_sim_s=0))
    except BaseException as error:
        write(root/'FAILED.json',dict(error=repr(error),new_sumo_sim_s=0));raise


def stage(host,tag):
    from . import ops
    root='/root/coopmwm20261102_'+tag
    files=ops.sources()
    frozen=ops.ssh.remote('gpuA',"import pathlib;print(pathlib.Path('/root/coopmwm20261101_warmqual1/FROZEN_COSTS.json').read_text())",timeout=20)
    fixtures=ops.ssh.remote('gpuA',"import pathlib;print(pathlib.Path('/root/coopmwm20261101_warmqual1/E2_INPUTS.json').read_text())",timeout=25)
    files.update({'FROZEN_COSTS.json':frozen,'E2_INPUTS.json':fixtures,
        'E2_LABELS.json':json.dumps([read(ops.EVIDENCE/f'runtime/CREDIT_WINDOW_{i:02}.json') for i in (0,1)])})
    staged=json.loads(ops.stage(host,root,files))
    binding=dict(hashes=staged['hashes'],host=host,root=root,new_sumo_sim_s=0,
        search='registered medium dimension-scaled MPPI; no training outcome selection')
    ops.stage(host,root,{'QUAL_BINDING.json':json.dumps(binding,sort_keys=True)})
    write(ops.EVIDENCE/f'{tag}_STAGE_{host}.json',binding)
    print(json.dumps(dict(host=host,root=root,files=len(staged['hashes']))))


def dispatch(host,tag):
    from . import ops
    root='/root/coopmwm20261102_'+tag
    source=f'''import pathlib,json,os,subprocess,time
r=pathlib.Path({root!r});assert r.is_dir()
assert not (r/'DISPATCH.json').exists() and not (r/'QUAL_STARTED.json').exists()
env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
 PYTHONPATH=str(r/'pytools')+':/root/rbeval20261031/phys_code_22cc32f65a80/src:/root/rbeval20261031/phys_code_22cc32f65a80',
 COOPMWM_COST_SOURCE=str(r/'FROZEN_COSTS.json'))
cmd=[{ops.ssh.PYTHON[host]!r},'-B','-u','-m','tools.coopmwm.warm_release_qual','run','--host',{host!r},'--tag',{tag!r}]
with (r/'QUAL.log').open('x') as log:
 p=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
s=(pathlib.Path('/proc')/str(p.pid)/'stat').read_text().split()
v=dict(pid=p.pid,start_ticks=s[21],boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),root=str(r),unix=time.time(),new_sumo_sim_s=0)
with (r/'DISPATCH.json').open('x') as f:json.dump(v,f)
print(json.dumps(v))'''
    value=json.loads(ops.ssh.remote(host,source,timeout=25))
    write(ops.EVIDENCE/f'{tag}_DISPATCH_{host}.json',value);print(json.dumps(value))


def status(host,tag):
    from . import ops
    root='/root/coopmwm20261102_'+tag
    value=json.loads(ops.ssh.remote(host,f'''import pathlib,json,time
r=pathlib.Path({root!r});assert r.is_dir()
d=json.loads((r/'DISPATCH.json').read_text());p=pathlib.Path('/proc')/str(d['pid'])
s=(p/'stat').read_text().split() if (p/'stat').exists() else None
assert not s or s[21]==d['start_ticks']
names=['CAUSAL_MEMORY.json','E2_COLD.json','E2_WARM.json','NATIVE_GEOMETRY.json','PARITY_K4_H120.json','PARITY_K8_H420.json','EMPTY_COLD_PARITY.json','SEARCH_GROUP2.json','SEARCH_GROUP4.json','SEARCH_GROUP6.json','FAILED.json','COMPLETE.json']
receipts={{n:json.loads((r/n).read_text()) for n in names if (r/n).exists()}}
print(json.dumps(dict(unix=time.time(),state=s[2] if s else 'exited',receipts=receipts,log=(r/'QUAL.log').read_text()[-12000:],new_sumo_sim_s=0)))''',timeout=25))
    write(ops.EVIDENCE/f'{tag}_STATUS_{host}_{int(value["unix"])}.json',value)
    # Keep the terminal short; all full receipts remain durable locally.
    print(json.dumps(dict(state=value['state'],receipts=list(value['receipts']),log=value['log'][-5000:])))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('stage','dispatch','status','run'))
    p.add_argument('--host',choices=('gpuA','gpuR'),required=True);p.add_argument('--tag',default='warmmppiqual1');a=p.parse_args()
    if a.mode=='run':run(Path('/root/coopmwm20261102_'+a.tag),a.host)
    else:globals()[a.mode](a.host,a.tag)
