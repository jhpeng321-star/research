"""Owned CPU-only dispatch/collection for the entrant-inclusive FIT diagnostic."""
import argparse
import base64
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
PYTHON = '/home/ubuntu/miniconda3/bin/python'
INT_COMMIT = '4015b641'
LANE_ENCODER_COMMIT = 'b192e0c8'


def remote(code):
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                             'cpuQ', PYTHON+' -B -'], input=code, text=True,
                            capture_output=True, timeout=120)
    if result.returncode:
        raise RuntimeError(result.stdout+'\n'+result.stderr)
    return result.stdout


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=('status', 'stage', 'launch', 'poll', 'trace', 'initial-geometry', 'collect'))
    p.add_argument('--version', default='v8end600')
    p.add_argument('--horizon', type=int, choices=(120, 300, 600), default=600)
    p.add_argument('--slots', type=int, default=512)
    p.add_argument('--entry-phase', choices=('before', 'after'), default='after')
    p.add_argument('--details', action='store_true')
    p.add_argument('--seed', type=int, default=969400)
    p.add_argument('--identity', type=int)
    args = p.parse_args()
    if not args.version.isalnum():
        p.error('version must be alphanumeric')
    if not 969400 <= args.seed <= 969431:
        p.error('FIT seed required')
    header = 'import os,json,time,hashlib\nfrom pathlib import Path\n'
    header += f'root=Path({("/tmp/rb-merge-fullfleet-"+args.version)!r})\n'
    header += f'details={args.details!r}\n'
    header += f'seed={args.seed!r}\n'
    header += f'identity={args.identity!r}\n'
    header += f'horizon={args.horizon!r}\nslots={args.slots!r}\n'
    header += f'entry_phase={args.entry_phase!r}\n'
    if args.mode == 'status':
        print(remote(header+"print(json.dumps(dict(load=os.getloadavg(), cpus=os.cpu_count(), old_owned_pids={p:Path('/proc/'+str(p)).exists() for p in (200540,233273)}, existing_dispatch=(root/'LAUNCH.json').exists())))"))
    elif args.mode == 'stage':
        archive = subprocess.check_output(['git', 'archive', '--format=tar', INT_COMMIT, 'src/wms',
                                           'configs/native_cav_head_v5.json',
                                           'configs/execparity_cav_idm_v1.json'],
                                           cwd='E:/agent-work/c4-rb-int')
        encoder_source = json.loads((Path('E:/agent-work/c4-rb-lane/tools/rblane/receipts/ENCODER_SOURCE.json')).read_text())
        checkpoint = Path(encoder_source['local_path']).read_bytes()
        import hashlib
        assert hashlib.sha256(checkpoint).hexdigest() == encoder_source['source_sha256'] == 'b8797242c6b736b57eb17b1aed77d2cf01d813770342f9e17ba33dc5b8b33436'
        def lane_file(name):
            return subprocess.check_output(['git', 'show', LANE_ENCODER_COMMIT+':'+name], cwd='E:/agent-work/c4-rb-lane')
        payload = dict(runtime_archive=base64.b64encode(archive).decode(),
            checkpoint=base64.b64encode(checkpoint).decode(),
            encoder=base64.b64encode(lane_file('tools/rblane/encoder.py')).decode(),
            encoder_tests=base64.b64encode(lane_file('tests/test_lc2013_encoder.py')).decode(),
            point_adapter=base64.b64encode((ROOT/'tools/rbeval/encoded_fleet.py').read_bytes()).decode(),
            point_tests=base64.b64encode((ROOT/'tests/unit/test_rb_merge_encoded_fleet.py').read_bytes()).decode(),
            merge_kernel=base64.b64encode((ROOT/'src/wms/physics/junction_merge.py').read_bytes()).decode(),
            network=base64.b64encode((ROOT/'src/wms/physics/merge_network.py').read_bytes()).decode(),
            insertion=base64.b64encode((ROOT/'src/wms/model/sumo_insertion.py').read_bytes()).decode(),
            native=base64.b64encode((ROOT/'src/wms/model/sumo_native.py').read_bytes()).decode(),
            native_tests=base64.b64encode((ROOT/'tests/unit/test_rb_merge_native.py').read_bytes()).decode(),
            departure=base64.b64encode((ROOT/'src/wms/model/sumo_departure.py').read_bytes()).decode(),
            departure_tests=base64.b64encode((ROOT/'tests/unit/test_rb_merge_departure_phase.py').read_bytes()).decode(),
            merge_tests=base64.b64encode((ROOT/'tests/unit/test_rb_merge.py').read_bytes()).decode(),
            validator=base64.b64encode((ROOT/'tools/rbeval/full_fleet.py').read_bytes()).decode(),
            tests=base64.b64encode((ROOT/'tests/unit/test_rb_merge_full_fleet.py').read_bytes()).decode(),
            targets={n: base64.b64encode((Path('E:/agent-work/rb-shared/targets')/n).read_bytes()).decode()
                     for n in ('ramp_delay_world.csv', 'ramp_merges.csv', 'insertions_10s.csv')})
        code = header+f'payload=json.loads({json.dumps(payload)!r})\n'
        code += '''import base64,io,tarfile,subprocess,sys
assert root.parent.is_dir()
root.mkdir(exist_ok=False)
runtime=root/'runtime'
runtime.mkdir()
with tarfile.open(fileobj=io.BytesIO(base64.b64decode(payload['runtime_archive']))) as tar:
    tar.extractall(runtime, filter='data')
# Overlay only MERGE's owned kernel; keep INT's entrant-identity and lane hooks.
(runtime/'src/wms/physics/junction_merge.py').write_bytes(base64.b64decode(payload['merge_kernel']))
(runtime/'src/wms/physics/merge_network.py').write_bytes(base64.b64decode(payload['network']))
(runtime/'src/wms/model/sumo_insertion.py').write_bytes(base64.b64decode(payload['insertion']))
(runtime/'src/wms/model/sumo_native.py').write_bytes(base64.b64decode(payload['native']))
(runtime/'src/wms/model/sumo_departure.py').write_bytes(base64.b64decode(payload['departure']))
adapter=runtime/'src/wms/model/sumo_merge.py'
text=adapter.read_text()
old='merge_speed_advice(kin, present, accel, physics, driver_params)'
if text.count(old)==1:
    text=text.replace(old,"merge_speed_advice(kin, present, accel, physics, driver_params, lookahead_speed=state.get('lc2013_lookahead'))")
else:
    assert text.count("lookahead_speed=state.get('lc2013_lookahead')")==1
old='apply_merge(moved, present, next_lane, deciders, model.physics, driver_params)'
if text.count(old)==1:
    text=text.replace(old,"apply_merge(moved, present, next_lane, deciders, model.physics, driver_params, lookahead_speed=out.get('lc2013_lookahead'))")
else:
    assert text.count("lookahead_speed=out.get('lc2013_lookahead')")==1
adapter.write_text(text)
encoder_dir=runtime/'tools/rblane'
encoder_dir.mkdir(parents=True)
(encoder_dir/'encoder.py').write_bytes(base64.b64decode(payload['encoder']))
inputs=root/'inputs'
inputs.mkdir()
(inputs/'wm_s.pt').write_bytes(base64.b64decode(payload['checkpoint']))
tools=root/'tools'/'rbeval'
tools.mkdir(parents=True)
(tools/'full_fleet.py').write_bytes(base64.b64decode(payload['validator']))
(tools/'encoded_fleet.py').write_bytes(base64.b64decode(payload['point_adapter']))
test=root/'test_full_fleet.py'
test.write_bytes(base64.b64decode(payload['tests']))
merge_test=root/'test_merge_source.py'
merge_test.write_bytes(base64.b64decode(payload['merge_tests']))
native_test=root/'test_native_physics.py'
native_test.write_bytes(base64.b64decode(payload['native_tests']))
point_test=root/'test_encoded_fleet.py'
point_test.write_bytes(base64.b64decode(payload['point_tests']))
encoder_test=root/'test_frozen_encoder.py'
encoder_test.write_bytes(base64.b64decode(payload['encoder_tests']))
departure_test=root/'test_departure_phase.py'
departure_test.write_bytes(base64.b64decode(payload['departure_tests']))
targets=root/'targets'
targets.mkdir()
for name,data in payload['targets'].items():
    (targets/name).write_bytes(base64.b64decode(data))
env=dict(os.environ, PYTHONPATH=str(runtime/'src')+':'+str(root)+':'+str(runtime), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
result=subprocess.run([sys.executable,'-m','pytest',str(test),str(merge_test),str(native_test),str(point_test),str(encoder_test),str(departure_test),'-q'],cwd=root,env=env,text=True,capture_output=True)
receipt=dict(command=result.args,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,load=os.getloadavg(),cpus=os.cpu_count())
(root/'CPU_TESTS.json').write_text(json.dumps(receipt,indent=2)+'\\n')
print(json.dumps(receipt))
assert result.returncode==0
preflight=subprocess.run([sys.executable,'-B',str(tools/'full_fleet.py'),'--root','/home/ubuntu/wmdata20261031/views','--runtime',str(runtime),'--targets',str(targets),'--seeds',*map(str,range(969400,969432)),'--encoder-checkpoint',str(inputs/'wm_s.pt'),'--preflight'],cwd=root,env=env,text=True,capture_output=True)
pre=dict(command=preflight.args,returncode=preflight.returncode,stdout=preflight.stdout,stderr=preflight.stderr)
(root/'PREFLIGHT.json').write_text(json.dumps(pre,indent=2)+'\\n')
print(json.dumps(dict(preflight_returncode=preflight.returncode,fit_worlds=32,stderr=preflight.stderr)))
assert preflight.returncode==0
'''
        print(remote(code))
    elif args.mode == 'launch':
        code = header+f'python={PYTHON!r}\n'
        code += '''import subprocess
assert json.loads((root/'CPU_TESTS.json').read_text())['returncode']==0
assert json.loads((root/'PREFLIGHT.json').read_text())['returncode']==0
assert os.getloadavg()[0]<28 and os.cpu_count()>=32
assert not (root/'results').exists()
with (root/'LAUNCH.json').open('x') as f:
    json.dump(dict(state='reserved',owner='S-RB-MERGE'),f)
script=root/'tools/rbeval/full_fleet.py'
command=[python,'-B',str(script),'--root','/home/ubuntu/wmdata20261031/views','--runtime',str(root/'runtime'),'--targets',str(root/'targets'),'--output',str(root/'results'),'--seeds',*map(str,range(969400,969432)),'--horizon',str(horizon),'--slots',str(slots),'--cf-model','sumo_idm','--warm-lc','--trace-merge','--native-model','sumo','--encoder-checkpoint',str(root/'inputs/wm_s.pt')]
command += ['--entry-phase',entry_phase]
env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
with (root/'RUN.log').open('x') as log:
    child=subprocess.Popen(command,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=lambda:os.nice(19))
record=dict(owner='S-RB-MERGE',pid=child.pid,started_unix=time.time(),command=command,validator_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),load=os.getloadavg(),cpus=os.cpu_count(),scope='FIT32 native full fleet H'+str(horizon)+', frozen motion encoder; one CPU thread; no SUMO/GPU/training')
(root/'LAUNCH.json').write_text(json.dumps(record,indent=2)+'\\n')
print(json.dumps(record))
'''
        print(remote(code))
    elif args.mode == 'poll':
        code = header+'''launch=json.loads((root/'LAUNCH.json').read_text())
pid=launch['pid']
result=root/'results'
rows=[s for s in range(969400,969432) if (result/(str(s)+'.json')).exists()]
summaries=[]
for seed in rows:
    row=json.loads((result/(str(seed)+'.json')).read_text())
    summaries.append(dict(seed=seed,predicted_ramp=row['delay']['predicted_ramp_veh_s'],recorded_ramp=float(row['delay']['recorded']['ramp_delay_vs']),ramp_source=row['delay']['ramp_source_veh_s'],inserted=row['demand']['inserted'],real_merges=row['merges']['recorded'],predicted_merges=row['merges']['predicted']))
failed=json.loads((result/'FAILED.json').read_text()) if (result/'FAILED.json').exists() else None
print(json.dumps(dict(pid=pid,alive=Path('/proc/'+str(pid)).exists(),completed=len(rows),complete=(result/'COMPLETE.json').exists(),failed=failed,rows=summaries if details else summaries[-1:],log=(root/'RUN.log').read_text() if failed or details else None,load=os.getloadavg())))
'''
        print(remote(code))
    elif args.mode == 'trace':
        code = header+'''path=root/'results'/(str(seed)+'.json')
if not path.exists():
    print(json.dumps(dict(seed=seed,available=False)))
else:
    row=json.loads(path.read_text())
    trace=row.get('merge_front_trace',[])
    latest=max((r['time_s'] for r in trace),default=None)
    heads=[r for r in trace if r['time_s']==latest]
    head=max(heads,key=lambda r:r['s_m']) if heads else None
    selected=identity if identity is not None else head['identity'] if head else None
    history=[r for r in trace if r['identity']==selected]
    if identity is not None:
        head=history[-1] if history else None
    summary=dict(seed=seed,available=True,head=head,head_first=history[0] if history else None,head_samples=len(history),safe_samples=sum(r['safe'] for r in history),blocked_leader_samples=sum(r['blocked_leader'] for r in history),blocked_follower_samples=sum(r['blocked_follower'] for r in history),body_overlap=row.get('body_overlap'))
    if details:
        summary['history']=history
    summary['selected_identity']=selected
    summary['merge_events']=[e for e in row['merges']['events'] if e['identity']==selected]
    print(json.dumps(summary))
'''
        print(remote(code))
    elif args.mode == 'initial-geometry':
        code = header+'''import sys
sys.path[:0]=[str(root/'runtime/src'),str(root)]
import numpy as np
import torch
torch.set_num_threads(1)
from tools.rbeval.full_fleet import safe,body_overlap_count,digest
from wms.physics.lane_chain import chain_s
output=root/'INITIAL_GEOMETRY.json'
if output.exists():
    report=json.loads(output.read_text())
else:
    views=Path('/home/ubuntu/wmdata20261031/views')
    rows=[]
    for seed in range(969400,969432):
        record=json.loads(safe(views/('world_'+str(seed)+'.json')).read_text())['native']
        ep=safe(views/'episodes'/record['episode'])
        meta=json.loads(safe(ep/'episode.json').read_text())
        assert meta['seed']==seed and meta['arm']=='native'
        t0=int(meta['t0_index'])
        kin=np.array(np.load(safe(ep/'veh_kin.npy'),mmap_mode='r',allow_pickle=False)[t0],copy=True)
        mask=np.load(safe(ep/'veh_mask.npy'),mmap_mode='r',allow_pickle=False)[t0].astype(bool)&(kin[:,1:4].sum(-1)>0)
        lane=kin[:,1:4].argmax(-1)
        physical=chain_s(torch.as_tensor(lane),torch.as_tensor(kin[:,0])).numpy()
        raw=body_overlap_count(kin[:,0],lane,kin[:,6],mask)
        chained=body_overlap_count(physical,lane,kin[:,6],mask)
        rows.append(dict(seed=seed,frame_pairs=raw[0],frame_worst_gap=raw[1],chain_pairs=chained[0],chain_worst_gap=chained[1],episode_sha256=digest(ep/'episode.json')))
    report=dict(contract='Recorded t0 geometry only; body overlap, not SUMO collision logs',rows=rows,frame_pairs=sum(r['frame_pairs'] for r in rows),chain_pairs=sum(r['chain_pairs'] for r in rows))
    with output.open('x') as f:
        json.dump(report,f,indent=2)
print(json.dumps(dict(contract=report['contract'],worlds=len(report['rows']),frame_pairs=report['frame_pairs'],chain_pairs=report['chain_pairs'],affected=[r for r in report['rows'] if r['frame_pairs'] or r['chain_pairs']])))
'''
        print(remote(code))
    else:
        code = header+'''result=root/'results'
files=[root/'LAUNCH.json',root/'CPU_TESTS.json',root/'PREFLIGHT.json',root/'INITIAL_GEOMETRY.json',result/'IDENTITY.json',result/'COMPLETE.json',result/'FAILED.json']
files += [result/(str(s)+'.json') for s in range(969400,969432)]
print(json.dumps({p.name:dict(content=p.read_text(),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files if p.exists()}))
'''
        data = json.loads(remote(code))
        dest = ROOT/('docs/rb_merge/fullfleet_'+args.version)
        dest.mkdir(exist_ok=False)
        for name, item in data.items():
            (dest/name).write_text(item['content'], encoding='utf-8', newline='\n')
            import hashlib
            assert hashlib.sha256((dest/name).read_bytes()).hexdigest() == item['sha256']
        (dest/'COLLECTION.json').write_text(json.dumps({n: v['sha256'] for n,v in data.items()}, indent=2)+'\n', newline='\n')
        print('Hash-verified collection:', dest, len(data))


if __name__ == '__main__':
    main()
