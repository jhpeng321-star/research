"""Operational regressions: real state transitions and pipes, zero GPU/SUMO."""
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess
import sys
import threading
from types import SimpleNamespace, ModuleType

import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/c4'))
sys.path.insert(0,str(ROOT))
import loop14 as loop  # noqa: E402
import loop14_resume as resume  # noqa: E402
import loop14_fast as fast  # noqa: E402
from tools.resumeprep import boxstage, control  # noqa: E402


def config(tmp_path):
    spec=importlib.util.spec_from_file_location('fixture_loop',ROOT/'tests/unit/test_loop14.py')
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    cfg=loop.read_json(m._config(tmp_path,m._registration(tmp_path)))
    cfg['hosts'].update(cpuP=cfg['hosts']['cpu'],cpuQ=cfg['hosts']['cpu'])
    cfg['fastloop']={'two_box':True,'boxes':['cpuP','cpuQ'],'workers_per_box':1}
    cfg['resumeprep']={'enabled':True}
    loop.init(cfg,SimpleNamespace(adopt_policy=None))
    return cfg


def test_operational_retry_preserves_failure_and_uses_new_attempt(tmp_path,monkeypatch):
    cfg=config(tmp_path)
    calls=[]
    def run(cfg,state,ph,root):
        calls.append(ph['attempt'])
        if len(calls)==1:
            raise OSError('transient transport')
        return {'ok':True}
    monkeypatch.setitem(loop.RUNNERS,'policy',run)
    assert resume.phase(loop,cfg,'policy',tmp_path)==0
    assert calls==[1,2]
    proof=loop.read_json(tmp_path/'RETRY_c00_policy.json')
    assert proof['failure']['attempt']==1 and proof['status']=='recovered'
    assert proof['charges_before']==proof['charges_after']==0


@pytest.mark.parametrize('message,expected_calls', [('finite False',1),('native-relative veto gate',1),('connection failed',2)])
def test_scientific_or_second_failure_never_third_attempt(tmp_path,monkeypatch,message,expected_calls):
    cfg=config(tmp_path)
    calls=[]
    def run(*args):
        calls.append(1)
        raise RuntimeError(message)
    monkeypatch.setitem(loop.RUNNERS,'policy',run)
    with pytest.raises(loop.Refused):
        resume.phase(loop,cfg,'policy',tmp_path)
    assert len(calls)==expected_calls
    if expected_calls==2:
        with pytest.raises(loop.Refused):
            resume.phase(loop,cfg,'policy',tmp_path)
        assert len(calls)==2


def test_shared_gpu_accepts_only_live_exact_owned_child(tmp_path,monkeypatch):
    cfg=config(tmp_path)
    monkeypatch.setattr(loop.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=0,stdout='123, python, 800 MiB\n'))
    monkeypatch.setattr(fast,'process_token',lambda pid:'start1')
    monkeypatch.setattr(loop,'fast_module',lambda:fast)
    proc=SimpleNamespace(pid=123,poll=lambda:None)
    resume.child(proc,'start1')
    try:
        assert loop.gpu_guard(cfg,cfg['hosts']['gpu'])['compute_apps']==0
        monkeypatch.setattr(fast,'process_token',lambda pid:'reused')
        with pytest.raises(loop.GpuBusy):
            loop.gpu_guard(cfg,cfg['hosts']['gpu'])
    finally:
        resume.release(proc)


def test_first_resource_refusal_retries_without_owner_wait(tmp_path,monkeypatch):
    cfg=config(tmp_path)
    calls=[]
    def run(*args):
        calls.append(1)
        if len(calls)==1:
            raise loop.Refused('disk guard: temporarily insufficient free bytes')
        return {'ok':True}
    monkeypatch.setitem(loop.RUNNERS,'policy',run)
    assert resume.phase(loop,cfg,'policy',tmp_path)==0
    assert len(calls)==2
    assert loop.read_json(tmp_path/'RETRY_c00_policy.json')['status']=='recovered'


def test_driver_setup_error_has_durable_single_retry(tmp_path,monkeypatch):
    cfg=config(tmp_path)
    calls=[]
    def once(*args):
        calls.append(1)
        if len(calls)==1:
            raise OSError('temporary deployment lookup failure')
    monkeypatch.setattr(resume,'_drive',once)
    resume.drive(loop,None,cfg,tmp_path)
    assert len(calls)==2
    assert loop.read_json(tmp_path/'DRIVER_RETRY.json')['status']=='recovered'


def test_retry_budget_is_shared_across_phases(tmp_path,monkeypatch):
    cfg=config(tmp_path)
    count=[]
    def first(*args):
        count.append(1)
        if len(count)==1:
            raise OSError('transport')
        return {'ok':True}
    monkeypatch.setitem(loop.RUNNERS,'policy',first)
    assert resume.phase(loop,cfg,'policy',tmp_path)==0
    def second(*args):
        raise OSError('another transport failure')
    monkeypatch.setitem(loop.RUNNERS,'collect',second)
    with pytest.raises(loop.Refused,match='second failure'):
        resume.phase(loop,cfg,'collect',tmp_path)
    assert loop.load_state(cfg)['cycles'][0]['phases']['collect']['attempt']==1


def test_shared_48gb_guard_requires_combined_budget_and_headroom(tmp_path,monkeypatch):
    cfg=config(tmp_path)
    cfg['resumeprep']['shared_gpu']=True
    for name in ('policy','wm'):
        cfg[name]['env']={'C4_GPU_MEM_FRACTION':'0.20'}
    memory=['30000, 49140']
    def query(argv,**kwargs):
        stdout=memory[0] if '--query-gpu=memory.free,memory.total' in argv else '900, python, 20000 MiB'
        return SimpleNamespace(returncode=0,stdout=stdout)
    monkeypatch.setattr(loop.subprocess,'run',query)
    monkeypatch.setattr(loop,'fast_module',lambda:fast)
    monkeypatch.setattr(fast,'process_token',lambda pid:None)
    assert loop.gpu_guard(cfg,cfg['hosts']['gpu'])['shared_main_fraction']==0.4
    memory[0]='20000, 49140'
    with pytest.raises(loop.GpuBusy,match='headroom'):
        loop.gpu_guard(cfg,cfg['hosts']['gpu'])
    cfg['wm']['env']['C4_GPU_MEM_FRACTION']='0.30'
    with pytest.raises(loop.Refused,match='combined'):
        loop.gpu_guard(cfg,cfg['hosts']['gpu'])


def test_wm_and_policy_overlap_but_single_writer_and_provenance(tmp_path,monkeypatch):
    cfg=config(tmp_path)
    cfg['fastloop']['overlap_policy']=True
    state=loop.load_state(cfg)
    for name in ('policy','collect','convert'):
        state['cycles'][0]['phases'][name]['status']='done'
    loop.save_state(cfg,state)
    started=threading.Event()
    wm_done=threading.Event()
    events=[]
    def overlap(loop_,cfg_,snapshot,root_):
        assert snapshot['accepted_wm']['sha256']==state['accepted_wm']['sha256']
        started.set()
        assert wm_done.wait(3)
        events.append('policy_done')
    def phase(loop_,cfg_,name,arm):
        if name=='wm_ft':
            assert started.wait(3)
            events.append('wm_done')
            wm_done.set()
            changed=loop.load_state(cfg)
            changed['cycles'][0]['phases']['wm_ft']['status']='done'
            loop.save_state(cfg,changed)
        else:
            assert events==['wm_done','policy_done']
            changed=loop.load_state(cfg)
            changed['stopped']='test_complete'
            loop.save_state(cfg,changed)
        return 0
    monkeypatch.setattr(loop,'fast_module',lambda:fast)
    monkeypatch.setattr(fast,'overlap',overlap)
    monkeypatch.setattr(resume,'phase',phase)
    resume.drive(loop,SimpleNamespace(prepare_horizon=lambda *a:None,retire_pair_cache=lambda *a:None),cfg,tmp_path)
    assert events==['wm_done','policy_done']


def test_box_stream_regular_files_hashes_and_short_argv(tmp_path):
    src=tmp_path/'source'
    src.mkdir()
    (src/'file.py').write_bytes(b'print(1)\n')
    dst=tmp_path/'destination'
    a,b=boxstage.commands('gpuR',str(src).replace('\\','/'),'gpuA',str(dst).replace('\\','/'),['file.py'])
    assert max(map(len,(a[-1],b[-1])))<8191
    producer=[sys.executable,*shlex.split(a[-1])[1:]]
    consumer=[sys.executable,*shlex.split(b[-1])[1:]]
    data=subprocess.check_output(producer)
    result=subprocess.run(consumer,input=data,capture_output=True,check=True)
    assert json.loads(result.stdout)['verified']==1
    assert (dst/'file.py').read_bytes()==b'print(1)\n'


@pytest.mark.parametrize('path',['../outside','/absolute'])
def test_box_bundle_rejects_unsafe_names(path):
    with pytest.raises(ValueError):
        boxstage.commands('gpuR','/root','gpuA','/root/own',[path])


@pytest.mark.skipif(sys.platform=='win32',reason='POSIX symlink materialization exercised on box')
def test_box_materializes_approved_links_and_refuses_outside_target(tmp_path):
    src=tmp_path/'source'
    src.mkdir()
    (src/'target.npy').write_bytes(b'approved numeric fixture')
    (src/'view.npy').symlink_to(src/'target.npy')
    dst=tmp_path/'destination'
    a,b=boxstage.commands('gpuR',str(src),'gpuA',str(dst),['view.npy'],True)
    data=subprocess.check_output([sys.executable,*shlex.split(a[-1])[1:]])
    subprocess.run([sys.executable,*shlex.split(b[-1])[1:]],input=data,check=True,capture_output=True)
    assert not (dst/'view.npy').is_symlink()
    assert (dst/'view.npy').read_bytes()==b'approved numeric fixture'
    outside=tmp_path/'outside.npy'
    outside.write_bytes(b'outside')
    (src/'bad.npy').symlink_to(outside)
    a,_=boxstage.commands('gpuR',str(src),'gpuA',str(tmp_path/'unused'),['bad.npy'],True)
    result=subprocess.run([sys.executable,*shlex.split(a[-1])[1:]],capture_output=True)
    assert result.returncode!=0 and b'unapproved source target' in result.stderr


def test_mini_registration_and_fail_before_sumo(tmp_path,monkeypatch):
    cfg=config(tmp_path)
    cfg['wm']['extra_set']=[]
    arm=tmp_path/'arm'
    arm.mkdir()
    monkeypatch.setattr(control,'fingerprint',lambda cfg:{'source':'s','config':'c'})
    loop.write_atomic(arm/'PREARM.json',{'status':'PASS','source':'s','config':'c'})
    calls=[]
    monkeypatch.setattr(loop,'run_phase',lambda cfg,name:calls.append(name) or 2)
    with pytest.raises(loop.Refused,match='mini policy gate failed'):
        control.mini(cfg,arm)
    one=loop.read_json(arm/'mini/CONFIG.json')
    assert loop.collect_sim_s(one)==1922
    assert fast.cases(one,[968142,968143])==[
        {'index':0,'seed':968142,'demand':'1900','penetration':'0.05','box':'cpuP'},
        {'index':1,'seed':968143,'demand':'1900','penetration':'0.4','box':'cpuQ'}]
    assert calls==['policy'] and loop.load_state(cfg)['sim_s_charged']==0


def test_mini_requires_matching_configuration(tmp_path,monkeypatch):
    cfg=config(tmp_path)
    loop.write_atomic(tmp_path/'PREARM.json',{'status':'PASS','source':'old','config':'old'})
    monkeypatch.setattr(control,'fingerprint',lambda cfg:{'source':'new','config':'new'})
    with pytest.raises(loop.Refused,match='new configuration'):
        control.mini(cfg,tmp_path)


def test_mini_end_to_end_orders_phases_and_charges_before_dispatch(tmp_path,monkeypatch):
    cfg=config(tmp_path)
    cfg['wm']['extra_set']=[]
    root=Path(cfg['loop_root'])
    (root/'seed_ledger.jsonl').write_text('')
    arm=tmp_path/'arm'
    arm.mkdir()
    monkeypatch.setattr(control,'fingerprint',lambda cfg:{'source':'s','config':'c'})
    loop.write_atomic(arm/'PREARM.json',{'status':'PASS','source':'s','config':'c'})
    calls=[]
    def runner(one,state,ph,where):
        name=loop._phase(state)['name']
        calls.append(name)
        if name=='policy':
            state['policy']={'checkpoint':'eligible','sha256':'p'}
        if name=='collect':
            assert loop.load_state(cfg)['sim_s_charged']==1922
            assert loop.load_state(cfg)['seeds_used']==[968142,968143]
        return {'ok':True}
    for name in ('policy','collect','convert','wm_ft'):
        monkeypatch.setitem(loop.RUNNERS,name,runner)
    control.mini(cfg,arm,policy_only=True)
    assert calls==['policy'] and loop.load_state(cfg)['sim_s_charged']==0
    loop.write_atomic(arm/'SHARED_SMOKE_HASHES_APPENDED.json',{'bindings_sha256':loop.sha256(arm/'SMOKE_BINDINGS.json')})
    control.mini(cfg,arm)
    assert calls==['policy','collect','convert','wm_ft']
    assert loop.read_json(arm/'MINI_PASS.json')['status']=='PASS'


def test_registered_mechanics_is_exactly_two_worlds_no_efficacy_claim(tmp_path,monkeypatch):
    cfg=config(tmp_path)
    cfg['collect'].update(worlds_per_cycle=2,demands='1900',penetrations='0.05,0.4',pen_weights='1,1',
                          branch_points=1,branch_first=360,branch_horizon=300,warmup=300)
    reg=loop.read_json(cfg['registration'])
    reg['seed_block']=[968142,968143]
    loop.write_atomic(Path(cfg['registration']),reg)
    state=loop.read_json(loop.state_path(cfg))
    state['registration_sha256']=loop.sha256(cfg['registration'])
    state['policy']={'checkpoint':'native','sha256':'p'}
    gate={'wm_sha256':[state['accepted_wm']['sha256']],
          'wm_config_sha256':[loop.sha256(state['accepted_wm']['config'])]}
    module=ModuleType('wms.control.nativeinit')
    module.verify_collection_gate=lambda policy:gate
    monkeypatch.setitem(sys.modules,'wms.control.nativeinit',module)
    monkeypatch.setattr(loop,'fast_module',lambda:fast)
    calls=[]
    def world(loop_,cfg_,snapshot,ph,root,item,box,attempt):
        assert loop.read_json(loop.state_path(cfg))['sim_s_charged']==1922
        calls.append((item['seed'],box))
        return {'collect':{'branch_collisions':0,'branch_teleports':0}}
    monkeypatch.setattr(fast,'_world',world)
    result=control.mechanics_collect(cfg,state,{'attempt':1},Path(cfg['loop_root']))
    assert sorted(calls)==[(968142,'cpuP'),(968143,'cpuQ')]
    assert result['efficacy_certification'] is False
    with pytest.raises(loop.Refused,match='already attempted'):
        control.mechanics_collect(cfg,state,{'attempt':2},Path(cfg['loop_root']))
