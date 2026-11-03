"""V2V accounting, late-rehearsal admission, and all-arm hook contracts."""
import sys
import types
from pathlib import Path
from unittest.mock import Mock
import pytest
from . import v2v_live as live, v2v_runner as runner
from .test_v2v_launch3_cpu import deployed  # noqa: F401 (real launch3 preflight fixture)

SMOKE_SIM_S=24*1200                 # one real-one-update smoke arm (24 episodes x 1200 sim-s)


def write_charges(root,entries):
    """Replace the authoritative ledger file with the given charge entries."""
    import json
    path=Path(root)/'CHARGES.jsonl'
    path.write_text(''.join(json.dumps(e,sort_keys=True)+'\n' for e in entries),encoding='utf-8')


@pytest.fixture
def charges_ledger(monkeypatch):
    """Windows has no flock; read the authoritative ledger directly (test_v2v_launch3_cpu pattern)."""
    import json
    from . import storage
    def ledger(r):
        p=Path(r)/'CHARGES.jsonl'
        return [json.loads(line) for line in p.read_text(encoding='utf-8').splitlines()] if p.exists() else []
    monkeypatch.setattr(storage,'ledger',ledger)


def watch_authority(tmp_path,root,*,binding_path,binding_sha256,control,mover):
    """Register LAUNCH3_AUTHORITY.json plus the legacy boundary proofs, as a real watcher root."""
    from .protocol import sha
    from . import storage
    legacy=tmp_path/'legacy';legacy.mkdir(parents=True,exist_ok=True)
    checkpoints={}
    for i,a in enumerate(('B','R')):
        p=legacy/f'{a}_checkpoint_020.pt';p.write_bytes(a.encode())
        checkpoints[a]=dict(path=str(p),update=20,sha256=sha(p))
    record=dict(binding_path=str(binding_path),binding_sha256=binding_sha256,
        control_root=str(control),mover_root=str(mover),
        legacy_boundary=dict(owners={a:dict(pid=i+1) for i,a in enumerate(('B','R'))},
            checkpoints=checkpoints))
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    storage.write(root/'LAUNCH3_AUTHORITY.json',record)
    (root/'AUTHORITY.md').write_text(sha(root/'LAUNCH3_AUTHORITY.json'))
    return record


@pytest.fixture
def fake_fcntl(monkeypatch):
    monkeypatch.setitem(sys.modules,'fcntl',
        types.SimpleNamespace(LOCK_EX=0,LOCK_NB=0,flock=lambda *a:None))


def binding(tmp_path):
    cfg=runner.prepare(dict(budget=dict(total=17471840,phases={})),root=tmp_path/'new',
        legacy_root=tmp_path/'old',source_root=tmp_path/'source',
        common_checkpoint_sha256='a'*64,v2v_dim=24)
    cfg.update(launch_authorized=True,phase_funding_authorized=True,
        launch_amendment_sha256='b'*64,funding_amendment_sha256='c'*64)
    return cfg


def test_smoke_exception_cannot_authorize_full_training(tmp_path):
    cfg=binding(tmp_path)
    release=dict(qualified=True,linux_cpu_pass=True,zero_init_parity=True,
        default_off_byte_identical=True,stub=False,v2v_dim=24)
    hooks={k:Mock() for k in runner.REQUIRED_HOOKS}
    with pytest.raises(ValueError,match='one-update smokes'):runner.admit(cfg,release,hooks)
    cfg.update(qualification_stage='one_update_smoke',smoke_authority_sha256='d'*64,
        rehearsal_late_60min=True,stop_after_update=1,A_stop_after_update=1)
    runner.admit(cfg,release,hooks)
    cfg['stop_after_update']=24
    with pytest.raises(ValueError,match='one-update smoke only'):runner.admit(cfg,release,hooks)
    for hook in hooks.values():hook.assert_not_called()


def test_shared_ledger_and_lock_are_both_required(tmp_path):
    cfg=binding(tmp_path)
    with pytest.raises(ValueError,match='ledger AND its lock'):
        live.accounting(tmp_path/'new',cfg)
    cfg['accounting_root']=str(tmp_path/'new')
    cfg['budget']['total']=17500001
    with pytest.raises(ValueError,match='cap/allocation'):live.accounting(tmp_path/'new',cfg)


def test_hooks_keep_bank_readers_and_all_enabled_policy_paths(tmp_path, monkeypatch):
    from . import replay_bank as rb, bank_imitation
    monkeypatch.setattr(rb, 'confirmations', rb.confirmations)    # restored after hooks() installs
    monkeypatch.setattr(bank_imitation, 'run', bank_imitation.run)  # audit M4: hooks() rebinds it
    from .v2v_runtime import Policy
    cfg=binding(tmp_path);hooks=live.hooks(tmp_path/'new',cfg)
    assert set(hooks)==set(runner.REQUIRED_HOOKS)
    assert hooks['bank_pairs'] is rb.bank_pairs and hooks['bank_quality'] is rb.bank_quality
    assert hooks['policy_factory'] is Policy
    hooks['configure'](tmp_path/'new','B',cfg)
    with pytest.raises(ValueError,match='own-policy episodes'):
        hooks['configure'](tmp_path/'new','B',dict(cfg,sumo_verify=dict(k=8)))


def test_resource_gate_never_reduces_ten_gib_launch_threshold(tmp_path,monkeypatch):
    import shutil
    from types import SimpleNamespace
    from . import warm_health
    monkeypatch.setattr(warm_health,'assert_running',lambda *args:None)
    monkeypatch.setattr(shutil,'disk_usage',lambda p:SimpleNamespace(free=10*1024**3-1))
    assert not live.resource_gate(tmp_path,{})['ready']
    monkeypatch.setattr(shutil,'disk_usage',lambda p:SimpleNamespace(free=10*1024**3))
    assert live.resource_gate(tmp_path,{})['ready']


def test_operator_remote_payloads_compile_before_dispatch(tmp_path,monkeypatch):
    from . import v2v_launch_ops as operator
    import json
    calls=[]
    def remote(host,code,**kwargs):
        compile(code,'<v2v-operator-payload>','exec');calls.append(code)
        return json.dumps(dict(prepared=True))
    monkeypatch.setattr(operator,'remote',remote)
    monkeypatch.setattr(operator,'read',lambda p:dict())
    monkeypatch.setattr(operator,'sha',lambda p:'a'*64)
    monkeypatch.setattr(operator.ops,'EVIDENCE',tmp_path)
    operator.prepare();operator.launch()
    assert len(calls)==2 and 'SUMO_HOME' in calls[1]
    assert 'ACCOUNTING.claim' in calls[0] and 'B_UPDATE_000.json' in calls[0]


def test_startup_never_reopens_assignment_named_paths_and_pins_receipt(tmp_path,monkeypatch):
    from . import storage,protocol
    root=tmp_path/'root';root.mkdir();(root/'AUTHORITY.md').write_text('registered')
    source=tmp_path/'source'
    runtime=dict(config=dict(source_root=str(source)),amendment_sha256='registered',cpu_result_sha256='cpu')
    cpu=dict(pass_=True,sim_s=0,skips=0,source_hashes={
        'pytools/traffic_shepherd/mixed_assignment.py':'unopened',
        'runtime_src/traffic_shepherd/mixed_assignment.py':'unopened',
        'pytools/tools/coopmwm/v2v_live.py':'code'})
    monkeypatch.setattr(storage,'read',lambda p:runtime if p.name=='V2V_RUNTIME.json' else cpu)
    seen=[]
    def sha(path):
        assert 'assignment' not in str(path)
        seen.append(str(path));return 'cpu' if path.name=='LS_CPU_RESULT.json' else 'code'
    monkeypatch.setattr(protocol,'sha',sha)
    monkeypatch.setattr(live,'__file__',str(source/'pytools/tools/coopmwm/v2v_live.py'))
    assert live.configuration(root)==runtime and len(seen)==2
    runtime['cpu_result_sha256']='changed'
    with pytest.raises(ValueError,match='receipt changed'):live.configuration(root)


# ---------------------------------------------------------------- restarted watcher remaining quote

NS='v2v1'


def smoke_ids(arm,count,first=986000):
    return [f'{NS}_warm_u001_{arm}_{first+i}' for i in range(count)]


def smoke_charges(root,arm,count,*,closed=True,first=986000):
    """Ledger rows of one arm's update-1 smoke episodes plus their job directories."""
    rows=[]
    for ident in smoke_ids(arm,count,first):
        job=Path(root)/'jobs'/ident;job.mkdir(parents=True,exist_ok=True)
        if closed:
            (job/'CLOSE.json').write_text('{}');(job/'RESULT.json').write_text('{}')
        rows.append(dict(id=ident,phase='v2v_train',sim_s=1200))
    return rows


def test_remaining_quote_credits_only_closed_reusable_smokes_of_this_root(tmp_path,charges_ledger):
    from . import v2v_launch3 as launch
    root=tmp_path/'run';root.mkdir();cfg=dict(job_namespace=NS)
    full=launch.V2V_PHASES['v2v_train']
    # fresh root: nothing charged -> full quote (byte-identical watcher start)
    assert live.remaining_phase_costs(root,cfg) is None
    write_charges(root,[])
    assert live.remaining_phase_costs(root,cfg) is None
    # the documented restart: smoke A and B update 1 charged and closed, R not run
    rows=smoke_charges(root,'A',24)+smoke_charges(root,'B',24)
    write_charges(root,rows+[dict(id='other_kind_x',phase='bank_verification_ext',sim_s=500)])
    remaining=live.remaining_phase_costs(root,cfg)
    assert remaining==dict(launch.V2V_PHASES,v2v_train=full-2*SMOKE_SIM_S)
    assert remaining['v2v_train']==2016000==SMOKE_SIM_S+3*23*24*1200
    # no namespace -> no credit
    assert live.remaining_phase_costs(root,{}) is None
    # charged but NOT closed (void/partial) episodes are never credited
    write_charges(root,rows+smoke_charges(root,'R',3,closed=False))
    assert live.remaining_phase_costs(root,cfg)['v2v_train']==full-2*SMOKE_SIM_S
    # foreign / non-u001 / unexpected-amount charges are not credited
    write_charges(root,rows+[dict(id='v2v1_warm_u002_A_986000',phase='v2v_train',sim_s=1200),
        dict(id='other_warm_u001_A_986000',phase='v2v_train',sim_s=1200),
        dict(id='smoke_A',phase='v2v_train',sim_s=SMOKE_SIM_S)])
    assert live.remaining_phase_costs(root,cfg)['v2v_train']==full-2*SMOKE_SIM_S
    bad=smoke_charges(root,'R',1,first=987000);bad[0]['sim_s']=2400
    write_charges(root,rows+bad)
    assert live.remaining_phase_costs(root,cfg)['v2v_train']==full-2*SMOKE_SIM_S
    # duplicate charge or any charged diagnostic/test phase -> no credit (full quote)
    write_charges(root,rows+rows[:1])
    assert live.remaining_phase_costs(root,cfg) is None
    write_charges(root,rows+[dict(id='v2vtest008_A_1',phase='v2v_interim',sim_s=1200)])
    assert live.remaining_phase_costs(root,cfg) is None
    # more than one registered smoke per arm is refused outright
    write_charges(root,rows+smoke_charges(root,'A',1,first=990000))
    with pytest.raises(ValueError,match='one registered smoke per arm'):
        live.remaining_phase_costs(root,cfg)


def test_watch_restarted_after_smokes_passes_exact_remaining_preflight(tmp_path,monkeypatch,deployed):
    from unittest.mock import patch
    from . import replay_start
    from . import v2v_launch3 as launch
    d=deployed;root=d.root
    cfg=dict(v2v_intent=True,gpu_deadline=10**12,legacy_root=str(tmp_path/'legacy'),job_namespace=NS)
    watch_authority(tmp_path,root,binding_path=d.path,
        binding_sha256=d.kwargs['binding_sha256'],control=d.control,mover=d.mover)
    with patch.object(replay_start,'alive',return_value=False):
        # fresh ledger: the full registered quote passes, unchanged
        assert live.remaining_phase_costs(root,cfg) is None
        fresh=live.watch_gate(root,cfg)
        assert fresh['ready'] and fresh['budget']['phase_costs']==dict(launch.V2V_PHASES)
        # the bug: after smoke A+B charged update 1 the full quote refuses although
        # the remaining work (R smoke + 3x23 train updates) fits exactly
        rows=smoke_charges(root,'A',24)+smoke_charges(root,'B',24)
        write_charges(root,rows)
        with pytest.raises(launch.Refused,match='budget phase shortfall: v2v_train'):
            live.launch3_check(root,cfg)
        # the fix: the restarted watcher's gate passes the exact remaining quote
        receipt=live.watch_gate(root,cfg)
        assert receipt['ready'] and receipt['budget']['phase_costs']['v2v_train']==2016000
        assert receipt['budget']['quote_total_sim_s']==sum(launch.V2V_PHASES.values())-2*SMOKE_SIM_S
        # every v2v_train sim-s is committed exactly: one more uncredited charge refuses
        write_charges(root,rows+[dict(id='void_x',phase='v2v_train',sim_s=1200)])
        with pytest.raises(launch.Refused,match='budget phase shortfall: v2v_train'):
            live.watch_gate(root,cfg)
        # an overdrawn ledger is still refused
        write_charges(root,rows+[dict(id='over',phase='v2v_train',sim_s=launch.V2V_PHASES['v2v_train'])])
        with pytest.raises(launch.Refused,match='overdrawn'):
            live.watch_gate(root,cfg)


def test_watch_gate_wiring_full_quote_and_watch_uses_it(tmp_path,monkeypatch,charges_ledger,fake_fcntl):
    from . import replay_start,v2v_launch3
    root=tmp_path/'run';root.mkdir()
    watch_authority(tmp_path,root,binding_path=tmp_path/'binding.json',
        binding_sha256='a'*64,control=tmp_path/'control',mover=tmp_path/'mover')
    cfg=dict(v2v_intent=True,gpu_deadline=10**12,legacy_root=str(tmp_path/'legacy'),job_namespace=NS)
    seen=[]
    monkeypatch.setattr(v2v_launch3,'wait_for_preflight',
        lambda *a,**k:seen.append(k) or {'ready':True})
    monkeypatch.setattr(replay_start,'alive',lambda receipt:False)
    monkeypatch.setattr(live,'configuration',lambda r:dict(config=cfg))
    # fresh ledger: watch's own gate keeps the full quote (phase_costs=None)
    assert live.watch_gate(root,cfg)=={'ready':True}
    assert len(seen)==1 and seen[0]['phase_costs'] is None
    write_charges(root,smoke_charges(root,'A',24)+smoke_charges(root,'B',24))
    assert live.watch_gate(root,cfg)=={'ready':True}
    assert seen[1]['phase_costs']['v2v_train']==2073600-2*SMOKE_SIM_S==2016000
    # child spawns keep their registered unit cost regardless of the restart quote
    assert live.launch3_check(root,cfg,'smoke','R')=={'ready':True}
    assert seen[2]['phase_costs']=={'v2v_train':SMOKE_SIM_S}
    # watch() itself goes through watch_gate before spawning any child
    calls=[]
    def gate(r,c):
        calls.append((r,c));raise ValueError('stop before smokes')
    monkeypatch.setattr(live,'watch_gate',gate)
    monkeypatch.setattr(replay_start,'owner',lambda:dict(pid=1,start_ticks='1',boot_id='b'))
    import subprocess
    monkeypatch.setattr(subprocess,'Popen',lambda *a,**k:pytest.fail('Popen without watch gate'))
    with pytest.raises(ValueError,match='stop before smokes'):
        live.watch(root)
    assert calls==[(root,cfg)]
