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

def test_remaining_quote_is_allocation_minus_ledger_spent(tmp_path,charges_ledger):
    from . import v2v_launch3 as launch
    root=tmp_path/'run';root.mkdir()
    # fresh root: nothing charged -> full quote (byte-identical watcher start)
    assert live.remaining_phase_costs(root) is None
    write_charges(root,[])
    assert live.remaining_phase_costs(root) is None
    # after smoke A+B charged their update 1 (the restart that used to refuse):
    # remaining = registered allocation - what the ledger already charged
    write_charges(root,[dict(id='v2v1_warm_u001_A_'+s,phase='v2v_train',sim_s=1200) for s in ('a','b')
        ]+[dict(id='v2v1_warm_u001_B_'+s,phase='v2v_train',sim_s=1200) for s in ('a','b')])
    assert live.remaining_phase_costs(root)==dict(
        launch.V2V_PHASES,v2v_train=launch.V2V_PHASES['v2v_train']-4*1200)
    # completed smokes: exact remaining v2v_train for the documented restart case
    write_charges(root,[dict(id='smoke_A',phase='v2v_train',sim_s=SMOKE_SIM_S),
                        dict(id='smoke_B',phase='v2v_train',sim_s=SMOKE_SIM_S)])
    remaining=live.remaining_phase_costs(root)
    assert remaining['v2v_train']==launch.V2V_PHASES['v2v_train']-2*SMOKE_SIM_S==2016000
    assert set(remaining)==set(launch.V2V_PHASES) and all(c>0 for c in remaining.values())
    # an exhausted (or overdrawn) phase is omitted: costs are never zero/negative
    write_charges(root,[dict(id='smoke_A',phase='v2v_train',sim_s=SMOKE_SIM_S),
        dict(id='smoke_B',phase='v2v_train',sim_s=SMOKE_SIM_S),
        dict(id='interim',phase='v2v_interim',sim_s=launch.V2V_PHASES['v2v_interim'])])
    remaining=live.remaining_phase_costs(root)
    assert 'v2v_interim' not in remaining and all(c>0 for c in remaining.values())
    write_charges(root,[dict(id='interim',phase='v2v_interim',sim_s=launch.V2V_PHASES['v2v_interim']+1)])
    assert 'v2v_interim' not in live.remaining_phase_costs(root)
    # fully charged slate -> None keeps the (then refusing) full quote; never an empty dict
    write_charges(root,[dict(id=name,phase=name,sim_s=amount) for name,amount in launch.V2V_PHASES.items()])
    assert live.remaining_phase_costs(root) is None
    # malformed entries are ignored here; requote re-checks the ledger authoritatively
    write_charges(root,[dict(id='smoke_A',phase='v2v_train',sim_s=SMOKE_SIM_S),
        dict(id='bad',phase='v2v_train',sim_s=-1),dict(id='alsobad',phase='v2v_train',sim_s=True)])
    assert live.remaining_phase_costs(root)['v2v_train']==launch.V2V_PHASES['v2v_train']-SMOKE_SIM_S


def test_watch_restarted_after_smokes_passes_exact_remaining_preflight(tmp_path,monkeypatch,deployed):
    from unittest.mock import patch
    from . import replay_start
    from . import v2v_launch3 as launch
    d=deployed;root=d.root
    cfg=dict(v2v_intent=True,gpu_deadline=10**12,legacy_root=str(tmp_path/'legacy'))
    watch_authority(tmp_path,root,binding_path=d.path,
        binding_sha256=d.kwargs['binding_sha256'],control=d.control,mover=d.mover)
    with patch.object(replay_start,'alive',return_value=False):
        # fresh ledger: the full registered quote is the exact remaining quote and passes
        assert live.remaining_phase_costs(root) is None
        fresh=live.launch3_check(root,cfg)
        assert fresh['ready'] and fresh['budget']['phase_costs']==dict(launch.V2V_PHASES)
        # the bug: after smoke A+B charged update 1 the full quote refuses although
        # the remaining work (R smoke + 3x23 train updates) fits exactly
        write_charges(root,[dict(id='smoke_A',phase='v2v_train',sim_s=SMOKE_SIM_S),
                            dict(id='smoke_B',phase='v2v_train',sim_s=SMOKE_SIM_S)])
        with pytest.raises(launch.Refused,match='budget phase shortfall: v2v_train'):
            live.launch3_check(root,cfg)
        # the fix: pass the exact remaining quote and the same preflight passes
        remaining=live.remaining_phase_costs(root)
        assert remaining['v2v_train']==launch.V2V_PHASES['v2v_train']-2*SMOKE_SIM_S==2016000
        receipt=live.launch3_check(root,cfg,phase_costs=remaining)
        assert receipt['ready'] and receipt['budget']['phase_costs']==remaining
        assert receipt['budget']['quote_total_sim_s']==sum(remaining.values())
        assert receipt['budget']['charged_to_date_sim_s']==2*SMOKE_SIM_S+811640
        # an overdrawn ledger is still refused through the remaining quote
        write_charges(root,[dict(id='smoke_A',phase='v2v_train',sim_s=SMOKE_SIM_S),
            dict(id='smoke_B',phase='v2v_train',sim_s=SMOKE_SIM_S),
            dict(id='over',phase='v2v_train',
                 sim_s=launch.V2V_PHASES['v2v_train']-2*SMOKE_SIM_S+1)])
        over=live.remaining_phase_costs(root)
        assert 'v2v_train' not in over and all(c>0 for c in over.values())
        with pytest.raises(launch.Refused,match='overdrawn'):
            live.launch3_check(root,cfg,phase_costs=over)


def test_watch_gate_wiring_full_quote_and_watch_uses_it(tmp_path,monkeypatch,charges_ledger,fake_fcntl):
    from . import replay_start,v2v_launch3
    root=tmp_path/'run';root.mkdir()
    watch_authority(tmp_path,root,binding_path=tmp_path/'binding.json',
        binding_sha256='a'*64,control=tmp_path/'control',mover=tmp_path/'mover')
    cfg=dict(v2v_intent=True,gpu_deadline=10**12,legacy_root=str(tmp_path/'legacy'))
    seen=[]
    monkeypatch.setattr(v2v_launch3,'wait_for_preflight',
        lambda *a,**k:seen.append(k) or {'ready':True})
    monkeypatch.setattr(replay_start,'alive',lambda receipt:False)
    monkeypatch.setattr(live,'configuration',lambda r:dict(config=cfg))
    # fresh ledger: watch's own gate keeps the full quote (phase_costs=None)
    assert live.watch_gate(root,cfg)=={'ready':True}
    assert len(seen)==1 and seen[0]['phase_costs'] is None
    write_charges(root,[dict(id='smoke_A',phase='v2v_train',sim_s=SMOKE_SIM_S),
                        dict(id='smoke_B',phase='v2v_train',sim_s=SMOKE_SIM_S)])
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
