"""V2V accounting, late-rehearsal admission, and all-arm hook contracts."""
from unittest.mock import Mock
import pytest
from . import v2v_live as live, v2v_runner as runner


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


def test_hooks_keep_bank_readers_and_all_enabled_policy_paths(tmp_path):
    from . import replay_bank as rb
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
