"""V2V accounting, late-rehearsal admission, and all-arm hook contracts."""
import sys
import types
from pathlib import Path
from unittest.mock import Mock
import pytest
from . import v2v_live as live, v2v_runner as runner
from .test_v2v_launch3_cpu import deployed  # noqa: F401 (real launch3 preflight fixture)

SMOKE_SIM_S=24*1200                 # one real-one-update smoke arm (24 episodes x 1200 sim-s)
SMOKE_EPISODE=1200                   # one registered update-1 smoke/train episode


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


def test_watch_restart_quote_is_practice_aware(tmp_path,charges_ledger):
    """Fix round 2 (d): with practice ON the update-1 smoke charges practice episodes and their
    matched A references too; the restarted watcher's quote covers the practice-aware allocation
    minus exactly the closed update-1 work of this root."""
    from . import v2v_launch3 as launch
    from .practice_starts import practice_count
    root=tmp_path/'run';root.mkdir()
    pcount=practice_count(24,0.25)                                       # 6
    cfg=dict(job_namespace=NS,practice_starts=True,practice_fraction=0.25,practice_repeat_cap=2)
    full=launch.V2V_PHASES['v2v_train']+pcount*2*24*SMOKE_EPISODE        # practice-aware allocation
    rows=[]
    for arm,practice in (('A',0),('B',1),('R',1)):
        rows+=smoke_charges(root,arm,24-practice*pcount)                 # normal smoke episodes
        for i in range(practice*pcount):                                 # practice episodes
            ident=f'{NS}_warm_u001_p{i+1:02d}_{arm}_{986000+i}'
            job=Path(root)/'jobs'/ident;job.mkdir(parents=True,exist_ok=True)
            (job/'CLOSE.json').write_text('{}');(job/'RESULT.json').write_text('{}')
            rows.append(dict(id=ident,phase='v2v_train',sim_s=SMOKE_EPISODE))
        for i in range(practice*pcount):                                 # matched A references
            ident=f'{NS}_warm_u001_{arm}_practiceref_pa{i+1:02d}_A_{986000+i}'
            job=Path(root)/'jobs'/ident;job.mkdir(parents=True,exist_ok=True)
            (job/'CLOSE.json').write_text('{}');(job/'RESULT.json').write_text('{}')
            rows.append(dict(id=ident,phase='v2v_train',sim_s=SMOKE_EPISODE))
    write_charges(root,rows)
    costs=live.remaining_phase_costs(root,cfg)
    assert costs['v2v_train']==full-84*SMOKE_EPISODE==2318400            # every closed u1 work
    assert costs['v2v_train']==full-(24+2*(24-6+6+6))*SMOKE_EPISODE
    # an open practice reference of the smoke is not credited (stays in the quote)
    open_ident=f'{NS}_warm_u001_B_practiceref_pa01_A_986000'
    (Path(root)/'jobs'/open_ident/'CLOSE.json').unlink()
    (Path(root)/'jobs'/open_ident/'RESULT.json').unlink()
    write_charges(root,rows)
    assert live.remaining_phase_costs(root,cfg)['v2v_train']==costs['v2v_train']+SMOKE_EPISODE


# ---------------------------------------------------------------- resume a started root (MAIN 07:35)

import json as _json


class StopLoop(Exception):
    pass


def _ids(u,arm,n,first=986000):
    return [f'{NS}_warm_u{u:03}_{arm}_{first+i}' for i in range(n)]


def _job(root,ident,*,closed=True,result=None,claim_mtime=None):
    import os
    job=Path(root)/'jobs'/ident;job.mkdir(parents=True,exist_ok=True)
    (job/'CLAIM.json').write_text(_json.dumps(dict(id=ident,charged_sim_s=1200)))
    if claim_mtime is not None:os.utime(job/'CLAIM.json',(claim_mtime,claim_mtime))
    if closed:
        (job/'CLOSE.json').write_text(_json.dumps(dict(actual_sim_s=1200.,charged_sim_s=1200)))
        if result is None:(job/'RESULT.json').write_text(_json.dumps(dict(id=ident,steps=[])))
        else:
            from .accel_storage import write
            write(job/'RESULT.json',result)


def started_root(tmp_path,*,sealed=None,current=None):
    """The documented 07:12 state: A sealed u3 and charged its live u4; B sealed u2 with 24
    closed u3 episodes; R sealed u2 with nothing of u3. Every sealed update: 24 closed charges."""
    sealed=sealed or dict(A=3,B=2,R=2)
    current=current or dict(A=('live',24),B=('closed',24),R=('closed',0))
    root=tmp_path/'run';root.mkdir(parents=True,exist_ok=True);rows=[]
    for arm,k in sealed.items():
        for u in range(1,k+1):
            (root/f'{arm}_UPDATE_{u:03}.json').write_text(_json.dumps(dict(update=u,arm=arm,trust=dict(beta=1.,graded=0))))
            (root/f'{arm}_checkpoint_{u:03}.pt').write_bytes(f'{arm}{u}'.encode())
            for ident in _ids(u,arm,24):
                _job(root,ident,claim_mtime=1000.+u*2000);rows.append(dict(id=ident,phase='v2v_train',sim_s=1200))
        kind,n=current[arm]
        for ident in _ids(k+1,arm,n):
            _job(root,ident,closed=kind=='closed',claim_mtime=1000.+(k+1)*2000)
            rows.append(dict(id=ident,phase='v2v_train',sim_s=1200))
    write_charges(root,rows)
    return root,rows


def test_resume_quote_is_exact_remaining_work(tmp_path,charges_ledger):
    from . import v2v_launch3 as launch
    root,rows=started_root(tmp_path);cfg=dict(job_namespace=NS)
    costs,per=live.resume_phase_costs(root,cfg,('A',),('B','R'))
    assert per['A']['sealed_update']==3 and per['A']['credited_episodes']==24     # live u4 owned by A
    assert per['B']['reusable']==_ids(3,'B',24) and per['B']['remaining_episodes']==22*24-24
    assert per['R']['remaining_episodes']==22*24 and per['R']['reusable']==[]
    assert costs['v2v_train']==(21*24-24+22*24-24+22*24)*1200==1814400
    spent=sum(r['sim_s'] for r in rows)
    assert spent+costs['v2v_train']==launch.V2V_PHASES['v2v_train']   # exactly 3 arms x 24 updates
    assert {k:v for k,v in costs.items() if k!='v2v_train'}=={k:v for k,v in launch.V2V_PHASES.items() if k!='v2v_train'}
    # every arm must be either live or resumed, never both
    with pytest.raises(ValueError,match='either live'):live.resume_phase_costs(root,cfg,('A','B'),('B','R'))
    with pytest.raises(ValueError,match='either live'):live.resume_phase_costs(root,cfg,('A',),('B',))
    # a partially closed in-progress update credits only its closed episodes
    root2,_=started_root(tmp_path/'p',current=dict(A=('live',24),B=('closed',10),R=('closed',0)))
    assert live.arm_remaining(root2,cfg,'B',live=False)['remaining_episodes']==22*24-10
    # charged but unclosed episodes of a resumed arm can never be retried: refuse
    root3,_=started_root(tmp_path/'u',current=dict(A=('live',24),B=('open',24),R=('closed',0)))
    with pytest.raises(ValueError,match='not closed'):live.arm_remaining(root3,cfg,'B',live=False)
    # ... while a live arm's in-flight (unclosed) charges are credited, capped at one update
    assert live.arm_remaining(root3,cfg,'B',live=True)['remaining_episodes']==22*24-24
    # the u3-link trap: a stray unsealed checkpoint k+1 (file or link) refuses a resume
    (root/'B_checkpoint_003.pt').write_bytes(b'legacy')
    with pytest.raises(ValueError,match='u3-link trap'):live.arm_remaining(root,cfg,'B',live=False)
    (root/'B_checkpoint_003.pt').unlink()
    # charges beyond k+1 without a sealed k+1 are inconsistent
    write_charges(root,rows+[dict(id=_ids(4,'R',1)[0],phase='v2v_train',sim_s=1200)])
    with pytest.raises(ValueError,match='beyond update 3'):live.arm_remaining(root,cfg,'R',live=False)
    # a sealed update without its own regular checkpoint file is refused
    write_charges(root,rows);(root/'R_checkpoint_002.pt').unlink()
    with pytest.raises(ValueError,match='lacks its own checkpoint'):live.sealed_update(root,'R')


def practice_started_root(tmp_path):
    """A started practice-ON root: A sealed u3 with a live closed u4; B sealed u2 whose live u3
    is fully closed (18 normal + 6 practice episode charges + 6 closed matched references);
    R sealed u2 with nothing of u3."""
    from .practice_starts import practice_count
    pcount=practice_count(24,0.25)                                       # 6
    root=tmp_path/'run';root.mkdir(parents=True,exist_ok=True);rows=[]
    def sealed(arm,u_count,practice=False):
        for u in range(1,u_count+1):
            (root/f'{arm}_UPDATE_{u:03}.json').write_text(_json.dumps(dict(update=u,arm=arm,
                trust=dict(beta=1.,graded=0))))
            (root/f'{arm}_checkpoint_{u:03}.pt').write_bytes(f'{arm}{u}'.encode())
            for ident in _ids(u,arm,24):
                _job(root,ident,claim_mtime=1000.+u*2000)
                rows.append(dict(id=ident,phase='v2v_train',sim_s=1200))
    sealed('A',3);sealed('B',2);sealed('R',2)
    for ident in _ids(4,'A',24):                                         # the live A update
        _job(root,ident,claim_mtime=9000.)
        rows.append(dict(id=ident,phase='v2v_train',sim_s=1200))
    normal=_ids(3,'B',18)
    practice_ids=[f'{NS}_warm_u003_p{i+1:02d}_B_{986000+i}' for i in range(pcount)]
    refs=[f'{NS}_warm_u003_B_practiceref_pa{i+1:02d}_A_{986000+i}' for i in range(pcount)]
    for ident in normal+practice_ids+refs:
        _job(root,ident,claim_mtime=7000.)
        rows.append(dict(id=ident,phase='v2v_train',sim_s=1200))
    write_charges(root,rows)
    return root,rows,pcount


def test_resume_quote_counts_practice_and_reference_charges(tmp_path,charges_ledger):
    """Fix round 2 (d): with practice ON, _train_rows counts the practice episode ids
    (_pNN_ slots, _practice_ replacements) and the matched practice reference ids
    (..._<arm>_practiceref_paNN_A_...); a resumed arm's quote covers its remaining episodes
    plus the registered practice A-reference cost of its remaining updates, crediting the
    closed reference charges of the in-progress update; the algebra stays within the
    practice-aware v2v_train allocation."""
    from . import v2v_launch3 as launch
    from .practice_starts import practice_count
    root,rows,pcount=practice_started_root(tmp_path)
    cfg=dict(job_namespace=NS,practice_starts=True,practice_fraction=0.25,practice_repeat_cap=2,
        budget=dict(phases=dict(v2v_train=launch.V2V_PHASES['v2v_train']
                                +launch.practice_reference_quote(0.25))))
    costs,per=live.resume_phase_costs(root,cfg,('A',),('B','R'))
    assert per['A']['credited_episodes']==24 and per['A']['remaining_reference_sim_s']==0
    # B: the 6 practice charges count as its (k+1) episodes like any other episode; the 6
    # closed references are credited against the remaining reference cost
    assert per['B']['charged_current_update']==24 and per['B']['credited_episodes']==24
    assert per['B']['credited_references']==6
    assert per['B']['remaining_episodes']==22*24-24
    assert per['B']['remaining_reference_sim_s']==(22*pcount-6)*1200
    assert per['B']['remaining_sim_s']==((22*24-24)+(22*pcount-6))*1200
    # R: charged nothing of u3 -> the full reference upper bound for its remaining updates
    assert per['R']['credited_references']==0
    assert per['R']['remaining_sim_s']==(22*24+22*pcount)*1200
    assert costs['v2v_train']==sum(p['remaining_sim_s'] for p in per.values())
    spent=sum(r['sim_s'] for r in rows)
    assert spent+costs['v2v_train']<=launch.V2V_PHASES['v2v_train']+launch.practice_reference_quote(0.25)
    # practice OFF keeps the old episodes-only quote (references are never credited then)
    costs_off,_=live.resume_phase_costs(root,dict(job_namespace=NS),('A',),('B','R'))
    assert costs_off['v2v_train']==(21*24-24+22*24-24+22*24)*1200==1814400
    # an unclosed reference of a resumed arm can never be retried: refuse
    root2,_,_=practice_started_root(tmp_path/'open')
    (root2/'jobs'/f'{NS}_warm_u003_B_practiceref_pa01_A_986000'/'CLOSE.json').unlink()
    with pytest.raises(ValueError,match='practice reference .* is not closed'):
        live.arm_remaining(root2,cfg,'B',live=False)
    # a reference charge beyond k+1 without a sealed k+1 is inconsistent
    root3,rows3,_=practice_started_root(tmp_path/'ahead')
    write_charges(root3,rows3+[dict(id=f'{NS}_warm_u004_B_practiceref_pa01_A_1',
                                    phase='v2v_train',sim_s=1200)])
    with pytest.raises(ValueError,match='reference charges beyond update 3'):
        live.arm_remaining(root3,cfg,'B',live=False)
    # an unexpected reference amount refuses
    root4,rows4,_=practice_started_root(tmp_path/'bad')
    write_charges(root4,rows4+[dict(id=f'{NS}_warm_u003_B_practiceref_pa07_A_1',
                                    phase='v2v_train',sim_s=2400)])
    with pytest.raises(ValueError,match='unexpected practice reference charge'):
        live.arm_remaining(root4,cfg,'B',live=False)


def test_adopt_preflight_is_real_launch3_exact_and_overdrawn_refused(tmp_path,monkeypatch,deployed):
    from unittest.mock import patch
    from . import replay_start
    from . import v2v_launch3 as launch
    d=deployed;root,rows=started_root(tmp_path)
    assert root==d.root
    cfg=dict(v2v_intent=True,gpu_deadline=10**12,legacy_root=str(tmp_path/'legacy'),job_namespace=NS)
    watch_authority(tmp_path,root,binding_path=d.path,
        binding_sha256=d.kwargs['binding_sha256'],control=d.control,mover=d.mover)
    original=launch.wait_for_preflight
    with patch.object(replay_start,'alive',return_value=False):
        # the old restart credit (u001 smokes only) refuses this started root ...
        with pytest.raises(launch.Refused,match='budget phase shortfall: v2v_train'):
            live.watch_gate(root,cfg)
        # ... the adopt quote is exact and passes the unchanged launch3 preflight
        receipt,costs,per=live.adopt_gate(root,cfg,('A',),('B','R'))
        assert receipt['ready'] and receipt['budget']['phase_costs']==costs
        assert costs['v2v_train']==1814400
        # per-child resume quotes are the arm's own exact remaining work
        seen=[]
        monkeypatch.setattr(launch,'wait_for_preflight',lambda *a,**k:seen.append(k['phase_costs']) or original(*a,**k))
        assert live.launch3_check(root,cfg,'resume','B')['ready']
        assert live.launch3_check(root,cfg,'resume','R')['ready']
        assert seen==[{'v2v_train':(22*24-24)*1200},{'v2v_train':22*24*1200}]
        monkeypatch.setattr(launch,'wait_for_preflight',original)
        # one uncredited train charge more than the structure allows: refused, never re-quoted down
        write_charges(root,rows+[dict(id='void_x',phase='v2v_train',sim_s=1200)])
        with pytest.raises(launch.Refused,match='budget phase shortfall: v2v_train'):
            live.adopt_gate(root,cfg,('A',),('B','R'),attempts=2)
        # an overdrawn ledger is still refused
        write_charges(root,rows+[dict(id='over',phase='v2v_train',sim_s=launch.V2V_PHASES['v2v_train'])])
        with pytest.raises(launch.Refused,match='overdrawn'):
            live.adopt_gate(root,cfg,('A',),('B','R'),attempts=1)
        # the live A seals u4 and charges u5 between quote and ledger read: stale quote refused,
        # re-quoted from the new ledger only when retries remain
        calls=[]
        def racing(*a,**k):
            calls.append(k['phase_costs'])
            if len(calls)==1:
                (root/'A_UPDATE_004.json').write_text('{}');(root/'A_checkpoint_004.pt').write_bytes(b'A4')
                write_charges(root,rows+[dict(id=i,phase='v2v_train',sim_s=1200) for i in _ids(5,'A',24)])
            return original(*a,**k)
        monkeypatch.setattr(launch,'wait_for_preflight',racing)
        def reset():
            calls.clear();write_charges(root,rows)
            for name in ('A_UPDATE_004.json','A_checkpoint_004.pt'):
                if (root/name).exists():(root/name).unlink()
        reset()
        with pytest.raises(launch.Refused,match='budget phase shortfall: v2v_train'):
            live.adopt_gate(root,cfg,('A',),('B','R'),attempts=1)
        reset()
        receipt,costs,_=live.adopt_gate(root,cfg,('A',),('B','R'),attempts=3)
        assert receipt['ready'] and len(calls)==2
        assert calls[1]['v2v_train']==calls[0]['v2v_train']-24*1200


def _failed_attempt(root):
    """Dead watcher + dead B/R children + the B guided stop exactly as left at 07:12."""
    error="FileExistsError(17, 'File exists')"
    dead=dict(pid=4242,start_ticks='1',boot_id='b')
    for name in ('WATCH_OWNER.json','WATCH_DISPATCH.json','train_B_DISPATCH.json','train_R_DISPATCH.json'):
        (root/name).write_text(_json.dumps(dead))
    (root/'train_A_DISPATCH.json').write_text(_json.dumps(dict(pid=5001,start_ticks='1',boot_id='b',label='train_A')))
    for name,value in (('watch_FAILED.json',dict(error='child exited')),('train_B_FAILED.json',dict(error=error)),
                       ('train_R_FAILED.json',dict(error="ValueError('durable guided engineering STOP')")),
                       ('WARM_B_FAILED_V2V_LIVE.json',dict(error=error)),
                       ('WARM_R_FAILED_V2V_LIVE.json',dict(error="ValueError('durable guided engineering STOP')")),
                       ('WARM_GUIDED_STOP.json',dict(arm='B',error=error)),
                       ('B_VIABILITY_003.json',dict(update=3)),('B_SEARCH_LEARNING_003.json',dict(update=3)),
                       ('B_SAFETY_COMPOSITE_003.json',dict(release=True)),('V2V_WATCH_STATUS.json',{}),
                       ('V2V_STARTED.json',dict(resumed_after_smoke=2)),
                       ('V2V_REAL_SMOKES_PASS.json',dict(arms=['A','B','R'],pass_=True))):
        (root/name).write_text(_json.dumps(value))
    for name in ('WATCH.log','train_B.log','train_R.log'):(root/name).write_text('log')


def test_resume_prepare_moves_dead_records_with_receipt_and_refuses_live_or_pause(tmp_path,monkeypatch,charges_ledger):
    from . import replay_start
    from .protocol import sha
    root,_=started_root(tmp_path);_failed_attempt(root)
    monkeypatch.setattr(replay_start,'alive',lambda d:d.get('pid')==5001)
    before={p.name:sha(p) for p in root.iterdir() if p.is_file()}
    receipt=live.resume_prepare(root,'r1',['R','B'])
    dest=root/'resume_r1_archive'
    moved={m['name'] for m in receipt['moved']}
    assert moved=={'WATCH_OWNER.json','WATCH_DISPATCH.json','watch_FAILED.json','WATCH.log','V2V_WATCH_STATUS.json',
        'train_B_DISPATCH.json','train_B_FAILED.json','train_B.log','train_R_DISPATCH.json','train_R_FAILED.json',
        'train_R.log','WARM_B_FAILED_V2V_LIVE.json','WARM_R_FAILED_V2V_LIVE.json','WARM_GUIDED_STOP.json',
        'B_VIABILITY_003.json','B_SEARCH_LEARNING_003.json'}
    for m in receipt['moved']:
        assert not (root/m['name']).exists() and sha(dest/m['name'])==m['sha256']==before[m['name']]
    assert (root/'train_A_DISPATCH.json').exists() and (root/'B_SAFETY_COMPOSITE_003.json').exists()
    assert receipt['resume_arms']==['B','R'] and receipt['sealed_updates']=={'B':2,'R':2}
    assert (dest/'ARCHIVE_RECEIPT.json').is_file()
    with pytest.raises(ValueError,match='already exists'):live.resume_prepare(root,'r1',['B','R'])
    # refusals: a live watcher, a foreign guided stop, a health pause, a bad label
    root2,_=started_root(tmp_path/'x');_failed_attempt(root2)
    monkeypatch.setattr(replay_start,'alive',lambda d:d.get('pid') in (5001,4242))
    with pytest.raises(ValueError,match='live V2V watcher'):live.resume_prepare(root2,'r1',['B','R'])
    monkeypatch.setattr(replay_start,'alive',lambda d:d.get('pid')==5001)
    (root2/'WARM_GUIDED_STOP.json').write_text(_json.dumps(dict(arm='B',error='something else')))
    with pytest.raises(ValueError,match='not the record'):live.resume_prepare(root2,'r1',['B','R'])
    (root2/'WARM_GUIDED_STOP.json').write_text(_json.dumps(dict(arm='B',error="FileExistsError(17, 'File exists')")))
    (root2/'WARM_R_FAILED_V2V_LIVE.json').write_text(_json.dumps(dict(error="HealthPause('registered health pause SAFETY')")))
    with pytest.raises(ValueError,match='health pause'):live.resume_prepare(root2,'r1',['B','R'])
    with pytest.raises(ValueError,match='attempt label'):live.resume_prepare(root2,'../x',['B'])
    assert not (root2/'resume_r1_archive').exists()
    # an unreleased registered pause still refuses after the move (assert_running)
    root3,_=started_root(tmp_path/'y');_failed_attempt(root3)
    (root3/'PAUSE_SAFETY.json').write_text(_json.dumps(dict(rule='SAFETY')))
    from .warm_health import HealthPause
    with pytest.raises(HealthPause):live.resume_prepare(root3,'r1',['B','R'])


def _adopt_env(tmp_path,monkeypatch,root):
    """Real watcher code; only process creation/identity and the launch3 preflight are stubbed."""
    import subprocess
    from . import replay_start,v2v_launch3
    cfg=dict(v2v_intent=True,gpu_deadline=10**12,legacy_root=str(tmp_path/'legacy'),job_namespace=NS,
        source_root=str(tmp_path/'source'),pace_s=1800)
    watch_authority(tmp_path,root,binding_path=tmp_path/'binding.json',
        binding_sha256='a'*64,control=tmp_path/'control',mover=tmp_path/'mover')
    (root/'V2V_RUNTIME.json').write_text('{}')
    monkeypatch.setattr(live,'configuration',lambda r:dict(config=cfg))
    live_pids={5001};spawned=[];quotes=[]
    monkeypatch.setattr(replay_start,'alive',lambda d:d.get('pid') in live_pids)
    monkeypatch.setattr(replay_start,'owner',lambda pid=None:dict(pid=pid or 77,start_ticks='1',boot_id='b',unix=0.))
    def popen(argv,**kwargs):
        pid=6000+len(spawned);spawned.append(argv);live_pids.add(pid)
        return types.SimpleNamespace(pid=pid)
    monkeypatch.setattr(subprocess,'Popen',popen)
    monkeypatch.setattr(v2v_launch3,'wait_for_preflight',
        lambda *a,**k:quotes.append(k['phase_costs']) or dict(ready=True,budget={}))
    def stop(seconds):
        raise StopLoop()
    monkeypatch.setattr(live.time,'sleep',stop)
    return cfg,live_pids,spawned,quotes


def test_adopt_keeps_live_A_resumes_B_and_R_without_double_A(tmp_path,monkeypatch,charges_ledger,fake_fcntl):
    root,_=started_root(tmp_path);_failed_attempt(root)
    cfg,live_pids,spawned,quotes=_adopt_env(tmp_path,monkeypatch,root)
    # adopt requires the resume_prepare receipt first (dead watcher/child records present)
    with pytest.raises(ValueError,match='watcher owner record present'):live.adopt(root,'r1',['B','R'])
    live.resume_prepare(root,'r1',['B','R'])
    with pytest.raises(StopLoop):live.adopt(root,'r1',['B','R'])
    # exactly two new children (B, R); the live A child is adopted, never re-dispatched
    assert [a[a.index('--arm')+1] for a in spawned]==['B','R']
    assert all(a[a.index('tools.coopmwm.v2v_live')+2]=='resume' and a[-2:]==['--attempt','r1'] for a in spawned)
    assert quotes[0]['v2v_train']==1814400
    assert quotes[1:]==[{'v2v_train':(22*24-24)*1200},{'v2v_train':22*24*1200}]
    plan=_json.loads((root/'V2V_RESUME_r1.json').read_text())
    assert plan['live_arms']==['A'] and plan['resume_arms']==['B','R']
    assert plan['arms']['B']['reusable']==_ids(3,'B',24) and plan['arms']['R']['resume_update']==3
    assert (root/'resume_r1_B_DISPATCH.json').exists() and (root/'resume_r1_R_DISPATCH.json').exists()
    assert _json.loads((root/'train_A_DISPATCH.json').read_text())['pid']==5001
    # one-use attempt: a second adopt with the same label refuses before any dispatch
    (root/'WATCH_OWNER.json').unlink()
    with pytest.raises(ValueError,match='one-use'):live.adopt(root,'r1',['B','R'])
    assert len(spawned)==2


def test_adopt_refuses_dead_A_or_unstarted_root(tmp_path,monkeypatch,charges_ledger,fake_fcntl):
    root,_=started_root(tmp_path);_failed_attempt(root)
    cfg,live_pids,spawned,quotes=_adopt_env(tmp_path,monkeypatch,root)
    live.resume_prepare(root,'r1',['B','R'])
    live_pids.discard(5001)              # the A child died: no double A, no silent restart
    with pytest.raises(ValueError,match='adopted training child exited: A'):live.adopt(root,'r1',['B','R'])
    assert spawned==[] and quotes==[] and not (root/'V2V_RESUME_r1.json').exists()
    (root/'V2V_STARTED.json').unlink();live_pids.add(5001)
    with pytest.raises(ValueError,match='only for a started root'):live.adopt(root,'r1',['B','R'])
    assert spawned==[]


def _real_episode(policy,ident,arm='B',seed=986000):
    from .v2v_runtime import from_weights
    from .test_v2v_intent_cpu import rollout
    from tools.coopnative.storage import canonical
    episode=rollout(from_weights({k:v.clone() for k,v in policy.state_dict().items()},enabled=True))
    episode=_json.loads(canonical(episode))
    episode.update(id=ident,arm=arm,charged_sim_s=1200,world=dict(seed=seed,cell=0))
    return episode


def test_resume_B_from_u2_reuses_u3_closed_episodes_of_its_own_checkpoint(tmp_path,monkeypatch,charges_ledger):
    import torch
    from .accel_learner import save
    from .test_v2v_intent_cpu import intent_policy,nonzero_head,CFG
    torch.set_num_threads(1)
    root,_=started_root(tmp_path,current=dict(A=('live',24),B=('closed',0),R=('closed',0)))
    (tmp_path/'u0').mkdir();(tmp_path/'u0b').mkdir()
    policy,optimizer,generator=intent_policy(tmp_path/'u0')
    nonzero_head(policy)
    (root/'B_checkpoint_002.pt').unlink()
    save(root/'B_checkpoint_002.pt',policy,optimizer,generator,2,0)
    rows=[_json.loads(line) for line in (root/'CHARGES.jsonl').read_text().splitlines()]
    for i,ident in enumerate(_ids(3,'B',3)):
        _job(root,ident,result=_real_episode(policy,ident,seed=986000+i),claim_mtime=7000.+i)
        rows.append(dict(id=ident,phase='v2v_train',sim_s=1200))
    write_charges(root,rows)
    (root/'B_UPDATE_002.json').write_text(_json.dumps(dict(update=2,trust=dict(beta=1.,graded=7))))
    (root/'B_SEARCH_LEARNING_002.json').write_text(_json.dumps(dict(arm='B',library=[])))
    (root/'V2V_STARTED.json').write_text(_json.dumps(dict(resumed_after_smoke=2)))
    cfg=dict(CFG,job_namespace=NS,pace_s=1800)
    per=live.arm_remaining(root,cfg,'B',live=False)
    assert per['reusable']==_ids(3,'B',3) and per['remaining_episodes']==22*24-3
    (root/'V2V_RESUME_r1.json').write_text(_json.dumps(dict(resume_arms=['B','R'],arms={'B':per})))
    state=live.resume_state(root,cfg,'B','r1')
    assert state['sealed_update']==2 and state['next_not_before_s']==7000.   # original u3 start
    progress=state['progress']
    assert progress['beta']==dict(beta=1.,graded=7) and progress['learned']==dict(arm='B',library=[])
    assert progress['running']==[] and progress['exact_count']==0
    receipt=_json.loads((root/'RESUME_r1_B_STATE.json').read_text())
    assert [r['id'] for r in receipt['reused']]==_ids(3,'B',3) and receipt['reused_sim_s']==3600
    assert all(r['max_abs_diff']<=live.REUSE_TOLERANCE for r in receipt['reused'])
    with pytest.raises(FileExistsError):live.resume_state(root,cfg,'B','r1')   # one-use receipt
    # episodes of any other policy are never reused
    other,_,_=intent_policy(tmp_path/'u0b',seed=11)
    nonzero_head(other,seed=5)
    with pytest.raises(ValueError,match='not produced by the sealed checkpoint'):
        live.verify_reuse(root,'B',other,_ids(3,'B',3))
    # the quoted state must still hold when the child starts
    (root/'V2V_RESUME_r2.json').write_text(_json.dumps(dict(resume_arms=['B'],arms={'B':dict(per,reusable=[])})))
    with pytest.raises(ValueError,match='changed since the adopt quote'):live.resume_state(root,cfg,'B','r2')


def test_resume_R_from_u2_stage_wiring(tmp_path,monkeypatch,charges_ledger):
    root,_=started_root(tmp_path)
    cfg=dict(job_namespace=NS,pace_s=1800,v2v_intent=True)
    value=dict(config=cfg,release=dict(qualified=True),amendment_sha256='x')
    monkeypatch.setattr(live,'configuration',lambda r:value)
    (root/'V2V_REAL_SMOKES_PASS.json').write_text(_json.dumps(dict(arms=['A','B','R'],pass_=True,next_not_before_s={})))
    (root/'V2V_STARTED.json').write_text(_json.dumps(dict(resumed_after_smoke=2)))
    per=live.arm_remaining(root,cfg,'R',live=False)
    (root/'V2V_RESUME_r1.json').write_text(_json.dumps(dict(resume_arms=['B','R'],arms={'R':per})))
    from . import v2v_runtime
    monkeypatch.setattr(v2v_runtime,'load_checkpoint',lambda p,config=None:(object(),None,None,dict(update=2)))
    seen={}
    monkeypatch.setattr(live,'run',lambda r,c,rel,arm:seen.update(cfg=c,release=rel,arm=arm))
    live.stage_arm(root,'resume','R','r1')
    c=seen['cfg']
    assert seen['arm']=='R' and c['resume_update']==2 and c['runtime_receipt_suffix']=='_V2V_RESUME_r1'
    assert c['next_not_before_s']=={'R':1000.+2*2000+1800}          # u2 paced start + pace
    assert c['resume_progress']['beta']==dict(beta=1.,graded=0) and c['resume_progress']['learned'] is None
    assert seen['release']['real_update_smokes_pass'] is True
    assert (root/'resume_r1_R_COMPLETE.json').exists()
    assert _json.loads((root/'RESUME_r1_R_STATE.json').read_text())['reused']==[]
    assert cfg.get('resume_update') is None                           # runtime config not mutated


def test_v2v_lessons_use_the_root_job_namespace(tmp_path,monkeypatch):
    """Legacy lesson_u006_<vid> charges already exist in the shared ledger: V2V lessons must not
    reuse that id namespace (or the u6/u12/u18 lessons would be refused as duplicates)."""
    from . import warm_runner as wr,replay_bank as rb,protocol
    cfg=binding(tmp_path)
    calls=[];captured={}
    hooks={k:Mock() for k in runner.REQUIRED_HOOKS}
    hooks['record_lesson']=lambda *a,**k:calls.append(k) or dict(ok=True)
    def fake_run_arm(root,arm,hooks=None):
        captured['lesson']=rb.record_lesson
    monkeypatch.setattr(runner,'admit',lambda *a:None)
    monkeypatch.setattr(wr,'run_arm',fake_run_arm)
    monkeypatch.setattr(protocol,'sha',lambda p:'a'*64)
    root=tmp_path/'new';root.mkdir()
    (root/'AUTHORITY.md').write_text('b'*64+'c'*64)
    runner.run_arm(root,'B',cfg,{},hooks)
    assert captured['lesson']('r','B',None,'r',5)['status']=='not_scheduled'
    captured['lesson']('r','B',None,'r',6,'runner_for',sources=())
    assert calls[-1]['attempt_kind']=='v2v1_lesson_u006' and calls[-1]['phase']=='v2v_lessons'


def test_gate_default_u3_reference_is_the_legacy_root(tmp_path):
    from .coopfix_exact_gate import legacy_u3
    assert legacy_u3(tmp_path/'v2v','B',dict(legacy_root=str(tmp_path/'warm1')))==tmp_path/'warm1'/'B_checkpoint_003.pt'
    assert legacy_u3(tmp_path/'warm1','R',{})==tmp_path/'warm1'/'R_checkpoint_003.pt'


def test_learning_pause_release_is_sha_bound_and_resume_prepare_accepts_only_released(tmp_path,monkeypatch,charges_ledger):
    """MAIN 09:00: the receipt-only learning pause is released only by its registered SHA-bound
    diagnosis/release; any later learning pause stays active; resume_prepare archives a HealthPause
    failure only once its named pause is released."""
    from . import replay_start
    from .protocol import sha
    from .warm_health import assert_running,pause,HealthPause
    from . import composite_health as ch
    root,_=started_root(tmp_path);_failed_attempt(root)
    for name in ('WARM_GUIDED_STOP.json',):(root/name).unlink()
    paused="HealthPause('registered health pause PAUSE_V2V_LEARNING_MECHANICS.json')"
    for arm in ('B','R'):(root/f'WARM_{arm}_FAILED_V2V_LIVE.json').write_text(_json.dumps(dict(error=paused)))
    monkeypatch.setattr(replay_start,'alive',lambda d:d.get('pid')==5001)
    with pytest.raises(HealthPause):pause(root,'V2V_LEARNING_MECHANICS',dict(failures=['ineffective_pass1']))
    with pytest.raises(HealthPause):assert_running(root,'B')
    with pytest.raises(ValueError,match='health pause needs a MAIN release'):live.resume_prepare(root,'r2',['B','R'])
    (root/ch.LEARNING_DIAGNOSIS).write_text(_json.dumps(dict(diagnosis='saturated tiny bank')))
    amendment='d'*64
    release=dict(authority=ch.LEARNING_AUTHORITY,released=True,pause=ch.LEARNING_PAUSE,
        original_pause_sha256=sha(root/ch.LEARNING_PAUSE),diagnosis_sha256=sha(root/ch.LEARNING_DIAGNOSIS),
        amendment_sha256=amendment)
    (root/ch.LEARNING_RELEASE).write_text(_json.dumps(release))
    with pytest.raises(HealthPause):assert_running(root,'B')          # not yet registered
    with (root/'AUTHORITY.md').open('a') as f:f.write('\namendmentSHA'+amendment+'\n')
    assert_running(root,'B');assert_running(root,'R')
    receipt=live.resume_prepare(root,'r2',['B','R'])
    assert {'WARM_B_FAILED_V2V_LIVE.json','WARM_R_FAILED_V2V_LIVE.json'}<={m['name'] for m in receipt['moved']}
    assert (root/ch.LEARNING_PAUSE).is_file()                          # the pause record stays in place
    # a NEW learning-mechanics failure writes a new pause name that is not released
    with pytest.raises(HealthPause):pause(root,'V2V_LEARNING_MECHANICS',dict(failures=['imitation_failed']))
    with pytest.raises(HealthPause):assert_running(root,'R')
    # a changed pause/diagnosis byte invalidates the release
    root2,_=started_root(tmp_path/'z')
    (root2/ch.LEARNING_PAUSE).write_text('{"rule":"x"}');(root2/ch.LEARNING_DIAGNOSIS).write_text('{}')
    (root2/ch.LEARNING_RELEASE).write_text(_json.dumps(dict(release,original_pause_sha256=sha(root2/ch.LEARNING_PAUSE),
        diagnosis_sha256='0'*64)))
    (root2/'AUTHORITY.md').write_text('amendmentSHA'+amendment)
    with pytest.raises(HealthPause):assert_running(root2,'B')
