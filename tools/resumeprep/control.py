"""Box-side prepare/prearm/mini-cycle/launch/verify CLI. Preparation never dispatches SUMO."""
from __future__ import annotations
import argparse
import copy
try:
    import fcntl
except ImportError:
    fcntl = None
import json
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'scripts/c4'), str(ROOT), str(ROOT/'src')]
import loop14 as loop  # noqa: E402
import main14_autolaunch as waiter  # noqa: E402

BASE_SHA = '5b2569418bf651b8ef2c3a2abdc110ac6860c32dd92dca8cdf3b2f48b1c8cbb2'
PANEL_SHA = '3eaa69a170c17302660d12dd4d7fddb4657e43fa334eece17844dd0fa6c0411e'


def prepare(old_arm, arm, snapshot=None, shared_gpu=False):
    old_cfg = loop.read_json(old_arm/'CONFIG.json')
    def mapped(value):
        return str(snapshot/Path(value).relative_to('/root')) if snapshot else value
    if snapshot:
        old_cfg['loop_root'] = mapped(old_cfg['loop_root'])
        old_cfg['registration'] = mapped(old_cfg['registration'])
    state = loop.load_state(old_cfg)
    if state.get('stopped') or state['accepted_wm']['sha256'] != BASE_SHA:
        raise loop.Refused('scientific stop or wrong accepted base')
    arm.mkdir(parents=True, exist_ok=False)
    root = arm/'loop'
    root.mkdir()
    cfg = copy.deepcopy(old_cfg)
    if snapshot:
        for key in ('initial_wm','initial_wm_config'):
            cfg['wm'][key] = mapped(cfg['wm'][key])
        for key in ('checkpoint','config'):
            state['accepted_wm'][key] = mapped(state['accepted_wm'][key])
        cfg['policy']['initial_real_returns'] = mapped(cfg['policy']['initial_real_returns'])
        cfg['policy'].setdefault('allowed_roots',[]).append(str(snapshot))
        cfg['policy']['allowed_roots'].append(str(snapshot.parent/'data'))
        for key in ('WMS_DATA_V5_NATIVE_Q','WMS_DATA_GENRAND_PLAIN_Q'):
            cfg['policy']['env'][key] = str(snapshot.parent/'data'/Path(cfg['policy']['env'][key]).name)
    cfg.update(loop_root=str(root), registration=str(arm/'REGISTRATION.json'))
    cfg['hosts']['gpu']['code_root'] = str(ROOT)
    for name in ('cpu','cpuP','cpuQ'):
        if name in cfg['hosts']:
            cfg['hosts'][name]['code_root'] = '/home/ubuntu/c4/main14/resumeprep/runtime'
    for host in cfg.get('mainstart',{}).get('collect_hosts',{}).values():
        host['code_root'] = '/home/ubuntu/c4/main14/resumeprep/runtime'
    cfg['resumeprep'] = {'enabled': True}
    cfg['policy'].update(native_init=True, updates_per_cycle=30)
    if '--native-init' not in loop.policy_cmd(cfg,state,arm/'native_support_check')[0]:
        raise loop.Refused('merge S-NATIVEINIT before preparing the successor')
    cfg['fastloop']['overlap_policy'] = True
    cfg['wm']['timeout_h'] = 4.5
    cfg['wm']['extra_set'] = [x for x in cfg['wm'].get('extra_set', []) if not x.startswith('curriculum.max_hours=')]+['curriculum.max_hours=4.5']
    cfg['policy']['env']['C4_GPU_MEM_FRACTION'] = '0.35'
    cfg['wm']['env']['C4_GPU_MEM_FRACTION'] = '0.50'
    if shared_gpu:
        cfg['resumeprep']['shared_gpu'] = True
        cfg['policy']['env']['C4_GPU_MEM_FRACTION'] = '0.20'
        cfg['wm']['env']['C4_GPU_MEM_FRACTION'] = '0.20'
    # Reserve smoke seeds in both successor main registration and its allocator.
    reg = copy.deepcopy(loop.read_json(old_cfg['registration']))
    reg['budget']['gpu_h_wm_per_cycle'] = 4.5
    reg.setdefault('exclude', []).append([968142, 968143])
    reg['resumeprep'] = {'owner_approved': '2026-10-26 ~20:20', 'smoke': 'S-C0DIAG-SMOKE-1',
                         'prior_charges': state['sim_s_charged'], 'new_cycle0': True}
    reg['budget']['sim_s_per_cycle'] = 29133
    loop.write_atomic(arm/'REGISTRATION.json', reg)
    args = cfg['policy'].get('extra_args', [])
    for flag, name in (('--gate-panel','gate_panel.json'),('--horizon-gate','horizon_gate_current.json')):
        if flag not in args:
            raise loop.Refused(f'missing productive gate {flag}')
        i = args.index(flag)+1
        (arm/name).write_bytes(Path(mapped(args[i])).read_bytes())
        args[i] = str(arm/name)
    if loop.sha256(arm/'gate_panel.json') != PANEL_SHA:
        raise loop.Refused('frozen panel changed')
    migrated = copy.deepcopy(state)
    migrated.setdefault('superseded_cycles', []).extend(migrated['cycles'])
    migrated.setdefault('superseded_policies', []).append(migrated['policy'])
    migrated.update(cycles=[loop.new_cycle(0)], policy=None, real_returns=cfg['policy']['initial_real_returns'],
                    registration=cfg['registration'], registration_sha256=loop.sha256(cfg['registration']))
    for key in ('policy_wm','real_credit','gate_table','gate_mean_table'):
        migrated.pop(key,None)
    # Existing charges stay in the append-only historical ledger with original
    # cycle labels; a new generation marker separates current-cycle budget use.
    ledger = Path(old_cfg['loop_root'])/'seed_ledger.jsonl'
    (root/'seed_ledger.jsonl').write_bytes(ledger.read_bytes())
    migrated['resumeprep_prior_charges'] = state['sim_s_charged']
    migrated['resumeprep_prior_ledger_rows'] = len(ledger.read_text().splitlines())
    loop.write_atomic(root/loop.STATE,migrated)
    loop.write_atomic(arm/'CONFIG.json',cfg)
    loop.write_atomic(arm/'PREPARED.json',{'base':state['accepted_wm'], 'old_state_sha256':loop.sha256(loop.state_path(old_cfg)),
                      'charged':state['sim_s_charged'], 'reserved_seeds':[968142,968143], 'live_state_unchanged':True})


def fingerprint(cfg):
    import hashlib
    files = sorted(p for d in ('src','traffic_shepherd','scripts/c4','configs','tools') for p in (ROOT/d).rglob('*.py' if d!='configs' else '*.toml')
                   if not any(t in str(p).lower() for t in loop.FORBIDDEN))
    args = cfg['policy'].get('extra_args', [])
    dependencies = {key:loop.sha256(cfg['wm'][key]) for key in ('initial_wm','initial_wm_config')}
    for flag in ('--gate-panel','--horizon-gate'):
        if flag in args:
            dependencies[flag] = loop.sha256(args[args.index(flag)+1])
    if cfg['policy'].get('initial_real_returns'):
        dependencies['initial_real_returns'] = loop.sha256(cfg['policy']['initial_real_returns'])
    return {'config':hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),
            'source':hashlib.sha256(json.dumps({str(p.relative_to(ROOT)):loop.sha256(p) for p in files},sort_keys=True).encode()).hexdigest(),
            'dependencies':hashlib.sha256(json.dumps(dependencies,sort_keys=True).encode()).hexdigest()}


def prearm(cfg, arm):
    # Two synthetic optimizer updates; productive panel is hash-checked separately.
    state = loop.load_state(cfg)
    synthetic = copy.deepcopy(cfg)
    synthetic['policy'].update(smoke=True, updates_per_cycle=2)
    synthetic['policy']['device'] = 'cpu'
    synthetic['policy']['native_init'] = False
    if cfg['policy'].get('longcritic'):
        # Synthetic prearm checks mechanics only; real longcritic/share selection
        # is exercised in mini-policy with the unchanged production config.
        synthetic['policy']['config'] = 'configs/c4_policy14_main14_nativelane.toml'
        synthetic['policy']['longcritic'] = False
    synthetic['policy']['extra_args'] = []
    argv, env = loop.policy_cmd(synthetic,state,arm/'prearm')
    code = loop.run_cmd(loop.host(cfg,'gpu'),argv,env=env,log=arm/'prearm.log',timeout_s=600)
    if code or loop.sha256(arm/'gate_panel.json') != PANEL_SHA:
        raise loop.Refused('prearm failure')
    loop.write_atomic(arm/'PREARM.json',{'status':'PASS', **fingerprint(cfg)})


def mini(cfg, arm, policy_only=False):
    """Real, tiny policy -> exactly registered smoke -> conversion -> two WM steps."""
    bind = fingerprint(cfg)
    if loop.read_json(arm/'PREARM.json') != {'status':'PASS',**bind}:
        raise loop.Refused('new configuration requires prearm')
    one = copy.deepcopy(cfg)
    root = arm/'mini'
    one.update(loop_root=str(root/'loop'),registration=str(root/'REGISTRATION.json'))
    one['fastloop']['overlap_policy'] = False
    one['policy']['updates_per_cycle'] = 10
    one['collect'].update(worlds_per_cycle=2, demands='1900',penetrations='0.05,0.4',pen_weights='1,1',
                          branch_points=1,branch_first=360,branch_horizon=300,warmup=300)
    one['collect'].pop('sumocredit',None)
    one['collect'].pop('long_group',None)
    one['collect'].pop('long_single_hold',None)
    one['wm']['extra_set'] += ['curriculum.stage_steps=[2]']
    reg = copy.deepcopy(loop.read_json(cfg['registration']))
    reg.update(seed_block=[968142,968143],exclude=[])
    reg['budget'].update(max_cycles=1,sim_s_per_cycle=1922,sim_s_total=1922,wm_steps_per_cycle=2)
    reg['budget']['policy_updates_per_cycle'] = 10
    if not root.exists():
        root.mkdir()
        loop.write_atomic(root/'REGISTRATION.json',reg)
        loop.write_atomic(root/'CONFIG.json',one)
        loop.init(one,argparse.Namespace(adopt_policy=None))
    elif loop.read_json(root/'CONFIG.json') != one:
        raise loop.Refused('mini config changed')
    # Gate rejection in this phase happens before any seed reservation/charge.
    mini_state = loop.load_state(one)
    if mini_state['cycles'][0]['phases']['policy']['status'] != 'done':
        if cfg.get('resumeprep', {}).get('operational_retries_per_error') == 3:
            import loop14_resume
            code = loop14_resume.phase(loop, one, 'policy', root)
        else:
            code = loop.run_phase(one, 'policy')
        if code != 0:
            raise loop.Refused('mini policy gate failed; no SUMO authorized')
    # Atomic parent-state/ledger operations under the same driver lock used by launch.
    state = loop.load_state(cfg)
    rows = [json.loads(x) for x in (Path(cfg['loop_root'])/'seed_ledger.jsonl').read_text().splitlines()]
    used = set(state['seeds_used']) | {s for row in rows for s in row.get('seeds',[])} | {row['seed'] for row in rows if 'seed' in row}
    if used & {968142,968143} or state['sim_s_charged']+1922>loop.read_json(cfg['registration'])['budget']['sim_s_total']:
        raise loop.Refused('registered smoke seeds consumed or total cap exhausted')
    candidate = loop.load_state(one)['policy']
    gate = None
    if cfg['policy'].get('native_init'):
        from wms.control.nativeinit import verify_collection_gate
        gate = verify_collection_gate(candidate)
        if (gate.get('wm_sha256') != [state['accepted_wm']['sha256']] or
                gate.get('wm_config_sha256') != [loop.sha256(state['accepted_wm']['config'])]):
            raise loop.Refused('mini gate WM provenance mismatch before reservation')
    bindings = {**bind, 'policy':candidate, 'wm':state['accepted_wm'],
                'native_gate':gate,'panel_sha256':PANEL_SHA,'sim_s':1922,'seeds':[968142,968143]}
    if gate is not None:
        bindings['native_gate_sha256'] = loop.sha256(Path(candidate['checkpoint']).with_name('native_gate_result.json'))
    loop.write_atomic(arm/'SMOKE_BINDINGS.json', bindings)
    if policy_only:
        return
    # This receipt MUST also be appended to the shared register before dispatch.
    if not (arm/'SHARED_SMOKE_HASHES_APPENDED.json').exists():
        raise loop.Refused('append SMOKE_BINDINGS hashes to shared registration before dispatch')
    if loop.read_json(arm/'SHARED_SMOKE_HASHES_APPENDED.json').get('bindings_sha256') != loop.sha256(arm/'SMOKE_BINDINGS.json'):
        raise loop.Refused('shared registration acknowledgment has stale bindings')
    loop.append(Path(cfg['loop_root'])/'seed_ledger.jsonl',{'event':'smoke_reservation','cycle':0,
                'seeds':[968142,968143],'sim_s_charged':1922,'registration':'S-C0DIAG-SMOKE-1'})
    state['seeds_used'] += [968142,968143]
    state['sim_s_charged'] += 1922
    loop.save_state(cfg,state)
    for phase in ('collect','convert','wm_ft'):
        original = loop.RUNNERS['collect']
        if phase == 'collect' and cfg['policy'].get('native_init'):
            # This separately registered two-world mechanics transaction is not
            # the four-world efficacy barrier of a normal main collection.
            loop.RUNNERS['collect'] = mechanics_collect
        try:
            if loop.run_phase(one,phase) != 0:
                raise loop.Refused(f'mini {phase} failed; charges retained; no automatic extra SUMO')
        finally:
            loop.RUNNERS['collect'] = original
    loop.write_atomic(arm/'MINI_PASS.json', {'status':'PASS',**bind,'charged':1922,'state':str(root/'loop'/loop.STATE)})


def mechanics_collect(cfg, state, ph, root):
    """Only the registered 1,922s transaction; no statistical efficacy decision."""
    c = cfg['collect']
    if (loop.collect_sim_s(cfg) != 1922 or c['worlds_per_cycle'] != 2 or
            c.get('sumocredit') or c['branch_points'] != 1 or c['branch_first'] != 360 or
            c['branch_horizon'] != 300 or c['warmup'] != 300 or c['demands'] != '1900' or
            c['penetrations'] != '0.05,0.4' or loop.check_registration(cfg)['seed_block'] != [968142,968143]):
        raise loop.Refused('mechanics smoke differs from exact registration')
    from wms.control.nativeinit import verify_collection_gate
    gate = verify_collection_gate(state['policy'])
    if (gate.get('wm_sha256') != [state['accepted_wm']['sha256']] or
            gate.get('wm_config_sha256') != [loop.sha256(state['accepted_wm']['config'])]):
        raise loop.Refused('mechanics smoke native gate provenance mismatch')
    if state['sim_s_charged'] or state['seeds_used'] or ph['attempt'] != 1:
        raise loop.Refused('mechanics smoke already attempted; no extra SUMO retry authorized')
    fast = loop.fast_module()
    cases = fast.cases(cfg,[968142,968143])
    loop.append(root/'seed_ledger.jsonl',{'event':'registered_mechanics_smoke','cycle':0,
                'seeds':[968142,968143],'sim_s_charged':1922})
    state.update(sim_s_charged=1922,seeds_used=[968142,968143])
    loop.save_state(cfg,state)
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(fast._world,loop,cfg,copy.deepcopy(state),ph,root,item,item['box'],1) for item in cases]
        worlds=[job.result() for job in jobs]
    return {'worlds':worlds,'seeds':[968142,968143],'sim_s_charged':1922,
            'branch_collisions':sum(w['collect'].get('branch_collisions',0) or 0 for w in worlds),
            'branch_teleports':sum(w['collect'].get('branch_teleports',0) or 0 for w in worlds),
            'registration':'S-C0DIAG-SMOKE-1','efficacy_certification':False}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','prearm','mini-policy','mini','launch','drive','verify'])
    p.add_argument('--arm',required=True,type=Path)
    p.add_argument('--old-arm',type=Path)
    p.add_argument('--snapshot-root',type=Path)
    p.add_argument('--shared-gpu',action='store_true')
    a=p.parse_args()
    if fcntl is None:
        raise RuntimeError('operational control requires Linux flock')
    if a.action=='prepare':
        prepare(a.old_arm,a.arm,a.snapshot_root,a.shared_gpu)
        return
    cfg=loop.load_config(a.arm/'CONFIG.json')
    if a.action=='verify':
        loop.status(cfg)
        pid=loop.read_json(a.arm/'PID.json')
        print(json.dumps({'pid':pid, 'same_incarnation':loop.fast_module().process_token(pid['pid'])==pid['start_ticks'],
                          'wait_owner':(a.arm/'WAIT_OWNER.json').exists()}))
        return
    with (a.arm/'driver.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | (0 if a.action=='drive' else fcntl.LOCK_NB))
        if a.action=='prearm':
            prearm(cfg,a.arm)
        elif a.action in ('mini-policy','mini'):
            mini(cfg,a.arm,policy_only=a.action=='mini-policy')
        elif a.action=='launch':
            proof=loop.read_json(a.arm/'MINI_PASS.json')
            if proof.get('status')!='PASS' or any(proof.get(k)!=v for k,v in fingerprint(cfg).items()):
                raise loop.Refused('configuration/source lacks matching mini-cycle PASS')
            if (a.arm/'WAIT_OWNER.json').exists() or (a.arm/'PID.json').exists():
                raise loop.Refused('stop or launch receipt exists')
            # The child blocks on this inherited startup lock until parent exit.
            with (a.arm/'driver.log').open('xb') as log:
                proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__)),'drive','--arm',str(a.arm)],
                     start_new_session=True,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            loop.write_atomic(a.arm/'PID.json',{'pid':proc.pid,'start_ticks':loop.fast_module().process_token(proc.pid)})
        elif a.action=='drive':
            try:
                waiter.drive(cfg,a.arm)
            except Exception as error:
                import loop14_resume
                name = 'OPERATIONAL_PAUSED.json' if isinstance(error, loop14_resume.OperationalPaused) else 'WAIT_OWNER.json'
                loop.write_atomic(a.arm/name,{'t':loop.now(),'error':str(error)})
                raise


if __name__=='__main__':
    main()
