"""Salvage completed C3 work, retain pre-SUMO disk charges, retry five worlds once."""
import copy
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from tools.longint import resume_c3 as base

REL, ARM, ROOT = base.REL, base.ARM, base.ROOT
OUT = ARM / 'loop/longint/c3resume/r2/collect_disk'
CONFIG = ARM / 'CONFIG_C3_DISK_REPAIR.json'
FAILED = {968187, 968189, 968191, 968193, 968195}
SALVAGE = {968188, 968190, 968192, 968194}
CPU = '/home/ubuntu/c4/main14/restart20261026'


def remote_json(loop, cfg, box, source):
    h = loop.host(cfg, box)
    p = subprocess.run([*h['ssh'], shlex.join([h['python'], '-B', '-c', source])],
                       capture_output=True, text=True, timeout=30)
    assert p.returncode == 0, p.stderr + p.stdout
    return json.loads(p.stdout)


def restored_receipt(loop, cfg, case, policy, wm):
    seed = case['seed']
    local = ARM / 'loop/worlds/c03' / f'w{seed}.a1'
    artifact = local / 'artifact'
    files = loop.fast_module().artifact_valid(artifact)
    assert files
    summary_path = local / 'collect/c03_a1_COLLECT_SUMMARY.json'
    summary = loop.read_json(summary_path)
    assert summary['episodes_failed'] == 0 and summary['episodes_complete'] == 1
    assert summary['branches'] == 4
    assert not any(summary[k] for k in ('collisions', 'teleports', 'branch_collisions', 'branch_teleports'))
    work = f"{cfg['collect']['work_root']}/w{seed}/{cfg['loop_id']}/c03"
    assert summary['policy'] == work + '/inputs/policy.pt'
    manifest = loop.read_json(artifact / 'pairs/C4_EPISODES_MANIFEST.json')
    assert len(manifest['episodes']) == 4
    for row in manifest['episodes']:
        assert row['seed'] == seed
        if row['branch']['kind'] == 'POLICY':
            assert row['branch']['policy_checkpoint'] == summary['policy']
    proof = remote_json(loop, cfg, 'cpuQ',
        'from pathlib import Path; import json,hashlib; '
        + 'paths=' + repr([work + '/inputs/policy.pt', work + '/inputs/wm/wm_s.pt']) + '; '
        + 'print(json.dumps([hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths]))')
    assert proof == [policy['sha256'], wm['sha256']]
    deltas = list(summary['pair_dJ'].values())
    result = {'case': case, 'box': 'cpuQ', 'attempt': 1, 'artifact': str(artifact),
        'hashes_sha256': base.sha(artifact / 'HASHES.json'),
        'salvaged_after_disk_refusal': True, 'source_summary_sha256': base.sha(summary_path),
        'collect': {'work': work, 'seeds': [seed], 'sim_s_charged': 2841, 'exit': 0,
            'episodes': 1, 'branches': 4, 'pair_sim_seconds': summary['pair_sim_seconds'],
            'branch_collisions': 0, 'branch_teleports': 0, 'sumo_pairs': len(deltas),
            'sumo_mean_policy_minus_native_J': sum(deltas) / len(deltas),
            'sumocredit': summary['sumocredit'], 'gate_table_used': False}}
    return result


def prepare(markdown_sha):
    loop = base.setup()
    held = base.locks()
    assert held and not OUT.exists() and OUT.parent.is_dir()
    cfg = loop.load_config(ARM / 'CONFIG_C3_RESUME_R2.json')
    state = loop.load_state(cfg)
    assert state['sim_s_charged'] == 250339 and not state.get('stopped')
    phases = state['cycles'][-1]['phases']
    assert phases['policy']['status'] == 'done' and phases['collect']['status'] == 'pending'
    assert 'disk guard' in phases['collect']['refused']
    for name in ('PID.json', 'TRAINER_PID.json'):
        pid = loop.read_json(OUT.parent / name)
        assert loop.fast_module().process_token(pid['pid']) != str(pid['start_ticks'])
    path = ARM / 'loop/worlds_c03.json'
    journal = loop.read_json(path)
    assert len(journal['worlds']) == 4 and journal['native_smoke']['verdict'] == 'inconclusive'
    OUT.mkdir()
    loop.write_atomic(OUT / 'BEFORE.json', {'state': state, 'journal': journal,
        'config': cfg, 'registration': loop.read_json(cfg['registration'])})
    loop.sync(cfg, 'cpuQ', CPU + '/autonomy/C3_DISK_DEDUP.json', 'gpu',
              str(OUT / 'CPUQ_DISK_DEDUP.json'), OUT / 'dedup_receipt_pull.log')
    dedup = loop.read_json(OUT / 'CPUQ_DISK_DEDUP.json')
    assert dedup['apply'] and dedup['preserved_content_and_paths'] and dedup['charged_sim_s'] == 0
    adopted = []
    for case in journal['cases']:
        if case['seed'] in SALVAGE:
            receipt = restored_receipt(loop, cfg, case, state['policy'], state['policy_wm'])
            journal['worlds'][str(case['seed'])] = receipt
            adopted.append(receipt)
    assert len(journal['worlds']) == 8
    # Every missing world failed at disk_guard BEFORE any input or work directory existed.
    missing = remote_json(loop, cfg, 'cpuP',
        'from pathlib import Path; import json; root=Path(' + repr(cfg['collect']['work_root']) + '); '
        + 'print(json.dumps({s:(root/("w"+str(s))).exists() for s in ' + repr(sorted(FAILED)) + '}))')
    assert not any(missing.values())
    assert {c['seed'] for c in journal['cases'] if str(c['seed']) not in journal['worlds']} == FAILED
    cfg['fastloop']['workers_per_box'] = 1
    cfg['fastloop']['boxes'] = ['cpuQ']
    cfg['collect_disk_repair'] = {'pending': sorted(FAILED), 'healthy_box': 'cpuQ',
        'retained_failed_charges': 14205, 'new_reserve': 14205, 'projected_total': 264544,
        'raw_mb_per_world': cfg['collect']['raw_mb_per_world'], 'input_mb_per_world': 8,
        'serial_conversion_temp_mb': 64, 'after_convert': 'WAIT_MAIN_C3_WMFT'}
    loop.write_atomic(CONFIG, cfg)
    repair = {'id': 'S-C3RESUME_COLLECT_DISK_REPAIR', 't': loop.now(),
        'authority': 'main 2026-10-28 06:20 fix-and-retry',
        'markdown_registration_sha256': markdown_sha, 'config_sha256': base.sha(CONFIG),
        'script_sha256': base.sha(__file__), 'failed_pre_sumo': sorted(FAILED),
        'salvaged_without_replay': sorted(SALVAGE), 'charged_before': 250339,
        'new_charge': 14205, 'projected_total': 264544, 'disk_floor_gib': 1.5,
        'cpuq_dedup_sha256': base.sha(OUT / 'CPUQ_DISK_DEDUP.json'),
        'guard_unchanged': copy.deepcopy(journal['native_smoke'])}
    reg = loop.read_json(cfg['registration'])
    assert not any(a.get('id') == repair['id'] for a in reg['amendments'])
    reg['amendments'].append(repair)
    loop.write_atomic(Path(cfg['registration']), reg)
    state['registration_sha256'] = base.sha(cfg['registration'])
    loop.save_state(cfg, state)
    loop.write_atomic(path, journal)
    loop.write_atomic(OUT / 'SALVAGED.json', adopted)
    loop.write_atomic(OUT / 'AUTHORITY.json', repair)
    base.event(loop, 'collection_disk_repair_registered', **repair)
    print(json.dumps(repair))


def install(loop):
    remote = base.module('c3disk_autonomy', REL / 'autonomy/remote.py')
    remote.R, remote.A, remote.ROOT, remote.loop = REL, ARM, ROOT, loop
    collector = base.module('c3disk_collector', REL / 'autonomy/collector.py')
    fast = loop.fast_module()

    def preflight(cfg, state, cases):
        available = remote.preflight(cfg, state, cases)
        assert 'cpuQ' in available
        journal = loop.read_json(ARM / 'loop/worlds_c03.json')
        pending = [c for c in cases if str(c['seed']) not in journal['worlds']]
        assert {c['seed'] for c in pending} <= FAILED
        free = remote_json(loop, cfg, 'cpuQ', 'import shutil,json; print(json.dumps(shutil.disk_usage('
                           + repr(cfg['collect']['work_root']) + ').free))')
        spec = cfg['collect_disk_repair']
        need = int(cfg['guards']['min_free_gb'] * 1024**3) + 1024**2 * (
            len(pending) * (spec['raw_mb_per_world'] + spec['input_mb_per_world'])
            + spec['serial_conversion_temp_mb'])
        proof = {'available_boxes': ['cpuQ'], 'pending': [c['seed'] for c in pending],
                 'charged_sim_s': 0, 'free_bytes': free, 'required_bytes': need,
                 'disk_floor_gib': cfg['guards']['min_free_gb'], 'status': 'PASS' if free >= need else 'REFUSED'}
        loop.write_atomic(OUT / 'DISK_PREFLIGHT.json', proof)
        assert free >= need, 'aggregate remaining-cohort disk readiness failed before charging'
        base.event(loop, 'disk_preflight', **proof)
        return ['cpuQ']

    fast.collect = lambda l, c, s, p, r: collector.collect(
        l, c, s, p, r, preflight, lambda value, **kw: base.status(loop, value, **kw), remote.priority)
    loop.fast_module = lambda: fast


def drive():
    loop = base.setup()
    held = base.locks()
    assert held and (ARM / 'loop/STOP_DRIVER').exists()
    cfg = loop.load_config(CONFIG)
    install(loop)
    try:
        for phase in ('collect', 'convert'):
            state = loop.load_state(cfg)
            assert state['cycles'][-1]['cycle'] == 3 and not state.get('stopped')
            assert state['sim_s_charged'] <= 280000
            base.status(loop, 'C3_DISK_REPAIR_' + phase.upper() + '_RUNNING')
            code = loop.run_phase(cfg, phase)
            if code:
                base.status(loop, 'C3_' + phase.upper() + '_STOPPED', exit=code)
                return code
        state = loop.load_state(cfg)
        assert state['sim_s_charged'] == 264544
        assert state['cycles'][-1]['phases']['wm_ft']['status'] == 'pending'
        base.status(loop, 'WAIT_MAIN_C3_WMFT', wm_base_decision='MAIN', policy=state['policy'],
                    conversion=state['cycles'][-1]['phases']['convert']['receipt'])
        return 0
    except Exception as error:
        base.status(loop, 'OPERATIONAL_PAUSED', reason=repr(error))
        raise


def start():
    loop = base.setup()
    assert not (OUT / 'PID.json').exists()
    with (OUT / 'driver.log').open('x') as log:
        child = subprocess.Popen([sys.executable, '-B', __file__, 'drive'], cwd=ROOT,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            env={**os.environ, 'PYTHONPATH': str(ROOT / 'src') + ':' + str(ROOT)})
    receipt = {'pid': child.pid, 'start_ticks': loop.fast_module().process_token(child.pid),
               'sid': os.getsid(child.pid), 'task': 'S-C3RESUME_DISK_REPAIR'}
    loop.write_atomic(OUT / 'PID.json', receipt)
    print(json.dumps(receipt))


if __name__ == '__main__':
    if sys.argv[1] == 'prepare':
        prepare(sys.argv[2])
    elif sys.argv[1] == 'start':
        start()
    elif sys.argv[1] == 'drive':
        sys.exit(drive())
