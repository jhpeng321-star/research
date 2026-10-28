"""Read hashed observable C3 summaries and verify the registered stopping boundary."""
import json
from pathlib import Path

from tools.longint.resume_c3 import ARM, CONFIG, OUT, REL, event, setup, sha


def report():
    loop = setup()
    repaired = (ARM / 'CONFIG_C3_DISK_REPAIR.json').exists()
    cfg = loop.load_config(ARM / 'CONFIG_C3_DISK_REPAIR.json' if repaired else CONFIG)
    state = loop.load_state(cfg)
    phases = state['cycles'][-1]['phases']
    assert state['cycles'][-1]['cycle'] == 3
    status = loop.read_json(REL / 'AUTONOMY_STATUS.json')['status']
    harmful_stop = status == 'C3_COLLECT_STOPPED' and state.get('stopped') == 'native smoke harmful'
    assert phases['policy']['status'] == 'done'
    if not harmful_stop:
        assert all(phases[p]['status'] == 'done' for p in ('collect', 'convert'))
        assert status == 'WAIT_MAIN_C3_WMFT'
    else:
        assert phases['convert']['status'] == 'pending'
    assert all(phases[p]['status'] == 'pending' for p in ('wm_ft', 'check'))
    assert all(phases[p]['attempt'] == 0 for p in ('wm_ft', 'check'))
    assert (ARM / 'loop/STOP_DRIVER').exists()
    journal = loop.read_json(ARM / 'loop/worlds_c03.json')
    completed = 4 if harmful_stop else 13
    assert len(journal['cases']) == 13 and len(journal['worlds']) == completed
    assert not cfg['collect'].get('probe', {}).get('enabled', False)
    rows = [json.loads(x) for x in (ARM / 'loop/seed_ledger.jsonl').read_text().splitlines()]
    charges = [r for r in rows if r.get('cycle') == 3 and r.get('event') == 'world_attempt']
    retained_failures = 5 if repaired else 0
    assert len(charges) == completed + retained_failures
    assert len({(r['seed'], r['attempt']) for r in charges}) == len(charges)
    assert all(r['attempt'] in (1, 2) and not 964000 <= r['seed'] <= 964199 for r in charges)
    assert {r['seed'] for r in charges if r['attempt'] == 2} == (
        {968187, 968189, 968191, 968193, 968195} if repaired else set())
    charged = sum(r['sim_s_charged'] for r in charges)
    assert charged == (completed + retained_failures) * 2841
    assert state['sim_s_charged'] == 213406 + charged <= 280000
    from tools.fastloop.world import verify
    from wms.control.nativeinit import artifact_world_delta, smoke_guard
    worlds, behavior_exports = [], []
    for case in journal['cases'][:completed]:
        receipt = journal['worlds'][str(case['seed'])]
        root = Path(receipt['artifact'])
        files = verify(root)
        assert not any(name.startswith(('PROBE_', 'probes/')) for name in files)
        for name in files:
            if any(token in name.lower() for token in ('coopmeas', 'behaviour', 'behavior')):
                path = root / name
                loop.guard(path)
                behavior_exports.append({'seed': case['seed'], 'source': str(path),
                    'sha256': files[name], 'value': loop.read_json(path) if path.suffix == '.json' else
                    'non-JSON collector behavior export; retained at source'})
        delta = artifact_world_delta(root, int(cfg['collect']['branch_points']))
        worlds.append({'seed': case['seed'], 'demand': case['demand'],
                       'penetration': case['penetration'], 'box': receipt['box'],
                       'attempt': receipt['attempt'],
                       'policy_minus_native_veh_s': delta,
                       'hashes_sha256': sha(root / 'HASHES.json')})
        manifest = loop.read_json(root / 'pairs/C4_EPISODES_MANIFEST.json')
        for item in manifest['episodes']:
            if item.get('branch', {}).get('kind') not in ('B', 'POLICY'):
                continue
            path = root / (item['source'].removeprefix('@artifact/') + '/summary.json')
            loop.guard(path)
            assert path.resolve().is_relative_to(root.resolve()) and not path.is_symlink()
            summary = loop.read_json(path)
            ledger = summary['ledger']
            assert abs(ledger['J_vehicle_seconds'] - ledger['speed_deficit_vehicle_seconds']
                       - ledger['source_wait_vehicle_seconds']) < 1e-5
            for key in summary:
                if any(token in key.lower() for token in ('coopmeas', 'behaviour', 'behavior')):
                    behavior_exports.append({'seed': case['seed'], 'arm': item['branch']['kind'],
                                             'source': str(path), 'key': key, 'value': summary[key]})
    guard = smoke_guard([w['policy_minus_native_veh_s'] for w in worlds[:4]])
    assert guard == journal['native_smoke']
    assert (guard['verdict'] == 'harmful') == harmful_stop
    verified = loop.read_json(OUT / 'POLICY_VERIFIED.json')
    result = {'task': 'S-C3RESUME', 'status': status,
        'policy': verified, 'charged_before': 213406, 'new_charged': charged,
        'charged_total': state['sim_s_charged'], 'remaining': 280000 - state['sim_s_charged'],
        'guard': {**guard, 'worlds': worlds[:4],
                  'units': 'policy - native veh-s; road speed-deficit + entrance waiting; sum of two paired starts/world',
                  'collection_decision': 'continue unless harmful; inconclusive is not relabelled pass',
                  'bootstrap': '5000 paired whole-world draws, seed20261026, percentile95CI'},
        'worlds': worlds, 'probes': False, 'behavior_exports': behavior_exports,
        'retained_failed_pre_sumo_attempts': retained_failures,
        'retained_failed_charges': retained_failures * 2841,
        'behavior_availability': 'exported' if behavior_exports else
            'COOPMEAS behavior rates are not exported by this registered collector',
        'conversion': phases['convert'].get('receipt'),
        'preflight': loop.read_json(REL / 'AUTONOMY_PREFLIGHT.json'),
        'wm_ft_started': False, 'wm_ft_attempt': phases['wm_ft']['attempt'],
        'check_attempt': phases['check']['attempt'], 'stop_driver_preserved': True}
    if repaired:
        result['collection_repair'] = loop.read_json(OUT / 'collect_disk/AUTHORITY.json')
        result['disk_preflight'] = loop.read_json(OUT / 'collect_disk/DISK_PREFLIGHT.json')
        result['disk_dedup'] = {k: v for k, v in loop.read_json(
            OUT / 'collect_disk/CPUQ_DISK_DEDUP.json').items() if k != 'changes'}
    target = OUT / 'RESULTS.json'
    assert not target.exists()
    loop.write_atomic(target, result)
    event(loop, 'final_verified', results=str(target), results_sha256=sha(target),
          charged=state['sim_s_charged'], guard=guard, selected_update=verified['gate']['update'])
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    report()
