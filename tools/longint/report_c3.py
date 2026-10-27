"""Read hashed observable C3 summaries and verify the registered stopping boundary."""
import json
from pathlib import Path

from tools.longint.resume_c3 import ARM, CONFIG, OUT, REL, event, setup, sha


def report():
    loop = setup()
    cfg = loop.load_config(CONFIG)
    state = loop.load_state(cfg)
    phases = state['cycles'][-1]['phases']
    assert state['cycles'][-1]['cycle'] == 3
    assert all(phases[p]['status'] == 'done' for p in ('policy', 'collect', 'convert'))
    assert all(phases[p]['status'] == 'pending' for p in ('wm_ft', 'check'))
    assert (ARM / 'loop/STOP_DRIVER').exists()
    assert loop.read_json(REL / 'AUTONOMY_STATUS.json')['status'] == 'WAIT_MAIN_C3_WMFT'
    journal = loop.read_json(ARM / 'loop/worlds_c03.json')
    assert len(journal['cases']) == len(journal['worlds']) == 13
    assert not cfg['collect'].get('probe', {}).get('enabled', False)
    rows = [json.loads(x) for x in (ARM / 'loop/seed_ledger.jsonl').read_text().splitlines()]
    charges = [r for r in rows if r.get('cycle') == 3 and r.get('event') == 'world_attempt']
    assert len(charges) == 13
    assert len({(r['seed'], r['attempt']) for r in charges}) == 13
    assert all(r['attempt'] == 1 and not 964000 <= r['seed'] <= 964199 for r in charges)
    assert sum(r['sim_s_charged'] for r in charges) == 36933
    assert state['sim_s_charged'] == 250339 <= 280000
    from tools.fastloop.world import verify
    from wms.control.nativeinit import artifact_world_delta, smoke_guard
    worlds, behavior_exports = [], []
    for case in journal['cases']:
        receipt = journal['worlds'][str(case['seed'])]
        root = Path(receipt['artifact'])
        verify(root)
        delta = artifact_world_delta(root, int(cfg['collect']['branch_points']))
        worlds.append({'seed': case['seed'], 'demand': case['demand'],
                       'penetration': case['penetration'], 'box': receipt['box'],
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
    assert guard['verdict'] != 'harmful'
    verified = loop.read_json(OUT / 'POLICY_VERIFIED.json')
    result = {'task': 'S-C3RESUME', 'status': 'WAIT_MAIN_C3_WMFT',
        'policy': verified, 'charged_before': 213406, 'new_charged': 36933,
        'charged_total': state['sim_s_charged'], 'remaining': 280000 - state['sim_s_charged'],
        'guard': {**guard, 'worlds': worlds[:4],
                  'units': 'policy - native veh-s; road speed-deficit + entrance waiting; sum of two paired starts/world',
                  'collection_decision': 'continue unless harmful; inconclusive is not relabelled pass',
                  'bootstrap': '5000 paired whole-world draws, seed20261026, percentile95CI'},
        'worlds': worlds, 'probes': False, 'behavior_exports': behavior_exports,
        'behavior_availability': 'exported' if behavior_exports else
            'COOPMEAS behavior rates are not exported by this registered collector',
        'conversion': phases['convert']['receipt'],
        'preflight': loop.read_json(REL / 'AUTONOMY_PREFLIGHT.json'),
        'wm_ft_started': False}
    target = OUT / 'RESULTS.json'
    assert not target.exists()
    loop.write_atomic(target, result)
    event(loop, 'final_verified', results=str(target), results_sha256=sha(target),
          charged=state['sim_s_charged'], guard=guard, selected_update=verified['gate']['update'])
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    report()
