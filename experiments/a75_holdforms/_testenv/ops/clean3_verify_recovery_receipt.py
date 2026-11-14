"""Offline integrity/phase-margin audit of saved public r12 receipts; no simulation."""
import argparse
import json
from pathlib import Path

from clean3_follow_remote import sealed


def verify(audit, progress):
    receipts = audit['receipts']
    launch = receipts['LAUNCHED.json']
    expected = {
        'SMOKE_PASS.json': 'SMOKE_PASS',
        'SMOKE_FIX_RECEIPTS_PASS.json': 'FIX_RECEIPTS_PASS',
        'SMOKE_REPORTING_PASS.json': 'COMPLETE',
        'SMOKE_LEVER_RECEIPTS_PASS.json': 'LEVER_RECEIPTS_PASS',
        'GATE_PASS.json': 'PASS',
    }
    for name, status in expected.items():
        value = sealed(receipts[name])
        assert value['status'] == status, (name, value['status'])
    smoke, gate = receipts['SMOKE_PASS.json'], receipts['GATE_PASS.json']
    fix = receipts['SMOKE_FIX_RECEIPTS_PASS.json']
    reporting = receipts['SMOKE_REPORTING_PASS.json']
    lever = receipts['SMOKE_LEVER_RECEIPTS_PASS.json']
    assert smoke['receipt_sha256'] == launch['smoke_sha256'] == gate['smoke_sha256']
    assert gate['receipt_sha256'] == launch['gate_sha256']
    assert smoke['clean2_fix_receipts_sha256'] == fix['receipt_sha256'] == lever['fix_receipts_sha256']
    assert fix['reporting_receipt_sha256'] == reporting['receipt_sha256']
    assert smoke['clean3_lever_receipts_sha256'] == lever['receipt_sha256']
    assert smoke['config_sha256'] == launch['runtime_sha256']
    assert smoke['source_manifest_sha256'] == launch['source_manifest_sha256']
    assert audit['config_file_sha256'] == launch['config_file_sha256']
    assert audit['source_commit'] == launch['runtime_commit']
    assert progress['launch'] == launch
    assert smoke['smoke_sim_s'] <= 20000
    assert all(gate['safety'][key] == 0 for key in ('collisions', 'teleports'))
    coverage = receipts['PHASE_COVERAGE_PASS.json']
    margins = {}
    for phase, need in coverage['need'].items():
        allocated = coverage['allocated'][phase]
        assert allocated * 100 >= need * 115, (phase, allocated, need)
        assert audit['budget']['phases'][phase] == allocated
        margins[phase] = dict(need=need, allocated=allocated,
                              margin_fraction=allocated/need-1)
    assert coverage['g4'] == audit['budget']['g4_cap_sim_s'] == 230400
    assert sum(v['total_sim_s'] for v in audit['previous_clean3_roots'].values()) == 22800
    assert audit['completed_g1']['charged_sim_s'] == audit['budget']['spent']['g1'] == 0
    assert audit['qualification']['returncode'] == 0
    assert audit['chain_owner']['alive']
    assert all(owner['alive'] for owner in audit['launch_owners'].values())
    assert all(owner['alive'] or owner['complete'] for owner in progress['owners'].values())
    assert not progress['faults'], progress['faults']
    assert not any(name in receipts for name in ('STOP.json', 'FAULT.json', 'PARTIAL_LAUNCH.json'))
    mined = lever['levers']['mined']['B']
    return dict(status='INTEGRITY_AND_PHASE_MARGIN_PASS',
        timestamp=progress['timestamp'], source_files_verified=audit['source_files_verified'],
        source_commit=audit['source_commit'], root=audit['root'],
        smoke_sim_s=smoke['smoke_sim_s'], gate_safety=gate['safety'], margins=margins,
        charged_sim_s=progress['charged_sim_s'],
        protected_total_sim_s=progress['protected_total_sim_s'], cap_sim_s=progress['cap_sim_s'],
        completed_combined_updates=sorted(progress['monitors'], key=int),
        mined_exposure_evidence=dict(admitted_examples=mined['examples'],
            practice_pool_contexts=mined['situations']['distinct_situations'],
            actual_B_smoke_demonstrations=len(mined['demonstrations']),
            observable_imitation_targets=mined['imitation_targets'],
            finding='Registration says every example practice-demonstrated; code checks every example in pool and at least one actual demonstration. Refer to launch operator before claiming all-example demonstration coverage.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit', required=True)
    parser.add_argument('--progress', required=True)
    parser.add_argument('--receipt', required=True)
    args = parser.parse_args()
    target = Path(args.receipt)
    assert target.parent.is_dir() and not target.exists()
    value = verify(json.loads(Path(args.audit).read_text()),
                   json.loads(Path(args.progress).read_text()))
    with target.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
    print(json.dumps(value, sort_keys=True))
