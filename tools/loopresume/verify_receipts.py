"""Verify the small handoff without models, remote writes, or simulation."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RECEIPTS = Path(__file__).with_name('receipts')


def read(name):
    return json.loads((RECEIPTS/name).read_text(encoding='utf-8'))


def main():
    amendment = json.loads(Path(__file__).with_name('AMENDMENT_C1.json').read_text(encoding='utf-8'))
    blocked = read('LOOPRESUME_PAIRING_BLOCKED.json')
    binding = read('LOOPRESUME_BINDINGS.json')
    tests = read('LOOPRESUME_TESTS_PASS.json')
    manifest = read('LOOPRESUME_SOURCE.json')
    assert tests['exit'] == 0
    assert tests['source_sha256'] == hashlib.sha256((RECEIPTS/'LOOPRESUME_SOURCE.json').read_bytes()).hexdigest()
    assert binding['charged'] == blocked['charged'] == 102607 <= amendment['execution']['charged_sim_s_cap']
    assert blocked['fit_worlds'] == 0 and blocked['development_worlds'] == 4 and blocked['real_pairs'] == 8
    assert not blocked['cycle1_launched'] and blocked['export_retries_exhausted'] == 3
    assert not blocked['scientific_threshold_or_partition_changed']
    partition = {str(s): int(hashlib.sha256(f'965140:{s}'.encode()).hexdigest(), 16) % 5
                 for s in amendment['cycle0']['collected_worlds']}
    assert partition == blocked['partition_remainders'] and set(partition.values()) == {0}
    assert len(amendment['cycle0']['reserved_unconsumed_worlds']) == 9
    assert amendment['future_guard']['effective_from_cycle'] == 1
    assert amendment['future_guard']['inconclusive_continues'] and amendment['future_guard']['margin'] == 0
    assert amendment['paircrit']['breakdown'] is False and amendment['paircrit']['conformal_ranges'] is False
    wm = read('LOOPRESUME_WM_ONLY.json')
    assert wm['charged'] == 102607 and wm['actor_launch_blocked']
    assert wm['converted_outputs']['pairing_not_fit_eligible']
    assert wm['confidence_recalibration']['gate_n'] == 64
    assert wm['confidence_recalibration']['gate_mean_n'] == 56
    assert read('LOOPRESUME_WM_PID.json')['pid'] != read('LOOPRESUME_LAUNCH.json')['pid']
    count = 0
    for name in manifest:
        path = ROOT/name
        if path.suffix == '.py' and path.is_file():
            ast.parse(path.read_text(encoding='utf-8-sig'), filename=name)
            count += 1
    ast.parse(Path(__file__).read_text(encoding='utf-8'))
    print(json.dumps({'status': 'PASS', 'registered_amendment': True, 'partition_block_reproduced': True,
                      'charged': 102607, 'remaining': 107393, 'reserved_unconsumed_worlds': 9,
                      'policy_cycle1_launched': False, 'wm_snapshot_status': wm['status'],
                      'source_files_in_remote_manifest': len(manifest), 'local_python_syntax_checked': count}))


if __name__ == '__main__':
    main()
