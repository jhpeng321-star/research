"""Validate the small final handoff receipt; no models, traffic data or simulation."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    receipt = json.loads((ROOT/'tools/restart/RESTART_FINAL.json').read_text())
    gate = receipt['guard']
    assert receipt['status'] == gate['status'] == 'SCIENTIFIC_WAIT_OWNER'
    assert gate['updates'] == 30 and gate['update10']['passed'] and gate['update10']['n_improved'] == 16
    assert gate['native_gate']['passed'] and gate['native_gate']['selected_update'] == 30
    real = gate['real_guard']
    assert real['verdict'] == 'inconclusive' and not real['passed']
    assert real['margin'] == 0 and real['ci95'][0] <= 0 < real['ci95'][1]
    dispatched, undispatched = set(gate['dispatched_worlds']), set(gate['undispatched_reserved_worlds'])
    assert dispatched == set(range(968144, 968148))
    assert undispatched == set(range(968148, 968157)) and not dispatched & undispatched
    assert gate['main_collection_charged'] == 4*2841
    assert gate['sim_s_charged'] == 87399+2*1922+4*2841 == 102607 < 210000
    mirror = receipt['mirror']['sha256']
    assert receipt['current_state_sha256'] == mirror['state']
    assert gate['policy']['sha256'] == mirror['policy']
    assert gate['wm']['sha256'] == mirror['base']
    assert receipt['mini_wm_receipt']['exit'] == 0 and receipt['mini_wm_receipt']['train_log_rows'] == 2
    assert receipt['mini_steps_finite'] and not receipt['scientific_failure_retried']
    auto = ROOT/'tools/restart/auto_restart.py'
    assert hashlib.sha256(auto.read_bytes().replace(b'\r\n', b'\n')).hexdigest() == receipt['own_auto_pid']['script_sha256']
    for name in ('auto_restart.py', 'operate.py', 'validate_smoke.py', 'repair_smoke.py'):
        path = ROOT/'tools/restart'/name
        ast.parse(path.read_text(), filename=str(path))
    print('PASS: first-gate stop, charge conservation, undispatched worlds, mirror hashes, operator syntax/source')


if __name__ == '__main__':
    main()
