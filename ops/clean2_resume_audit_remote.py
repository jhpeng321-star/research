"""Read-only continuation audit, sent through the accepted F6 transport.

Run with _sol/clean2_transport.py --source-package --research-env. This never
dispatches physics, changes training state, or clears a stop.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import runpy
import time

from tools.coopmwm import launch_clean2, launch_training
from tools.coopmwm.exact_grade import verify_sealed
from tools.coopmwm.protocol import digest
from tools.coopmwm.storage import ledger


root = Path('/home/ubuntu/coopmwm20261105_clean2')
source = Path(CLEAN2_SOURCE_ROOT)
cfg = launch_training.checked_config(root/'CONFIG.json')
assert Path(cfg['source_root']) == source
launch_clean2.verify_previous(cfg)
launch_training.health(root)
launched = launch_training.read(root/'LAUNCHED.json')
assert launched['runtime_sha256'] == digest(cfg)
assert launched['config_file_sha256'] == hashlib.sha256((root/'CONFIG.json').read_bytes()).hexdigest()
assert launched['source_manifest_sha256'] == cfg['source_manifest_sha256']
manifest = launch_training.read(source/'SOURCE_MANIFEST.json')
assert manifest['commit'] == launched['runtime_commit']
control = Path('/home/ubuntu/cpuv_setup_20261105')/('clean2_chain_'+manifest['commit'][:8])
chain_module = runpy.run_path(str(source/'ops/clean2_chain.py'), run_name='clean2_read_only_audit')
chain = chain_module['Chain'](source, control)
bindings = chain.check_receipts('gate', cfg)
assert bindings['gate_sha256'] == launched['gate_sha256']
assert bindings['smoke_sha256'] == launched['smoke_sha256']
qualification = launch_training.read(control.parent/('clean2_tests_'+manifest['commit'][:8])/'RESULT.json')
assert qualification['returncode'] == 0 and qualification['source_commit'] == manifest['commit']
status = launch_training.read(control/'CHAIN_STATUS.json')
assert status['step'] == 'MONITOR' and status['status'] == 'RUNNING'
dispatch = launch_training.read(control/'CHAIN_DISPATCH.json')


def owner_state(owner):
    path = Path('/proc')/str(owner['pid'])/'stat'
    fields = path.read_text().split() if path.exists() else []
    alive = bool(fields and fields[2] != 'Z' and fields[21] == str(owner['start_ticks']))
    assert alive, 'recorded owner exited: '+str(owner)
    return dict(owner, alive=alive)


owners = {role: owner_state(owner) for role, owner in launched['owners'].items()}
chain_owner = owner_state(dispatch)
rows = ledger(root)
phases = {phase: sum(row['sim_s'] for row in rows if row['phase'] == phase)
          for phase in sorted({row['phase'] for row in rows})}
assert all(amount <= cfg['clean2_budget']['budget']['phases'][phase] for phase, amount in phases.items())
spent = sum(phases.values())
previous = ledger(Path(cfg['clean2_restart']['clean1_root']))
global_spent = spent+sum(row['sim_s'] for row in previous)+960
assert global_spent <= 5000000
monitors = {}
for update in range(1, 25):
    path = root/f'LEARNING_MONITOR_u{update:03}.json'
    if not path.is_file():
        continue
    row = launch_training.read(path)
    verify_sealed(row, 'receipt_sha256')
    assert row['config_sha256'] == digest(cfg)
    safety = {name: sum(arm['metrics']['safety'][name] for arm in row['arms'])
              for name in ('collisions', 'teleports')}
    assert not any(safety.values())
    monitors[str(update)] = dict(receipt_sha256=row['receipt_sha256'], safety=safety,
        actions={arm['arm']: dict(action=arm['action'], diagnose=arm['diagnose']) for arm in row['arms']})
result = dict(schema='coopmwm.clean2.resume_audit.v1', timestamp=datetime.now(timezone.utc).isoformat(),
    unix=time.time(), read_only=True, launched=launched, source_files=len(manifest['files']),
    qualification=qualification, qualification_bindings=bindings,
    smoke=launch_training.read(root/'SMOKE_PASS.json'), gate=launch_training.read(root/'GATE_PASS.json'),
    chain_status=status, chain_owner=chain_owner, owners=owners, monitors=monitors,
    ledger_rows=len(rows), phase_charged_sim_s=phases, clean2_charged_sim_s=spent,
    global_charged_sim_s=global_spent, clean2_cap_sim_s=cfg['clean2_budget']['budget']['total'],
    clean1_preserved_stop_sha256=cfg['clean2_budget']['clean1_stop_sha256'])
print(json.dumps(result, sort_keys=True, allow_nan=False))
