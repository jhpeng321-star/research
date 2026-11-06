"""Read public clean3 chain/ledger receipts without dispatching or signalling work."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time


def public(path):
    if not path.is_file():
        return None
    raw = path.read_bytes()
    return dict(sha256=hashlib.sha256(raw).hexdigest(), value=json.loads(raw))


def identity(value):
    if not isinstance(value, dict) or not value.get('pid'):
        return None
    pid = int(value['pid'])
    path = Path('/proc')/str(pid)
    try:
        fields = (path/'stat').read_text().rsplit(')', 1)[1].split()
        ticks = fields[19]
        return dict(pid=pid, state=fields[0], start_ticks=ticks,
                    same_identity=str(value.get('start_ticks')) == ticks,
                    command=(path/'cmdline').read_bytes().replace(b'\x00', b' ').decode())
    except FileNotFoundError:
        return dict(pid=pid, exited=True)


home = Path('/home/ubuntu')
controls = []
for control in sorted((home/'cpuv_setup_20261105').glob('clean3_chain_*')):
    dispatch = public(control/'CHAIN_DISPATCH.json')
    status = public(control/'CHAIN_STATUS.json')
    started = public(control/'CHAIN_STARTED.json')
    controls.append(dict(control=str(control),
        status=status, dispatch=dispatch,
        started=started, owner_process=identity(started['value']) if started else None,
        status_process=identity(status['value']) if status else None,
        process=identity(dispatch['value']) if dispatch else None))
roots = []
for root in sorted(home.glob('coopmwm20261106_clean3*')):
    if not root.is_dir() or '_src_' in root.name:
        continue
    ledger = root/'CHARGES.jsonl'
    phases = {}
    raw = ledger.read_bytes() if ledger.is_file() else b''
    for line in raw.splitlines():
        row = json.loads(line)
        phases[row['phase']] = phases.get(row['phase'], 0)+row['sim_s']
    launch = public(root/'LAUNCHED.json')
    owners = (launch or {}).get('value', {}).get('owners', {})
    monitors = sorted(root.glob('LEARNING_MONITOR_u[0-9][0-9][0-9].json'))
    roots.append(dict(root=str(root), charged_sim_s=sum(phases.values()), phases=phases,
        ledger_sha256=hashlib.sha256(raw).hexdigest(), ledger_rows=len(raw.splitlines()),
        receipts={name: public(root/name) for name in (
            'CHAIN_STATUS.json', 'LAUNCHED.json', 'STOP.json', 'FAULT.json',
            'SMOKE_PASS.json', 'SMOKE_LEVER_RECEIPTS_PASS.json', 'GATE_PASS.json',
            'PHASE_COVERAGE_PASS.json', 'CLEAN3_GO_NOGO.json')},
        owners={role: identity(owner) for role, owner in owners.items()},
        latest_monitor=public(monitors[-1]) if monitors else None))
qualifications = [dict(control=str(control), result=public(control/'RESULT.json'),
    dispatch=public(control/'DISPATCH.json'))
    for control in sorted((home/'cpuv_setup_20261105').glob('clean3_tests_*'))]
print(json.dumps(dict(schema='coopmwm.clean3.resume_readonly.v1',
    timestamp=datetime.now(timezone.utc).isoformat(), unix=time.time(),
    status='READ_ONLY_INVENTORY', controls=controls, roots=roots,
    qualifications=qualifications), sort_keys=True))
