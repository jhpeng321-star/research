"""Read-only r12 recovery audit; reads named public receipts and owned /proc IDs."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path('/home/ubuntu/coopmwm20261106_clean3_r6')
CONTROL = Path('/home/ubuntu/cpuv_setup_20261105/clean3_chain_3e5bc9f9')
SOURCE = Path('/home/ubuntu/coopmwm20261106_clean3_src_3e5bc9f9')


def read(path):
    return json.loads(path.read_text()) if path.is_file() else None


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def owner(value):
    if not value:
        return None
    stat = Path('/proc')/str(value['pid'])/'stat'
    fields = stat.read_text().rsplit(')', 1)[1].split() if stat.exists() else []
    alive = bool(fields and fields[0] != 'Z' and fields[19] == str(value['start_ticks']))
    result = dict(value, alive=alive)
    if alive:
        result['state'] = fields[0]
        result['command'] = (stat.parent/'cmdline').read_bytes().replace(b'\0', b' ').decode()
    return result


def charges(root):
    path = root/'CHARGES.jsonl'
    phases, rows = {}, 0
    if path.is_file():
        for line in path.read_bytes().splitlines():
            row = json.loads(line)
            phases[row['phase']] = phases.get(row['phase'], 0)+row['sim_s']
            rows += 1
    return dict(rows=rows, phases=phases, total_sim_s=sum(phases.values()), sha256=sha(path))


manifest = read(SOURCE/'SOURCE_MANIFEST.json')
changed = [name for name, expected in manifest['files'].items() if sha(SOURCE/name) != expected]
assert not changed, changed
qualification = read(CONTROL.parent/'clean3_tests_3e5bc9f9'/'RESULT.json')
assert qualification['returncode'] == 0 and qualification['source_commit'] == manifest['commit']
receipt_names = ('PREPARED.json', 'SMOKE_STARTED.json', 'SMOKE_PASS.json', 'SMOKE_FAILED.json',
    'SMOKE_FIX_RECEIPTS_PASS.json', 'SMOKE_REPORTING_PASS.json', 'SMOKE_LEVER_RECEIPTS_PASS.json',
    'GATE_STARTED.json', 'GATE_PASS.json', 'GATE_FAILED.json', 'PHASE_COVERAGE_PASS.json',
    'LAUNCHED.json', 'STOP.json', 'FAULT.json', 'PARTIAL_LAUNCH.json')
receipts = {name: read(ROOT/name) for name in receipt_names if (ROOT/name).is_file()}
cfg = read(ROOT/'CONFIG.json')
prior_roots = [Path('/home/ubuntu/coopmwm20261106_clean3'+suffix)
               for suffix in ('', '_r2', '_r3', '_r4', '_r5')]
prior = {str(root): charges(root) for root in prior_roots}
events = [read(path) for path in sorted(CONTROL.glob('STATUS_*.json'))]
logs = {name: (CONTROL/name).read_text(errors='replace')[-10000:]
        for name in ('CHAIN.log', 'smoke.log', 'gate.log') if (CONTROL/name).is_file()}
launch = receipts.get('LAUNCHED.json') or {}
out = dict(schema='coopmwm.clean3.resume_w4_readonly.v1',
    timestamp=datetime.now(timezone.utc).isoformat(), root=str(ROOT), control=str(CONTROL),
    status=read(CONTROL/'CHAIN_STATUS.json'), events=events,
    chain_owner=owner(read(CONTROL/'CHAIN_STARTED.json')),
    launch_owners={role: owner(value) for role, value in launch.get('owners', {}).items()},
    source_commit=manifest['commit'], source_files_verified=len(manifest['files']),
    source_manifest_sha256=sha(SOURCE/'SOURCE_MANIFEST.json'), qualification=qualification,
    receipts=receipts, receipt_file_sha256={name: sha(ROOT/name) for name in receipts},
    root_charges=charges(ROOT), previous_clean3_roots=prior,
    completed_g1=read(CONTROL/'CLEAN3_GO_NOGO.json'),
    mined_admission=[read(path) for path in sorted(CONTROL.glob('MINED_ADMISSION_*.json'))],
    budget=(cfg or {}).get('clean3_budget'), restart=(cfg or {}).get('clean3_restart'),
    config_file_sha256=sha(ROOT/'CONFIG.json'), authority_sha256=sha(ROOT/'AUTHORITY.md'),
    chain_authority=read(CONTROL/'AUTHORITY.json'), logs=logs,
    new_dispatch=False, new_physics=False, process_signals=False)
print(json.dumps(out, sort_keys=True))
