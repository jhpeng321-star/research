"""Read only named clean2 control, checkpoint, monitor and lesson evidence."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from tools.coopmwm import launch_production as production, launch_training as training
from tools.coopmwm import replay_bank as rb, practice_starts as ps
from tools.coopmwm import track2_backend as p1, rerun_cpus as rc
from tools.coopmwm.exact_grade import verify_sealed
from tools.coopmwm.protocol import digest
from tools.coopmwm.storage import ledger

root = Path('/home/ubuntu/coopmwm20261105_clean2')
cfg = training.checked_config(root/'CONFIG.json')
launched = training.read(root/'LAUNCHED.json')
assert launched['runtime_sha256'] == digest(cfg)
assert launched['config_file_sha256'] == rc.sha(root/'CONFIG.json')


def state(owner):
    path = Path('/proc')/str(owner['pid'])/'stat'
    fields = path.read_text().split() if path.exists() else []
    return dict(owner, alive=bool(fields and fields[2] != 'Z'
        and fields[21] == str(owner['start_ticks'])))


controls = {}
for name in ('STOP.json', 'FAULT.json', 'CHAIN_STATUS.json'):
    path = root/name
    controls[name] = dict(sha256=rc.sha(path), value=training.read(path)) if path.is_file() else None
owners = {arm: state(owner) for arm, owner in launched['owners'].items()}
sealed = {}
for arm in ('A', 'B', 'R'):
    sealed[arm] = []
    for u in range(1, 25):
        path = root/f'{arm}_UPDATE_{u:03}.json'
        if not path.is_file():
            continue
        row = training.read(path)
        verify_sealed(row, 'update_sha256')
        checkpoint = root/f'{arm}_checkpoint_{u:03}.pt'
        assert row['checkpoint_sha256'] == rc.sha(checkpoint)
        sealed[arm].append(dict(update=u, update_sha256=row['update_sha256'],
            checkpoint_sha256=row['checkpoint_sha256']))
monitors = {}
for u in range(1, 25):
    path = root/f'LEARNING_MONITOR_u{u:03}.json'
    if path.is_file():
        row = training.read(path)
        verify_sealed(row, 'receipt_sha256')
        assert row['config_sha256'] == digest(cfg)
        monitors[str(u)] = dict(receipt_sha256=row['receipt_sha256'],
            arms=[dict(arm=a['arm'], action=a['action'], diagnose=a['diagnose'],
                safety=a['metrics']['safety']) for a in row['arms']])
rows = ledger(root)
phases = {phase: sum(r['sim_s'] for r in rows if r['phase'] == phase)
    for phase in sorted({r['phase'] for r in rows})}
lessons = {}
native = production.backend_config(cfg)
with p1._bank_view(native):
    frozen = rb.lesson_list(root, 'B', cfg.get('replay_sources', {}).get('B', ()))
    decisions = {r['verification_id']: r for r in rb.bank_decisions(root, 'B',
        sources=cfg.get('replay_sources', {}).get('B', ()))}
    links = ps._links(root, 'B')
    # Cost/spec construction only: no reservation, model loading or simulator.
    specs = []
    for vid in frozen['verification_ids']:
        receipt = decisions[vid]
        snap = rb._decision_snapshot(root, 'B', receipt, links)
        specs.append(rb.lesson_policy_spec(snap, receipt['chosen_plan'], receipt))
    for u in rc.SCHEDULE['B']['lessons']:
        path = root/f'evaluation/LESSON_u{u:03}.json'
        receipt = training.read(path) if path.is_file() else None
        if receipt:
            verify_sealed(receipt, 'receipt_sha256')
        lessons[str(u)] = dict(completed=receipt is not None,
            receipt_sha256=receipt['receipt_sha256'] if receipt else None,
            branches=len(specs), quoted_sim_s=sum(int(s['sim_s']) for s in specs),
            manifest=[dict(id=f'lesson_u{u:03}_{i:04}', sim_s=int(s['sim_s']),
                window=s['window']) for i, s in enumerate(specs)])
prior = ledger(Path(cfg['clean2_restart']['clean1_root']))
print(json.dumps(dict(schema='coopmwm.clean2.fault_audit.v1', read_only=True,
    timestamp=datetime.now(timezone.utc).isoformat(), config_sha256=digest(cfg),
    owners=owners, controls=controls, sealed_updates=sealed, monitors=monitors,
    lesson_schedule=lessons, lesson_phase_cap_sim_s=cfg['clean2_budget']['budget']['phases']['v2v_lessons'],
    phases=phases, clean2_charged_sim_s=sum(phases.values()),
    global_charged_sim_s=sum(phases.values())+sum(r['sim_s'] for r in prior)+960,
    ledger_sha256=hashlib.sha256((root/'CHARGES.jsonl').read_bytes()).hexdigest()),
    sort_keys=True, allow_nan=False))
