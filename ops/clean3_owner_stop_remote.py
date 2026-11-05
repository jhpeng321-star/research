"""Register the 02:13 owner stop at the already sealed clean2 boundary."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from tools.coopmwm import launch_training as training
from tools.coopmwm.exact_grade import verify_sealed
from tools.coopmwm.protocol import digest
from tools.coopmwm.storage import ledger

root = Path('/home/ubuntu/coopmwm20261105_clean2')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
cfg = training.checked_config(root/'CONFIG.json')
launched = training.read(root/'LAUNCHED.json')
assert digest(cfg) == launched['runtime_sha256']
assert sha(root/'CONFIG.json') == launched['config_file_sha256']
owners = dict(launched['owners'])
owners['chain'] = dict(pid=296173, start_ticks='6536748')
for role, owner in owners.items():
    path = Path('/proc')/str(owner['pid'])/'stat'
    fields = path.read_text().split() if path.exists() else []
    assert not (fields and fields[2] != 'Z' and fields[21] == str(owner['start_ticks'])), role
assert (root/'STOP.json').is_file() and (root/'FAULT.json').is_file()
protected = {name: sha(root/name) for name in ('STOP.json', 'FAULT.json', 'CHARGES.jsonl', 'CONFIG.json')}
sealed = {}
for arm in ('A', 'B', 'R'):
    updates = []
    for u in range(1, 25):
        path = root/f'{arm}_UPDATE_{u:03}.json'
        if path.is_file():
            row = training.read(path)
            verify_sealed(row, 'update_sha256')
            assert row['checkpoint_sha256'] == sha(root/f'{arm}_checkpoint_{u:03}.pt')
            updates.append(u)
    assert updates and updates == list(range(1, max(updates)+1))
    sealed[arm] = max(updates)
assert len(set(sealed.values())) == 1, 'require a common sealed boundary'
now = datetime.now(timezone.utc).isoformat()
decision = dict(schema='coopmwm.clean2.owner_stop_clean3.v1',
    timestamp=now, owner_verbatim='yes, do that',
    registration='2026-11-06 02:13 - Owner: stop clean2 now; cpuV joins mining',
    reason='Owner stops clean2 at its next sealed boundary for G1 certification and mining before clean3',
    prior_fault_preserved=True, fault_is_not_relabelled_as_owner_stop=True,
    sealed_updates=sealed, original_owners_exited=True,
    protected_sha256=protected, training_resume_authorized=False,
    preserve_all_research_evidence=True)
path = root/'STOPPED_BY_OWNER.json'
if path.exists():
    existing = training.read(path)
    assert existing['registration'] == decision['registration']
    assert existing['protected_sha256'] == protected and existing['sealed_updates'] == sealed
    decision = existing
else:
    with path.open('x') as stream:
        json.dump(decision, stream, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
authority = root/'AUTHORITY.md'
marker = '\n## OWNER_STOP_CLEAN3_20261106_0213\n'
if marker not in authority.read_text():
    with authority.open('a') as stream:
        stream.write(marker+'Registered '+now+' (system clock). Owner (verbatim): "yes, do that". '
            'REGISTRATION.md 2026-11-06 02:13: stop clean2 at its next sealed boundary; '
            'keep evidence; cpuV cores shared with G1 certification first and mining the rest. '
            'All original owners and chain are exited; common sealed boundary '+str(sealed)+'. '
            'Existing budget FAULT/STOP remain unchanged; no release, resume or replay. '
            'STOPPED_BY_OWNER.json SHA256 '+sha(path)+'.\n')
        stream.flush()
        os.fsync(stream.fileno())
assert protected == {name: sha(root/name) for name in protected}
mining = Path('/home/ubuntu/coopmwm20261106_s_mining')
mining_rows = ledger(mining)
print(json.dumps(dict(timestamp=now, status='STOPPED_BY_OWNER_REGISTERED',
    root=str(root), decision=decision, owner_stop_sha256=sha(path),
    authority_sha256=sha(authority), clean2_charged_sim_s=sum(r['sim_s'] for r in ledger(root)),
    mining_charged_sim_s=sum(r['sim_s'] for r in mining_rows),
    mining_control={name: training.read(mining/name) for name in
        ('CHAIN_STATUS.json', 'MINING_STATUS.json', 'DISPATCH_STARTED.json') if (mining/name).is_file()}),
    sort_keys=True, allow_nan=False))
