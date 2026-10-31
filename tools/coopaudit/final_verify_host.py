"""Read-only end-of-panel verification; own RUN lock, ledger and receipts only."""
import fcntl
import hashlib
import json
from pathlib import Path
import time

root=Path('/root/coopaudit20261031')
selection=json.loads((root/'CORRECTED_DESIGN.json').read_text())
storage=Path(selection['storage_root'])
assert json.loads((root/'V2_MAIN_COMPLETE.json').read_text())['ok']
assert not (root/'STOP').exists()
with (root/'RUN.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    ledger=[json.loads(s) for s in (root/'accounting.jsonl').read_text().splitlines()]
    assert len(ledger)==len({r['id'] for r in ledger})
    actual_new=0
    full_prefix_gates=0
    traces=0
    complete=0
    for directory in sorted(storage.glob('v2_*')):
        if not directory.is_dir():
            continue
        assert not (directory/'failure.json').exists()
        result=json.loads((directory/'result.json').read_text())
        assert result['ok'] and result['branch_mode']=='full_prefix_rerun'
        actual_new+=result['actual_sim_s']
        gates=list(directory.glob('branch_*.json'))
        full_prefix_gates+=len(gates)
        for p in gates:
            gate=json.loads(p.read_text())
            assert gate['actual_sim_s']==900 and gate['prefix_verified']
        traces+=len(list((directory/'traces').glob('*.jsonl.gz')))
        if directory.name.startswith('v2_s'):
            complete+=1
    charged=sum(r['sim_s'] for r in ledger)
    assert actual_new==222300 and charged==243780 and complete==28 and full_prefix_gates==traces==247
    pid=json.loads((root/'v2-main_PID.json').read_text())
    stat=Path('/proc')/str(pid['pid'])/'stat'
    active=stat.exists() and stat.read_text().split()[21]==pid['proc_start_ticks'] and stat.read_text().split()[2]!='Z'
    assert not active
print(json.dumps(dict(ok=True,status='COMPLETE',unix=time.time(),main_worlds=complete,
                      full_prefix_gates=full_prefix_gates,trace_files=traces,
                      actual_total_sim_s=21180+actual_new,charged_sim_s=charged,cap_sim_s=500000,
                      retained_unused_reservation_sim_s=300,run_lock_free=True,own_supervisor_inactive=True,
                      source=pid,ledger_sha256=hashlib.sha256((root/'accounting.jsonl').read_bytes()).hexdigest()),indent=1))
