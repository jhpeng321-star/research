"""Receipt the existing durable lean baseline; never runs a forward or dispatch."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

from tools.candB.pilot_lean_observe import append, DIRECTORY, LEDGER
from tools.candB.pilot_wmfast_dispatch import ROOT


def main():
    code = f'''
import hashlib,json,time
from pathlib import Path
output=Path({ROOT!r})/'pilot_EXPLORATORY-LEAN'
names=['LEAN_STARTUP.json','frozen_scoring_baseline.json','pilot_STARTUP_REUSE_PROOF.json']
names += [f'pilot_startup_score_{{i}}.json' for i in range(5)]
doc=dict(epoch=time.time(),output=str(output),files={{}},sha256={{}})
for name in names:
    data=(output/name).read_bytes()
    doc['sha256'][name]=hashlib.sha256(data).hexdigest()
    doc['files'][name]=json.loads(data)
print(json.dumps(doc))
'''
    result = subprocess.run(['ssh', '-o', 'ConnectTimeout=10', 'gpuA',
        '/root/miniconda3/bin/python', '-'], input=code, text=True,
        capture_output=True, timeout=30)
    if result.returncode:
        raise RuntimeError(result.stderr)
    doc = json.loads(result.stdout)
    frozen = doc['files']['frozen_scoring_baseline.json']
    startup = doc['files']['LEAN_STARTUP.json']
    assert startup['label'] == 'EXPLORATORY-LEAN'
    assert startup['phase_origin_update'] == frozen['update'] == 191
    assert len(frozen['panel']) == 4
    for key in ('raw_dJ_vs_CTRL0_dense', 'raw_dJ_vs_CTRL0_long'):
        assert frozen['independent'][key] == 0.
    proof = doc['files']['pilot_STARTUP_REUSE_PROOF.json']
    assert proof['saved_independent_reused'] and proof['policy_equal'] and proof['config_equal']
    assert not proof['completed_baseline_recomputed']
    marker = DIRECTORY / 'pilot_repaired_STARTUP_ACK.json'
    with marker.open('x') as stream:
        json.dump(doc, stream, indent=1, allow_nan=False)
    text = (f'\n- S-PILOTB EXPLORATORY-LEAN BASELINE DURABLE epoch{doc["epoch"]}: '
        f'repaired startup {startup["startup_s"]:.6f}s ({startup["startup_s"]/60:.2f}min), '
        f'original phase191; four fixed native/gate starts, saved independent K8 score reused; '
        f'zero raw dense/300s dJ vs CTRL0. Frozen baseline SHA256 '
        f'{doc["sha256"]["frozen_scoring_baseline.json"]}; all five startup scores durable. '
        'Update1 has begun under original two-worker/.28 ES-stack-OFF recipe; '
        'WMFAST waits for >=1 finite completed update and clean checkpoint/old-process exit. '
        'No baseline recomputation or extra GPU query. Full receipt '
        '`E:/agent-work/c4-candB/tools/candB/pilot_repaired_STARTUP_ACK.json`.\n')
    append(DIRECTORY/'PILOT.md', text)
    append(LEDGER, text)
    registration = dict(epoch=time.time(), ledger=str(LEDGER),
        sha256=hashlib.sha256(LEDGER.read_bytes()).hexdigest())
    with (DIRECTORY/'pilot_repaired_STARTUP_REGISTERED.json').open('x') as stream:
        json.dump(registration, stream, indent=1)
    print(json.dumps(dict(startup_s=startup['startup_s'], baseline_sha256=doc['sha256'][
        'frozen_scoring_baseline.json'], receipt=str(marker), registration=registration)))


if __name__ == '__main__':
    main()
