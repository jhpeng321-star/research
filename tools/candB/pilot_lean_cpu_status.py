"""Recover only named lean CPU receipts after a disconnected test caller."""
import subprocess
import json
from pathlib import Path

code = r'''
from pathlib import Path
import fcntl,json
p=Path('/root/autodl-tmp/c4/speedprof_CPU_pilotlean_2350')
for f in sorted(p.glob('PROBE_*.json'))[-1:]:
    print(f.name, f.read_text())
    doc=json.loads(f.read_text())
    if doc.get('status')=='completed': print((p/doc['log']).read_text())
with (p/'CPU.lock').open('a') as lock:
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print('CPU_TEST_LOCK_FREE')
    except BlockingIOError:
        print('CPU_TEST_STILL_RUNNING')
'''
if __name__ == '__main__':
    result = subprocess.run(['ssh','gpuA','/root/miniconda3/bin/python','-'],input=code,
        capture_output=True,text=True,timeout=20)
    print(result.stdout)
    print(result.stderr)
    with Path(__file__).with_name('pilot_lean_CPU_recovered.json').open('x') as stream:
        json.dump(dict(stdout=result.stdout,stderr=result.stderr,returncode=result.returncode,
            earlier_full_test='PROBE_1793375453325102700: 24 passed/3 missing-fixture-field failures; real K4 dense/RNG proof PASS'),stream,indent=1)
