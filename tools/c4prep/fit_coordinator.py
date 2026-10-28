"""Bound the owned historical fit child to one hour, with explicit fallback."""
import argparse
from pathlib import Path
import subprocess
import sys
import time

from common import CODE, write_new, now

p = argparse.ArgumentParser()
p.add_argument('--output', required=True)
a = p.parse_args()
dest = Path(a.output)
begun = time.monotonic()
with (dest / 'replay.log').open('x') as log:
    child = subprocess.Popen([sys.executable, '-B', str(CODE / 'tools/c4prep/calibrate.py'),
        'replay', '--entrant', 'on', '--launch-bounded', '--output', str(dest)],
        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    ticks = Path(f'/proc/{child.pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
    write_new(dest / 'REPLAY_PID.json', {'pid': child.pid, 'start_ticks': ticks, 't': now()})
    try:
        exit_code = child.wait(timeout=3600)
        reason = f'fit child exited {exit_code}'
    except subprocess.TimeoutExpired:
        # Popen is the owned unreaped child: PID cannot have been recycled.
        child.terminate()
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
        exit_code, reason = 124, 'authorized 3600s fit budget exhausted'
if not (dest / 'FIT_DECISION.json').exists():
    write_new(dest / 'FIT_DECISION.json', {'native_anchor': False, 'pessimistic_credit': True,
        'quantile': .1, 'k': 8, 'reason': reason, 'fitted_scale': None,
        'cycles': [0, 1, 2], 'elapsed_s': time.monotonic() - begun})
write_new(dest / 'RESULT.json', {'exit': exit_code, 'elapsed_s': time.monotonic() - begun})
