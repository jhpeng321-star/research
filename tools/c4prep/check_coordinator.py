"""Detached, source-bound LONGMEM qualification with durable stage receipts."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from common import CODE, OUT, now, read, write_new


def main():
    commit = read(CODE / 'SOURCE.json')['commit']
    root = OUT / ('launch_checks_' + commit[:12])
    root.mkdir(exist_ok=False)
    stages = [
        ('CPU', ['tests/unit/test_c4launch.py', 'tests/unit/test_c4prep.py',
                 'tests/unit/test_antiexploit.py', 'tests/unit/test_longmem.py'],
         {**os.environ, 'CUDA_VISIBLE_DEVICES': ''}, 900),
        ('CUDA', ['tests/unit/test_longmem.py::test_saved_spill_shares_parameter_views_without_changing_gradients',
                  'tests/unit/test_longmem.py::test_production_actor_bridge_futures_and_entries[False-True-True-8-cuda]'],
         dict(os.environ), 600),
    ]
    results = []
    for name, tests, env, timeout in stages:
        # pytest creates basetemp itself, but requires its parent to exist.
        (root / name).mkdir(exist_ok=False)
        started = time.monotonic()
        argv = [sys.executable, '-B', '-m', 'pytest', '-q', '-o', 'addopts=',
                '--basetemp', str(root / name / 'tmp'), *tests]
        with (root / (name + '.log')).open('x') as log:
            child = subprocess.Popen(argv, cwd=CODE, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT)
            ticks = Path(f'/proc/{child.pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
            write_new(root / (name + '_PID.json'),
                      {'pid': child.pid, 'start_ticks': ticks, 'argv': argv, 't': now()})
            try:
                exit_code = child.wait(timeout=timeout)
                timed_out = False
            except subprocess.TimeoutExpired:
                # This is our unreaped direct child, never an inferred/peer PID.
                child.terminate()
                exit_code = child.wait(timeout=30)
                timed_out = True
        result = {'stage': name, 'exit': exit_code, 'timed_out': timed_out,
                  'elapsed_s': time.monotonic() - started, 't': now()}
        results.append(result)
        write_new(root / (name + '_RESULT.json'), result)
        if exit_code or timed_out:
            break
    receipt = {'source_commit': commit, 'passed': len(results) == 2 and
               all(r['exit'] == 0 and not r['timed_out'] for r in results),
               'results': results, 't': now(), 'root': str(root)}
    write_new(OUT / ('CHECKS_' + commit[:12] + '.json'), receipt)
    print(json.dumps(receipt), flush=True)
    return 0 if receipt['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
