"""Bounded serial continuation of predeclared factors; never reads efficacy to schedule."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def save(path, obj):
    with path.open('x') as stream:
        json.dump(obj, stream, indent=1)


def alive(receipt):
    path = Path('/proc')/str(receipt['pid'])/'stat'
    try:
        fields = path.read_text().split()
        return (fields[2] != 'Z' and fields[21] == receipt['start_ticks']
                and Path('/proc/sys/kernel/random/boot_id').read_text().strip() == receipt['boot_id'])
    except FileNotFoundError:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait-for', required=True)
    parser.add_argument('--factors', nargs='+', required=True)
    parser.add_argument('--lane', type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    runtime = Path(__file__).resolve().parents[2]
    root = runtime.parent/'rb2break'
    wait = Path(args.wait_for)
    assert wait.parent == root
    old = json.loads((wait/'DISPATCH.json').read_text())
    assert old['owner'] == 'S-RB2-BREAK'
    deadline = time.monotonic()+12*3600
    while alive(old):
        assert time.monotonic() < deadline
        time.sleep(15)
    assert (wait/'COMPLETE.json').is_file(), 'prior job failed; no automatic scientific retry'
    from tools.rb2break.run import FACTORS, DEV
    assert set(args.factors) <= set(FACTORS) and len(set(args.factors)) == len(args.factors)
    completed = []
    for factor in args.factors:
        out = root/f'screen_{factor}_s0'
        assert not out.exists(), 'preserve existing dispatch'
        while True:
            free = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free',
                '--format=csv,noheader,nounits'], text=True).strip())
            if os.getloadavg()[0] < 16 and free > 12000:
                break
            assert time.monotonic() < deadline
            time.sleep(30)
        out.mkdir()
        argv = ['nice', '-n', '19', sys.executable, '-B', '-u', str(runtime/'tools/rb2break/run.py'),
            '--out', str(out), '--runtime-root', str(runtime), '--wm-config', str(runtime/'configs/wm_phys.toml'),
            '--model-name', 'RB2BREAK_screen_'+factor, '--factor', factor,
            '--dev-worlds', *map(str, DEV), '--futures', '0', '--memory-gib', '5']
        with (out/'worker.log').open('x') as log:
            child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, cwd=runtime)
        receipt = dict(owner='S-RB2-BREAK', pid=child.pid, argv=argv,
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            start_ticks=Path('/proc/'+str(child.pid)+'/stat').read_text().split()[21], unix=time.time())
        save(out/'DISPATCH.json', receipt)
        result = child.wait()
        save(out/'EXIT.json', dict(returncode=result, unix=time.time()))
        assert result == 0 and (out/'COMPLETE.json').is_file(), 'failure is terminal'
        completed.append(factor)
        print('COMPLETE_FACTOR', factor, flush=True)
    save(root/f'QUEUE_{args.lane}_COMPLETE.json', dict(factors=completed, unix=time.time()))


if __name__ == '__main__':
    main()
