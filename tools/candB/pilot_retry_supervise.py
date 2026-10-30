"""One exclusive S4 retry dispatch and >=600s resource sampling, own STOP only."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from tools.candB.pilot_lean_supervise import sample, write

OLD_ROOT = Path('/root/autodl-tmp/c4/candB_pilot_run_WMFAST_EXPLORATORY-LEAN_20261031')


def absent_incarnation(path, name):
    owner = json.loads((path/name).read_text())
    proc = Path('/proc')/str(owner['pid'])/'stat'
    if proc.exists():
        assert proc.read_text().rsplit(')',1)[1].split()[19] != str(owner['start_ticks']), name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--python', required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    assert json.loads(Path('pilot_RETRY_CPU_RESULT.json').read_text())['returncode'] == 0
    for name in ('TRAINER_OWNERSHIP.json', 'OWNERSHIP.json'):
        if (OLD_ROOT/name).exists():
            absent_incarnation(OLD_ROOT, name)
    write(root/'OWNERSHIP.json', dict(pid=os.getpid(),
        start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
        runtime=str(Path.cwd()), epoch=time.time()))
    last = 0.
    child = output = None
    while True:
        if (root/'STOP').exists():
            if child is not None and output.is_dir():
                (output/'STOP').touch(exist_ok=True)
            if child is None or child.poll() is not None:
                return
        if child is not None and child.poll() is not None:
            write(root/'EXIT.json', dict(returncode=child.returncode, epoch=time.time()))
            return
        if time.time()-last >= 600:
            observation = sample()
            last = observation['epoch']
            with (root/'GPU.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(label='EXPLORATORY-LEAN', **observation))+'\n')
                stream.flush(); os.fsync(stream.fileno())
            if child is None and observation['free_MiB'] >= 46000:
                output = root/'pilot_EXPLORATORY-LEAN_RETRY0300'
                env = dict(os.environ, RUNTIME=str(Path.cwd()), PYTHON=args.python,
                    PILOT_ADMISSION_FREE_MIB=str(observation['free_MiB']),
                    PILOT_ADMISSION_EPOCH=str(last))
                command = [args.python, '-B', '-u', '-m', 'tools.candB.pilot_retry_launch', str(output)]
                with (root/'trainer.log').open('x') as log:
                    child = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                        stdout=log, stderr=subprocess.STDOUT)
                write(root/'TRAINER_OWNERSHIP.json', dict(pid=child.pid,
                    start_ticks=Path(f'/proc/{child.pid}/stat').read_text().rsplit(')',1)[1].split()[19],
                    command=command, output=str(output), epoch=time.time(), admission=observation))
        time.sleep(2)


if __name__ == '__main__':
    main()
