"""Exclusive repair->clean update1->WMFAST full-resume chain; no repeated baseline."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from tools.candB.pilot_lean_supervise import sample, write
from tools.candB.pilot_lean_deploy import ROOT as OLD_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--python', required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    assert json.loads(Path('pilot_WMFAST_CPU_RESULT.json').read_text())['returncode'] == 0
    old = Path(OLD_ROOT)
    assert (old/'EXIT.json').exists(), 'old trainer/supervisor must finish before recovery'
    for name in ('TRAINER_OWNERSHIP.json', 'OWNERSHIP.json'):
        owner = json.loads((old/name).read_text())
        proc = Path('/proc')/str(owner['pid'])/'stat'
        if proc.exists():
            assert proc.read_text().rsplit(')',1)[1].split()[19] != str(owner['start_ticks'])
    write(root/'OWNERSHIP.json', dict(pid=os.getpid(), start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
        runtime=str(Path.cwd()), epoch=time.time()))
    last = json.loads((old/'GPU.jsonl').read_text().splitlines()[-1])['epoch']
    child = None
    mode = 'repair'
    parent = old/'pilot_EXPLORATORY-LEAN'
    while True:
        if (root/'STOP').exists():
            if child and output.is_dir():
                (output/'STOP').touch(exist_ok=True)
            if child is None:
                return
        if child and child.poll() is not None:
            write(root/f'{mode}_EXIT.json', dict(returncode=child.returncode, epoch=time.time()))
            if mode == 'repair' and not (root/'STOP').exists():
                from tools.candB.pilot_wmfast import validate_parent
                prior, origin = validate_parent(output)
                assert prior['updates'][-1]['phase_update'] == 1
                assert prior['seed'] == 965140
                write(root/'BOUNDARY.json', dict(phase_origin=origin, baseline_recomputed=False,
                    phase_update=1, parent=str(output)))
                mode, parent, child = 'continue', output, None
            else:
                write(root/'EXIT.json', dict(returncode=child.returncode, epoch=time.time()))
                return
        if time.time()-last >= 600:
            observation = sample()
            last = observation['epoch']
            with (root/'GPU.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(label='EXPLORATORY-LEAN', **observation))+'\n')
                stream.flush(); os.fsync(stream.fileno())
            if child is None and observation['free_MiB'] >= 38000:
                env = dict(os.environ, RUNTIME=str(Path.cwd()), PYTHON=args.python,
                    PILOT_ADMISSION_FREE_MIB=str(observation['free_MiB']), PILOT_ADMISSION_EPOCH=str(last))
                output = root/('pilot_EXPLORATORY-LEAN' if mode == 'repair' else 'pilot_WMFAST_EXPLORATORY-LEAN')
                command = [args.python, '-B', '-u', '-m', 'tools.candB.pilot_wmfast_launch',
                    mode, '--parent', str(parent), str(output)]
                with (root/f'{mode}_trainer.log').open('x') as log:
                    child = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                write(root/f'{mode}_TRAINER_OWNERSHIP.json', dict(pid=child.pid,
                    start_ticks=Path(f'/proc/{child.pid}/stat').read_text().rsplit(')',1)[1].split()[19],
                    command=command, output=str(output), epoch=time.time(), admission=observation))
        time.sleep(2)


if __name__ == '__main__':
    main()
