"""Exclusive corrected continuation controller; waits for existing trainer/supervisor."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from tools.candB.pilot_lean_supervise import sample, write


def alive(owner):
    proc = Path('/proc')/str(owner['pid'])/'stat'
    return proc.exists() and proc.read_text().rsplit(')',1)[1].split()[19] == str(owner['start_ticks'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--python', required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    write(root/'WMFAST_CONTROL_OWNERSHIP.json', dict(pid=os.getpid(),
        start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
        runtime=str(Path.cwd()), epoch=time.time(), no_dispatch_before_clean_phase1=True))
    try:
        assert json.loads(Path('pilot_WMFAST_CPU_RESULT.json').read_text())['returncode'] == 0
        prior_supervisor = json.loads((root/'OWNERSHIP.json').read_text())
        prior_trainer = json.loads((root/'repair_TRAINER_OWNERSHIP.json').read_text())
        # The original supervisor refuses a row without independent_J and exits
        # after its already-armed repair's clean phase1 STOP. It owns GPU polling
        # until then. Never duplicate its trainer or continuation.
        while alive(prior_supervisor) or alive(prior_trainer):
            if (root/'WMFAST_CONTROL_STOP').exists():
                return
            time.sleep(2)
        assert not (root/'continue_TRAINER_OWNERSHIP.json').exists(), 'existing continuation; recover it'
        from tools.candB.pilot_wmfast import validate_parent
        parent = root/'pilot_EXPLORATORY-LEAN'
        prior, origin = validate_parent(parent)
        assert prior['updates'][-1]['phase_update'] == 1
        write(root/'WMFAST_CLEAN_BOUNDARY.json',dict(parent=str(parent), phase_origin=origin,
            phase_update=1, baseline_recomputed=False, epoch=time.time()))
        last = json.loads((root/'GPU.jsonl').read_text().splitlines()[-1])['epoch']
        child = None
        output = root/'pilot_WMFAST_EXPLORATORY-LEAN'
        while True:
            if (root/'WMFAST_CONTROL_STOP').exists() or (root/'STOP').exists():
                if child and output.is_dir():
                    (output/'STOP').touch(exist_ok=True)
                if child is None:
                    return
            if child and child.poll() is not None:
                write(root/'EXIT.json',dict(returncode=child.returncode,epoch=time.time()))
                return
            if time.time()-last >= 600:
                observation = sample()
                last = observation['epoch']
                with (root/'GPU.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(label='EXPLORATORY-LEAN',**observation))+'\n')
                    stream.flush(); os.fsync(stream.fileno())
                if child is None and observation['free_MiB'] >= 38000:
                    env = dict(os.environ,RUNTIME=str(Path.cwd()),PYTHON=args.python,
                        PILOT_ADMISSION_FREE_MIB=str(observation['free_MiB']),PILOT_ADMISSION_EPOCH=str(last))
                    command = [args.python,'-B','-u','-m','tools.candB.pilot_wmfast_launch','continue',
                        '--parent',str(parent),str(output)]
                    with (root/'continue_trainer.log').open('x') as log:
                        child = subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                    write(root/'continue_TRAINER_OWNERSHIP.json',dict(pid=child.pid,
                        start_ticks=Path(f'/proc/{child.pid}/stat').read_text().rsplit(')',1)[1].split()[19],
                        runtime=str(Path.cwd()),command=command,output=str(output),epoch=time.time(),admission=observation))
            time.sleep(2)
    except Exception as error:
        import traceback
        write(root/'WMFAST_CONTROL_FAILURE.json',dict(error=str(error),traceback=traceback.format_exc(),epoch=time.time()))
        raise


if __name__ == '__main__':
    main()
