"""One exclusive lean dispatch and >=600s resource sampling, own STOP only."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def write(path, data):
    with path.open('x') as stream:
        json.dump(dict(label='EXPLORATORY-LEAN', **data), stream, indent=1)


def sample():
    gpu = subprocess.check_output(['nvidia-smi','--query-gpu=memory.total,memory.free,memory.used',
        '--format=csv,noheader,nounits'], text=True).strip()
    apps = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory',
        '--format=csv,noheader,nounits'],text=True).strip()
    total, free, used = map(int, gpu.split(','))
    return dict(epoch=time.time(),total_MiB=total,free_MiB=free,used_MiB=used,apps=apps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--python', required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    write(root/'OWNERSHIP.json',dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],
        runtime=str(Path.cwd()),epoch=time.time()))
    # Last sample from the superseded own chain; preserve global query spacing.
    previous = Path('/root/autodl-tmp/c4/candB_pilot_final_run_EXPLORATORY_20261030/GPU_ADMISSION.jsonl')
    last = json.loads(previous.read_text().splitlines()[-1])['t']
    child = None
    while True:
        if (root/'STOP').exists():
            if child and (root/'pilot_EXPLORATORY-LEAN').is_dir():
                (root/'pilot_EXPLORATORY-LEAN/STOP').touch(exist_ok=True)
            if child is None: return
        if child and child.poll() is not None:
            write(root/'EXIT.json',dict(returncode=child.returncode,epoch=time.time()))
            return
        if time.time()-last >= 600:
            observation = sample()
            last = observation['epoch']
            with (root/'GPU.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(label='EXPLORATORY-LEAN',**observation))+'\n')
                stream.flush(); os.fsync(stream.fileno())
            if child is None and not observation['apps'] and observation['free_MiB']>=43000:
                env = dict(os.environ,RUNTIME=str(Path.cwd()),PYTHON=args.python,
                    PILOT_ADMISSION_FREE_MIB=str(observation['free_MiB']),PILOT_ADMISSION_EPOCH=str(last))
                output = root/'pilot_EXPLORATORY-LEAN'
                command=['bash','tools/candB/pilot_lean_launch.sh',str(output),'--exploratory-owner-20261030']
                with (root/'trainer.log').open('x') as log:
                    child=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                write(root/'TRAINER_OWNERSHIP.json',dict(pid=child.pid,
                    start_ticks=Path(f'/proc/{child.pid}/stat').read_text().rsplit(')',1)[1].split()[19],
                    command=command,output=str(output),epoch=time.time(),admission=observation))
        time.sleep(2)


if __name__ == '__main__':
    main()
