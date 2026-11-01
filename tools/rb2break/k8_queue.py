"""Frozen dev-only K8 continuation after owned K1 workers release their slots."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from tools.rb2break.factor_queue import alive, save
from tools.rb2break.run import DEV


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lane', type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    runtime = Path(__file__).resolve().parents[2]
    root = runtime.parent/'rb2break'
    predecessor = root/('actual2_cav_settings_s0' if args.lane == 0 else 'actual2_native_uncommanded_s0')
    receipt = json.loads((predecessor/'DISPATCH.json').read_text())
    assert receipt['owner'] == 'S-RB2-BREAK'
    deadline = time.monotonic()+12*3600
    while alive(receipt):
        assert time.monotonic() < deadline
        time.sleep(15)
    assert (predecessor/'COMPLETE.json').is_file(), 'failed predecessor: no automatic retry'
    out = root/f'devk8_native2_s{args.lane}'
    assert not out.exists(), 'preserve prior dispatch'
    while True:
        free = int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free',
                                           '--format=csv,noheader,nounits'], text=True).strip())
        if os.getloadavg()[0] < 16 and free > 12000:
            break
        assert time.monotonic() < deadline
        time.sleep(30)
    out.mkdir()
    argv = ['nice','-n','19',sys.executable,'-B','-u','-m','tools.rb2break.run',
        '--out',str(out),'--runtime-root',str(runtime),'--wm-config',str(runtime/'configs/wm_phys.toml'),
        '--model-name','RB2BREAK_devK8_native_actual2','--factor','native_uncommanded',
        '--dev-worlds',*map(str,DEV),'--futures',*map(str,range(8)),
        '--shard',str(args.lane),'--shards','2','--memory-gib','5']
    with (out/'worker.log').open('x') as log:
        child = subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,cwd=runtime)
    save(out/'DISPATCH.json',dict(owner='S-RB2-BREAK',pid=child.pid,argv=argv,unix=time.time(),
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        start_ticks=Path('/proc/'+str(child.pid)+'/stat').read_text().split()[21]))
    code = child.wait()
    save(out/'EXIT.json',dict(returncode=code,unix=time.time()))
    assert code == 0 and (out/'COMPLETE.json').is_file(), 'scientific failure is terminal'


if __name__ == '__main__':
    main()
