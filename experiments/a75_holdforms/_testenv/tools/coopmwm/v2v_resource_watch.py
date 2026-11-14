"""Additive MAIN16:47 resource gate; qualified V2V numerical sources unchanged."""
import hashlib
import json
from pathlib import Path
import shutil
import time

CONTROL=Path('/home/ubuntu/coopmwm20261103_v2v_resource1647')
ROOT=Path('/home/ubuntu/coopmwm20261103_v2v2')
MOVER=Path('/home/ubuntu/coopmwm20261103_rolling1647_v3')
REQUIRED=6*1024**3


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def owner_alive(d):
    p=Path('/proc',str(d['pid']))
    if not p.exists():return False
    s=(p/'stat').read_text().split()
    return (s[21]==d['start_ticks'] and s[2]!='Z'
        and d['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip())


def decision(free,mover_alive,mover_stopped):
    return dict(ready=free>=REQUIRED and mover_alive and not mover_stopped,
        disk_free_bytes=free,required_free_bytes=REQUIRED,mover_alive=mover_alive,
        mover_stopped=mover_stopped,rule='MAIN2026-11-03 16:27/16:47 additive launch gate')


def gate(root,cfg):
    from tools.coopmwm.warm_health import assert_running
    for arm in ('B','R'):assert_running(root,arm)
    p=MOVER/'OWNER.json'
    alive=p.exists() and owner_alive(json.loads(p.read_text()))
    stopped=(MOVER/'STOP_GPU_SPACE.json').exists()
    return decision(shutil.disk_usage(root).free,alive,stopped)


def run():
    import sys
    authority=json.loads((CONTROL/'AUTHORITY.json').read_text())
    assert digest(__file__)==authority['source_sha256']
    cpu=json.loads((CONTROL/'CPU_RESULT.json').read_text());assert cpu['pass_']
    assert cpu['source_sha256']==digest(__file__)
    assert authority['amendment_sha256'] in (ROOT/'AUTHORITY.md').read_text()
    value=json.loads((ROOT/'V2V_RUNTIME.json').read_text())
    source=Path(value['config']['source_root'])
    sys.path[:0]=[str(source/'pytools'),str(source/'runtime_src'),'/home/ubuntu/e3final/runtime']
    from tools.coopmwm import v2v_live
    # configuration() still checks every inherited qualified source hash.
    v2v_live.configuration(ROOT)
    v2v_live.resource_gate=gate
    try:v2v_live.watch(ROOT)
    except BaseException as e:
        p=CONTROL/'FAILED.json'
        if not p.exists():
            with p.open('x') as f:json.dump(dict(error=repr(e),unix=time.time(),new_sim_s=0),f)
        raise


if __name__=='__main__':run()
