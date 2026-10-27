"""Sharded invocation of the unchanged registered gate's panel, rollout and statistics.

Each group keeps its original global index/CRN seed. No SUMO, training or GPU calls.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HOME = Path.home()/'longhzn'
ROOT = HOME/'runtime'
HORIZONS = (60,90,120,180,300)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_gate():
    os.environ['COOPW_ROOT'] = str(ROOT)
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.path[:0] = [str(ROOT),str(ROOT/'src'),str(ROOT/'tools'),str(ROOT/'tools/r10b_checks')]
    spec = importlib.util.spec_from_file_location('registered_gate', HOME/'horizon_gate.py')
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    import torch
    torch.set_num_threads(1)
    return gate


def priority():
    # Read-only process metadata: never signal or manipulate peer processes.
    text = subprocess.check_output(['ps','-eo','pid=,comm=,args='],text=True)
    active = []
    for line in text.splitlines():
        parts = line.strip().split(None,2)
        if len(parts) < 3:
            continue
        pid, comm, command = parts
        if comm in ('sumo','sumo-gui') or (('collect' in command or 'fastloop' in command)
                and ('/c4/main14/' in command or '/c4/coopoff220261027/' in command)
                and 'longhzn' not in command):
            active.append(int(pid))
    return active


def wait_priority():
    while priority() or os.getloadavg()[0] > 24:
        time.sleep(10)


def worker(shard, shards):
    os.nice(19)
    gate = load_gate()
    acc = gate._accept()
    original = json.loads((HOME/'original_gate.json').read_text())
    assert sha(ROOT/'tools/coopw/horizon_gate.py') == sha(HOME/'horizon_gate.py')
    assert sha(ROOT/'configs/c4_wm_loop14_r16_ft.toml') == sha(HOME/'wm1_config.toml')
    acc.check_roots([HOME/'pairs300'])
    configs = {'wm0': ROOT/'configs/c4_wm_loop14_r16_ft.toml',
               'wm1': ROOT/'configs/c4_wm_loop14_r16_ft.toml', 'r16': HOME/'r16_config.json'}
    models = {}
    for name, config in configs.items():
        wait_priority()
        models[name] = acc.load_arm(config, HOME/(name+'.pt'))
    ds = acc.build_sets(models['wm1'][0], [HOME/'pairs300'],600)[0]
    panel = gate.panel_groups(ds,96,20261026)
    assert [key for key,_ in panel] == [r['pair_key'] for r in original['rows']]
    assert all(len(group['SEED']) == original['rows'][g]['arms'] for g,(_,group) in enumerate(panel))
    out = HOME/'rows'
    out.mkdir(exist_ok=True)
    for g,(key,group) in enumerate(panel):
        if g % shards != shard:
            continue
        for name,(_,settings,model) in models.items():
            target = out/f'{name}_{g:03d}.json'
            if target.exists():
                continue
            wait_priority()
            start = time.time()
            runs = []
            for index in [group['B'], *group['SEED']]:
                wait_priority()
                r = acc.free_rollout(model,settings,ds[index],600,gate.SEED0+7919*g)
                assert len(r['speed']) == 601 and len(r['rec_speed']) == 601
                sel = r['t0_present'] & ~r['is_cav']
                runs.append({str(h):(acc.hdv_delay(r['speed'],r['mask'],sel,2*h),
                                    acc.hdv_delay(r['rec_speed'],r['rec_mask'],sel,2*h)) for h in HORIZONS})
            row = {'group':g,'pair_key':key,'arms':len(runs)-1,'model':name,
                   'dJ':{str(h): [[run[str(h)][0]-runs[0][str(h)][0],
                                    run[str(h)][1]-runs[0][str(h)][1]] for run in runs[1:]] for h in HORIZONS},
                   'elapsed_s':time.time()-start,'seed':gate.SEED0+7919*g,'steps':600}
            tmp = target.with_suffix('.tmp')
            tmp.write_text(json.dumps(row))
            tmp.replace(target)
            print(json.dumps({'model':name,'group':g,'seconds':row['elapsed_s']}),flush=True)


def coordinate(box):
    assert box in (0,1)
    os.nice(19)
    assert not priority(), 'live collection: launch later'
    env = {**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1',
           'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
    pids = []
    children = []
    for shard in range(box*16,(box+1)*16):
        log = (HOME/f'worker{shard:02d}.log').open('x')
        child = subprocess.Popen([sys.executable,'-B',str(HOME/'run_cpu.py'),'worker',str(shard),'32'],
                                 env=env,stdout=log,stderr=subprocess.STDOUT)
        ticks = Path(f'/proc/{child.pid}/stat').read_text().rsplit(')',1)[1].split()[19]
        pids.append({'pid':child.pid,'start_ticks':ticks,'shard':shard})
        children.append(child)
    (HOME/'WORKERS.json').write_text(json.dumps(pids,indent=1))
    codes = [child.wait() for child in children]
    (HOME/'DONE.json').write_text(json.dumps({'box':box,'exit_codes':codes,'all_ok':all(c==0 for c in codes)}))
    return int(any(codes))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode',choices=('worker','coordinate'))
    p.add_argument('index',type=int)
    p.add_argument('shards',type=int,nargs='?',default=32)
    a = p.parse_args()
    if a.mode == 'worker':
        worker(a.index,a.shards)
    else:
        sys.exit(coordinate(a.index))
