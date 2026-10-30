import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

BASE = Path('/root/autodl-tmp/c4')
ROOT = BASE/'candB_rankcheck_d17274bd_parallel_20261030'
if sys.argv[1] in ('export', 'launch-plan', 'queue', 'queue-worker'):
    ROOT = BASE/'candB_rankcheck_3cd6481e_plan_20261030'
OLD = BASE/'candB_rankcheck_05f7bfba_graph_20261030/measurement'
def ticks(pid):
    return Path(f'/proc/{pid}/stat').read_text().split(') ')[1].split()[19]
def write(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=1)
if sys.argv[1] == 'queue':
    with (ROOT/'QUEUE.log').open('x') as log:
        p = subprocess.Popen([sys.executable, __file__, 'queue-worker'], cwd=ROOT,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    write(ROOT/'QUEUE_OWNERSHIP.json', dict(pid=p.pid, start_ticks=ticks(p.pid)))
    print('QUEUED own', p.pid)
elif sys.argv[1] == 'queue-worker':
    out = ROOT/'measurement_candB_zero'
    owned = json.loads((out/'OWNERSHIP.json').read_text())
    while Path(f"/proc/{owned['pid']}").exists() and ticks(owned['pid']) == str(owned['start_ticks']):
        time.sleep(5)
    document = json.loads((out/'RAW.json').read_text())
    assert document['shard_finished'] and document['zero_identity']['bitwise']
    subprocess.run([sys.executable, __file__, 'launch-plan',
                    'MAPPO_native_lanes+MAPPO_speed_partial'], check=True)
elif sys.argv[1] == 'stop-old':
    owned = json.loads((OLD/'CHILD.json').read_text())
    assert owned['pid'] == 953251 and ticks(953251) == str(owned['start_ticks'])
    write(OLD/'OWNED_AMENDMENT_STOP.json', dict(**owned, stopped_unix=time.time(),
        reason='MAIN 19:10 parallel/K4 deadline amendment; all saved arms retained'))
    os.kill(953251, signal.SIGTERM)
    print('Stopped exact owned incarnation', 953251)
elif sys.argv[1] == 'inspect':
    paths = ['restart20261026/runtime_longint/configs/c4_wm_loop14_r16_ft.toml',
        'restart20261026/arm/loop/wm/c02/wm_s.pt', 'wmcombined_wm2/wm_s.pt',
        'candA_prod_keyerr/policy_000191.pt', 'candA_keyerr/configs/execparity_cav_idm_v1.json',
        'restart20261026/arm/loop/longint/gate_wm2/HORIZON_GATE_WM2_REG.json',
        'candA_keyerr/authority_gen/BOUNDARY_WM2.json',
        'restart20261026/arm/loop/returns/pairing_c03.a1.json',
        'wmspeed20261028/assets/controllers/mappo.pt', 'mappores20261028/inputs/mappo_v5_final.pt']
    print([(p, (BASE/p).is_file()) for p in paths])
elif sys.argv[1] == 'export':
    source = BASE/'candB_rankcheck_d17274bd_parallel_20261030/measurement_CIRCLES/COMMAND.json'
    command = json.loads(source.read_text())
    args = command['argv'][:-4]
    args[args.index('--out')+1] = str(ROOT/'EXPORT_RECEIPT.json')
    args[args.index('--gpu-memory-fraction')+1] = '.04'
    args += ['--export-starts', str(ROOT/'TRAIN_STARTS.pt')]
    env = dict(os.environ, **command['environment'])
    env['PYTHONPATH'] = str(ROOT/'src')+':'+str(ROOT)
    subprocess.run(args, cwd=ROOT, env=env, check=True)
elif sys.argv[1] in ('launch', 'launch-plan'):
    arm = sys.argv[2]
    out = ROOT/('measurement_'+arm+(sys.argv[3] if len(sys.argv) > 3 else ''))
    assert ROOT.is_dir() and not out.exists()
    # Import only the definitions; the old launcher's top-level dispatch is not imported.
    code = (ROOT/'rankcheck_gpuA_launch.py').read_text().split("if '--worker' in sys.argv:")[0]
    namespace = {'__file__': str(ROOT/'rankcheck_gpuA_launch.py')}
    exec(compile(code, 'prior_launcher.py', 'exec'), namespace)
    args = namespace['arguments'](BASE/'wmspeed20261028/assets/controllers/mappo.pt')
    args[args.index('--device')+1] = 'cuda'
    args[args.index('--gpu-memory-fraction')+1] = '.19' if arm != 'PI' else '.40'
    args[args.index('--out')+1] = str(out/'RAW.json')
    args += ['--measurement-arms', *arm.split('+'), '--future-prefix', '4']
    if arm == 'candB_zero':
        args += ['--stop-after', '1']
    if sys.argv[1] == 'launch-plan':
        args += ['--input-starts', str(ROOT/'TRAIN_STARTS.pt')]
    env = namespace['environment']()
    env['CUDA_VISIBLE_DEVICES'] = '0'
    if not Path(args[0]).exists():
        args[0] = '/root/miniconda3/bin/python'
        mapping = {
            str(BASE/'restart20261026/arm/loop/wm/c02/wm_s.pt'): str(BASE/'wmcombined_wm2/wm_s.pt'),
            str(BASE/'wmspeed20261028/assets/controllers/mappo.pt'): str(ROOT/'inputs/mappo.pt'),
        }
        for i, v in enumerate(args):
            if v in mapping:
                args[i] = mapping[v]
            elif i and not Path(v).exists() and args[i-1] in ('--current-policy', '--current-policy-driver',
                                    '--horizon-gate', '--boundary', '--pairing-receipt', '--wm-config'):
                args[i] = str(ROOT/'inputs'/Path(v).name)
    out.mkdir()
    write(out/'PREFLIGHT.json', dict(time_unix=time.time(),
        nvidia_smi=subprocess.check_output(['nvidia-smi'], text=True)))
    write(out/'COMMAND.json', dict(argv=args, environment={k:v for k,v in env.items()
        if k.startswith(('C4_', 'WMS_', 'OMP_', 'MKL_', 'OPENBLAS_', 'CUDA_', 'PYTHON'))}))
    with (out/'run.log').open('x') as log:
        p = subprocess.Popen(args, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            preexec_fn=lambda: os.nice(19))
    write(out/'OWNERSHIP.json', dict(pid=p.pid, start_ticks=ticks(p.pid),
        started_unix=time.time(), argv=args))
    print('LAUNCHED', arm, p.pid, ticks(p.pid))
