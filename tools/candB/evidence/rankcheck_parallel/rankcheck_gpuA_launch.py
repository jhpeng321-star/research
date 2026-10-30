"""Operational wrapper only: one serial WM measurement, no optimiser/SUMO entry."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
BASE = Path('/root/autodl-tmp/c4')
PYTHON = BASE/'candB_rankcheck_0264b967_20261030/venv/bin/python'
OUT = ROOT/'measurement'
BAD = re.compile(r'sealed|holdout|heldout|cal_split|chunk5|j4b|inc1|hetdata_truth|vtype', re.I)
WM_SHA = 'b8797242c6b736b57eb17b1aed77d2cf01d813770342f9e17ba33dc5b8b33436'
MAPPO_SHA = 'aecd7d70ccce2995a10d03d9400f587040b1d3b435eebeab2561768ba57c3564'

def digest(p):
    assert not BAD.search(str(p)), p
    with open(p, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def write(p, d):
    with p.open('x') as stream:
        json.dump(d, stream, indent=1)

def ticks(pid):
    return Path(f'/proc/{pid}/stat').read_text().split(') ')[1].split()[19]

def locate_mappo():
    matches = []
    for parent, dirs, files in os.walk(BASE):
        dirs[:] = [d for d in dirs if d not in ('data', 'data_cache', 'venv', '.git')
                   and not BAD.search(str(Path(parent)/d))]
        for name in files:
            path = Path(parent)/name
            if (name.endswith('.pt') and not BAD.search(str(path))
                    and ('mappo' in str(path).lower() or name == 'policy_00170.pt')
                    and path.stat().st_size < 2_000_000 and digest(path) == MAPPO_SHA):
                matches.append(path)
    if not matches:
        raise FileNotFoundError('registered MAPPO aecd7d70 not found in allowed c4 tree')
    return sorted(matches, key=lambda p: len(str(p)))[0]

def arguments(mappo):
    loop = BASE/'restart20261026/arm/loop'
    return [str(PYTHON), '-B', '-u', '-m', 'tools.candB.rankcheck', '--measure-only',
        '--wm', str(loop/'wm/c02/wm_s.pt'), '--wm-config',
        str(BASE/'restart20261026/runtime_longint/configs/c4_wm_loop14_r16_ft.toml'),
        '--mappo', str(mappo), '--mappo-speed', str(ROOT/'rankcheck_mappospeed_policy_00025.pt'),
        '--current-policy', str(BASE/'candA_prod_keyerr/policy_000191.pt'),
        '--current-policy-driver', str(BASE/'candA_keyerr/configs/execparity_cav_idm_v1.json'),
        '--horizon-gate', str(loop/'longint/gate_wm2/HORIZON_GATE_WM2_REG.json'),
        '--boundary', str(BASE/'candA_keyerr/authority_gen/BOUNDARY_WM2.json'),
        '--pairing-receipt', str(loop/'returns/pairing_c03.a1.json'),
        '--cct2', f'authority/CCT2_{WM_SHA}_mix.json',
        '--n-starts', '16', '--device', 'cpu', '--gpu-memory-fraction', '.40',
        '--out', str(OUT/'CANDB_RANKCHECK.json')]

def environment():
    env = dict(os.environ)
    env.update(PYTHONPATH=str(ROOT/'src')+':'+str(ROOT), C4_ALLOWED_ROOTS=str(BASE),
        CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1',
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',
        WMS_DATA_V5_NATIVE_Q=str(BASE/'restart20261026_inputs/data/v5_native_q'),
        WMS_DATA_GENRAND_PLAIN_Q=str(BASE/'restart20261026_inputs/data/genrand_plain_q'),
        C4_PROBES_DEVICE='cpu')
    # No parallel-forward/VJP pool: the entire measurement has one .40 allocator.
    for name in list(env):
        if name.startswith('WMS_SPEED_'):
            del env[name]
    # Existing qualified exact forward optimizations from tools/candB/launch.sh;
    # keep every process-pool setting OFF so aggregate allocator remains .40.
    env.update(WMS_WMSPEED_DEDUPE_MOBIL='1', WMS_WMSPEED_MOBIL_LAZY_SAFE='1',
        WMS_SPEED_PIVOT_DIAG_OFF='1', WMS_SPEED_FAST_MERGE_LANE='1',
        WMS_SPEED_FAST_CAV_DRIVE='1', WMS_SPEED_FAST_LONG_HISTORY='1',
        WMS_SPEED_GRAPH_CVAR='1')
    return env

if '--worker' in sys.argv:
    args = json.loads((OUT/'COMMAND.json').read_text())['argv']
    started = time.time()
    child = subprocess.Popen(args, cwd=ROOT, env=environment())
    write(OUT/'CHILD.json', dict(pid=child.pid, start_ticks=ticks(child.pid), argv=args,
        started_unix=started, gpu_fraction=.40))
    code = child.wait()
    write(OUT/'EXIT.json', dict(returncode=code, started_unix=started,
        ended_unix=time.time(), elapsed_s=time.time()-started))
    sys.exit(code)

assert ROOT.is_dir() and not OUT.exists()
for name, sha in json.loads((ROOT/'SOURCE_MANIFEST.json').read_text()).items():
    assert digest(ROOT/name) == sha, name
mappo = locate_mappo()
print('MAPPO', mappo, flush=True)
OUT.mkdir()
gpu = subprocess.check_output(['nvidia-smi'], text=True)
write(OUT/'PREFLIGHT.json', dict(time_unix=time.time(), nvidia_smi=gpu,
    source=json.loads((ROOT/'SOURCE.json').read_text()), launcher_sha256=digest(__file__),
    mappo=str(mappo), mappo_sha256=digest(mappo),
    speed_sha256=digest(ROOT/'rankcheck_mappospeed_policy_00025.pt')))
args = arguments(mappo)
write(OUT/'COMMAND.json', dict(argv=args, environment={k: v for k, v in environment().items()
    if k.startswith(('C4_', 'WMS_', 'OMP_', 'MKL_', 'OPENBLAS_', 'PYTORCH_'))
    or k in ('PYTHONPATH', 'CUDA_VISIBLE_DEVICES')}))
with (OUT/'run.log').open('x') as log:
    child = subprocess.Popen([str(PYTHON), str(Path(__file__).resolve()), '--worker'],
        cwd=ROOT, env=environment(), stdin=subprocess.DEVNULL, stdout=log,
        stderr=subprocess.STDOUT, start_new_session=True, preexec_fn=lambda: os.nice(19))
write(OUT/'OWNERSHIP.json', dict(pid=child.pid, start_ticks=ticks(child.pid),
    time_unix=time.time(), gpu_fraction=.40, runtime=str(ROOT)))
print('LAUNCHED', child.pid, 'runtime', ROOT, flush=True)
