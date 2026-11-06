"""Launch only the registered offline CPU inference process, once."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import resource
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path('/root/autodl-tmp/c4/v5')
REPO = ROOT / 'repo_r13e_8c7ff0a1'
WORK = ROOT / 'diag_emerg'
OUT = WORK / 'offline_train48_workers4'
assert ROOT.is_dir() and WORK.is_dir() and REPO.is_dir()
assert not OUT.exists() and not (WORK / 'offline_owned_process.json').exists()
assert shutil.disk_usage(ROOT).free > 100_000_000
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (min(65536, hard), hard))
sys.path.insert(0, str(REPO))
from tools.r13e_checks import offline as study  # noqa: E402 (authorized archive import)
study.cpu_limits()
print(json.dumps(study.source_provenance()), flush=True)
env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
           OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', WMS_JAM_WORKERS='4')
command = ['/root/miniconda3/bin/python', '-B', '-u', str(REPO/'tools/r13e_checks/offline.py'),
           '--checkpoint', str(ROOT/'runs/c4_wm_v10_round12c/wm_s.pt'),
           '--train-root', str(ROOT/'data/v5_native'), '--train-root', str(ROOT/'data/genrand_plain'),
           '--states', str(WORK/'train_states48.json.gz'), '--out', str(OUT),
           '--workers', '4', '--one-step-panel']
with (WORK/'offline.log').open('x') as log, open(os.devnull) as stdin:
    process = subprocess.Popen(command, cwd=REPO, env=env, stdin=stdin, stdout=log, stderr=log,
                               start_new_session=True)
start_ticks = Path(f'/proc/{process.pid}/stat').read_text().rsplit(')', 1)[1].split()[19]
receipt = {'pid': process.pid, 'start_ticks': start_ticks, 'argv': command,
           'started_utc': datetime.now(timezone.utc).isoformat(), 'nice': 19, 'workers': 4,
           'nofile_soft': resource.getrlimit(resource.RLIMIT_NOFILE)[0],
           'checkpoint_sha256': study.file_digest(ROOT/'runs/c4_wm_v10_round12c/wm_s.pt'),
           'offline_tool_sha256': hashlib.sha256((REPO/'tools/r13e_checks/offline.py').read_bytes()).hexdigest()}
(WORK/'offline_owned_process.json').write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps(receipt, indent=2))
