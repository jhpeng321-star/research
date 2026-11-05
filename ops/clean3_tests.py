"""Run the retained full clean2 selection plus every integrated clean3 suite."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

repo = Path(__file__).resolve().parents[1]
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
parent = repo/'_out'
assert parent.is_dir()
control = parent/('clean3_cpu_'+head[:8])
control.mkdir()
paths = json.loads((repo/'CLEAN2_TEST_SELECTION.json').read_text())['paths']
paths += ['tools/coopmwm/'+name for name in (
    'test_executable_labels_cpu.py', 'test_coordination_opportunities_cpu.py',
    'test_sparse_synergy_credit_cpu.py', 'test_g4_runtime_cpu.py', 'test_clean3_budget_cpu.py',
    'test_group_lane_cpu.py', 'test_g1_certification_cpu.py', 'test_bank_perseed_cpu.py',
    'test_bank_imitation_cpu.py', 'test_replay_bank_v2v_cpu.py', 'test_clean3_restart_cpu.py')]
paths += ['tests/unit/test_c4_evalv5.py', 'tests/unit/test_c4_policy2.py']
paths = list(dict.fromkeys(paths))
assert all((repo/path).is_file() for path in paths)
command = [sys.executable, '-B', '-m', 'pytest', '-q', '--import-mode=importlib',
    '-o', 'consider_namespace_packages=true', *paths, '--tb=short',
    '--deselect=tests/unit/test_c4_policy2.py::test_env_executes_lane_commands_in_sumo',
    '--basetemp='+str(control/'tmp'), '--junitxml='+str(control/'results.xml')]
env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
    OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
started = time.time()
(control/'DISPATCH.json').write_text(json.dumps(dict(pid=os.getpid(), head=head,
    unix=started, command=command, paths=paths), indent=2))
with (control/'pytest.log').open('x', encoding='utf-8') as log:
    result = subprocess.run(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
after = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
(control/'RESULT.json').write_text(json.dumps(dict(returncode=result.returncode,
    head=head, unchanged_source_head=head == after, seconds=time.time()-started), indent=2))
raise SystemExit(result.returncode or int(head != after))
