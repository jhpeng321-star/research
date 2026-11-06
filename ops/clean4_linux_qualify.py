"""cpuV: full Linux qualification of an assembled clean4 source tree (zero physics).

Usage (on cpuV): python ops/clean4_linux_qualify.py <src> ; writes
/home/ubuntu/cpuv_setup_20261105/clean4_tests_<commit8>/{DISPATCH*.json,pytest.log,RESULT.json}.
Same environment prefix as the qualified clean2/clean3 runs (T2C/intent/pre-F4 baselines, MAPPO),
selection = ops/clean4_test_selection.json (clean2 selection + clean3 suites + every map/clean4/
lane-drop/W1/W2 suite). COOPMWM_MAP_SPEC is NOT exported (tests bind maps themselves).
"""
import json
from pathlib import Path
import subprocess
import sys
import time

src = Path(sys.argv[1])
old = json.loads(open('/home/ubuntu/cpuv_setup_20261105/clean2_tests_285663a1/DISPATCH_STARTED.json').read())['command']
commit = json.loads((src/'SOURCE_MANIFEST.json').read_text())['commit']
control = Path('/home/ubuntu/cpuv_setup_20261105/clean4_tests_'+commit[:8])
control.mkdir()
prefix = old.split(' && /home/ubuntu/miniconda3/bin/python')[0].replace('/home/ubuntu/coopmwm20261105_clean2_src_285663a1', str(src))
paths = json.loads((src/'ops'/'clean4_test_selection.json').read_text())['paths']
paths = [p for p in paths if (src/'pytools'/p).is_file()]
cmd = (prefix+' && unset COOPMWM_MAP_SPEC && cd '+str(src)+'/pytools && /home/ubuntu/miniconda3/bin/python -B -m pytest -q '
       '--import-mode=importlib -o consider_namespace_packages=true '+' '.join(paths)+' --tb=short -p no:cacheprovider '
       '--deselect=tests/unit/test_c4_policy2.py::test_env_executes_lane_commands_in_sumo '
       '--deselect=tools/coopmwm/test_group_lane_cpu.py::test_g5_env_default_off_matches_pre_g5_env_byte_identically')
runner = ("import json,subprocess,sys,time;t=time.time();"
          "r=subprocess.run(['bash','-c',sys.argv[1]],stdout=open(sys.argv[2]+'/pytest.log','w'),stderr=subprocess.STDOUT);"
          "open(sys.argv[2]+'/RESULT.json','w').write(json.dumps(dict(returncode=r.returncode,seconds=time.time()-t,source_commit=sys.argv[3])))")
(control/'DISPATCH_STARTED.json').write_text(json.dumps(dict(unix=time.time(), source_commit=commit, command=cmd, paths=len(paths))))
p = subprocess.Popen(['setsid', 'nohup', 'nice', '-n', '19', '/home/ubuntu/miniconda3/bin/python', '-c', runner, cmd, str(control), commit],
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
                     cwd='/home/ubuntu')
(control/'DISPATCH.json').write_text(json.dumps(dict(pid=p.pid, source_commit=commit, new_sim_s=0, paths=len(paths))))
print(p.pid, control)
