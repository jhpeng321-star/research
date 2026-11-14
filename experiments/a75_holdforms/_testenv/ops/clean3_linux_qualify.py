import json, os, subprocess, time, shlex
from pathlib import Path
old = json.loads(open('/home/ubuntu/cpuv_setup_20261105/clean2_tests_285663a1/DISPATCH_STARTED.json').read())['command']
src = '/home/ubuntu/coopmwm20261106_clean3_src_3e5bc9f9'
commit = json.loads(open(src+'/SOURCE_MANIFEST.json').read())['commit']
control = Path('/home/ubuntu/cpuv_setup_20261105/clean3_tests_'+commit[:8]); control.mkdir()
prefix = old.split(' && /home/ubuntu/miniconda3/bin/python')[0].replace('/home/ubuntu/coopmwm20261105_clean2_src_285663a1', src)
paths = json.loads(open(src+'/TEST_SELECTION.json').read())['paths']
cmd = (prefix + ' && cd ' + src + '/pytools && /home/ubuntu/miniconda3/bin/python -B -m pytest -q --import-mode=importlib -o consider_namespace_packages=true '
       + ' '.join(paths) + ' --tb=short -p no:cacheprovider --deselect=tests/unit/test_c4_policy2.py::test_env_executes_lane_commands_in_sumo --deselect=tools/coopmwm/test_group_lane_cpu.py::test_g5_env_default_off_matches_pre_g5_env_byte_identically'
      )
runner = ("import json,subprocess,sys,time;t=time.time();"
  "r=subprocess.run(['bash','-c',sys.argv[1]],stdout=open(sys.argv[2]+'/pytest.log','w'),stderr=subprocess.STDOUT);"
  "open(sys.argv[2]+'/RESULT.json','w').write(json.dumps(dict(returncode=r.returncode,seconds=time.time()-t,source_commit=sys.argv[3])))")
(control/'DISPATCH_STARTED.json').write_text(json.dumps(dict(unix=time.time(), source_commit=commit, command=cmd)))
p = subprocess.Popen(['setsid','nohup','/home/ubuntu/miniconda3/bin/python','-c',runner,cmd,str(control),commit],
    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True, cwd='/home/ubuntu')
(control/'DISPATCH.json').write_text(json.dumps(dict(pid=p.pid, source_commit=commit, new_sim_s=0)))
print(p.pid, prefix[-300:])
