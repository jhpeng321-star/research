"""Sharded Linux qualification of an assembled source tree (zero physics, <= N cores).

Usage (on the host): python ops/clean5a_linux_qualify.py <src> <control_parent> [--shards 4]
Writes <control_parent>/clean5a_tests_<commit8>/{DISPATCH.json, shard_<k>.log, RESULT.json}.
Env prefix = the qualified clean2/clean3 prefix (T2C/intent/pre-F4 baselines from <src>/qualification_baselines,
COOPMWM_MAPPO = frozen d9042cc5); COOPMWM_MAP_SPEC / COOPMWM_FROZEN_TRUNK unset (tests bind themselves).
Selection = <src>/ops/clean4_test_selection.json + every clean4/clean5a/frozen_trunk/g4ld/nm_eval suite.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument('src'); p.add_argument('control_parent'); p.add_argument('--shards', type=int, default=4)
a = p.parse_args()
src = Path(a.src)
commit = json.loads((src/'SOURCE_MANIFEST.json').read_text())['commit']
control = Path(a.control_parent)/('clean5a_tests_'+commit[:8])
control.mkdir(parents=True)
base = src/'qualification_baselines'
prefix = (f'. /home/ubuntu/cpus_env/env.sh {src} && export T2C_TRACK_BASE_SOURCE={base}/rerun_cpus_13f184b5.py '
          f'COOPMWM_TRACK_BASELINE_DIR={base}/track_base COOPMWM_INTENT_BASELINE_DIR={base}/intent '
          f'COOPMWM_PRE_F4_BASELINE_DIR={base}/pre_f4 COOPMWM_MAPPO=/home/ubuntu/mappospeed/production2/policy_00169.pt '
          '&& unset COOPMWM_MAP_SPEC COOPMWM_FROZEN_TRUNK')
paths = json.loads((src/'ops'/'clean4_test_selection.json').read_text())['paths']
extra = sorted(str(q.relative_to(src/'pytools')) for q in (src/'pytools'/'tools'/'coopmwm').glob('test_*cpu.py')
               if any(k in q.name for k in ('clean4', 'clean5a', 'g4ld', 'nm_eval', 'lane_drop', 'map_')))
paths = [q for q in dict.fromkeys(paths+extra) if (src/'pytools'/q).is_file()]
shards = [paths[k::a.shards] for k in range(a.shards)]
deselect = ('--deselect=tests/unit/test_c4_policy2.py::test_env_executes_lane_commands_in_sumo '
            '--deselect=tools/coopmwm/test_group_lane_cpu.py::test_g5_env_default_off_matches_pre_g5_env_byte_identically')
procs = []
started = time.time()
for k, shard in enumerate(shards):
    cmd = (prefix+f' && cd {src}/pytools && nice -n 19 /home/ubuntu/miniconda3/bin/python -B -m pytest -q '
           '--import-mode=importlib -o consider_namespace_packages=true '+' '.join(shard)+
           f' --tb=short -p no:cacheprovider {deselect} --basetemp=/tmp/clean5a_q_{commit[:8]}_{k}')  # outside /home/ubuntu (staging tests mirror it)
    log = (control/f'shard_{k}.log').open('x')
    procs.append(subprocess.Popen(['bash', '-c', cmd], stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL))
(control/'DISPATCH.json').write_text(json.dumps(dict(source_commit=commit, shards=a.shards, paths=len(paths),
    pids=[q.pid for q in procs], unix=started, new_sim_s=0)))
codes = [q.wait() for q in procs]
summary = []
for k in range(a.shards):
    tail = (control/f'shard_{k}.log').read_text(errors='replace').strip().splitlines()[-1:]
    summary.append(tail[0] if tail else '')
(control/'RESULT.json').write_text(json.dumps(dict(returncode=max(codes), shard_returncodes=codes, summary=summary,
    seconds=time.time()-started, source_commit=commit, paths=len(paths), new_sim_s=0)))
print(json.dumps(dict(returncode=max(codes), summary=summary)))
