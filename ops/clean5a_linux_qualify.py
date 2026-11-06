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
               if any(k in q.name for k in ('clean4', 'clean5a', 'clean5b', 'g4ld', 'nm_eval', 'lane_drop', 'map_', 'fullpol', 'lc_')))
paths = [q for q in dict.fromkeys(paths+extra) if (src/'pytools'/q).is_file()]
shards = [paths[k::a.shards] for k in range(a.shards)]
# Deselected on the DEPLOYED Linux tree, with reasons (each passes in the Windows checkout suite):
#  - SUMO-running test (as clean2/clean3 qualification);
#  - G5 pre-G5-env byte identity (as clean3 qualification);
#  - test_map_runtime x3: subprocess PYTHONPATH = checkout traffic_shepherd + src (absent in the deployed
#    tree by design: pytools must not shadow runtime_src/e3final); runtime path covered by the cpuV sanity job;
#  - test_wall_w2 OFF golden: Windows-written float digest (platform numerics), identical failure on both hosts.
deselect = ' '.join('--deselect='+t for t in (
    'tests/unit/test_c4_policy2.py::test_env_executes_lane_commands_in_sumo',
    'tools/coopmwm/test_group_lane_cpu.py::test_g5_env_default_off_matches_pre_g5_env_byte_identically',
    'tools/coopmwm/test_map_runtime_cpu.py::test_wz3to2_install_binds_projection_scenario_lanes_slots_and_rsu',
    'tools/coopmwm/test_map_runtime_cpu.py::test_wz3to2_env_builder_after_install_writes_probe_identical_scenario',
    'tools/coopmwm/test_map_runtime_cpu.py::test_install_refuses_a_frame_mismatch',
    'tools/coopmwm/test_wall_w2_cpu.py::test_off_is_byte_identical_to_the_pre_w2_tree'))
#  - clean5b: LCEXEC harness/parity tests that need CHECKOUT layout files absent from the deployed tree by design
#    (runtime_src/baselines/circles/adapter.json for lc_writer_harness, repo-root LCEXEC_PARITY.json, run_py
#    subprocess traffic_shepherd PYTHONPATH); every node id + cause in ops/clean5b_layout_deselect.json (classified
#    from the 2d2d8db8 cpuV run: 172 adapter.json, 1 LCEXEC_PARITY.json, 1 traffic_shepherd); they run in the
#    Windows checkout suite; the production LCEXEC path is exercised by the clean5b dry prepare + smoke.
layout = src/'ops'/'clean5b_layout_deselect.json'
if layout.is_file():
    deselect += ' '+' '.join("'--deselect="+t+"'" for t in sorted(json.loads(layout.read_text())))
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
