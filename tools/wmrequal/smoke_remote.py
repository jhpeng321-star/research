from pathlib import Path
import os
import subprocess
import sys
p=Path('/root/autodl-tmp/c4/wmrequal20261030')
rt=Path('/root/autodl-tmp/c4/restart20261026/runtime_longint')
env=dict(os.environ,PYTHONPATH=f'{p}:{p}/tools/wmrequal:{rt}/src:{rt}',
         OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
         PYTHONDONTWRITEBYTECODE='1',WMS_SPEED_PIVOT_DIAG_OFF='1',
         WMS_SPEED_BATCH_REPLAY='1',WMS_SPEED_FAST_CAV_DRIVE='1',
         WMS_SPEED_FAST_LONG_HISTORY='1',WMS_LOOPLONG_SPEED='1',
         C4_SUMO_LEADER_RULE='1',C4_EXECUTION_PARITY='2')
r=subprocess.run([sys.executable,'-B',str(p/'tools/wmrequal/score.py'),
                  '--panel','validation','--smoke'],env=env,cwd=p)
raise SystemExit(r.returncode)
