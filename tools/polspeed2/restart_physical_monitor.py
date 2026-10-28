"""Preserve an owned physical-headroom STOP before retrying the incomplete case."""
from pathlib import Path
import json
import shutil
import subprocess
import sys

base = Path('/root/autodl-tmp/c4/polspeed2_20261028')
old = base/'out/six_N4'
result = json.loads((old/'RESULT.json').read_text())
assert 's_per_update' not in result
assert result['peak_owned_gpu_fraction'] < .60 and result['peak_total_gpu_fraction'] > .95
assert (base/'STOP').exists()
archive = base/'failed_physical_headroom'
archive.mkdir(exist_ok=False)
old.rename(archive/'six_N4')
(base/'launch/six_N4').rename(archive/'launch_six_N4')
(base/'STOP').rename(archive/'STOP')
shutil.copyfile(base/'suite.log', archive/'suite.log')
subprocess.run([sys.executable, str(base/'runtime/tools/polspeed2/run_remaining.py')], check=True)
