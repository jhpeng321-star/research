"""Schedule follow-up recipes after the existing owned sweep exits."""
from pathlib import Path
import subprocess
import sys

base = Path('/root/autodl-tmp/c4/polspeed2_20261028')
assert base.is_dir() and not (base/'STOP').exists()
assert not (base/'FOLLOWUP_PID').exists()
with (base/'followup.log').open('w') as log:
    process = subprocess.Popen([sys.executable, '-u', str(base/'runtime/tools/polspeed2/run_remaining.py'),
        '--wait'], cwd=base, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True)
(base/'FOLLOWUP_PID').write_text(str(process.pid))
print('followup pid', process.pid)
