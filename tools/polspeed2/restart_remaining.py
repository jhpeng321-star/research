"""Continue after a diagnosed worker-memory failure, keeping its original log."""
from pathlib import Path
import shutil
import subprocess
import sys

base = Path('/root/autodl-tmp/c4/polspeed2_20261028')
assert base.is_dir() and not (base/'STOP').exists()
pid = int((base/'COORDINATOR_PID').read_text())
stat = Path(f'/proc/{pid}/stat')
assert not stat.exists() or stat.read_text().rsplit(')', 1)[1].split()[0] == 'Z'
target = base/'remaining_before_gate.log'
assert not target.exists()
shutil.copyfile(base/'remaining.log', target)
subprocess.run([sys.executable, str(base/'runtime/tools/polspeed2/run_remaining.py')], check=True)
