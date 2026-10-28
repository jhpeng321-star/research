"""Launch one additional owned recipe after the previous coordinator exits."""
from pathlib import Path
import argparse
import subprocess
import sys

from benchmark import BASE, VARIANTS

parser = argparse.ArgumentParser()
parser.add_argument('name', choices=VARIANTS)
args = parser.parse_args()
assert BASE.is_dir() and not (BASE/'STOP').exists()
pid = int((BASE/'COORDINATOR_PID').read_text())
stat = Path(f'/proc/{pid}/stat')
assert not stat.exists() or stat.read_text().rsplit(')', 1)[1].split()[0] == 'Z'
assert not (BASE/'out'/args.name).exists()
logfile = BASE/(args.name+'.coordinator.log')
assert not logfile.exists()
with logfile.open('w') as log:
    process = subprocess.Popen([sys.executable, '-u', str(Path(__file__).with_name('benchmark.py')),
        args.name], cwd=BASE, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True)
(BASE/'COORDINATOR_PID').write_text(str(process.pid))
print('coordinator', process.pid, args.name)
