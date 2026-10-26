"""Offline tests with all scratch/logs in this task's box directory."""
from pathlib import Path
import os
import subprocess
import sys

root = Path('/root/autodl-tmp/c4/paircrit20261026')
os.environ['PYTHONPATH'] = str(root / 'src') + ':' + str(root)
os.environ['NATIVEINIT_BASELINE'] = str(root / 'parent')
result = subprocess.run([sys.executable, '-B', '-m', 'pytest', '-q',
    'tests/unit/test_paircrit.py', 'tests/unit/test_longcritic.py', 'tests/unit/test_nativeinit.py',
    '--basetemp=' + str(root / 'pytest')], capture_output=True, text=True)
(root / 'tests.log').write_text(result.stdout + result.stderr)
print(result.stdout + result.stderr)
raise SystemExit(result.returncode)
