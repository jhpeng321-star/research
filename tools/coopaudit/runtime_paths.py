import importlib.util
import os
import shutil
import sys
from pathlib import Path
print('interpreter',sys.executable,'PATH',os.environ.get('PATH'))
for name in ('sumo','sumolib'):
    spec=importlib.util.find_spec(name)
    print(name,spec.origin if spec else None)
print('sumo',shutil.which('sumo'))
for p in (Path('/root/miniconda3/bin/sumo'),Path('/root/miniconda3/lib/python3.12/site-packages/sumo/bin/sumo')):
    print(str(p),p.is_file())
