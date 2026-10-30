import runpy
import sys
from pathlib import Path
p=Path('/root/autodl-tmp/c4/wmrequal20261030')
assert p.is_dir()
sys.path.insert(0,str(p/'tools/wmrequal'))
runpy.run_path(str(p/'tools/wmrequal/real_panel.py'),run_name='__main__')
