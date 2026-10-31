"""Read-only source-location probe, no model/data loading or CUDA allocation."""
import importlib.util
from importlib.metadata import version, PackageNotFoundError
import json
import os
import sys
from pathlib import Path

root = Path('/root/autodl-tmp/c4')
fix = root/'setupaudit20261030/fix'
sys.path[:0] = [str(root/'speedhead20261030/tools/speedhead'),
                str(fix/'src'), str(fix), str(fix/'tools/setupaudit')]
os.environ['CUDA_VISIBLE_DEVICES'] = ''
print(json.dumps(dict(path=sys.path), indent=1))
for name in ('panel2', 'measure', 'wm_worker'):
    spec = importlib.util.find_spec(name)
    print(name, None if spec is None else spec.origin)
print('load', os.getloadavg(), 'cpus', os.cpu_count())
for name in ('cpu.max', 'cpu.stat', 'memory.current', 'memory.max'):
    path = Path('/sys/fs/cgroup')/name
    print(name, path.read_text() if path.is_file() else 'unavailable')
for name in ('scikit-learn', 'numpy', 'joblib', 'torch'):
    try:
        print(name, version(name))
    except PackageNotFoundError:
        print(name, 'unavailable')
