from pathlib import Path
import importlib.util
import torch
print('torch', torch.__version__, 'cloudpickle', importlib.util.find_spec('cloudpickle'))
for name in ('polspeed20261028', 'polspeed2_20261028'):
    path = Path('/root/autodl-tmp/c4') / name
    print(path, list(path.iterdir()) if path.exists() else 'absent')
