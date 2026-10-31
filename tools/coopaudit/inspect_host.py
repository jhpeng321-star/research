"""Read permitted runtime/config paths only, no SUMO or process signals."""
import json
import os
from pathlib import Path

roots = ['/root/wmdata2'] if os.getuid() == 0 else ['/home/ubuntu/wmdata20261031', '/home/ubuntu/e3final']
for root in roots:
    p = Path(root)
    if p.is_dir():
        print(json.dumps(dict(root=root, entries=[str(x) for x in p.iterdir()])))
        for name in ('tools', 'pytools/tools', 'inputs/INPUTS.json'):
            item = p/name
            if item.is_dir():
                print(json.dumps(dict(path=str(item), entries=[str(x) for x in item.iterdir()])))
            elif item.is_file():
                print(str(item), item.read_text())
        if os.getuid() == 0:
            print((p/'tools/wmdata2/worker.py').read_text()[:6500])
            print((p/'tools/ctrlarm/zero.py').read_text())
            print((p/'tools/speedhead/baselines.py').read_text())
