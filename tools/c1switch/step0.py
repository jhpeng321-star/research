"""Single owned low-priority queue for the pre-registered twelve native trunks."""
import argparse
import json
import os
from pathlib import Path
import time

from .design import worlds
from .runner import run
from .storage import Ledger, exclusive, write


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('/root/c1switch20261031'))
    a = p.parse_args()
    root = a.root
    with exclusive(root):
        for i, world in enumerate(worlds('step0')):
            out = root/'jobs'/f'step0_s{world["seed"]}'
            if (out/'result.json').exists():
                result = json.loads((out/'result.json').read_text())
                if not result.get('ok') or result['world'] != world or result['actual_sim_s'] != 4890:
                    raise ValueError('existing receipt not a complete matching world')
                continue
            if out.exists() or (root/'STOP').exists():
                raise ValueError('incomplete attempt/STOP; never redispatch')
            while os.getloadavg()[0] > 20 or (root/'YIELD').exists():
                write(root/'STEP0_STATUS.json', dict(status='WAIT_PRIORITY', index=i, unix=time.time()))
                if (root/'STOP').exists():
                    return
                time.sleep(30)
            write(root/'STEP0_STATUS.json', dict(status='RUNNING', index=i, world=world, unix=time.time()))
            run(root, 'step0', world)
            write(root/'STEP0_STATUS.json', dict(status='WORLD_COMPLETE', index=i, unix=time.time()))
        rows = Ledger(root).rows()
        charged = sum(r['sim_s'] for r in rows)
        if charged != 59160:
            raise ValueError('unexpected complete Step0 accounting')
        write(root/'STEP0_STATUS.json', dict(status='COMPLETE', worlds=12, charged_sim_s=charged, unix=time.time()))


if __name__ == '__main__':
    main()
