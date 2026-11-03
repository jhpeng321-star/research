"""Add MAIN12:35's fixed R-u8 episodes without touching the live A/B interim."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import os
from pathlib import Path
import time

from .accel_storage import ledger, reserve, write
from .protocol import sha
from .replay_start import accounting, binding, config, interim_specs, owner


def supplement_specs(panel, existing):
    items = interim_specs(panel, ['R'])
    if len(items) != 96 or len({s['id'] for s in items}) != 96:
        raise ValueError('R interim requires the registered 96 distinct worlds')
    if {s['id'] for s in items}.intersection(existing):
        raise ValueError('adopt existing R interim; never retry physics')
    return items


def run(root, control):
    import fcntl
    from . import final_runtime as fr
    from .nullfix_geometry import install
    from .warm_protocol import worlds
    root, control = Path(root), Path(control)
    lock = (control/'RUN.claim').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    write(control/'OWNER.json', owner())
    v = config(root); accounting(v)
    s = Path(v['source_root'])
    install(str(s/'FROZEN_GEOMETRY_SOURCE.py'), str(s/'FROZEN_NETWORK.net.xml.gz'))
    assert (root/'R_UPDATE_008.json').is_file()
    items = supplement_specs(worlds('final_test'), {x['id'] for x in ledger(root)})
    assert sum(x['sim_s'] for x in items) == 115200
    # The complete supplement is precharged before any physics, in the existing
    # restricted interim phase. Existing A/B charges and jobs are never changed.
    reserve(root, items, 'interim_u8_test')
    write(control/'MANIFEST.json', dict(unix=time.time(), specs=items, sim_s=115200,
        checkpoint_sha256=sha(root/'R_checkpoint_008.pt'), outcomes_closed=True))
    b = binding(root, v, 'R')
    with ProcessPoolExecutor(max_workers=1,
            mp_context=multiprocessing.get_context('spawn')) as pool:
        for spec in items:
            meta = pool.submit(fr.execute_one, root, b, spec).result()
            assert meta['outcomes_closed'] and meta['checkpoint_update'] == 8
    write(control/'COMPLETE.json', dict(unix=time.time(), episodes=96,
        sim_s=115200, outcomes_closed=True))


def main():
    p = argparse.ArgumentParser();p.add_argument('root');p.add_argument('control')
    a = p.parse_args()
    os.nice(max(0, 19-os.getpriority(os.PRIO_PROCESS, 0)))
    try:run(a.root, a.control)
    except BaseException:
        # Never disclose outcome-bearing exception text in public metadata.
        write(Path(a.control)/'FAILED.json', dict(unix=time.time(),
            failed=True, outcomes_closed=True, retry_forbidden=True))
        raise


if __name__ == '__main__':main()
