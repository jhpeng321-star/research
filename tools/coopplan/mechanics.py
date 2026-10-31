"""Single registered 2040 sim-s mechanics slot, never automatically repeated."""
import argparse
from pathlib import Path
import traceback

from . import plant
from .protocol import worlds, canonical
from .storage import write, stop


def run(root):
    root = Path(root)
    plant.setup(root)
    world = worlds()[0]
    b,path = plant.fresh(root,'mechanics_mappo',world,1200)
    try:
        from tools.leverprobe.backend import observations
        snapshot = dict(world=world,time_s=300,rows=observations(b.env),history=[],prefix=b.fingerprint())
        write(root/'states'/'mechanics_t300.json',snapshot)
        plant.replay(b,[],1200)
        reference = dict(totals=b.totals,prefix=b.fingerprint())
        write(path/'RESULT.json',reference)
    finally:
        b.close()
    a = plant.branch(root,'mechanics_t300',dict(id='repeat_a',members=[]),120)
    b = plant.branch(root,'mechanics_t300',dict(id='repeat_b',members=[]),120)
    if canonical(a) != canonical(b):
        raise ValueError('identical full-rerun branches differ')
    write(root/'MECHANICS.json',dict(pass_=True,sim_s=2040,baseline_reference=reference,branch=a))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root',type=Path)
    args = p.parse_args()
    try:
        run(args.root)
    except BaseException:
        stop(args.root,traceback.format_exc())
        raise
