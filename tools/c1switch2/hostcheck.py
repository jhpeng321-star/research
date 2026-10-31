"""Real-checkpoint CPU feedback probe. No SUMO allocation and no GPU admission."""
import argparse
import json
import os
import time

from tools.c1switch.design import sha
from tools.c1switch.model import BB_SHA, WM2_SHA
from tools.c1switch.storage import canonical

from .core import means
from .design import DESIGN
from .engine import Engine


def belief(now=320):
    history = []
    for i in range(121):
        t = now-60+i*.5
        own = [dict(vehicle_id=vid, timestamp_s=t, s_m=s-17*(now-t), lane=lane,
                    v_mps=17., a_mps2=0., age_s=0., source='cav_self')
               for vid, s, lane in [('a', -1000, 0), ('b', -1010, 1)]]
        history.append(dict(now_s=t, self_reports=own, detections=[], coverage=[], counters=None))
    return dict(schema='c1switch_observable/1', history=history,
        controller_state=dict(incumbent='native', now_s=now, pi={}, commands={}, wall=None))


def probe(model, checkpoint, *, steps=120):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU probe requires CUDA hidden')
    binding = dict(model=model, wm_checkpoint_sha256=WM2_SHA, bb_checkpoint_sha256=BB_SHA,
                   checkpoint_path=checkpoint, objective_design_sha256=sha(DESIGN))
    started = time.monotonic()
    instance = Engine(binding, device='cpu')
    raw = belief()
    before = canonical(raw)
    samples = instance.imagine_all(raw, steps=steps)
    fresh = means(samples)
    if canonical(raw) != before:
        raise ValueError('real-checkpoint inference modified input')
    raw['controller_state'].update(incumbent='wall', wall=dict(schema='c1switch2.wall/1',
        started_s=300., last_s=319., cooldown_s=None, members=['a', 'b'], speed_mps=17.,
        phase=1, assembly_since_s=300., assembled_at_s=301.))
    continued = means(instance.imagine_all(raw, steps=steps))
    if continued['wall'] == fresh['wall']:
        raise ValueError('real-checkpoint WALL feedback failed to preserve incumbent memory')
    return dict(status='PASS', model=model, steps=steps, K=8, device='cpu',
        fresh=fresh, incumbent_wall=continued, elapsed_s=time.monotonic()-started,
        samples_sha256=canonical(samples), attestation=instance.attestation,
        new_sim_s=0, new_gpu_s=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=('wm', 'bb'), required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--steps', type=int, default=120)
    args = parser.parse_args()
    print(json.dumps(probe(args.model, args.checkpoint, steps=args.steps)))


if __name__ == '__main__':
    main()
