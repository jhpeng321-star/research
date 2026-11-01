"""Owner-bound exact protocol; unused COOPNATIVE worlds reassigned prospectively."""
from tools.coopnative.protocol import (worlds, batch_worlds, guard, install_seed_scope,
    checked, canonical, digest, sha, FORBIDDEN, REGISTRY)

CAP=6_000_000
CAPS=dict(train=4_608_000,checks=590_400,dev=115_200,mechanics=2400,
          calibration=240_000,discovery=240_000)
UPDATES=80
CHECK_EVERY=4


def authority(text):
    lines=text.splitlines()
    a=next((x for x in lines if '16:22 MAIN:' in x and 'S-COOPMWM' in x and '6,000,000' in x),None)
    b=next((x for x in lines if '16:25 MAIN:' in x and 'SPEED MODE' in x and 'FULL-v4' in x),None)
    if not a or not b:
        raise ValueError('owner MAPPO re-scope and speed amendment required')
    return [a,b]


def registry_conflicts(text):
    from tools.coopnative.protocol import registry_conflicts as prior
    safe='\n'.join('' if ('S-COOPMWM' in l or 'S-COOPNATIVE DESIGN READY' in l) else l
                   for l in text.splitlines())
    return prior(safe)


def budget():
    return dict(total=CAP,phases=CAPS,unallocated=CAP-sum(CAPS.values()),
                updates_per_arm=80,episodes_per_update=24,sim_s_per_episode=1200,
                train_controlled_sim_s=3_456_000,new_dev_mappo_sim_s=0)
