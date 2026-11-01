"""Extract observable departure distributions from explicitly named FIT tapes."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from tools.rbfit.idm_fit import native_episode, read_episode


def extract(a, world):
    kin, mask, vid, times = (a[k] for k in ('veh_kin', 'veh_mask', 'vid', 'time_s'))
    seen = set()
    rows = []
    for t in range(len(times)):
        if times[t] > 900:
            break
        active = np.flatnonzero(mask[t] & (kin[t, :, 1:4].sum(-1) > 0))
        ids = vid[t] if vid.ndim == 2 else vid
        if t == 0:
            seen.update(int(ids[i]) for i in active)
            continue
        for i in active:
            identity = int(ids[i])
            if identity in seen:
                continue
            seen.add(identity)
            if not 240 <= times[t] <= 900:
                continue
            k = kin[t, i]
            lane = int(k[1:4].argmax())
            rows.append(dict(world=world, time_s=float(times[t]), lane=lane,
                             position_m=float(k[0]), speed_mps=float(k[4]),
                             length_m=float(k[6])))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    assert args.out.is_dir() and not (args.out/'DIAGNOSIS.json').exists()
    begin = time.monotonic()
    rows, provenance = [], []
    for world in range(969400, 969432):
        a, receipt = read_episode(native_episode(args.root, world), world)
        rows.extend(extract(a, world))
        provenance.append(receipt)
    groups = {}
    for e, name in ((0, 'mainline'), (1, 'ramp')):
        group = [r for r in rows if (r['lane'] == 2) == bool(e)]
        groups[name] = dict(n=len(group), speed_quantiles=np.quantile(
            [r['speed_mps'] for r in group], [0, .1, .5, .9, 1]).tolist(),
            lane_counts=[sum(r['lane'] == ln for r in group) for ln in (0, 1, 2)],
            position_quantiles=np.quantile([r['position_m'] for r in group], [0, .5, 1]).tolist())
    result = dict(worlds=list(range(969400, 969432)), groups=groups,
                  seconds=time.monotonic()-begin, provenance=provenance)
    (args.out/'DIAGNOSIS.json').write_text(json.dumps(result, indent=2)+'\n')
    (args.out/'births.json').write_text(json.dumps(rows)+'\n')
    print(json.dumps(groups), flush=True)


if __name__ == '__main__':
    main()
