"""Cooperation probe summary: per setting x family x world seed: tried / beat-policy / synergy-measured /
synergy-admitted (per CRN seed index), plus every measured synergy with its one-sided LB95."""
import json
from pathlib import Path
import sys

ROOT = Path('/home/ubuntu/coopprobe20261106')


def load(setting):
    p = ROOT/'results'/f'{setting}.jsonl'
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def main():
    out = {}
    for setting in ('wall', 'hp', 'lowcoop'):
        rows = load(setting)
        table = {}
        for r in rows:
            if 'group_id' not in r:
                continue
            key = (r['family'], r['world']['seed'])
            t = table.setdefault(key, dict(tried=0, beat_policy=0, beat_policy_lb=0, measured=0, admitted=0,
                                           examples=[]))
            t['tried'] += 1
            if r.get('group_adv') is not None and r['group_adv'] > 0:
                t['beat_policy'] += 1
            if (r.get('adv_lb95') or -1) > 0:
                t['beat_policy_lb'] += 1
            if r.get('synergy_lb95') is not None:
                t['measured'] += 1
                t['admitted'] += int(r['synergy_lb95'] > 0)
                t['examples'].append(dict(sid=r['state_id'], seed_index=r['seed_index'], n=r['n_members'],
                                          group=r['group_id'][:90], adv=round(r['group_adv'], 2),
                                          best_solo_adv=round(r['best_solo_adv'], 2),
                                          synergy=round(r['synergy'], 3), lb95=round(r['synergy_lb95'], 3),
                                          collisions=r['safety'].get('collisions'),
                                          teleports=r['safety'].get('teleports'), tag=r['tag']))
        out[setting] = {f'{k[0]}|{k[1]}': v for k, v in sorted(table.items(), key=lambda kv: (kv[0][0], kv[0][1]))}
    json.dump(out, sys.stdout, indent=1)


if __name__ == '__main__':
    main()
