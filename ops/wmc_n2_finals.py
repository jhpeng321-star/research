"""Offline N2 parity (c4/wmverify tools.wzprop.wmverify_n2.parity, FULLPOL e36ece92) of every non-comparator final
of p20 strategy searches. Run from the c4/wmverify worktree in rl_driving:

    python wmc_n2_finals.py OUT.json SEARCH1.json [SEARCH2.json ...]
"""
import json
import sys

from tools.wzprop import wmverify_n2 as N


def main(out, *paths):
    res = {}
    for p in paths:
        for r in json.load(open(p))['final']:
            s = r['strategy']
            if r.get('comparator') or s['name'] in res:
                continue
            try:
                v = N.parity(s, share=0.2, seed=989950)
            except Exception as e:  # recorded, never imputed
                v = dict(verdict='error', reason=repr(e))
            res[s['name']] = dict(verdict=v.get('verdict'), reason=v.get('reason'))
            print(s['name'], res[s['name']]['verdict'], res[s['name']]['reason'] or '', flush=True)
    json.dump(res, open(out, 'w'), indent=1)


if __name__ == '__main__':
    main(*sys.argv[1:])
