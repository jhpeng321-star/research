"""CLEAN5C4 onset-bin host reducer (read-only; stdout only; stdlib only).

For every hold activation of every non-u0 WMVERIFY job: the acting CAV (holder = the member on SUMO lane 2,
else the first member; S-MULTICRIT-ACT reduce_act rule) and the observable tape rows at the onset second ->
tools.coopmwm.fullpol_situational.onset_bin(own, rows) - the SAME function fullpol_bank.offer_ok calls at
runtime (own row + vehicles within SENSOR_RANGE_M; unobserved -> its own bin).

Usage: python3 ops/sit_reduce.py <tree> <root_dir> [<root_dir> ...] > ONSET_<tag>.jsonl
  <tree>: a directory holding tools/coopmwm/fullpol_situational.py (the reviewed commit)
"""
import gzip
import json
import os
import re
import sys

ID_RE = re.compile(r'^(?:(D\d+_p\d+)_)?s(\d+)_(.+)$')


def tape_rows(path, times):
    want, out = set(times), {}
    with gzip.open(path, 'rt') as fh:
        for line in fh:
            d = json.loads(line)
            t = int(round(d['time_s']))
            if t in want:
                out[t] = [dict(vid=r['vid'], is_cav=bool(r.get('is_cav')), lane=int(r['lane']), s_m=float(r['s_m']),
                               v_mps=None if r.get('v_mps') is None else float(r['v_mps'])) for r in d['vehicles']]
    return out


def main(tree, roots):
    sys.path.insert(0, tree)
    from tools.coopmwm.fullpol_situational import onset_bin
    for root in roots:
        name = os.path.basename(os.path.normpath(root))
        jd = os.path.join(root, 'jobs')
        for ident in sorted(os.listdir(jd)):
            m = ID_RE.match(ident)
            path = os.path.join(jd, ident)
            if not m or ident.startswith('identity') or m.group(3) == 'u0' or not os.path.exists(os.path.join(path, 'DONE.json')):
                continue
            cell, seed, arm = m.group(1) or 'D3200_p60', int(m.group(2)), m.group(3)
            acts = json.load(gzip.open(os.path.join(path, 'SUMMARY.json.gz'), 'rt')).get('activations') or []
            if not acts:
                continue
            rows = tape_rows(os.path.join(path, 'plant', 'observable.jsonl.gz'), [int(a['t']) for a in acts])
            for a in acts:
                t, lanes = int(a['t']), a.get('sumo_lanes') or []
                hold = next((mm for mm, ln in zip(a['members'], lanes) if ln == 2), a['members'][0] if a['members'] else None)
                at = rows.get(t) or []
                own = next((r for r in at if r['vid'] == hold), None)
                sys.stdout.write(json.dumps(dict(root=name, cell=cell, seed=seed, arm=arm, t=t, holder=hold,
                                                 own_found=own is not None, n_rows=len(at),
                                                 bin=onset_bin(own, at)), separators=(',', ':'))+'\n')
            sys.stdout.flush()


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
