"""Guidance-doc priority (MAIN 2026-11-09, CLEAN5C5B u1: unconditional duo12_v20 crowded out the WM targets).

EV-BC ranks qualifying families by lb95 (fullpol_learn.qualifying). The unconditional row's lb95 is per EPISODE
(927 veh-s) while the situational rows' are per activation window, so it always won. This rewrites ONLY the ranking
numbers of unconditional rows (row_class 'unconditional_fallback'): lb95/mean become fallback_scale x the smallest
situational lb95/mean in the same cell, so situational rows (bin-gated, admitted, feasible) are chosen first and the
unconditional plan is the fallback - the ORACLE ordering. Original values are kept in row['ranking']. Rows digest
re-hashed with the parent cited. No code change: the doc joins via bank growth at an update boundary.

  python ops/priority_doc.py --doc UNION_UNC.json --out UNION_UNC_PRIORITY.json
"""
import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.coopmwm.fullpol_learn import guidance_doc  # noqa: E402

RULE = ('priority v2 (MAIN 2026-11-09): situational rows rank first by lb95; then unconditional WM rows, then '
        'unconditional rule rows (each tier by original lb95), all below scale x the smallest situational value of the '
        'cell; originals in row.ranking')


def prioritise(doc, scale=0.5, exclude=()):
    excl = set(exclude)
    rows = [r for r in copy.deepcopy(doc['rows']) if r['family'] not in excl]
    uncond = lambda r: bool((r.get('situational') or {}).get('unconditional'))  # noqa: E731
    for r in rows:
        r['row_class'] = 'unconditional_fallback' if uncond(r) else 'situational'
    for cell in sorted({r['regime'] for r in rows}):
        sit = [x for x in rows if x['regime'] == cell and x['row_class'] == 'situational' and x['lb95_veh_s'] > 0]
        unc = [x for x in rows if x['regime'] == cell and x['row_class'] == 'unconditional_fallback']
        if not sit or not unc:
            continue                      # nothing to yield to (or nothing to demote)
        # v2 (MAIN 2026-11-09): unconditional WM rows rank above unconditional rule rows; within a tier by the
        # original lb95; all of them below every situational row of the cell.
        unc.sort(key=lambda x: (x.get('plan_source') != 'wm', -x['lb95_veh_s'], x['family']))
        top_lb, top_mean = scale*min(x['lb95_veh_s'] for x in sit), scale*min(x['mean_veh_s'] for x in sit)
        n = len(unc)
        for k, r in enumerate(unc):
            r['ranking'] = dict(rule=RULE, original_lb95_veh_s=r['lb95_veh_s'], original_mean_veh_s=r['mean_veh_s'],
                                original_ub95_veh_s=r.get('ub95_veh_s'), unit='per episode (unconditional confirmation)',
                                fallback_rank=k+1, fallback_tier=r.get('plan_source'))
            r['lb95_veh_s'] = top_lb*(n-k)/n
            r['mean_veh_s'] = top_mean*(n-k)/n
    out = guidance_doc(rows, sources=list(doc['source']['sources'])+[dict(
        name='parent guidance (priority rewrite)', sha256=doc['source']['sha256'], baseline='mappo', role='parent')],
        provenance=doc['provenance']+' | '+RULE, weight=doc['weight'], clip=doc['clip'])
    out['source']['rule'] = doc['source'].get('rule', out['source']['rule'])
    for k in set(doc)-set(out):
        out[k] = copy.deepcopy(doc[k])
    if excl:
        if 'clean5c' in out:
            out['clean5c']['templates'] = [t for t in out['clean5c']['templates'] if t['family'] not in excl]
        if 'situational' in out:
            out['situational']['pairs'] = [q for q in out['situational']['pairs'] if q['family'] not in excl]
    out['priority'] = dict(rule=RULE, scale=scale, parent_rows_sha256=doc['source']['sha256'],
                           excluded=sorted(excl))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--doc', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--scale', type=float, default=0.5)
    ap.add_argument('--exclude', nargs='*', default=(), help='families dropped (rows, templates, pairs)')
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise SystemExit('refusing to overwrite '+str(out))
    doc = prioritise(json.loads(Path(a.doc).read_bytes()), a.scale, a.exclude)
    data = (json.dumps(doc, indent=1, sort_keys=True)+'\n').encode('utf-8')
    out.write_bytes(data)
    sha = hashlib.sha256(data).hexdigest()
    Path(str(out)+'.sha256').write_bytes(f'{sha}  {out.name}\n'.encode('utf-8'))
    print(json.dumps(dict(out=str(out), file_sha256=sha, rows_sha256=doc['source']['sha256'],
                          order={c: [r['family'] for r in sorted((r for r in doc['rows'] if r['regime'] == c),
                                                                 key=lambda r: (-r['lb95_veh_s'], -r['mean_veh_s'], r['family']))]
                                 for c in sorted({r['regime'] for r in doc['rows']})})))
