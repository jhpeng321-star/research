"""Identity verdict after the table edits (zero SUMO): the IDENT_U0 / IDENT_F episodes run from this code tree must
equal the stored A42 confirmation records (WAVE_A42C_U0__*.json / WAVE_A42C_BIG_BANK__*.json) world by world:
total delay difference exactly 0.0, identical safety counters, same number of activations.

usage: ident_compare.py --raw IDENT_RAW.json --a42-root /root/a66/refs/a42/root --out IDENTITY_CV1.json
PASS -> exit 0; FAILED -> exit 5 (nothing else may run). The verdict file is written once.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cvlib as L  # noqa: E402

PAIRS = (('IDENT_U0', 'A42C_U0'), ('IDENT_F', 'A42C_BIG_BANK'))


def stored(a42_root, name):
    out = {}
    for p in sorted(Path(a42_root).glob(f'WAVE_{name}__*.json')):
        for r in json.loads(p.read_text(encoding='utf-8'))['records']:
            out[r['world_id']] = r
    return out


def compare(raw, a42_root):
    rows = []
    for mine, theirs in PAIRS:
        ref = stored(a42_root, theirs)
        for w, r in raw['waves'][mine].items():
            a = ref[w]
            acts = [int(r['activations']), int((a.get('a37') or {}).get('activations') or 0)]
            rows.append(dict(wave=mine, world_id=w, seed=r['seed'], j_now=r['J'], j_stored=a['J_veh_s'],
                             abs_diff=abs(r['J']-a['J_veh_s']), safety_equal=r['safety'] == a['safety'],
                             activations=acts))
    ok = len(rows) == 8 and all(x['abs_diff'] == 0.0 and x['safety_equal']
                                and x['activations'][0] == x['activations'][1] for x in rows)
    return dict(schema='claim_validation.identity.v1', run_id=L.RUN_ID,
                rule='U0 and the original bank F on worlds 998600-998603 from the amendment-72 code tree equal the '
                     'stored A42 records exactly: J difference 0.0, identical safety counters, same activations',
                verdict='PASS' if ok else 'FAILED', worlds=rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--raw', required=True)
    p.add_argument('--a42-root', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args(argv)
    doc = L.seal(compare(L.read_json(a.raw), a.a42_root))
    state = L.write_once(a.out, L.text_of(doc))
    print(json.dumps(dict(verdict=doc['verdict'], file=state, out=a.out,
                          max_abs_diff=max(x['abs_diff'] for x in doc['worlds']))))
    return 0 if doc['verdict'] == 'PASS' else 5


if __name__ == '__main__':
    raise SystemExit(main())
