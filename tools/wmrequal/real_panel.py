"""Extract real labels only for the exact existing verdict panel keys."""
import json
import argparse
from pathlib import Path
from labels import recorded, safe

BASE = Path('/root/autodl-tmp/c4')
OUT = BASE/'wmrequal20261030'


def entries_for(panel):
    root = (BASE/'panel120/package' if panel == 'panel120' else
            BASE/'wmdeep20261027/score_pairs' if panel == 'pairs300' else BASE/'e3data')
    doc = json.loads(safe(root/'C4_EPISODES_MANIFEST.json').read_text())
    if panel == 'pairs300':
        ref = json.loads(safe(BASE/'cct2mix20261029/WM2mix_pairs300.json').read_text())
        keys = {r['pair_key'] for r in ref['rows']}
        entries = [e for e in doc['episodes'] if e.get('pair_key') in keys and
                   e.get('branch',{}).get('kind') in ('B','SEED')]
        assert len(keys) == 48 and len(entries) == 144
    elif panel == 'panel120':
        entries = doc['episodes']
        assert len(entries) == 120
    else:
        split = json.loads(safe(root/'SPLIT.json').read_text())
        entries = [e for e in doc['episodes'] if split['worlds'][str(e['seed'])] == panel]
        assert len(entries) == 144
    return root, entries


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--identity-map',type=Path,help='observable episode -> recorded vehicle_ids JSON; never assignment data')
    ap.add_argument('--out',type=Path,default=OUT/'REAL.json')
    a=ap.parse_args()
    identity_map=json.loads(safe(a.identity_map).read_text()) if a.identity_map else {}
    result = dict(schema='wmrequal_recorded/1', panels={})
    target=safe(a.out)
    if target.is_file():
        result=json.loads(target.read_text())
    for panel in ('validation','pairs300','calibration','panel120'):
        if panel in result['panels']:
            continue
        root, entries = entries_for(panel)
        rows = []
        for e in entries:
            names=identity_map.get(e['episode'])
            r = recorded(root,e,120 if panel == 'panel120' else 300,identities=names)
            rows.append(dict(pair_key=e['pair_key'],world=e['world'],arm=e['arm'],
                native=e.get('branch',{}).get('kind') == 'B',seed=e['seed'],
                demand=e['demand_vph'],penetration=e['penetration'],**r))
        result['panels'][panel] = rows
        target.write_text(json.dumps(result,indent=1,allow_nan=False))
        print(panel, 'episodes',len(rows),'source_Q0_exact',sum(r['source_available'] for r in rows),
              'Q0_range',[min(r['queue0_total'] for r in rows),max(r['queue0_total'] for r in rows)],flush=True)
    target.write_text(json.dumps(result,indent=1,allow_nan=False))


if __name__ == '__main__':
    main()
