"""Input availability only: safe fixed identities, corpus metadata, no outcomes."""
from pathlib import Path
import hashlib
import json
from operate import ROOT, allowed


def main():
    panel = json.loads((ROOT/'docs/paper_results/c1pidiag/PANEL.json').read_text())['states']
    index_path = Path('E:/research-dense/data_cache/pack_v3/INDEX.json')
    idx = json.loads(index_path.read_text())
    # The pack's run items are metadata, not model observations or latent truth.
    items = next(v for v in idx.values() if isinstance(v, list) and v and
                 isinstance(v[0], dict) and 'tick_start' in v[0])
    by_key = {e['key']:e for e in items}
    rows = []
    for e in panel:
        allowed(e['key'])
        item = by_key.get(e['key'])
        probes = []
        if item:
            for path in (item['shard_dir'], 'E:/research-dense/'+item['root_rel']+'/'+e['key'].split('/')[-1]):
                p = Path(allowed(path))
                probes.append(dict(path=str(p), exists=p.exists()))
        rows.append(dict(sid=e['sid'], key=e['key'], t0=e['t0_s'],
            pack_t0=item['t0_s'] if item else None,
            pack_max_pret0_span_s=max(0.,e['t0_s']-item['t0_s']) if item else None,
            original_source_probes=probes))
    value = dict(pack_index_sha256=hashlib.sha256(index_path.read_bytes()).hexdigest(),
        rows=rows, full60_available_in_pack=sum(r['pack_max_pret0_span_s'] is not None and r['pack_max_pret0_span_s']>=60 for r in rows),
        caveat='No source root is assumed deleted merely because these named original paths are absent; no cross-arm or shifted-t0 replacement is valid.')
    out=ROOT/'docs/paper_results/c1pidiag/HISTORY_RECOVERY.json'
    with out.open('x') as f: json.dump(value,f,indent=2)
    print('pack full60 eligible',value['full60_available_in_pack'],'of',len(rows),
          'original roots present',sum(p['exists'] for r in rows for p in r['original_source_probes']))


if __name__=='__main__': main()
