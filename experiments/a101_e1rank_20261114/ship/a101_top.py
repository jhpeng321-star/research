"""A101 S-E1-RANK top files (prep 2026-11-14). Zero SUMO, zero WM.
Rows come unchanged from a86_s5/STRATEGY_SEARCH_D2800_p60.json list `final`, matched by strategy.name, written with
the exact S5 writer row expression (a86s5.write_niche_top: A.offer_condition / A.is_extended of ship a75ss cae54dc1).
Self-check: the same expression over the 12 shortlist rows reproduces a86_s5 NICHECAP2 file byte for byte.
  python -I -B a101_top.py   -> TOP_OFFER_CONDITIONS_A101{A,B,C}_D2800_p60.json
"""
import hashlib, json, sys
from pathlib import Path
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
S5 = Path('E:/research-dense/experiments/a86_s5_closinglane_20261113')
RUN = Path('E:/research-dense/runs/active/C4-V7FT-20261017')
for p in (str(S5 / 'ship'), str(S5 / 'testtree')):
    sys.path.insert(0, p)
import a75ss as A  # noqa: E402

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

PINS = {S5 / 'ship' / 'a75ss.py': 'cae54dc11be5f5220f19eb7399efac2699c7cdf363a90b7fa202766d5b906bc0',
        S5 / 'ship' / 'a86s5.py': '71dda3e8436d1202ef54235f1c02db0cd38e485329cf17a036cd36bd491e152e',
        RUN / 'a86_s5' / 'STRATEGY_SEARCH_D2800_p60.json': '30f10455703519de3f9b998883ec79a761fe4e7cd8ff06bcb299cc321aa850a9',
        RUN / 'a86_s5' / 'TOP_OFFER_CONDITIONS_NICHECAP2_FOR_SUMO_D2800_p60.json':
            '2209791012e41ce35f184cd4c155632e65bdb7b59442495f005c5a76b0fbf6fb',
        RUN / 'a101_e1rank' / 'A101_GROUPS.json': 'a0269a7169137df9b32c3208674187276875672df1a28b3e14738eb46c407114'}
for p, w in PINS.items():
    assert sha(p) == w, p
assert A.offer_condition is A.ORIG.get('offer_condition', A.offer_condition)
res = json.loads((RUN / 'a86_s5' / 'STRATEGY_SEARCH_D2800_p60.json').read_text())
byname = {}
for r in res['final']:
    assert r['strategy']['name'] not in byname
    byname[r['strategy']['name']] = r

def row(r):   # verbatim row expression of a86s5.write_niche_top
    return dict(family=r['strategy']['name'], strategy_key=r['key'], score=r['score'], gain_vs_base=r['gain_vs_base'],
                gain_vs_base_lcb=r['gain_vs_base_lcb'], feasible=r['feasible'], extended=A.is_extended(r['strategy']),
                offer_condition=A.offer_condition(r['strategy']))

ref_raw = (RUN / 'a86_s5' / 'TOP_OFFER_CONDITIONS_NICHECAP2_FOR_SUMO_D2800_p60.json').read_bytes()
ref = json.loads(ref_raw)
mine = [row(byname[x['family']]) for x in ref]
assert mine == ref, 'content mismatch on shortlist'
assert json.dumps(mine, indent=1).encode() == ref_raw, 'bytes mismatch on shortlist'
for x in ref:
    assert x['strategy_key'] == A.strategy_key(byname[x['family']]['strategy'])
print('shortlist reproduced byte-for-byte:', len(ref))
G = json.loads((RUN / 'a101_e1rank' / 'A101_GROUPS.json').read_text())['groups']
allg = [n for g in 'ABC' for n in G[g]]
assert len(allg) == 31 == len(set(allg)) and not set(allg) & {x['family'] for x in ref}
for g in 'ABC':
    rows = [row(byname[n]) for n in G[g]]
    for n, r in zip(G[g], rows):
        s = byname[n]
        assert not s['comparator'] and not s['banned'], n
        assert r['strategy_key'] == A.strategy_key(s['strategy']), n
    out = HERE / f'TOP_OFFER_CONDITIONS_A101{g}_D2800_p60.json'
    out.write_bytes(json.dumps(rows, indent=1).encode())
    print(g, len(rows), sha(out), 'extended', sum(r['extended'] for r in rows))
