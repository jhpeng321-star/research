"""AMENDMENT 95(a) hand-written candidates (prep 2026-11-14). Zero SUMO, zero WM.
strategy_search.seeds() (17, in order) + dense_seeds()['duo12_v20'], UNCHANGED (no a75ss.install, no S4 cfg, no
uncond / lane-set / min_roles filter), written as a75ss TOP_OFFER_CONDITIONS rows (score fields null).
  python -I -B a95_hand.py            -> HAND_ALL.json (18 rows)
"""
import hashlib, json, sys
from pathlib import Path
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
S4DIR = Path('E:/research-dense/experiments/a85_s4_diffsearch_20261113')
for p in (str(S4DIR / 'a85ship'), str(S4DIR / 'testtree')):
    sys.path.insert(0, p)
from tools.wzprop import strategy_search as SS  # noqa: E402
import a75ss as A  # noqa: E402
PINS = {S4DIR / 'a85ship' / 'a75ss.py': 'cae54dc11be5f5220f19eb7399efac2699c7cdf363a90b7fa202766d5b906bc0',
        S4DIR / 'testtree' / 'tools' / 'wzprop' / 'strategy_search.py':
            '02e119c11a056c13bc54309c5f71e0ca8dd73aeb3c49c5a901e008fda36f54fc'}
for p, w in PINS.items():
    assert hashlib.sha256(p.read_bytes()).hexdigest() == w, p
assert SS.validate is A.ORIG['validate']          # not installed: original code objects
plans = list(SS.seeds())
assert len(plans) == 17
plans += [s for s in SS.dense_seeds() if s['name'] == 'duo12_v20']
assert len(plans) == 18
rows = [dict(family=s['name'], strategy_key=A.strategy_key(s), score=None, gain_vs_base=None, gain_vs_base_lcb=None,
             feasible=None, extended=A.is_extended(s), offer_condition=A.offer_condition(s)) for s in plans]
assert all(r['strategy_key'] == SS.strategy_key(s) for r, s in zip(rows, plans))
(HERE / 'HAND_ALL.json').write_bytes(json.dumps(rows, indent=1).encode())
print(len(rows), hashlib.sha256((HERE / 'HAND_ALL.json').read_bytes()).hexdigest())
