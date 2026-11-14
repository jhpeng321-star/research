"""AMENDMENT 93 random-draw control (prep 2026-11-14). Zero SUMO, zero WM, zero training.

Draws plans with the S4 search's OWN random plan generator (strategy_search.random_strategy as patched by
a75ss.install(S4 cfg)) and the S4 configuration, RNG random.Random(20261114), in generation order, and applies the
search's own validity checks in the order the S4 search applies them to a random plan:
  1. generator validity: SS.random_strategy validates internally (a ValueError = rejected draw 'validate_error');
  2. S4 lane-set exclusion (a85s4.LaneFilter.bad: role-lane set in exclude_lane_sets) -> 'excluded_lane_set';
  3. run(): SS.uncond (unconditional cfg: gate / start_gate / phases -> None, rand_ name re-keyed),
     len(roles) >= SS.MIN_ROLES -> 'min_roles', SS.banned -> 'banned', duplicate strategy key -> 'duplicate_key'.
Passing plans are written in generation order as TOP_OFFER_CONDITIONS rows (a75ss writer row format) with
score / gain_vs_base / gain_vs_base_lcb / feasible = null: no WM number exists for them and none is looked up.

  python -I -B a93_draw.py draw     -> STREAM_PASS.json (first N_STREAM passing plans) + STREAM_DRAWS.json
  python -I -B a93_draw.py finalize MAPCHECK.json
        -> TOP_OFFER_CONDITIONS_D2800_p60.json (first 12 passing plans the zero-SUMO pin mapping accepts) + DRAWS.json
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
S4DIR = Path('E:/research-dense/experiments/a85_s4_diffsearch_20261113')
SHIP, TREE = S4DIR / 'a85ship', S4DIR / 'testtree'
for p in (str(SHIP), str(TREE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from tools.wzprop import strategy_search as SS  # noqa: E402
import a75ss as A  # noqa: E402
import a85s4 as S4  # noqa: E402

SEED = 20261114
N_KEEP = 12
N_STREAM = 40
CELL = 'D2800_p60'
CFG = S4DIR / 'cfg' / 'A85S4_SEARCH_CFG.json'
S4_RESULT = Path('E:/research-dense/runs/active/C4-V7FT-20261017/a85_s4/STRATEGY_SEARCH_D2800_p60.json')
PINS = {SHIP / 'a75ss.py': 'cae54dc11be5f5220f19eb7399efac2699c7cdf363a90b7fa202766d5b906bc0',
        SHIP / 'a85s4.py': '7ca82976e7183ce205d0591470a60281b553c2d03e8643d479d739b86c419261',
        TREE / 'tools' / 'wzprop' / 'strategy_search.py': '02e119c11a056c13bc54309c5f71e0ca8dd73aeb3c49c5a901e008fda36f54fc',
        CFG: '0a80c0f4c6a8858faf4430c8e55be04a9abc77779337cd91921e1df873b82f5e'}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def setup():
    for p, want in PINS.items():
        if sha(p) != want:
            raise SystemExit(f'pin mismatch {p}')
    cfg = dict(SS.DEFAULTS, **json.loads(CFG.read_text(encoding='utf-8')))
    # the config the S4 search actually ran with: ONLY its 'config' key is read (no score / result field is touched)
    used = json.loads(S4_RESULT.read_text(encoding='utf-8'))['config']
    if json.dumps(used, sort_keys=True) != json.dumps(cfg, sort_keys=True):
        raise SystemExit('cfg file (merged with DEFAULTS) != config recorded by the S4 search')
    A.install(cfg)                                   # a75ss.main does this before search_cell
    S4.set_globals(cfg)                              # search_cell globals
    SS.ONSET_MAX = cfg.get('onset_max')
    SS.SEARCH_GRID = cfg.get('search_grid')
    return cfg


def row(c):
    return dict(family=c['name'], strategy_key=SS.strategy_key(c), score=None, gain_vs_base=None,
                gain_vs_base_lcb=None, feasible=None, extended=A.is_extended(c), offer_condition=A.offer_condition(c))


def draw():
    cfg = setup()
    flt = S4.LaneFilter(S4.excluded_sets(cfg))
    dense, hold_only = bool(cfg['dense']), bool(cfg['hold_only'])
    rc = tuple(cfg['role_counts']) if cfg.get('role_counts') else None
    rng = random.Random(SEED)
    draws, passed, seen = [], [], set()
    i = 0
    while len(passed) < N_STREAM:
        i += 1
        try:
            s = SS.random_strategy(rng, cfg['max_roles'], dense, hold_only, rc)
        except ValueError as e:
            draws.append(dict(draw=i, status='rejected', reason=f'validate_error: {e}'))
            continue
        rec = dict(draw=i, name=s['name'], lanes=sorted({int(r['lane']) for r in s['roles']}))
        if flt.bad(s):
            draws.append(dict(rec, status='rejected', reason='excluded_lane_set'))
            continue
        c = SS.uncond(s)
        k = SS.strategy_key(c)
        rec.update(key=k, uncond_name=c['name'])
        if len(c['roles']) < SS.MIN_ROLES:
            draws.append(dict(rec, status='rejected', reason='min_roles'))
            continue
        if SS.banned(c):
            draws.append(dict(rec, status='rejected', reason='banned'))
            continue
        if k in seen:
            draws.append(dict(rec, status='rejected', reason='duplicate_key'))
            continue
        SS.validate(c)
        seen.add(k)
        draws.append(dict(rec, status='pass', pass_rank=len(passed) + 1))
        passed.append(row(c))
    meta = dict(seed=SEED, generator='strategy_search.random_strategy patched by a75ss.install(S4 cfg)',
                cfg_sha256=sha(CFG), code={str(p.name): w for p, w in PINS.items()}, n_draws=i)
    (HERE / 'STREAM_PASS.json').write_bytes(json.dumps(passed, indent=1).encode())
    (HERE / 'STREAM_DRAWS.json').write_bytes(json.dumps(dict(meta, draws=draws), indent=1).encode())
    print('draws', i, 'passed', len(passed), 'rejected', sum(d['status'] == 'rejected' for d in draws))


def finalize(mapcheck):
    stream = json.loads((HERE / 'STREAM_PASS.json').read_text())
    sd = json.loads((HERE / 'STREAM_DRAWS.json').read_text())
    mc = json.loads(Path(mapcheck).read_text())
    if mc['stream_sha256'] != sha(HERE / 'STREAM_PASS.json'):
        raise SystemExit('MAPCHECK is not for this stream')
    refused = {r['family']: r['reason'] for r in mc['rows'] if r['reason']}
    keep, draws = [], []
    for d in sd['draws']:
        d = dict(d)
        if d['status'] == 'pass':
            if len(keep) >= N_KEEP:
                break
            r = stream[d['pass_rank'] - 1]
            if r['family'] in refused:
                d.update(status='rejected', reason='pin_mapping_refused: ' + refused[r['family']])
            else:
                keep.append(r)
                d['top_rank'] = len(keep)
        draws.append(d)
    if len(keep) != N_KEEP:
        raise SystemExit('stream too short')
    top = HERE / f'TOP_OFFER_CONDITIONS_{CELL}.json'
    top.write_bytes(json.dumps(keep, indent=1).encode())
    out = dict({k: v for k, v in sd.items() if k != 'draws'}, n_draws_used=draws[-1]['draw'],
               rule='first 12 distinct plans (strategy key) passing the S4 search validity checks and the zero-SUMO '
                    'A80 pin mapping (map_rows), in generation order; no WM score used',
               stream_sha256=mc['stream_sha256'], mapcheck_sha256=sha(mapcheck), top_sha256=sha(top),
               n_rejected=sum(d['status'] == 'rejected' for d in draws), draws=draws)
    (HERE / 'DRAWS.json').write_bytes(json.dumps(out, indent=1).encode())
    print('top', sha(top), 'draws used', out['n_draws_used'], 'rejected', out['n_rejected'])


if __name__ == '__main__':
    if sys.argv[1] == 'draw':
        draw()
    else:
        finalize(sys.argv[2])
