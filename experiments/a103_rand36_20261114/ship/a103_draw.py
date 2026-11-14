"""AMENDMENT 103 S-E1-RAND36 random-draw control (prep 2026-11-14). Zero SUMO, zero WM, zero training.

Draws plans with the S5 search's OWN random plan generator and configuration: a86s5.LaneFilter.random =
strategy_search.random_strategy (as patched by a75ss.install(S5 cfg)) with the S5 lane-set exclusion
exclude_lane_sets [[1],[2],[1,2]]; S5 cfg A86S5_SEARCH_CFG.json merged with SS.DEFAULTS (asserted equal to the
'config' key recorded by the S5 search output). RNG random.Random(20261115), generation order. Instead of
LaneFilter.random's silent redraw loop each draw is taken once and an excluded-lane-set draw is recorded as rejected
(the next accepted draws are identical to what LaneFilter.random would return from the same RNG stream).
Checks in the order the S5 search applies them to a random plan:
  1. generator validity (ValueError -> 'validate_error');
  2. S5 lane-set exclusion (a86s5.LaneFilter.bad) -> 'excluded_lane_set';
  3. run(): SS.uncond, len(roles) >= SS.MIN_ROLES -> 'min_roles'; SS.banned -> 'banned'; duplicate key -> 'duplicate_key';
  4. A103 rule: key equals a plan the S5 search evaluated at any recorded stage (screened, final, archives, best)
     -> 's5_evaluated_key'; key equals any A93 draw key (A93 STREAM_DRAWS.json, all 50 draws) -> 'a93_draw_key';
  5. zero-SUMO A80 pin mapping (a80_s2b.map_row, run on Box 3 by mapcheck_a103.py) -> 'pin_mapping_refused'.
Rows: a75ss writer format with score / gain_vs_base / gain_vs_base_lcb / feasible = null (no WM number exists).

Generator evidence: `verify` reproduces the S5 search's generation 0 (72 random plans; seed 20261187, screen-state
sample of the 18 D2800_p60 states first, exactly as a86s5.initial_population) and asserts its keys equal the first
72 entries of the recorded S5 'screened' list, in order.

  python -I -B a103_draw.py verify
  python -I -B a103_draw.py draw                -> STREAM_PASS.json + STREAM_DRAWS.json
  python -I -B a103_draw.py finalize MAPCHECK.json
        -> TOP_OFFER_CONDITIONS_A103{A,B,C}_D2800_p60.json (12 each, draw order) + TOP_ALL36 + DRAWS.json
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
S5DIR = Path('E:/research-dense/experiments/a86_s5_closinglane_20261113')
SHIP, TREE = S5DIR / 'ship', S5DIR / 'testtree'
for p in (str(SHIP), str(TREE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from tools.wzprop import strategy_search as SS  # noqa: E402
import a75ss as A  # noqa: E402
import a86s5 as S5  # noqa: E402

SEED = 20261115
N_KEEP = 36
N_STREAM = 60
CELL = 'D2800_p60'
RUN = Path('E:/research-dense/runs/active/C4-V7FT-20261017')
CFG = S5DIR / 'cfg' / 'A86S5_SEARCH_CFG.json'
STATES = TREE / 'data' / 'regime_STATES.json'
S5_RESULT = RUN / 'a86_s5' / 'STRATEGY_SEARCH_D2800_p60.json'
A93_DRAWS = Path('E:/research-dense/experiments/a93_randomcontrol_20261114/STREAM_DRAWS.json')
PINS = {SHIP / 'a75ss.py': 'cae54dc11be5f5220f19eb7399efac2699c7cdf363a90b7fa202766d5b906bc0',
        SHIP / 'a86s5.py': '71dda3e8436d1202ef54235f1c02db0cd38e485329cf17a036cd36bd491e152e',
        TREE / 'tools' / 'wzprop' / 'strategy_search.py': '02e119c11a056c13bc54309c5f71e0ca8dd73aeb3c49c5a901e008fda36f54fc',
        CFG: '3adc266397fad96a11430e4251f6155eaf4a8ad90b47ffe7b0ba50b1e9b9c215',
        STATES: '74aef8f8047a073e16097aa1e104fbf72ab32ca1dd16c7a53f4ad58c70a154e3',
        S5_RESULT: '30f10455703519de3f9b998883ec79a761fe4e7cd8ff06bcb299cc321aa850a9',
        A93_DRAWS: 'd5455bb0178f59c8ee4175ac4039bba509a9f3af5c1387f0462d0479507de4b5'}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def setup():
    for p, want in PINS.items():
        if sha(p) != want:
            raise SystemExit(f'pin mismatch {p}')
    cfg = dict(SS.DEFAULTS, **json.loads(CFG.read_text(encoding='utf-8')))
    res = json.loads(S5_RESULT.read_text(encoding='utf-8'))
    # only the 'config' key and the plan KEYS are read from the S5 output; no score / outcome field is used
    if json.dumps(res['config'], sort_keys=True) != json.dumps(cfg, sort_keys=True):
        raise SystemExit('cfg file (merged with DEFAULTS) != config recorded by the S5 search')
    A.install(cfg)
    S5.set_globals(cfg)
    SS.ONSET_MAX = cfg.get('onset_max')
    SS.SEARCH_GRID = cfg.get('search_grid')
    return cfg, res


def s5_keys(res):
    ks = {r['key'] for r in res['screened']} | {r['key'] for r in res['final']}
    for lst in res['archives'].values():
        ks |= {r['key'] for r in lst}
    ks |= {b['key'] for b in res['best'].values() if b}
    return ks


def a93_keys():
    d = json.loads(A93_DRAWS.read_text())
    return {x['key'] for x in d['draws'] if x.get('key')}


def verify():
    cfg, res = setup()
    states = [m for m in json.loads(STATES.read_text()) if m.get('valid') and m['cell'] == CELL]
    flt = S5.LaneFilter(S5.excluded_sets(cfg))
    _scr, pop, _rc = S5.initial_population(random.Random(cfg['seed']), states, cfg, flt)
    keys = []
    for c in pop:
        c = SS.uncond(c)
        if c is None or len(c['roles']) < SS.MIN_ROLES or SS.banned(c):
            continue
        k = SS.strategy_key(c)
        if k not in keys:
            keys.append(k)
    rec = [r['key'] for r in res['screened'][:int(cfg['n_init'])]]
    assert len(pop) == int(cfg['n_init']) == 72 and all(s['name'].startswith('rand_') for s in pop)
    assert keys == rec, 'S5 generation 0 not reproduced'
    gsha = hashlib.sha256(json.dumps(keys).encode()).hexdigest()
    print('S5_GEN0_REPRODUCED', len(keys), 'rejected_random', flt.rejected['random'], 'rejected_seeds',
          len(flt.rejected['seed']), 'gen0_key_sha', gsha)
    return gsha, flt.rejected['random']


def row(c):
    return dict(family=c['name'], strategy_key=SS.strategy_key(c), score=None, gain_vs_base=None,
                gain_vs_base_lcb=None, feasible=None, extended=A.is_extended(c), offer_condition=A.offer_condition(c))


def draw():
    gsha, grej = verify()
    cfg, res = setup()
    flt = S5.LaneFilter(S5.excluded_sets(cfg))
    dense, hold_only = bool(cfg['dense']), bool(cfg['hold_only'])
    rc = tuple(cfg['role_counts']) if cfg.get('role_counts') else None
    s5k, a93k = s5_keys(res), a93_keys()
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
        rec = dict(draw=i, name=s['name'], lanes=sorted(S5.lane_set(s)))
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
        if k in s5k:
            draws.append(dict(rec, status='rejected', reason='s5_evaluated_key'))
            continue
        if k in a93k:
            draws.append(dict(rec, status='rejected', reason='a93_draw_key'))
            continue
        SS.validate(c)
        seen.add(k)
        rec['form'] = S5.form_family(c)
        rec['niche'] = S5.niche_label(c)
        draws.append(dict(rec, status='pass', pass_rank=len(passed) + 1))
        passed.append(row(c))
    meta = dict(seed=SEED, generator='a86s5.LaneFilter.random = strategy_search.random_strategy patched by '
                                     'a75ss.install(S5 cfg) + S5 exclude_lane_sets (redraw loop unrolled, rejects recorded)',
                cfg_sha256=sha(CFG), code={p.name: w for p, w in PINS.items()}, n_draws=i,
                s5_gen0_reproduced=dict(n=72, key_list_sha256=gsha, rejected_random=grej),
                n_s5_keys_excluded=len(s5k), n_a93_keys_excluded=len(a93k))
    (HERE / 'STREAM_PASS.json').write_bytes(json.dumps(passed, indent=1).encode())
    (HERE / 'STREAM_DRAWS.json').write_bytes(json.dumps(dict(meta, draws=draws), indent=1).encode())
    print('draws', i, 'passed', len(passed), 'rejected', Counter(d['reason'].split(':')[0] for d in draws
                                                                  if d['status'] == 'rejected'))


def finalize(mapcheck):
    stream = json.loads((HERE / 'STREAM_PASS.json').read_text())
    sd = json.loads((HERE / 'STREAM_DRAWS.json').read_text())
    mc = json.loads(Path(mapcheck).read_text())
    if mc['stream_sha256'] != sha(HERE / 'STREAM_PASS.json'):
        raise SystemExit('MAPCHECK is not for this stream')
    refused = {r['family']: r['reason'] for r in mc['rows'] if r['reason']}
    form = {r['family']: r['form'] for r in mc['rows']}
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
                d['keep_rank'] = len(keep)
                d['root'] = 'abc'[(len(keep) - 1) // 12]
                d['a80_form'] = form[r['family']]
        draws.append(d)
    if len(keep) != N_KEEP:
        raise SystemExit('stream too short')
    tops = {}
    allf = HERE / f'TOP_ALL36_A103_{CELL}.json'
    allf.write_bytes(json.dumps(keep, indent=1).encode())
    for j, g in enumerate('ABC'):
        f = HERE / f'TOP_OFFER_CONDITIONS_A103{g}_{CELL}.json'
        f.write_bytes(json.dumps(keep[12 * j:12 * j + 12], indent=1).encode())
        tops[g] = sha(f)
    kept = [d for d in draws if d.get('keep_rank')]
    out = dict({k: v for k, v in sd.items() if k != 'draws'}, n_draws_used=draws[-1]['draw'],
               rule='first 36 distinct plans (strategy key) passing the S5 search validity checks, not equal to any '
                    'S5-evaluated key or A93 draw key, and accepted by the zero-SUMO A80 pin mapping (map_row), in '
                    'generation order; split 1-12 / 13-24 / 25-36 into roots a / b / c; no WM score used',
               stream_sha256=mc['stream_sha256'], mapcheck_sha256=sha(mapcheck), all36_sha256=sha(allf),
               top_sha256=tops,
               key_list_sha256=hashlib.sha256(json.dumps([r['strategy_key'] for r in keep]).encode()).hexdigest(),
               n_rejected=sum(d['status'] == 'rejected' for d in draws),
               rejected_by_reason=dict(Counter(d['reason'].split(':')[0] for d in draws if d['status'] == 'rejected')),
               a80_forms=dict(Counter(d['a80_form'] for d in kept)),
               a80_forms_by_root={g: dict(Counter(d['a80_form'] for d in kept if d['root'] == g)) for g in 'abc'},
               role_lane_sets=dict(Counter(''.join(map(str, d['lanes'])) for d in kept)),
               n_roles=dict(Counter(str(len(r['offer_condition'].get('roles', []))) if isinstance(
                   r.get('offer_condition'), dict) else 'na' for r in keep)),
               draws=draws)
    (HERE / 'DRAWS.json').write_bytes(json.dumps(out, indent=1).encode())
    print('all36', out['all36_sha256'], 'keys', out['key_list_sha256'], 'tops', tops)
    print('draws used', out['n_draws_used'], 'rejected', out['rejected_by_reason'])
    print('forms', out['a80_forms'], 'by root', out['a80_forms_by_root'], 'lanes', out['role_lane_sets'],
          'n_roles', out['n_roles'])


if __name__ == '__main__':
    {'verify': verify, 'draw': draw}.get(sys.argv[1], lambda: finalize(sys.argv[2]))()
