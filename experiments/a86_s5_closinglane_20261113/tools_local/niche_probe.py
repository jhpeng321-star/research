"""CPU-only probe (no WM): S5 generation-0 niches, hand seeds under the filter, novelty saturation.
usage: python -I -B niche_probe.py ROOT"""
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(sys.argv[1])
sys.path[:0] = [str(ROOT / 'ship'), str(ROOT / 'testtree')]
from tools.wzprop import strategy_search as SS  # noqa: E402
import a75ss as A  # noqa: E402
import a86s5 as S5  # noqa: E402

cfg5 = json.loads((ROOT / 'cfg' / 'A86S5_SEARCH_CFG.json').read_text())
sts = [m for m in json.loads((ROOT / 'testtree/data/regime_STATES.json').read_text())
       if m.get('valid') and m['cell'] == 'D2800_p60']


def gen0(cfg, excl):
    cfg = dict(SS.DEFAULTS, **cfg)
    A.uninstall()
    A.install(cfg)
    S5.set_globals(cfg)
    SS.ONSET_MAX, SS.SEARCH_GRID = cfg.get('onset_max'), cfg.get('search_grid')
    flt = S5.LaneFilter(excl)
    _s, pop, _ = S5.initial_population(random.Random(cfg['seed']), sts, cfg, flt)
    return cfg, flt, pop


cfg, flt, pop = gen0(cfg5, S5.excluded_sets(cfg5))
print('S5 gen0 pop', len(pop), 'rejected seeds', len(flt.rejected['seed']), 'rejected random draws', flt.rejected['random'])
c = Counter(S5.niche_label(s) for s in pop)
print('S5 gen0 niches', len(c), dict(sorted(c.items())))
print('S5 gen0 forms', dict(Counter(S5.form_family(s) for s in pop)))
nov = [SS.novelty(SS.uncond(s)) for s in pop]
print('S5 gen0 novelty: min %.3f median %.3f  share at cap(>=%.1f) %.2f' % (
    min(nov), sorted(nov)[len(nov) // 2], SS.NOV_CAP, sum(x >= SS.NOV_CAP for x in nov) / len(nov)))
cfg4 = json.loads((ROOT / 'cfg' / 'S4_REF' / 'A85S4_SEARCH_CFG.json').read_text())
cfg, flt, pop = gen0(cfg4, S5.excluded_sets(cfg4))
c = Counter(S5.niche_label(s) for s in pop)
nov = [SS.novelty(SS.uncond(s)) for s in pop]
print('S4 gen0 niches', len(c), 'novelty median %.3f share at cap %.2f' % (
    sorted(nov)[len(nov) // 2], sum(x >= SS.NOV_CAP for x in nov) / len(nov)))
s2b = json.loads(Path('E:/research-dense/runs/active/C4-V7FT-20261017/a75_s2b/STRATEGY_SEARCH_D2800_p60.json').read_text())
fn = [r.get('novelty', 0.0) for r in s2b['final'] if not r['comparator']]
print('S2b finals novelty median %.3f share at cap %.2f n %d' % (sorted(fn)[len(fn) // 2],
      sum(x >= SS.NOV_CAP for x in fn) / len(fn), len(fn)))
c = Counter(S5.niche_label(r['strategy']) for r in s2b['final'] if not r['comparator'])
print('S2b finals niches', len(c), c.most_common(6))
top = [r for r in s2b['final'] if not r['comparator'] and not r['banned'] and A.activated(r)][:12]
print('S2b global top12 niches', Counter(S5.niche_label(r['strategy']) for r in top).most_common())
cap, glob = S5.capped_top(s2b['final'], 12, 2)
print('S2b niche-capped top12 niches', Counter(S5.niche_label(r['strategy']) for r in cap).most_common())
