"""CPU-only: count WM rollouts the search WOULD request, using the StubWM fake (no WM, no GPU). Activation and
dedup are the real code; only scores (hence parent choice in generations >= 1) are fake.
usage: python -I -B stub_rollout_count.py ROOT CFG.json [k=v ...]"""
import json, sys, time
from pathlib import Path
ROOT = Path(sys.argv[1])
sys.path[:0] = [str(ROOT / 'ship'), str(ROOT / 'testtree')]
from tools.wzprop import strategy_search as SS
from tools.wzprop import wm_backend as W
import a75ss as A
import a86s5 as S4
cfg = json.loads(Path(sys.argv[2]).read_text()); cfg = cfg.get('config', cfg)
for kv in sys.argv[3:]:
    k, v = kv.split('='); cfg[k] = json.loads(v)
cfg = dict(SS.DEFAULTS, **cfg)
sts = [m for m in json.loads((ROOT / 'testtree/data/regime_STATES.json').read_text()) if m.get('valid') and m['cell'] == 'D2800_p60']
log = []
A.uninstall(); A.install(cfg)
t = time.time()
res = S4.search_cell(W.StubWM(), sts, dict(cfg), log=log.append)
A.uninstall()
print(json.dumps(dict(seed=cfg['seed'], excl=cfg.get('exclude_lane_sets'), n_init=cfg['n_init'], gens=cfg['generations'],
      kids=cfg['n_children'], imm=cfg['n_immigrants'], evaluated_per_gen=[r['evaluated'] for r in log],
      rollouts_per_gen=[r['rollouts'] for r in log], total=res['wm_rollouts'], n_final=len(res['final']),
      final_rollouts=res['wm_rollouts'] - log[-1]['rollouts'], a85=res.get('a85', {}).get('generated_per_generation'),
      cpu_s=round(time.time() - t, 1))))
