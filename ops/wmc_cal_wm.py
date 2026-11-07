"""WM predictions for the SUMO-verified families (harm calibration input; zero SUMO).
    python ops/wmc_cal_wm.py CAL_INPUT.json STATES.json OUT.json CHECKPOINT [workers] [n_states]"""
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.wzprop import strategy_search as SS  # noqa: E402
from tools.wzprop import wm_backend as W  # noqa: E402


def main(cal, states_path, out, ckpt, workers=8, n_states=30, K=6):
    d = json.loads(Path(cal).read_text())
    states = [m for m in json.loads(Path(states_path).read_text()) if m.get('valid') and m['cell'] == d['cell']]
    if len(states) > n_states:
        states = random.Random(SS.DEFAULTS['seed'] + 1).sample(states, n_states)
    kw = dict(checkpoint=ckpt, map_spec='maps/wz3to2/MAP_SPEC.json', device='cuda', max_rows=32)
    wm = W.ParallelWM('wma:WMA_READY.json', workers, local_kw=dict(kw, device='cpu'), mem_fraction=0.12, **kw)
    try:
        specs = [r['spec'] for r in d['rows']]
        per, n = SS.evaluate(wm, states, specs, K, 300, groups=True)
    finally:
        wm.close()
    res = {}
    for r in d['rows']:
        k = SS.strategy_key(r['spec'])
        res[r['family']] = dict(summary=SS.summarize(per[k], 0.5), rows=per[k])
    Path(out).write_text(json.dumps(dict(cell=d['cell'], n_states=len(states), K=K, rollouts=n, checkpoint=ckpt,
                                         families=res), indent=1, default=str))


if __name__ == '__main__':
    a = sys.argv[1:]
    main(a[0], a[1], a[2], a[3], *(int(x) for x in a[4:]))
