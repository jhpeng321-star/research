"""S-SUMOES WM 60 s direction screen on cpuQ CPUs (no SUMO). Guidance only; never the fitness.

Reads screens/request_uXX.json; for each of the 8 directions d and each of the 8 frozen
TRAIN starts (S-WMDATA CTRL0 t300 starts, START_SEEDS) runs paired 60 s WM2+mix rollouts
with the E3 C1 trunk and head = theta +/- sigma*d (identical future/entry/lane draws on
both sides = CRN), gain = J60(minus) - J60(plus) (real_J, first two 30 s bins; 90 s
descriptive). Rule: an antithetic pair +/-d is kept iff its gain sign is confident
(median > 0 and q10 > 0, or the mirrored orientation median < 0 and q90 < 0); rank by
|median|; keep the top 3. Writes screens/screen_uXX.json bound to the request sha256.
Known limitation: the frozen WM runtime composes residuals base-relative (no caprel).
"""
from pathlib import Path
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time

OWN = Path('/home/ubuntu/sumoes20261031')
START_SEEDS = (969401, 969402, 969404, 969405, 969407, 969408, 969410, 969411)
STEPS = 120
KEEP = 3
_G = {}


def _init():
    os.nice(0)
    sys.path.insert(0, str(Path(__file__).parent))
    import wm_smoke  # noqa: F401  (sets sys.path + env exactly like the smoke)
    import torch
    from dataclasses import replace
    from wms.control.counterfactual import native_fn
    from wms.control.dreamer import DreamerConfig
    from wms.control.policy import load_policy_components
    from wms.control.policy14_cli import load_members
    from wms.control.train_cli import GroupSignal
    import wm_starts
    torch.set_num_threads(1)
    models, _ = load_members([str(wm_smoke.WM_PATH)], [str(wm_smoke.WM_CONFIG)], 'cpu')
    model = models[0]
    payload = torch.load(wm_smoke.CANDIDATE, map_location='cpu', weights_only=False)
    cfg = replace(DreamerConfig(**payload['config']), horizon=STEPS // 2, entrant_learned_mix=True)
    model.cav_exec_cap = cfg.cav_exec_cap
    cp, ce, _, _ = load_policy_components(wm_smoke.C1, device='cpu')
    cp.eval()
    ce = ce.encoder if isinstance(ce, GroupSignal) else ce
    ce.eval()
    starts = {s: wm_starts.make_start(model, Path(f'/home/ubuntu/wmdata20261031/jobs/s{s}_ctrl0'))
              for s in START_SEEDS}
    _G.update(models=models, model=model, cfg=cfg, cp=cp, ce=ce, starts=starts,
              native=native_fn('head', a_max=1.2))


def _rollout(task):
    import copy
    import torch
    from panel2 import Entries, costs, expand
    from wms.control.dreamer import _entry_process, with_entry_slots
    from wms.control.entrant_mix import make_sampler
    from wms.control.futures import FuturesConfig, SampledFutures, lane_uniform_draws
    from wms.control.nativeinit import pair_merge_streams
    from wms.physics.batched import road_end_exit_bound
    import wm_starts
    key, seed, theta = task
    g = _G
    model, models, cfg, cp = g['model'], g['models'], g['cfg'], g['cp']
    head = cp.residual_head
    theta = torch.tensor(theta, dtype=torch.float64)
    with torch.no_grad():
        head.weight.copy_(theta[:136].reshape(1, 136).to(head.weight))
        head.bias.copy_(theta[136:].to(head.bias))
        begin = time.monotonic()
        torch.manual_seed(965140)
        b = with_entry_slots(copy.deepcopy(g['starts'][seed]), STEPS // 2, 1920,
                             torch.Generator().manual_seed(1_000_003 * 191), queue_state=True)
        plan = SampledFutures(FuturesConfig.from_trainer(cfg), 190).plan(
            models, b.state, b.rollout_kwargs, STEPS)[0]
        initial = expand(b.state, 1)
        pair_merge_streams(initial, torch.zeros(1, dtype=torch.long), 1, 190)
        initial['exit_s_m'] = torch.full_like(initial['s_m'], road_end_exit_bound(model.physics))
        kw = expand({**b.rollout_kwargs, **plan[0]}, 1)
        lanes = lane_uniform_draws(models, 1, STEPS, initial['s_m'].shape[1], 190, device='cpu')
        if lanes is not None:
            kw['lane_uniform'] = expand(lanes, 1)
        learned = None
        if model.physics.entry_learned:
            from wms.model.entry_learned import load_learned_entry
            learned = load_learned_entry()
        process = _entry_process(b.entry_spec, [0], initial['s_m'].shape[1],
                                 cfg.entry_insert_rule, learned, couple_draws=True, sample_index=0)
        entries = Entries(process)
        kw['entries'] = entries
        kw['entrant_driver_sampler'] = make_sampler(model, initial, kw, 190, [0])
        roll = model.rollout(initial, n_steps=STEPS, cav_controller=wm_starts.callback(
            cp, g['ce'], g['native']), **kw)
        bins = costs(roll, entries, b.entry_spec['k0'])['time_bins_30s']['real_J'][0]
    j60, j90 = float(sum(bins[:2])), float(sum(bins[:3]))
    return key, seed, j60, j90, time.monotonic() - begin


def decide(gains):
    import numpy as np
    rows = []
    for d, values in enumerate(gains):
        v = np.asarray(values, dtype=float)
        med, q10, q90 = float(np.median(v)), float(np.percentile(v, 10)), float(np.percentile(v, 90))
        confident = (med > 0 and q10 > 0) or (med < 0 and q90 < 0)
        rows.append(dict(direction=d, median=med, q10=q10, q90=q90, confident=confident))
    ranked = sorted([r for r in rows if r['confident']], key=lambda r: -abs(r['median']))
    kept = sorted(r['direction'] for r in ranked[:KEEP])
    return kept, rows


def main():
    update = int(sys.argv[1])
    procs = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    assert os.getpriority(os.PRIO_PROCESS, 0) == 19
    req_path = OWN / 'screens' / f'request_u{update:02d}.json'
    out = OWN / 'screens' / f'screen_u{update:02d}.json'
    assert not out.exists(), 'screen exists; never overwrite'
    text = req_path.read_text()
    request = json.loads(text)
    assert request['update'] == update
    theta, sigma, dirs = request['theta'], request['sigma'], request['directions']
    tasks = []
    for d, direction in enumerate(dirs):
        for side, sign in (('plus', 1.0), ('minus', -1.0)):
            vec = [t + sign * sigma * e for t, e in zip(theta, direction)]
            tasks += [((d, side), seed, vec) for seed in START_SEEDS]
    begin = time.monotonic()
    ctx = mp.get_context('fork')
    with ctx.Pool(procs, initializer=_init) as pool:
        results = pool.map(_rollout, tasks, chunksize=1)
    table = {(k, s): (j60, j90) for k, s, j60, j90, _ in results}
    gains60 = [[table[((d, 'minus'), s)][0] - table[((d, 'plus'), s)][0] for s in START_SEEDS]
               for d in range(len(dirs))]
    gains90 = [[table[((d, 'minus'), s)][1] - table[((d, 'plus'), s)][1] for s in START_SEEDS]
               for d in range(len(dirs))]
    kept, rows = decide(gains60)
    payload = dict(schema='sumoes_screen/1', update=update,
                   request_sha256=hashlib.sha256(text.encode()).hexdigest(), kept=kept,
                   predicted_gain=[rows[d]['median'] for d in kept],
                   rejected=[d for d in range(len(dirs)) if d not in kept],
                   all_gains=gains60, gains90_descriptive=gains90, decision_rows=rows,
                   rule='confident sign of J60(minus)-J60(plus) over 8 starts '
                        '(median>0&q10>0 or median<0&q90<0), top 3 by |median|',
                   start_seeds=list(START_SEEDS), steps=STEPS, rollouts=len(tasks),
                   wall_s=time.monotonic() - begin, max_rollout_s=max(r[4] for r in results),
                   wm='WM2 b8797242 + learned mix (setupaudit fix runtime, base-relative law)',
                   trunk='E3 C1 policy_latest.pt 64f70af4')
    with out.open('x') as f:
        json.dump(payload, f, indent=1, allow_nan=False)
        f.write('\n')
    print(json.dumps(dict(update=update, kept=kept, wall_s=payload['wall_s'],
                          medians=[round(r['median'], 2) for r in rows])), flush=True)


if __name__ == '__main__':
    main()
