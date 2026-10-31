"""S-SUMOES WM-screen feasibility smoke on cpuQ (no SUMO): one 60 s WM2+mix rollout from a
WMDATA CTRL0 start with the E3 C1 trunk and a given head. Adapted from the frozen
S-SPEEDHEAD wm_worker (make_starts/callback_for 'ctrl0' path); timing + finiteness only."""
from pathlib import Path
from dataclasses import replace
import copy
import json
import os
import sys
import time

BASE = Path('/home/ubuntu/sumoes20261031/wmscreen/c4')
FIX = BASE / 'setupaudit20261030/fix'
sys.path[:0] = [str(FIX / 'src'), str(FIX), str(FIX / 'tools/setupaudit')]
os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                  OPENBLAS_NUM_THREADS='1', C4_ALLOWED_ROOTS=str(BASE) + ':/home/ubuntu',
                  C4_EXECUTION_PARITY='2', C4_SUMO_LEADER_RULE='1',
                  C4_EXECUTION_DRIVER=str(FIX / 'configs/execparity_cav_idm_v1.json'),
                  WMS_WMSPEED_DEDUPE_MOBIL='1', WMS_WMSPEED_MOBIL_LAZY_SAFE='1',
                  WMS_SPEED_PIVOT_DIAG_OFF='1', WMS_SPEED_FAST_MERGE_LANE='1',
                  WMS_SPEED_FAST_CAV_DRIVE='1', WMS_SPEED_FAST_LONG_HISTORY='1',
                  WMS_SPEED_GRAPH_CVAR='1')
import torch  # noqa: E402

torch.set_num_threads(1)
WM_PATH = BASE / 'wmcombined_wm2/wm_s.pt'
WM_CONFIG = BASE / 'restart20261026/runtime_longint/configs/c4_wm_loop14_r16_ft.toml'
CANDIDATE = BASE / 'setupaudit20261030/policy_000190.pt'
C1 = Path('/home/ubuntu/e3final/inputs/policy_latest.pt')


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 969404
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    t = time.monotonic()
    from panel2 import Entries, costs, expand
    from wms.control.counterfactual import native_fn
    from wms.control.dreamer import DreamerConfig, _entry_process, with_entry_slots
    from wms.control.entrant_mix import make_sampler
    from wms.control.futures import FuturesConfig, SampledFutures, lane_uniform_draws
    from wms.control.nativeinit import pair_merge_streams
    from wms.control.policy import load_policy_components
    from wms.control.policy14_cli import load_members
    from wms.control.train_cli import GroupSignal
    from wms.physics.batched import road_end_exit_bound
    sys.path.insert(0, str(Path(__file__).parent))
    import wm_starts
    models, receipt = load_members([str(WM_PATH)], [str(WM_CONFIG)], 'cpu')
    model = models[0]
    payload = torch.load(CANDIDATE, map_location='cpu', weights_only=False)
    cfg = replace(DreamerConfig(**payload['config']), horizon=steps // 2, entrant_learned_mix=True)
    model.cav_exec_cap = cfg.cav_exec_cap
    print('loaded', round(time.monotonic() - t, 1), receipt if isinstance(receipt, str) else '',
          flush=True)
    batch = wm_starts.make_start(model, Path(f'/home/ubuntu/wmdata20261031/jobs/s{seed}_ctrl0'))
    cp, ce, _, _ = load_policy_components(C1, device='cpu')
    torch.nn.init.zeros_(cp.residual_head.weight)
    torch.nn.init.zeros_(cp.residual_head.bias)
    cp.eval()
    ce = ce.encoder if isinstance(ce, GroupSignal) else ce
    ce.eval()
    native = native_fn('head', a_max=1.2)
    with torch.no_grad():
        fi = 0
        begin = time.monotonic()
        torch.manual_seed(965140)
        b = with_entry_slots(copy.deepcopy(batch), steps // 2, 1920,
                             torch.Generator().manual_seed(1_000_003 * 191), queue_state=True)
        plan = SampledFutures(FuturesConfig.from_trainer(cfg), 190).plan(
            models, b.state, b.rollout_kwargs, steps)[0]
        initial = expand(b.state, 1)
        pair_merge_streams(initial, torch.zeros(1, dtype=torch.long), 1, 190 + fi)
        initial['exit_s_m'] = torch.full_like(initial['s_m'], road_end_exit_bound(model.physics))
        kw = expand({**b.rollout_kwargs, **plan[fi]}, 1)
        lanes = lane_uniform_draws(models, 1, steps, initial['s_m'].shape[1], 190, device='cpu')
        if lanes is not None:
            kw['lane_uniform'] = expand(lanes, 1)
        learned = None
        if model.physics.entry_learned:
            from wms.model.entry_learned import load_learned_entry
            learned = load_learned_entry()
        process = _entry_process(b.entry_spec, [0], initial['s_m'].shape[1],
                                 cfg.entry_insert_rule, learned, couple_draws=True, sample_index=fi)
        entries = Entries(process)
        kw['entries'] = entries
        kw['entrant_driver_sampler'] = make_sampler(model, initial, kw, 190, [0])
        cb = wm_starts.callback(cp, ce, native)
        roll = model.rollout(initial, n_steps=steps, cav_controller=cb, **kw)
        score = costs(roll, entries, b.entry_spec['k0'])
        print('rollout_s', round(time.monotonic() - begin, 1), 'keys', list(score)[:6], flush=True)
        print(json.dumps({h: {k: float(v) if hasattr(v, '__float__') else str(type(v))
                              for k, v in score[h].items() if k in ('J', 'real_J')}
                          for h in score if isinstance(score[h], dict)})[:800])


if __name__ == '__main__':
    main()
