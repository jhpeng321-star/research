"""FULLPOL EV-BC pace audit (CPU fixture; report only): P(target set) per update under the registered
KL cap for imitation steps x lr variants. Usage: python -m tools.coopmwm.fullpol_pace_fixture [steps lr ...]
Reproduces the 2026-11-06 numbers sent to MAIN (20/3e-4 -> 0.80 at u24; 60/3e-4 -> 0.8 by u9)."""
import copy
import sys

import numpy as np
import torch


def main():
    from tools.coopmwm import exploration  # noqa: E402
    exploration.regime = lambda rows: 'ordinary'
    torch.set_num_threads(1)
    from tools.coopmwm import fullpol_learn as fl, fullpol_ev as fe
    from tools.coopmwm import test_fullpol_ev_cpu as T
    cfg = dict(track2_backend=True, run_root='E:/Temp/pace', training_seed=123, v2v=True, v2v_dim=48, v2v_intent=True,
               merge_features=True, host='cpuV', layer_config=dict(hidden_dim=8, learning_rate=.0003, kl_budget=.045,
               clip_ratio=.2, max_grad_norm=.5), trigger_candidates=8, trigger_partner_slots=4, decision_interval_s=60,
               trigger_lr=.0003, layer_epochs=1, trigger_ppo=dict(clip_ratio=.2, value_weight=.5, entropy_weight=.01,
               max_grad_norm=.5), worlds_by_update={'1': [dict(seed=1)]}, practice_starts=False, runtime='u',
               checkpoint='u', accel_bounds=[-3, 2.6], gpu_deadline=9e9, workers=1, load_max=1,
               budget=dict(total=1, phases=dict(t=1)), safety_reference={}, track2_reward=dict(weight=.5, provenance='x'),
               imitation_epochs=20)
    rows = [T.grow('wall_h_solo2', 15169., baseline='mappo')]
    on = dict(T.ld_cfg(cfg, v2v=False), **fe.recommended(provenance='x', regime_guidance=T.guidance(rows)))
    on['trigger_partner_slots'] = 4
    p = T.make(on)
    eps = []
    for k in range(8):
        samples = T._decisions(T.make(on), 6)
        eps += [T._episode(s, f'e{k}_{i}', seed=989220+k) for i, s in enumerate(samples)]
    recs, _ = fl.ev_bc_records(on, 'B', eps)
    def ptarget():
        vals = []
        with torch.no_grad():
            for r in recs:
                b = r['batch']
                pr = p.trigger.distribution(b.context, b.descriptors, b.mask)[0].softmax(-1)
                for i in range(len(b.actions)):
                    if float(r['row_weight'][i]) > 0:
                        vals.append(float(pr[i][r['equivalent'][i]].sum()))
        return np.mean(vals), len(vals)
    base_state = copy.deepcopy(p.trigger.state_dict())
    base_opt = copy.deepcopy(p.trigger_optimizer.state_dict())
    args = sys.argv[1:] or ['20', '3e-4', '60', '3e-4', '20', '1e-3']
    for steps, lr in zip(map(int, args[::2]), map(float, args[1::2])):
        p.trigger.load_state_dict(base_state)
        p.trigger_optimizer.load_state_dict(base_opt)
        for g in p.trigger_optimizer.param_groups:
            g['lr'] = lr
        trace = []
        for u in range(1, 25):
            out = fl.bc_steps(p, recs, epochs=steps, kl_budget=.045, max_grad_norm=.5)
            trace.append((round(ptarget()[0], 3), out['passes'][-1]['kl']))
        first = next((u+1 for u, t in enumerate(trace) if t[0] >= .8), None)
        print(f'steps={steps} lr={lr}: p@6={trace[5][0]} p@12={trace[11][0]} p@24={trace[23][0]} first_u_p>=.8={first}')


if __name__ == '__main__':
    main()
