"""c4/learnfix offline replay (CPU, zero SUMO, read-only on the run root; nothing is written there).

Rebuilds one recorded arm update (F8U / CLEAN5C4) exactly as launch_training.update does - same runtime from the
u-1 checkpoint/sidecar, the recorded episodes (RESULT.json.gz), the P2 rewards, the fairness groups, the recorded
G4 envelope - and runs train + the EV-BC supervision twice from the same start: inherited code vs learnfix.
Reports, on the update's own decision ticks: choice-row KL(before || after), P(none) on choice rows, and (B, with
guidance) P(target) on the learnfix act rows and P(none) on its explicit none rows.

  python ops/learnfix_replay.py <run_root> <arm> <update> > REPLAY_<run>_<arm>_u<update>.json
"""
import gc
import glob
import gzip
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

torch.set_num_threads(int(os.environ.get('LF_THREADS', '2')))


def load_episodes(root, arm, u):
    out = []
    for j in sorted(glob.glob(os.path.join(root, 'jobs', f't2c_warm_u{u:03}_*{arm}_*'))):
        f = os.path.join(j, 'RESULT.json.gz')
        if not os.path.exists(f) or f'_{arm}_' not in os.path.basename(j):
            continue
        d = json.load(gzip.open(f))
        if d.get('arm') != arm:
            continue
        out.append(d)
        gc.collect()
    return out


@torch.no_grad()
def measure(policy, ticks, ev_records):
    from tools.coopmwm.learnfix import choice_mask
    probs = [policy.trigger.distribution(b.context, b.descriptors, b.mask)[0].softmax(-1) for b in ticks]
    choice = [p[choice_mask(b)] for p, b in zip(probs, ticks)]
    p_none = torch.cat([c[:, 0] for c in choice if len(c)])
    out = dict(choice_rows=int(len(p_none)), p_none_choice=float(p_none.mean()) if len(p_none) else None)
    act, none = [], []
    for r in ev_records:
        b = r['batch']
        p = policy.trigger.distribution(b.context, b.descriptors, b.mask)[0].softmax(-1)
        pt = (p*r['equivalent'].float()).sum(-1)
        for k in range(len(pt)):
            if float(r['row_weight'][k]) <= 0:
                continue
            (act if int(b.actions[k]) else none).append(float(pt[k]))
    out.update(act_rows=len(act), p_target_act=sum(act)/len(act) if act else None,
               none_rows=len(none), p_none_on_none_rows=sum(none)/len(none) if none else None)
    return out, probs


def choice_kl(before, after, ticks):
    from tools.coopmwm.learnfix import choice_mask
    num, den = 0., 0
    for a, b, t in zip(before, after, ticks):
        c = choice_mask(t)
        if not bool(c.any()):
            continue
        num += float(torch.distributions.kl_divergence(torch.distributions.Categorical(probs=a[c]),
                                                       torch.distributions.Categorical(probs=b[c])).sum())
        den += int(c.sum())
    return num/den if den else 0.


def run(root, arm, u, fix):
    from tools.coopmwm import (launch_production as production, track2_backend as p1, track2_reward as p2,
                               g4_runtime as g4, fullpol_learn as fl, learnfix)
    from tools.coopmwm.fair_vehicle_log import update_step as fair_update_step
    from tools.coopmwm.launch_training import read
    cfg = read(Path(root)/'CONFIG.json')
    from tools.coopmwm import clean4_restart as c4      # replay only: the recorded run's own approved envelope
    budget = cfg.get('clean4_budget') or {}
    if budget.get('run') in c4.APPROVED_BY_RUN:
        c4.APPROVED_BY_RUN[budget['run']] = budget['approved_sim_s']
    native = production.backend_config(cfg)
    native['update'] = u
    if fix:
        native['learnfix'] = learnfix.value()
    else:
        native.pop('learnfix', None)
    episodes = load_episodes(root, arm, u)
    from tools.coopmwm.fair_vehicle_log import enabled as fair_enabled
    if fair_enabled(native):
        fair_update_step(native, arm, u, episodes)          # B/R: attach groups from A's labels (read only)
    hooks = p1.build_arm_runtime(native, arm)
    runtime = hooks['runtime']
    runtime.safety_checked = True
    ticks = [p1._batch(s['sample']['track2_trigger']) for e in episodes for s in e['steps']
             if s['sample'].get('track2_trigger') and not e.get('practice')]
    fix_cfg = dict(native, learnfix=learnfix.value())          # the learnfix labels define the measured EV rows
    ev_eval, _ = fl.ev_bc_records(fix_cfg, arm, episodes) if arm == 'B' else ([], None)
    before, p0 = measure(runtime.policy, ticks, ev_eval)
    envelope = p2.reward_update(native, arm, u, episodes)
    synergy = None
    if arm != 'A' and g4.enabled(native, u):
        path = g4._envelope_path(native, u)
        synergy = json.loads(Path(path).read_text(encoding='utf-8'))
    kw = dict(rewards=envelope) if synergy is None else dict(rewards=envelope, synergy=synergy)
    learning = hooks['train_update'](episodes, **kw)
    after_ppo, p1_ = measure(runtime.policy, ticks, ev_eval)
    kl_ppo = choice_kl(p0, p1_, ticks)
    production.bank_move_supervision(runtime)
    from tools.coopmwm.fullpol_learn import supervise
    sup = supervise(runtime, episodes)
    after, p2_ = measure(runtime.policy, ticks, ev_eval)
    return dict(run=os.path.basename(root), arm=arm, update=u, learnfix=fix, episodes=len(episodes),
                ticks=len(ticks), before=before, after_ppo=after_ppo, after=after,
                choice_kl_ppo=kl_ppo, choice_kl_total=choice_kl(p0, p2_, ticks),
                ppo=learning.get('fullpol_ppo_kl'), learnfix_receipt={k: v for k, v in (learning.get('learnfix') or {}).items()
                                                                      if k != 'steps'},
                learnfix_steps=len((learning.get('learnfix') or {}).get('steps', ())),
                ev_bc=None if sup is None else dict(fit=sup['fit'], rows=(sup.get('ev_bc') or {}).get('rows'),
                                                    learnfix=(sup.get('ev_bc') or {}).get('learnfix')))


if __name__ == '__main__':
    root, arm, u = sys.argv[1], sys.argv[2], int(sys.argv[3])
    modes = sys.argv[4].split(',') if len(sys.argv) > 4 else ['base', 'fix']
    out = [run(root, arm, u, m == 'fix') for m in modes]
    print(json.dumps(out, indent=1, default=str))
