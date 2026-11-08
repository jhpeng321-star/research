"""c4/learnfix2 calibration + offline check (CPU, zero SUMO, read-only on the run root).

calibrate: the single none-logit bias b so that mean choice-row P(none) = target on the u1 recorded contexts of B and R
           under the u0 trigger (track2_000).
check:     pre-registered criteria clean5c/learnfix2/CRITERIA.json (C1-C4) on one replayed update per arm.

  python ops/learnfix2_check.py calibrate <run_root> [target]
  python ops/learnfix2_check.py check <run_root> <arm> <update> <bias>
"""
import gzip
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

torch.set_num_threads(int(os.environ.get('LF_THREADS', '2')))


def _cfg(root):
    from tools.coopmwm import clean4_restart as c4, launch_production as production
    from tools.coopmwm.launch_training import read
    cfg = read(Path(root)/'CONFIG.json')
    b = cfg.get('clean4_budget') or {}
    if b.get('run') in c4.APPROVED_BY_RUN:
        c4.APPROVED_BY_RUN[b['run']] = b['approved_sim_s']
    return production.backend_config(cfg)


def _episodes(root, arm, u):
    out = []
    for j in sorted(glob.glob(os.path.join(root, 'jobs', f't2c_warm_u{u:03}_{arm}_*'))):
        f = os.path.join(j, 'RESULT.json.gz')
        if os.path.exists(f):
            d = json.load(gzip.open(f))
            if d.get('arm') == arm:
                out.append(d)
    return out


def _policy(native, root, arm, u_state):
    from tools.coopmwm import track2_backend as tb
    from tools.coopmwm.v2v_runtime import load_checkpoint
    base = load_checkpoint(os.path.join(root, f'{arm}_checkpoint_{u_state:03}.pt'), config=native)[0]
    p = tb.EpisodePolicy(base, native, arm)
    p.restore_worker(torch.load(os.path.join(root, f'{arm}_track2_{u_state:03}.pt'), map_location='cpu',
                                weights_only=False)['worker'])
    return p


@torch.no_grad()
def _choice_logits(trigger, ticks, bias=0.):
    out = []
    for b in ticks:
        logits = trigger.distribution(b.context, b.descriptors, b.mask)[0]
        c = b.mask[:, 1:].any(1)
        if bool(c.any()):
            lg = logits[c].clone()
            lg[:, 0] += bias
            out.append((lg, b.mask[c]))
    return out


def _stats(rows):
    pn, ent, valid = [], [], []
    for lg, m in rows:
        p = lg.softmax(-1)
        pn.append(p[:, 0])
        ent.append(torch.distributions.Categorical(logits=lg).entropy())
        valid.append(m.sum(1).float())
    pn, ent, valid = torch.cat(pn), torch.cat(ent), torch.cat(valid)
    return dict(rows=int(len(pn)), p_none=float(pn.mean()), entropy=float(ent.mean()),
                log_valid=float(valid.log().mean()))


def _ticks(episodes):
    from tools.coopmwm import track2_backend as tb
    return [tb._batch(s['sample']['track2_trigger']) for e in episodes for s in e['steps']
            if s['sample'].get('track2_trigger') and not e.get('practice')]


def calibrate(root, target=0.90):
    native = _cfg(root)
    rows = []
    for arm in ('B', 'R'):
        p = _policy(native, root, arm, 0)
        rows += _choice_logits(p.trigger, _ticks(_episodes(root, arm, 1)))
    lo, hi = 0., 10.
    for _ in range(60):
        mid = .5*(lo+hi)
        pn = float(torch.cat([(lg.clone().index_add_(1, torch.tensor([0]), torch.full((len(lg), 1), mid))).softmax(-1)[:, 0]
                              for lg, _ in rows]).mean())
        lo, hi = (mid, hi) if pn < target else (lo, mid)
    b = round(.5*(lo+hi), 4)
    print(json.dumps(dict(root=root, target=target, none_bias=b, rows=sum(len(lg) for lg, _ in rows),
                          before=_stats(rows), after=_stats([(lg.clone().index_add_(1, torch.tensor([0]),
                                                                                    torch.full((len(lg), 1), b)), m)
                                                             for lg, m in rows]))))


def check(root, arm, u, bias):
    from tools.coopmwm import (track2_backend as p1, track2_reward as p2, g4_runtime as g4, fullpol_learn as fl,
                               learnfix, launch_production as production)
    from tools.coopmwm.fair_vehicle_log import enabled as fair_enabled, update_step as fair_update_step
    native = _cfg(root)
    native['update'] = u
    native['learnfix'] = native.get('learnfix') or learnfix.value()
    episodes = _episodes(root, arm, u)
    if fair_enabled(native):
        fair_update_step(native, arm, u, episodes)
    hooks = p1.build_arm_runtime(native, arm)
    runtime = hooks['runtime']
    runtime.safety_checked = True
    ticks = _ticks(episodes)
    old = _stats(_choice_logits(runtime.policy.trigger, ticks))                       # C4: old-code control
    trig = runtime.policy.trigger
    trig.none_bias = torch.nn.Parameter(torch.tensor(float(bias)))                    # learnfix2 init
    runtime.policy.trigger_optimizer = trig.optimizer(lr=native['trigger_lr'])
    runtime.cfg['learnfix2'] = learnfix.value2(bias)
    fix_cfg = dict(runtime.cfg)
    ev, _ = fl.ev_bc_records(fix_cfg, arm, episodes) if arm == 'B' else ([], None)

    @torch.no_grad()
    def p_target():
        vals = []
        for r in ev:
            b = r['batch']
            p = trig.distribution(b.context, b.descriptors, b.mask)[0].softmax(-1)
            pt = (p*r['equivalent'].float()).sum(-1)
            vals += [float(pt[k]) for k in range(len(pt)) if float(r['row_weight'][k]) > 0 and int(b.actions[k]) > 0]
        return (sum(vals)/len(vals) if vals else None), len(vals)
    before_rows = _choice_logits(trig, ticks)
    u0 = _stats(before_rows)
    pt0 = p_target()
    envelope = p2.reward_update(native, arm, u, episodes)
    synergy = None
    if g4.enabled(native, u):
        path = g4._envelope_path(native, u)
        if Path(path).exists():
            synergy = json.loads(Path(path).read_text(encoding='utf-8'))
    learning = hooks['train_update'](episodes, rewards=envelope, **({} if synergy is None else dict(synergy=synergy)))
    ppo_rows = _choice_logits(trig, ticks)
    production.bank_move_supervision(runtime)
    from tools.coopmwm.fullpol_learn import supervise
    sup = supervise(runtime, episodes)
    after_rows = _choice_logits(trig, ticks)
    kl = 0.
    n = 0
    for (a, m), (b, _) in zip(before_rows, after_rows):
        kl += float(torch.distributions.kl_divergence(torch.distributions.Categorical(logits=a),
                                                      torch.distributions.Categorical(logits=b)).sum())
        n += len(a)
    def _kl(xs, ys):
        num = den = 0.
        for (a, _), (b, _) in zip(xs, ys):
            num += float(torch.distributions.kl_divergence(torch.distributions.Categorical(logits=a),
                                                           torch.distributions.Categorical(logits=b)).sum())
            den += len(a)
        return num/max(den, 1)
    kl_ppo, kl_bc = _kl(before_rows, ppo_rows), _kl(ppo_rows, after_rows)
    after = _stats(after_rows)
    pt1 = p_target()
    res = dict(run=os.path.basename(root), arm=arm, update=u, none_bias=bias, old_code=old, u0_biased=u0, after=after,
               choice_kl=kl/max(n, 1), choice_kl_ppo=kl_ppo, choice_kl_bc=kl_bc, after_ppo=_stats(ppo_rows),
               p_target=dict(before=pt0, after=pt1), bc_fit=None if sup is None else sup.get('report_bc_kl'),
               none_bias_after=float(trig.none_bias.detach()),
               learnfix=dict((learning.get('learnfix') or {}), steps=len((learning.get('learnfix') or {}).get('steps', ()))),
               credit=(learning.get('learnfix_net') or {}).get('credit_cap'),
               ev_bc=None if sup is None else (sup.get('ev_bc') or {}).get('learnfix'))
    res['learnfix'].pop('advantages', None)
    res['criteria'] = dict(
        C1=u0['p_none'] >= 0.85, C2=res['choice_kl'] >= 0.005,
        C3=(None if pt0[0] is None else pt1[0] > pt0[0]),
        C4=old["p_none"] <= 0.25 and old["entropy"] >= 0.9*old["log_valid"], C5=after["p_none"] >= 0.75)
    print(json.dumps(res, default=str))


if __name__ == '__main__':
    if sys.argv[1] == 'calibrate':
        calibrate(sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 0.90)
    else:
        check(sys.argv[2], sys.argv[3], int(sys.argv[4]), float(sys.argv[5]))
