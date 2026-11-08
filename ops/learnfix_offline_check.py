"""LEARNCHECK (2026-11-09): pre-launch offline proof that the c4/learnfix training learns. CPU, zero SUMO.

Replays ONE recorded update of arm B (or R) from the u-1 checkpoint on the update's recorded samples, exactly the
way launch_training.update builds it (same runtime, RESULT.json.gz episodes, P2 rewards, fairness groups, recorded
G4 envelope), with ``--code old`` (cfg without ``learnfix`` = the inherited path) or ``--code new`` (the registered
learnfix value; plus ``learnfix_demo`` when ``--demo`` is given). The run root is only read; outputs go to --out.

Reports: choice-row KL(before || after) (rows with > 1 valid action) after PPO and after the whole update (PPO +
EV-BC); P(target) on learnfix act rows (teacher-admitted for CLEAN5C4) and on demo act rows; P(none) on choice rows;
advantage RMS vs the entropy term; the fairness-credit probe (largest-harm episode with acting rows: raw and
normalised advantages of its acting rows with the paired-A harm vs with that episode's harm set to 0).

  python ops/learnfix_offline_check.py check --root R --arm B --update 1 --code new [--demo P --demo-sha256 S] --out F
  python ops/learnfix_offline_check.py summarize F1.json F2.json ...      # markdown + overall PASS/FAIL
"""
import argparse
import gc
import glob
import gzip
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

FORBIDDEN = ('sealed', 'holdout', 'heldout', 'cal_split', 'chunk5', 'j4b', 'inc1')
CRITERIA = dict(kl_lo=0.01, kl_hi=0.045, p_none_rise_max=0.05, old_ppo_kl_max=0.01)


class _Captured(Exception):
    pass


def guard(path):
    low = str(path).lower()
    bad = [w for w in FORBIDDEN if w in low]
    if bad:
        raise SystemExit(f'refusing forbidden path {path} ({bad})')
    return path


# ------------------------------------------------------------------ pure metrics (unit-tested)

def choice_mask(mask):
    return mask[:, 1:].any(1)


def choice_kl(before, after, masks):
    """Mean KL(before || after) over choice rows; before/after are lists of [n, slots] log-prob tensors."""
    num, den = 0., 0
    for a, b, m in zip(before, after, masks):
        c = choice_mask(m)
        if not bool(c.any()):
            continue
        p, q = a[c], b[c]
        valid = m[c]
        t = torch.where(valid, p.exp()*(p-q), torch.zeros_like(p))
        num += float(t.sum())
        den += int(c.sum())
    return num/den if den else 0.


def p_none_choice(logps, masks):
    vals = [lp[choice_mask(m)][:, 0].exp() for lp, m in zip(logps, masks) if bool(choice_mask(m).any())]
    v = torch.cat(vals) if vals else torch.zeros(0)
    return dict(rows=int(len(v)), mean=float(v.mean()) if len(v) else None)


def p_target(records, logps):
    """Mean P(equivalent target set) over act-labelled rows (action != 0, row_weight > 0)."""
    vals = []
    for r, lp in zip(records, logps):
        pt = (lp.exp()*r['equivalent'].float()).sum(-1)
        for k in range(len(pt)):
            if float(r['row_weight'][k]) > 0 and int(r['batch'].actions[k]) != 0:
                vals.append(float(pt[k]))
    return dict(rows=len(vals), mean=sum(vals)/len(vals) if vals else None)


def verdict(res):
    """Per-replay criteria (see report header). None = not applicable."""
    out = {}
    if res['code'] == 'new':
        kl = res['choice_kl_total']
        out['c1_kl_in_band'] = CRITERIA['kl_lo'] <= kl <= CRITERIA['kl_hi']
        pt = res['p_target_act']
        out['c2_p_target_rises'] = (None if pt['before']['mean'] is None
                                    else pt['after']['mean'] > pt['before']['mean'])
        dm = res.get('p_target_demo')
        out['c2_demo_p_target_rises'] = (None if not dm or dm['before']['mean'] is None
                                         else dm['after']['mean'] > dm['before']['mean'])
        pn = res['p_none_choice']
        out['c3_p_none_rise_ok'] = pn['after']['mean']-pn['before']['mean'] <= CRITERIA['p_none_rise_max']
        f = res.get('fairness')
        out['c4_fairness'] = None if not f or f.get('status') != 'probed' else bool(
            f['all_raw_lower'] and f['mean_norm_lower'])
    else:
        out['c5_old_ppo_kl_near_zero'] = res['choice_kl_ppo'] <= CRITERIA['old_ppo_kl_max']
    return out


def summarize(results):
    lines = ['| run | arm | u | code | KL ppo | KL total | P(tgt) act b->a (n) | P(tgt) demo b->a (n) | '
             'P(none) choice b->a | adv RMS | ent term | fair | verdict |', '|' + '---|'*13]
    ok = True
    for r in results:
        v = r['verdict']
        bad = [k for k, x in v.items() if x is False]
        ok = ok and not bad

        def ba(d):
            if not d or d['before']['mean'] is None:
                return 'n/a'
            return f"{d['before']['mean']:.3f}->{d['after']['mean']:.3f} ({d['before']['rows']})"
        f = r.get('fairness') or {}
        fair = ('n/a' if f.get('status') != 'probed' else
                f"{'ok' if f['all_raw_lower'] and f['mean_norm_lower'] else 'FAIL'} (harm {f['harm_veh_s']:.0f}, "
                f"{f['acting_rows']} rows, raw d={f['mean_raw_delta']:.3g}, norm d={f['mean_norm_delta']:.3g})")
        pn = r['p_none_choice']
        lines.append(f"| {r['run']} | {r['arm']} | {r['update']} | {r['code']} | {r['choice_kl_ppo']:.4f} | "
                     f"{r['choice_kl_total']:.4f} | {ba(r['p_target_act'])} | {ba(r.get('p_target_demo'))} | "
                     f"{pn['before']['mean']:.3f}->{pn['after']['mean']:.3f} | {r['adv']['rms']:.3g} | "
                     f"{r['adv']['entropy_term']:.3g} | {fair} | {'FAIL: '+','.join(bad) if bad else 'pass'} |")
    return ok, '\n'.join(lines)


# ------------------------------------------------------------------ replay

def load_episodes(root, arm, u):
    out = []
    for j in sorted(glob.glob(os.path.join(root, 'jobs', f't2c_warm_u{u:03}_{arm}_*'))):
        f = guard(os.path.join(j, 'RESULT.json.gz'))
        if not os.path.exists(f):
            continue
        d = json.load(gzip.open(f))
        if d.get('arm') == arm:
            out.append(d)
        gc.collect()
    return out


@torch.no_grad()
def logps(trigger, batches):
    return [torch.log_softmax(trigger.distribution(b.context, b.descriptors, b.mask)[0], -1) for b in batches]


def _probe(runtime, episodes, kw, zero_episode=None):
    """Run train() only up to the temporal targets (no parameter is touched): (harm, advantages)."""
    from tools.coopmwm import fullpol_fair
    seen = {}
    real_pen, real_tt = fullpol_fair.penalties, runtime._temporal_targets

    def pen(*a, **k):
        harm, receipt = real_pen(*a, **k)
        harm = dict(harm)
        if zero_episode is not None:
            harm[zero_episode] = 0.
        seen['harm'] = harm
        return harm, receipt

    def tt(episodes_, targets):
        seen['returns'], seen['adv'] = real_tt(episodes_, targets)
        raise _Captured
    fullpol_fair.penalties, runtime._temporal_targets = pen, tt
    try:
        runtime.train(episodes, **kw)
    except _Captured:
        pass
    finally:
        fullpol_fair.penalties = real_pen
        del runtime._temporal_targets
    return seen.get('harm'), seen['adv']


def fairness_probe(runtime, episodes, kw):
    from tools.coopmwm import learnfix
    from tools.coopmwm.fullpol_learn import regime_key
    from tools.coopmwm.track2_backend import _batch
    by_key = {(e['id'], s['time_s']): s['sample'] for e in episodes for s in e['steps']}
    acting = learnfix.acting_rows(by_key)
    harm, adv_with = _probe(runtime, episodes, kw)
    if harm is None:
        return dict(status='fair reward off (no harm term)')
    cand = sorted(((h, e) for e, h in harm.items()
                   if h > 0 and any(any(v) for k, v in acting.items() if k[0] == e)), reverse=True)
    if not cand:
        return dict(status='no episode with harm > 0 and an acting row', harms=harm)
    h, eid = cand[0]
    _, adv_without = _probe(runtime, episodes, kw, zero_episode=eid)
    cells = {e['id']: regime_key(e) for e in episodes}

    def norm(adv):
        ticks = [dict(batch=_batch(by_key[k]['track2_trigger']), adv=list(adv[k]), cell=cells[k[0]])
                 for k in sorted(adv) if 'track2_trigger' in by_key[k]]
        learnfix.advantages(ticks)
        return {k: t['norm_adv'] for k, t in zip(sorted(k for k in adv if 'track2_trigger' in by_key[k]), ticks)}
    nw, nwo = norm(adv_with), norm(adv_without)
    rows = [(k, i) for k, v in acting.items() if k[0] == eid and k in adv_with for i, a in enumerate(v) if a]
    raw_d = [adv_with[k][i]-adv_without[k][i] for k, i in rows]
    norm_d = [float(nw[k][i]-nwo[k][i]) for k, i in rows]
    others = [adv_with[k][i]-adv_without[k][i] for k in adv_with if k[0] == eid
              for i in range(len(adv_with[k])) if (k, i) not in set(rows)]
    return dict(status='probed', episode_id=eid, harm_veh_s=h, acting_rows=len(rows),
                all_raw_lower=all(d < 0 for d in raw_d), mean_raw_delta=sum(raw_d)/len(raw_d),
                mean_norm_delta=sum(norm_d)/len(norm_d), mean_norm_lower=sum(norm_d) < 0,
                nonacting_rows=len(others), nonacting_mean_raw_delta=(sum(others)/len(others) if others else None),
                episodes_with_harm=sum(1 for x in harm.values() if x > 0))


def run(root, arm, u, code, demo=None, demo_sha=None):
    guard(root)
    if arm not in ('B', 'R'):
        raise SystemExit('arm must be B or R (A is frozen and its fair hook writes)')
    from tools.coopmwm import (launch_production as production, track2_backend as p1, track2_reward as p2,
                               g4_runtime as g4, fullpol_learn as fl, learnfix, demo_bc)
    from tools.coopmwm.fair_vehicle_log import enabled as fair_enabled, update_step as fair_update_step
    from tools.coopmwm.launch_training import read
    from tools.coopmwm import clean4_restart as c4
    cfg = read(Path(root)/'CONFIG.json')
    spec = (cfg.get('newmap') or {}).get('map_spec')
    if spec and not os.environ.get('COOPMWM_MAP_SPEC'):     # the run's own process map (as its workers had)
        os.environ['COOPMWM_MAP_SPEC'] = str(guard(spec))
    budget = cfg.get('clean4_budget') or {}
    if budget.get('run') in c4.APPROVED_BY_RUN:          # replay only: the recorded run's own approved envelope
        c4.APPROVED_BY_RUN[budget['run']] = budget['approved_sim_s']
    native = production.backend_config(cfg)
    native['update'] = u
    native.pop('learnfix', None)
    native.pop('learnfix_demo', None)
    if code == 'new':
        native['learnfix'] = learnfix.value()
        if demo and arm == 'B':
            native['learnfix_demo'] = dict(path=str(guard(demo)), sha256=demo_sha)
    episodes = load_episodes(root, arm, u)
    if not episodes:
        raise SystemExit(f'no {arm} episodes for u{u} in {root}')
    if fair_enabled(native):
        fair_update_step(native, arm, u, episodes)
    hooks = p1.build_arm_runtime(native, arm)
    runtime = hooks['runtime']
    runtime.safety_checked = True
    trig = runtime.policy.trigger
    batches = [p1._batch(s['sample']['track2_trigger']) for e in episodes if not e.get('practice')
               for s in e['steps'] if s['sample'].get('track2_trigger')]
    masks = [b.mask for b in batches]
    label_cfg = dict(native, learnfix=learnfix.value())          # same measured act rows for both codes
    label_cfg.pop('learnfix_demo', None)
    ev_eval, _ = fl.ev_bc_records(label_cfg, arm, episodes) if arm == 'B' else ([], None)
    demo_eval = []
    if demo:
        demo_eval = [r for r in demo_bc.demo_records(demo_bc.load(guard(demo), demo_sha)['items'])
                     if r['meta']['label'] == 'act']
    envelope = p2.reward_update(native, arm, u, episodes)
    synergy = None
    if g4.enabled(native, u):
        synergy = json.loads(Path(g4._envelope_path(native, u)).read_text(encoding='utf-8'))
    kw = dict(rewards=envelope) if synergy is None else dict(rewards=envelope, synergy=synergy)

    fair = fairness_probe(runtime, episodes, kw)
    _, adv_raw = _probe(runtime, episodes, kw)                    # the code's own advantages (pre G4/EV terms)
    lp0 = logps(trig, batches)
    ev0, dm0 = logps(trig, [r['batch'] for r in ev_eval]), logps(trig, [r['batch'] for r in demo_eval])
    learning = hooks['train_update'](episodes, **kw)
    lp1 = logps(trig, batches)
    production.bank_move_supervision(runtime)
    sup = fl.supervise(runtime, episodes)
    lp2 = logps(trig, batches)
    ev2, dm2 = logps(trig, [r['batch'] for r in ev_eval]), logps(trig, [r['batch'] for r in demo_eval])

    fix_receipt = learning.get('learnfix') or {}
    with torch.no_grad():
        h = [-(lp[choice_mask(m)].exp()*lp[choice_mask(m)].nan_to_num(neginf=0.)).sum(-1)
             for lp, m in zip(lp0, masks) if bool(choice_mask(m).any())]
    mean_h = float(torch.cat(h).mean()) if h else 0.
    if code == 'new':
        adv = dict(kind='normalised (learnfix)', rms=fix_receipt.get('adv_rms', float('nan')),
                   entropy_weight=fix_receipt.get('entropy_weight'), mean_entropy=fix_receipt.get('mean_entropy'),
                   raw_rms=(fix_receipt.get('advantages') or {}).get('raw_rms'))
        adv['entropy_term'] = (adv['entropy_weight'] or 0.)*(adv['mean_entropy'] or 0.)
    else:
        by_key = {(e['id'], s['time_s']): s['sample'] for e in episodes for s in e['steps']}
        vals = [a for k, row in adv_raw.items() for a, c in
                zip(row, choice_mask(p1._batch(by_key[k]['track2_trigger']).mask).tolist()) if c]
        ew = float(native['trigger_ppo']['entropy_weight'])
        adv = dict(kind='raw return - value (inherited)', rms=math.sqrt(sum(a*a for a in vals)/len(vals)) if vals
                   else 0., entropy_weight=ew, mean_entropy=mean_h, entropy_term=ew*mean_h)
    res = dict(run=os.path.basename(root.rstrip('/')), arm=arm, update=u, code=code, episodes=len(episodes),
               ticks=len(batches), demo=demo, choice_kl_ppo=choice_kl(lp0, lp1, masks),
               choice_kl_total=choice_kl(lp0, lp2, masks),
               p_none_choice=dict(before=p_none_choice(lp0, masks), after_ppo=p_none_choice(lp1, masks),
                                  after=p_none_choice(lp2, masks)),
               p_target_act=dict(before=p_target(ev_eval, ev0), after=p_target(ev_eval, ev2)),
               p_target_demo=(dict(before=p_target(demo_eval, dm0), after=p_target(demo_eval, dm2))
                              if demo_eval else None),
               adv=adv, fairness=fair,
               ppo_receipt=learning.get('fullpol_ppo_kl'),
               learnfix_receipt={k: v for k, v in fix_receipt.items() if k != 'steps'},
               learnfix_steps=len(fix_receipt.get('steps', ())),
               ev_bc=None if sup is None else dict(fit={k: v for k, v in sup['fit'].items() if k != 'passes'},
                                                   accepted=sum(p['accepted'] for p in sup['fit'].get('passes', ())),
                                                   rows=(sup.get('ev_bc') or {}).get('rows'),
                                                   learnfix=(sup.get('ev_bc') or {}).get('learnfix'),
                                                   demo_bc=sup.get('demo_bc')))
    res['verdict'] = verdict(res)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('check')
    c.add_argument('--root', required=True)
    c.add_argument('--arm', default='B')
    c.add_argument('--update', type=int, required=True)
    c.add_argument('--code', choices=('old', 'new'), required=True)
    c.add_argument('--demo')
    c.add_argument('--demo-sha256')
    c.add_argument('--out', required=True)
    s = sub.add_parser('summarize')
    s.add_argument('files', nargs='+')
    a = ap.parse_args(argv)
    if a.cmd == 'check':
        torch.set_num_threads(int(os.environ.get('LC_THREADS', '2')))
        res = run(a.root, a.arm, a.update, a.code, a.demo, a.demo_sha256)
        Path(a.out).write_text(json.dumps(res, indent=1, default=str), encoding='utf-8')
        print(json.dumps(res['verdict']))
        return 0
    ok, table = summarize([json.loads(Path(f).read_text(encoding='utf-8')) for f in a.files])
    print(f"RESULT: {'PASS' if ok else 'FAIL'}\n\n{table}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
