"""c4/strongguide offline check (CPU, zero SUMO, read-only on the run root; see clean5c/strongguide/CRITERIA.json).

  python ops/strong_check.py <run_root> <arm> <update> current|strong
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


def main(root, arm, u, mode):
    from tools.coopmwm import (clean4_restart as c4, launch_production as production, track2_backend as p1,
                               track2_reward as p2, g4_runtime as g4, fullpol_learn as fl, learnfix)
    from tools.coopmwm.bank_growth import update_view
    from tools.coopmwm.fair_vehicle_log import enabled as fair_enabled, update_step as fair_update_step
    from tools.coopmwm.launch_training import read
    cfg = read(Path(root)/'CONFIG.json')
    b = cfg.get('clean4_budget') or {}
    if b.get('run') in c4.APPROVED_BY_RUN:
        c4.APPROVED_BY_RUN[b['run']] = b['approved_sim_s']
    native = production.backend_config(cfg)
    native['update'] = u
    native, growth = update_view(native, arm, u)                 # the run's own frozen U<u> snapshot (read only)
    if mode == 'strong':
        native['learnfix2'] = learnfix.apply_strong(native['learnfix2'], learnfix.STRONG)
    episodes = []
    for j in sorted(glob.glob(os.path.join(root, 'jobs', f't2c_warm_u{u:03}_{arm}_*'))):
        f = os.path.join(j, 'RESULT.json.gz')
        if os.path.exists(f):
            d = json.load(gzip.open(f))
            if d.get('arm') == arm:
                episodes.append(d)
    if fair_enabled(native):
        fair_update_step(native, arm, u, episodes)
    hooks = p1.build_arm_runtime(native, arm)
    runtime = hooks['runtime']
    runtime.safety_checked = True
    trig = runtime.policy.trigger
    recorded = {(e['id'], s['time_s']): s['sample']['track2_trigger'] for e in episodes for s in e['steps']
                if s['sample'].get('track2_trigger') and not e.get('practice')}
    keys = sorted(recorded)
    ticks = [p1._batch(recorded[k]) for k in keys]
    ev, _ = fl.ev_bc_records(native, arm, episodes) if arm == 'B' else ([], None)
    verified = {t['family'] for t in native['clean5c']['templates']} if native.get('clean5c') else set()

    def free_slot(entry):
        if entry is None or entry.get('family') not in ('free_program_initiate', 'free_program_join'):
            return False
        return not set(fl.option_labels(entry)) & verified

    def mask_of(trigger):
        return torch.tensor([[j > 0 and free_slot(e) for j, e in enumerate(row)] for row in trigger['tables']])
    masks = [mask_of(recorded[k]) for k in keys]

    @torch.no_grad()
    def measure():
        p_none = n = 0.
        free_exp = free_exp_masked = 0.
        for t, m in zip(ticks, masks):
            lg = trig.distribution(t.context, t.descriptors, t.mask)[0]
            c = t.mask[:, 1:].any(1)
            p = lg.softmax(-1)
            p_none += float(p[c, 0].sum())
            n += int(c.sum())
            free_exp += float((p*m).sum())
            pm = lg.masked_fill(m, -torch.inf).softmax(-1)
            free_exp_masked += float((pm*m).sum())
        pt, ptm = [], []
        for r in ev:
            bb = r['batch']
            lg = trig.distribution(bb.context, bb.descriptors, bb.mask)[0]
            m = mask_of(recorded[(r['meta']['episode_id'], r['meta']['time_s'])])
            for k in range(len(bb.actions)):
                if float(r['row_weight'][k]) > 0 and int(bb.actions[k]) > 0:
                    pt.append(float((lg[k].softmax(-1)*r['equivalent'][k].float()).sum()))
                    ptm.append(float((lg[k].masked_fill(m[k], -torch.inf).softmax(-1)*r['equivalent'][k].float()).sum()))
        return dict(p_none_choice=p_none/max(n, 1), free_expected=free_exp, free_expected_verified_only=free_exp_masked,
                    p_target=(sum(pt)/len(pt) if pt else None), p_target_verified_only=(sum(ptm)/len(ptm) if ptm else None),
                    target_rows=len(pt))
    start = [trig.distribution(t.context, t.descriptors, t.mask)[0].detach() for t in ticks]
    before = measure()
    envelope = p2.reward_update(native, arm, u, episodes)
    synergy = None
    if g4.enabled(native, u):
        path = g4._envelope_path(native, u)
        if Path(path).exists():
            synergy = json.loads(Path(path).read_text(encoding='utf-8'))
    hooks['train_update'](episodes, rewards=envelope, **({} if synergy is None else dict(synergy=synergy)))
    production.bank_move_supervision(runtime)
    sup = fl.supervise(runtime, episodes)
    after = measure()
    kl = learnfix.choice_kl(trig, [dict(batch=t) for t in ticks], start)
    res = dict(run=os.path.basename(root), arm=arm, update=u, mode=mode, growth=growth, before=before, after=after,
               total_choice_kl=kl, ppo_choice_kl=(getattr(runtime, 'learnfix2_update_start', None) or {}).get('ppo_kl'),
               bc=None if sup is None else sup.get('report_bc_kl'), bc_passes=None if sup is None else len(sup['fit']['passes']),
               demo=None if sup is None else sup.get('demo_bc'), learnfix2=native['learnfix2'])
    print(json.dumps(res, default=str))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4])
