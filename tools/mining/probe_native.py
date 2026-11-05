"""Cooperation probe -> clean3-consumable native bank (owner ~07:2x walls-only launch).

For an explicit list of probe plans on one context: the unchanged mining-op2 native path
(mine_verify.run_context logic): first-seed sv.verify (policy + candidates), then
verify_perseed.confirm_all8 (seed 2 + per-seed extended later-harm windows + per-admitted
batched solos + optional extra seeds 3..), all charged to the coop_probe ledger. The sealed
confirmations_all8 records are what clean3's MINED_MOVES rows point to (run_root = bank dir).

run <setting> <arm> <ctx_dir> <plans.json> <extra_seeds> <tag>
extract                      -> <ROOT>/minebank/MINED_MOVES(.jsonl|_ANNOTATED.jsonl|_EXAMPLES.jsonl)
"""
import json
import os
from pathlib import Path
import sys
import time

from tools.mining import probe

ROOT = probe.ROOT
BANK = ROOT/'minebank'/'bank'


def run(setting, arm, ctx_dir, plans, extra_seeds, tag, workers=16):
    import torch
    from tools.coopmwm import sumo_verify as sv, replay_bank as rb, verify_perseed as vp
    from tools.coopmwm.storage import write as swrite
    ctx_dir = Path(ctx_dir)
    ctx = json.loads((ctx_dir/'CONTEXT.json').read_text())
    probe.check_world(ctx['world'])
    assert ctx['arm'] == arm
    snap = torch.load(ctx_dir/'SNAPSHOT.pt', weights_only=False)
    assert sv.snapshot_sha(snap) == ctx['snapshot_sha256']
    sid = ctx['state_id']
    bank = Path(rb.bank_root(BANK, arm))
    (bank/'snapshots').mkdir(parents=True, exist_ok=True)
    if not (bank/'snapshots'/(sid+'.pt')).exists():
        torch.save(snap, bank/'snapshots'/(sid+'.pt'))
        swrite(bank/'snapshots'/(sid+'.json'), dict(state_id=sid, arm=arm, job=sid.rsplit('_t', 1)[0],
               world=ctx['world'], t0=int(ctx['t0']), group=dict(members=ctx['group']), regime=None,
               policy_update=0, record_sha256=ctx['snapshot_sha256'], unix=time.time(),
               mining=dict(pressure=ctx.get('pressure'), selection=ctx.get('selection'), wm_ranked=True,
                           probe_setting=setting, variant=ctx.get('variant'))))

    def charge(kind):
        return lambda vid, specs: probe.charge(f'{tag}_{kind}_{vid}', sum(int(s['sim_s']) for s in specs),
                                               setting, len(specs))
    runner = sv.SubprocessBranchRunner(bank/'sumo_verification'/'branches', workers=workers, timeout_s=3600)
    r = sv.verify(snap, plans, runner, charge('first'), 'wm', workers=workers, root=bank/'sumo_verification',
                  tail_s=120)
    (bank/'decisions').mkdir(exist_ok=True)
    (bank/'decisions'/(sid+'.json')).write_text(json.dumps(dict(state_id=sid, verification_id=r['verification_id'],
        status=r['status'], policy_update=0, arm=arm, available_k=len(plans), issued_unix=time.time(),
        matched_B=None)))
    out = dict(state_id=sid, tag=tag, verification_id=r['verification_id'], first_status=r['status'])
    if r['status'] not in ('accepted', 'rejected'):
        out.update(status='first_'+r['status'], reason=r.get('reason'))
        return out
    rb._confirm_charge = lambda run_root, phase, kind: charge(kind)

    def runner_for(work):
        return sv.SubprocessBranchRunner(Path(work)/'branches', workers=workers, timeout_s=3600)
    receipt = dict(r, _branches=str(bank/'sumo_verification'/'branches'), _policy_version='u0')
    conf = vp.confirm_all8(BANK, arm, receipt, snap, runner_for, str(BANK), 'coop_probe', workers=workers,
                           version='u0', candidates=plans, slim_rejected_tapes=False,
                           extra_seeds=int(extra_seeds))
    out.update(status='confirmed', admitted=conf['admitted'], confirmation_sha256=conf['confirmation_sha256'])
    return out


def extract(families=('wall', 'easing', 'ramp_assist')):
    """mine_extract logic for verified acceleration-plan joint moves (no lane effect): a (plan, seed)
    example is mined iff admitted (LB95 vs policy > 0, own-seed safety, later-harm) AND cooperative
    (that seed's synergy LB95 > 0) AND 0 collisions/teleports in the group window, >= 2 members."""
    from tools.coopmwm import replay_bank as rb, verify_perseed as vp
    from tools.coopmwm.protocol import digest
    from tools.coopmwm.exact_grade import verify_sealed
    examples, mined = [], []
    for arm in ('B', 'R'):
        bank = Path(rb.bank_root(BANK, arm))
        for f in sorted((bank/vp.RECORDS).glob('*.json')):
            c = json.loads(f.read_text())
            verify_sealed(c, 'confirmation_sha256')
            vid = c['verification_id']
            plans = {}
            for s in (bank/'sumo_verification'/'branches'/vid).glob('cand*.spec.json'):
                p = json.loads(s.read_text())['plan']
                plans[p['id']] = p
            by_plan = {}
            for e in c['per_seed_v1']['examples']:
                plan = e.get('plan') or plans.get(e['plan_id'])
                fam = plan.get('family') if plan else None
                syn = e.get('synergy') or {}
                win = (e.get('tape') or {}).get('branch', {}).get('window') or {}
                row = dict(arm=arm, verification_id=vid, t0=c['t0'], plan_id=e['plan_id'], family=fam,
                           members=e.get('members') or (plan or {}).get('members'), seed=e['seed_index'],
                           admitted=e['admitted'], cooperative=e.get('cooperative'),
                           status=e.get('cooperative_status'), advantage_veh_s=e.get('advantage_veh_s'),
                           adv_lower95=e.get('lower95_one_sided_veh_s'),
                           synergy_veh_s=syn.get('pooled_total_veh_s'),
                           synergy_lower95=syn.get('lower95_one_sided_veh_s'), best_solo=syn.get('best_solo_member'),
                           safety_ok=e.get('safety_ok'), window_deltas=e.get('window_deltas'),
                           group_collisions=win.get('collisions'), group_teleports=win.get('teleports'),
                           later_harm=e.get('later_harm'))
                examples.append(row)
                ok = (fam in families and plan and len(plan['members']) >= 2 and row['admitted']
                      and row['cooperative'] is True and row['safety_ok'] is not False
                      and not row['group_collisions'] and not row['group_teleports'])
                by_plan.setdefault(e['plan_id'], []).append((ok, row, plan))
            first = json.loads((bank/'sumo_verification'/'verifications'/(vid+'.json')).read_text())
            ep = first.get('episode') or {}
            n_reps = 2+max([0]+[int(e['seed_index'])-2 for e in c['per_seed_v1']['examples']])
            for pid, items in by_plan.items():
                plan = items[0][2]
                for ok, r, _ in items:
                    if not ok:
                        continue
                    s_ = str(r['seed'])
                    mined.append(dict(schema='coopmwm-mined-move-v1', arm=arm, run_root=str(BANK),
                        verification_id=vid, plan_sha256=digest(plan), replay_sources={},
                        seed=r['seed'], situation_id=f"{ep.get('id')}_t{c['t0']}", world=ep.get('world'),
                        t0=c['t0'], family=plan['family'], members=plan['members'], duration_s=plan['duration_s'],
                        profile=plan['profile'], policy_profile=plan['policy_profile'],
                        perturbation_profile=plan['perturbation_profile'],
                        synergy_veh_s={s_: r['synergy_veh_s']}, synergy_lower95={s_: r['synergy_lower95']},
                        advantage_veh_s={s_: r['advantage_veh_s']}, adv_lower95={s_: r['adv_lower95']},
                        best_solo=r['best_solo'], n_reps=n_reps,
                        statistic='per-seed one-sided 95% circular-block (30 s) paired bootstrap B=2000, '
                                  'best solo minus group on the common window (verify_perseed._cooperation)',
                        base='frozen MAPPO u0 (clean2 A_checkpoint_000.pt sha256 5ea22964...)',
                        source='coop_probe 2026-11-06 (owner probe 05:10; walls-only launch 07:2x; B re-verify)'))
    d = ROOT/'minebank'
    (d/'MINED_EXAMPLES.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in examples))
    (d/'MINED_MOVES_ALL_ARMS.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in mined))
    b_rows = [r for r in mined if r['arm'] == 'B']   # MAIN 07:4x: file B rows only; R has its own procedure
    (d/'MINED_MOVES.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in b_rows))
    print(json.dumps(dict(examples=len(examples), mined_rows_all=len(mined), B_rows=len(b_rows))))
    return mined


if __name__ == '__main__':
    if sys.argv[1] == 'run':
        setting, arm, ctx_dir, plans_path, extra, tag = sys.argv[2:8]
        plans = json.loads(Path(plans_path).read_text())
        claim = ROOT/'claims'/('native_'+tag+'.claim')
        claim.parent.mkdir(exist_ok=True)
        with claim.open('x') as f:
            f.write(str(os.getpid()))
        res = run(setting, arm, ctx_dir, plans, int(extra), tag, int(os.environ.get('PROBE_WORKERS', '16')))
        print(json.dumps(res), flush=True)
        with (ROOT/'results'/'native.jsonl').open('a') as f:
            f.write(json.dumps(dict(res, unix=time.time()))+'\n')
    elif sys.argv[1] == 'extract':
        extract()
