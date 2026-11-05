"""Summarise sealed all8 confirmations -> per-example rows + v1 MINED_MOVES pointers.

A plan is emitted only if: G5 family; >= 1 within-seed example admitted AND
cooperative (synergy = best solo J - group J, one-sided 95% block-bootstrap lower
bound > 0, common window); composite safety ok on that seed; the group tape has
0 collisions/teleports in the window; and group_lane.verify_tape passes (actual
issued+completed lane change and executed bound profile).
"""
import json
from pathlib import Path
import sys


def main(root, run_root_out, bank_name='bank'):
    from tools.coopmwm import replay_bank as rb, verify_perseed as vp
    from tools.coopmwm.accel_storage import read
    from tools.coopmwm.group_lane import verify_tape, FAMILIES as G5
    from tools.coopmwm.protocol import digest
    from tools.coopmwm.exact_grade import verify_sealed
    bank_root = Path(root)/bank_name
    tag = '' if bank_name == 'bank' else '_'+bank_name
    examples, mined = [], []
    for arm in ('B', 'R'):
        bank = Path(rb.bank_root(bank_root, arm))
        for f in sorted((bank/vp.RECORDS).glob('*.json')):
            c = json.loads(f.read_text())
            verify_sealed(c, 'confirmation_sha256')
            vid = c['verification_id']
            first = json.loads((bank/'sumo_verification'/'verifications'/(vid+'.json')).read_text())
            plans = {p['id']: p for p in (first.get('chosen_plan') and [first['chosen_plan']] or [])}
            spec_dir = bank/'sumo_verification'/'branches'/vid
            for s in spec_dir.glob('cand*.spec.json'):
                p = json.loads(s.read_text())['plan']
                plans[p['id']] = p
            by_plan = {}
            for e in c['per_seed_v1']['examples']:
                plan = e.get('plan') or plans.get(e['plan_id'])
                fam = plan.get('family') if plan else None
                syn = e.get('synergy') or {}
                win = (e.get('tape') or {}).get('branch', {}).get('window') or {}
                row = dict(arm=arm, verification_id=vid, t0=c['t0'], plan_id=e['plan_id'], family=fam,
                           members=e.get('members'), seed=e['seed_index'], admitted=e['admitted'],
                           cooperative=e.get('cooperative'), status=e.get('cooperative_status'),
                           advantage_veh_s=e.get('advantage_veh_s'),
                           adv_lower95=e.get('lower95_one_sided_veh_s'),
                           synergy_veh_s=syn.get('pooled_total_veh_s'),
                           synergy_lower95=syn.get('lower95_one_sided_veh_s'),
                           best_solo=syn.get('best_solo_member'),
                           safety_ok=e.get('safety_ok'), window_deltas=e.get('window_deltas'),
                           group_collisions=win.get('collisions'), group_teleports=win.get('teleports'),
                           later_harm=e.get('later_harm'), lane_effect=None)
                if fam in G5 and e['admitted'] and e.get('cooperative') is True:
                    try:
                        t = e['tape']
                        tape = read(Path(t['branches_dir'])/t['verification_id']/t['branch']['branch']/'RESULT.json')
                        if digest(tape) != t['branch']['result_sha256']:
                            raise ValueError('tape hash mismatch')
                        verify_tape(plan, tape, c['t0'])
                        row['lane_effect'] = True
                    except Exception as err:
                        row['lane_effect'] = 'FAIL: '+repr(err)[:200]
                examples.append(row)
                ok = (fam in G5 and row['admitted'] and row['cooperative'] is True and row['safety_ok'] is not False
                      and row['lane_effect'] is True and not row['group_collisions'] and not row['group_teleports'])
                by_plan.setdefault(e['plan_id'], []).append((ok, row, plan))
            for pid, items in by_plan.items():
                good = [r for ok, r, _ in items if ok]
                if good:
                    plan = items[0][2]
                    mined.append(dict(schema='coopmwm-mined-move-v1', arm=arm, run_root=run_root_out,
                                      verification_id=vid, plan_sha256=digest(plan), replay_sources={},
                                      _family=plan['family'], _seeds=sorted(r['seed'] for r in good),
                                      _synergy_veh_s={str(r['seed']): r['synergy_veh_s'] for r in good},
                                      _synergy_lower95={str(r['seed']): r['synergy_lower95'] for r in good}))
    (Path(root)/f'EXAMPLES{tag}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in examples))
    (Path(root)/f'MINED_MOVES_ANNOTATED{tag}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in mined))
    (Path(root)/f'MINED_MOVES{tag}.jsonl').write_text(''.join(
        json.dumps({k: v for k, v in r.items() if not k.startswith('_')})+'\n' for r in mined))
    print(json.dumps(dict(examples=len(examples), mined=len(mined))))


if __name__ == '__main__':
    main(*sys.argv[1:])
