"""A98 zero-SUMO end-to-end test (stand-ins only; prints structure, never an outcome number).
usage (inside the A98 tree, env sourced): python a98_e2e.py --a91-root /root/a91/root --final-root <drytest root> --shm /dev/shm/a98_..
1. builds a FAKE "A94 root" in /dev/shm from the A91 confirm records (job ids relabelled a91c_ -> a94c_, receipts
   re-sealed, cached reports copied, chain_confirm.out marker, panel pin) and runs a98_addon.copy_a94 on it;
2. runs analyse_core + summary on: BANK_S4 <- BANK_TURNS, BANK_S4S5 <- L2_ONLY (a98 side) and the copied fake A94
   arms BANK_SEARCHED <- ROLL_BOTH, BANK_HAND <- ROLL_BOTH, DUO_HOLD_60C, MAPPO60C;
3. runs rule_rows on the A91 mechanics records A91I_BANK_TURNS / A91I_L2_ONLY.  -> A98_E2E_OK
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

import a81_bigconfirm as B  # noqa: E402
import a98_addon as M  # noqa: E402

FAKE94 = dict(BANK_SEARCHED='ROLL_BOTH', BANK_HAND='ROLL_BOTH', DUO_HOLD_60C='DUO_HOLD_60C', MAPPO60C='MAPPO60C')
OWN = dict(BANK_S4='BANK_TURNS', BANK_S4S5='L2_ONLY')
FAMS = dict(BANK_S4=('a89l2_d86b1d6ac653e8ad', 'mut_1fca373046a3c66e'), BANK_S4S5=('a89l2_d86b1d6ac653e8ad',))
BANKS = dict(BANK_S4='A91_TURNS_D2800_p60.json', BANK_S4S5='A89_ARMS_D2800_p60.json')
MAPPO60C = '2abed31707b61121569af2936c23689827b52b28422df162751b230aa6a93de8'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--a91-root', required=True)
    p.add_argument('--final-root', required=True)
    p.add_argument('--shm', required=True)
    a = p.parse_args()
    shm = Path(a.shm)
    if not str(shm).startswith('/dev/shm/') or shm.exists():
        raise SystemExit('--shm must be a new directory under /dev/shm')
    M.point()
    N = B._mods()[0]
    D2 = B._D2()
    src = Path(a.a91_root)
    panel91 = json.loads((src/'A91_PANEL_C.json').read_text(encoding='utf-8'))
    worlds = [r['world_id'] for r in panel91['worlds']]
    shm.mkdir(parents=True)
    try:
        fake = shm/'a94'/'root'
        fake.mkdir(parents=True)
        (fake.parent/'chain_confirm.out').write_text('x A94_CONFIRM_RUN_DONE\n')
        (fake/'A94_PANEL_C.sha256').write_text(f'0  A94_PANEL_C.json panel_sha256 {panel91["panel_sha256"]}\n')
        for arm, old in FAKE94.items():
            v = N.verify(N.read(src/f'WAVE_A91C_{old}__0000-0095.json'), 'receipt_sha256', old)
            recs = []
            for r in v['records']:
                d = r.get('work_dir', r['job_id'])
                r2 = dict(r, job_id=r['job_id'].replace('a91c_', 'a94c_', 1))
                d2 = r2.get('work_dir', r2['job_id'])
                (fake/'evaluation'/d2).mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src/'evaluation'/d/'vehicles_report.json', fake/'evaluation'/d2/'vehicles_report.json')
                recs.append(r2)
            body = {k: x for k, x in v.items() if k != 'receipt_sha256'}
            N.publish(fake/f'WAVE_A94C_{arm}__0000-0095.json', N.seal(dict(body, records=recs), 'receipt_sha256'))
        h94, s94 = M.copy_a94(fake, shm/'ref94', worlds, panel91['panel_sha256'])
        have, reports = {}, {}
        for arm, old in OWN.items():
            v = N.verify(N.read(src/f'WAVE_A91C_{old}__0000-0095.json'), 'receipt_sha256', old)
            have[arm] = {r['world_id']: r for r in v['records']}
            reports[arm] = D2._own_reports(src, have[arm], worlds)          # cached reports, read-only
        for arm, recs in h94.items():
            have[arm] = recs
            reports[arm] = D2._own_reports(shm/'ref94', recs, worlds)
        D, panel = B.setup('c', Path(a.final_root))
        v = M.analyse_core(D, have, reports, worlds, dict(panel91['contract']['bootstrap']), fams=FAMS,
                           meta=dict(test='a91-layout stand-ins'))
        out = shm/'A98_RESULTS_E2E_TEST.json'
        N.publish(out, v)
        back = N.verify(N.read(out), 'analysis_sha256', out.name)
        line = json.dumps(M.summary(back), default=str)
        from tools.coopmwm import a75_holdforms as H
        fails, rr = [], {}
        for arm, old in OWN.items():
            w = N.verify(N.read(src/f'WAVE_A91I_{old}__0000-0001.json'), 'receipt_sha256', old)
            hv = {r['world_id']: r for r in w['records']}
            doc = json.loads((src/'a91_banks'/BANKS[arm]).read_text(encoding='utf-8'))
            plans = {q['family']: q for q in H.a75_bank_plans(doc, 'D2800_p60')[0]}
            zones = {f: H.zone_bounds(q['spec']) for f, q in plans.items()}
            rr[arm] = M.rule_rows(src, hv, arm, FAMS[arm], plans, zones, sorted(hv), fails, MAPPO60C)
        print(json.dumps(dict(copied_arms=sorted(h94), n_receipts={k: len(x) for k, x in s94.items()},
                              contrasts=[(r['treatment'], r['reference'], r['kind']) for r in back['contrasts']],
                              arms=sorted(back['arms']), bank_keys=sorted(back['arms']['BANK_S4']),
                              summary_keys=sorted(json.loads(line)),
                              mech_errors={k: sum(len(x['errors']) for x in o['rows']) for k, o in rr.items()},
                              mech_failures=len(fails), analysis_sha256=back['analysis_sha256'])))
        if fails:
            raise SystemExit(f'rule checks failed {fails}')
        print('A98_E2E_OK')
    finally:
        shutil.rmtree(shm, ignore_errors=True)


if __name__ == '__main__':
    main()
