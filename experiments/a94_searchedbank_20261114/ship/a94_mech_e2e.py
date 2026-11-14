"""A94 zero-SUMO test of the mechanics RULE CHECKS (a94_bank._rule_rows) on EXISTING A91 mechanics records (copied to
/dev/shm; nothing is written under /root/a91 or /root/a94/root).

usage (inside the A94 tree, env sourced): python a94_mech_e2e.py --a91-root /root/a91/root --shm /dev/shm/a94_me_<pid>
Stand-ins: BANK_SEARCHED <- A91I_BANK_TURNS (A75_BANK, two plans; specs / zones from the A91 turns bank), BANK_HAND
<- A91I_ROLL_BOTH (A75_PLAN, one plan; specs / zones from the A89 arms bank). Exercises a80_s2b.plan_checks per family, the per-family zone check on the
sealed vehicle logs, overlap / engaged counters. Prints only structure + error COUNTS, then MECH_E2E_OK.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

import a81_bigconfirm as B  # noqa: E402
import a94_bank as M  # noqa: E402

STAND_IN = dict(BANK_SEARCHED='BANK_TURNS', BANK_HAND='ROLL_BOTH')
FAMS = dict(BANK_SEARCHED=('a89l2_d86b1d6ac653e8ad', 'mut_1fca373046a3c66e'), BANK_HAND=('d2_d86b1d6ac653e8ad',))
BANKS = dict(BANK_SEARCHED='A91_TURNS_D2800_p60.json', BANK_HAND='A89_ARMS_D2800_p60.json')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--a91-root', required=True)
    p.add_argument('--shm', required=True)
    a = p.parse_args()
    shm = Path(a.shm)
    if not str(shm).startswith('/dev/shm/') or shm.exists():
        raise SystemExit('--shm must be a new directory under /dev/shm')
    M.point()
    N = B._mods()[0]
    from tools.coopmwm import a75_holdforms as H
    src = Path(a.a91_root)
    shm.mkdir(parents=True)
    try:
        recs = {}
        for arm, old in STAND_IN.items():
            name = f'WAVE_A91I_{old}__0000-0001.json'
            shutil.copyfile(src/name, shm/name)
            v = N.verify(N.read(shm/name), 'receipt_sha256', name)
            recs[f'A94I_{arm}'] = {r['world_id']: r for r in v['records']}
            for r in v['records']:
                d = r.get('work_dir', r['job_id'])
                (shm/'evaluation'/d).mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src/'evaluation'/d/N.VEHICLE_LOG, shm/'evaluation'/d/N.VEHICLE_LOG)
        B._load = lambda root, wave, specs: (recs[wave], None)          # test-only: records without specs
        worlds = sorted(next(iter(recs.values())))
        fails, out = [], {}
        for arm in STAND_IN:
            doc = json.loads((src/'a91_banks'/BANKS[arm]).read_text(encoding='utf-8'))
            plans = {q['family']: q for q in H.a75_bank_plans(doc, M.CELL)[0]}
            zones = {f: H.zone_bounds(q['spec']) for f, q in plans.items()}
            out[arm] = M._rule_rows(None, shm, {f'A94I_{arm}': None}, arm, FAMS[arm], plans, zones, worlds, fails)
        print(json.dumps(dict(worlds=worlds, arms=sorted(out), row_keys=sorted(out['BANK_SEARCHED']['rows'][0]),
                              n_errors={arm: sum(len(x['errors']) for x in o['rows']) for arm, o in out.items()},
                              n_failures=len(fails), activated_keys={arm: sorted(o['activated']) for arm, o in out.items()},
                              per_family_keys=sorted(out['BANK_SEARCHED']['rows'][0]['per_family']),
                              overlap_keys=sorted(out['BANK_SEARCHED']['rows'][0]['overlap'])), indent=1))
        if fails:
            print('failures:', fails)
            raise SystemExit('MECH_E2E: rule checks failed on the A91 mechanics records')
        print('MECH_E2E_OK')
    finally:
        shutil.rmtree(shm, ignore_errors=True)


if __name__ == '__main__':
    main()
