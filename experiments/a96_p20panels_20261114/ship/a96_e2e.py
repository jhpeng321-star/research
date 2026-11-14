"""A96 zero-SUMO end-to-end tests on EXISTING A91 records (copied to /dev/shm; nothing written under /root/a91 or any
A96 root). Stand-ins only; prints structure and error COUNTS, never an outcome number.

usage (inside an A96 tree, env sourced):
  python a96_e2e.py analyse --a91-root /root/a91/root --final-root <drytest root with the A96 panel> --shm /dev/shm/a96..
  python a96_e2e.py mech --a91-root /root/a91/root --shm /dev/shm/a96..
analyse: A96 analyse_core + summary on the A91 confirm layout (searched <- BANK_TURNS (two-plan A75_BANK), hand <-
  ROLL_BOTH (one plan), DUO_HOLD <- DUO_HOLD_60C, MAPPO20C <- MAPPO60C) -> ANALYSE_E2E_OK
mech: A96 rule_rows (per-plan a80_s2b.plan_checks, zone on the sealed vehicle logs, overlap / engaged) on the A91
  mechanics records A91I_BANK_TURNS / A91I_ROLL_BOTH (expected executor = their MAPPO60C) -> MECH_E2E_OK
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

import a96_p20 as M  # noqa: E402

MAPPO60C_SHA256 = '2abed31707b61121569af2936c23689827b52b28422df162751b230aa6a93de8'
L2, TAPER, ROLL = 'a89l2_d86b1d6ac653e8ad', 'mut_1fca373046a3c66e', 'd2_d86b1d6ac653e8ad'
FAMS = {M.SEARCHED: (L2, TAPER), M.HAND: (ROLL,)}
STAND_IN_C = {M.SEARCHED: 'BANK_TURNS', M.HAND: 'ROLL_BOTH', 'DUO_HOLD': 'DUO_HOLD_60C', 'MAPPO20C': 'MAPPO60C'}
BANKS = {M.SEARCHED: 'A91_TURNS_D2800_p60.json', M.HAND: 'A89_ARMS_D2800_p60.json'}


def copy_wave(N, src, shm, name, log_name):
    shutil.copyfile(src/name, shm/name)
    v = N.verify(N.read(shm/name), 'receipt_sha256', name)
    for r in v['records']:
        d = r.get('work_dir', r['job_id'])
        (shm/'evaluation'/d).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src/'evaluation'/d/log_name, shm/'evaluation'/d/log_name)
    return v


def analyse(a, N, src, shm):
    D2 = M.D2
    panel91 = json.loads((src/'A91_PANEL_C.json').read_text(encoding='utf-8'))
    worlds = [r['world_id'] for r in panel91['worlds']]
    have, reports = {}, {}
    for arm in M.ARMS:
        v = copy_wave(N, src, shm, f'WAVE_A91C_{STAND_IN_C[arm]}__0000-0095.json', 'vehicles_report.json')
        have[arm] = {r['world_id']: r for r in v['records']}
        reports[arm] = D2._own_reports(shm, have[arm], worlds)
    D, panel = M.setup(Path(a.final_root))
    if len(panel['worlds']) != 192:
        raise SystemExit('A96 panel missing in --final-root')
    v = M.analyse_core(D, have, reports, worlds, dict(panel91['contract']['bootstrap']),
                       fams={b: FAMS[b] for b in M.BANK_ARMS}, meta=dict(test='a91-layout stand-ins'))
    out = shm/'A96_RESULTS_E2E_TEST.json'
    N.publish(out, v)
    back = N.verify(N.read(out), 'analysis_sha256', out.name)
    line = json.dumps(M.summary(back), default=str)
    assert json.loads(line)['primary']['verdict'] in ('SUPPORTED', 'NOT SUPPORTED')
    b = back['arms'][M.SEARCHED]
    print(json.dumps(dict(part=M.PART, hand_run=back['hand_run'], primary=back['primary']['contrast'],
                          direction_present='direction' in back['primary'], arms=sorted(back['arms']),
                          contrasts=[(r['treatment'], r['reference'], r['kind']) for r in back['contrasts']],
                          bank_keys=sorted(b), overlap_keys=sorted(b['overlap']), engaged=sorted(b['cavs_engaged_at_once']),
                          summary_keys=sorted(json.loads(line)), analysis_sha256=back['analysis_sha256'])))
    print('ANALYSE_E2E_OK')


def mech(a, N, src, shm):
    from tools.coopmwm import a75_holdforms as H
    fails, out = [], {}
    for arm, old in ((M.SEARCHED, 'BANK_TURNS'), (M.HAND, 'ROLL_BOTH')):
        v = copy_wave(N, src, shm, f'WAVE_A91I_{old}__0000-0001.json', N.VEHICLE_LOG)
        have = {r['world_id']: r for r in v['records']}
        doc = json.loads((src/'a91_banks'/BANKS[arm]).read_text(encoding='utf-8'))
        plans = {q['family']: q for q in H.a75_bank_plans(doc, 'D2800_p60')[0]}
        zones = {f: H.zone_bounds(q['spec']) for f, q in plans.items()}
        out[arm] = M.rule_rows(shm, have, arm, FAMS[arm], plans, zones, sorted(have), fails, exe=MAPPO60C_SHA256)
    print(json.dumps(dict(arms=sorted(out), n_errors={k: sum(len(x['errors']) for x in o['rows']) for k, o in out.items()},
                          n_failures=len(fails), row_keys=sorted(out[M.SEARCHED]['rows'][0]),
                          activated_keys={k: sorted(o['activated']) for k, o in out.items()})))
    if fails:
        print('failures:', fails)
        raise SystemExit('MECH_E2E: rule checks failed on the A91 mechanics records')
    print('MECH_E2E_OK')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('what', choices=('analyse', 'mech'))
    p.add_argument('--a91-root', required=True)
    p.add_argument('--final-root')
    p.add_argument('--shm', required=True)
    a = p.parse_args()
    shm = Path(a.shm)
    if not str(shm).startswith('/dev/shm/') or shm.exists():
        raise SystemExit('--shm must be a new directory under /dev/shm')
    N = M._N()
    shm.mkdir(parents=True)
    try:
        (analyse if a.what == 'analyse' else mech)(a, N, Path(a.a91_root), shm)
    finally:
        shutil.rmtree(shm, ignore_errors=True)


if __name__ == '__main__':
    main()
