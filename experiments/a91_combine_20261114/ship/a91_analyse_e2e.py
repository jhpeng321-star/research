"""A91 zero-SUMO end-to-end test of the analysis + console summary on the EXISTING A89 confirm layout (copied to
/dev/shm; nothing is written under /root/a89 or /root/a91/root).

usage (inside the A91 tree, env sourced, after the A91 pin stage):
  python a91_analyse_e2e.py --a89-root /root/a89/root --final-root /root/a91/root --shm /dev/shm/a91_e2e_<pid>
Arm stand-ins (A91 arm <- A89 confirm wave): L2_ONLY <- ROLL_L2_ONLY, BANK_TURNS <- TAPER (an A75_PLAN record; it
exercises the per-family / overlap code with one family), ROLL_BOTH <- ROLL_BOTH, DUO_HOLD_60C, MAPPO60C <- same.
Prints ONLY structure (keys, verdict labels present, json round trip, sha) - never a number - then ANALYSE_E2E_OK.
The A89 outcomes are not inspected or used for any choice.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]

import a81_bigconfirm as B  # noqa: E402
import a91_combine as M  # noqa: E402

STAND_IN = dict(L2_ONLY='ROLL_L2_ONLY', BANK_TURNS='TAPER', ROLL_BOTH='ROLL_BOTH', DUO_HOLD_60C='DUO_HOLD_60C',
                MAPPO60C='MAPPO60C')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--a89-root', required=True)
    p.add_argument('--final-root', required=True)
    p.add_argument('--shm', required=True)
    a = p.parse_args()
    shm = Path(a.shm)
    if not str(shm).startswith('/dev/shm/') or shm.exists():
        raise SystemExit('--shm must be a new directory under /dev/shm')
    M.point()
    N = B._mods()[0]
    D2 = B._D2()
    src = Path(a.a89_root)
    panel89 = json.loads((src/'A89_PANEL_C.json').read_text(encoding='utf-8'))
    worlds = [r['world_id'] for r in panel89['worlds']]
    bootstrap = dict(panel89['contract']['bootstrap'])
    shm.mkdir(parents=True)
    have, reports, sources = {}, {}, {}
    try:
        for arm, old in STAND_IN.items():
            name = f'WAVE_A89C_{old}__0000-0095.json'
            shutil.copyfile(src/name, shm/name)
            v = N.verify(N.read(shm/name), 'receipt_sha256', name)
            have[arm] = {r['world_id']: r for r in v['records']}
            for r in v['records']:
                d = r.get('work_dir', r['job_id'])
                (shm/'evaluation'/d).mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src/'evaluation'/d/'vehicles_report.json', shm/'evaluation'/d/'vehicles_report.json')
            reports[arm] = D2._own_reports(shm, have[arm], worlds)
            sources[arm] = v['receipt_sha256']
        D, panel = B.setup('c', Path(a.final_root))           # the A91 core / setting (read-only)
        v = M.analyse_core(D, have, reports, worlds, bootstrap, sources=sources, panel_sha256=panel89['panel_sha256'],
                           root=shm, seeds=['a89-layout-test'])
        out = shm/'A91_RESULTS_E2E_TEST.json'
        N.publish(out, v)
        back = N.verify(N.read(out), 'analysis_sha256', out.name)
        line = json.dumps(M.summary(back), default=str)            # the exact console-summary path of main()
        assert json.loads(line)['primary']['verdict'] in ('SUPPORTED', 'NOT SUPPORTED')
        print(json.dumps(dict(keys=sorted(back), primary_keys=sorted(back['primary']),
                              contrasts=[(r['treatment'], r['reference'], r['kind'], sorted(r)[:6]) for r in back['contrasts']],
                              arms=sorted(back['arms']), bank_keys=sorted(back['arms']['BANK_TURNS']),
                              overlap_keys=sorted(back['arms']['BANK_TURNS']['overlap']),
                              summary_keys=sorted(json.loads(line)), summary_len=len(line),
                              analysis_sha256=back['analysis_sha256']), indent=1))
        print('ANALYSE_E2E_OK')
    finally:
        shutil.rmtree(shm, ignore_errors=True)


if __name__ == '__main__':
    main()
