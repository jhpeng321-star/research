"""A94 zero-SUMO end-to-end test of the analysis + console summary on the EXISTING A91 confirm layout (copied to
/dev/shm; nothing is written under /root/a91 or /root/a94/root).

usage (inside the A94 tree, env sourced, after a root holding the A94 panel exists - the drytest root):
  python a94_analyse_e2e.py --a91-root /root/a91/root --final-root <root> --shm /dev/shm/a94_e2e_<pid>
Arm stand-ins (A94 arm <- A91 confirm wave): BANK_SEARCHED <- BANK_TURNS (a two-plan A75_BANK record: exercises the
per-plan / overlap / engaged code), BANK_HAND <- ROLL_BOTH (only when A94_HAND_SHA256 is a sha; with "none" the 3-arm
fallback layout is tested), DUO_HOLD_60C, MAPPO60C <- same.
Prints ONLY structure (keys, verdict labels present, json round trip, sha) - never a number - then ANALYSE_E2E_OK.
The A91 outcomes are not inspected or used for any choice.
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

STAND_IN = dict(BANK_SEARCHED='BANK_TURNS', BANK_HAND='ROLL_BOTH', DUO_HOLD_60C='DUO_HOLD_60C', MAPPO60C='MAPPO60C')
A91_FAMS = dict(BANK_SEARCHED=('a89l2_d86b1d6ac653e8ad', 'mut_1fca373046a3c66e'), BANK_HAND=('d2_d86b1d6ac653e8ad',))


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
    bootstrap = dict(panel91['contract']['bootstrap'])
    shm.mkdir(parents=True)
    have, reports, sources = {}, {}, {}
    try:
        for arm, old in ((a, STAND_IN[a]) for a in M.ARMS):
            name = f'WAVE_A91C_{old}__0000-0095.json'
            shutil.copyfile(src/name, shm/name)
            v = N.verify(N.read(shm/name), 'receipt_sha256', name)
            have[arm] = {r['world_id']: r for r in v['records']}
            for r in v['records']:
                d = r.get('work_dir', r['job_id'])
                (shm/'evaluation'/d).mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src/'evaluation'/d/'vehicles_report.json', shm/'evaluation'/d/'vehicles_report.json')
            reports[arm] = D2._own_reports(shm, have[arm], worlds)
            sources[arm] = v['receipt_sha256']
        D, panel = B.setup('c', Path(a.final_root))           # the A94 core / setting (read-only)
        if panel is None or len(panel['worlds']) != M.CONFIRM_N:
            raise SystemExit('A94 panel missing in --final-root')
        v = M.analyse_core(D, have, reports, worlds, bootstrap, sources=sources, panel_sha256=panel91['panel_sha256'],
                           root=shm, seeds=['a91-layout-test'], fams={a: A91_FAMS[a] for a in M.BANK_ARMS},
                           bank_meta=dict(test=True))
        out = shm/'A94_RESULTS_E2E_TEST.json'
        N.publish(out, v)
        back = N.verify(N.read(out), 'analysis_sha256', out.name)
        line = json.dumps(M.summary(back), default=str)            # the exact console-summary path of main()
        assert json.loads(line)['primary']['verdict'] in ('SUPPORTED', 'NOT SUPPORTED')
        bs = back['arms']['BANK_SEARCHED']
        print(json.dumps(dict(keys=sorted(back), primary_keys=sorted(back['primary']),
                              contrasts=[(r['treatment'], r['reference'], r['kind'], sorted(r)[:6]) for r in back['contrasts']],
                              hand_run=back['hand_run'], primary=back['primary']['contrast'],
                              direction_present='direction' in back['primary'],
                              arms=sorted(back['arms']), bank_keys=sorted(bs),
                              overlap_keys=sorted(bs['overlap']), engaged_keys=sorted(bs['cavs_engaged_at_once']),
                              offers_by_plan_families=sorted(bs['offers_by_plan']),
                              summary_keys=sorted(json.loads(line)), summary_len=len(line),
                              analysis_sha256=back['analysis_sha256']), indent=1))
        print('ANALYSE_E2E_OK')
    finally:
        shutil.rmtree(shm, ignore_errors=True)


if __name__ == '__main__':
    main()
