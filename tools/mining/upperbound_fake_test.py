"""No-physics orchestration test for upperbound.py (fake branch runner + fake bootstrap; temp root).

usage: UB_ROOT=/tmp/ubtest python -m tools.mining.upperbound_fake_test <probe ctx dir>
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys


def main(src):
    root = Path(os.environ['UB_ROOT'])
    assert not root.exists() and str(root).startswith('/tmp/')
    from tools.mining import upperbound as ub
    from tools.coopmwm import sumo_verify as sv
    d = root/'collect'/'fake'/'ctx'
    d.mkdir(parents=True)
    shutil.copy(Path(src)/'SNAPSHOT.pt', d/'SNAPSHOT.pt')
    ctx = json.loads((Path(src)/'CONTEXT.json').read_text())
    prof = ctx['policy_profile']
    el = [r for r in ctx['rows'] if r['is_cav'] and r['lane'] in (0, 1) and ub.S_LO <= r['s_m'] <= ub.S_HI
          and prof.get(r['vid']) and prof[r['vid']][0] is not None]
    ctx['eligible'] = [r['vid'] for r in sorted(el, key=lambda r: (-r['s_m'], r['vid']))]
    ctx['world'] = dict(ctx['world'])
    (d/'CONTEXT.json').write_text(json.dumps(ctx))
    (root/'REGISTRATION.json').write_text('{}')
    ub._write(root/'results'/'SITUATIONS.json', dict(collections={}, situations={'1': str(d)}))

    def fake_run_branch(spec, snapshot, runner):
        plan = spec['plan']
        h = int(hashlib.sha256((plan or {}).get('id', 'policy').encode()).hexdigest()[:6], 16)
        J = 10000. if plan is None else 10000.-((h % 600)-250)
        return dict(branch=spec['branch'], index=spec['index'], status='graded', J_veh_s=J,
                    window=dict(collisions=0., teleports=0., hard_brake=1., ttc_below_3_vs=2., team_delay_vs=J),
                    branch_sha256=spec['branch_sha256'], result_sha256='x', actual_sim_s=spec['sim_s'])
    sv.run_branch = fake_run_branch
    ub.stats = lambda a, b, w: dict(total=a['J']-b['J'], lb95=a['J']-b['J']-40., unit='fake')
    sit = ub.Situation(d)
    print('eligible', ctx['eligible'], 'sets', ub.member_sets(ctx))
    print('screen', [p['id'] for p in ub.screen_slate(ctx)])
    ub.evaluate_phases([sit])
    s = ub.summary()
    print(json.dumps(s['aggregate'], indent=1))
    print(json.dumps({k: v for k, v in s['situations'][0].items() if k != 'searched_adv'}, indent=1, default=str))
    print('ledger', ub.spent(), sum(1 for _ in open(root/'CHARGES.jsonl')))
    print('searched', len(s['situations'][0]['searched_adv']))


if __name__ == '__main__':
    main(sys.argv[1])
