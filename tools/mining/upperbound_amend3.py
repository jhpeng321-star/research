"""Upper-bound amendment A3 (registered in AMENDMENT_A3.json before execution; same ledger, hard cap).

Later-harm check of the chosen move on the search seed: policy + move re-run with an extended window
[t0, t0+660] (tail >= 600 s, as the native confirm_all8 later-harm windows), for the acting situations
in the registered order while the pair fits under the 150,000 cap. Also re-checks determinism: the
[t0, t0+180] increments of the extended tapes must equal the original J values.
"""
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

from tools.mining import upperbound as ub
from tools.mining import upperbound_amend as am

EXT = 660


def one(sid_dir, move):
    from tools.coopmwm import sumo_verify as sv
    from tools.coopmwm.branch_eval import totals_at
    from tools.coopmwm.exact_grade import _write_new
    sit = ub.Situation(sid_dir)
    am.load(sit)
    plan = sit.plans[move]
    specs = sv.build_branches(sit.snap, [plan], 'wm', tail_s=EXT-int(plan['duration_s']))
    assert all(s['window']['end_s'] == sit.t0+EXT for s in specs)
    cost = sum(int(s['sim_s']) for s in specs)
    if ub.spent()+cost > ub.CAP:
        return dict(sid=sit.sid, amendment='A3', status='skipped_budget', need=cost)
    vid = specs[0]['verification_id']
    key = f'{vid}_r1_ext{EXT}'
    work = ub.ROOT/'verify_ext'/sit.sid
    runner = sv.SubprocessBranchRunner(work/'branches', workers=2, timeout_s=3600)
    _write_new(ub.ROOT/'claims'/f'{key}.json', dict(key=key, sid=sit.sid, pid=os.getpid(), unix=time.time()))
    ub.charge(key, cost, 'a3_later_harm', len(specs))
    runner.prepare(specs, sit.snap)
    with ThreadPoolExecutor(max_workers=2) as tp:
        br = list(tp.map(lambda s: sv.run_branch(s, sit.snap, runner), specs))
    row = dict(sid=sit.sid, amendment='A3', seed=1, move=move, window=[sit.t0, sit.t0+EXT],
               statuses=[b['status'] for b in br])
    if any(b['status'] != 'graded' for b in br):
        return row
    rows = [dict(branches_dir=str(work/'branches'), verification_id=vid, branch=b, J=b['J_veh_s'], window=b['window'])
            for b in br]
    st = ub.stats(rows[0], rows[1], dict(start_s=sit.t0, end_s=sit.t0+EXT))
    t180 = []
    for r in rows:
        tp_ = ub.tape(r)
        t180.append(totals_at(tp_, sit.t0+ub.SPAN)['team_delay_vs']-totals_at(tp_, sit.t0)['team_delay_vs'])
    orig = sit.res[1]
    row.update(adv_ext=rows[0]['J']-rows[1]['J'], adv_ext_lb95=st['lb95'],
               safety_ext=ub.safety(rows[0], rows[1]), J_policy_ext=rows[0]['J'], J_move_ext=rows[1]['J'],
               adv_180_from_ext=t180[0]-t180[1], adv_180_orig=orig['policy']['J']-orig[move]['J'],
               determinism_ok=abs(t180[0]-orig['policy']['J']) < 1e-6 and abs(t180[1]-orig[move]['J']) < 1e-6)
    return row


def main():
    reg = json.loads((ub.ROOT/'AMENDMENT_A3.json').read_text())
    assert (ub.ROOT/'AMEND_DONE.json').exists() and not (ub.ROOT/'AMEND3_DONE.json').exists()
    sits_ = json.loads((ub.ROOT/'results'/'SITUATIONS.json').read_text())
    dirs = {}
    for d in sits_['situations'].values():
        dirs[json.loads((ub.ROOT.joinpath(d, 'CONTEXT.json')).read_text())['state_id']] = d
    best = json.loads((ub.ROOT/'results'/'BEST.json').read_text())
    order, total = [], ub.spent()
    for sid in reg['order']:
        sit = ub.Situation(dirs[sid])
        cost = 2*(sit.t0+EXT)
        if total+cost <= ub.CAP:
            order.append(sid)
            total += cost
    ub.log('a3_start', spent=ub.spent(), order=order)
    with ThreadPoolExecutor(max_workers=max(1, len(order))) as tp:
        for row in tp.map(lambda sid: one(dirs[sid], best[sid]), order):
            ub._append(ub.ROOT/'results'/'amend_a3.jsonl', row)
    ub._write(ub.ROOT/'AMEND3_DONE.json', dict(unix=time.time(), spent=ub.spent(), run=order))
    ub.log('a3_done', spent=ub.spent())


if __name__ == '__main__':
    main()
