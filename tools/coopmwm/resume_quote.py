"""Receipt-only latest-brief re-quote; no dispatch, process control or physics."""
import json
from .accel_protocol import PHASES
from .branch_eval import budget_quote


def quote(cap=7500000):
    endpoints = (48, 48, 24)
    branches = budget_quote(96, [300, 500, 700, 900], ['MAPPO', 'A', 'B'])
    base = dict(PHASES, train=sum(endpoints)*24*1200,
                cold_B_retained=24*1200, failed_A_retained=12*1200,
                paired_branch=branches['total_sim_s'], branch_reuse_equivalence=600,
                discovery=200000, exact_plan_grading=1000000)
    scenarios = []
    for train_every, dev_every in ((4, 8), (4, 16), (8, 16)):
        phases = dict(base,
            checks=sum(u//train_every for u in endpoints)*12*1200,
            dev=sum(u//dev_every for u in endpoints)*24*1200)
        # A's completed u4 check remains charged when future checks move to u8.
        if train_every == 8:
            phases['completed_A_u4_check_retained'] = 12*1200
        total = sum(phases.values())
        scenarios.append(dict(train_every=train_every, dev_every=dev_every,
            phases=phases, allocated_sim_s=total, over_cap_sim_s=max(0, total-cap)))
    selected = scenarios[1]  # MAIN23:54: TRAIN4/dev16, R branches deferred.
    minimum = selected['allocated_sim_s']
    return dict(cap_sim_s=cap, source='MAIN02:24 cap7500000; MAIN23:54 original allocation; every-plan exact grading reserve1000000',
        updates=dict(A=48, B=48, R=24), episodes_per_update=24,
        episode_sim_s=1200, branch=branches, scenarios=scenarios,
        selected=selected, allocated_sim_s=minimum, unallocated_sim_s=cap-minimum,
        minimum_allocated_sim_s=minimum, minimum_over_cap_sim_s=max(0, minimum-cap),
        over_brief_6500000_sim_s=max(0, minimum-6500000),
        admitted=minimum<=cap, new_sumo_sim_s=0,
        accounting=['All superseded set-point charges retained: 811640.',
            'All 24 cold B reservations retained: 28800, including unclaimed jobs.',
            'All 12 failed A reservations retained: 14400.',
            'A completed training u1..4 is inside the 48-update train allocation, not added twice.',
            'Completed A u4 TRAIN check is additional only for future every8 schedule.',
            'Original mechanics/calibration retained; MAIN23:54 discovery reserve reduced to 200000.',
            'All 96 episode-test worlds and A/B branches protected; R branches deferred pending owner decision.',
            'MAIN02:24 adds 1000000 exact-plan grading sim-s; complete prefix+duration+120 reserved before each grade, no refunds.',
            'Expected 720 minimum B/R plan starts over 48/24 updates; ~1000sim-s/grade is a quote, not a per-grade cap.',
            'Extra guided starts consume the same finite reserve; insufficient complete-grade budget is STOP, never imputation.',
            'MAPPO branch reference reuse assumed to PASS; failure adds 115200, not included.'],
        disposition='STOP before new physics: owner budget/scope decision required' if minimum>cap else 'prospective registration still required')


def snapshot():
    from . import ops
    from .storage import write
    state = json.loads(ops.ssh.remote('cpuQ', '''import pathlib,json,time
r=pathlib.Path('/home/ubuntu/coopmwm20261101_accel1');assert r.is_dir()
d=json.loads((r/'A_WARM5_DISPATCH.json').read_text());p=pathlib.Path('/proc')/str(d['pid'])
s=(p/'stat').read_text().split() if (p/'stat').exists() else None
boot=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
assert not s or (boot==d['boot_id'] and s[21]==d['start_ticks'])
paused=json.loads((r/'A_WARM5_CONTINUATION_PAUSED.json').read_text())
assert paused['update']==4
charges=[json.loads(x) for x in (r/'CHARGES.jsonl').read_text().splitlines()]
print(json.dumps(dict(unix=time.time(),A_owner=d,A_state=s[2] if s else 'exited',
 A_pause=paused,charged_sim_s=811640+sum(c['sim_s'] for c in charges),
 updates=sorted(p.name for p in r.glob('*_UPDATE_*.json')),new_sumo_sim_s=0)))
''', timeout=25))
    value = dict(quote(), live=state)
    write(ops.EVIDENCE/f'RESUME_BUDGET_BLOCK_{int(state["unix"])}.json', value)
    print(json.dumps(value, indent=2))


if __name__ == '__main__':
    snapshot()
