"""Amendment 57 member-lane filter (CPU only, no SUMO): PART_L<k> commands only the planned members on SUMO lane k;
every other member keeps the U0 command and gets no lane hold; activation eligibility is the full group; filter off
(no a57_member_lane in the checkpoint) = the unchanged A37 CELL_BANK hook."""
from __future__ import annotations

import hashlib
import json

import pytest

from tools.coopmwm import a37_cellbank as C
from tools.coopmwm import a57_partlane as P
from tools.coopmwm import nm_eval_runner as R
from tools.coopmwm.test_a37_cellbank_cpu import plans, role, spec
from tools.coopmwm.test_learned_trigger_cpu import wm_rows

CELL = 'D2800_p60'
# wm_rows: c0 = WM lane 0 = SUMO lane 2, c1 = SUMO 1, c2 = SUMO 0 (SUMO lane = 2 - WM lane)
TRIO = spec('p_trio', [role(0, 20.0), role(1, 21.0), role(2, 22.0, onset_s=2)])
SUMO_OF = {'c0': 2, 'c1': 1, 'c2': 0}


class Ctl:
    def apply(self, rows, commands, now):
        return {vid: dict(u0=commands[vid]) for vid in commands}


def drive(hook, ts=range(300, 306)):
    c = Ctl()
    hook(c, None, None)
    rows = wm_rows(speed=25.0)
    cmds = {r['vid']: 0.3 for r in rows if r['is_cav']}
    return c, [c.apply(rows, cmds, t) for t in ts]


def make(lane=None, bank=None):
    h = C.make_hook('CELL_BANK', CELL, bank=bank or C.bank_doc(plans(TRIO), CELL))
    return h if lane is None else P.apply_filter(h, lane)


def plan_commanded(outs):
    return {vid for out in outs for vid, d in out.items() if 'oracle_plan' in d}


def test_filter_off_is_the_unchanged_hook():
    c_ref, ref = drive(make())
    assert plan_commanded(ref) == {'c0', 'c1', 'c2'}
    assert 'a57_part' not in make().receipts()


@pytest.mark.parametrize('lane', (0, 1, 2))
def test_only_members_on_the_lane_are_commanded(lane):
    c, outs = drive(make(lane))
    _, ref = drive(make())
    want = {v for v, s in SUMO_OF.items() if s == lane}
    assert plan_commanded(outs) == want
    for out, r in zip(outs, ref):
        for vid in SUMO_OF:
            if vid in want:
                assert out[vid] == r[vid]                     # the member on the lane: exactly the joint command
            else:
                assert out[vid] == dict(u0=0.3)               # every other member: the U0 command, as a non-member
    assert set(getattr(c, 'g5_lane_targets', {}) or {}) == want   # lane hold only for the commanded member


@pytest.mark.parametrize('lane', (0, 1, 2))
def test_activation_eligibility_is_the_full_group(lane):
    h = make(lane)
    drive(h)
    ref = make()
    drive(ref)
    assert [a['members'] for a in h.core.acts] == [a['members'] for a in ref.core.acts] == [['c2', 'c1', 'c0']]
    r = h.receipts()['a57_part']
    assert r['member_lane'] == lane and r['activations'] == 1 and r['activations_no_member_on_lane'] == 0
    assert r['members_planned_total'] == 3 and r['members_on_lane_total'] == 1
    assert r['acts'][0]['on_lane'] == [v for v, s in SUMO_OF.items() if s == lane]
    assert r['vehicles_with_accel_command'] == 1 and r['dropped_accel'] > 0 and r['a57_sha256'] == P.a57_digest()
    assert h.core.receipts() == ref.core.receipts()       # the executor state is the full-group state


def test_plan_without_member_on_lane_commands_nobody():
    duo = spec('p_duo', [role(1, 20.0), role(2, 21.0)])
    h = make(0, C.bank_doc(plans(duo), CELL))
    _, outs = drive(h)
    assert plan_commanded(outs) == set()
    r = h.receipts()['a57_part']
    assert r['activations'] == 1 and r['activations_no_member_on_lane'] == 1 and r['members_on_lane_total'] == 0


def test_bad_lane_refused():
    with pytest.raises(ValueError):
        make(3)


def test_runner_hook_filter_switch(tmp_path):
    bank = tmp_path/'bank.json'
    bank.write_text(json.dumps(C.bank_doc(plans(TRIO), CELL)), encoding='utf-8')
    ck = dict(a37_arm='CELL_BANK', cell_key=CELL, a37_sha256=C.a37_digest(),
              a37_bank=dict(path=str(bank), file_sha256=hashlib.sha256(bank.read_bytes()).hexdigest()))
    world = dict(cell_key=CELL)
    off, loaded = R.a37_hook(ck, world, 'CELL_BANK')
    assert loaded == C.a37_digest() and not hasattr(off, 'a57_member_lane')
    assert plan_commanded(drive(off)[1]) == {'c0', 'c1', 'c2'}
    on, _ = R.a37_hook(dict(ck, a57_member_lane=1, a57_sha256=P.a57_digest()), world, 'CELL_BANK')
    assert on.a57_member_lane == 1 and plan_commanded(drive(on)[1]) == {'c1'}
    with pytest.raises(ValueError):
        R.a37_hook(dict(ck, a57_member_lane=1, a57_sha256='0'*64), world, 'CELL_BANK')
    with pytest.raises(ValueError):
        R.a37_hook(dict(ck, a37_arm='A37_PLAN', a37_family='p_trio', a57_member_lane=1, a57_sha256=P.a57_digest()),
                   world, 'A37_PLAN')


def test_a37_digest_untouched_by_a57():
    assert 'a57' not in ' '.join(C.EXECUTED) and len(P.a57_digest()) == 64
