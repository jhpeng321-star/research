"""Amendment 61 within-lane pick (CPU only, no SUMO): PART_L2A commands only the lower-rank lane-2 member, PART_L2B
only the other; activation receipts equal the unfiltered core; no pick = the A57 lane filter unchanged."""
from __future__ import annotations

import hashlib
import json

import pytest

from tools.coopmwm import a37_cellbank as C
from tools.coopmwm import a57_partlane as P
from tools.coopmwm import a61_rankpick as K
from tools.coopmwm import nm_eval_runner as R
from tools.coopmwm.test_a37_cellbank_cpu import plans, spec
from tools.coopmwm.test_learned_trigger_cpu import wm_rows

CELL = 'D2800_p60'


def role(lane, rank, v_w=20.0, onset_s=0):
    return dict(lane=lane, rank=rank, v_w=v_w, dmax=1.0, hold_s=120, onset_s=onset_s, lc_target=None)


# lanes trigger: two members on SUMO lane 2 (WM lane 0) ranked by position (front = rank 0), one on SUMO lane 1
PLAN = spec('p_l2pair', [role(2, 1, 21.0), role(1, 0, 20.0), role(2, 0, 22.0, onset_s=2)], trigger='lanes',
            zone='full', spread_m=500.0)


def rows():
    r = wm_rows(speed=25.0)
    r.append(dict(vid='c3', s_m=-1100.0, v_mps=25.0, lane=0, is_cav=True, length_m=5.0))   # behind c0 on SUMO lane 2
    return r


class Ctl:
    def apply(self, rows, commands, now):
        return {vid: dict(u0=commands[vid]) for vid in commands}


def drive(hook, ts=range(300, 306)):
    c = Ctl()
    hook(c, None, None)
    rs = rows()
    cmds = {r['vid']: 0.3 for r in rs if r['is_cav']}
    return c, [c.apply(rs, cmds, t) for t in ts]


def make(pick=None):
    h = P.apply_filter(C.make_hook('CELL_BANK', CELL, bank=C.bank_doc(plans(PLAN), CELL)), 2)
    return h if pick is None else K.apply_pick(h, 2, pick)


def commanded(outs):
    return {vid for out in outs for vid, d in out.items() if 'oracle_plan' in d}


def test_lane_filter_alone_commands_both_lane2_members():
    assert commanded(drive(make())[1]) == {'c0', 'c3'}
    assert 'a61_pick' not in make().receipts()


@pytest.mark.parametrize('pick,want', (('A', {'c0'}), ('B', {'c3'})))
def test_pick_commands_one_member(pick, want):
    h = make(pick)
    c, outs = drive(h)
    _, ref = drive(make())
    assert commanded(outs) == want
    for o, r in zip(outs, ref):
        for vid in ('c0', 'c1', 'c2', 'c3'):
            assert o[vid] == (r[vid] if vid in want else dict(u0=0.3))
    assert set(c.g5_lane_targets) == want
    rr = h.receipts()
    k = rr['a61_pick']
    assert k['pick'] == pick and k['activations'] == 1 and k['activations_two_lane_members'] == 1
    assert k['activations_fewer_than_two'] == 0 and k['acts'][0]['on_lane'] == ['c0', 'c3']
    assert k['acts'][0]['picked'] == sorted(want) and k['a61_sha256'] == K.a61_digest()
    ref_h = make()
    drive(ref_h)
    assert h.core.receipts() == ref_h.core.receipts()           # full-group executor state unchanged
    assert rr['a57_part']['member_lane'] == 2


def test_pick_needs_the_lane_filter_and_valid_pick():
    with pytest.raises(ValueError):
        K.apply_pick(C.make_hook('CELL_BANK', CELL, bank=C.bank_doc(plans(PLAN), CELL)), 2, 'A')
    with pytest.raises(ValueError):
        K.apply_pick(make(), 2, 'C')
    with pytest.raises(ValueError):
        K.apply_pick(make(), 1, 'A')


def test_runner_switch(tmp_path):
    bank = tmp_path/'bank.json'
    bank.write_text(json.dumps(C.bank_doc(plans(PLAN), CELL)), encoding='utf-8')
    ck = dict(a37_arm='CELL_BANK', cell_key=CELL, a37_sha256=C.a37_digest(), a57_member_lane=2,
              a57_sha256=P.a57_digest(),
              a37_bank=dict(path=str(bank), file_sha256=hashlib.sha256(bank.read_bytes()).hexdigest()))
    world = dict(cell_key=CELL)
    off, _ = R.a37_hook(ck, world, 'CELL_BANK')
    assert not hasattr(off, 'a61_member_pick') and commanded(drive(off)[1]) == {'c0', 'c3'}
    on, _ = R.a37_hook(dict(ck, a61_member_pick='B', a61_sha256=K.a61_digest()), world, 'CELL_BANK')
    assert on.a61_member_pick == 'B' and commanded(drive(on)[1]) == {'c3'}
    with pytest.raises(ValueError):
        R.a37_hook(dict(ck, a61_member_pick='B', a61_sha256='0'*64), world, 'CELL_BANK')
    no_lane = {k: v for k, v in ck.items() if k not in ('a57_member_lane', 'a57_sha256')}
    with pytest.raises(ValueError):
        R.a37_hook(dict(no_lane, a61_member_pick='A', a61_sha256=K.a61_digest()), world, 'CELL_BANK')


def test_a57_digest_unchanged_by_a61():
    assert P.a57_digest().startswith('f09d0196e474')
