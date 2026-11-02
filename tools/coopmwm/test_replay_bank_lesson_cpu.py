"""CPU regression for the LESSON-LEARNED CHECK (MAIN 2026-11-03 01:22): after EVERY B and R
update in B-REPLAY, one exact SUMO branch of the NEW policy continues from each entry of the
FROZEN lesson list (same window, same CRN prefix/generator state). recapture =
(J_policy_u3 - J_new_policy) / (J_policy_u3 - J_best_plan); plus the imitation action error
(m/s^2) of the new policy vs the plan actions on the chosen branch tape. Charges are
lesson_u<k>-prefixed in bank_verification_ext (the same entry is re-measured every update, so
unqualified ids would hit the durable duplicate-charge guard from the second update on).
"""
import json

import pytest
import torch

from tools.coopmwm import replay_bank as rb
from tools.coopmwm import sumo_verify as sv
from tools.coopmwm.exact_grade import BudgetBlocked, verify_sealed
from tools.coopmwm.storage import ledger
from tools.coopmwm.test_replay_bank_confirm_cpu import fp, snapshot, plan


class LessonTapeRunner:
    """'Policy continues' J depends on the snapshot's policy sha (u3 = 'p' -> 500, any NEW
    policy -> 470), so a genuinely recapturing policy is measurable; candidate branches return
    J = J_u3 - 10*mean(profile) and write a RESULT.json tape with member actions (context and
    action included) for the imitation-error check."""

    def __init__(self, work):
        self.work = work

    def J(self, p):
        prof = [x for row in p['profile'] for x in row]
        return 500.-10.*sum(prof)/len(prof)

    def run(self, spec, snap):
        t0, t_end = spec['window']['start_s'], spec['window']['end_s']
        w = snap['episode']['world']['seed']
        d = self.work/'branches'/spec['verification_id']/spec['branch'];d.mkdir(parents=True, exist_ok=True)
        row = dict(time_s=t0, vehicles=[dict(lane='m1', s_m=10., v_mps=8., length_m=5.),
                                        dict(lane='m1', s_m=0., v_mps=10., length_m=5.)])
        import gzip
        from tools.coopnative.storage import canonical
        with gzip.open(d/'observable.jsonl.gz', 'wt', encoding='utf-8') as f:
            f.write(canonical(row)+'\n')
        J = (500. if snap.get('policy', {}).get('state_dict_sha256') == 'p' else 470.) if spec['plan'] is None \
            else self.J(spec['plan'])
        prefixes = {str(t): fp(t, 3.*w+0.5*t*t/100.) for t in range(300, t0+1)}
        r = dict(prefixes=prefixes, final=fp(t_end, 3.*w+0.5*t0*t0/100.+J), actual_sim_s=t_end-300)
        p = spec['plan']
        if p is not None:
            from tools.coopmwm.accel_storage import write as awrite
            from tools.coopmwm.protocol import digest
            tape = dict(steps=[dict(time_s=t, sample=dict(context=dict(members=list(p['members'])),
                action=[0.5, -0.5],
                plan_executions=[dict(plan_id=p['id'], proposal_members=list(p['members']))]))
                for t in range(spec['window']['start_s'], spec['window']['start_s']+p['duration_s'])])
            awrite(d/'RESULT.json', tape)
            r['result_sha256'] = digest(tape)
        return r


class FakePolicy:
    """A NEW policy: state dict differs from the u3 weights; mean action [0.3, -0.4] m/s^2."""

    def state_dict(self):
        return {'w': torch.zeros(3)}

    def __call__(self, context):
        return torch.tensor([0.3, -0.4]), torch.tensor(0.)


def admitted_entry(root, t0, pid, gain):
    """One first-seed accepted, fully confirmed (admitted) bank decision at t0, with the
    collector's snapshot file and decision link (as lesson_check's own-bank path reads them)."""
    bank = rb.bank_root(root, 'B')
    snap = snapshot(t0)
    first = sv.verify(snap, [plan(pid, gain)], LessonTapeRunner(bank/'sumo_verification'),
                      lambda vid, specs: None, root=bank/'sumo_verification')
    assert first['status'] == 'accepted', first.get('reason')
    receipt = {k: v for k, v in first.items() if not k.startswith('_')}
    out = rb.confirm(root, 'B', receipt, snap, lambda work: LessonTapeRunner(work), str(root),
                     'bank_verification_ext', workers=2, version='u4')
    assert out['admitted'] is True, out
    state_id = f'snap_{pid}'
    (bank/'snapshots').mkdir(exist_ok=True)
    torch.save(snap, bank/'snapshots'/(state_id+'.pt'))
    (bank/'decisions').mkdir(exist_ok=True)
    (bank/'decisions'/(state_id+'.json')).write_text(json.dumps(
        dict(state_id=state_id, verification_id=receipt['verification_id'], status='accepted',
             policy_update=4, arm='B', available_k=1, issued_unix=1.)))
    return first


def _runner(work):
    return LessonTapeRunner(work)


def test_recapture_frozen_list_action_error_and_charges(tmp_path):
    root = tmp_path;bank = rb.bank_root(root, 'B');bank.mkdir(parents=True)
    first = admitted_entry(root, 420, 'a', 4.)
    out = rb.record_lesson(root, 'B', FakePolicy(), str(root), 4, _runner)
    e = out['entries'][0]
    assert e['status'] == 'graded'
    # u3 policy J=500, the new policy J=470, the plan J=460 -> recapture 0.75
    assert e['J_policy_u3_veh_s'] == 500. and e['J_best_plan_veh_s'] == 460.
    assert e['J_new_policy_veh_s'] == 470. and e['recapture'] == 0.75
    assert out['recapture_mean'] == 0.75 and out['graded_entries'] == 1
    # imitation action error on the bank state: mean |[0.3,-0.4]-[0.5,-0.5]| = 0.15 m/s^2
    assert e['imitation_action_error_mps2'] == pytest.approx(0.15)
    assert out['imitation_action_error_mean_mps2'] == pytest.approx(0.15)
    # same window as the first-seed entry (CRN: identical prefixes/generator, only weights differ)
    orig = json.loads((bank/'sumo_verification'/'verifications'/f'{first["verification_id"]}.json').read_text())
    assert e['new_policy_branch']['window'] == orig['window']
    # the frozen list is sealed and holds exactly this first admitted entry
    lst = json.loads((bank/'LESSON_LIST_B.json').read_text())
    verify_sealed(lst, 'lesson_list_sha256')
    assert lst['verification_ids'] == [first['verification_id']]
    saved = json.loads((root/'LESSON_CHECK_u004_B.json').read_text())
    verify_sealed(saved, 'lesson_check_sha256')
    # charges: one lesson_u004 policy-branch line in bank_verification_ext
    le = [x for x in ledger(root) if x['id'].startswith('lesson_u004_')]
    assert len(le) == 1 and le[0]['phase'] == 'bank_verification_ext' and le[0]['id'].endswith('_policy')
    # SECOND update: a new admitted entry exists, but the list stays FROZEN (same entry re-measured)
    admitted_entry(root, 460, 'b', 2.)
    out5 = rb.record_lesson(root, 'B', FakePolicy(), str(root), 5, _runner)
    assert out5['list_size'] == 1 and len(out5['entries']) == 1
    assert out5['entries'][0]['verification_id'] == first['verification_id']
    assert out5['recapture_mean'] == 0.75
    assert len([x for x in ledger(root) if x['id'].startswith('lesson_u005_')]) == 1   # unique per update


def test_empty_bank_leaves_the_list_open(tmp_path):
    root = tmp_path;rb.bank_root(root, 'B').mkdir(parents=True)
    out = rb.record_lesson(root, 'B', FakePolicy(), str(root), 4, _runner)
    assert out['list_size'] == 0 and 'reason' in out and out['entries'] == []
    assert not (rb.bank_root(root, 'B')/'LESSON_LIST_B.json').exists()
    saved = json.loads((root/'LESSON_CHECK_u004_B.json').read_text())
    verify_sealed(saved, 'lesson_check_sha256')


def test_budget_blocked_and_missing_receipts_are_never_imputed(tmp_path, monkeypatch):
    root = tmp_path;rb.bank_root(root, 'B').mkdir(parents=True)
    first = admitted_entry(root, 420, 'a', 4.)

    def blocked(vid, specs):
        raise BudgetBlocked('phase cap')
    monkeypatch.setattr(rb, '_confirm_charge', lambda run_root, phase, kind: blocked)
    out = rb.record_lesson(root, 'B', FakePolicy(), str(root), 4, _runner)
    e = out['entries'][0]
    assert e['status'] == 'budget_blocked' and e['recapture'] is None
    assert out['recapture_mean'] is None and out['graded_entries'] == 0
    monkeypatch.setattr(rb, 'bank_decisions', lambda *a, **k: [])   # receipt vanished -> never imputed
    out = rb.record_lesson(root, 'B', FakePolicy(), str(root), 5, _runner)
    assert out['entries'][0]['status'] == 'missing_or_ungraded' and out['entries'][0]['recapture'] is None
    assert out['entries'][0]['verification_id'] == first['verification_id']
