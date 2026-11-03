"""CPU regression for the V2V bank pairs and new-policy branch wiring (MAIN 2026-11-03,
part 3/3 BANK/IMITATION). Flag ``v2v`` DEFAULT OFF: bank_pairs output stays byte-identical
and the branch runner uses the plain controller. When ON, every imitation pair of an
admitted bank entry carries the partners' v2v features of its second, computed from the
winning plan's recorded branch trace (positions, speeds, accelerations; the last issued
command at the previous second); pairs whose trace lacks partner commands are excluded and
counted; the lesson check's new-policy branches run with the v2v observation block enabled
(last_commands computed per step)."""
import gzip
import json
import os

import pytest
import torch

from tools.coopmwm import bank_v2v
from tools.coopmwm import replay_bank as rb
from tools.coopmwm import sumo_verify as sv
from tools.coopmwm.exact_grade import verify_sealed
from tools.coopnative.storage import canonical
from tools.coopmwm.test_replay_bank_confirm_cpu import (bank_budget, fp, snapshot, plan,
                                                        FakeRunner)
from tools.coopmwm.test_replay_bank_lesson_cpu import (admitted_entry, FakePolicy,
                                                       LessonTapeRunner)


class V2VTapeRunner(FakeRunner):
    """FakeRunner whose candidate branches write a RESULT.json plan tape (context members,
    actions, plan_executions - as the real plant does) AND an observable trace covering every
    plan second plus the previous one, with CAV partner rows carrying issued commands.
    ``prev_second``/``issue_commands`` vary the trace to exercise the exclusion rules."""

    def __init__(self, work, prev_second=True, issue_commands=True):
        super().__init__(work)
        self.prev_second = prev_second
        self.issue_commands = issue_commands
        self.t0 = None

    def frame(self, t):
        i = t-self.t0
        return dict(time_s=t, vehicles=[
            dict(vid='c1', lane=0, is_cav=True, length_m=5., s_m=10.+i, v_mps=8.,
                 issued_acceleration_mps2=1.+0.1*i, command_present=True),
            dict(vid='c2', lane=0, is_cav=True, length_m=5., s_m=24.+1.5*i, v_mps=10.5,
                 issued_acceleration_mps2=(-2.-0.1*i) if self.issue_commands else None,
                 command_present=bool(self.issue_commands))])

    def run(self, spec, snap):
        r = super().run(spec, snap)
        p = spec['plan']
        if p is None:
            return r
        from tools.coopmwm.accel_storage import write as awrite
        from tools.coopmwm.protocol import digest
        d = self.work/'branches'/spec['verification_id']/spec['branch']
        tape = dict(steps=[dict(time_s=t, sample=dict(context=dict(members=list(p['members'])),
            action=[0.5, -0.5], absolute_accels={v: 0.5 for v in p['members']},
            plan_executions=[dict(plan_id=p['id'], proposal_members=list(p['members']))]))
            for t in range(spec['window']['start_s'], spec['window']['start_s']+p['duration_s'])])
        awrite(d/'RESULT.json', tape)
        r['result_sha256'] = digest(tape)
        self.t0 = spec['window']['start_s']
        first = self.t0-1 if self.prev_second else self.t0
        with gzip.open(d/'observable.jsonl.gz', 'wt', encoding='utf-8') as f:
            for t in range(first, self.t0+p['duration_s']):
                f.write(canonical(self.frame(t))+'\n')
        return r


def _bank(root, bank, runner):
    first = sv.verify(snapshot(420), [plan('a', 1.)], runner(bank/'sumo_verification'),
                      lambda vid, specs: None, root=bank/'sumo_verification')
    assert first['status'] == 'accepted', first.get('reason')
    receipt = {k: v for k, v in first.items() if not k.startswith('_')}
    out = rb.confirm(root, 'B', receipt, snapshot(420), lambda work: runner(work), str(root),
                     'bank_verification_ext', workers=2, version='u4')
    assert out['admitted'] is True, out


def test_pair_v2v_block_from_synthetic_trace():
    frames = {t: dict(time_s=t, vehicles=[
        dict(vid='c1', is_cav=True, s_m=10., v_mps=8., issued_acceleration_mps2=1.),
        dict(vid='c2', is_cav=True, s_m=24., v_mps=10.5, issued_acceleration_mps2=-2.),
        dict(vid='hd', is_cav=False, s_m=3., v_mps=9., issued_acceleration_mps2=9.)])
        for t in (419, 420)}
    block, missing = bank_v2v.pair_v2v_block(frames[420], frames[419], ['c1', 'c2'])
    assert missing == [] and sorted(block) == ['c1', 'c2']
    assert bank_v2v.STUB is False and bank_v2v.V2V_DIM == 24   # OBS part landed and is picked up
    # partner c2 by ego c1 - OBSERVATION layout: gap, lane offset, speed, accel,
    # previous-second command, presence
    assert block['c1'][:6] == pytest.approx([14., 0., 10.5, -2., -2., 1.])
    assert block['c1'][6:] == [0.]*18                       # zero padded to V2V_DIM
    assert block['c2'][:6] == pytest.approx([-14., 0., 8., 1., 1., 1.])
    # non-CAV rows are never partners
    frame = dict(time_s=420, vehicles=[dict(vid='c1', is_cav=True, s_m=10., v_mps=8.,
                                           issued_acceleration_mps2=1.)])
    solo, _ = bank_v2v.pair_v2v_block(frame, frame, ['c1'])
    assert solo['c1'] == [0.]*bank_v2v.V2V_DIM
    # partners beyond the shared comm range are invisible (drifted far apart)
    far = dict(time_s=420, vehicles=[
        dict(vid='c1', is_cav=True, s_m=10., v_mps=8., issued_acceleration_mps2=1.),
        dict(vid='c3', is_cav=True, s_m=500., v_mps=9., issued_acceleration_mps2=.5)])
    out_block, _ = bank_v2v.pair_v2v_block(far, far, ['c1'])
    assert out_block['c1'] == [0.]*bank_v2v.V2V_DIM


def test_attach_pairs_excludes_and_counts_missing_commands(tmp_path):
    trace = tmp_path/'observable.jsonl.gz'
    frames = {t: dict(time_s=t, vehicles=[
        dict(vid='c1', is_cav=True, s_m=10., v_mps=8., issued_acceleration_mps2=1.),
        dict(vid='c2', is_cav=True, s_m=24., v_mps=10.5,
             issued_acceleration_mps2=(-2. if t in (419, 420) else None), command_present=t in (419, 420))])
        for t in (419, 420)}
    with gzip.open(trace, 'wt', encoding='utf-8') as f:
        for t in (419, 420):
            f.write(canonical(frames[t])+'\n')
    pairs = [(dict(context=dict(members=['c1', 'c2']),
                   behaviour=dict(proposal_members=['c1', 'c2'])), 1.) for _ in range(2)]
    # both seconds present with commands: kept
    kept, counts = bank_v2v.attach_pairs(pairs[:1], trace, 420)
    assert counts == dict(pairs=1, kept=1, excluded_missing_prev_second=0,
                          excluded_missing_partner_command=0, trace_missing=False)
    assert kept[0][0]['context']['v2v']['c1'][-4:] == [0.]*4
    # the previous second of the FIRST pair (420) exists; a pair at 421 has no frame -> excluded
    kept, counts = bank_v2v.attach_pairs(pairs, trace, 420)
    assert counts['pairs'] == 2 and counts['kept'] == 1
    assert counts['excluded_missing_prev_second'] == 1
    # a trace without any issued command at the previous second -> every pair excluded
    frames419 = json.loads(json.dumps(frames[419]))
    frames419['vehicles'][1]['issued_acceleration_mps2'] = None
    frames419['vehicles'][1]['command_present'] = False
    trace2 = tmp_path/'no_cmd.jsonl.gz'
    with gzip.open(trace2, 'wt', encoding='utf-8') as f:
        f.write(canonical(frames419)+'\n');f.write(canonical(frames[420])+'\n')
    kept, counts = bank_v2v.attach_pairs(pairs[:1], trace2, 420)
    assert counts['kept'] == 0 and counts['excluded_missing_partner_command'] == 1
    # missing trace file: everything excluded, disclosed
    kept, counts = bank_v2v.attach_pairs(pairs[:1], tmp_path/'absent.jsonl.gz', 420)
    assert kept == [] and counts['trace_missing'] is True and counts['pairs'] == 1


def test_bank_pairs_default_off_is_byte_identical(tmp_path):
    root = tmp_path;bank = rb.bank_root(root, 'B');bank.mkdir(parents=True)
    _bank(root, bank, lambda work: V2VTapeRunner(work))
    p1, s1 = rb.bank_pairs(root, 'B', 12345)
    p2, s2 = rb.bank_pairs(root, 'B', 12345, v2v=False)
    assert json.dumps([p1, s1]) == json.dumps([p2, s2])
    assert 'v2v' not in s1 and len(p1) == 120
    assert all('v2v' not in p[0].get('context', {}) for p in p1)
    p3, s3 = rb.bank_pairs(root, 'B', 12345, v2v=True)
    assert json.dumps([p1, s1]) != json.dumps([p3, s3])
    assert 'v2v' in s3


def test_bank_pairs_v2v_attaches_partner_features_from_the_branch_trace(tmp_path):
    root = tmp_path;bank = rb.bank_root(root, 'B');bank.mkdir(parents=True)
    _bank(root, bank, lambda work: V2VTapeRunner(work))
    pairs, s = rb.bank_pairs(root, 'B', 12345, v2v=True)
    assert len(pairs) == 120
    assert s['v2v']['stub'] is False and s['v2v']['cfg'] == bank_v2v.DEFAULT_CFG
    assert s['v2v']['counts'] == dict(pairs=120, kept=120, excluded_missing_prev_second=0,
                                      excluded_missing_partner_command=0, trace_missing=False)
    for i, (sample, w) in enumerate(pairs[:2]):
        block = sample['context']['v2v']
        assert sorted(block) == ['c1', 'c2']
        # pair i is second 420+i: partner c2 by ego c1 (OBS layout: gap, lane offset,
        # speed, accel, previous-second command, presence)
        assert block['c1'][:6] == pytest.approx(
            [14.+0.5*i, 0., 10.5, -2.-0.1*i, -1.9-0.1*i, 1.])
        # partner c1 by ego c2 (relative position carries the other sign)
        assert block['c2'][:6] == pytest.approx(
            [-(14.+0.5*i), 0., 8., 1.+0.1*i, 0.9+0.1*i, 1.])
        assert len(block['c1']) == bank_v2v.V2V_DIM and w == 1.
        assert block['c1'][6:] == [0.]*18
    assert s['entries'][0]['verification_id'] == s['entries'][0]['verification_id']
    assert s['counts']['used_admitted'] == 1            # the entry itself is unchanged


def test_bank_pairs_v2v_excludes_pairs_lacking_the_previous_second(tmp_path):
    root = tmp_path;bank = rb.bank_root(root, 'B');bank.mkdir(parents=True)
    _bank(root, bank, lambda work: V2VTapeRunner(work, prev_second=False))
    pairs, s = rb.bank_pairs(root, 'B', 12345, v2v=True)
    # only the FIRST plan second lacks its previous-second frame; the rest of the trace covers it
    assert s['v2v']['counts'] == dict(pairs=120, kept=119, excluded_missing_prev_second=1,
                                      excluded_missing_partner_command=0, trace_missing=False)
    assert len(pairs) == 119
    # the first kept pair is second 421 (i=1 in the trace frame arithmetic)
    assert pairs[0][0]['context']['v2v']['c1'][:6] == pytest.approx([14.5, 0., 10.5, -2.1, -2., 1.])


def test_bank_pairs_v2v_excludes_pairs_lacking_partner_commands(tmp_path):
    root = tmp_path;bank = rb.bank_root(root, 'B');bank.mkdir(parents=True)
    _bank(root, bank, lambda work: V2VTapeRunner(work, issue_commands=False))
    pairs, s = rb.bank_pairs(root, 'B', 12345, v2v=True)
    assert pairs == []
    assert s['v2v']['counts']['excluded_missing_partner_command'] == 120
    assert s['v2v']['counts']['kept'] == 0


def test_branch_runner_controller_is_flag_gated(monkeypatch):
    monkeypatch.delenv(bank_v2v.ENV, raising=False)
    assert bank_v2v.branch_controller_class() is None          # default: plain runner, byte-identical
    with bank_v2v.enabled(True):
        cls = bank_v2v.branch_controller_class()
        assert cls is not None and any(k.__name__ == 'VerifyController' for k in cls.__mro__)
    assert bank_v2v.branch_controller_class() is None
    assert os.environ.get(bank_v2v.ENV) is None                # restored


class TinyPolicy:
    """Stands in for the actor inside Controller.apply (CPU only, no SUMO)."""

    def __init__(self):
        self.contexts = []

    def sample(self, context, generator, deterministic=False, memory=None):
        self.contexts.append(context)
        n = len(context['members'])
        sample = dict(context=context, action=[[0.2]]*n, own_action=[[0.2]]*n,
                      previous_noise=[[0.]]*n, deterministic=True, rho=.5, window=1.,
                      logp=0., value=0.)
        return sample, {v: 0. for v in context['members']}


def _row(vid, s, v, lane=0, cav=True):
    return dict(vid=vid, s_m=float(s), v_mps=float(v), lane=lane, is_cav=cav, length_m=5.)


def test_runtime_branch_runner_computes_last_commands_per_step(monkeypatch):
    monkeypatch.delenv(bank_v2v.ENV, raising=False)
    with bank_v2v.enabled(True):
        cls = bank_v2v.branch_controller_class()
    policy = TinyPolicy()
    c = cls(policy, torch.Generator().manual_seed(1), snapshot(300), None)
    c.apply([_row('c1', 10., 8.), _row('c2', 24., 10.5), _row('hd', 5., 9., cav=False)],
            dict(c1=.5, c2=-1.), 418)
    c.apply([_row('c1', 18., 7.6), _row('c2', 35.5, 9.9), _row('hd', 14., 9., cav=False)],
            dict(c1=.4, c2=-.9), 419)
    # the v2v block reached the policy observation at every step
    assert [sorted(x['v2v']) for x in policy.contexts] == [['c1', 'c2'], ['c1', 'c2']]
    block = policy.contexts[1]['v2v']
    # partner c2 by ego c1 at 419 (OBS layout): relative position 17.5, lane offset 0,
    # speed 9.9, this-second accel 9.9-10.5, and the LAST ISSUED command of the previous
    # second (in effect at 419) = -0.9; presence 1.
    assert block['c1'][:6] == pytest.approx([17.5, 0., 9.9, -0.6, -0.9, 1.])
    assert block['c2'][:6] == pytest.approx([-17.5, 0., 7.6, -0.4, 0.4, 1.])
    # non-CAV 'hd' is never a partner and absent members get the zero block
    assert block['c1'][6:] == [0.]*18 and len(block['c1']) == bank_v2v.V2V_DIM
    gone = c.v2v_block(dict(members=['c1', 'gone']))
    assert gone['gone'] == [0.]*bank_v2v.V2V_DIM
    assert os.environ.get(bank_v2v.ENV) is None


def test_lesson_check_v2v_enables_the_branch_runner_block_and_discloses(tmp_path):
    root = tmp_path;bank = rb.bank_root(root, 'B');bank.mkdir(parents=True)
    admitted_entry(root, 420, 'v', 4.)
    seen = []

    class Probe(LessonTapeRunner):
        def run(self, spec, snap):
            seen.append(os.environ.get(bank_v2v.ENV))
            return super().run(spec, snap)

    out = rb.record_lesson(root, 'B', FakePolicy(), str(root), 4, lambda work: Probe(work), v2v=True)
    assert seen == ['1']                                   # flag set while the branch ran
    assert os.environ.get(bank_v2v.ENV) is None            # and restored afterwards
    e = out['entries'][0]
    assert e['status'] == 'graded' and e['recapture'] == 0.75   # physics untouched by the flag
    assert out['v2v']['enabled'] is True
    saved = json.loads((root/'LESSON_CHECK_u004_B.json').read_text())
    verify_sealed(saved, 'lesson_check_sha256')
    assert saved['v2v']['enabled'] is True


def test_lesson_check_default_never_sets_the_flag_or_the_key(tmp_path):
    root = tmp_path;bank = rb.bank_root(root, 'B');bank.mkdir(parents=True)
    admitted_entry(root, 420, 'v', 4.)
    seen = []

    class Probe(LessonTapeRunner):
        def run(self, spec, snap):
            seen.append(os.environ.get(bank_v2v.ENV))
            return super().run(spec, snap)

    out = rb.record_lesson(root, 'B', FakePolicy(), str(root), 4, lambda work: Probe(work))
    assert seen == [None]
    assert 'v2v' not in out and os.environ.get(bank_v2v.ENV) is None
