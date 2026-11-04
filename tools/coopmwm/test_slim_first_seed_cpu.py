"""CPU tests: slim_first_seed_tapes (spec slim_extend_spec.md 13:30). No SUMO: reuses the
deterministic per-second plant of test_verify_perseed_cpu (exact team_delay_vs fingerprints,
gz tapes), the legacy confirmation path of replay_bank and the FakePolicy of the lesson tests."""
import hashlib
import json
import shutil
import time
from pathlib import Path

import pytest
import torch

from tools.coopmwm import bank_statistics as bs
from tools.coopmwm import branch_eval
from tools.coopmwm import coopfix_exact_gate as ceg
from tools.coopmwm import replay_bank as rb
from tools.coopmwm import slim_tapes as st
from tools.coopmwm import sumo_verify as sv
from tools.coopmwm.exact_grade import _seal, verify_sealed
from tools.coopmwm.protocol import digest
from tools.coopmwm.test_replay_bank_confirm_cpu import bank_budget, plan, snapshot  # noqa: F401
from tools.coopmwm.test_replay_bank_lesson_cpu import FakePolicy, LessonTapeRunner
from tools.coopmwm.test_verify_perseed_cpu import PHASE, SeedPlant, _budget  # noqa: F401

K = 4                      # candidates per synthetic decision


class ActionPlant(SeedPlant):
    """SeedPlant whose committed-plan tape steps also carry context/action rows, so the
    imitation readers (_action_error, bank_pairs) can consume the tapes (identity-safe:
    the receipt's result_sha256 is updated to the rewritten tape)."""

    def run(self, spec, snap):
        from tools.coopmwm.accel_storage import read, write as awrite
        r = super().run(spec, snap)
        p = spec['plan']
        if p is not None:
            d = self.work/'branches'/spec['verification_id']/spec['branch']
            tape = read(d/'RESULT.json')
            tape['steps'] = [dict(s, sample=dict(s['sample'],
                                                 context=dict(members=list(p['members'])),
                                                 action=[0.5, -0.5])) for s in tape['steps']]
            (d/st.PAYLOAD).unlink(missing_ok=True)     # the gz payload is written 'xb'
            (d/st.MARKER).unlink(missing_ok=True)
            awrite(d/'RESULT.json', tape)
            r['result_sha256'] = digest(tape)
        return r


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def decision(root, tag, t0, gains=None, k=K, arm='B'):
    """One first-seed sv.verify decision with its own episode id; gains[(plan_id, seed)] =
    per-second delay reduction (default: every candidate loses -> the receipt is REJECTED)."""
    bank = rb.bank_root(root, arm)
    snap = snapshot(t0)
    snap['episode'] = dict(snap['episode'], id=f'warm_u005_{arm}_{tag}', arm=arm)
    plans = [plan(f'{tag}p{i+1}', 0.1) for i in range(k)]
    plant = ActionPlant(bank/'sumo_verification', gains or {})
    r = sv.verify(snap, plans, plant, sv.local_charge(root/'charges'), 'wm',
                  root=bank/'sumo_verification')
    return snap, r


def confirm(root, snap, r, arm='B'):
    """Sealed LEGACY confirmation of one accepted first-seed decision (replay_bank.confirm)."""
    pid = r['chosen_plan']['id']

    def runner_for(work):
        return ActionPlant(work, {(pid, 1): 0.4, (pid, 2): 0.4})
    receipt = {k: v for k, v in r.items() if not k.startswith('_')}
    return rb.confirm(root, arm, receipt, snap, runner_for, str(root), PHASE, workers=2,
                      version='u4')


def link_decision(bank, snap, r, state_id):
    """The collector-side decision link + snapshot the legacy readers need."""
    (bank/'snapshots').mkdir(parents=True, exist_ok=True)
    torch.save(snap, bank/'snapshots'/(state_id+'.pt'))
    (bank/'decisions').mkdir(exist_ok=True)
    (bank/'decisions'/(state_id+'.json')).write_text(json.dumps(
        dict(state_id=state_id, verification_id=r['verification_id'], status='accepted',
             policy_update=4)))


def retired_file(root, vids, name='ALL8_RETIRED.json'):
    p = root/name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(_seal(dict(rule='MAIN: retired from all8', verification_ids=list(vids)),
                                    'retired_sha256')))
    return p


def slimmed_dirs(bank):
    return {str(d.parent.resolve()) for d in Path(bank).rglob(st.SLIM)}


def series_of(branches_dir, vid, branch, t0, t_end):
    tape = bs._read_result(branches_dir, vid, branch)
    return [branch_eval.totals_at(tape, t)['team_delay_vs'] for t in range(t0, t_end+1)]


def branches_of(receipt, slimmed=False):
    want = [b['branch'] for b in receipt['branches']
            if b['branch'] != 'policy' and b.get('status') == 'graded']
    if receipt.get('chosen') is not None:
        want = [b for b in want if b != receipt['branches'][receipt['chosen']+1]['branch']]
    return set(want)


# ---------------------------------------------------------------- selection rules

def test_first_seed_plan_rules(tmp_path):
    root = tmp_path
    _budget(root)
    bank = rb.bank_root(root, 'B')
    # accepted + sealed legacy confirmation: graded candidates minus policy minus chosen
    snap, acc = decision(root, 'ga', 420, {('gap1', 1): 0.6})
    assert acc['status'] == 'accepted'
    confirm(root, snap, acc, 'B')
    entries = st.first_seed_plan(bank, acc)
    assert {b['branch'] for _, b, _ in entries} == branches_of(acc)
    for bdir, b, branches_dir in entries:
        assert branches_dir == bank/'sumo_verification'/'branches'
        assert b['verification_id'] == acc['verification_id']
        assert b['result_sha256'] and b['window']['start_s'] == 420 and 'J_veh_s' in b
    # rejected + registered retired: chosen is None -> every graded candidate
    _, rej = decision(root, 'gr', 460)
    assert rej['status'] == 'rejected' and rej['chosen'] is None
    assert st.first_seed_plan(bank, rej) == []            # not retired yet
    retired_file(root, [rej['verification_id']])
    entries = st.first_seed_plan(bank, rej)
    assert {b['branch'] for _, b, _ in entries} == branches_of(rej)
    assert all(b['branch'] != 'policy' for _, b, _ in entries)
    # rule 4: no confirmation and not retired
    _, pen = decision(root, 'gp', 500)
    assert st.first_seed_plan(bank, pen) == []
    # rule 4: uncompleted receipt
    _, stopped = decision(root, 'gs', 540)
    stopped = dict(stopped, status='stopped', chosen=None)
    assert st.first_seed_plan(bank, stopped) == []
    # LS-source receipts (tapes outside the bank) are refused, never slimmed
    with pytest.raises(st.SlimError):
        st.first_seed_plan(bank, dict(acc, _branches=str(root/'lscollect1'/'branches')))


def test_rule5_all8_recorded_decisions_belong_to_slim_plan(tmp_path):
    # NB: the test name must avoid BAD-guard words ('sealed' etc): pytest derives the tmp
    # directory name from the test id and the guard refuses any protected-looking path.
    from tools.coopmwm.test_slim_tapes_cpu import build_bank
    root, rec = build_bank(tmp_path/'all8')
    bank = rb.bank_root(root, 'B')
    assert st.first_seed_plan(bank, rec) == []
    assert len(st.slim_plan(bank, rec)) == 13             # the reviewed rule covers seed 1


def test_selection_excludes_protected_and_referenced(tmp_path):
    root = tmp_path
    _budget(root)
    bank = rb.bank_root(root, 'B')
    snap, acc = decision(root, 'ga', 420, {('gap1', 1): 0.6})
    confirm(root, snap, acc, 'B')
    vid = acc['verification_id']
    cases = {
        'bank_lesson_list': lambda: (bank/'LESSON_LIST_B.json').write_text(json.dumps(
            _seal(dict(arm='B', verification_ids=[vid]), 'lesson_list_sha256'))),
        'run_root_lesson_list': lambda: (root/'LESSON_LIST_R.json').write_text(json.dumps(
            _seal(dict(arm='R', verification_ids=[vid]), 'lesson_list_sha256'))),
        'isolated_claim': lambda: mkdir_json(root/'isolated'/'u006_B'/'CLAIM.json',
                                             dict(update=6, arm='B', verification_ids=[vid])),
        'policy_synergy_claim': lambda: mkdir_json(root/'pssynergy_ctl'/'u007_B'/'CLAIM.json',
                                                   dict(verification_ids=[vid])),
        'gate_plan': lambda: mkdir_json(root/'gate_coop6'/'COOPFIX_EXACT_PLAN.json',
                                        dict(gate='coopfix_exact',
                                             states=[dict(seed=986066, verification_id=vid,
                                                          jobs=[], quotes={})])),
        'gate_report': lambda: mkdir_json(root/'gate_coop7'/'COOPFIX_GATE.json',
                                          dict(states=[dict(seed=986048, verification_id=vid)])),
        'queue_pending': lambda: mkdir_json(root/'cpus_ctl'/'QUEUE.json',
                                            dict(queues={'B': [dict(verification_id=vid)],
                                                         'R': []})),
        'control_attempt': lambda: mkdir_json(root/'cpus_ctl'/'attempts'/f'B_{vid}.json',
                                              dict(arm='B', verification_id=vid, status='failed')),
    }
    for name, make in cases.items():
        make()
        assert vid in st.protected_ids(bank), name
        assert st.first_seed_plan(bank, acc) == [], name
    # BAD-guarded protection files are never opened (a 'sealed' control dir is invisible)
    mkdir_json(root/'sealed_ctl'/'QUEUE.json', dict(queues={'B': [dict(verification_id=vid)]}))
    assert all('sealed' not in s for s in st.protection_report(bank))
    assert vid in st.protected_ids(bank)                  # still protected by the earlier lists


def mkdir_json(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body))


def test_retired_registry_makes_rejected_eligible(tmp_path):
    root = tmp_path
    _budget(root)
    bank = rb.bank_root(root, 'B')
    _, rej = decision(root, 'gr', 460)
    vid = rej['verification_id']
    retired_file(root, [vid])
    assert vid in st.retired_ids(bank)
    assert {b['branch'] for _, b, _ in st.first_seed_plan(bank, rej)} == branches_of(rej)
    # a registry WITHOUT verification_ids protects nothing
    (root/'ALL8_RETIRED_X.json').write_text(json.dumps(dict(notes='no ids')))
    assert st.retired_ids(bank) == {vid}


def test_env_and_extra_protection_paths(tmp_path, monkeypatch):
    root = tmp_path
    _budget(root)
    bank = rb.bank_root(root, 'B')
    snap, acc = decision(root, 'ga', 420, {('gap1', 1): 0.6})
    confirm(root, snap, acc, 'B')
    vid = acc['verification_id']
    coop3 = tmp_path/'external'/'coop3_root'          # outside every discovered-glob depth
    coop3.mkdir(parents=True)
    (coop3/'LESSON_LIST_B.json').write_text(json.dumps(
        _seal(dict(arm='B', verification_ids=[vid]), 'lesson_list_sha256')))
    assert st.first_seed_plan(bank, acc) != []            # eligible without the coop3 list
    monkeypatch.setenv(st.PROTECT_ENV, str(coop3/'LESSON_LIST_B.json'))
    assert st.first_seed_plan(bank, acc) == []            # protected via the env paths
    monkeypatch.delenv(st.PROTECT_ENV)
    assert st.first_seed_plan(bank, acc, protected=st.protected_ids(bank, extra=[str(coop3)])) == []
    assert st.first_seed_plan(bank, acc) != []            # explicit extra only, no env


# ---------------------------------------------------------------- OFF / read-only identity

def test_quote_is_read_only_and_dry_run_moves_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(time, 'time', lambda: 1700000000.0)
    root = tmp_path
    _budget(root)
    bank = rb.bank_root(root, 'B')
    snap, acc = decision(root, 'ga', 420, {('gap1', 1): 0.6})
    confirm(root, snap, acc, 'B')
    _, rej = decision(root, 'gr', 460)
    retired_file(root, [rej['verification_id']])

    def tree():
        return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*')) if p.is_file()}
    before = tree()
    q = st.first_seed_quote(bank)
    assert q['default_off'] is True and q['bytes_removed_total'] > 0
    assert tree() == before                               # the quote never writes
    archive = tmp_path/'archive_target'
    dry = st.first_seed_migrate(bank, archive, dry_run=True)
    assert dry['dry_run'] is True and dry['bytes_moved'] > 0 and not archive.exists()
    assert tree() == before                               # dry run: nothing moves
    assert dry['bytes_moved'] == q['bytes_removed_total']


# ---------------------------------------------------------------- slim + readers

def test_slim_first_seed_keeps_policy_chosen_and_protected(tmp_path, monkeypatch):
    monkeypatch.setattr(time, 'time', lambda: 1700000000.0)
    root = tmp_path
    _budget(root)
    bank = rb.bank_root(root, 'B')
    snap, acc = decision(root, 'ga', 420, {('gap1', 1): 0.6, ('gap1', 2): 0.6})
    confirm(root, snap, acc, 'B')
    _, rej = decision(root, 'gr', 460)
    retired_file(root, [rej['verification_id']])
    plans = st.first_seed_plan(bank, acc)+st.first_seed_plan(bank, rej)
    assert len(plans) == (K-1)+K
    # full-tape series of every eligible branch BEFORE the move
    want = {(str(d), b['branch']): series_of(bd, b['verification_id'], b,
                                             b['window']['start_s'], b['window']['end_s'])
            for d, b, bd in plans}
    archive = tmp_path/'archive_target'
    out = st.first_seed_migrate(bank, archive)
    assert out['errors'] == [] and out['partial'] is None
    assert out['moved'] == (K-1)+K and out['bytes_moved'] > 0 and out['bytes_removed'] == 0
    for vid, receipt in st.first_seed_receipts(bank):
        vdir = bank/'sumo_verification'/'branches'/vid
        assert (vdir/'policy'/'RESULT.json').exists() and not (vdir/'policy'/st.SLIM).exists()
        if receipt.get('chosen') is not None:
            chosen = receipt['branches'][receipt['chosen']+1]['branch']
            assert (vdir/chosen/'RESULT.json').exists()
            assert not (vdir/chosen/st.SLIM).exists()
        for b in receipt['branches']:
            bd = vdir/b['branch']
            assert (bd/'observable.jsonl.gz').exists()     # traces untouched
            assert (vdir/f'{b["branch"]}.spec.json').exists()   # candidate spec files untouched
    assert len(slimmed_dirs(bank)) == (K-1)+K
    # SLIM fallback series bit-identical to the full-tape series for EVERY eligible branch
    for d, b, bd in plans:
        got = series_of(bd, b['verification_id'], b, b['window']['start_s'], b['window']['end_s'])
        assert got == want[(str(d), b['branch'])]
        assert not (d/st.MARKER).exists() and not (d/st.PAYLOAD).exists()
        verify_sealed(st.load_slim(d), 'slim_sha256')
        assert (archive/d.relative_to(bank)/st.MARKER).exists()
    # idempotent re-run
    again = st.first_seed_migrate(bank, archive)
    assert again['moved'] == 0 and again['already_slim'] == (K-1)+K and again['errors'] == []
    again_q = st.first_seed_quote(bank)
    assert again_q['bytes_removed_total'] == 0 and again_q['slim_bytes_total'] > 0


def test_readers_unchanged_after_first_seed_slimming(tmp_path, monkeypatch):
    monkeypatch.setattr(time, 'time', lambda: 1700000000.0)
    root = tmp_path
    _budget(root)
    bank = rb.bank_root(root, 'B')
    snap, acc = decision(root, 'ga', 420, {('gap1', 1): 0.6, ('gap1', 2): 0.6})
    confirm(root, snap, acc, 'B')
    _, rej = decision(root, 'gr', 460)
    retired_file(root, [rej['verification_id']])
    link_decision(bank, snap, acc, 'acc')
    archive = tmp_path/'archive_target'
    assert st.first_seed_migrate(bank, archive)['errors'] == []
    bs._CACHE.clear()
    # (1) per_seed_v1 preview over the slimmed bank (chosen/policy tapes of BOTH decisions)
    rep = bs.report_per_seed(rb, root, 'B')
    assert rep['bank_rule'] == 'per_seed_v1' and rep['examples_admitted'] >= 1
    assert rep['data_missing_examples'] == 0
    # (2) the lesson action-error reader of the KEPT chosen branch still works
    r = next(x for x in rb.bank_decisions(root, 'B')
             if x['verification_id'] == acc['verification_id'])
    assert rb._action_error(FakePolicy(), r, r['chosen_plan']) == pytest.approx(0.15)
    # (3) gate certification consumes the slimmed bank through the per_seed_v1 rule
    with ceg.bank_rule():
        candidates, statuses = ceg.certified_candidates(root, 'B')
    assert acc['verification_id'] in statuses
    # (4) relabel_reference_mismatch stays a no-op over a bank without all8 records
    from tools.coopmwm import verify_perseed as vp
    for vid, receipt in st.first_seed_receipts(bank):
        assert vp.relabel_reference_mismatch(receipt, bank) is receipt


def test_bank_pairs_and_lesson_check_after_first_seed_slim(tmp_path, monkeypatch):
    """The legacy readers (bank_pairs, lesson_check._action_error) read the KEPT policy/chosen
    tapes; the lesson-listed decision is fully protected, the retired one is slimmed."""
    monkeypatch.setattr(time, 'time', lambda: 1700000000.0)
    root = tmp_path
    _budget(root)
    bank = rb.bank_root(root, 'B')
    snap, acc = decision(root, 'ga', 420, {('gap1', 1): 0.6, ('gap1', 2): 0.6})
    confirm(root, snap, acc, 'B')
    link_decision(bank, snap, acc, 's1')
    lst = rb.lesson_list(root, 'B')
    assert lst and lst['verification_ids'] == [acc['verification_id']]
    assert st.first_seed_plan(bank, acc) == []            # now lesson-listed: protected
    _, rej = decision(root, 'gr', 460)
    retired_file(root, [rej['verification_id']])
    archive = tmp_path/'archive_target'
    out = st.first_seed_migrate(bank, archive)
    assert out['moved'] == K and out['errors'] == []
    bs._CACHE.clear()
    pairs, summary = rb.bank_pairs(root, 'B', training_seed=1)
    assert summary['counts']['used_admitted'] == 1        # chosen-branch entry intact
    check = rb.lesson_check(root, 'B', FakePolicy(), str(root), 4, lambda w: LessonTapeRunner(w))
    e = check['entries'][0]
    assert e['verification_id'] == acc['verification_id']
    assert e.get('imitation_action_error_mps2') == pytest.approx(0.15)   # chosen tape readable


# ---------------------------------------------------------------- migration safety

def test_first_seed_migration_partial_failure_keeps_both_sides(tmp_path, monkeypatch):
    monkeypatch.setattr(time, 'time', lambda: 1700000000.0)
    root = tmp_path
    _budget(root)
    bank = rb.bank_root(root, 'B')
    snap, acc = decision(root, 'ga', 420, {('gap1', 1): 0.6})
    confirm(root, snap, acc, 'B')
    _, rej = decision(root, 'gr', 460)
    retired_file(root, [rej['verification_id']])
    plan_acc = st.first_seed_plan(bank, acc)
    d0 = plan_acc[0][0]
    payload_sha = sha(d0/st.PAYLOAD)
    real_unlink = Path.unlink
    state = {'n': 0}

    def flaky_unlink(self, missing_ok=False):
        if self.name == st.MARKER and self.parent == d0:   # SECOND unlink of the FIRST branch
            state['n'] += 1
            if state['n'] == 1:
                raise OSError('simulated crash after the first source unlink')
        return real_unlink(self, missing_ok=missing_ok)
    monkeypatch.setattr(Path, 'unlink', flaky_unlink)
    archive = tmp_path/'archive_target'
    out = st.first_seed_migrate(bank, archive)
    monkeypatch.undo()
    assert out['partial'] == acc['verification_id'] and out['errors']
    assert out['errors'][0]['partial'] is True
    # BOTH sides kept: the verified archive copy of the already-unlinked payload AND the
    # remaining source marker, plus SLIM.json and the sealed PARTIAL.json record
    rel = d0.relative_to(bank)
    assert sha(archive/rel/st.PAYLOAD) == payload_sha
    assert not (d0/st.PAYLOAD).exists() and (d0/st.MARKER).exists()
    assert (d0/st.SLIM).exists() and (d0/st.PARTIAL).exists()
    verify_sealed(json.loads((d0/st.PARTIAL).read_text()), 'partial_sha256')
    # the run STOPPED: the second decision's tapes are all still there
    for b in rej['branches']:
        assert (bank/'sumo_verification'/'branches'/rej['verification_id']/b['branch'] /
                st.MARKER).exists()
    # a re-run restores the partial branch (full_restored) and moves the remaining 6 tapes
    out2 = st.first_seed_migrate(bank, archive)
    assert out2['moved'] == 2*(K-1) and out2['errors'] == []
    assert st.slim_branch(plan_acc[0], bank=bank, archive_dir=archive)['status'] == 'full_restored'


def test_first_seed_migration_copy_failure_keeps_source(tmp_path, monkeypatch):
    monkeypatch.setattr(time, 'time', lambda: 1700000000.0)
    root = tmp_path
    _budget(root)
    bank = rb.bank_root(root, 'B')
    snap, acc = decision(root, 'ga', 420, {('gap1', 1): 0.6})
    confirm(root, snap, acc, 'B')
    plan_ = st.first_seed_plan(bank, acc)
    d0 = plan_[0][0]
    real_copy2 = shutil.copy2

    def corrupt_copy2(src, dst, *a, **kw):
        real_copy2(src, dst, *a, **kw)
        if Path(dst).name == st.PAYLOAD:
            Path(dst).write_bytes(b'corrupted')            # the sha check must catch this
        return dst
    monkeypatch.setattr(shutil, 'copy2', corrupt_copy2)
    out = st.first_seed_migrate(bank, tmp_path/'archive_target')
    monkeypatch.undo()
    assert out['errors'] and len(out['errors']) == K-1 and out['moved'] == 0
    assert out['partial'] is None
    assert (d0/st.MARKER).exists() and (d0/st.PAYLOAD).exists()
    assert not (d0/st.SLIM).exists()
    out2 = st.first_seed_migrate(bank, tmp_path/'archive_target')   # clean re-run succeeds
    assert out2['moved'] == K-1 and out2['errors'] == []


def test_archive_inside_bank_refused(tmp_path):
    root = tmp_path
    _budget(root)
    bank = rb.bank_root(root, 'B')
    with pytest.raises(st.SlimError):
        st.first_seed_migrate(bank, bank/'archive')
    with pytest.raises(st.SlimError):
        st.first_seed_migrate(bank, tmp_path)              # a parent of the bank


# ---------------------------------------------------------------- quote / CLI

def test_first_seed_quote_bytes_per_decision(tmp_path, monkeypatch):
    monkeypatch.setattr(time, 'time', lambda: 1700000000.0)
    root = tmp_path
    _budget(root)
    bank = rb.bank_root(root, 'B')
    snap, acc = decision(root, 'ga', 420, {('gap1', 1): 0.6})
    confirm(root, snap, acc, 'B')
    _, rej = decision(root, 'gr', 460)
    retired_file(root, [rej['verification_id']])
    q = st.first_seed_quote(bank)
    planned = st.first_seed_plan(bank, acc)+st.first_seed_plan(bank, rej)
    measured = sum((d/st.MARKER).stat().st_size+((d/st.PAYLOAD).stat().st_size
                                                 if (d/st.PAYLOAD).exists() else 0)
                   for d, _, _ in planned)
    assert q['bytes_removed_total'] == measured > 0
    assert q['decisions'] == 2 and q['eligible_decisions'] == 2
    assert q['branches'] == (K-1)+K
    assert q['per_decision'][acc['verification_id']]['branches'] == K-1
    assert q['per_decision'][rej['verification_id']]['branches'] == K
    assert q['per_decision'][acc['verification_id']]['kept_reasons'] == []
    assert q['kept_decisions'] == {}
    # a NOT retired rejected decision shows its keep reason
    _, pen = decision(root, 'gp', 500)
    q2 = st.first_seed_quote(bank)
    assert q2['per_decision'][pen['verification_id']]['kept_reasons'] == \
        ['confirm_pending_or_not_retired']
    assert q2['eligible_decisions'] == 2
    assert q2['kept_decisions'] == {'confirm_pending_or_not_retired': 1}


def test_cli_first_seed_quote_migrate_and_all(tmp_path, monkeypatch):
    import contextlib
    import io
    monkeypatch.setattr(time, 'time', lambda: 1700000000.0)
    root = tmp_path
    _budget(root)
    bank = rb.bank_root(root, 'B')
    snap, acc = decision(root, 'ga', 420, {('gap1', 1): 0.6})
    confirm(root, snap, acc, 'B')
    _, rej = decision(root, 'gr', 460)
    retired_file(root, [rej['verification_id']])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert st.main(['first-seed', 'quote', '--root', str(root), '--arm', 'B']) == 0
    out = json.loads(buf.getvalue())
    assert out['scope'] == 'first_seed' and out['bytes_removed_total'] > 0
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        st.main(['first-seed', 'migrate', '--root', str(root), '--arm', 'B',
                 '--archive', str(tmp_path/'arc'), '--dry-run'])
    assert json.loads(buf.getvalue())['dry_run'] is True and not (tmp_path/'arc').exists()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert st.main(['first-seed', 'migrate', '--root', str(root), '--arm', 'B',
                        '--archive', str(tmp_path/'arc')]) == 0
    moved = json.loads(buf.getvalue())
    assert moved['moved'] == (K-1)+K and moved['errors'] == []
    assert (tmp_path/'arc'/'B'/'sumo_verification'/'branches'/rej['verification_id'] /
            'cand1'/st.MARKER).exists()                   # bank-relative archive layout
    assert not (tmp_path/'arc'/'B'/'sumo_verification'/'branches'/rej['verification_id'] /
                'policy'/st.MARKER).exists()              # the policy reference is never moved
    # --arm all: per-arm bytes dict (R is an empty bank)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert st.main(['first-seed', 'quote', '--root', str(root), '--arm', 'all']) == 0
    both = json.loads(buf.getvalue())
    assert set(both) == {'B', 'R'} and both['B']['bytes_removed_total'] == 0
    assert both['R']['decisions'] == 0
    # --protect adds an external list (coop3 layout) and keeps the decision
    (tmp_path/'coop3').mkdir(parents=True)
    (tmp_path/'coop3'/'LESSON_LIST_B.json').write_text(json.dumps(
        _seal(dict(verification_ids=[acc['verification_id']]), 'lesson_list_sha256')))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        st.main(['first-seed', 'quote', '--root', str(root), '--arm', 'all',
                 '--protect', str(tmp_path/'coop3')])
    q = json.loads(buf.getvalue())['B']
    assert q['per_decision'][acc['verification_id']]['kept_reasons'] == ['protected_reference']
