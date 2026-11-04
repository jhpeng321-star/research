"""CPU tests: slim_rejected_tapes (owner 2026-11-04). No SUMO: reuses the deterministic
per-second plant of test_verify_perseed_cpu (exact team_delay_vs fingerprints, gz tapes)."""
import copy
import hashlib
import json
import shutil
import time
from pathlib import Path

import pytest

from tools.coopmwm import bank_statistics as bs
from tools.coopmwm import branch_eval
from tools.coopmwm import coopfix_exact_gate as ceg
from tools.coopmwm import replay_bank as rb
from tools.coopmwm import slim_tapes as st
from tools.coopmwm import verify_perseed as vp
from tools.coopmwm.exact_grade import _seal, verify_sealed
from tools.coopmwm.protocol import digest
from tools.coopmwm.test_replay_bank_confirm_cpu import fp, plan, snapshot, bank_budget  # noqa: F401
from tools.coopmwm.test_verify_perseed_cpu import (ExtraPlant, PHASE, _budget,  # noqa: F401
                                                   plants_of, record, setup)  # noqa: F401

# p1 wins seed 1 (cooperative), p2 also admitted on seed 1 (not cooperative), p3 wins seed 2
# (cooperative); every other candidate is explicitly rejected on BOTH seeds. Rejected branch
# tapes: 6 on seed 1 (cand3..cand8 of the first verification) + 7 on seed 2 (the seed-2
# verification) = 13; the extended verifications hold only admitted candidates.
GAINS = {('p1', 1): 0.6, ('p2', 1): 0.5, ('p3', 2): 0.6,
         ('p1', 1, 'solo'): 0.1, ('p2', 1, 'solo'): 0.5, ('p3', 2, 'solo'): 0.1}
N_SLIM = 13
N_TAPES = 32            # 9 first + 9 seed2 + 3 ext1 + 2 ext2 + 3*3 solos
N_TAPE_FILES = 2*N_TAPES     # every branch dir holds the RESULT.json marker AND the .gz payload


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def scrub(v, root):
    """Replace the run's root path inside record strings (paths differ across runs)."""
    if isinstance(v, str):
        return v.replace(str(root), '<R>')
    if isinstance(v, list):
        return [scrub(x, root) for x in v]
    if isinstance(v, dict):
        return {k: scrub(x, root) for k, x in v.items()}
    return v


def tree(root):
    """{relative path: sha256} of every file under root."""
    root = Path(root)
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*')) if p.is_file()}


def build_bank(tmp_path, slim=None):
    root, snap, plans, r, runner_for, plants = setup(tmp_path, GAINS)
    _budget(root)
    rb.confirm_pending(root, 'B', runner_for, str(root), PHASE, verify_mode=vp.ALL8,
                       slim_rejected_tapes=slim)
    bs._CACHE.clear()
    return root, record(root)


def slimmed_dirs(bank, rec):
    """[(branch_dir, branch)] the option must slim for ONE record, by the same plan rule."""
    return [(d, b) for d, b, _ in st.slim_plan(bank, rec)]


def slimmed_names(bank):
    return {str(p.parent.resolve()) for p in Path(bank).rglob('SLIM.json')}


# ---------------------------------------------------------------- option defaults / OFF identity

def test_option_default_off(monkeypatch):
    monkeypatch.delenv(st.ENV, raising=False)
    assert st.enabled(None) is False and st.enabled(False) is False and st.enabled(True) is True
    monkeypatch.setenv(st.ENV, '1')
    assert st.enabled(None) is True and st.enabled(False) is False


def test_off_is_byte_identical_and_keeps_every_tape(tmp_path, monkeypatch):
    """Default OFF: no SLIM.json anywhere, no tape removed, sealed record identical across an
    absent option / explicit False / env '0' (same frozen clock, same process)."""
    monkeypatch.setattr(time, 'time', lambda: 1700000000.0)
    monkeypatch.delenv(st.ENV, raising=False)
    outs = []
    for name, kw, env in (('absent', {}, None), ('false', dict(slim_rejected_tapes=False), None),
                          ('env0', {}, '0')):
        if env is None:
            monkeypatch.delenv(st.ENV, raising=False)
        else:
            monkeypatch.setenv(st.ENV, env)
        root, rec = build_bank(tmp_path/name, slim=kw.get('slim_rejected_tapes'))
        bank = rb.bank_root(root, 'B')
        assert not list(bank.rglob('SLIM.json'))
        n_tapes = len(list(bank.rglob('RESULT.json')))+len(list(bank.rglob('RESULT.json.gz')))
        assert n_tapes == N_TAPE_FILES
        verify_sealed(rec, 'confirmation_sha256')     # 'confirmation_sha256' covers the root
        body = scrub({k: v for k, v in rec.items() if k != 'confirmation_sha256'}, root)
        outs.append((json.dumps(body, sort_keys=True), n_tapes))
    assert outs[0] == outs[1] == outs[2]


# ---------------------------------------------------------------- ON: what is kept / removed

def test_on_slims_rejected_only_and_keeps_admitted_reference_solo(tmp_path, monkeypatch):
    monkeypatch.setattr(time, 'time', lambda: 1700000000.0)
    root, rec = build_bank(tmp_path/'on', slim=True)
    bank = rb.bank_root(root, 'B')
    got = slimmed_names(bank)
    want = {str(d.resolve()) for d, _ in slimmed_dirs(bank, rec)}
    assert got == want and len(want) == N_SLIM
    for p in Path(bank).rglob('SLIM.json'):
        d = p.parent
        assert not (d/'RESULT.json').exists() and not (d/'RESULT.json.gz').exists()
        assert (d/'observable.jsonl.gz').exists()          # safety readers untouched
    assert len(list(bank.rglob('RESULT.json')))+len(list(bank.rglob('RESULT.json.gz'))) == N_TAPE_FILES-2*N_SLIM
    # (a) every admitted example's plan tape still there and identity-checked
    for e in rec['per_seed_v1']['examples']:
        if e['admitted']:
            tape, entry = vp.imitation_entry(e, rec['t0'], 1.)
            assert digest(tape) == e['tape']['branch']['result_sha256']
            assert not (Path(e['tape']['branches_dir'])/rec['verification_id']/
                        e['tape']['branch']['branch']/st.SLIM).exists()
    # (b) every policy reference tape kept (first, seed2, both extended, solos)
    for root_dir in (bank/'sumo_verification', bank/vp.SEED2_ROOT, bank/vp.EXT_ROOT,
                     bank/vp.EXT2_ROOT, bank/vp.SOLO_ROOTS[1], bank/vp.SOLO_ROOTS[2]):
        for vr in (root_dir/'verifications').glob('*.json'):
            receipt = json.loads(vr.read_text())
            d = root_dir/'branches'/receipt['verification_id']/receipt['branches'][0]['branch']
            assert (d/'RESULT.json').exists() and not (d/st.SLIM).exists(), d
    # (c) EVERY solo tape kept (all branches of the solo verifications)
    for k in (1, 2):
        for vr in (bank/vp.SOLO_ROOTS[k]/'verifications').glob('*.json'):
            receipt = json.loads(vr.read_text())
            for b in receipt['branches']:
                d = bank/vp.SOLO_ROOTS[k]/'branches'/receipt['verification_id']/b['branch']
                assert (d/'RESULT.json').exists() and not (d/st.SLIM).exists(), d
    # the sealed record itself is untouched and still verifies
    record_path = bank/vp.RECORDS/(rec['verification_id']+'.json')
    verify_sealed(json.loads(record_path.read_text()), 'confirmation_sha256')
    q = st.quote(bank)
    assert q['bytes_removed_total'] == 0 and q['slim_bytes_total'] > 0


def test_record_and_neighbours_untouched_by_slim_record(tmp_path, monkeypatch):
    """slim_record never modifies the sealed record or anything outside the planned dirs."""
    monkeypatch.setattr(time, 'time', lambda: 1700000000.0)
    root, rec = build_bank(tmp_path/'off', slim=False)
    bank = rb.bank_root(root, 'B')
    record_path = bank/vp.RECORDS/(rec['verification_id']+'.json')
    before = tree(bank)
    out = st.slim_record(bank, rec)
    assert out['slimmed'] == N_SLIM and not out['errors'] and out['bytes_removed'] > 0
    after = tree(bank)
    rel = str(record_path.relative_to(bank))
    assert after[rel] == before[rel]
    verify_sealed(json.loads(record_path.read_text()), 'confirmation_sha256')
    planned = {str(Path(d).relative_to(bank)) for d, _ in slimmed_dirs(bank, rec)}
    changed = {k for k in set(before) | set(after) if before.get(k) != after.get(k)}
    assert changed <= {str(Path(d)/n) for d in planned
                       for n in ('RESULT.json', 'RESULT.json.gz', 'SLIM.json')}
    assert st.load_slim(bank/next(iter(planned)))['schema'] == st.SCHEMA


def test_first_seed_chosen_branch_kept_when_seed1_rejected_it(tmp_path):
    """Unit rule: the legacy CHOSEN branch of the first-seed verification is never slimmed, even
    when the all8 record rejected that candidate on seed 1 (legacy bank_pairs/lesson readers)."""
    from tools.coopmwm.accel_storage import write as awrite
    bank = tmp_path/'bank'
    first = bank/'sumo_verification'
    vid, t0, t_end = 'v0', 600, 720
    tape = dict(prefixes={str(t): fp(t, 100.+0.5*(t-t0)) for t in range(300, t_end+1)}, final=None, steps=[])
    (first/'verifications').mkdir(parents=True)
    branches = [dict(branch='policy', index=0, status='graded', J_veh_s=60.,
                     window=dict(start_s=t0, end_s=t_end), result_sha256=digest(tape))]
    for i in (1, 2, 3):
        d = first/'branches'/vid/f'cand{i}';d.mkdir(parents=True)
        awrite(d/'RESULT.json', tape)
        branches.append(dict(branch=f'cand{i}', index=i, status='graded', J_veh_s=60.,
                             window=dict(start_s=t0, end_s=t_end), result_sha256=digest(tape)))
    receipt = _seal(dict(verification_id=vid, status='accepted', chosen=2,
                         window=dict(start_s=t0, end_s=t_end), branches=branches), 'receipt_sha256')
    (first/'verifications'/(vid+'.json')).write_text(json.dumps(receipt))
    row = dict(seed_index=1, admitted=False, status='rejected',
               tape=dict(branches_dir=str(first/'branches'), verification_id=vid))
    rec = _seal(dict(verification_id=vid, bank_rule='per_seed_v1', first_seed=dict(
        verification_id=vid, receipt_sha256=receipt['receipt_sha256'], branches_dir=str(first/'branches')),
        per_seed_v1=dict(examples=[dict(row, candidate_index=0), dict(row, candidate_index=2)]),
        seed2={}, extended=None, extended_seed2=None), 'confirmation_sha256')
    planned = st.slim_plan(bank, rec)
    assert [b['branch'] for _, b, _ in planned] == ['cand1']      # cand3 IS the chosen branch
    out = st.slim_record(bank, rec)
    assert out['slimmed'] == 1 and not out['errors']
    assert (first/'branches'/vid/'cand3'/'RESULT.json').exists()
    assert (first/'branches'/vid/'cand1'/'SLIM.json').exists()
    got = bs._read_result(first/'branches', vid, dict(branch='cand1', result_sha256=digest(tape)))
    for t in range(t0, t_end+1):
        assert branch_eval.totals_at(got, t)['team_delay_vs'] == \
            branch_eval.totals_at(tape, t)['team_delay_vs']


# ---------------------------------------------------------------- readers after slimming

def test_readers_work_after_slimming(tmp_path, monkeypatch):
    monkeypatch.setattr(time, 'time', lambda: 1700000000.0)
    root, rec = build_bank(tmp_path/'slim', slim=True)
    bank = rb.bank_root(root, 'B')
    vid = rec['verification_id']
    # (1) per_seed_v1 preview (the bank_statistics preview entry point)
    rep = bs.report_per_seed(rb, root, 'B')
    assert rep['bank_rule'] == 'per_seed_v1' and rep['examples_admitted'] == 3
    assert rep['data_missing_examples'] == 0
    # (2) imitation readers: admitted examples' tapes intact (bank_pairs per_seed_v1 path)
    rows = vp.bank_examples(root, 'B')
    assert len(rows) == 3
    for e in rows:
        tape, entry = vp.imitation_entry(e, rec['t0'], 1.)
        assert digest(tape) == e['tape']['branch']['result_sha256']
    # (3) relabel_reference_mismatch re-derives the label from the KEPT plan/reference/solo tapes
    stale = copy.deepcopy(rec)
    target = next(e for e in stale['per_seed_v1']['examples']
                  if e['admitted'] and e['seed_index'] == 1 and e['cooperative'] is True)
    target['cooperative_status'] = 'reference_mismatch'
    out = vp.relabel_reference_mismatch(stale, bank)
    fixed = next(e for e in out['per_seed_v1']['examples'] if e['example_id'] == target['example_id'])
    assert fixed['cooperative_status'] == 'cooperative' and fixed.get('reader_relabel')
    assert fixed['cooperative'] is True
    # (4) gate certification reads the slimmed bank through the per_seed_v1 rule
    with ceg.bank_rule():
        candidates, statuses = ceg.certified_candidates(root, 'B')
    cert = next(c for c in candidates if c[0]['verification_id'] == vid)
    assert sorted(cert[2]) == [1, 2]
    assert statuses[vid]['cooperative'] is True


def test_slim_tape_stats_bit_identical_to_sealed_admission(tmp_path, monkeypatch):
    """bank_statistics._read_result of a slimmed branch rebuilds a tape whose paired bootstrap
    result equals the sealed admission stats exactly (same floats, same RNG draws)."""
    monkeypatch.setattr(time, 'time', lambda: 1700000000.0)
    root, rec = build_bank(tmp_path/'stats', slim=True)
    checked = 0
    for e in rec['per_seed_v1']['examples']:
        if e['status'] != 'rejected' or 'admission' not in e:
            continue
        t = e['tape']
        pol = bs._read_result(t['branches_dir'], t['verification_id'], dict(branch='policy'))
        slim = bs._read_result(t['branches_dir'], t['verification_id'], t['branch'])
        st2 = bs._stats([(pol, slim, t['window'])])
        assert st2['lower95_one_sided_veh_s'] == e['admission']['lower95_one_sided_veh_s']
        assert st2['pooled_total_veh_s'] == e['admission']['pooled_total_veh_s']
        assert st2['per_seed'][0]['n_units'] == e['admission']['per_seed'][0]['n_units']
        checked += 1
    assert checked == N_SLIM
    # the slimmed branch no longer has a full tape
    e = next(e for e in rec['per_seed_v1']['examples'] if e['status'] == 'rejected')
    t = e['tape']
    assert not (Path(t['branches_dir'])/t['verification_id']/t['branch']['branch']/'RESULT.json').exists()
    # a tampered SLIM.json is refused (seal checked on every read)
    d = Path(t['branches_dir'])/t['verification_id']/t['branch']['branch']
    body = st.load_slim(d);body['J_veh_s'] += 1.0
    (d/st.SLIM).write_text(json.dumps(body))
    with pytest.raises(ValueError):
        bs._read_result(t['branches_dir'], t['verification_id'], t['branch'])


# ---------------------------------------------------------------- migration

def test_migration_moves_with_sha256_and_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(time, 'time', lambda: 1700000000.0)
    root, rec = build_bank(tmp_path/'off', slim=False)
    bank = rb.bank_root(root, 'B')
    dirs = slimmed_dirs(bank, rec)
    before = {str(d): {n: sha(d/n) for n in ('RESULT.json', 'RESULT.json.gz') if (d/n).exists()}
              for d, _ in dirs}
    archive = tmp_path/'archive_target'
    dry = st.migrate(bank, archive, dry_run=True)                # dry run: nothing moves
    assert dry['dry_run'] and dry['bytes_moved'] > 0 and not archive.exists()
    assert all((d/'RESULT.json').exists() for d, _ in dirs)
    out = st.migrate(bank, archive)
    assert out['moved'] == N_SLIM and out['bytes_moved'] == dry['bytes_moved']
    assert out['errors'] == [] and out['bytes_removed'] == 0     # migration never deletes
    for d, _ in dirs:
        rel = d.relative_to(bank)
        assert (archive/rel/st.MARKER).exists()
        for n, want in before[str(d)].items():
            assert sha(archive/rel/n) == want                    # exact bytes at destination
        assert not (d/st.MARKER).exists() and not (d/st.PAYLOAD).exists()
        assert (d/st.SLIM).exists()
    bs._CACHE.clear()                                            # readers still work
    rep = bs.report_per_seed(rb, root, 'B')
    assert rep['examples_admitted'] == 3 and rep['data_missing_examples'] == 0
    again = st.migrate(bank, archive)                            # idempotent re-run
    assert again['moved'] == 0 and again['already_slim'] == N_SLIM and again['errors'] == []
    with pytest.raises(st.SlimError):
        st.migrate(bank, bank/'archive')                         # archive inside the bank
    with pytest.raises(st.SlimError):
        st.migrate(bank, tmp_path)                               # a parent of the bank


def test_migration_failure_keeps_source_intact(tmp_path, monkeypatch):
    monkeypatch.setattr(time, 'time', lambda: 1700000000.0)
    root, rec = build_bank(tmp_path/'off', slim=False)
    bank = rb.bank_root(root, 'B')
    d0 = slimmed_dirs(bank, rec)[0][0]
    real_copy2 = shutil.copy2

    def corrupt_copy2(src, dst, *a, **kw):
        real_copy2(src, dst, *a, **kw)
        if Path(dst).name == st.PAYLOAD:
            Path(dst).write_bytes(b'corrupted')                  # the sha check must catch this
        return dst
    monkeypatch.setattr(shutil, 'copy2', corrupt_copy2)
    out = st.migrate(bank, tmp_path/'archive_target')
    assert out['errors'] and out['moved'] == 0
    assert (d0/st.MARKER).exists() and (d0/st.PAYLOAD).exists()  # source untouched
    assert not (d0/st.SLIM).exists()
    dest = tmp_path/'archive_target'/d0.relative_to(bank)
    assert not dest.exists() or not (dest/st.MARKER).exists()    # partial copies cleaned
    monkeypatch.undo()
    out2 = st.migrate(bank, tmp_path/'archive_target')           # clean re-run succeeds
    assert out2['moved'] == N_SLIM and out2['errors'] == []


# ---------------------------------------------------------------- quote / CLI

def test_quote_bytes_saved(tmp_path, monkeypatch):
    monkeypatch.setattr(time, 'time', lambda: 1700000000.0)
    root, rec = build_bank(tmp_path/'off', slim=False)
    bank = rb.bank_root(root, 'B')
    q = st.quote(bank)
    assert q['option'] == st.ENABLE_KEY and q['default_off'] is True and q['records'] == 1
    measured = sum((d/st.MARKER).stat().st_size+((d/st.PAYLOAD).stat().st_size if (d/st.PAYLOAD).exists() else 0)
                   for d, _ in slimmed_dirs(bank, rec))
    assert q['bytes_removed_total'] == measured > 0
    assert q['per_record'][rec['verification_id']]['branches'] == N_SLIM
    root2, rec2 = build_bank(tmp_path/'on', slim=True)
    q2 = st.quote(rb.bank_root(root2, 'B'))
    assert q2['bytes_removed_total'] == 0 and q2['slim_bytes_total'] > 0
    assert q2['slim_bytes_total'] < measured                     # SLIM.json is tiny


def test_cli_migrate_and_quote(tmp_path, monkeypatch):
    import contextlib
    import io
    monkeypatch.setattr(time, 'time', lambda: 1700000000.0)
    root, rec = build_bank(tmp_path/'off', slim=False)
    bank = rb.bank_root(root, 'B')
    st.main(['migrate', '--root', str(root), '--arm', 'B', '--archive', str(tmp_path/'arc')])
    d0 = slimmed_dirs(bank, rec)[0][0]
    assert (tmp_path/'arc'/d0.relative_to(bank)/st.MARKER).exists()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert st.main(['quote', '--root', str(root), '--arm', 'B']) == 0
    out = json.loads(buf.getvalue())
    assert out['bytes_removed_total'] == 0 and out['slim_bytes_total'] > 0
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        st.main(['migrate', '--root', str(root), '--arm', 'B', '--archive', str(tmp_path/'arc2'),
                 '--dry-run'])
    assert json.loads(buf.getvalue())['dry_run'] is True


# ---------------------------------------------------------------- FIX ROUND (2026-11-04)

def test_option_accepts_config_strings(monkeypatch):
    """The BINDING/config key may hold a string ('1'/'true'/'on' ON, '0'/''/'false' OFF)."""
    assert st.enabled('1') is True and st.enabled('true') is True and st.enabled('ON') is True
    assert st.enabled(' yes ') is True
    assert st.enabled('0') is False and st.enabled('false') is False and st.enabled('') is False
    assert st.enabled('no') is False and st.enabled(None) is False


def test_rejected_candidates_any_seed_index():
    """rejected_candidates handles ANY seed_index (extra_seeds append seed_index >= 3 examples);
    data_missing examples and admitted ones never appear."""
    rec = dict(per_seed_v1=dict(examples=[
        dict(seed_index=7, candidate_index=3, admitted=False, status='rejected'),
        dict(seed_index=9, candidate_index=1, admitted=False, status='rejected'),
        dict(seed_index=9, candidate_index=2, admitted=True, status='admitted'),
        dict(seed_index=9, candidate_index=4, admitted=False, status='data_missing'),
        dict(seed_index=1, candidate_index=5, admitted=False, status='rejected')]))
    assert st.rejected_candidates(rec) == {7: {3}, 9: {1}, 1: {5}}


def test_quote_survives_any_planning_error(tmp_path, monkeypatch):
    """quote() must not crash when a record's planning raises ANY exception (not only SlimError):
    the record is reported with zero branches instead."""
    monkeypatch.setattr(time, 'time', lambda: 1700000000.0)
    root, rec = build_bank(tmp_path/'off', slim=False)
    bank = rb.bank_root(root, 'B')
    def boom(b, r):
        raise RuntimeError('unexpected planning crash')
    monkeypatch.setattr(st, 'slim_plan', boom)
    q = st.quote(bank)                                   # must NOT raise
    assert q['records'] == 1 and q['branches'] == 0
    assert q['per_record'][rec['verification_id']] == dict(branches=0, bytes_removed=0)


def test_slim_record_never_raises_and_keeps_every_tape(tmp_path, monkeypatch):
    """slim_record catches ANY Exception per branch (not only SlimError/(OSError,ValueError,
    KeyError)) and NEVER raises; nothing changes on disk and the tapes are all kept. A crash of
    slim_record itself (after sealing) never fails the sealed record either."""
    monkeypatch.setattr(time, 'time', lambda: 1700000000.0)
    root, rec = build_bank(tmp_path/'off', slim=False)
    bank = rb.bank_root(root, 'B')
    before = tree(bank)
    def boom_branch(entry, bank=None, archive_dir=None, dry_run=False):
        raise RuntimeError('unexpected branch crash')
    monkeypatch.setattr(st, 'slim_branch', boom_branch)
    out = st.slim_record(bank, rec)                      # must NOT raise
    monkeypatch.undo()
    assert out['slimmed'] == 0 and out['kept'] == N_SLIM and out['partial'] is False
    assert len(out['errors']) == N_SLIM and all('RuntimeError' in e['error'] for e in out['errors'])
    assert tree(bank) == before                          # nothing was touched anywhere
    def boom_record(b, r, dry_run=False, archive_dir=None):
        raise RuntimeError('record-level slim crash')
    monkeypatch.setattr(st, 'slim_record', boom_record)
    root2, rec2 = build_bank(tmp_path/'on', slim=True)   # confirm_all8 must survive this
    monkeypatch.undo()
    verify_sealed(rec2, 'confirmation_sha256')
    assert not list(rb.bank_root(root2, 'B').rglob(st.SLIM))     # tapes kept, nothing slimmed
    assert len(rec2['per_seed_v1']['examples']) == 16


def test_partial_move_keeps_both_sides_marks_and_stops(tmp_path, monkeypatch):
    """THE data-loss edge: the archive copies are NEVER deleted once ANY source file was
    unlinked. Here the payload unlinks fine and the marker unlink fails: both sides are kept,
    a sealed PARTIAL.json marker records the state, and the run STOPS (nothing else moves)."""
    monkeypatch.setattr(time, 'time', lambda: 1700000000.0)
    root, rec = build_bank(tmp_path/'off', slim=False)
    bank = rb.bank_root(root, 'B')
    d0 = slimmed_dirs(bank, rec)[0][0]                   # plan[0]: the very first branch
    keep = {n: sha(d0/n) for n in (st.MARKER, st.PAYLOAD)}
    real_unlink = Path.unlink
    def fail_marker_unlink(self, missing_ok=False):
        if self == d0/st.MARKER:
            raise PermissionError(13, 'stub: marker unlink fails AFTER the payload was unlinked')
        return real_unlink(self, missing_ok=missing_ok)
    monkeypatch.setattr(Path, 'unlink', fail_marker_unlink)
    archive = tmp_path/'archive_target'
    out = st.migrate(bank, archive)
    monkeypatch.undo()
    rel = d0.relative_to(bank)
    assert (d0/st.MARKER).exists()                       # the failing unlink left it in place
    assert not (d0/st.PAYLOAD).exists()                  # the payload WAS unlinked first
    assert (archive/rel/st.MARKER).exists() and (archive/rel/st.PAYLOAD).exists()
    assert sha(archive/rel/st.MARKER) == keep[st.MARKER]          # copies NOT deleted
    assert sha(archive/rel/st.PAYLOAD) == keep[st.PAYLOAD]        # (the only full payload left)
    partial = json.loads((d0/st.PARTIAL).read_text())
    verify_sealed(partial, 'partial_sha256')
    assert partial['state'] == 'partial_move' and partial['unlinked'] == [st.PAYLOAD]
    assert partial['kept_source'] == [st.MARKER] and partial['archive'] == str(archive/rel)
    assert out['partial'] == rec['verification_id']      # STOP: nothing else moved this run
    assert out['moved'] == 0 and out['kept'] == 1
    assert len(out['errors']) == 1 and out['errors'][0]['partial'] is True
    out2 = st.migrate(bank, archive)                     # clean re-run moves the REST
    assert out2['partial'] is None and out2['errors'] == []
    assert out2['moved'] == N_SLIM-1 and out2['full_restored'] == 1   # d0: SLIM+marker present
    for d, _ in slimmed_dirs(bank, rec)[1:]:
        assert (d/st.SLIM).exists() and not (d/st.MARKER).exists() and not (d/st.PAYLOAD).exists()
    assert (d0/st.SLIM).exists() and (d0/st.MARKER).exists() and not (d0/st.PAYLOAD).exists()
    bs._CACHE.clear()                                    # readers still work (slim fallback)
    rep = bs.report_per_seed(rb, root, 'B')
    assert rep['examples_admitted'] == 3 and rep['data_missing_examples'] == 0


# ---------------------------------------------------------------- extra_seeds compatibility

def test_extra_seeds_rejected_tapes_slimmed_on_every_seed_index(tmp_path, monkeypatch):
    """extra_seeds=4 (merged c4/extra-seeds layout): rejected_candidates handles ANY seed_index
    and slim_plan covers the extra-seed roots confirm_seed{k}_all8 / confirm_extended_seed{k}_all8.
    p1 (candidate 0) is cooperative on seeds 1/2 and re-verified on seeds 3..6: seed 3 admits,
    seed 4 rejects on its own LB (group branch only - no extended window ran), seed 5 fails its
    own later-harm window (group + extended branches), seed 6 admits. Exactly 16 branches are
    slimmed (7 first-seed + 6 seed-2 + the 3 extra-seed ones); every admitted (plan, seed) tape,
    every policy reference, every extended/solo tape and all observables are kept."""
    monkeypatch.setattr(time, 'time', lambda: 1700000000.0)
    gains = {('p1', 1): 0.6, ('p1', 2): 0.6, ('p1', 1, 'solo'): 0.1, ('p1', 2, 'solo'): 0.1,
             ('p1', 3): 0.6, ('p1', 3, 'solo'): 0.1, ('p1', 5): 0.6, ('p1', 6): 0.6, ('p1', 6, 'solo'): 0.6,
             ('p3', 2): 0.6, ('p3', 2, 'solo'): 0.6}
    root, snap, plans, r, _, _ = setup(tmp_path/'ext', gains)
    _budget(root)
    runner_for, _ = plants_of(ExtraPlant, gains, {('p1', 5): 0.5})
    rb.confirm_pending(root, 'B', runner_for, str(root), PHASE, verify_mode=vp.ALL8,
                       extra_seeds=4, slim_rejected_tapes=True)
    bank = rb.bank_root(root, 'B')
    rec = record(root)
    assert rec['extra_seeds'] == 4 and sorted(rec['extra']) == ['3', '4', '5', '6']
    rej = st.rejected_candidates(rec)
    assert set(rej) == {1, 2, 4, 5}                      # ANY seed_index, not only {1, 2}
    assert rej[4] == {0} and rej[5] == {0}
    # exactly the planned branches were slimmed (16 total)
    got = slimmed_names(bank)
    want = {str(d.resolve()) for d, _, _ in st.slim_plan(bank, rec)}
    assert got == want and len(want) == 16
    roots = {d.parents[2] for d, _, _ in st.slim_plan(bank, rec)}
    assert roots == {bank/'sumo_verification', bank/vp.SEED2_ROOT,
                     bank/'confirm_seed4_all8', bank/'confirm_seed5_all8',
                     bank/'confirm_extended_seed5_all8'}
    # the extra-seed roots appear EXACTLY as named (each sub-verification has its own id)
    for k in (4, 5):
        gvid = rec['extra'][str(k)]['verification_id']
        d = bank/f'confirm_seed{k}_all8'/'branches'/gvid/'cand1'
        assert (d/st.SLIM).exists() and not (d/st.MARKER).exists(), d
        pol = d.parent/'policy'
        assert (pol/st.MARKER).exists() and not (pol/st.SLIM).exists()
    assert not (bank/'confirm_extended_seed4_all8').exists()      # no extended window ran for s4
    e5ext = rec['extra']['5']['extended']['verification_id']
    d = bank/'confirm_extended_seed5_all8'/'branches'/e5ext/'cand1'
    assert (d/st.SLIM).exists() and not (d/st.MARKER).exists(), d
    pol = d.parent/'policy'
    assert (pol/st.MARKER).exists() and not (pol/st.SLIM).exists()
    for k in (3, 6):
        gvid = rec['extra'][str(k)]['verification_id']
        evids = {vp.seed_kinds(k)[0]: gvid,
                 vp.seed_kinds(k)[1]: rec['extra'][str(k)]['extended']['verification_id']}
        for name, evid in evids.items():
            for b in ('policy', 'cand1'):
                d = bank/name/'branches'/evid/b
                assert (d/st.MARKER).exists() and not (d/st.SLIM).exists(), d   # admitted: kept
    # every example's own tape: rejected slimmed, admitted full
    for e in rec['per_seed_v1']['examples']:
        t = e['tape']
        d = Path(t['branches_dir'])/t['verification_id']/t['branch']['branch']
        if e['status'] == 'rejected':
            assert (d/st.SLIM).exists() and not (d/st.MARKER).exists(), d
        else:
            assert (d/st.MARKER).exists() and not (d/st.SLIM).exists(), d
    # policy references of the first/seed2/extended verifications and the solos are untouched
    for root_dir in (bank/'sumo_verification', bank/vp.SEED2_ROOT, bank/vp.EXT_ROOT,
                     bank/vp.EXT2_ROOT, bank/'confirm_extended_seed5_all8',
                     bank/'confirm_solo_all8_s1', bank/'confirm_solo_all8_s2',
                     bank/'confirm_solo_all8_s3', bank/'confirm_solo_all8_s6'):
        for vr in (root_dir/'verifications').glob('*.json'):
            receipt = json.loads(vr.read_text())
            d = root_dir/'branches'/receipt['verification_id']/receipt['branches'][0]['branch']
            assert (d/st.MARKER).exists() and not (d/st.SLIM).exists(), d
    # a slimmed EXTRA-seed branch feeds the stats readers exactly like the sealed admission
    e5 = next(e for e in rec['per_seed_v1']['examples'] if e['seed_index'] == 5)
    t = e5['tape']
    pol = bs._read_result(t['branches_dir'], t['verification_id'], dict(branch='policy'))
    slim = bs._read_result(t['branches_dir'], t['verification_id'], t['branch'])
    st2 = bs._stats([(pol, slim, t['window'])])
    assert st2['lower95_one_sided_veh_s'] == e5['admission']['lower95_one_sided_veh_s']
    # admitted extra-seed examples are still readable imitation entries
    rows = vp.bank_examples(root, 'B')
    assert len(rows) == 5                                # p1 s1/s2/s3/s6 + p3 s2
    for x in rows:
        tape, entry = vp.imitation_entry(x, rec['t0'], 1.)
        assert digest(tape) == x['tape']['branch']['result_sha256']
    rep = bs.report_per_seed(rb, root, 'B')
    assert rep['examples_admitted'] == 5 and rep['data_missing_examples'] == 0
    q = st.quote(bank)
    assert q['bytes_removed_total'] == 0 and q['slim_bytes_total'] > 0
