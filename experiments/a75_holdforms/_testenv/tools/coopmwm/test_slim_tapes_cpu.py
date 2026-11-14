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
from tools.coopmwm.test_verify_perseed_cpu import PHASE, _budget, record, setup  # noqa: F401

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
    plan = st.slim_plan(bank, rec)
    assert [b['branch'] for _, b, _ in plan] == ['cand1']      # cand3 IS the chosen branch
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
    plan = slimmed_dirs(bank, rec)
    before = {str(d): {n: sha(d/n) for n in ('RESULT.json', 'RESULT.json.gz') if (d/n).exists()}
              for d, _ in plan}
    archive = tmp_path/'archive_target'
    dry = st.migrate(bank, archive, dry_run=True)                # dry run: nothing moves
    assert dry['dry_run'] and dry['bytes_moved'] > 0 and not archive.exists()
    assert all((d/'RESULT.json').exists() for d, _ in plan)
    out = st.migrate(bank, archive)
    assert out['moved'] == N_SLIM and out['bytes_moved'] == dry['bytes_moved']
    assert out['errors'] == [] and out['bytes_removed'] == 0     # migration never deletes
    for d, _ in plan:
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
