"""slim_rejected_tapes (disk saving; owner 2026-11-04; DEFAULT OFF, byte-identical when off).

The training box cannot expand its disk. verify_mode='all8_both_seeds'
(verify_perseed.confirm_all8) writes a full RESULT.json.gz (6-8 MB) for EVERY verified
candidate and extended-window branch (e.g. replay_bank/<arm>/confirm_extended_all8/branches/
<vid>/cand<i>/) - ~137 MB per B decision. After a decision's per-seed admission is sealed
(<bank>/confirmations_all8/<vid>.json, sealed 'confirmation_sha256'), later reads need only:

  (a) ADMITTED examples' plan-branch tapes (imitation: replay_bank.bank_pairs per_seed_v1,
      verify_perseed.imitation_entry / bank_examples);
  (b) policy REFERENCE tapes (branch 'policy' of every verification; grading/bootstrap,
      verify_perseed.relabel_reference_mismatch, coopfix_exact_gate.state_receipts
      'reference_adopted');
  (c) solo tapes (bank_statistics._best_solo, verify_perseed._cooperation).

READER AUDIT (grep RESULT.json / branches_dir / tape readers in bank_statistics, replay_bank,
verify_perseed, exact_grade, policy_synergy, coopfix_exact_gate): after the all8 record is
sealed, NO reader opens a NON-admitted candidate branch tape. bank_pairs/bank_examples read
admitted examples only; the legacy boot_v2/t1727 views and the lesson readers (bank_pairs
legacy path, replay_bank._action_error) read policy + first-seed CHOSEN branches;
solo and reference tapes are kept; every safety reader uses observable.jsonl.gz, which is
never touched. For extra safety the first-seed CHOSEN branch is kept even when seed 1
rejected it, and examples whose data went missing (status='data_missing') are never slimmed
(only explicitly status='rejected' ones lose their full tapes).

For every explicitly rejected (candidate, seed) the branch dir keeps a small SLIM.json: the
per-second cumulative team_delay_vs series over the verification window (exactly what
bank_statistics._step_series / branch_eval.totals_at consume), the receipt's grading summary
fields (window J_veh_s, status, actual_sim_s), per-vehicle window outcomes when the tape
carried them, the original payload identities (result_sha256 from the sealed receipt, the
marker's payload sha256) and a sealed 'slim_sha256'. The full RESULT.json + RESULT.json.gz
are then REMOVED (live path, option ON only) or MOVED to an operator-provided archive
directory with sha256 verification (migration CLI, never deletes: copy -> verify the bytes at
the destination against the source -> only then drop the source; a failure AFTER any source
file was already unlinked keeps BOTH sides, records PARTIAL.json and stops the run - the
archive copies are never deleted once the source side is incomplete). Admitted, reference and
solo tapes, every observable.jsonl.gz trace and the sealed record itself are untouched; every
CRN seed is covered (extra_seeds records: confirm_seed{k}_all8 group verifications and
confirm_extended_seed{k}_all8 extended windows for k >= 3); the option
key is 'slim_rejected_tapes' (confirm_pending/confirm_pending_all8/confirm_all8 parameter,
default None = the C4NM_SLIM_REJECTED_TAPES env var (or the BINDING's slim_rejected_tapes
key in the bank_builder), unset = OFF: nothing is written or removed anywhere).

bank_statistics._read_result falls back to a SLIM.json reconstruction (slim_tapes.slim_tape),
so even an unexpected future read of a slimmed branch keeps the statistics readers working
(integrity-checked against the sealed receipt's result_sha256).

One-off migration of EXISTING sealed decisions (the operator rsyncs the archive target to
another host; archive layout is bank-relative so it restores 1:1):
  python -m tools.coopmwm.slim_tapes migrate --root R --arm B --archive /target/dir [--dry-run]
  python -m tools.coopmwm.slim_tapes quote   --root R --arm B          (bytes saved, no writes)
"""
import hashlib
import json
import re
from pathlib import Path

from .archive_move import BAD

SCHEMA = 'coopmwm/slim-tapes/v1'
SLIM = 'SLIM.json'
MARKER = 'RESULT.json'
PAYLOAD = 'RESULT.json.gz'
ENABLE_KEY = 'slim_rejected_tapes'
ENV = 'C4NM_SLIM_REJECTED_TAPES'


class SlimError(ValueError):
    """A branch could not be slimmed safely; its full tape is kept and nothing was removed."""


class SlimPartial(SlimError):
    """A migration move failed AFTER a source file was already unlinked: the branch keeps both
    sides (remaining source files AND the verified archive copies) plus a PARTIAL.json marker;
    slimming stops for the run (an operator reconciles the branch before anything else moves)."""


PARTIAL = 'PARTIAL.json'


def enabled(flag=None):
    """The slim_rejected_tapes option: explicit True/False wins (strings like the BINDING's
    '1'/'true'/'on' are honoured too); None reads the env var."""
    if flag is None:
        import os
        return os.environ.get(ENV) == '1'
    if isinstance(flag, str):
        return flag.strip().lower() in ('1', 'true', 'yes', 'on')
    return bool(flag)


def _guard(path):
    """Never open sealed/ or holdout-like paths (repo-wide archive_move.BAD rule)."""
    if BAD.search(str(path)):
        raise SlimError(f'refusing to touch a protected path: {path}')
    return Path(path)


# ---------------------------------------------------------------- SLIM.json read side

def load_slim(bdir):
    """Verified SLIM.json body of one branch dir."""
    body = json.loads((Path(bdir)/SLIM).read_text(encoding='utf-8'))
    from .exact_grade import verify_sealed
    verify_sealed(body, 'slim_sha256')
    return body


def slim_tape(branches_dir, vid, branch):
    """Stats-capable tape view of a slimmed branch (bank_statistics._read_result fallback).

    Carries exactly the per-second cumulative team_delay_vs fingerprints of the verification
    window (branch_eval.totals_at-compatible) plus per-vehicle outcomes when stored. Integrity:
    the SLIM.json seal AND the caller's sealed receipt result_sha256 are both checked; the
    floats round-trip exactly, so per-second differences are bit-identical to the full tape."""
    from . import branch_eval
    bdir = Path(branches_dir)/vid/branch['branch']
    body = load_slim(bdir)
    if branch.get('result_sha256') and body['result_sha256'] != branch['result_sha256']:
        raise ValueError(f'slim tape {bdir/SLIM} differs from its sealed receipt')
    prefixes = {str(t): dict(metrics={branch_eval.METRIC: float(v)}) for t, v in body['series'].items()}
    tape = dict(prefixes=prefixes, final=None,
                slim=dict(schema=SCHEMA, verification_id=vid, branch=body['branch']))
    if body.get('per_vehicle_window_delay_vs') is not None:
        from .bank_statistics import PER_VEHICLE_KEY
        tape[PER_VEHICLE_KEY] = body['per_vehicle_window_delay_vs']
    return tape


# ---------------------------------------------------------------- what may be slimmed

def rejected_candidates(record):
    """{seed_index: {candidate_index}} EXPLICITLY rejected per seed (data_missing keeps tapes).
    ANY seed_index: extra_seeds (verify_perseed BUILD option) append seed_index >= 3 examples
    with the same admitted/status fields; missing keys simply stay absent."""
    out = {}
    for e in (record.get('per_seed_v1') or {}).get('examples') or []:
        if e.get('admitted') is False and e.get('status') == 'rejected':
            out.setdefault(int(e['seed_index']), set()).add(int(e['candidate_index']))
    return out


def _verification_plan(receipt_root, vid, branch_indices, receipt_sha256=None):
    """Candidate branch entries of one sealed verification whose (1-based) branch index is
    slimmed. 'policy' (the reference, index 0), ungraded branches and anything not in
    ``branch_indices`` is never returned."""
    from . import bank_statistics as bs
    receipt = bs._receipt(_guard(receipt_root), dict(verification_id=vid, receipt_sha256=receipt_sha256))
    branches_dir = Path(receipt_root)/'branches'
    win = receipt['window']
    out = []
    for b in receipt['branches']:
        if b['branch'] == 'policy' or b.get('status') != 'graded':
            continue
        if int(b['index']) not in branch_indices:
            continue
        out.append((branches_dir/vid/b['branch'],
                    dict(branch=b['branch'], index=int(b['index']), J_veh_s=b['J_veh_s'],
                         status=b['status'], actual_sim_s=b.get('actual_sim_s'),
                         result_sha256=b.get('result_sha256'), verification_id=vid,
                         receipt_sha256=receipt['receipt_sha256'],
                         window=dict(start_s=int(win['start_s']), end_s=int(win['end_s']))),
                    branches_dir))
    return out


def slim_plan(bank, record):
    """Branch entries ONE sealed all8 record slims: every explicitly rejected candidate branch
    on each seed, plus its extended-window branch - on the first two CRN seeds AND on every
    extra seed (record['extra']: group root confirm_seed{k}_all8, extended root
    confirm_extended_seed{k}_all8; the extra group verification holds only the cooperative
    candidates, so ``extra['candidates']`` maps position -> branch index like the extended
    ``candidate_indices``). NEVER returned: 'policy' branches, admitted (plan, seed) branches
    and their extended-window branches, anything under the solo roots, and the first-seed
    CHOSEN branch (kept even when seed 1 rejected it, for the legacy bank_pairs/lesson
    readers). The sealed record itself is never modified. A record whose first-seed receipt is
    not on disk next to its branches dir (e.g. an LS-source decision whose receipt lives in a
    label file) raises SlimError: nothing is slimmed without the receipt's 'chosen' index and
    per-branch identities."""
    from . import bank_statistics as bs
    from . import verify_perseed as vp
    from .exact_grade import verify_sealed
    verify_sealed(record, 'confirmation_sha256')
    if record.get('bank_rule') != bs.PER_SEED or 'examples' not in (record.get('per_seed_v1') or {}):
        raise SlimError(f"{record.get('verification_id')}: not a per_seed_v1 all8 record")
    bank = _guard(bank)
    rej = rejected_candidates(record)
    out = []
    first = record['first_seed']
    froot = Path(first['branches_dir']).parent      # receipts live next to branches/
    try:
        freceipt = bs._receipt(_guard(froot), dict(verification_id=record['verification_id'],
                                                   receipt_sha256=first.get('receipt_sha256')))
    except bs._Missing as e:
        raise SlimError(f"{record['verification_id']}: first-seed receipt not found next to "
                        f"{first['branches_dir']} ({e}); keeping every tape") from e
    if freceipt.get('chosen') is not None:          # legacy readers read the chosen branch tape
        rej[1] = rej.get(1, set())-{int(freceipt['chosen'])}
    out += _verification_plan(froot, record['verification_id'], {i+1 for i in rej.get(1, set())},
                              first.get('receipt_sha256'))
    s2 = record.get('seed2') or {}
    if s2.get('verification_id'):
        root = s2.get('root') or str(bank/vp.SEED2_ROOT)
        out += _verification_plan(Path(root), s2['verification_id'], {i+1 for i in rej.get(2, set())},
                                  s2.get('receipt_sha256'))
    for k, key, root in ((1, 'extended', vp.EXT_ROOT), (2, 'extended_seed2', vp.EXT2_ROOT)):
        ext = record.get(key) or {}
        if not ext.get('verification_id'):
            continue
        idxs = ext.get('candidate_indices') or []   # 0-based candidates of that seed's ext verification
        slim = {j+1 for j, i in enumerate(idxs) if int(i) in rej.get(k, set())}   # position -> branch index
        out += _verification_plan(bank/root, ext['verification_id'], slim, ext.get('receipt_sha256'))
    for k_str, extra in sorted((record.get('extra') or {}).items()):    # extra_seeds >= 3
        k = int(k_str)
        rk = rej.get(k, set())
        g = extra.get('candidates') or []           # 0-based candidates of that seed's GROUP verification
        groot = extra.get('root') or str(bank/vp.seed_kinds(k)[0])
        out += _verification_plan(Path(groot), extra['verification_id'],
                                  {j+1 for j, i in enumerate(g) if int(i) in rk},
                                  extra.get('receipt_sha256'))
        ext = extra.get('extended') or {}
        if ext.get('verification_id'):
            idxs = ext.get('candidate_indices') or []
            out += _verification_plan(bank/vp.seed_kinds(k)[1], ext['verification_id'],
                                      {j+1 for j, i in enumerate(idxs) if int(i) in rk},
                                      ext.get('receipt_sha256'))
    return out


# ---------------------------------------------------------------- one branch

def _sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build_slim(branch, tape, payload_sha256=None):
    """Sealed SLIM.json body for one branch entry; raises SlimError when the tape cannot
    faithfully feed the statistics readers (the caller then keeps everything)."""
    from . import bank_statistics as bs
    from . import branch_eval
    from .protocol import digest
    if branch.get('result_sha256') and digest(tape) != branch['result_sha256']:
        raise SlimError(f"{branch['branch']} of {branch['verification_id']}: tape differs from "
                        'its sealed receipt result_sha256')
    window = branch['window']
    series = {}
    for t in range(int(window['start_s']), int(window['end_s'])+1):
        series[str(t)] = float(branch_eval.totals_at(tape, t)[branch_eval.METRIC])
    delta = series[str(int(window['end_s']))]-series[str(int(window['start_s']))]
    if not abs(delta-float(branch['J_veh_s'])) <= 1e-9*max(1., abs(delta)):
        raise SlimError(f"{branch['branch']} of {branch['verification_id']}: series does not "
                        'reproduce the sealed J_veh_s')
    body = dict(schema=SCHEMA, verification_id=branch['verification_id'], branch=branch['branch'],
                index=int(branch['index']), window=dict(window), J_veh_s=float(branch['J_veh_s']),
                status=branch.get('status'), actual_sim_s=branch.get('actual_sim_s'),
                result_sha256=branch.get('result_sha256'), payload_sha256=payload_sha256,
                series=series, per_vehicle_window_delay_vs=tape.get(bs.PER_VEHICLE_KEY))
    from .exact_grade import _seal
    return _seal(body, 'slim_sha256')


def slim_branch(entry, bank=None, archive_dir=None, dry_run=False):
    """Slim ONE branch entry of slim_plan. archive_dir=None (live path): write SLIM.json then
    remove RESULT.json(.gz). archive_dir=<dir> (migration): MOVE the full files into the
    bank-relative archive layout (copy -> sha256-verify the copies against the source -> only
    then drop the source); never deletes without verification. Returns a row dict; a SlimError
    (or any other failure) leaves the branch fully intact."""
    from .accel_storage import read
    bdir, branch, _ = entry
    _guard(bdir)
    row = dict(verification_id=branch['verification_id'], branch=branch['branch'], dir=str(bdir),
               status=None, bytes_removed=0, bytes_moved=0, slim_bytes=0, archive=None)
    slim_path, marker, payload = bdir/SLIM, bdir/MARKER, bdir/PAYLOAD
    if slim_path.exists():
        if marker.exists() or payload.exists():
            row['status'] = 'full_restored'
        else:
            row['status'] = 'already_slim'
            row['slim_bytes'] = slim_path.stat().st_size
        return row
    if not marker.exists():
        row['status'] = 'no_tape'
        return row
    tape = read(marker)                              # verifies the gz payload digest itself
    psha = _sha_file(payload) if payload.exists() else None
    body = build_slim(branch, tape, payload_sha256=psha)
    removed = marker.stat().st_size + (payload.stat().st_size if payload.exists() else 0)
    row['bytes_removed'] = row['bytes_moved'] = removed
    if dry_run:
        row['status'] = 'dry_run'
        return row
    if archive_dir is not None:
        dest = _archive_dest(Path(archive_dir), bdir, bank)
        _move_verified(bdir, dest, body)
        row.update(status='moved', archive=str(dest), bytes_removed=0)
    else:
        from .exact_grade import _write_new
        _write_new(slim_path, body)
        marker.unlink()                              # marker FIRST: a later failure at worst
        payload.unlink(missing_ok=True)              # leaves full_restored, never a half tape
        row['status'] = 'slimmed'
    row['slim_bytes'] = slim_path.stat().st_size
    return row


def _archive_dest(archive_dir, bdir, bank=None):
    """Bank-relative archive layout (restores 1:1 after the operator's rsync); branch dirs
    outside the bank keep a stable hashed name."""
    rel = None
    if bank is not None:
        try:
            rel = Path(bdir).relative_to(Path(bank))
        except ValueError:
            rel = None
    if rel is None:
        rel = Path('_external')/f"{hashlib.sha256(str(bdir).encode()).hexdigest()[:16]}_{bdir.name}"
    return Path(archive_dir)/rel


def _move_verified(bdir, dest, body):
    """Migration move: copy RESULT.json(.gz) into dest, verify sha256 there (both files against
    the source digests, the uncompressed payload against the marker identity and the sealed
    receipt digest), write SLIM.json into the branch dir, then drop the source files. Before the
    FIRST source unlink any failure removes the copies and restores the branch untouched. Once
    ANY source file has been unlinked the archive copies are NEVER deleted (they are then the
    only complete remaining payload): a later failure keeps BOTH sides, records a PARTIAL.json
    marker in the branch dir and raises SlimPartial so the caller stops."""
    import shutil
    import time
    from .exact_grade import _write_new, _seal
    from .protocol import digest
    dest = _guard(dest)
    dest.mkdir(parents=True, exist_ok=True)
    src = {n: bdir/n for n in (PAYLOAD, MARKER) if (bdir/n).exists()}
    want = {n: _sha_file(p) for n, p in src.items()}
    copied = []
    unlinked = []
    try:
        for n, p in src.items():
            dst = dest/n
            shutil.copy2(p, dst)
            copied.append(dst)
            if _sha_file(dst) != want[n]:
                raise SlimError(f'{dst} sha256 differs from the source {p}')
        dmarker, dpayload = dest/MARKER, dest/PAYLOAD
        if not dmarker.exists():
            raise SlimError(f'{dmarker} missing after copy')
        if body['payload_sha256'] is not None:
            if not dpayload.exists():
                raise SlimError(f'{dpayload} missing after copy')
            import gzip
            dm = json.loads(dmarker.read_text(encoding='utf-8'))
            payload_bytes = gzip.decompress(dpayload.read_bytes())
            if hashlib.sha256(payload_bytes).hexdigest() != dm['sha256']:
                raise SlimError(f'{dpayload} uncompressed payload digest mismatch')
            if digest(json.loads(payload_bytes.decode('utf-8'))) != body['result_sha256']:
                raise SlimError(f'{dpayload} payload is not the sealed tape')
        _write_new(bdir/SLIM, body)
        for n in src:
            (bdir/n).unlink()
            unlinked.append(n)
    except Exception as e:
        if unlinked:
            # DATA-LOSS GUARD: at least one source file is already gone - the verified archive
            # copies are the only complete remaining payload. Keep BOTH sides, record the
            # PARTIAL marker, stop (the caller must not move anything else past this branch).
            try:
                partial = _seal(dict(schema=SCHEMA, state='partial_move',
                                     verification_id=body['verification_id'], branch=body['branch'],
                                     dir=str(bdir), archive=str(dest), unlinked=unlinked,
                                     kept_source=[n for n in src if n not in unlinked],
                                     error=repr(e), unix=time.time()), 'partial_sha256')
                _write_new(bdir/PARTIAL, partial)
            except Exception:
                pass                       # marker is best-effort; the raise below is the guarantee
            raise SlimPartial(f'PARTIAL move of {bdir}: {unlinked} already unlinked; kept BOTH '
                              f'the remaining source files and the archive copies {dest} '
                              f'({e!r}); nothing else will be moved this run') from e
        for dst in copied:
            dst.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------- record / bank level

def slim_record(bank, record, dry_run=False, archive_dir=None):
    """Slimming of ONE sealed all8 record (live path: full tapes REMOVED after SLIM.json is
    durable; archive_dir set: moved instead). The record is never modified and this NEVER raises:
    any Exception after sealing is caught and reported in 'errors' with the branch's full tape
    kept; a SlimPartial (source file already unlinked, both sides kept) additionally STOPS the
    record's loop - nothing else moves until the operator reconciles that branch."""
    out = dict(verification_id=record['verification_id'], records=0, slimmed=0, moved=0, kept=0,
               already_slim=0, bytes_removed=0, bytes_moved=0, slim_bytes=0, partial=False, errors=[])
    try:
        plan = slim_plan(bank, record)
    except Exception as e:      # any planning failure keeps every tape and is reported
        out['errors'].append(dict(error=repr(e)))
        return out
    out['records'] = len(plan)
    for entry in plan:
        try:
            row = slim_branch(entry, bank=bank, archive_dir=archive_dir, dry_run=dry_run)
        except SlimPartial as e:
            out['errors'].append(dict(branch=entry[1]['branch'], dir=str(entry[0]), error=str(e),
                                      partial=True))
            out['kept'] += 1
            out['partial'] = True
            break                        # STOP: both sides kept; operator reconciles first
        except Exception as e:   # ANY failure after sealing is recorded; the tape is kept
            out['errors'].append(dict(branch=entry[1]['branch'], dir=str(entry[0]), error=repr(e)))
            out['kept'] += 1
            continue
        out[row['status']] = out.get(row['status'], 0)+1
        out['bytes_removed'] += row['bytes_removed']
        out['bytes_moved'] += row['bytes_moved']
        out['slim_bytes'] += row['slim_bytes']
    return out


def records(bank):
    """{vid: sealed all8 record} of the bank (read-only, same rule as bank_statistics.all8_records)."""
    from . import bank_statistics as bs
    from .exact_grade import verify_sealed
    out = {}
    for p in sorted((Path(bank)/bs.ALL8_RECORDS).glob('*.json')):
        v = json.loads(p.read_text(encoding='utf-8'))
        verify_sealed(v, 'confirmation_sha256')
        if v.get('bank_rule') != bs.PER_SEED or 'examples' not in (v.get('per_seed_v1') or {}):
            raise ValueError(f'{p}: not a per_seed_v1 all8 record')
        out[v['verification_id']] = v
    return out


def quote(bank):
    """Bytes slimming WOULD remove/move per sealed record and in total (read-only, no writes).
    A record whose planning fails for ANY reason (not only SlimError) is reported with zero
    branches instead of crashing the quote."""
    bank = Path(bank)
    per, total, slim_total, branches = {}, 0, 0, 0
    for vid, rec in records(bank).items():
        removed, n = 0, 0
        try:
            plan = slim_plan(bank, rec)
        except Exception:
            plan = []
        for bdir, _, _ in plan:
            marker, payload, slim = bdir/MARKER, bdir/PAYLOAD, bdir/SLIM
            if slim.exists() and not marker.exists() and not payload.exists():
                slim_total += slim.stat().st_size
                continue
            if not marker.exists():
                continue
            removed += marker.stat().st_size + (payload.stat().st_size if payload.exists() else 0)
            n += 1
        per[vid] = dict(branches=n, bytes_removed=removed)
        total += removed
        branches += n
    return dict(bank=str(bank), option=ENABLE_KEY, env=ENV, default_off=True, records=len(per),
                branches=branches, bytes_removed_total=total, slim_bytes_total=slim_total,
                per_record=per)


def migrate(bank, archive_dir, dry_run=False):
    """One-off migration: apply the rule to every EXISTING sealed all8 record of the bank,
    MOVING each non-admitted branch's full RESULT.json(.gz) into archive_dir (bank-relative
    layout, sha256-verified) instead of deleting. Never touches admitted/reference/solo tapes,
    observable traces or the sealed records."""
    bank, archive_dir = Path(bank), Path(archive_dir)
    a, b = archive_dir.resolve(), bank.resolve()
    if a == b or a in b.parents or b in a.parents:
        raise SlimError('archive dir must be outside the bank')
    out = dict(bank=str(bank), archive=str(archive_dir), dry_run=bool(dry_run), records=0,
               bytes_moved=0, bytes_removed=0, slim_bytes=0, partial=None, errors=[])
    for vid, rec in sorted(records(bank).items()):
        row = slim_record(bank, rec, dry_run=dry_run, archive_dir=archive_dir)
        row['verification_id'] = vid
        out['records'] += 1
        out['bytes_moved'] += row['bytes_moved']
        out['bytes_removed'] += row['bytes_removed']
        out['slim_bytes'] += row['slim_bytes']
        out['errors'] += [dict(verification_id=vid, **e) for e in row['errors']]
        for k in ('slimmed', 'moved', 'kept', 'already_slim', 'full_restored', 'no_tape'):
            out[k] = out.get(k, 0)+row.get(k, 0)
        if row.get('partial'):      # STOP the whole migration after a partial move
            out['partial'] = vid
            break
    return out


def main(argv=None):
    """python -m tools.coopmwm.slim_tapes quote|migrate --root R --arm B [--archive DIR] [--dry-run]"""
    import argparse
    ap = argparse.ArgumentParser(prog='slim_tapes')
    sub = ap.add_subparsers(dest='cmd', required=True)
    q = sub.add_parser('quote', help='bytes slimming would save (read-only)')
    q.add_argument('--root', required=True)
    q.add_argument('--arm', required=True)
    m = sub.add_parser('migrate', help='move full non-admitted tapes to --archive (SLIM.json stays)')
    m.add_argument('--root', required=True)
    m.add_argument('--arm', required=True)
    m.add_argument('--archive', required=True)
    m.add_argument('--dry-run', action='store_true')
    a = ap.parse_args(argv)
    from .replay_bank import bank_root
    bank = bank_root(a.root, a.arm)
    if a.cmd == 'quote':
        print(json.dumps(quote(bank), indent=1))
        return 0
    print(json.dumps(migrate(bank, Path(a.archive), dry_run=a.dry_run), indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
