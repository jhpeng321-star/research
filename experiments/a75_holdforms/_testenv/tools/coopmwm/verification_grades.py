"""Reuse decision-time SUMO verification grades; never dispatch another branch.

The reused contrast is the decision-local exact candidate vs policy comparison
on the verifier's COMMON window (max candidate duration + 120, episode capped).
It is labelled separately from a later marginal replay of the realised episode.
Later-truncated plans still need their registered executed-portion marginal grade.
"""
import copy
import json
from pathlib import Path
import time

from . import exact_grade as xg, sumo_verify as sv
from .protocol import checked, digest


def _enqueue(queue, spec):
    try:
        queue.enqueue(spec)
    except FileExistsError:
        # Rollout completion and the grade supervisor can import simultaneously.
        if queue.spec(spec['grade_id'])['spec_sha256'] != spec['spec_sha256']:
            raise ValueError('concurrent grade spec differs; preserve both attempts')


def record_smoke(root, verification, live, k=sv.TOP_K):
    """Record a real-smoke gate after all k+1 branches and live equivalence pass.

    The caller must supply actual TRAIN receipts; this helper cannot run SUMO.
    CPU fixtures exercise it only in their isolated temporary roots. ``k`` is the
    registered verification width (4 today; 8 after the learned-search switch).
    """
    gate = sv.check_live_continuation(verification, live)
    if verification['k'] != k or len(verification['branches']) != k+1:
        raise ValueError(f'real smoke requires {k} candidates and policy')
    if any(b['status'] != 'graded' or not (b.get('audit') or {}).get('crn_verified_t0')
           for b in verification['branches']):
        raise ValueError(f'real smoke requires {k+1} exact-prefix/CRN passes')
    p = checked(Path(root)/'SMOKE_PASSED.json')
    if p.exists():
        old = json.loads(p.read_text(encoding='utf-8'))
        xg.verify_sealed(old, 'smoke_sha256')
        if not old['pass_']:
            raise ValueError('existing smoke did not pass')
        return old
    value = xg._seal(dict(gate, all_five_branches_graded=True, unix=time.time()), 'smoke_sha256')
    xg._write_new(p, value)
    return value


def reuse(queue, spec, original, step, verification_root):
    """Publish an exact-grade-compatible receipt before its queue item becomes visible.

    Claims/attempts are never replaced. A completed import is idempotent. No
    pending queue item can race into a second SUMO charge during publication.
    """
    root = checked(Path(verification_root))
    smoke = json.loads((root/'SMOKE_PASSED.json').read_text(encoding='utf-8'))
    xg.verify_sealed(smoke, 'smoke_sha256')
    if not smoke.get('pass_') or not smoke.get('all_five_branches_graded'):
        raise ValueError('verification real smoke not qualified')
    identity = step['sample']['behaviour']['sumo_verification']
    vid = identity['verification_id']
    if Path(vid).name != vid or '/' in vid or '\\' in vid:
        raise ValueError('invalid verification receipt identity')
    verification = json.loads(checked(root/'verifications'/f'{vid}.json').read_text(encoding='utf-8'))
    xg.verify_sealed(verification, 'receipt_sha256')
    if identity['receipt_sha256'] != verification['receipt_sha256']:
        raise ValueError('verification source SHA differs from executed tape')
    if (verification['episode']['world'] != original['world'] or
            verification['episode']['arm'] != spec['episode']['arm'] or
            verification['episode']['update'] != spec['episode']['update'] or
            verification['t0'] != spec['window']['start_s'] or
            digest(verification['chosen_plan']) != spec['plan']['plan_sha256']):
        raise ValueError('verification source does not match the graded plan')
    sv.check_live_continuation(verification, original)
    if spec['plan']['duration_s'] != verification['chosen_plan']['duration_s']:
        raise ValueError('truncated plan requires executed-portion marginal grading')
    if verification['grade_veh_s'] <= 0:
        raise ValueError('accepted verification must beat policy')
    # MAIN (harden-int): failed CANDIDATE branches are excluded, never imputed; the policy
    # branch, the chosen branch and every graded branch still need exact prefix/CRN evidence.
    used = [verification['branches'][0], verification['branches'][verification['chosen']+1]]
    if any(b['status'] != 'graded' for b in used) or any(
            not (b.get('audit') or {}).get('crn_verified_t0') for b in verification['branches'] if b['status'] == 'graded'):
        raise ValueError('verification branch prefix/CRN evidence incomplete')
    selected = verification['branches'][verification['chosen']+1]
    policy = verification['branches'][0]
    source = dict(verification_id=vid, receipt_sha256=verification['receipt_sha256'],
                  smoke_sha256=smoke['smoke_sha256'],
                  contrast='decision-local exact verification; future unknown plans absent',
                  window_rule='common max-candidate-duration + 120, episode capped')
    imported = copy.deepcopy(spec)
    imported.pop('spec_sha256')
    imported.update(sim_s=0, verification_source=source, overlap_unresolved=False,
                    window=dict(verification['window'], nominal_end_s=verification['window']['end_s']),
                    overlapping_plan_starts=[], replayed_plans=[], overlap_clash_members=[])
    imported = xg._seal(imported, 'spec_sha256')
    body = dict(status='graded', grade_veh_s=verification['grade_veh_s'],
                J_with_veh_s=selected['J_veh_s'], J_without_veh_s=policy['J_veh_s'],
                window_with=selected['window'], window_without=policy['window'],
                noise=dict(crn_verified_t0=True, noise_mode_used='crn'),
                prefix=dict(checked_from_s=300, checked_to_s=verification['t0'], pass_=True),
                actual_sim_s=0, verification_source=source, grade_source='sumo_verification',
                no_fresh_branch=True)
    gid = imported['grade_id']
    existing = queue.receipt(gid)
    if existing is not None:
        xg.verify_sealed(existing, 'receipt_sha256')
        if existing.get('verification_source') != source or existing['spec_sha256'] != imported['spec_sha256']:
            raise ValueError('different grade already exists; preserve it')
        _enqueue(queue, imported)  # repair an interrupted receipt-before-spec publication
        return gid
    if (queue.root/'queue'/f'{gid}.json').exists() or (queue.root/'claims'/f'{gid}.json').exists() or queue.attempts(gid):
        # A concurrent importer may have published between the first receipt
        # read and this queue check. Only adopt the identical verified receipt.
        winner = queue.receipt(gid)
        if winner is not None:
            xg.verify_sealed(winner, 'receipt_sha256')
            if winner.get('verification_source') == source and winner['spec_sha256'] == imported['spec_sha256']:
                _enqueue(queue, imported)
                return gid
        raise ValueError('grade already queued/claimed; never duplicate or replace it')
    result = xg.receipt(imported, body, 0, verification['timing']['started_unix'])
    try:
        xg._write_new(queue.root/'receipts'/f'{gid}.json', result)
    except FileExistsError:
        winner = queue.receipt(gid)
        xg.verify_sealed(winner, 'receipt_sha256')
        if winner.get('verification_source') != source or winner['spec_sha256'] != imported['spec_sha256']:
            raise ValueError('concurrent grade receipt differs; preserve it')
    _enqueue(queue, imported)
    return gid


def enqueue_episode(queue, original, meta):
    """Old/unverified starts use marginal grading; verified full starts reuse branches."""
    imported = []
    starts = {int(s['time_s']): s for _, s in xg.executed_plan_steps(original)}
    for spec in xg.job_specs(original, meta):
        step = starts[spec['window']['start_s']]
        verified = (step.get('proposal') or {}).get('sumo_verified')
        full = spec['plan']['duration_s'] == step['proposal']['plan']['duration_s']
        if verified and full:
            imported.append(reuse(queue, spec, original, step, original['sumo_verification_root']))
        elif queue.enqueue(spec):
            imported.append(spec['grade_id'])
    return imported
