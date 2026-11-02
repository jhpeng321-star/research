"""Narrow per-plan code-fault isolation; safety/physics/accounting stay fatal."""
import traceback
from .protocol import digest


def execute(record,call):
    try:
        return call()
    except (KeyError,TypeError,AttributeError,NameError,NotImplementedError) as error:
        # ValueError/AssertionError/RuntimeError include validation, parity,
        # accounting, physics and CUDA failures: deliberately propagate those.
        return dict(accepted=False,reason='per-query code error; policy continues',
            record_sha256=digest(record),compute_counts_unknown=True,
            query_error=dict(kind='code',error=repr(error),traceback=traceback.format_exc(),
                inference_retry=False))


def validate_response(record,result):
    error=result.get('query_error')
    if error is None:return
    if (error.get('kind')!='code' or not error.get('traceback') or
            error.get('inference_retry') is not False or result.get('accepted') is not False or
            result.get('record_sha256')!=digest(record)):
        raise ValueError('invalid per-query error response')


def malformed(kind,record,result):
    """HARDEN-3: reason string if a delivered GPU result breaks the client ABI, else None.

    Only shape/identity is checked (never a value threshold). A malformed plan or
    centre result becomes a per-query decline (policy continues), not a global STOP.
    """
    if not isinstance(result,dict):return 'result is not a JSON object'
    if result.get('record_sha256')!=digest(record):return 'record_sha256 missing or mismatched'
    if kind=='centre':
        if result.get('declined'):return None
        if result.get('candidate_scores')!=0:return 'centre candidate_scores must be 0'
        traj=result.get('policy_traj')
        if not isinstance(traj,list) or len(traj)!=len(record['group']):return 'centre policy_traj shape'
        return None
    if not isinstance(result.get('accepted'),bool):return 'accepted is not a boolean'
    if not result['accepted']:return None
    plan=result.get('plan')
    if not isinstance(plan,dict) or plan.get('members')!=record['group'] or 'id' not in plan:
        return 'accepted plan missing or not for the requested group'
    if not isinstance(plan.get('duration_s'),int) or 'profile' not in plan:return 'accepted plan duration/profile'
    top=result.get('topk')
    if top is not None and (not isinstance(top,list) or any(not isinstance(c,dict) or
            c.get('members')!=record['group'] or not isinstance(c.get('duration_s'),int) for c in top)):
        return 'topk candidates malformed'
    return None


def declined(kind,record,reason,**extra):
    """Explicit policy-continues response for a malformed/unavailable result (never a forecast)."""
    if kind=='centre':
        return dict(record_sha256=digest(record),candidate_scores=0,policy_traj=None,receipt=None,
            declined=reason,inference_retry=False,**extra)
    return dict(accepted=False,reason=reason+'; policy continues',record_sha256=digest(record),
        compute_counts_unknown=True,candidates_searched=0,imagined_futures=0,
        query_error=dict(kind='malformed',error=reason,inference_retry=False),**extra)
