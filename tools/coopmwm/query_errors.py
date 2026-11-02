"""Narrow per-plan code-fault isolation; safety/physics/accounting stay fatal."""
import traceback
from .protocol import digest


# Worker/run invariants that must stay fatal even when raised inside one query.
FATAL_WORDS=('parity','immutable','accounting','physics','safety','budget','deadline','identity','duplicate','cuda')


def per_query(error):
    """True for a fault confined to this query (decline it), False to propagate.

    MAIN17:26: any per-query error is a decline, never a worker death. Plan/group
    validation ValueError/AssertionError (e.g. the 4-6 member family_seeds
    'complete observable group of two or three required') is per-query; the
    named invariants above stay fatal. A CUDA OOM of one large group is released
    and declined; other RuntimeErrors (CUDA context faults) still propagate.
    """
    if isinstance(error,(KeyError,TypeError,AttributeError,NameError,NotImplementedError,IndexError,ZeroDivisionError,StopIteration)):
        return True
    if type(error).__name__=='OutOfMemoryError':
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        return True
    if isinstance(error,(ValueError,AssertionError)) and not isinstance(error,UnicodeError):
        text=(type(error).__name__+' '+str(error)).lower()
        return not any(word in text for word in FATAL_WORDS)
    return False


def execute(record,call):
    try:
        return call()
    except Exception as error:
        if not per_query(error):raise
        # kind stays 'code' so already-deployed brokers accept the decline.
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


def plan_shape(record,result):
    """HARDEN-3: reason if an ACCEPTED plan result breaks the client ABI, else None.

    Shape only (never a value threshold): the client indexes plan['members'],
    duration_s, the per-second profile and the top-k candidates (k=8 verification).
    """
    if not result.get('accepted'):return None
    plan=result.get('plan')
    if not isinstance(plan,dict) or 'id' not in plan or plan.get('members')!=record['group']:
        return 'accepted plan missing or not for the requested group'
    if not isinstance(plan.get('duration_s'),int) or 'profile' not in plan:return 'accepted plan duration/profile'
    top=result.get('topk')
    if top is not None and (not isinstance(top,list) or any(not isinstance(c,dict) or 'id' not in c or
            c.get('members')!=record['group'] or not isinstance(c.get('duration_s'),int) for c in top)):
        return 'topk candidates malformed'
    return None
